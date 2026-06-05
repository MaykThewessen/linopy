"""Tests for ModelSnapshot.advance — incremental in-place snapshot advance.

Contract: capture(model_post) is equivalent to capture(model_pre).advance(diff, model_post)
for any diff that was successfully applied (no rebuild). Equivalence is
checked field-by-field, including unchanged containers (which advance must
not touch).
"""

from __future__ import annotations

import numpy as np
import pytest

from linopy import Model
from linopy.persistent import ModelDiff, ModelSnapshot, RebuildReason


def _build_two_container_model() -> Model:
    m = Model()
    x = m.add_variables(0.0, 10.0, coords=[range(4)], name="x")
    y = m.add_variables(-5.0, 5.0, coords=[range(3)], name="y")
    m.add_constraints(2 * x >= 1.0, name="cx")
    m.add_constraints(x.sum() + y.sum() <= 20.0, name="cxy")
    m.add_objective(x.sum() + 2 * y.sum())
    return m


def _snapshot_equal(a: ModelSnapshot, b: ModelSnapshot) -> None:
    assert a.structural_key == b.structural_key
    assert a.obj_quad_present == b.obj_quad_present
    assert a.obj_sense == b.obj_sense
    np.testing.assert_array_equal(a.obj_c, b.obj_c)
    assert a.var_buffers.keys() == b.var_buffers.keys()
    for k in a.var_buffers:
        va, vb = a.var_buffers[k], b.var_buffers[k]
        np.testing.assert_array_equal(va.lower, vb.lower)
        np.testing.assert_array_equal(va.upper, vb.upper)
        np.testing.assert_array_equal(va.active_labels, vb.active_labels)
        assert va.type == vb.type
    assert a.con_buffers.keys() == b.con_buffers.keys()
    for k in a.con_buffers:
        ca, cb = a.con_buffers[k], b.con_buffers[k]
        np.testing.assert_array_equal(ca.indptr, cb.indptr)
        np.testing.assert_array_equal(ca.indices, cb.indices)
        np.testing.assert_array_equal(ca.data, cb.data)
        np.testing.assert_array_equal(ca.rhs, cb.rhs)
        np.testing.assert_array_equal(ca.sign, cb.sign)
        np.testing.assert_array_equal(ca.active_labels, cb.active_labels)


def test_advance_matches_capture_for_rhs_only() -> None:
    m = _build_two_container_model()
    snap_pre = ModelSnapshot.capture(m)

    m.constraints["cx"].update(rhs=np.array([2.0, 2.0, 2.0, 2.0]))

    diff = ModelDiff.from_snapshot(snap_pre, m, same_model=True)
    assert diff.rebuild_reason is RebuildReason.NONE
    assert diff.changed_constraints == {"cx"}
    assert "cxy" not in diff.changed_constraints

    snap_pre.advance(diff, m)
    snap_full = ModelSnapshot.capture(m)
    _snapshot_equal(snap_pre, snap_full)


def test_advance_matches_capture_for_var_bounds() -> None:
    m = _build_two_container_model()
    snap_pre = ModelSnapshot.capture(m)

    m.variables["x"].update(upper=np.array([7.0, 7.0, 7.0, 7.0]))

    diff = ModelDiff.from_snapshot(snap_pre, m, same_model=True)
    assert diff.changed_variables == {"x"}
    assert "y" not in diff.changed_variables

    snap_pre.advance(diff, m)
    snap_full = ModelSnapshot.capture(m)
    _snapshot_equal(snap_pre, snap_full)


def test_advance_matches_capture_for_objective_linear() -> None:
    m = _build_two_container_model()
    snap_pre = ModelSnapshot.capture(m)

    x = m.variables["x"]
    y = m.variables["y"]
    m.add_objective(3 * x.sum() + 4 * y.sum(), overwrite=True)

    diff = ModelDiff.from_snapshot(snap_pre, m, same_model=True)
    assert diff.obj_c_indices is not None
    assert diff.changed_variables == set()
    assert diff.changed_constraints == set()

    snap_pre.advance(diff, m)
    snap_full = ModelSnapshot.capture(m)
    _snapshot_equal(snap_pre, snap_full)


def test_advance_matches_capture_for_combined_changes() -> None:
    m = _build_two_container_model()
    snap_pre = ModelSnapshot.capture(m)

    m.constraints["cx"].update(rhs=np.array([1.5, 1.5, 1.5, 1.5]))
    m.variables["y"].update(lower=np.array([-3.0, -3.0, -3.0]))

    diff = ModelDiff.from_snapshot(snap_pre, m, same_model=True)
    assert "cx" in diff.changed_constraints
    assert "y" in diff.changed_variables

    snap_pre.advance(diff, m)
    snap_full = ModelSnapshot.capture(m)
    _snapshot_equal(snap_pre, snap_full)


def test_advance_does_not_touch_unchanged_containers() -> None:
    m = _build_two_container_model()
    snap_pre = ModelSnapshot.capture(m)
    cxy_buf_id = id(snap_pre.con_buffers["cxy"])
    y_buf_id = id(snap_pre.var_buffers["y"])

    m.constraints["cx"].update(rhs=np.array([2.0, 2.0, 2.0, 2.0]))
    diff = ModelDiff.from_snapshot(snap_pre, m, same_model=True)
    snap_pre.advance(diff, m)

    assert id(snap_pre.con_buffers["cxy"]) == cxy_buf_id
    assert id(snap_pre.var_buffers["y"]) == y_buf_id


def test_advance_empty_diff_is_noop() -> None:
    m = _build_two_container_model()
    snap_pre = ModelSnapshot.capture(m)
    diff = ModelDiff.from_snapshot(snap_pre, m, same_model=True)
    assert diff.is_empty

    snap_pre.advance(diff, m)
    snap_full = ModelSnapshot.capture(m)
    _snapshot_equal(snap_pre, snap_full)


def test_advance_rejects_rebuild_required_diff() -> None:
    m = _build_two_container_model()
    snap_pre = ModelSnapshot.capture(m)
    diff = ModelDiff(rebuild_reason=RebuildReason.SPARSITY)
    with pytest.raises(ValueError, match="rebuild-required"):
        snap_pre.advance(diff, m)


def test_advance_clears_coef_dirty_on_touched_constraints() -> None:
    m = _build_two_container_model()
    snap_pre = ModelSnapshot.capture(m)

    m.constraints["cx"].update(rhs=np.array([2.0, 2.0, 2.0, 2.0]))
    assert m.constraints.data["cx"]._coef_dirty in (True, False)

    diff = ModelDiff.from_snapshot(snap_pre, m, same_model=True)
    snap_pre.advance(diff, m)
    assert m.constraints.data["cx"]._coef_dirty is False

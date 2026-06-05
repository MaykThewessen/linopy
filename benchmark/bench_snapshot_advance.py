"""Microbench: ModelSnapshot.advance vs full re-capture.

Builds a multi-container model. Mutates a small fraction of one container's
RHS, then measures:
  (A) full re-capture: ``ModelSnapshot.capture(model)``
  (B) incremental advance: ``snap.advance(diff, model)``

Reports median of N runs at three scales. Run as a script:

    python benchmark/bench_snapshot_advance.py
"""

from __future__ import annotations

import statistics
import time
from typing import Callable

import numpy as np

from linopy import Model
from linopy.persistent import ModelDiff, ModelSnapshot


def _build_model(n_per_container: int, n_containers: int) -> Model:
    m = Model()
    for i in range(n_containers):
        x = m.add_variables(
            0.0, 10.0, coords=[range(n_per_container)], name=f"x{i}"
        )
        m.add_constraints(2 * x >= 1.0, name=f"c{i}")
    m.add_objective(sum((m.variables[f"x{i}"].sum() for i in range(n_containers))))
    return m


def _median_ms(fn: Callable[[], None], reps: int = 7) -> float:
    fn()  # warmup
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(samples)


def bench(n_per_container: int, n_containers: int) -> None:
    """Measure only the snapshot-refresh step (post-apply book-keeping).

    In ``Solver._update_locked`` the diff is already computed for the
    backend apply step; the snapshot is then either re-captured fresh or
    advanced. This bench isolates that final refresh choice. The diff
    here is reused across reps (representative of the steady-state of a
    warm-update loop where each iteration produces a fresh diff and a
    fresh refresh).
    """
    m = _build_model(n_per_container, n_containers)
    snap_recap = ModelSnapshot.capture(m)
    snap_adv = ModelSnapshot.capture(m)

    m.constraints["c0"].update(rhs=np.full(n_per_container, 0.5))
    diff = ModelDiff.from_snapshot(snap_recap, m, same_model=True)

    def do_recapture() -> None:
        ModelSnapshot.capture(m)

    def do_advance() -> None:
        snap_adv.advance(diff, m)

    t_recap = _median_ms(do_recapture)
    t_adv = _median_ms(do_advance)

    n_rows = n_per_container * n_containers
    print(
        f"n_per={n_per_container:>7d} n_containers={n_containers:>3d} "
        f"n_rows={n_rows:>9d}  recapture={t_recap:7.2f} ms  "
        f"advance={t_adv:7.2f} ms  speedup={t_recap / t_adv:5.1f}x"
    )


if __name__ == "__main__":
    print("Mutating 1 of N constraint containers; measuring snapshot refresh only.")
    bench(n_per_container=1_000, n_containers=4)
    bench(n_per_container=10_000, n_containers=10)
    bench(n_per_container=50_000, n_containers=20)

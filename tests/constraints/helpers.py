import numpy as np

from backend.constraints.feasibility import Problem
from backend.optimization.encoding import FleetRoute


def make_problem(
    demand, capacity, duration=None, windows=None, service=None, shift_end=1e9, rho_max=1.0
):
    """Small hand-built Problem: `demand` is per customer (N,), depot prepended."""
    n = len(demand)
    dur = np.full((n + 1, n + 1), 10.0) if duration is None else np.asarray(duration, float)
    np.fill_diagonal(dur, 0.0)
    w = np.full((n, 2), [0.0, 1e9]) if windows is None else np.asarray(windows, float)
    cap = np.atleast_1d(np.asarray(capacity, float))
    return Problem(
        duration=dur,
        demand=np.concatenate([[0.0], demand]),
        service=np.concatenate([[0.0], np.zeros(n) if service is None else service]),
        windows=np.vstack([[0.0, np.inf], w]),
        capacity=cap,
        shift_end=np.full(len(cap), float(shift_end)),
        t_start=np.zeros(len(cap)),
        rho_max=rho_max,
    )


def fleet(*routes, n=None):
    """FleetRoute from bare stop lists, e.g. fleet([1, 2, 3], [4])."""
    rs = [np.array([0, *r, 0], np.int64) for r in routes]
    n = n or max(max(r, default=0) for r in routes)
    a = np.full(n, -1, np.int64)
    for v, r in enumerate(routes):
        a[np.asarray(r, int) - 1] = v
    return FleetRoute(rs, a)

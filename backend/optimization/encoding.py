"""Continuous particle <-> discrete route-set encoding (random-key style) [SPEC v4 6-9].

Particle X_i = [Y_i | Z_i] in [0, 1]^(2N): Y = assignment keys, Z = ordering keys.
A swarm is one array X of shape (M, 2N); every function here takes the whole swarm and
is NumPy-vectorised over particles AND customers. No per-customer Python loop exists in
the decode path; the only Python loop is over vehicles when materialising route lists.

Index conventions (used everywhere downstream):
  * customers are 0..N-1 in key/assignment arrays, 1..N in routes (depot = 0, matching
    the cost matrices from `road.cost_matrix`);
  * vehicles are 0..M_veh-1 (spec's a_j is one-indexed; see `assign`).

CRITICAL INVARIANT [SPEC 8.3 / v4 10]: `decode` is a PURE function. It never writes to
X. Repair / 2-opt act on its OUTPUT (FleetRoute / dense order arrays), never on X,
pbest, gbest or mbest. `test_encoding.py` asserts X is bit-identical after decode.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FleetRoute:
    """One particle's decoded plan: `routes[v] = [0, j1, ..., jk, 0]` (matrix indices,
    depot 0 at both ends) for each vehicle v [SPEC v4 8]. `assignment[j]` is the
    zero-indexed vehicle of customer j (0..N-1)."""

    routes: list[np.ndarray]
    assignment: np.ndarray


def dim(n_customers: int) -> int:
    """Particle dimension D = 2N [SPEC v4 6]."""
    return 2 * n_customers


def split(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(Y, Z) views of a (M, 2N) swarm: Y = X[:, :N], Z = X[:, N:] [SPEC v4 6]."""
    X = np.atleast_2d(X)
    n = X.shape[1] // 2
    return X[:, :n], X[:, n:]


def assign(Y: np.ndarray, n_vehicles: int) -> np.ndarray:
    """Zero-indexed vehicle per customer, a = min(M_veh - 1, floor(M_veh * y)) [SPEC v4 7].

    Spec writes a_j = 1 + min(M_veh - 1, floor(M_veh * y_j)) (one-indexed); this
    module drops the "1 +" so `a` indexes arrays directly.
    Example: y_j = 0.62, M_veh = 20 -> floor(12.4) = 12 here, spec's a_j = 13.
    Shape (M, N) int64 in, same shape out; y = 1.0 maps to the last vehicle.
    """
    return np.minimum(n_vehicles - 1, np.floor(n_vehicles * Y)).astype(np.int64)


def to_binary_assignment(a: np.ndarray, n_vehicles: int) -> np.ndarray:
    """x_jv = 1 if a_j == v else 0 [SPEC v4 9]. (N,) -> (M_veh, N); (M, N) -> (M, M_veh, N)."""
    return (a[..., None, :] == np.arange(n_vehicles)[:, None]).astype(np.int8)


def decode_dense(X: np.ndarray, n_vehicles: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised decode of a whole swarm -> (assignment, order, ptr) [SPEC v4 7-8].

    assignment: (M, N) vehicle of each customer (see `assign`).
    order:      (M, N) customers 0..N-1 sorted by (vehicle, z) per particle, i.e. all
                vehicle-0 stops in z order, then vehicle 1, ... (one lexsort per swarm).
    ptr:        (M, M_veh + 1) segment bounds: order[m, ptr[m, v]:ptr[m, v + 1]] is pi_v.
    Pure: X is never written.
    """
    Y, Z = split(X)
    M, N = Y.shape
    a = assign(Y, n_vehicles)
    order = np.lexsort((Z, a), axis=1)  # primary key a, secondary z  [SPEC v4 8]
    flat = a + n_vehicles * np.arange(M)[:, None]
    counts = np.bincount(flat.ravel(), minlength=M * n_vehicles).reshape(M, n_vehicles)
    ptr = np.zeros((M, n_vehicles + 1), dtype=np.int64)
    np.cumsum(counts, axis=1, out=ptr[:, 1:])
    return a, order, ptr


def decode(X: np.ndarray, n_vehicles: int) -> list[FleetRoute]:
    """One `FleetRoute` per particle: R_v = [0, pi_v + 1 ..., 0] [SPEC v4 8].

    Wraps `decode_dense`; the only Python loop is over (particle, vehicle) slices, never
    over customers. Pure: X is never written (see module docstring).
    """
    a, order, ptr = decode_dense(X, n_vehicles)
    order1 = order + 1  # matrix indices: depot 0, customers 1..N
    return [
        FleetRoute(
            routes=[
                np.concatenate(([0], order1[m, ptr[m, v] : ptr[m, v + 1]], [0]))
                for v in range(n_vehicles)
            ],
            assignment=a[m],
        )
        for m in range(len(a))
    ]


def encode(
    fleet: FleetRoute, n_customers: int, n_vehicles: int, rng: np.random.Generator
) -> np.ndarray:
    """Inverse of `decode` up to key jitter (spec: warm start, incumbent-to-particle map).

    y_j = (v + u) / M_veh with u ~ U(0, 1) so floor(M_veh * y_j) == v;
    z_j = (rank + u') / len(pi_v) so argsort within v reproduces pi_v.
    Returns (2N,); `decode(encode(r)) == r` holds for any r covering every customer once.
    """
    x = np.empty(dim(n_customers))
    for v, r in enumerate(fleet.routes):
        j = np.asarray(r[1:-1]) - 1  # strip depot ends, back to 0..N-1
        k = len(j)
        x[j] = (v + rng.random(k)) / n_vehicles
        x[n_customers + j] = (np.arange(k) + rng.random(k)) / max(k, 1)
    return x


def random_particle(n_customers: int, rng: np.random.Generator, m: int = 1) -> np.ndarray:
    """Uniform swarm sample in [0, 1]^(m x 2N) (spec: swarm initialisation)."""
    return rng.random((m, dim(n_customers)))


def clip(x: np.ndarray) -> np.ndarray:
    """Project back into [0, 1]^D after a QPSO position update (spec: bound handling)."""
    return np.clip(x, 0.0, 1.0)

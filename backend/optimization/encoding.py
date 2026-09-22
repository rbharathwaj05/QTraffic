"""Continuous particle <-> discrete route-set encoding (random-key style)."""

from __future__ import annotations

import numpy as np

# Particle layout: x = [assignment keys (N,) | ordering keys (N,)], all in [0, 1].
# assignment key a_i -> vehicle floor(a_i * K); ordering key o_i sorts customers within a vehicle.


def dim(n_customers: int) -> int:
    """Particle dimension D = 2N (spec: solution encoding)."""
    raise NotImplementedError


def decode(x: np.ndarray, n_customers: int, n_vehicles: int) -> list[np.ndarray]:
    """x -> list of K routes (customer id arrays, depot implicit at both ends).

    vehicle(i) = min(floor(x[i] * K), K-1); within each vehicle, sort customers by
    x[N + i] ascending (spec: decoding).
    """
    raise NotImplementedError


def encode(
    routes: list[np.ndarray], n_customers: int, n_vehicles: int, rng: np.random.Generator
) -> np.ndarray:
    """Inverse of `decode` up to key jitter: a_i = (k + u)/K with u ~ U(0,1) for vehicle k,
    o_i = rank / len(route) + small jitter, so `decode(encode(r)) == r`
    (spec: warm start, incumbent-to-particle mapping).
    """
    raise NotImplementedError


def random_particle(n_customers: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform sample in [0, 1]^D (spec: swarm initialisation)."""
    raise NotImplementedError


def clip(x: np.ndarray) -> np.ndarray:
    """Project back into [0, 1]^D after a QPSO position update (spec: bound handling)."""
    raise NotImplementedError

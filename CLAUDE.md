# qtraffic — standing instructions for Claude Code

Read fully before touching code. Dense by design.

## Standing invariants

- Every optimization-relevant formula in code MUST carry a comment tag referencing its
  spec section, e.g. `# [SPEC 7.4] QPSO position update`.
- NEVER let repair or 2-opt write back into pbest/gbest/mbest — those remain raw key
  vectors in [0,1]^2N permanently. This is the single most important invariant in the
  whole system (spec §8.3 / v4 §10).
- All hot loops (decode, repair, fitness) must be NumPy-vectorized or Numba-jitted.
  A `for particle in swarm: ...` Python loop over >1 particle is a defect, not a style
  choice — flag it and rewrite before proceeding.
- Every stochastic component (φ, u, s, breeding noise, warm-start perturbation) must
  accept a numpy Generator/seed so runs are reproducible for benchmarking.
- Runtime/wall-clock is NEVER part of the fitness function F. It is measured and
  logged on a separate axis. If you catch yourself adding a time penalty into F, stop.
- Objective normalization bounds (T_min/max, D_min/max, C_min/max, R_min/max) are
  computed ONCE per scenario and frozen. Never recompute them mid-run.
- Distinguish M (swarm particle count) from M_veh (vehicle count) in all code — never
  reuse the same variable name for both.

1. `backend/config.py::QTrafficConfig` is the only place a tunable lives. Never hard-code a
   parameter that exists there. New tunables go there first, with default + spec citation.
2. Weight groups sum to 1 and are enforced by `QTrafficConfig.validate()`:
   `w_t+w_d+w_c+w_r`, `eta_a+eta_o`, `warm_fraction+diverse_fraction`.
3. Spec ranges are enforced: `r_E in [0.10,0.20]`, `r_B in [0.05,0.10]`, `r_inject in [0.20,0.30]`,
   `rho_max = 0.95`, `theta_soft <= theta_hard <= theta_override`.
4. Contracts before bodies. Every function in `optimization/`, `constraints/`, `road/`, `traffic/`
   has a signature + docstring stating the formula. Implement what the docstring says; if the
   docstring is wrong, fix the docstring in the same commit and say why.
5. Fitness is minimised. `F = w_t T/T_ref + w_d D/D_ref + w_c V/V_ref + w_r R`; V is the
   residual violation *after* repair (max `MAX_REPAIR_ITERATIONS=8`), weighted by `w_p`.
6. Particle encoding is random-key: `x in [0,1]^(2N)`, first N = assignment keys, last N =
   ordering keys. `decode(encode(r)) == r` must hold.
7. Re-planning goes through `traffic/controller.py::ReplanController` only. No module calls the
   optimiser directly in response to an event. Cooldown `T_cool=120 s` sim time; override at
   `theta_override`.
8. Wall-clock budgets: local re-plan `< T_response_local (10 s)`, fleet `< T_response_fleet (30 s)`.
   Optimiser `run()` must honour `time_budget_s`.
9. All algorithms (`QPSO`, `EBQPSO`, `PSO`, `GA`, `ACO`) expose the same
   `run(T, time_budget_s) -> SwarmResult` so `benchmark.py` treats them uniformly.
10. Randomness: every stochastic function takes `rng: np.random.Generator`. No global seeds.
11. Hot loops are NumPy/numba over arrays from `road/osm_loader.graph_to_arrays`; no networkx
    calls inside fitness or position updates.
12. Import name is `backend` (`from backend.config import QTrafficConfig`). Package name on PyPI
    metadata is `qtraffic`.
13. `make check` (ruff + black + pytest) must pass before any commit. Stub tests are `skip`,
    never deleted; flip them to real tests when the body lands.
14. Sim time is float seconds from scenario start. Wall-clock time is only used for budgets.

## Module map

| Path | Owns |
|---|---|
| `backend/config.py` | `QTrafficConfig`: every tunable, defaults, `validate()`. |
| `backend/main.py` | FastAPI app + router mounting only. |
| `backend/api/` | HTTP/WS endpoints: `fleet`, `traffic` (event injection), `optimization` (trigger/status), `websocket` (live state feed). No logic. |
| `backend/road/` | OSM graph load/cache (`osm_loader`), OSRM HTTP (`osrm_client`), OD cost matrix + traffic factors (`cost_matrix`), OD-pair↔edge index (`path_index`), haversine/snap/polyline (`geometry`). |
| `backend/optimization/` | Random-key `encoding`, `fitness` (+`ProblemContext`), `qpso` base, `ebqpso` (elite breeding, diversity injection, warm start), baselines `pso`/`ga`/`aco`, `benchmark` harness incl. OR-Tools reference. |
| `backend/constraints/` | `feasibility` predicates (capacity, time windows, insertion), `checker` violation magnitudes → `ViolationReport`, `repair` eject/reinsert loop + residual penalty. |
| `backend/traffic/` | `events` model + generators, `congestion` BPR/diurnal/factor composition, `simulator` event queue → edge factors → cost matrix, `controller` hysteresis/cooldown re-plan gate. |
| `backend/fleet/` | Vehicle/customer entities, current plan (`route_manager`), mutable fleet `state`. Contract TBD (phase 1). |
| `backend/simulation/` | Sim `clock`, orchestration `engine` (traffic step → controller → optimiser → fleet), `replay` of recorded runs. Contract TBD (phase 1). |
| `frontend/` | Vite + React placeholder (`package.json` only). |
| `data/` | `city` (GraphML/OSRM cache), `customers`, `vehicles`, `scenarios` (event scripts). |
| `benchmarks/` | CSV output of `optimization/benchmark.py`. |
| `tests/` | Mirrors `backend/`; one `test_<module>.py` per module. |

## Commands

```
pip install -e .[dev]
make check          # ruff check . && black --check . && pytest -q
docker compose up   # redis + osrm + backend
```

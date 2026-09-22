# qtraffic

Dynamic fleet-routing platform: QPSO / EB-QPSO optimisation of a capacitated vehicle
routing problem with time windows (CVRPTW) over a live OpenStreetMap road graph, with
simulated traffic events driving incremental re-planning.

Phases 0–5 are implemented (road graph, cost matrices, fleet data model, random-key
encoding, canonical fitness). Constraints, optimisers, traffic simulation, API and
frontend are scaffolded and still raise `NotImplementedError`. `CLAUDE.md` holds the
standing invariants; this file describes what exists and how it fits together.

---

## 1. Tech stack

| Layer | Technology | Used for |
|---|---|---|
| Language | Python 3.11+ | whole backend |
| Numerics | NumPy (Numba planned for kernels) | vectorised decode / fitness over the whole swarm |
| Road graph | osmnx + networkx | OSM download, simplification, largest-SCC pruning, GraphML cache, KD-tree snapping |
| Routing engine | OSRM (`osrm/osrm-backend`, MLD) | one-time `/table` (OD durations, distances) and `/route` (path node annotations) |
| Shared state | Redis 7 | monotone `traffic_version` per scenario (in-process fallback when no client) |
| API | FastAPI + uvicorn + websockets | HTTP/WS endpoints (scaffold) |
| Reference solver | OR-Tools | optimality reference in the benchmark harness (planned) |
| Frontend | Vite + React | live map / dashboard (placeholder) |
| Tooling | ruff, black, pytest, docker compose | `make check`, local services |

---

## 2. Repository layout

```
backend/
  config.py            QTrafficConfig: every tunable, defaults, validate()
  main.py              FastAPI app (health only)
  road/                Phase 1-2: graph, OSRM, cost matrices, path index, scenario precompute
  fleet/               Phase 3: Customer / Vehicle schema, scenario generator
  optimization/        Phase 4-7: encoding, fitness, qpso; ebqpso/pso/ga/aco stubs
  constraints/         Phase 6: feasibility, checker, repair
  traffic/             stub: events, congestion, simulator, controller
  simulation/          stub: clock, engine, replay
  api/                 stub: fleet, traffic, optimization, websocket
scripts/
  inspect_graph.py     print node/edge counts and depot for the configured city
  generate_scenario.py write S1..S5 customer/vehicle/depot JSON
  run_qpso.py          run QPSO on one scenario, write convergence curves to benchmarks/
data/
  city/                GraphML cache, OSM .pbf, OSRM .osrm* (gitignored)
  scenarios/<name>/    customers.json, vehicles.json, depot.json (committed)
                       matrices.npz, path_index.pkl, points.npy (gitignored, OSRM-derived)
tests/                 mirrors backend/, one test_<module>.py per module
```

---

## 3. Architecture

### 3.1 Data flow

```
 OSM (osmnx)                     OSRM (docker)
     |                                |
     v                                v
 road/osm_loader ----------> road/osrm_client
   MultiDiGraph                 table()  route()
   GraphML cache                  |        |
   snap_points_to_nodes           v        v
     |                    road/cost_matrix   road/path_index
     |                    D, T_base (frozen) pair -> edges, edge -> pairs
     |                    factor, TrafficVersion    CSR view, invalidate_edge
     |                            \        /
     v                             v      v
 fleet/scenario            road/build_scenario  ->  data/scenarios/<id>/
   Customer, Vehicle              |
   S1..S5 JSON                    v
     |                    fitness.TrafficState  (duration, distance, congestion)
     |                            |
     v                            v
 optimization/encoding  ---->  optimization/fitness.evaluate  ---->  F per particle
   X (M, 2N) random keys          T, D, C gathers + R' vs deployed plan
   decode_dense -> (a, order, ptr)     normalised by frozen Bounds
```

### 3.2 Road layer (`backend/road/`)

* **`osm_loader`** — `load_road_graph(config)` downloads or reads a cached drive
  network (`city_query` or `city_bbox`), keeps the largest strongly connected
  component, adds `speed_kph` / `travel_time` per edge. `snap_points_to_nodes` is one
  batched KD-tree query. `graph_to_arrays` flattens to `(lat, lon, edge_uv, length)`
  for later Numba kernels.
* **`osrm_client`** — `OSRMClient.table()` tiles source×destination blocks so each
  request carries at most `max_table_size` coordinates (never one call per pair);
  `route()` returns duration, distance, geometry and OSM node annotations.
* **`cost_matrix`** — `build_cost_matrices` runs `/table` **once per scenario** and
  rejects unroutable pairs (no NaN reaches the optimiser). `CostMatrix` holds frozen
  `duration_s` (T_base), `distance_m` (D), a mutable `factor` and a `TrafficVersion`.
  `update_factors` recomputes `factor[i, j] = Σ t0_e f_e / Σ t0_e` over the path's
  edges with one `bincount` pass and bumps the version only when something changed.
  `effective_duration = duration_s * factor` is the optimiser-facing lookup: pure
  array, zero OSRM calls.
* **`path_index`** — `PathIndex` maps every OD pair to its edge ids and inverts that
  map, so a traffic event on edge `e` finds affected pairs in O(1)
  (`invalidate_edge`). Pairs whose OSRM path left the simplified graph are marked
  `partial`; unknown edges fall back to that set. Persisted with pickle.
* **`build_scenario`** — `build_scenario(id, n, config, rng)` writes
  `matrices.npz`, `path_index.pkl`, `points.npy`; `load_scenario` reads them back with
  no OSRM or graph access.

### 3.3 Fleet layer (`backend/fleet/`)

* **`Customer`** (frozen dataclass): `customer_id, lat, lon, node_id, demand,
  service_time, time_window_start, time_window_end`.
* **`Vehicle`**: `vehicle_id, capacity, current_node, shift_end, status,
  assigned_customers, remaining_route`.
* **`scenario.generate`** — random points inside the graph bbox → one batched snap →
  distinct nodes excluding the depot. Demand and service time are uniform over
  config ranges. Time windows: `tw_anytime_fraction` of customers get `[0, shift_end)`,
  the rest a window of width `U(tw_width_range)` placed uniformly inside the shift.
  `check_feasible` raises unless `Σ demand ≤ rho_max · Σ capacity`.
* Scalability ladder (all generated on the Chennai graph and committed):

  | Scenario | Customers | Vehicles | Purpose |
  |---|---|---|---|
  | S1 | 10 | 5 | correctness |
  | S2 | 50 | 10 | integration |
  | S3 | 100 | 20 | optimisation |
  | S4 | 300 | 50 | target |
  | S5 | 300 | 100 | final demo |

### 3.4 Encoding (`backend/optimization/encoding.py`)

Random-key representation, spec v4 §6–9. A particle is `X = [Y | Z] ∈ [0,1]^(2N)`;
a swarm is one `(M, 2N)` array and every function is vectorised over it.

* `assign(Y, M_veh) = min(M_veh − 1, floor(M_veh · Y))` (zero-indexed; spec's
  `y = 0.62, M_veh = 20 → 13` is `12` here).
* `decode_dense(X, M_veh) → (assignment, order, ptr)`: one `lexsort` over
  `(vehicle, z)`, `bincount` + `cumsum` for segment bounds. No Python loop over
  particles, customers or vehicles.
* `decode(X, M_veh) → list[FleetRoute]` materialises `routes[v] = [0, j…, 0]`
  (depot = 0, customers 1..N, matching the cost-matrix indices).
* `encode(FleetRoute)` is the inverse up to key jitter: `decode(encode(r)) == r`.
* `to_binary_assignment` gives `x_jv` as `(M_veh, N)` or `(M, M_veh, N)`.
* **Invariant:** `decode` is pure; repair and 2-opt act on its output, never on `X`,
  pbest, gbest or mbest.

### 3.5 Fitness (`backend/optimization/fitness.py`)

Single objective for every algorithm, minimised:

```
F      = w_t·T' + w_d·D' + w_c·C' + w_r·R'        X' = (X − X_min) / (X_max − X_min)
F_eval = F + w_p·P                                P = residual violation after repair
```

* `T = Σ T_path_ij(t)`, `D = Σ D_path_ij`, `C = Σ ρ_ij(t)·D_ij` are one fancy-index
  gather each over a flat visit sequence `[0, π_0, 0, 0, π_1, 0, …]` built from
  `decode_dense` output; nothing is re-routed here.
* `R'` (spec v4 §19) compares against the **deployed** plan passed in as an argument:
  `A = share of customers that changed vehicle`, `O = mean |pos_new − pos_cur| / L_j`
  over customers that stayed, `R' = η_a·A + η_o·O`.
* `Bounds` are computed **once per scenario** by `compute_normalization_bounds` (random
  sample of particles) and passed explicitly; they are never recomputed inside a loop.
* Wall-clock time is never part of `F`.
* `evaluate(plans, traffic, current, bounds, cfg, penalties=None) → (M,)` accepts the
  dense tuple (preferred) or a `list[FleetRoute]`.

### 3.6 Configuration

`QTrafficConfig` in `backend/config.py` is the only home for tunables: swarm size and
iteration budget, elite/breeding fractions, objective weights (`w_t + w_d + w_c + w_r
= 1`), route-change split (`η_a + η_o = 1`), repair penalty `w_p`, hysteresis
thresholds, cooldown, warm-start split, diversity injection, stagnation window,
response budgets, `rho_max`, city/graph settings, scenario-generation ranges and
infra URLs. `validate()` enforces the sum-to-one groups and spec ranges.

---

## 4. Workflow

### 4.1 Setup

```bash
pip install -e .[dev]
make check                 # ruff check . && black --check . && pytest -q
```

### 4.2 Road data and OSRM (one-time per city)

```bash
python scripts/inspect_graph.py                 # downloads + caches the osmnx graph
# put a Geofabrik extract at data/city/city.osm.pbf, then:
make osrm-build                                 # osrm-extract / partition / customize
docker compose up osrm redis                    # OSRM on :5000, Redis on :6379
```

### 4.3 Scenario generation

```bash
python scripts/generate_scenario.py             # S1..S5 -> data/scenarios/<name>/*.json
python scripts/generate_scenario.py --only S2 --seed 3
python -m backend.road.build_scenario --id S2 --n 50   # OSRM matrices + path index
```

### 4.4 Per-scenario setup (what an optimiser will do)

1. `matrix, index, points = road.build_scenario.load_scenario(id)`
2. `traffic = fitness.TrafficState.from_cost_matrix(matrix)`
3. `bounds = fitness.compute_normalization_bounds(traffic, N, M_veh, rng)` — once
4. Loop: `plans = encoding.decode_dense(X, M_veh)`;
   `F = fitness.evaluate(plans, traffic, current_plan, bounds, cfg)`

### 4.5 Live traffic update path

`traffic.congestion` produces per-edge factors → `cost_matrix.update_factors(matrix,
edge_factor, index)` rewrites `factor` and bumps `TrafficVersion` →
`TrafficState.from_cost_matrix(matrix)` gives the optimiser a fresh matrix.
`D` and `T_base` never change after build.

### 4.6 Development rules (short form; full list in `CLAUDE.md`)

* Every optimisation-relevant formula carries a `[SPEC …]` comment tag.
* Hot loops are NumPy/Numba over arrays; a per-particle Python loop is a defect.
* Every stochastic function takes `rng: np.random.Generator`.
* Normalisation bounds are frozen per scenario; repair never writes back into
  pbest/gbest/mbest.
* `make check` must pass before every commit.

---

## 5. Status by phase

| Phase | Scope | State |
|---|---|---|
| 0 | scaffold, config, stub tests | done |
| 1 | OSM graph loader, geometry helpers | done |
| 2 | OSRM client, cost matrices, path index, scenario precompute | done |
| 3 | Customer/Vehicle schema, scenario generator S1–S5 | done |
| 4 | random-key encode/decode, vectorised over the swarm | done |
| 5 | canonical normalised fitness `evaluate()` | done |
| 6 | feasibility, fixed-order checker, bounded repair | done |
| 7 | plain QPSO engine, 2-opt polish, `scripts/run_qpso.py` | done |
| 8+ | EB-QPSO / baselines, traffic simulation, controller, API, frontend | stub |

Test suite at Phase 7: `91 passed, 18 skipped` (skips are Phase 0 stub tests, kept
until their bodies land).

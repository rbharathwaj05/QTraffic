# qtraffic

Dynamic fleet-routing platform: QPSO / EB-QPSO optimisation over a live OSM road graph
with simulated traffic events.

Phase 0 = scaffold only. No algorithm logic yet; every optimisation/constraint/road/traffic
function raises `NotImplementedError`. See `CLAUDE.md` for invariants and module map.

```
pip install -e .[dev]
make check      # ruff + black + pytest
```

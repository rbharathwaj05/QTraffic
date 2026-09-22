"""Re-plan controller: decides if/when/how wide to re-optimise after traffic changes."""

# STATUS (through Phase 6): contract only. Signatures + docstrings define the formulas;
# every body raises NotImplementedError until its phase lands. Tests for this module skip.

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from backend.config import QTrafficConfig


class Scope(str, Enum):
    NONE = "none"
    LOCAL = "local"  # only affected vehicles, budget T_response_local
    FLEET = "fleet"  # whole fleet, budget T_response_fleet


@dataclass(frozen=True)
class Decision:
    scope: Scope
    delta: float  # relative cost increase that triggered it
    reason: str


class ReplanController:
    """Hysteresis + cooldown gate in front of the optimiser (spec: re-plan controller)."""

    def __init__(self, cfg: QTrafficConfig) -> None:
        self.cfg = cfg
        self.last_replan_t: float = -np.inf
        self.armed: bool = False

    def cost_delta(self, cost_now: float, cost_planned: float) -> float:
        """delta = (cost_now - cost_planned) / cost_planned (spec: re-plan trigger metric)."""
        raise NotImplementedError

    def decide(
        self, t_sim: float, delta: float, affected_vehicles: int, total_vehicles: int
    ) -> Decision:
        """delta >= theta_override            -> FLEET, ignore cooldown
        delta >= theta_hard                  -> LOCAL/FLEET if cooldown elapsed
        theta_soft <= delta < theta_hard     -> arm; fire on next call still above theta_soft
        delta < theta_soft                   -> disarm, NONE
        LOCAL when affected/total <= 0.5 else FLEET (spec: hysteresis, cooldown, scope)."""
        raise NotImplementedError

    def record(self, t_sim: float) -> None:
        """Mark a re-plan as executed at t_sim; resets armed and cooldown clock."""
        raise NotImplementedError

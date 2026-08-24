"""Strategy selector -- the central controller of the *proposed* pipeline (STUB).

Takes drift detection + drift-type classification output and routes to a
retraining strategy proportional to the drift type:

    sudden        -> full retrain on recent data
    gradual       -> incremental / partial model update
    incremental   -> incremental / partial model update
    recurring     -> retrieve a cached prior model (near-zero cost)

and logs the decision plus its cost (wall-clock training time; FLOPs/energy proxy
where available). This is what the naive always-full-retrain baseline is compared
against.

The routing table is stable; the *actions* depend on components still being
designed (the incremental-update mechanism and the cache retrieval), so the
bodies are stubbed until those are settled.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class Strategy(str, Enum):
    FULL_RETRAIN = "full_retrain"
    INCREMENTAL_UPDATE = "incremental_update"
    CACHE_RETRIEVAL = "cache_retrieval"


# Drift type -> strategy. Central, explicit, easy to cite in the paper.
STRATEGY_BY_DRIFT_TYPE = {
    "sudden": Strategy.FULL_RETRAIN,
    "gradual": Strategy.INCREMENTAL_UPDATE,
    "incremental": Strategy.INCREMENTAL_UPDATE,
    "recurring": Strategy.CACHE_RETRIEVAL,
}


@dataclass
class StrategyDecision:
    index: int
    drift_type: str
    strategy: Strategy
    cost_s: float = 0.0
    cache_hit: Optional[bool] = None
    note: str = ""


@dataclass
class StrategySelector:
    """Route a classified drift to an action and log the decision (stubbed)."""

    log: List[StrategyDecision] = field(default_factory=list)

    def route(self, drift_type: str) -> Strategy:
        return STRATEGY_BY_DRIFT_TYPE[drift_type]

    def apply(self, drift_type: str, index: int, **ctx) -> StrategyDecision:
        """Execute the chosen strategy and return a logged decision. STUB.

        Depends on the incremental-update mechanism and cache retrieval, both of
        which are pending design decisions.
        """
        raise NotImplementedError(
            "Strategy actions depend on the (pending) incremental-update and "
            "cache-retrieval designs. Routing table is defined; wire actions once "
            "those components are approved."
        )

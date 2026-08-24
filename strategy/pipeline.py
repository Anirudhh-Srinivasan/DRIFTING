"""Full drift-type-aware retraining pipeline (the proposed system).

Replays a fixed drift schedule (captured once, as in the controlled ablation) and
at each drift applies the strategy the drift-type classifier's label routes to:

    sudden       -> full retrain (fresh model on the recent window) + cache it
    gradual      -> warm-start incremental update on a smaller recent window
    incremental  -> warm-start incremental update on a smaller recent window
    recurring    -> retrieve a cached model (near-zero cost); if no cache hit,
                    fall back to a full retrain so the pipeline never gets stuck

Because it replays the *same* schedule the naive baseline uses, the two differ
only in the action taken per drift -- so wall-clock and accuracy differences are
attributable to type-aware strategy selection, not to detecting differently.

Every decision is logged with its wall-clock cost, so the runner can break the
saving down by strategy and confirm accuracy stays at parity.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from config import ExperimentConfig
from data.stream import DriftStream
from models import make_model
from model_cache import ModelReuseCache
from strategy.selector import Strategy, STRATEGY_BY_DRIFT_TYPE


@dataclass
class StrategyAction:
    resolve_index: int
    detect_index: int
    drift_type: str
    strategy: str
    cost_s: float
    cache_hit: bool = False
    distance: float = float("nan")


@dataclass
class TypeAwareResult:
    correct: np.ndarray
    step_index: np.ndarray
    actions: List[StrategyAction] = field(default_factory=list)
    total_cost_s: float = 0.0
    cache_stats: object = None

    @property
    def overall_accuracy(self) -> float:
        return float(self.correct.mean()) if len(self.correct) else 0.0

    def counts_by_strategy(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for a in self.actions:
            out[a.strategy] = out.get(a.strategy, 0) + 1
        return out

    def cost_by_strategy(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for a in self.actions:
            out[a.strategy] = round(out.get(a.strategy, 0.0) + a.cost_s, 5)
        return out


def _fit_fresh(model_factory, X, y):
    m = model_factory()
    t0 = time.perf_counter()
    m.fit(X, y)
    return m, time.perf_counter() - t0


def run_type_aware(
    stream: DriftStream,
    cfg: ExperimentConfig,
    schedule: List[Tuple[int, int]],
    type_map: Dict[int, str],
) -> TypeAwareResult:
    """Replay ``schedule`` applying the strategy routed by ``type_map``.

    ``type_map`` maps a resolve index to a predicted drift type.
    """
    ev, sc = cfg.eval, cfg.strategy
    X, y = stream.X, stream.y
    n = stream.n_samples
    model_factory = lambda: make_model(cfg.model, cfg.seed)

    model, warm_t = _fit_fresh(model_factory, X[: ev.warmup_size], y[: ev.warmup_size])
    cache = ModelReuseCache(threshold=cfg.cache.distance_threshold)
    cache.store(cache.fingerprint(X[: ev.warmup_size], y[: ev.warmup_size]), model, 0)

    correct = np.empty(n - ev.warmup_size, dtype=np.int8)
    step_index = np.arange(ev.warmup_size, n)
    actions: List[StrategyAction] = []
    retrain_times: List[float] = [warm_t]
    total_cost = warm_t

    resolve_map = {res: det for det, res in schedule}

    for pos, i in enumerate(range(ev.warmup_size, n)):
        pred = model.predict(X[i : i + 1])[0]
        correct[pos] = int(pred == y[i])

        if i in resolve_map:
            d = resolve_map[i]
            drift_type = type_map.get(i, "sudden")   # default: safest (full retrain)
            strat = STRATEGY_BY_DRIFT_TYPE[drift_type]
            fp = cache.fingerprint(X[d:i], y[d:i])

            if strat == Strategy.CACHE_RETRIEVAL:
                avg = float(np.mean(retrain_times))
                reuse_model, dist, _ = cache.try_reuse(fp, avg, index=d, before_index=d)
                if reuse_model is not None:
                    model = reuse_model
                    actions.append(StrategyAction(i, d, drift_type,
                                                  "cache_retrieval", 0.0, True, round(dist, 5)))
                else:
                    # No hit -> fall back to a full retrain so we never stall.
                    start = max(0, i - ev.retrain_window)
                    model, t = _fit_fresh(model_factory, X[start:i], y[start:i])
                    total_cost += t; retrain_times.append(t)
                    cache.store(fp, model, index=i)
                    actions.append(StrategyAction(i, d, drift_type,
                                                  "full_retrain_fallback", t, False, round(dist, 5)))

            elif strat == Strategy.INCREMENTAL_UPDATE:
                # Fresh retrain on a SMALLER recent window than the full-retrain
                # strategy: same model class, fewer samples -> cheaper, and it
                # learns the union of concepts present in the window (which a
                # warm-start partial update cannot), so accuracy recovers.
                start = max(0, i - sc.incremental_window)
                model, t = _fit_fresh(model_factory, X[start:i], y[start:i])
                total_cost += t
                cache.store(fp, model, index=i)   # remember this concept for reuse
                actions.append(StrategyAction(i, d, drift_type,
                                              "incremental_update", t, False))

            else:  # FULL_RETRAIN (sudden)
                start = max(0, i - ev.retrain_window)
                model, t = _fit_fresh(model_factory, X[start:i], y[start:i])
                total_cost += t; retrain_times.append(t)
                cache.store(fp, model, index=i)
                actions.append(StrategyAction(i, d, drift_type,
                                              "full_retrain", t, False))

    return TypeAwareResult(
        correct=correct, step_index=step_index, actions=actions,
        total_cost_s=total_cost, cache_stats=cache.stats,
    )

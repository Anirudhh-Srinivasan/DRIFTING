"""Controlled ablation harness: isolate the cache effect exactly.

The live cache-reuse pipeline (``strategy/cache_reuse.py``) and the naive baseline
each run their own detector, so they can end up acting at slightly different
points -- which muddies "how many retrains did the cache actually save?".

This module removes that confound. A single prequential loop, ``run_policy``, is
parameterised by an *action policy* and can either:

* **detect live** (``schedule=None``) -- run the detector with cooldown/settle and
  *record* the resulting schedule of drift resolutions; or
* **replay** a fixed ``schedule`` -- act at exactly those pre-recorded points,
  detector disabled.

The ablation then is: capture the schedule once from the ``always_retrain``
policy, and replay that *same* schedule under both ``always_retrain`` and
``cache_reuse``. Detection points are identical by construction, so the only
difference is the action taken at each point. Hence:

    retrains_avoided  ==  number of cache reuses      (pure caching effect)

and any accuracy difference is attributable solely to reuse-vs-retrain, not to
the two pipelines detecting differently.

Both policies share the same warm-up model (same seed) and the same retrain
window, so a non-recurrence point produces identical retraining in both -- the
pipelines only diverge where the cache legitimately reuses.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from config import ExperimentConfig
from data.stream import DriftStream
from detectors import make_detector
from models import make_model
from model_cache import ModelReuseCache

ALWAYS_RETRAIN = "always_retrain"
CACHE_REUSE = "cache_reuse"


@dataclass
class Action:
    resolve_index: int      # where the action was taken (detect + settle)
    detect_index: int       # where the drift was detected
    action: str             # "retrain" | "reuse"
    distance: float         # concept distance to nearest cached concept (nan if n/a)
    cost_s: float           # wall-clock spent (0 for reuse)
    matched_index: int = -1


@dataclass
class PolicyResult:
    policy: str
    correct: np.ndarray
    step_index: np.ndarray
    schedule: List[Tuple[int, int]]      # [(detect_index, resolve_index), ...]
    actions: List[Action] = field(default_factory=list)
    total_retrain_time_s: float = 0.0
    cache_stats: object = None

    @property
    def n_retrains(self) -> int:
        return sum(1 for a in self.actions if a.action == "retrain")

    @property
    def n_reuses(self) -> int:
        return sum(1 for a in self.actions if a.action == "reuse")

    @property
    def overall_accuracy(self) -> float:
        return float(self.correct.mean()) if len(self.correct) else 0.0


def _fit(model_factory, X, y):
    model = model_factory()
    t0 = time.perf_counter()
    model.fit(X, y)
    return model, time.perf_counter() - t0


def run_policy(
    stream: DriftStream,
    cfg: ExperimentConfig,
    policy: str,
    schedule: Optional[List[Tuple[int, int]]] = None,
) -> PolicyResult:
    """Run one policy, either detecting live (records schedule) or replaying one."""
    ev, cc = cfg.eval, cfg.cache
    X, y = stream.X, stream.y
    n = stream.n_samples
    model_factory = lambda: make_model(cfg.model, cfg.seed)

    model, warm_t = _fit(model_factory, X[: ev.warmup_size], y[: ev.warmup_size])

    cache = None
    if policy == CACHE_REUSE:
        cache = ModelReuseCache(threshold=cc.distance_threshold)
        cache.store(cache.fingerprint(X[: ev.warmup_size], y[: ev.warmup_size]),
                    model, index=0)

    correct = np.empty(n - ev.warmup_size, dtype=np.int8)
    step_index = np.arange(ev.warmup_size, n)
    actions: List[Action] = []
    retrain_times: List[float] = [warm_t]
    total_time = warm_t
    recorded: List[Tuple[int, int]] = []

    live = schedule is None
    if live:
        detector = make_detector(cfg.primary_detector, cfg.detectors)
        pending = None
        cooldown_until = ev.warmup_size
    else:
        # resolve_index -> detect_index
        resolve_map = {res: det for det, res in schedule}

    for pos, i in enumerate(range(ev.warmup_size, n)):
        xi = X[i : i + 1]
        pred = model.predict(xi)[0]
        err = int(pred != y[i])
        correct[pos] = 1 - err

        # -- decide whether to resolve/act at this step --------------------- #
        resolve_here = False
        d = None
        if live:
            if pending is not None and (i - pending >= cc.settle or i == n - 1):
                resolve_here, d, pending = True, pending, None
        else:
            if i in resolve_map:
                resolve_here, d = True, resolve_map[i]

        if resolve_here:
            recorded.append((d, i))
            start = max(0, i - ev.retrain_window)
            if policy == CACHE_REUSE:
                fp_new = cache.fingerprint(X[d:i], y[d:i])
                avg_retrain = float(np.mean(retrain_times))
                reuse_model, dist, matched = cache.try_reuse(
                    fp_new, avg_retrain_time_s=avg_retrain, index=d, before_index=d
                )
                if reuse_model is not None:
                    model = reuse_model
                    actions.append(Action(i, d, "reuse", round(dist, 5), 0.0, matched))
                else:
                    model, t = _fit(model_factory, X[start:i], y[start:i])
                    total_time += t
                    retrain_times.append(t)
                    cache.store(fp_new, model, index=i)
                    actions.append(Action(i, d, "retrain", round(dist, 5), t))
            else:  # ALWAYS_RETRAIN
                model, t = _fit(model_factory, X[start:i], y[start:i])
                total_time += t
                retrain_times.append(t)
                actions.append(Action(i, d, "retrain", float("nan"), t))

            if live:
                detector = make_detector(cfg.primary_detector, cfg.detectors)
                cooldown_until = i + cc.settle

        # -- feed the detector (live mode only) ----------------------------- #
        if live:
            drift = detector.update(err)
            if drift and pending is None and i >= cooldown_until:
                pending = i

    return PolicyResult(
        policy=policy,
        correct=correct,
        step_index=step_index,
        schedule=recorded if live else list(schedule),
        actions=actions,
        total_retrain_time_s=total_time,
        cache_stats=cache.stats if cache else None,
    )


def controlled_ablation(stream: DriftStream, cfg: ExperimentConfig):
    """Capture the schedule from always-retrain, replay both policies on it.

    Returns ``(baseline_result, cache_result)`` sharing identical action points.
    """
    baseline = run_policy(stream, cfg, ALWAYS_RETRAIN, schedule=None)
    cache = run_policy(stream, cfg, CACHE_REUSE, schedule=baseline.schedule)
    return baseline, cache

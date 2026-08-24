"""Cache-reuse pipeline: detect -> reuse-cached-model-on-recurrence, else retrain.

This is the recurring-drift half of the proposed system, validated in isolation
before the full drift-type-aware selector is built. It mirrors the naive
baseline's prequential loop exactly, changing only the *response* to a detected
drift:

    baseline:    drift  -> ALWAYS full retrain
    this:        drift  -> fingerprint the new concept;
                            if it matches a cached concept  -> REUSE (near-zero cost)
                            else                             -> full retrain + cache it

So any difference in retrain count / wall-clock cost between this and the baseline
is attributable purely to cache reuse -- a clean ablation for the paper.

Fingerprinting the *new* concept requires a few post-detection samples (the
detector fires partway into the new regime, and we want to summarise the new
distribution, not the old). We therefore gather ``settle`` samples after a
detection before deciding reuse-vs-retrain; during that brief settle the current
model keeps predicting (its errors still count, exactly as in the baseline).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List

import numpy as np

from config import ExperimentConfig
from data.stream import DriftStream
from detectors import make_detector
from models import make_model
from model_cache import ModelReuseCache


@dataclass
class CacheReuseEvent:
    index: int
    action: str            # "reuse" or "retrain"
    distance: float        # concept distance to nearest cached concept
    cost_s: float          # wall-clock spent (0 for reuse)
    matched_index: int = -1


@dataclass
class CacheRunResult:
    pipeline: str
    detector: str
    model: str
    n_samples: int
    warmup_size: int
    correct: np.ndarray
    step_index: np.ndarray
    detections: List[int] = field(default_factory=list)
    decisions: List[CacheReuseEvent] = field(default_factory=list)
    total_retrain_time_s: float = 0.0
    cache_stats: object = None

    @property
    def n_retrains(self) -> int:
        return sum(1 for d in self.decisions if d.action == "retrain")

    @property
    def n_reuses(self) -> int:
        return sum(1 for d in self.decisions if d.action == "reuse")

    @property
    def overall_accuracy(self) -> float:
        return float(self.correct.mean()) if len(self.correct) else 0.0


def _fit(model_factory, X, y):
    model = model_factory()
    t0 = time.perf_counter()
    model.fit(X, y)
    return model, time.perf_counter() - t0


def run_cache_reuse(stream: DriftStream, cfg: ExperimentConfig) -> CacheRunResult:
    ev, cc = cfg.eval, cfg.cache
    X, y = stream.X, stream.y
    n = stream.n_samples

    detector = make_detector(cfg.primary_detector, cfg.detectors)
    model_factory = lambda: make_model(cfg.model, cfg.seed)
    cache = ModelReuseCache(threshold=cc.distance_threshold)

    # Warm-up: train, and seed the cache with the warm-up concept.
    model, warm_t = _fit(model_factory, X[: ev.warmup_size], y[: ev.warmup_size])
    fp0 = cache.fingerprint(X[: ev.warmup_size], y[: ev.warmup_size])
    cache.store(fp0, model, index=0)

    correct = np.empty(n - ev.warmup_size, dtype=np.int8)
    step_index = np.arange(ev.warmup_size, n)
    detections: List[int] = []
    decisions: List[CacheReuseEvent] = []
    retrain_times: List[float] = [warm_t]
    total_time = warm_t

    # When a detection is pending, we wait for `settle` samples before deciding.
    pending_from = None
    # After acting on a drift, ignore new detections for a cooldown (the model is
    # recovering; its residual error would otherwise re-trigger the detector).
    cooldown_until = ev.warmup_size

    for pos, i in enumerate(range(ev.warmup_size, n)):
        xi = X[i : i + 1]
        pred = model.predict(xi)[0]
        err = int(pred != y[i])
        correct[pos] = 1 - err

        # Resolve a pending detection once enough post-drift samples exist.
        if pending_from is not None and (i - pending_from >= cc.settle or i == n - 1):
            d = pending_from
            pending_from = None
            # Fingerprint the NEW concept from strictly post-detection samples
            # [d, i]. The detector fires *after* the true onset, so this window is
            # (almost) entirely the new concept and does not bleed into the old one.
            fp_new = cache.fingerprint(X[d:i], y[d:i])
            avg_retrain = float(np.mean(retrain_times))

            reuse_model, dist, matched = cache.try_reuse(
                fp_new, avg_retrain_time_s=avg_retrain, index=d, before_index=d
            )
            if reuse_model is not None:
                model = reuse_model            # near-zero cost recurrence reuse
                decisions.append(CacheReuseEvent(d, "reuse", round(dist, 5), 0.0, matched))
            else:
                start = max(0, i - ev.retrain_window)
                model, t = _fit(model_factory, X[start:i], y[start:i])
                total_time += t
                retrain_times.append(t)
                cache.store(fp_new, model, index=i)
                decisions.append(CacheReuseEvent(d, "retrain", round(dist, 5), t))

            # Reset the detector so the pre-action error spike can't immediately
            # re-fire, and start a cooldown over the recovery window.
            detector = make_detector(cfg.primary_detector, cfg.detectors)
            cooldown_until = i + cc.settle

        drift = detector.update(err)
        if drift and pending_from is None and i >= cooldown_until:
            detections.append(i)
            pending_from = i

    return CacheRunResult(
        pipeline="cache_reuse",
        detector=cfg.primary_detector,
        model=cfg.model.kind,
        n_samples=n,
        warmup_size=ev.warmup_size,
        correct=correct,
        step_index=step_index,
        detections=detections,
        decisions=decisions,
        total_retrain_time_s=total_time,
        cache_stats=cache.stats,
    )

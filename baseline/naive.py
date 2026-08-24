"""Naive baseline: detect -> ALWAYS full retrain.

This is the comparison point for every headline result. It runs a standard
prequential (test-then-train) loop:

1. Train an initial model on the first ``warmup_size`` samples.
2. For each subsequent sample: predict it, record whether the prediction was
   wrong (the *error* signal), then feed that error to the drift detector.
3. When the detector fires, do a **full retrain** on the most recent
   ``retrain_window`` samples -- regardless of what *kind* of drift it was. That
   indiscriminate response is exactly what the drift-type-aware pipeline is meant
   to improve on.

Everything the evaluation needs is logged: per-step correctness (for the
accuracy-over-stream curve), detection indices, and each retrain's wall-clock
cost and window.

The proposed (type-aware) pipeline will reuse this same loop but swap the
"always full retrain" action for a strategy-selector decision.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, List

import numpy as np

from config import ExperimentConfig
from data.stream import DriftStream
from detectors import make_detector
from models import make_model


@dataclass
class RetrainEvent:
    index: int                 # stream position where retrain was triggered
    train_time_s: float        # wall-clock cost of this retrain
    window_start: int
    window_end: int


@dataclass
class RunResult:
    """Everything downstream evaluation/plots need from one pipeline run."""

    pipeline: str
    detector: str
    model: str
    n_samples: int
    warmup_size: int
    correct: np.ndarray                      # (n-warmup,) 0/1 correctness, post-warmup
    step_index: np.ndarray                   # stream index for each entry in `correct`
    detections: List[int] = field(default_factory=list)
    retrains: List[RetrainEvent] = field(default_factory=list)
    total_retrain_time_s: float = 0.0

    @property
    def n_retrains(self) -> int:
        return len(self.retrains)

    @property
    def overall_accuracy(self) -> float:
        return float(self.correct.mean()) if len(self.correct) else 0.0


def _fit(model_factory: Callable, X, y, seed):
    model = model_factory()
    t0 = time.perf_counter()
    model.fit(X, y)
    return model, time.perf_counter() - t0


def run_naive_baseline(stream: DriftStream, cfg: ExperimentConfig) -> RunResult:
    ev = cfg.eval
    X, y = stream.X, stream.y
    n = stream.n_samples

    detector = make_detector(cfg.primary_detector, cfg.detectors)
    model_factory = lambda: make_model(cfg.model, cfg.seed)

    # 1. Warm-up training.
    model, warm_t = _fit(model_factory, X[: ev.warmup_size], y[: ev.warmup_size], cfg.seed)

    correct = np.empty(n - ev.warmup_size, dtype=np.int8)
    step_index = np.arange(ev.warmup_size, n)
    detections: List[int] = []
    retrains: List[RetrainEvent] = [
        RetrainEvent(index=0, train_time_s=warm_t, window_start=0, window_end=ev.warmup_size)
    ]
    total_time = warm_t

    # 2. Prequential loop over the rest of the stream.
    for pos, i in enumerate(range(ev.warmup_size, n)):
        xi = X[i : i + 1]
        pred = model.predict(xi)[0]
        err = int(pred != y[i])
        correct[pos] = 1 - err

        drift = detector.update(err)
        if drift:
            detections.append(i)
            # 3. ALWAYS full retrain on the most recent window.
            start = max(0, i - ev.retrain_window)
            model, t = _fit(model_factory, X[start:i], y[start:i], cfg.seed)
            total_time += t
            retrains.append(RetrainEvent(index=i, train_time_s=t,
                                         window_start=start, window_end=i))

    return RunResult(
        pipeline="naive_always_retrain",
        detector=cfg.primary_detector,
        model=cfg.model.kind,
        n_samples=n,
        warmup_size=ev.warmup_size,
        correct=correct,
        step_index=step_index,
        detections=detections,
        retrains=retrains,
        total_retrain_time_s=total_time,
    )

"""Feature extraction around a detected drift.

Given the prequential error trajectory, the schedule of drift resolutions, and
the stream, we compute -- for each detected drift -- a small set of interpretable
features that the rule-based (and, later, ML) classifier uses to name the drift
type. These map directly onto the three signals the Methodology names:

* **rate of error increase / detection speed** -> ``ramp_length``, ``error_slope``,
  ``error_jump``. A sudden drift produces a short, steep error rise (the detector
  fires almost immediately); gradual/incremental produce a long, shallow rise.
* **shape of the transition** -> ``oscillation``, ``monotonicity``. During gradual
  drift, samples alternate between the old and new concept, so the error bounces
  (high oscillation, monotonicity ~0.5). Incremental drift slides steadily, so the
  error climbs near-monotonically (low oscillation, high monotonicity).
* **similarity to previously seen concepts** -> ``concept_distance``: the
  symmetric-KL distance from the new concept to the nearest *earlier* concept
  (shared with the model-reuse cache). A recurrence sits near zero.

Features are computed from a window ending at the detection point ``d`` (the rise
happens *before* the detector fires) plus the settled post-detection window
``[d, r]`` for the concept fingerprint. Nothing here looks at ground-truth labels.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Tuple

import numpy as np

from config import ExperimentConfig
from data.stream import DriftStream
from model_cache.fingerprint import compute_fingerprint, ConceptMemory


@dataclass
class DriftFeatures:
    detect_index: int
    resolve_index: int
    ramp_length: int          # #samples the error took to rise before detection
    error_slope: float        # (peak - baseline) / ramp_length
    error_jump: float         # peak rolling error minus pre-drift baseline
    oscillation: float        # sign-change rate of the error rise (0..1)
    monotonicity: float       # fraction of upward steps during the rise (0..1)
    concept_distance: float   # symmetric-KL distance to nearest earlier concept

    def as_dict(self):
        return asdict(self)


def _rolling(x: np.ndarray, w: int) -> np.ndarray:
    if len(x) == 0:
        return x.astype(float)
    k = np.ones(w) / w
    return np.convolve(x.astype(float), k, mode="same")


class DriftFeatureExtractor:
    """Compute :class:`DriftFeatures` for every drift in a schedule."""

    def __init__(self, cfg: ExperimentConfig, roll_window: int = 150,
                 rise_margin: float = 0.05, fp_window: int = 800):
        self.cfg = cfg
        self.roll_window = roll_window
        self.rise_margin = rise_margin
        # Fixed post-detection window for the concept fingerprint, INDEPENDENT of
        # the pipeline's `settle`. This keeps the concept_distance feature on the
        # same scale between calibration streams (large settle) and deployment
        # (small settle), so the calibrated recurring threshold transfers.
        self.fp_window = fp_window

    def extract(
        self,
        stream: DriftStream,
        schedule: List[Tuple[int, int]],
        correct: np.ndarray,
        step_index: np.ndarray,
    ) -> List[DriftFeatures]:
        cfg = self.cfg
        X, y = stream.X, stream.y
        warmup = cfg.eval.warmup_size

        # Error trajectory aligned to absolute stream indices.
        err = 1 - correct.astype(float)                 # over [warmup, n)
        roll = _rolling(err, self.roll_window)

        def roll_at(abs_idx: int) -> float:
            pos = abs_idx - warmup
            pos = min(max(pos, 0), len(roll) - 1)
            return float(roll[pos])

        # Concept memory: seed with the warm-up concept, then add each concept as
        # it settles -- exactly mirroring the model-reuse cache.
        memory = ConceptMemory()
        memory.add(0, compute_fingerprint(X[:warmup], y[:warmup]))

        feats: List[DriftFeatures] = []
        for (d, r) in schedule:
            # -- pre-drift baseline error (stable region well before d) ------ #
            lo = max(warmup, d - 1000)
            hi = max(lo + 1, d - 300)
            base_pos_lo, base_pos_hi = lo - warmup, hi - warmup
            base_slice = roll[base_pos_lo:base_pos_hi]
            baseline = float(np.median(base_slice)) if len(base_slice) else roll_at(d)

            # -- ramp: walk back from d while error stays above baseline ----- #
            thresh = baseline + self.rise_margin
            ramp_start = d
            t = d
            floor = warmup
            while t > floor and roll_at(t) > thresh:
                t -= 1
            ramp_start = t
            ramp_length = max(1, d - ramp_start)

            peak = max(roll_at(k) for k in range(ramp_start, d + 1))
            error_jump = max(0.0, peak - baseline)
            error_slope = error_jump / ramp_length

            # -- shape of the rise ------------------------------------------- #
            rise = np.array([roll_at(k) for k in range(ramp_start, d + 1)])
            diffs = np.diff(rise)
            if len(diffs) >= 2:
                signs = np.sign(diffs)
                nz = signs[signs != 0]
                sign_changes = int(np.sum(nz[1:] != nz[:-1])) if len(nz) > 1 else 0
                oscillation = sign_changes / (len(diffs) - 1)
                monotonicity = float(np.mean(diffs > 0))
            else:
                oscillation, monotonicity = 0.0, 1.0

            # -- concept similarity (recurrence signal) ---------------------- #
            # Fixed-size window from the detection point, so the distance scale is
            # independent of the pipeline's settle/resolve gap.
            hi = min(len(X), d + self.fp_window)
            fp = compute_fingerprint(X[d:hi], y[d:hi])
            dist, _, _ = memory.nearest(fp, before_index=d)
            memory.add(r, fp)

            feats.append(DriftFeatures(
                detect_index=d, resolve_index=r,
                ramp_length=ramp_length, error_slope=round(error_slope, 5),
                error_jump=round(error_jump, 5), oscillation=round(oscillation, 4),
                monotonicity=round(monotonicity, 4),
                concept_distance=round(float(dist), 5),
            ))
        return feats

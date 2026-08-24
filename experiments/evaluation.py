"""Evaluation metrics shared by every pipeline.

Metrics are computed against the ground-truth ``DriftEvent``s so detection
quality is judged separately from downstream task accuracy, per the prompt.

Detection matching rule: a detection at index ``d`` is a *true positive* for a
ground-truth event with onset ``o`` if ``o <= d <= o + tolerance``. Each event is
credited to its first matching detection (that gives the detection delay);
detections that match no event are *false alarms*.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from data.stream import DriftStream
from baseline.naive import RunResult


def _rolling_accuracy(correct: np.ndarray, window: int) -> np.ndarray:
    if len(correct) == 0:
        return correct.astype(float)
    c = correct.astype(float)
    kernel = np.ones(window) / window
    # 'same' keeps alignment with step_index; edges are lightly biased but fine
    # for a trajectory plot.
    return np.convolve(c, kernel, mode="same")


def match_detections(events, detections: List[int], tolerance: int):
    """Return (per-event delay dict, n_true_positives, false_alarm_indices)."""
    delays: Dict[int, int] = {}          # event onset -> detection delay
    matched_detections = set()
    for ev in events:
        for d in detections:
            if d in matched_detections:
                continue
            if ev.onset <= d <= ev.onset + tolerance:
                delays[ev.onset] = d - ev.onset
                matched_detections.add(d)
                break
    false_alarms = [d for d in detections if d not in matched_detections]
    return delays, len(matched_detections), false_alarms


def recovery_times(stream: DriftStream, result: RunResult,
                   target: float = 0.85, window: int = 300) -> Dict[int, int]:
    """Samples after each drift onset until rolling accuracy recovers to ``target``.

    Returns onset -> recovery length (in samples); -1 if it never recovers within
    the segment before the next drift.
    """
    roll = _rolling_accuracy(result.correct, window)
    idx = result.step_index
    onsets = sorted(e.onset for e in stream.events)
    out: Dict[int, int] = {}
    for k, onset in enumerate(onsets):
        next_onset = onsets[k + 1] if k + 1 < len(onsets) else stream.n_samples
        mask = (idx >= onset) & (idx < next_onset)
        if not mask.any():
            out[onset] = -1
            continue
        seg_idx = idx[mask]
        seg_acc = roll[mask]
        recovered = np.where(seg_acc >= target)[0]
        out[onset] = int(seg_idx[recovered[0]] - onset) if len(recovered) else -1
    return out


def evaluate(stream: DriftStream, result: RunResult, cfg) -> Dict:
    ev = cfg.eval
    delays, n_tp, false_alarms = match_detections(
        stream.events, result.detections, ev.detection_tolerance
    )
    n_events = len(stream.events)
    delay_vals = list(delays.values())

    rec = recovery_times(stream, result, window=ev.accuracy_window)
    rec_vals = [v for v in rec.values() if v >= 0]

    return {
        "pipeline": result.pipeline,
        "detector": result.detector,
        "model": result.model,
        "overall_accuracy": round(result.overall_accuracy, 4),
        "n_ground_truth_drifts": n_events,
        "n_detections": len(result.detections),
        "n_true_positive_detections": n_tp,
        "detection_recall": round(n_tp / n_events, 3) if n_events else 0.0,
        "n_false_alarms": len(false_alarms),
        "false_alarm_rate": round(len(false_alarms) / max(1, len(result.detections)), 3),
        "mean_detection_delay": round(float(np.mean(delay_vals)), 1) if delay_vals else None,
        "per_event_detection_delay": {int(k): int(v) for k, v in delays.items()},
        "mean_recovery_samples": round(float(np.mean(rec_vals)), 1) if rec_vals else None,
        "per_event_recovery_samples": {int(k): int(v) for k, v in rec.items()},
        "n_retrains": result.n_retrains,
        "total_retrain_time_s": round(result.total_retrain_time_s, 4),
    }


def rolling_accuracy_curve(result: RunResult, window: int):
    """(x indices, rolling accuracy) for the accuracy-over-stream plot."""
    return result.step_index, _rolling_accuracy(result.correct, window)

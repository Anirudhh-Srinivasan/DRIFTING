"""Build labelled drift-feature datasets for calibrating/evaluating the classifier.

For clean, balanced supervision we generate *single-drift* streams: a short
stable warm-up on one concept, then exactly one drift of a known type. Different
seeds yield different concept geometries (the synthetic generator keys concept
transforms on the seed), so each seed is an independent example of that drift
type. Recurring drift needs the concept to have appeared earlier, so its stream
is stable(0) -> sudden(1) -> recurring(0), and we label the final (recurrence)
drift.

``build_dataset`` runs the always-retrain prequential loop (so drifts are
detected under a realistic recovering model), extracts features at the detection
that matches the target drift's onset, and pairs them with the ground-truth type.
"""

from __future__ import annotations

import copy
from dataclasses import replace
from typing import List, Tuple

from config import ExperimentConfig, ScenarioConfig, DriftSegment, default_config
from data import load_stream
from strategy.controlled import run_policy, ALWAYS_RETRAIN
from drift_classifier.features import DriftFeatureExtractor, DriftFeatures

DRIFT_TYPES = ("sudden", "gradual", "incremental", "recurring")


def single_drift_scenario(kind: str) -> Tuple[ScenarioConfig, int]:
    """Return (scenario, target_onset) for a one-drift stream of ``kind``."""
    # Segment sizes mirror the main scenario (~4000/segment): the drift response
    # needs room to recover before the next drift, otherwise a fast detection
    # leaves the retrain window straddling the old concept and the model never
    # relearns -- which, for change-based detectors, silently hides later drifts.
    if kind == "sudden":
        segs = [DriftSegment("stable", 0, 4000),
                DriftSegment("sudden", 1, 4000)]
        onset = 4000
    elif kind == "gradual":
        segs = [DriftSegment("stable", 0, 4000),
                DriftSegment("gradual", 2, 5000, transition=1800)]
        onset = 4000
    elif kind == "incremental":
        segs = [DriftSegment("stable", 0, 4000),
                DriftSegment("incremental", 3, 5000, transition=2500)]
        onset = 4000
    elif kind == "recurring":
        segs = [DriftSegment("stable", 0, 4000),
                DriftSegment("sudden", 1, 4000),
                DriftSegment("recurring", 0, 4000)]
        onset = 8000
    else:
        raise ValueError(kind)
    return ScenarioConfig(segments=segs), onset


def _cfg_for(kind: str, seed: int, base: ExperimentConfig) -> Tuple[ExperimentConfig, int]:
    scenario, onset = single_drift_scenario(kind)
    cfg = copy.deepcopy(base)
    cfg.seed = seed
    cfg.scenario = scenario
    cfg.eval.warmup_size = 2000
    cfg.eval.retrain_window = 2000
    # settle == retrain_window makes the retrain window exactly the post-detection
    # window [d, i] (start = i - retrain_window = d), so the model actually learns
    # the *new* concept instead of a window straddling the drift. Without this a
    # fast detection leaves a stale model and later (recurring) drifts are never
    # detected. Latency is irrelevant here -- these streams exist only to produce
    # labelled drift features, not to measure downstream accuracy.
    cfg.cache.settle = 2000
    return cfg, onset


def build_dataset(seeds: List[int], base: ExperimentConfig | None = None
                  ) -> Tuple[List[DriftFeatures], List[str], dict]:
    """Return (features, labels, stats) over all (kind, seed) single-drift streams.

    ``stats`` reports detection recall per type (how often the target drift was
    detected and thus produced a feature vector).
    """
    base = base or default_config()
    feats: List[DriftFeatures] = []
    labels: List[str] = []
    seen = {k: 0 for k in DRIFT_TYPES}
    detected = {k: 0 for k in DRIFT_TYPES}

    for kind in DRIFT_TYPES:
        for seed in seeds:
            cfg, onset = _cfg_for(kind, seed, base)
            stream = load_stream(cfg)
            res = run_policy(stream, cfg, ALWAYS_RETRAIN, schedule=None)
            extractor = DriftFeatureExtractor(cfg)
            all_feats = extractor.extract(stream, res.schedule, res.correct, res.step_index)
            seen[kind] += 1
            # Match the detection whose onset lands in [onset, onset+tol].
            tol = cfg.eval.detection_tolerance
            match = None
            for f in all_feats:
                if onset <= f.detect_index <= onset + tol:
                    match = f
                    break
            if match is not None:
                feats.append(match)
                labels.append(kind)
                detected[kind] += 1

    stats = {
        "detection_recall": {k: round(detected[k] / seen[k], 3) for k in DRIFT_TYPES},
        "n_examples": {k: detected[k] for k in DRIFT_TYPES},
    }
    return feats, labels, stats

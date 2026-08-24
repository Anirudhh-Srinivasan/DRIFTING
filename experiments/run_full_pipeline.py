"""End-to-end proposed pipeline vs naive baseline (the paper's headline experiment).

Steps:
  1. Calibrate the rule-based drift-type classifier on separate seeds.
  2. Build the main multi-drift stream.
  3. Pass 1 -- run the naive always-retrain baseline live; capture the drift
     schedule and the error trajectory.
  4. Extract features at each detected drift, classify its type, and record the
     drift-type classification accuracy against ground truth.
  5. Pass 2 -- run the type-aware pipeline on the *same* schedule, routing each
     drift to full retrain / warm-start incremental update / cache retrieval.
  6. Compare: retraining wall-clock, per-strategy breakdown, cache reuse, and
     overall accuracy (must stay at parity).

Because both pipelines replay the identical schedule, the difference is purely
the type-aware strategy selection.

Usage:
    .venv/bin/python -m experiments.run_full_pipeline
    .venv/bin/python -m experiments.run_full_pipeline --calib 1-6 --model logreg
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import default_config
from data import load_stream
from drift_classifier import RuleBasedDriftClassifier, DriftFeatureExtractor
from experiments.classification_data import build_dataset
from experiments.evaluation import _rolling_accuracy
from strategy.controlled import run_policy, ALWAYS_RETRAIN
from strategy.pipeline import run_type_aware


def _parse_seeds(spec: str) -> List[int]:
    out: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-"); out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def _gt_kind(stream, index, tol):
    best = None
    for e in stream.events:
        if e.onset <= index <= e.onset + tol:
            if best is None or (index - e.onset) < (index - best.onset):
                best = e
    return best.kind if best else "false_alarm"


def _plot(stream, base, proposed, type_map, cfg, path):
    fig, ax = plt.subplots(figsize=(11, 4.8))
    w = cfg.eval.accuracy_window
    ax.plot(base.step_index, _rolling_accuracy(base.correct, w),
            lw=1.0, color="#9aa0a6", label="naive (always retrain)")
    ax.plot(proposed.step_index, _rolling_accuracy(proposed.correct, w),
            lw=1.2, color="#1f4e79", label="type-aware")
    for e in stream.events:
        c = "#8338ec" if e.is_recurrence else "#d1495b"
        ax.axvline(e.onset, color=c, ls="--", lw=1.0, alpha=0.7)
    markers = {"full_retrain": ("o", "#d1495b"), "incremental_update": ("s", "#edae49"),
               "cache_retrieval": ("^", "#2a9d8f"), "full_retrain_fallback": ("x", "#d1495b")}
    for a in proposed.actions:
        mk, col = markers.get(a.strategy, ("o", "#333"))
        ax.scatter([a.resolve_index], [0.04], marker=mk, color=col, s=55, zorder=5)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("stream index"); ax.set_ylabel(f"accuracy (rolling, w={w})")
    ax.set_title("Type-aware pipeline vs naive baseline "
                 f"({cfg.model.kind}, {cfg.primary_detector.upper()})\n"
                 "○ full retrain  □ incremental  △ cache reuse")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", default="1-6", help="classifier calibration seeds")
    ap.add_argument("--detector", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    cfg = default_config()
    if args.detector: cfg.primary_detector = args.detector
    if args.model: cfg.model.kind = args.model
    if args.seed is not None: cfg.seed = args.seed
    os.makedirs(cfg.results_dir, exist_ok=True)
    tol = cfg.eval.detection_tolerance

    print(f"[1/6] Calibrating drift-type classifier (seeds {args.calib}) ...")
    cf, cl, cstats = build_dataset(_parse_seeds(args.calib), cfg)
    clf = RuleBasedDriftClassifier()
    clf.calibrate(cf, cl)

    print(f"[2/6] Building main stream (seed={cfg.seed}) ...")
    stream = load_stream(cfg)
    print(f"      {stream.n_samples} samples, {len(stream.events)} drifts")

    print("[3/6] Pass 1 -- naive baseline (always retrain), capturing schedule ...")
    base = run_policy(stream, cfg, ALWAYS_RETRAIN, schedule=None)

    print("[4/6] Extracting features + classifying each detected drift ...")
    feats = DriftFeatureExtractor(cfg).extract(stream, base.schedule, base.correct, base.step_index)
    type_map: Dict[int, str] = {}
    type_rows = []
    n_type_correct = 0
    n_type_total = 0
    for f in feats:
        pred = clf.classify(f)
        type_map[f.resolve_index] = pred
        gt = _gt_kind(stream, f.detect_index, tol)
        type_rows.append((f.detect_index, gt, pred))
        if gt != "false_alarm":
            n_type_total += 1
            n_type_correct += int(pred == gt)
    type_acc = n_type_correct / n_type_total if n_type_total else 0.0

    print("[5/6] Pass 2 -- type-aware pipeline on the same schedule ...")
    proposed = run_type_aware(stream, cfg, base.schedule, type_map)

    print("[6/6] Comparing ...\n")
    cs = proposed.cache_stats
    summary = {
        "detector": cfg.primary_detector,
        "model": cfg.model.kind,
        "n_drift_points": len(base.schedule),
        "drift_type_classification_accuracy": round(type_acc, 3),
        "per_drift": [
            {"detect_index": d, "true_type": gt, "predicted_type": p}
            for (d, gt, p) in type_rows
        ],
        "naive_baseline": {
            "n_retrains": base.n_retrains,
            "total_cost_s": round(base.total_retrain_time_s, 4),
            "overall_accuracy": round(base.overall_accuracy, 4),
        },
        "type_aware": {
            "strategy_counts": proposed.counts_by_strategy(),
            "strategy_cost_s": proposed.cost_by_strategy(),
            "total_cost_s": round(proposed.total_cost_s, 4),
            "overall_accuracy": round(proposed.overall_accuracy, 4),
            "cache_hits": cs.hits,
            "compute_saved_s_est": round(cs.compute_saved_s, 4),
        },
        "wall_clock_speedup": round(base.total_retrain_time_s / proposed.total_cost_s, 3)
        if proposed.total_cost_s else None,
        "cost_reduction_pct": round(
            100 * (base.total_retrain_time_s - proposed.total_cost_s)
            / base.total_retrain_time_s, 1) if base.total_retrain_time_s else None,
        "accuracy_delta": round(proposed.overall_accuracy - base.overall_accuracy, 4),
    }
    with open(os.path.join(cfg.results_dir, "full_pipeline_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    _plot(stream, base, proposed, type_map, cfg,
          os.path.join(cfg.results_dir, "full_pipeline_accuracy.png"))

    print(f"  drift-type classification accuracy: {type_acc:.3f}")
    print("  per-drift (true -> predicted):")
    for (d, gt, p) in type_rows:
        flag = "" if gt == p else "   <-- misclassified"
        print(f"    idx {d:>6}: {gt:>12} -> {p:<12}{flag}")
    print(f"\n  naive baseline: retrains={base.n_retrains}  "
          f"cost={base.total_retrain_time_s:.4f}s  acc={base.overall_accuracy:.4f}")
    print(f"  type-aware:     {proposed.counts_by_strategy()}")
    print(f"                  cost={proposed.total_cost_s:.4f}s  "
          f"acc={proposed.overall_accuracy:.4f}  cache_hits={cs.hits}")
    print(f"  cost reduction: {summary['cost_reduction_pct']}%  "
          f"(speedup {summary['wall_clock_speedup']}x)")
    print(f"  accuracy delta: {summary['accuracy_delta']:+.4f}")
    print(f"\nArtifacts written to ./{cfg.results_dir}/")


if __name__ == "__main__":
    main()

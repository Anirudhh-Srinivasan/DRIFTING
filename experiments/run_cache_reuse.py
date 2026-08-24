"""Validate the model-reuse cache on the recurring-drift scenario.

Runs the naive always-retrain baseline and the cache-reuse pipeline on the *same*
stream and reports:
  * per-detection concept distance + action (reuse/retrain) -- the calibration
    evidence for the distance threshold (recurrences sit near 0; novel concepts
    far), matched against ground-truth recurrences;
  * cache hit rate and compute saved (wall-clock retrain time avoided);
  * retrains: baseline vs cache-reuse (the headline compute-savings number);
  * overall accuracy for both (should be near-equal -- savings must not cost
    accuracy).

Usage:
    .venv/bin/python -m experiments.run_cache_reuse
    .venv/bin/python -m experiments.run_cache_reuse --threshold 1.0 --detector adwin
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import default_config
from data import load_stream
from baseline.naive import run_naive_baseline
from strategy.cache_reuse import run_cache_reuse
from experiments.evaluation import (
    evaluate, rolling_accuracy_curve, match_detections,
)


def _ground_truth_kind(stream, index, tol):
    """Kind + recurrence flag of the ground-truth drift a detection matches."""
    best = None
    for e in stream.events:
        if e.onset <= index <= e.onset + tol:
            if best is None or (index - e.onset) < (index - best.onset):
                best = e
    if best is None:
        return "false_alarm", False
    return best.kind, best.is_recurrence


def _plot(stream, base, cache, cfg, path):
    fig, ax = plt.subplots(figsize=(11, 4.8))
    xb, ab = rolling_accuracy_curve(base, cfg.eval.accuracy_window)
    ax.plot(xb, ab, lw=1.0, color="#9aa0a6", label="baseline (always retrain)")
    from strategy.cache_reuse import CacheRunResult  # local import for typing
    xc = cache.step_index
    from experiments.evaluation import _rolling_accuracy
    ac = _rolling_accuracy(cache.correct, cfg.eval.accuracy_window)
    ax.plot(xc, ac, lw=1.2, color="#1f4e79", label="cache-reuse")

    for e in stream.events:
        c = "#8338ec" if e.is_recurrence else "#d1495b"
        ax.axvline(e.onset, color=c, ls="--", lw=1.0, alpha=0.8)
    for d in cache.decisions:
        if d.action == "reuse":
            ax.scatter([d.index], [0.03], marker="^", color="#2a9d8f", s=60, zorder=5)

    ax.set_ylim(0, 1.02)
    ax.set_xlabel("stream index")
    ax.set_ylabel(f"accuracy (rolling, w={cfg.eval.accuracy_window})")
    ax.set_title("Cache-reuse vs naive baseline "
                 f"({cache.model}, {cache.detector.upper()}) — ▲ = cache hit, "
                 "purple = recurrence")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--settle", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    cfg = default_config()
    if args.detector: cfg.primary_detector = args.detector
    if args.model: cfg.model.kind = args.model
    if args.threshold is not None: cfg.cache.distance_threshold = args.threshold
    if args.settle is not None: cfg.cache.settle = args.settle
    if args.seed is not None: cfg.seed = args.seed
    os.makedirs(cfg.results_dir, exist_ok=True)
    tol = cfg.eval.detection_tolerance

    print(f"[1/4] Building stream (seed={cfg.seed}) ...")
    stream = load_stream(cfg)
    recurrences = [e for e in stream.events if e.is_recurrence]
    print(f"      {stream.n_samples} samples, {len(stream.events)} drifts "
          f"({len(recurrences)} recurring)")

    print("[2/4] Baseline (always retrain) ...")
    base = run_naive_baseline(stream, cfg)

    print(f"[3/4] Cache-reuse (threshold={cfg.cache.distance_threshold}) ...")
    cache = run_cache_reuse(stream, cfg)

    # --- per-detection calibration table ---------------------------------- #
    print("\n[4/4] Per-detection concept distance & action "
          "(calibration evidence):")
    print(f"  {'index':>7} {'gt_type':>12} {'recur?':>6} {'distance':>9} {'action':>8}")
    for d in cache.decisions:
        kind, is_rec = _ground_truth_kind(stream, d.index, tol)
        print(f"  {d.index:>7} {kind:>12} {str(is_rec):>6} "
              f"{d.distance:>9.4f} {d.action:>8}")

    # --- headline comparison ---------------------------------------------- #
    cs = cache.cache_stats
    base_m = evaluate(stream, base, cfg)
    _, _, base_fa = match_detections(stream.events, base.detections, tol)
    _, _, cache_fa = match_detections(stream.events, cache.detections, tol)

    # Correct reuse == reuse action landing on a genuine recurrence.
    correct_reuse = sum(
        1 for d in cache.decisions
        if d.action == "reuse" and _ground_truth_kind(stream, d.index, tol)[1]
    )

    summary = {
        "detector": cfg.primary_detector,
        "model": cfg.model.kind,
        "distance_threshold": cfg.cache.distance_threshold,
        "n_ground_truth_recurrences": len(recurrences),
        "baseline": {
            "n_retrains": base.n_retrains,
            "total_retrain_time_s": round(base.total_retrain_time_s, 4),
            "overall_accuracy": base_m["overall_accuracy"],
        },
        "cache_reuse": {
            "n_retrains": cache.n_retrains,
            "n_reuses": cache.n_reuses,
            "correct_reuses_on_recurrence": correct_reuse,
            "total_retrain_time_s": round(cache.total_retrain_time_s, 4),
            "overall_accuracy": round(cache.overall_accuracy, 4),
            "cache_hit_rate": round(cs.hit_rate, 3),
            "compute_saved_s": round(cs.compute_saved_s, 4),
            "hit_distances": cs.hit_distances,
        },
        "retrains_avoided": base.n_retrains - cache.n_retrains,
    }

    with open(os.path.join(cfg.results_dir, "cache_reuse_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    _plot(stream, base, cache, cfg,
          os.path.join(cfg.results_dir, "cache_reuse_accuracy.png"))

    print("\n=== Cache-reuse vs baseline ===")
    print(f"  retrains:        baseline={base.n_retrains}  "
          f"cache-reuse={cache.n_retrains}  (avoided={summary['retrains_avoided']})")
    print(f"  cache reuses:    {cache.n_reuses}  "
          f"(correct on recurrence={correct_reuse}/{len(recurrences)})")
    print(f"  cache hit rate:  {cs.hit_rate:.3f}")
    print(f"  compute saved:   {cs.compute_saved_s:.4f} s "
          f"(baseline retrain time={base.total_retrain_time_s:.4f} s)")
    print(f"  accuracy:        baseline={base_m['overall_accuracy']:.4f}  "
          f"cache-reuse={cache.overall_accuracy:.4f}")
    print(f"\nArtifacts written to ./{cfg.results_dir}/")


if __name__ == "__main__":
    main()

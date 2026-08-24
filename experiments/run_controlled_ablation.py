"""Controlled ablation: cache effect isolated on an identical detection schedule.

Both pipelines act at exactly the same drift points (schedule captured from the
always-retrain policy and replayed under both). The only difference is the action
taken, so:

    retrains_avoided  ==  cache reuses            (pure caching)
    compute_saved     ==  the baseline's actual retrain wall-clock at the exact
                          points where the cache reused instead

This is the number for the paper's headline table. Accuracy should stay at parity
(or better), confirming savings don't cost detection quality.

Usage:
    .venv/bin/python -m experiments.run_controlled_ablation
    .venv/bin/python -m experiments.run_controlled_ablation --detector ddm --threshold 1.0
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import default_config
from data import load_stream
from strategy.controlled import controlled_ablation
from experiments.evaluation import _rolling_accuracy


def _gt_kind(stream, index, tol):
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
    w = cfg.eval.accuracy_window
    ax.plot(base.step_index, _rolling_accuracy(base.correct, w),
            lw=1.0, color="#9aa0a6", label="always-retrain")
    ax.plot(cache.step_index, _rolling_accuracy(cache.correct, w),
            lw=1.2, color="#1f4e79", label="cache-reuse")
    for e in stream.events:
        c = "#8338ec" if e.is_recurrence else "#d1495b"
        ax.axvline(e.onset, color=c, ls="--", lw=1.0, alpha=0.8)
    for a in cache.actions:
        if a.action == "reuse":
            ax.scatter([a.resolve_index], [0.03], marker="^",
                       color="#2a9d8f", s=60, zorder=5)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("stream index")
    ax.set_ylabel(f"accuracy (rolling, w={w})")
    ax.set_title("Controlled ablation — identical schedule, action differs only "
                 f"({cache.policy}; ▲ = cache reuse)")
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

    print(f"[1/3] Building stream (seed={cfg.seed}) ...")
    stream = load_stream(cfg)
    n_rec = sum(1 for e in stream.events if e.is_recurrence)
    print(f"      {stream.n_samples} samples, {len(stream.events)} drifts "
          f"({n_rec} recurring)")

    print("[2/3] Capturing schedule from always-retrain, replaying both policies ...")
    base, cache = controlled_ablation(stream, cfg)

    # Exact compute saved: baseline's actual retrain cost at the resolve points
    # where the cache chose to reuse.
    base_cost_by_resolve = {a.resolve_index: a.cost_s for a in base.actions}
    reuse_points = [a.resolve_index for a in cache.actions if a.action == "reuse"]
    exact_saved = sum(base_cost_by_resolve.get(r, 0.0) for r in reuse_points)

    correct_reuse = sum(
        1 for a in cache.actions
        if a.action == "reuse" and _gt_kind(stream, a.detect_index, tol)[1]
    )
    wrong_reuse = len(reuse_points) - correct_reuse

    print("\n[3/3] Per-action table (identical schedule for both):")
    print(f"  {'resolve':>7} {'gt_type':>12} {'recur?':>6} {'distance':>9} "
          f"{'cache_action':>13}")
    for a in cache.actions:
        kind, is_rec = _gt_kind(stream, a.detect_index, tol)
        dist = "nan" if a.distance != a.distance else f"{a.distance:.4f}"
        print(f"  {a.resolve_index:>7} {kind:>12} {str(is_rec):>6} {dist:>9} "
              f"{a.action:>13}")

    summary = {
        "detector": cfg.primary_detector,
        "model": cfg.model.kind,
        "distance_threshold": cfg.cache.distance_threshold,
        "settle": cfg.cache.settle,
        "n_shared_drift_points": len(base.schedule),
        "n_ground_truth_recurrences": n_rec,
        "always_retrain": {
            "n_retrains": base.n_retrains,
            "total_retrain_time_s": round(base.total_retrain_time_s, 4),
            "overall_accuracy": round(base.overall_accuracy, 4),
        },
        "cache_reuse": {
            "n_retrains": cache.n_retrains,
            "n_reuses": cache.n_reuses,
            "correct_reuses_on_recurrence": correct_reuse,
            "wrong_reuses": wrong_reuse,
            "total_retrain_time_s": round(cache.total_retrain_time_s, 4),
            "overall_accuracy": round(cache.overall_accuracy, 4),
        },
        # Pure caching effect -- identical schedule guarantees this.
        "retrains_avoided_by_cache": base.n_retrains - cache.n_retrains,
        "exact_compute_saved_s": round(exact_saved, 4),
        "accuracy_delta": round(cache.overall_accuracy - base.overall_accuracy, 4),
    }
    with open(os.path.join(cfg.results_dir, "controlled_ablation_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    _plot(stream, base, cache, cfg,
          os.path.join(cfg.results_dir, "controlled_ablation_accuracy.png"))

    print("\n=== Controlled ablation (identical schedule) ===")
    print(f"  shared drift points:   {len(base.schedule)}")
    print(f"  always-retrain:        retrains={base.n_retrains}  "
          f"acc={base.overall_accuracy:.4f}  time={base.total_retrain_time_s:.4f}s")
    print(f"  cache-reuse:           retrains={cache.n_retrains}  reuses={cache.n_reuses}  "
          f"acc={cache.overall_accuracy:.4f}  time={cache.total_retrain_time_s:.4f}s")
    print(f"  retrains avoided:      {summary['retrains_avoided_by_cache']} "
          f"(== reuses; correct on recurrence={correct_reuse}/{n_rec}, "
          f"wrong={wrong_reuse})")
    print(f"  exact compute saved:   {exact_saved:.4f}s")
    print(f"  accuracy delta:        {summary['accuracy_delta']:+.4f} (cache − baseline)")
    print(f"\nArtifacts written to ./{cfg.results_dir}/")


if __name__ == "__main__":
    main()

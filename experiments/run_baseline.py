"""End-to-end runner for the naive always-retrain baseline.

Usage (from the project root):

    .venv/bin/python -m experiments.run_baseline
    .venv/bin/python -m experiments.run_baseline --detector ddm --model logreg

Produces, under ``results/``:
    * baseline_metrics.json     -- the evaluation summary
    * baseline_accuracy.png     -- accuracy-over-stream with drift markers
    * stream_summary.json       -- ground-truth drift events (for the paper)

This is the working comparison point the novel components will be measured
against. It is intentionally self-contained and reproducible (single seed).
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")  # headless; save figures without a display
import matplotlib.pyplot as plt

from config import default_config
from data import load_stream
from baseline.naive import run_naive_baseline
from experiments.evaluation import evaluate, rolling_accuracy_curve


def _plot(stream, result, cfg, path):
    x, acc = rolling_accuracy_curve(result, cfg.eval.accuracy_window)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(x, acc, lw=1.1, color="#1f4e79", label="rolling accuracy")

    colors = {"sudden": "#d1495b", "gradual": "#edae49",
              "incremental": "#66a182", "recurring": "#8338ec"}
    seen_kinds = set()
    for e in stream.events:
        c = colors.get(e.kind, "#888")
        lbl = e.kind if e.kind not in seen_kinds else None
        seen_kinds.add(e.kind)
        ax.axvline(e.onset, color=c, ls="--", lw=1.2, alpha=0.9, label=lbl)
        if e.is_recurrence:
            ax.annotate("recur", (e.onset, 0.02), fontsize=7, color=c,
                        rotation=90, va="bottom")

    for r in result.retrains[1:]:  # skip the warm-up entry
        ax.axvline(r.index, color="#bbbbbb", lw=0.5, alpha=0.5)

    ax.set_ylim(0, 1.02)
    ax.set_xlabel("stream index")
    ax.set_ylabel(f"accuracy (rolling, w={cfg.eval.accuracy_window})")
    ax.set_title("Naive always-retrain baseline — accuracy over stream "
                 f"({result.model}, {result.detector.upper()})")
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", default=None, help="adwin|kswin|ddm|eddm")
    ap.add_argument("--model", default=None, help="random_forest|logreg")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    cfg = default_config()
    if args.detector:
        cfg.primary_detector = args.detector
    if args.model:
        cfg.model.kind = args.model
    if args.seed is not None:
        cfg.seed = args.seed

    os.makedirs(cfg.results_dir, exist_ok=True)

    print(f"[1/4] Building synthetic stream (seed={cfg.seed}) ...")
    stream = load_stream(cfg, source="synthetic")
    print(f"      {stream.n_samples} samples, {stream.n_features} features, "
          f"{len(stream.class_names)} classes, {len(stream.events)} drifts")

    print(f"[2/4] Running naive baseline (detector={cfg.primary_detector}, "
          f"model={cfg.model.kind}) ...")
    result = run_naive_baseline(stream, cfg)

    print("[3/4] Evaluating ...")
    metrics = evaluate(stream, result, cfg)

    # Persist everything.
    with open(os.path.join(cfg.results_dir, "baseline_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    stream_summary = {
        "n_samples": stream.n_samples,
        "n_features": stream.n_features,
        "class_names": stream.class_names,
        "events": [vars(e) for e in stream.events],
    }
    with open(os.path.join(cfg.results_dir, "stream_summary.json"), "w") as f:
        json.dump(stream_summary, f, indent=2)

    print("[4/4] Plotting accuracy-over-stream ...")
    _plot(stream, result, cfg, os.path.join(cfg.results_dir, "baseline_accuracy.png"))

    print("\n=== Baseline summary ===")
    for k, v in metrics.items():
        if not isinstance(v, dict):
            print(f"  {k}: {v}")
    print(f"\nArtifacts written to ./{cfg.results_dir}/")


if __name__ == "__main__":
    main()

"""Calibrate and evaluate the rule-based drift-type classifier.

Splits seeds into a calibration set and a disjoint test set, calibrates the rule
thresholds on the former (against simulated ground truth), and reports the full
4x4 confusion matrix + per-class precision/recall/F1 + overall accuracy on the
held-out test set. Thresholds and detection recall are logged too.

Usage:
    .venv/bin/python -m experiments.run_drift_classification
    .venv/bin/python -m experiments.run_drift_classification --calib 1-8 --test 100-104
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import default_config
from drift_classifier import RuleBasedDriftClassifier, DRIFT_TYPES
from experiments.classification_data import build_dataset


def _parse_seeds(spec: str) -> List[int]:
    """'1-8' -> [1..8]; '1,3,5' -> [1,3,5]; combinations allowed."""
    out: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def _confusion(y_true, y_pred, order):
    idx = {t: i for i, t in enumerate(order)}
    m = np.zeros((len(order), len(order)), dtype=int)
    for t, p in zip(y_true, y_pred):
        m[idx[t], idx[p]] += 1
    return m


def _per_class(cm, order):
    out = {}
    for i, t in enumerate(order):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        out[t] = {"precision": round(prec, 3), "recall": round(rec, 3),
                  "f1": round(f1, 3), "support": int(cm[i, :].sum())}
    return out


def _plot_cm(cm, order, acc, path):
    fig, ax = plt.subplots(figsize=(5.6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(order))); ax.set_yticks(range(len(order)))
    ax.set_xticklabels(order, rotation=30, ha="right"); ax.set_yticklabels(order)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"Rule-based drift-type confusion (test acc={acc:.3f})")
    vmax = cm.max() if cm.max() else 1
    for i in range(len(order)):
        for j in range(len(order)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > vmax / 2 else "black", fontsize=11)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", default="1-8", help="calibration seeds, e.g. 1-8")
    ap.add_argument("--test", default="101-106", help="held-out test seeds")
    args = ap.parse_args()

    cfg = default_config()
    os.makedirs(cfg.results_dir, exist_ok=True)
    calib_seeds = _parse_seeds(args.calib)
    test_seeds = _parse_seeds(args.test)
    order = list(DRIFT_TYPES)

    print(f"[1/4] Building calibration set (seeds {calib_seeds}) ...")
    cf, cl, cstats = build_dataset(calib_seeds, cfg)
    print(f"      calibration examples: {cstats['n_examples']} "
          f"(detection recall {cstats['detection_recall']})")

    print("[2/4] Calibrating rule thresholds on ground truth ...")
    clf = RuleBasedDriftClassifier()
    thresholds = clf.calibrate(cf, cl)
    for node, d in thresholds.items():
        print(f"      {node}: {d}")

    print(f"[3/4] Building held-out test set (seeds {test_seeds}) ...")
    tf, tl, tstats = build_dataset(test_seeds, cfg)
    print(f"      test examples: {tstats['n_examples']} "
          f"(detection recall {tstats['detection_recall']})")

    preds = [clf.classify(f) for f in tf]
    cm = _confusion(tl, preds, order)
    acc = float(np.mean(np.array(tl) == np.array(preds))) if tl else 0.0
    per_class = _per_class(cm, order)

    print("\n[4/4] Confusion matrix (rows = true, cols = predicted):")
    header = "true\\pred".ljust(13) + "".join(t[:5].rjust(7) for t in order)
    print("  " + header)
    for i, t in enumerate(order):
        print("  " + t.ljust(13) + "".join(str(cm[i, j]).rjust(7) for j in range(len(order))))

    print("\n  per-class:")
    for t in order:
        pc = per_class[t]
        print(f"    {t:>12}: precision={pc['precision']:.3f} recall={pc['recall']:.3f} "
              f"f1={pc['f1']:.3f} (support={pc['support']})")
    print(f"\n  overall test accuracy: {acc:.3f}")

    summary = {
        "calibration_seeds": calib_seeds,
        "test_seeds": test_seeds,
        "calibration": cstats,
        "test_detection": tstats,
        "thresholds": thresholds,
        "confusion_matrix": {"order": order, "matrix": cm.tolist()},
        "per_class": per_class,
        "overall_accuracy": round(acc, 4),
    }
    with open(os.path.join(cfg.results_dir, "drift_classification_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    _plot_cm(cm, order, acc,
             os.path.join(cfg.results_dir, "drift_classification_confusion.png"))
    print(f"\nArtifacts written to ./{cfg.results_dir}/")


if __name__ == "__main__":
    main()

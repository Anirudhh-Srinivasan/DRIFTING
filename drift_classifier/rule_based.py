"""Rule-based drift-type classifier with ground-truth-calibrated thresholds.

The rule *structure* is fixed and interpretable -- a 3-level decision tree that
mirrors how the drift types differ in the detector signal:

    1. concept_distance ≤ τ_recur           -> recurring
       (a recurrence returns to a cached concept, so its fingerprint distance
        collapses to ~0; every novel concept is far. This test comes first
        because a recurrence is *dynamically* an abrupt switch and would
        otherwise look "sudden".)
    2. else, <sudden-feature> ≥ τ_sudden     -> sudden
       (a sudden switch produces a large, near-monotonic error jump; gradual and
        incremental produce a small, slow rise.)
    3. else split gradual vs incremental on the most discriminative feature.

The *thresholds* -- and, at each node, *which* feature to split on -- are not
hand-picked: ``calibrate`` chooses them from labelled features by optimal 1-D
splits on the simulated ground truth, so every number is defensible. The
gradual/incremental split is the genuinely hard one (the two are near-identical
in the error signal); the confusion matrix reports how well it does rather than
hiding it in an aggregate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from drift_classifier.base import DriftTypeClassifier
from drift_classifier.features import DriftFeatures

# Candidate features considered at each decision node during calibration.
_RECUR_FEATURES = ["concept_distance"]
_SUDDEN_FEATURES = ["error_jump", "monotonicity", "ramp_length"]
_GRAD_INC_FEATURES = ["ramp_length", "monotonicity", "error_jump", "oscillation"]


def _values(feats: Sequence[DriftFeatures], name: str) -> np.ndarray:
    return np.array([getattr(f, name) for f in feats], dtype=float)


def _best_1d_split(vals: np.ndarray, positive: np.ndarray):
    """Best threshold + direction separating ``positive`` (bool) from the rest.

    Returns ``(threshold, direction, accuracy)`` where direction is ">=" (value
    ≥ threshold predicts positive) or "<=". Chooses the split maximising accuracy.
    """
    if len(vals) == 0:
        return 0.0, ">=", 0.0
    order = np.argsort(vals)
    sv = vals[order]
    # Candidate thresholds: midpoints between consecutive distinct values, plus
    # the extremes so an all-positive/all-negative node still resolves.
    mids = [(sv[i] + sv[i + 1]) / 2 for i in range(len(sv) - 1) if sv[i] != sv[i + 1]]
    candidates = [sv[0] - 1e-9, *mids, sv[-1] + 1e-9]
    best = (candidates[0], ">=", -1.0)
    for t in candidates:
        for direction in (">=", "<="):
            pred = vals >= t if direction == ">=" else vals <= t
            acc = float(np.mean(pred == positive))
            if acc > best[2]:
                best = (t, direction, acc)
    return best


@dataclass
class _Node:
    feature: str
    threshold: float
    direction: str          # ">=" or "<=": condition that selects the node's target
    accuracy: float         # split accuracy on the calibration subset


class RuleBasedDriftClassifier(DriftTypeClassifier):
    name = "rule_based"

    def __init__(self):
        self.recur: Optional[_Node] = None
        self.sudden: Optional[_Node] = None
        self.grad_inc: Optional[_Node] = None
        self.grad_inc_target: str = "gradual"   # which label the ">=/<=" selects

    # -- calibration -------------------------------------------------------- #
    def _pick_node(self, feats, positive, feature_names) -> _Node:
        best = None
        for fname in feature_names:
            t, d, acc = _best_1d_split(_values(feats, fname), positive)
            if best is None or acc > best.accuracy:
                best = _Node(fname, t, d, acc)
        return best

    def calibrate(self, feats: List[DriftFeatures], labels: List[str]) -> dict:
        labels = np.array(labels)

        # Node 1: recurring vs everything else.
        is_recur = labels == "recurring"
        self.recur = self._pick_node(feats, is_recur, _RECUR_FEATURES)

        # Node 2: sudden vs {gradual, incremental} among the non-recurring.
        rest_mask = ~is_recur
        rest = [f for f, m in zip(feats, rest_mask) if m]
        rest_labels = labels[rest_mask]
        is_sudden = rest_labels == "sudden"
        self.sudden = self._pick_node(rest, is_sudden, _SUDDEN_FEATURES)

        # Node 3: gradual vs incremental among that subset.
        gi_mask = np.isin(rest_labels, ["gradual", "incremental"])
        gi = [f for f, m in zip(rest, gi_mask) if m]
        gi_labels = rest_labels[gi_mask]
        is_gradual = gi_labels == "gradual"
        self.grad_inc = self._pick_node(gi, is_gradual, _GRAD_INC_FEATURES)
        self.grad_inc_target = "gradual"

        return self.thresholds()

    def thresholds(self) -> dict:
        def d(n: _Node):
            return None if n is None else {
                "feature": n.feature, "threshold": round(n.threshold, 5),
                "direction": n.direction, "calib_split_accuracy": round(n.accuracy, 3),
            }
        return {"recurring_vs_rest": d(self.recur),
                "sudden_vs_rest": d(self.sudden),
                "gradual_vs_incremental": d(self.grad_inc)}

    # -- inference ---------------------------------------------------------- #
    @staticmethod
    def _fires(node: _Node, f: DriftFeatures) -> bool:
        v = getattr(f, node.feature)
        return v >= node.threshold if node.direction == ">=" else v <= node.threshold

    def classify(self, f: DriftFeatures) -> str:
        if self.recur is None:
            raise RuntimeError("Classifier not calibrated; call calibrate() first.")
        if self._fires(self.recur, f):
            return "recurring"
        if self._fires(self.sudden, f):
            return "sudden"
        # grad_inc node selects the target label when it "fires", else the other.
        other = "incremental" if self.grad_inc_target == "gradual" else "gradual"
        return self.grad_inc_target if self._fires(self.grad_inc, f) else other

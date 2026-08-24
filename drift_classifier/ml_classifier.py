"""ML drift-type classifier over the same features (STUB -- comparison, later).

A lightweight model (small decision tree or logistic regression) trained on the
labelled simulated drift features, to compare against the rule-based approach on
the identical feature set and confusion matrix. Interface is fixed; kept stubbed
until the rule-based results are reviewed.
"""

from __future__ import annotations

from typing import List

from drift_classifier.base import DriftTypeClassifier
from drift_classifier.features import DriftFeatures


class MLDriftClassifier(DriftTypeClassifier):
    name = "ml"

    def __init__(self, model=None):
        self._model = model

    def fit(self, feats: List[DriftFeatures], labels: List[str]):
        raise NotImplementedError(
            "ML comparison classifier is deferred until the rule-based results "
            "are reviewed; it will train on the same DriftFeatures dataset."
        )

    def classify(self, features: DriftFeatures) -> str:
        raise NotImplementedError("Train the model first.")

"""Interface for the drift-type classifier.

Given the features extracted around a detected drift (see ``features.py``),
classify it as sudden / gradual / incremental / recurring. Two implementations:

* ``rule_based.py`` -- interpretable decision tree, thresholds calibrated on
  simulated ground truth.
* ``ml_classifier.py`` -- a small ML model over the same features (comparison).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from drift_classifier.features import DriftFeatures

DRIFT_TYPES = ("sudden", "gradual", "incremental", "recurring")


class DriftTypeClassifier(ABC):
    name: str = "base"

    @abstractmethod
    def classify(self, features: DriftFeatures) -> str:
        """Return one of ``DRIFT_TYPES`` for a single detected drift."""

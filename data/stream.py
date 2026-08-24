"""Shared stream data structures.

Both the synthetic generator and the (future) real CICIDS2017 loader return a
``DriftStream``, so everything downstream -- detectors, baseline, evaluation --
is agnostic to where the data came from. This is the single interface seam
between "how we get data" and "what we do with it".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class DriftEvent:
    """Ground-truth record of one injected drift.

    ``onset`` is the stream index where the new concept starts to appear.
    ``end`` marks where the transition completes (== onset for sudden drift).
    ``kind`` is one of sudden / gradual / incremental / recurring.
    """

    onset: int
    end: int
    kind: str
    from_concept: int
    to_concept: int
    # True when ``to_concept`` was already seen earlier in the stream. This is
    # the ground-truth signal the model-reuse cache is meant to exploit.
    is_recurrence: bool = False


@dataclass
class DriftStream:
    """A streaming classification problem with ground-truth drift labels.

    Attributes
    ----------
    X : (n_samples, n_features) float array
    y : (n_samples,) int array of class indices (0 == BENIGN)
    concept_ids : (n_samples,) int array -- which concept generated each sample
    events : chronological list of injected drifts
    class_names : human-readable label names, index-aligned with ``y``
    feature_names : column names (CICIDS2017-style when available)
    """

    X: np.ndarray
    y: np.ndarray
    concept_ids: np.ndarray
    events: List[DriftEvent] = field(default_factory=list)
    class_names: Optional[List[str]] = None
    feature_names: Optional[List[str]] = None

    def __post_init__(self) -> None:
        n = len(self.X)
        assert len(self.y) == n, "X and y length mismatch"
        assert len(self.concept_ids) == n, "concept_ids length mismatch"

    @property
    def n_samples(self) -> int:
        return len(self.X)

    @property
    def n_features(self) -> int:
        return self.X.shape[1]

    def binary_labels(self) -> np.ndarray:
        """Benign (0) vs attack (1) view of the multi-class labels."""
        return (self.y != 0).astype(int)

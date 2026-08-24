"""Concept fingerprints and distribution similarity.

A *fingerprint* is a compact summary of the data distribution a model was trained
on. Two fingerprints are compared with a **symmetric KL divergence** between
per-class diagonal Gaussians (the approach agreed for the paper's Methodology):

* For each class present, summarise its features as a diagonal Gaussian
  N(mean, diag(var)) estimated from the window.
* Distance between two concepts = mean over shared classes of the symmetric KL
  divergence of those per-class Gaussians, normalised by the feature count so the
  scale is comparable across configurations.

Rationale: drift here moves the class-conditional feature distributions, so a
per-class mean+covariance summary captures exactly what changes; symmetric KL is
a principled, standard distributional distance and is cheap for diagonal
Gaussians. A genuine recurrence returns to a previously seen concept, so its
distance to the stored fingerprint collapses to near sampling noise -- which is
what lets the cache decide "reuse" with a simple threshold.

This module is shared: the model-reuse cache uses it to key/retrieve models, and
the (future) drift-type classifier uses the same distance to tell *recurring*
drift from *sudden*.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

# Numerical floor for variances so KL stays finite.
_VAR_EPS = 1e-6


@dataclass
class ConceptFingerprint:
    """Per-class diagonal-Gaussian summary of a data window.

    ``per_class[k] == (mean, var)`` with shape (n_features,). Classes absent from
    the window are simply not present in the dict.
    """

    per_class: Dict[int, tuple]
    n_features: int

    @property
    def classes(self):
        return set(self.per_class.keys())


def compute_fingerprint(X: np.ndarray, y: np.ndarray) -> ConceptFingerprint:
    """Summarise a labelled window as per-class diagonal Gaussians."""
    per_class = {}
    d = X.shape[1]
    for k in np.unique(y):
        Xk = X[y == k]
        if len(Xk) == 0:
            continue
        mean = Xk.mean(axis=0)
        var = Xk.var(axis=0) + _VAR_EPS
        per_class[int(k)] = (mean.astype(np.float64), var.astype(np.float64))
    return ConceptFingerprint(per_class=per_class, n_features=d)


def _sym_kl_diag_gaussian(m0, v0, m1, v1) -> float:
    """Symmetric KL between two diagonal Gaussians (summed over dimensions)."""
    # KL(N0||N1) for diagonal Gaussians:
    #   0.5 * sum( v0/v1 + (m1-m0)^2/v1 - 1 + log(v1/v0) )
    diff2 = (m1 - m0) ** 2
    kl_01 = 0.5 * np.sum(v0 / v1 + diff2 / v1 - 1.0 + np.log(v1 / v0))
    kl_10 = 0.5 * np.sum(v1 / v0 + diff2 / v0 - 1.0 + np.log(v0 / v1))
    return float(kl_01 + kl_10)


def concept_distance(a: ConceptFingerprint, b: ConceptFingerprint) -> float:
    """Mean per-class symmetric KL over shared classes, normalised by n_features.

    Returns ``inf`` if the two fingerprints share no classes (incomparable).
    """
    shared = a.classes & b.classes
    if not shared:
        return float("inf")
    total = 0.0
    for k in shared:
        m0, v0 = a.per_class[k]
        m1, v1 = b.per_class[k]
        total += _sym_kl_diag_gaussian(m0, v0, m1, v1)
    return total / (len(shared) * a.n_features)


@dataclass
class ConceptMemory:
    """Ordered store of previously seen concept fingerprints.

    Used to detect recurrence: compare a new concept's fingerprint against
    everything seen before it and report the closest match.
    """

    def __init__(self):
        self.entries = []  # list of (added_at_index, ConceptFingerprint, payload)

    def add(self, index: int, fp: ConceptFingerprint, payload=None) -> None:
        self.entries.append((index, fp, payload))

    def nearest(self, fp: ConceptFingerprint, before_index: Optional[int] = None):
        """Closest prior fingerprint to ``fp``.

        Returns ``(distance, index, payload)`` for the best match, or
        ``(inf, None, None)`` if the memory is empty / nothing qualifies. If
        ``before_index`` is set, only entries added strictly before it count, so a
        concept can't "match itself" within the same drift.
        """
        best = (float("inf"), None, None)
        for idx, entry_fp, payload in self.entries:
            if before_index is not None and idx >= before_index:
                continue
            dist = concept_distance(fp, entry_fp)
            if dist < best[0]:
                best = (dist, idx, payload)
        return best

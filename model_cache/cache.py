"""Model-reuse cache.

Stores previously trained models keyed by the concept fingerprint they were
trained on. When drift is detected, the pipeline fingerprints the *new* concept
and asks the cache for a reusable model: if the nearest stored fingerprint is
within ``threshold`` (symmetric-KL concept distance), the cached model is
returned and no retraining happens -- near-zero cost. Otherwise it's a miss and
the caller trains + stores a fresh model.

Design choices (agreed, documented for the paper):
  * Fingerprint: per-class mean+covariance (see ``fingerprint.py``).
  * Similarity: symmetric KL between per-class diagonal Gaussians.
  * Match threshold: ``threshold`` (concept distance). Calibrated against the
    simulated recurrences -- a true recurrence sits near sampling noise while a
    novel concept is far, so the two are well separated; see
    ``experiments/run_cache_reuse.py`` for the calibration.

Compute saved per hit is credited as the wall-clock a full retrain *would* have
cost, estimated by the running mean of measured retrains (a hit avoids exactly
that work). This is what feeds the paper's compute-savings result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from model_cache.fingerprint import (
    ConceptFingerprint,
    ConceptMemory,
    compute_fingerprint,
    concept_distance,
)


@dataclass
class CachedConcept:
    fingerprint: ConceptFingerprint
    model: object
    trained_at_index: int


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    compute_saved_s: float = 0.0          # summed retrain time avoided on hits
    hit_indices: List[int] = field(default_factory=list)
    hit_distances: List[float] = field(default_factory=list)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


class ModelReuseCache:
    """Fingerprint -> model store with symmetric-KL nearest retrieval."""

    def __init__(self, threshold: float):
        self.threshold = threshold
        self._memory = ConceptMemory()   # fingerprints + model payloads
        self.stats = CacheStats()

    # -- fingerprinting (delegates to the shared module) -------------------- #
    @staticmethod
    def fingerprint(X, y) -> ConceptFingerprint:
        return compute_fingerprint(X, y)

    # -- retrieval ---------------------------------------------------------- #
    def lookup(self, fp: ConceptFingerprint, before_index: Optional[int] = None):
        """Return ``(model_or_None, distance, matched_index)``.

        A hit (model is not None) means ``distance <= threshold``.
        """
        dist, idx, payload = self._memory.nearest(fp, before_index=before_index)
        if payload is not None and dist <= self.threshold:
            return payload, dist, idx
        return None, dist, idx

    def try_reuse(self, fp: ConceptFingerprint, avg_retrain_time_s: float,
                  index: int, before_index: Optional[int] = None):
        """High-level hit/miss with bookkeeping.

        On a hit, credits ``avg_retrain_time_s`` as compute saved and returns the
        cached model. On a miss, returns None (caller should retrain + ``store``).
        """
        model, dist, matched = self.lookup(fp, before_index=before_index)
        if model is not None:
            self.stats.hits += 1
            self.stats.compute_saved_s += avg_retrain_time_s
            self.stats.hit_indices.append(index)
            self.stats.hit_distances.append(round(dist, 5))
            return model, dist, matched
        self.stats.misses += 1
        return None, dist, matched

    # -- population --------------------------------------------------------- #
    def store(self, fp: ConceptFingerprint, model: object, index: int) -> None:
        self._memory.add(index, fp, payload=model)

    @property
    def size(self) -> int:
        return len(self._memory.entries)

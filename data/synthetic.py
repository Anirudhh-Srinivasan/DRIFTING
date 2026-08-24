"""Synthetic CICIDS2017-like data generator.

CICIDS2017 ships ~78 numeric flow features plus a ``Label`` column. Since it has
no labelled drift segments, we generate a distribution-matched *stand-in* whose
feature/label structure mirrors it, so the full pipeline runs end-to-end today.
The real loader (``data/cicids_loader.py``) implements the same interface and can
be swapped in later without touching anything downstream.

Core idea -- a *concept* is an affine transform of a fixed set of per-class
centroids in feature space:

    centroid_concept[k] = R_concept @ base_centroid[k] + t_concept

where ``R_concept`` is an (optional) rotation and ``t_concept`` a translation,
both drawn deterministically from the concept id + seed. Samples for class ``k``
under a concept are drawn from an isotropic Gaussian around that concept's
centroid for ``k``. Because different concepts move the class centroids, a
classifier trained under one concept mislabels samples from another -- which is
exactly what drives the prequential error signal the detectors consume.

The generator is *stateless per concept*: asking for concept 0 again always
yields the same distribution. That determinism is what makes "recurring" drift a
true recurrence (and lets the model-reuse cache legitimately hit).
"""

from __future__ import annotations

from typing import List

import numpy as np

from config import DataConfig


class ConceptSpace:
    """Deterministic factory of concept-conditional class distributions."""

    def __init__(self, cfg: DataConfig, seed: int):
        self.cfg = cfg
        self.seed = seed
        self.n_classes = len(cfg.class_names)

        # Base centroids are fixed for the whole experiment: one per class,
        # spread out on a hypercube so the classes are separable *within* a
        # concept.
        base_rng = np.random.default_rng(seed)
        self.base_centroids = base_rng.normal(
            0.0, 1.5, size=(self.n_classes, cfg.n_features)
        )

    def _concept_rng(self, concept_id: int) -> np.random.Generator:
        # Distinct, reproducible stream per concept.
        return np.random.default_rng(self.seed * 1000 + concept_id)

    def _transform(self, concept_id: int):
        """Return (rotation, translation) for a concept.

        Concept 0 is the identity concept (base centroids untouched) so it reads
        as the "natural" starting distribution.
        """
        d = self.cfg.n_features
        if concept_id == 0:
            return np.eye(d), np.zeros(d)

        rng = self._concept_rng(concept_id)
        t = rng.normal(0.0, self.cfg.concept_shift, size=d)

        if not self.cfg.concept_rotate:
            return np.eye(d), t

        # A random rotation via QR of a Gaussian matrix (Haar-ish orthogonal).
        a = rng.normal(size=(d, d))
        q, r = np.linalg.qr(a)
        # Fix sign ambiguity so the transform is deterministic.
        q *= np.sign(np.diag(r))
        return q, t

    def centroids(self, concept_id: int) -> np.ndarray:
        r, t = self._transform(concept_id)
        return self.base_centroids @ r.T + t

    def sample(
        self,
        concept_id: int,
        n: int,
        rng: np.random.Generator,
        class_weights: np.ndarray | None = None,
    ):
        """Draw ``n`` labelled samples from a concept.

        ``class_weights`` lets a segment tilt the class mixture; defaults to
        uniform over classes.
        """
        centroids = self.centroids(concept_id)
        if class_weights is None:
            class_weights = np.ones(self.n_classes) / self.n_classes
        y = rng.choice(self.n_classes, size=n, p=class_weights)
        noise = rng.normal(0.0, self.cfg.class_spread, size=(n, self.cfg.n_features))
        X = centroids[y] + noise
        return X.astype(np.float32), y.astype(int)


def feature_names(cfg: DataConfig) -> List[str]:
    """CICIDS2017-flavoured column names so downstream code/plots look realistic."""
    stems = [
        "Flow_Duration", "Total_Fwd_Packets", "Total_Bwd_Packets",
        "Fwd_Pkt_Len_Mean", "Bwd_Pkt_Len_Mean", "Flow_Bytes_s",
        "Flow_Pkts_s", "Flow_IAT_Mean", "Fwd_IAT_Mean", "Bwd_IAT_Mean",
        "Fwd_PSH_Flags", "SYN_Flag_Count", "ACK_Flag_Count", "URG_Flag_Count",
        "Pkt_Len_Var", "Down_Up_Ratio", "Avg_Pkt_Size", "Init_Win_Fwd",
        "Init_Win_Bwd", "Active_Mean", "Idle_Mean",
    ]
    names = list(stems[: cfg.n_features])
    # Pad if n_features exceeds the stem list.
    while len(names) < cfg.n_features:
        names.append(f"Feature_{len(names)}")
    return names

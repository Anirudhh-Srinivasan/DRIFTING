"""Turn concepts into a stream with controlled, labelled drift.

Given a ``ScenarioConfig`` (an ordered list of segments), this builds a single
``DriftStream`` and records ground-truth ``DriftEvent``s. Each drift *type* is
realised as a specific way of moving between concepts:

* **sudden**      -- abrupt switch: the whole segment is drawn from the new
  concept starting at the segment boundary. ``onset == end``.
* **gradual**     -- during a transition window, each sample comes from the old
  concept with probability ``p`` and the new one with ``1-p``, where ``p`` ramps
  linearly from 1 -> 0. Samples oscillate between concepts (the textbook
  definition of gradual drift); after the window all samples are the new concept.
* **incremental** -- the concept *parameters* interpolate: during the window we
  draw from an intermediate concept whose centroids are a linear blend of old and
  new, moving in small steps. No oscillation; the distribution slides over.
* **recurring**   -- a sudden switch back to a concept seen earlier in the
  stream. Flagged with ``is_recurrence=True`` -- the target scenario for the
  model-reuse cache.

The first segment is expected to be ``stable`` (warm-up); it emits no event.
"""

from __future__ import annotations

from typing import List

import numpy as np

from config import DataConfig, ScenarioConfig, DriftSegment
from data.stream import DriftStream, DriftEvent
from data.synthetic import ConceptSpace, feature_names


def _blend_centroids(space: ConceptSpace, c_from: int, c_to: int, alpha: float):
    """Linear interpolation between two concepts' centroids (alpha in [0,1])."""
    return (1 - alpha) * space.centroids(c_from) + alpha * space.centroids(c_to)


def _sample_from_centroids(centroids, n, cfg: DataConfig, rng, n_classes):
    weights = np.ones(n_classes) / n_classes
    y = rng.choice(n_classes, size=n, p=weights)
    noise = rng.normal(0.0, cfg.class_spread, size=(n, cfg.n_features))
    X = centroids[y] + noise
    return X.astype(np.float32), y.astype(int)


def build_stream(
    data_cfg: DataConfig, scenario: ScenarioConfig, seed: int
) -> DriftStream:
    space = ConceptSpace(data_cfg, seed)
    rng = np.random.default_rng(seed + 7)
    n_classes = space.n_classes

    X_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []
    cid_parts: List[np.ndarray] = []
    events: List[DriftEvent] = []

    cursor = 0
    current_concept = scenario.segments[0].to_concept
    seen_concepts = set()

    for i, seg in enumerate(scenario.segments):
        seg_start = cursor

        if seg.kind == "stable":
            X, y = space.sample(seg.to_concept, seg.n_samples, rng)
            cids = np.full(seg.n_samples, seg.to_concept)
            X_parts.append(X); y_parts.append(y); cid_parts.append(cids)
            current_concept = seg.to_concept
            cursor += seg.n_samples

        elif seg.kind in ("sudden", "recurring"):
            X, y = space.sample(seg.to_concept, seg.n_samples, rng)
            cids = np.full(seg.n_samples, seg.to_concept)
            X_parts.append(X); y_parts.append(y); cid_parts.append(cids)
            events.append(DriftEvent(
                onset=seg_start, end=seg_start, kind=seg.kind,
                from_concept=current_concept, to_concept=seg.to_concept,
                is_recurrence=(seg.to_concept in seen_concepts),
            ))
            current_concept = seg.to_concept
            cursor += seg.n_samples

        elif seg.kind == "gradual":
            w = min(seg.transition, seg.n_samples)
            # Transition window: probabilistic mix, p(old) ramps 1 -> 0.
            p_old = np.linspace(1.0, 0.0, w)
            from_c = space.centroids(current_concept)
            to_c = space.centroids(seg.to_concept)
            Xw = np.empty((w, data_cfg.n_features), dtype=np.float32)
            yw = np.empty(w, dtype=int)
            for j in range(w):
                use_old = rng.random() < p_old[j]
                cent = from_c if use_old else to_c
                k = rng.integers(n_classes)
                yw[j] = k
                Xw[j] = cent[k] + rng.normal(0.0, data_cfg.class_spread, data_cfg.n_features)
            # Remainder: fully the new concept.
            rem = seg.n_samples - w
            Xr, yr = space.sample(seg.to_concept, rem, rng)
            X_parts.append(Xw); y_parts.append(yw)
            X_parts.append(Xr); y_parts.append(yr)
            cid_parts.append(np.full(w, -1))  # -1 == mixed/transition region
            cid_parts.append(np.full(rem, seg.to_concept))
            events.append(DriftEvent(
                onset=seg_start, end=seg_start + w, kind="gradual",
                from_concept=current_concept, to_concept=seg.to_concept,
                is_recurrence=(seg.to_concept in seen_concepts),
            ))
            current_concept = seg.to_concept
            cursor += seg.n_samples

        elif seg.kind == "incremental":
            w = min(seg.transition, seg.n_samples)
            # Transition window: centroids slide from old to new in small steps.
            alphas = np.linspace(0.0, 1.0, w)
            Xw = np.empty((w, data_cfg.n_features), dtype=np.float32)
            yw = np.empty(w, dtype=int)
            for j in range(w):
                cent = _blend_centroids(space, current_concept, seg.to_concept, alphas[j])
                k = rng.integers(n_classes)
                yw[j] = k
                Xw[j] = cent[k] + rng.normal(0.0, data_cfg.class_spread, data_cfg.n_features)
            rem = seg.n_samples - w
            Xr, yr = space.sample(seg.to_concept, rem, rng)
            X_parts.append(Xw); y_parts.append(yw)
            X_parts.append(Xr); y_parts.append(yr)
            cid_parts.append(np.full(w, -1))
            cid_parts.append(np.full(rem, seg.to_concept))
            events.append(DriftEvent(
                onset=seg_start, end=seg_start + w, kind="incremental",
                from_concept=current_concept, to_concept=seg.to_concept,
                is_recurrence=(seg.to_concept in seen_concepts),
            ))
            current_concept = seg.to_concept
            cursor += seg.n_samples

        else:
            raise ValueError(f"Unknown segment kind: {seg.kind!r}")

        # A concept counts as "seen" once its (post-transition) segment is placed.
        # We mark it after handling the segment so a recurrence refers to a
        # concept that genuinely appeared earlier.
        if i == 0:
            seen_concepts.add(seg.to_concept)
        else:
            seen_concepts.add(current_concept)

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    concept_ids = np.concatenate(cid_parts, axis=0)

    return DriftStream(
        X=X, y=y, concept_ids=concept_ids, events=events,
        class_names=list(data_cfg.class_names),
        feature_names=feature_names(data_cfg),
    )

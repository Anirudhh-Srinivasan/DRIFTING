"""Central, config-driven experiment settings.

Everything that affects an experiment's outcome is declared here so runs are
reproducible: a single ``seed`` drives all randomness, and every component reads
its parameters from an ``ExperimentConfig`` instance rather than hard-coding them.

Design decisions captured here (documented for the paper's Methodology section):

* Underlying task is multi-class attack-type classification. A binary
  benign/attack view is derivable from the labels (class 0 == BENIGN).
* Drift is injected at the level of the class-conditional feature distribution
  (see ``data/synthetic.py``): each *concept* applies a distinct affine transform
  to the class centroids, so a model trained on an old concept degrades when the
  concept changes -- which is what makes the prequential error signal (fed to the
  detectors) rise.
* The default scenario strings together one drift of each type plus recurring
  concepts, so a single run exercises sudden / gradual / incremental / recurring
  and the (future) model-reuse cache.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


# --------------------------------------------------------------------------- #
# Concept / feature-space parameters                                          #
# --------------------------------------------------------------------------- #
@dataclass
class DataConfig:
    """Parameters for the synthetic CICIDS2017-like generator."""

    n_features: int = 20
    # Class 0 is BENIGN; the rest stand in for CICIDS2017 attack families.
    class_names: List[str] = field(
        default_factory=lambda: ["BENIGN", "DoS", "PortScan", "BruteForce", "Botnet"]
    )
    # Within-class spread (isotropic Gaussian std around each centroid).
    class_spread: float = 0.6
    # Magnitude of the per-concept affine shift applied to class centroids.
    concept_shift: float = 2.5
    # Whether to rotate the feature space per concept (in addition to shifting).
    concept_rotate: bool = True


# --------------------------------------------------------------------------- #
# Drift scenario                                                              #
# --------------------------------------------------------------------------- #
@dataclass
class DriftSegment:
    """One segment of the simulated stream.

    ``kind`` is one of: ``"stable"``, ``"sudden"``, ``"gradual"``,
    ``"incremental"``, ``"recurring"``. ``to_concept`` is the concept the segment
    transitions *into* (or holds, for ``stable``). ``n_samples`` is the segment
    length; ``transition`` is the width of the transition window for gradual /
    incremental drift (ignored otherwise).
    """

    kind: str
    to_concept: int
    n_samples: int = 4000
    transition: int = 1500


@dataclass
class ScenarioConfig:
    """Ordered list of segments defining the full stream.

    Default: warm up on concept 0, then one drift of each type, then bring back
    two previously seen concepts (recurring) to exercise the model-reuse cache.
    """

    segments: List[DriftSegment] = field(
        default_factory=lambda: [
            DriftSegment("stable", to_concept=0, n_samples=4000),
            DriftSegment("sudden", to_concept=1, n_samples=4000),
            DriftSegment("gradual", to_concept=2, n_samples=5000, transition=1800),
            DriftSegment("incremental", to_concept=3, n_samples=5000, transition=2500),
            DriftSegment("recurring", to_concept=0, n_samples=4000),
            DriftSegment("recurring", to_concept=1, n_samples=4000),
        ]
    )


# --------------------------------------------------------------------------- #
# Detector / model / evaluation parameters                                    #
# --------------------------------------------------------------------------- #
@dataclass
class DetectorConfig:
    """Which detectors to run and their key hyper-parameters.

    Kept explicit so thresholds are traceable in the paper. river defaults are
    used unless overridden here.
    """

    adwin_delta: float = 0.002
    kswin_alpha: float = 0.005
    kswin_window_size: int = 100
    kswin_stat_size: int = 30
    # DDM / EDDM use river defaults; exposed here if we need to tune later.


@dataclass
class ModelConfig:
    """Underlying task classifier used by the pipelines."""

    # "logreg" or "random_forest"
    kind: str = "random_forest"
    rf_n_estimators: int = 60
    rf_max_depth: int = 12


@dataclass
class CacheConfig:
    """Model-reuse cache parameters.

    ``distance_threshold`` is the symmetric-KL concept distance below which a new
    concept is judged a recurrence of a cached one (calibrated against the
    simulated recurrences -- see experiments/run_cache_reuse.py).
    ``settle`` is how many post-detection samples to gather before fingerprinting
    the new concept (so we fingerprint the *new* distribution, not the old one).
    """

    distance_threshold: float = 1.0
    # Post-detection samples gathered before the reuse-vs-retrain decision. Kept
    # small: the concept-distance separation is large (recurrence ~0.03 vs novel
    # >2), so a short window fingerprints reliably, and a small settle minimises
    # the accuracy/latency cost the cache pipeline pays relative to the baseline.
    settle: int = 150
    # Window (most-recent samples) used to fingerprint a concept.
    fingerprint_window: int = 1500


@dataclass
class StrategyConfig:
    """Parameters for the type-aware retraining strategies.

    The incremental/partial update (for gradual & incremental drift) is a *fresh*
    retrain on a *smaller* recent window than the full-retrain strategy -- same
    model class, fewer samples. Training from scratch on the recent window lets
    the model learn the union of the concepts present in it, so accuracy recovers,
    while the smaller window makes it cheaper than a full retrain.

    (A warm-start partial update was evaluated first and rejected: sklearn's
    RandomForest warm-start can only *add* trees, so the stale majority keeps
    out-voting new trees on a large shift, costing accuracy. Fresh-on-smaller-
    window restores parity at ~half the full-retrain cost. See
    experiments/run_full_pipeline.py.)
    """

    incremental_window: int = 1000     # smaller than eval.retrain_window (=cheaper)


@dataclass
class EvalConfig:
    """Evaluation windows and matching tolerances."""

    warmup_size: int = 2000            # samples used to train the initial model
    retrain_window: int = 2000         # #recent samples a full retrain uses
    # A detection counts as a true positive if it falls within this many samples
    # *after* a ground-truth drift onset; otherwise it is a false alarm.
    detection_tolerance: int = 2500
    # Accuracy trajectory is reported as a rolling window.
    accuracy_window: int = 300


@dataclass
class ExperimentConfig:
    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    detectors: DetectorConfig = field(default_factory=DetectorConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    # Primary detector driving the baseline's retrain decisions.
    primary_detector: str = "adwin"
    results_dir: str = "results"


def default_config() -> ExperimentConfig:
    return ExperimentConfig()

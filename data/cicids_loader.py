"""Real CICIDS2017 loader -- stubbed behind the same interface as the synthetic
generator.

When you have the CICIDS2017 CSVs (the MachineLearningCVE / GeneratedLabelledFlows
release), point ``csv_dir`` at the folder and implement the body below. The
contract is intentionally identical to ``data.drift_simulation.build_stream``:
load + preprocess the real flows into (X, y), then hand them to the *same* drift
simulator so drift segments (sudden / gradual / incremental / recurring) are
injected on top of real data by reordering/resampling. Nothing downstream changes.

Implementation notes for later (kept here so the plan is on record for the paper):

1. Read every ``*.csv`` in ``csv_dir``; concatenate. Strip whitespace from column
   names (CICIDS2017 headers have leading spaces).
2. Drop/ýimpute the known bad rows: ``Flow Bytes/s`` and ``Flow Packets/s``
   contain Inf/NaN; coerce to numeric and drop non-finite rows.
3. Map the ``Label`` column to integer class ids with BENIGN == 0, matching the
   synthetic convention.
4. Standardise features (fit scaler on a warm-up prefix only, to avoid leaking
   future statistics into a streaming setting).
5. Build per-class sample pools, then call ``simulate_on_real`` (to be written)
   which mirrors ``build_stream`` but samples rows from those pools instead of
   Gaussians -- so the drift geometry comes from real class distributions
   reappearing/switching, and ground-truth ``DriftEvent``s are recorded the same way.
"""

from __future__ import annotations

from config import DataConfig, ScenarioConfig
from data.stream import DriftStream


def build_stream_from_cicids(
    csv_dir: str,
    data_cfg: DataConfig,
    scenario: ScenarioConfig,
    seed: int,
) -> DriftStream:
    """Load real CICIDS2017 and inject the same drift scenario. NOT YET IMPLEMENTED.

    Returns a ``DriftStream`` with identical structure to the synthetic path.
    """
    raise NotImplementedError(
        "Real CICIDS2017 loader is stubbed. Provide the CSV directory and "
        "implement the steps in this module's docstring. The synthetic generator "
        "(data.drift_simulation.build_stream) is the drop-in used until then."
    )

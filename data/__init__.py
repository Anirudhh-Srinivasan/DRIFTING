"""Data layer: synthetic CICIDS2017-like generator + drift simulation.

``load_stream`` is the single entry point downstream code should use; it selects
the synthetic generator (default) or the real-CICIDS loader based on config.
"""

from __future__ import annotations

from config import ExperimentConfig
from data.stream import DriftStream, DriftEvent
from data.drift_simulation import build_stream


def load_stream(cfg: ExperimentConfig, source: str = "synthetic",
                csv_dir: str | None = None) -> DriftStream:
    """Return a ``DriftStream`` from the chosen source.

    source == "synthetic" (default): use the distribution-matched generator.
    source == "cicids": use the real loader (currently stubbed).
    """
    if source == "synthetic":
        return build_stream(cfg.data, cfg.scenario, cfg.seed)
    if source == "cicids":
        from data.cicids_loader import build_stream_from_cicids
        if csv_dir is None:
            raise ValueError("csv_dir is required for source='cicids'")
        return build_stream_from_cicids(csv_dir, cfg.data, cfg.scenario, cfg.seed)
    raise ValueError(f"Unknown data source: {source!r}")


__all__ = ["load_stream", "DriftStream", "DriftEvent", "build_stream"]

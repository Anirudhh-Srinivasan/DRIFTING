"""river-backed drift detectors, wrapped to the ``BaseDriftDetector`` interface.

All four consume the pipeline's prequential *error* stream:

* DDM / EDDM are binary detectors -- they expect 0/1 (correct/incorrect) and
  track the error rate and its standard deviation.
* ADWIN / KSWIN are general change detectors over a real-valued signal; we feed
  them the same 0/1 error so every detector sees an identical input and the
  comparison is apples-to-apples.

Tested against river 0.26.x. The ``signal()`` dicts expose the internal
statistics each backend makes available -- these are the candidate features for
the drift-type classifier (built later).
"""

from __future__ import annotations

from typing import Dict

from river import drift
from river.drift.binary import DDM, EDDM

from config import DetectorConfig
from detectors.base import BaseDriftDetector


class ADWINDetector(BaseDriftDetector):
    name = "adwin"

    def __init__(self, cfg: DetectorConfig):
        self._d = drift.ADWIN(delta=cfg.adwin_delta)

    def update(self, error: float) -> bool:
        self._d.update(float(error))
        return self._d.drift_detected

    @property
    def drift_detected(self) -> bool:
        return self._d.drift_detected

    def signal(self) -> Dict[str, float]:
        # ADWIN's adaptive window width shrinks sharply at a change point -- a
        # strong candidate feature for distinguishing sudden vs gradual drift.
        return {
            "adwin_width": float(getattr(self._d, "width", 0.0)),
            "adwin_estimation": float(getattr(self._d, "estimation", 0.0)),
        }


class KSWINDetector(BaseDriftDetector):
    name = "kswin"

    def __init__(self, cfg: DetectorConfig):
        self._d = drift.KSWIN(
            alpha=cfg.kswin_alpha,
            window_size=cfg.kswin_window_size,
            stat_size=cfg.kswin_stat_size,
        )

    def update(self, error: float) -> bool:
        self._d.update(float(error))
        return self._d.drift_detected

    @property
    def drift_detected(self) -> bool:
        return self._d.drift_detected

    def signal(self) -> Dict[str, float]:
        return {"kswin_p_value": float(getattr(self._d, "p_value", 1.0))}


class DDMDetector(BaseDriftDetector):
    name = "ddm"

    def __init__(self, cfg: DetectorConfig):
        self._d = DDM()

    def update(self, error: float) -> bool:
        # DDM expects the misclassification indicator (1 == wrong).
        self._d.update(int(error))
        return self._d.drift_detected

    @property
    def drift_detected(self) -> bool:
        return self._d.drift_detected

    @property
    def warning_detected(self) -> bool:
        return bool(self._d.warning_detected)

    def signal(self) -> Dict[str, float]:
        # Error-rate level; its slope over a window feeds the type classifier.
        return {"ddm_p": float(getattr(self._d, "p", 0.0)),
                "ddm_warning": float(self._d.warning_detected)}


class EDDMDetector(BaseDriftDetector):
    name = "eddm"

    def __init__(self, cfg: DetectorConfig):
        self._d = EDDM()

    def update(self, error: float) -> bool:
        self._d.update(int(error))
        return self._d.drift_detected

    @property
    def drift_detected(self) -> bool:
        return self._d.drift_detected

    @property
    def warning_detected(self) -> bool:
        return bool(self._d.warning_detected)

    def signal(self) -> Dict[str, float]:
        return {"eddm_warning": float(self._d.warning_detected)}


_REGISTRY = {
    "adwin": ADWINDetector,
    "kswin": KSWINDetector,
    "ddm": DDMDetector,
    "eddm": EDDMDetector,
}


def make_detector(name: str, cfg: DetectorConfig) -> BaseDriftDetector:
    key = name.lower()
    if key not in _REGISTRY:
        raise ValueError(f"Unknown detector {name!r}; choices: {list(_REGISTRY)}")
    return _REGISTRY[key](cfg)


def all_detector_names():
    return list(_REGISTRY)

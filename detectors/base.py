"""Common detector interface.

Every detector -- whatever river class backs it -- exposes the same tiny surface
so the pipelines and the (future) drift-type classifier don't care which one is
running:

    detected = detector.update(error)   # error in {0,1} or a real value
    detector.drift_detected             # bool, last update's verdict
    detector.warning_detected           # bool, if the backend supports warnings
    detector.signal()                   # dict of interpretable internal state

The ``signal()`` dict is deliberately open-ended: it is the raw material the
drift-type classifier will consume (e.g. ADWIN window width, DDM error-rate
slope, alarm timing). Keeping it per-detector avoids baking in feature choices
before that component's design is settled.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict


class BaseDriftDetector(ABC):
    name: str = "base"

    @abstractmethod
    def update(self, error: float) -> bool:
        """Feed one observation; return True iff drift is detected on this step."""

    @property
    @abstractmethod
    def drift_detected(self) -> bool:
        ...

    @property
    def warning_detected(self) -> bool:
        """Backends without a warning zone always report False."""
        return False

    @abstractmethod
    def signal(self) -> Dict[str, float]:
        """Interpretable internal state, for the drift-type classifier."""

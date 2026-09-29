"""Detector parameters and their two string forms.

A mirror of `TouchDetectorParams` in `TouchDetector.hpp`. Two things matter for
the C++ parity tests:

* every numeric field holds a `numpy.float32` where the C++ declares a `float`,
  because the detector's geometry is done in single precision and rounding shows
  up in the reported distances and thresholds;
* `describe()` reproduces `DescribeTouchDetectorParams()` byte for byte, and
  `parse()` reads that same string back, so a test can take the parameters out
  of a C++ report and replay it here.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

F32 = np.float32


@dataclass
class DetectorParams:
    """TouchDetectorParams, field for field."""

    keypointConfidence: float = F32(0.2)
    ballMinConfidence: float = F32(0.15)
    contactDistanceRatio: float = F32(0.045)
    personHeightRatio: float = F32(0.12)
    minContactPx: float = F32(22.0)
    maxContactPx: float = F32(64.0)
    releaseRatio: float = F32(1.4)
    minContactFrames: int = 2
    maxGapFrames: int = 3
    minSplitFrames: int = 24
    minRiseFrames: int = 2
    maxClusterFrames: int = 300
    candidateCount: int = 6
    preferTrackedBall: bool = True
    speedWindowSec: float = float(0.15)
    refractoryFrames: int = 30
    personRefractoryFrames: int = 30
    eventTimeMode: int = 0
    minSpeedChangePxPerSec: float = F32(0.0)
    minSpeedRatio: float = F32(1.0)
    requireTrackedBall: bool = False

    # --- single precision for everything the C++ declares as float -----------
    _FLOAT_FIELDS = (
        "keypointConfidence",
        "ballMinConfidence",
        "contactDistanceRatio",
        "personHeightRatio",
        "minContactPx",
        "maxContactPx",
        "releaseRatio",
        "minSpeedChangePxPerSec",
        "minSpeedRatio",
    )
    # `speedWindowSec` is the one field the C++ declares as `double`. Narrowing
    # it to float would move the far edge of the ball speed window by 6e-9 s,
    # which is enough to include or exclude the sample sitting on the edge. It
    # stays a Python float, which is a C double.
    _DOUBLE_FIELDS = ("speedWindowSec",)

    def __post_init__(self) -> None:
        for name in self._FLOAT_FIELDS:
            setattr(self, name, F32(getattr(self, name)))

    def describe(self) -> str:
        """The string `DescribeTouchDetectorParams()` writes (fixed, precision 4)."""
        return (
            "keypointConfidence={p.keypointConfidence:.4f}"
            " ballMinConfidence={p.ballMinConfidence:.4f}"
            " contactDistanceRatio={p.contactDistanceRatio:.4f}"
            " personHeightRatio={p.personHeightRatio:.4f}"
            " minContactPx={p.minContactPx:.4f}"
            " maxContactPx={p.maxContactPx:.4f}"
            " releaseRatio={p.releaseRatio:.4f}"
            " minContactFrames={p.minContactFrames}"
            " maxGapFrames={p.maxGapFrames}"
            " minSplitFrames={p.minSplitFrames}"
            " minRiseFrames={p.minRiseFrames}"
            " maxClusterFrames={p.maxClusterFrames}"
            " candidateCount={p.candidateCount}"
            " preferTrackedBall={prefer}"
            " speedWindowSec={p.speedWindowSec:.4f}"
            " refractoryFrames={p.refractoryFrames}"
            " personRefractoryFrames={p.personRefractoryFrames}"
            " eventTimeMode={p.eventTimeMode}"
            " minSpeedChangePxPerSec={p.minSpeedChangePxPerSec:.4f}"
            " minSpeedRatio={p.minSpeedRatio:.4f}"
            " requireTrackedBall={required}"
        ).format(
            p=self,
            prefer="true" if self.preferTrackedBall else "false",
            required="true" if self.requireTrackedBall else "false",
        )

    @classmethod
    def parse(cls, text: str) -> "DetectorParams":
        """Read back a `describe()` string (as found in a C++ report)."""
        known = {f.name for f in fields(cls)}
        values: dict[str, object] = {}
        for token in text.split():
            if "=" not in token:
                continue
            key, _, raw = token.partition("=")
            if key not in known:
                continue
            if key in cls._DOUBLE_FIELDS:
                values[key] = float(raw)
            elif key in cls._FLOAT_FIELDS:
                values[key] = F32(float(raw))
            elif raw in ("true", "false"):
                values[key] = raw == "true"
            else:
                values[key] = int(float(raw))
        return cls(**values)

    def replace(self, **changes: object) -> "DetectorParams":
        """A copy with some fields changed (what the CLI flags do)."""
        merged = {f.name: getattr(self, f.name) for f in fields(self)}
        merged.update(changes)
        return DetectorParams(**merged)

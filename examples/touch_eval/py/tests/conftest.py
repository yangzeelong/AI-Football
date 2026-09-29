"""Shared fixtures for the differential tests.

The C++ implementation is gone from this repository, but its output is not: the
report, the events file and the dumped ground truth of the last C++ run are
frozen in `tests/reference/`, and these tests still hold the Python
implementation to them. That is deliberate - a self comparison against the
Python's own output would pass no matter what the port got wrong.

The observations are the one artefact too large to freeze (90 MB). They stay in
the run directory, and `TOUCH_EVAL_OBSERVATIONS` points at another copy.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import pytest

ROOT = Path(os.environ.get("NEXUSFLOW_ROOT", Path(__file__).resolve().parents[4]))

#: The C++ output of the fp16 run, frozen before the implementation was deleted.
#: There is only one precision path now, and this is it.
#:
#: It also holds `annotations.csv`: the annotations the C++ run was scored
#: against, rebuilt verbatim from the dump's `raw_line` in source order. The
#: tests parse that copy rather than `data/移动传球.touches.csv`, because the
#: live file is a working copy - hand edited, and since reduced by the
#: `foot`/`player` columns - and a test that reads it is not a test of the port.
REFERENCE = Path(
    os.environ.get("TOUCH_EVAL_REFERENCE", Path(__file__).resolve().parent / "reference")
)

#: The evidence the reference run was scored against.
OBSERVATIONS = Path(
    os.environ.get(
        "TOUCH_EVAL_OBSERVATIONS",
        ROOT / "output/touch_eval_moving_pass/observations.jsonl",
    )
)

#: How much of the recording the parity tests walk; the whole thing by default.
FRAMES = int(os.environ.get("TOUCH_EVAL_PARITY_FRAMES", "0")) or None

#: The detector drops events up to `maxClusterFrames` after the last frame it
#: saw, so a prefix comparison cannot judge the tail.
TAIL_FRAMES = 400


@dataclass
class Reference:
    """What one parity test needs: the frozen output and the observations."""

    observations: Path
    events: Path
    report: dict
    ground_truth: Path

    @property
    def params_text(self) -> str:
        return self.report["params"]


@pytest.fixture(scope="session")
def reference() -> Reference:
    report_path = REFERENCE / "touch_report.json"
    if not report_path.is_file():
        pytest.skip(f"frozen reference missing under {REFERENCE}")
    if not OBSERVATIONS.is_file():
        pytest.skip(f"observations missing at {OBSERVATIONS}")
    return Reference(
        observations=OBSERVATIONS,
        events=REFERENCE / "touch_events.jsonl",
        report=json.loads(report_path.read_text()),
        ground_truth=REFERENCE / "ground_truth.json",
    )


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def close_enough(mine: float, theirs: float) -> bool:
    """The C++ wrote 6 significant digits, so that is the resolution compared."""
    if mine == theirs:
        return True
    scale = max(abs(mine), abs(theirs), 1e-9)
    return abs(mine - theirs) / scale < 1e-5

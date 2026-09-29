"""Differential tests for the annotation parser.

`--dump_gt` writes every field the parser recovered, so comparing that file with
`ground_truth_json(parse_annotation_file(...))` checks the whole structure at
once: times, frame pairs, flags, indexes, warnings and the totals.
"""

from __future__ import annotations

import json

import pytest

from conftest import REFERENCE, ROOT
from touch_eval import gt

#: Which annotations the reference was produced from. The comparison parses the
#: frozen copy, not `data/移动传球.touches.csv`: the live file is a working copy,
#: edited by hand (and since reduced by the `foot`/`player` columns), and a test
#: that reads it stops being a test of the port.

#: Every annotation file the repository uses, with the JSON the C++ wrote for it.
#: The JSON is the frozen one: `output/` gets rewritten by every run, and then
#: the comparison would be against this implementation's own output.
CASES = [
    (REFERENCE / "annotations.csv", REFERENCE / "ground_truth.json"),
]


def _diff(mine: dict, theirs: dict, prefix: str = "") -> list[str]:
    """Every leaf that differs, as `path: python / cpp` lines."""
    problems: list[str] = []
    if isinstance(mine, dict) and isinstance(theirs, dict):
        for key in sorted(set(mine) | set(theirs)):
            if key not in mine:
                problems.append(f"{prefix}.{key}: missing in python")
            elif key not in theirs:
                problems.append(f"{prefix}.{key}: missing in cpp")
            else:
                problems.extend(_diff(mine[key], theirs[key], f"{prefix}.{key}"))
        return problems
    if isinstance(mine, list) and isinstance(theirs, list):
        if len(mine) != len(theirs):
            problems.append(f"{prefix}: length {len(mine)} vs {len(theirs)}")
            return problems
        for index, (left, right) in enumerate(zip(mine, theirs)):
            problems.extend(_diff(left, right, f"{prefix}[{index}]"))
        return problems
    if isinstance(mine, bool) or isinstance(theirs, bool):
        if mine != theirs:
            problems.append(f"{prefix}: {mine!r} vs {theirs!r}")
        return problems
    if isinstance(mine, (int, float)) and isinstance(theirs, (int, float)):
        if abs(float(mine) - float(theirs)) > 1e-9 * max(1.0, abs(float(theirs))):
            problems.append(f"{prefix}: {mine!r} vs {theirs!r}")
        return problems
    if mine != theirs:
        problems.append(f"{prefix}: {mine!r} vs {theirs!r}")
    return problems


@pytest.mark.parity
@pytest.mark.parametrize("annotation,reference_json", CASES, ids=lambda item: item.name)
def test_ground_truth_matches_cpp(annotation, reference_json) -> None:
    if not annotation.is_file() or not reference_json.is_file():
        pytest.skip(f"missing {annotation} or {reference_json}")
    theirs = json.loads(reference_json.read_text())
    # The C++ records the path exactly as the command line spelled it, so the
    # comparison uses whatever that run recorded, not this checkout's spelling.
    mine = gt.ground_truth_json(
        gt.parse_annotation_text(annotation.read_text(), theirs["source"])
    )
    problems = _diff(mine, theirs)
    assert not problems, "\n".join(problems[:40])


@pytest.mark.parity
def test_legacy_label_parses_without_warnings() -> None:
    """The `.label` notation must survive a round trip through the parser."""
    label = ROOT / "data/移动传球.label"
    if not label.is_file():
        pytest.skip(f"missing {label}")
    ground_truth = gt.parse_annotation_file(str(label))
    assert ground_truth.source_kind == "label"
    assert ground_truth.default_point_window_sec == 1.0
    assert ground_truth.events, "the label parser found no rows"
    assert ground_truth.touch_count() == sum(e.count for e in ground_truth.events)

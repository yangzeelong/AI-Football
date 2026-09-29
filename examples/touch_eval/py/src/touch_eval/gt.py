"""Ground truth annotations: the `.label` and `.touches.csv` notations.

A port of the parsing half of `TouchGroundTruth.cpp`. Two formats matter:

* the legacy `.label`, one handwritten line per touch (`00:07S（2次）`), where a
  whole second is the annotation's resolution - one second of window;
* the v2 CSV, one row per touch, with optional frame columns, an `ignore` flag
  for rows that are not ground truth at all, and a `low_quality_data` flag for
  frames too degraded to judge.

Both are turned into the same `GroundTruth`, because every rule downstream (the
window a row owns, the slot count, the coverage check) reads that structure.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Iterable

# ---------------------------------------------------------------------------
# helpers that mirror the C++ byte level utilities
# ---------------------------------------------------------------------------

_SPACE = " \t\r\n\f\v"

#: Separators `TrimNote()` strips off the front of a remark, as code points.
_NOTE_SEPARATORS = frozenset(
    [
        0x20, 0x09, 0x2D, 0x7E, 0x2C, 0x3A, 0x2E, 0x5F, 0x2F, 0x29, 0x5D, 0x7C,
        0xFF0C,  # ，
        0x3001,  # 、
        0xFF1A,  # ：
        0x3002,  # 。
        0xFF1B,  # ；
        0xFF09,  # ）
        0x2014,  # —
        0x2013,  # –
        0xFF5E,  # ～
        0x2212,  # −
        0x00B7,  # ·
    ]
)


def trim(text: str) -> str:
    return text.strip(_SPACE)


def lower_ascii(text: str) -> str:
    return "".join(
        chr(ord(ch) + 32) if "A" <= ch <= "Z" else ch for ch in text
    )


def strip_bom(line: str) -> str:
    return line[1:] if line.startswith("\ufeff") else line


def split_csv_line(line: str) -> list[str]:
    """`SplitCsvLine()`: quotes doubled, a full width comma also separates."""
    fields: list[str] = []
    current: list[str] = []
    in_quotes = False
    index = 0
    while index < len(line):
        value = line[index]
        if in_quotes:
            if value == '"':
                if index + 1 < len(line) and line[index + 1] == '"':
                    current.append('"')
                    index += 1
                else:
                    in_quotes = False
            else:
                current.append(value)
            index += 1
            continue
        if value == '"':
            in_quotes = True
        elif value == "," or (value == "\uff0c" and line.startswith("\uff0c", index)):
            fields.append(trim("".join(current)))
            current = []
            if value != ",":
                index += 2
        else:
            current.append(value)
        index += 1
    fields.append(trim("".join(current)))
    return fields


def trim_note(text: str) -> str:
    """`TrimNote()`: drop leading separators, then trailing whitespace."""
    begin = 0
    while begin < len(text):
        if ord(text[begin]) not in _NOTE_SEPARATORS:
            break
        begin += 1
    end = len(text)
    while end > begin and text[end - 1] in _SPACE:
        end -= 1
    return text[begin:end]


def _atof(text: str) -> float:
    """`atof()`: the numeric prefix, 0 when there is none."""
    match = re.match(r"\s*[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?", text)
    if not match:
        return 0.0
    try:
        return float(match.group(0))
    except ValueError:  # pragma: no cover - the regex already guarantees this
        return 0.0


def _is_number_char(ch: str) -> bool:
    return ch.isdigit() or ch in ".+- "


def parse_csv_time(text: str) -> float | None:
    """`ParseCsvTime()`: `12.34`, `1:23.45`, `00:01:02.5`, a trailing `s`."""
    body = text
    while True:
        stripped = trim(body)
        if stripped and stripped[-1] in "sS":
            body = stripped[:-1]
            continue
        body = stripped
        break
    if not body:
        return None

    parts: list[str] = []
    current: list[str] = []
    index = 0
    while index < len(body):
        ch = body[index]
        if ch == ":":
            parts.append("".join(current))
            current = []
        elif ch == "\uff1a":
            # A full width colon is three bytes; anything else is malformed.
            parts.append("".join(current))
            current = []
            index += 2
        else:
            current.append(ch)
        index += 1
    parts.append("".join(current))

    for part in parts:
        for ch in part:
            if not _is_number_char(ch):
                return None

    total = 0.0
    for part in parts:
        total = total * 60.0 + _atof(part)
    return total


class FlagValue:
    UNSET = 0
    SET = 1
    UNKNOWN = 2


_UNSET_FLAGS = frozenset(["", "no", "n", "false", "0", "-", "ok", "none", "正常", "否", "无"])
_SET_FLAGS = frozenset(
    [
        "yes", "y", "true", "1", "x", "low", "blur", "blurry", "低", "低质量", "模糊",
        "低质量数据", "ignore", "ignored", "exclude", "excluded", "skip", "drop",
        "无效", "忽略", "不计", "剔除",
    ]
)


def read_flag(text: str) -> int:
    """`ReadFlag()`: how a yes/no column reads, case insensitively."""
    key = lower_ascii(trim(text))
    if key in _UNSET_FLAGS:
        return FlagValue.UNSET
    if key in _SET_FLAGS:
        return FlagValue.SET
    return FlagValue.UNKNOWN


def normalize_foot_hint(text: str) -> str:
    """`NormalizeFootHint()`: an empty string when the column says nothing."""
    key = lower_ascii(trim(text))
    if not key:
        return ""
    if key == "left" or key == "l" or "左脚" in text:
        return "left"
    if key == "right" or key == "r" or "右脚" in text:
        return "right"
    if key == "both" or "左右脚" in text:
        return "both"
    return ""


_NO_TOUCH_TYPES = frozenset(
    ["no_touch", "no-touch", "notouch", "none", "negative", "no", "无触球", "无", "空"]
)
_TOUCH_TYPES = frozenset(["", "touch", "t", "yes", "触球", "有"])


def is_no_touch_type(text: str) -> bool:
    return lower_ascii(trim(text)) in _NO_TOUCH_TYPES


def is_touch_type(text: str) -> bool:
    return lower_ascii(trim(text)) in _TOUCH_TYPES


# ---------------------------------------------------------------------------
# the structures both notations produce
# ---------------------------------------------------------------------------


@dataclass
class GroundTruthEvent:
    index: int = 0
    count: int = 1
    start_sec: float = 0.0
    end_sec: float = 0.0
    start_frame: int = -1
    end_frame: int = -1
    frames_stated: bool = False
    low_quality: bool = False
    ignored: bool = False
    foot_hint: str = ""
    player_hint: str = ""
    note: str = ""
    raw_line: str = ""
    source_line: int = 0
    source_row: int = 0
    warnings: list[str] = field(default_factory=list)

    def is_interval(self) -> bool:
        """`IsInterval()`: a span the annotation wrote out, not an instant."""
        return self.end_sec > self.start_sec

    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec


@dataclass
class GroundTruth:
    events: list[GroundTruthEvent] = field(default_factory=list)
    no_touch: list[GroundTruthEvent] = field(default_factory=list)
    ignored: list[GroundTruthEvent] = field(default_factory=list)
    source: str = ""
    source_kind: str = ""
    fps: float = 0.0
    default_point_window_sec: float = 0.0
    warnings: list[str] = field(default_factory=list)
    skipped_rows: int = 0

    def touch_count(self) -> int:
        """`TouchCount()`: the slots the touches claim, `(2次)` counting twice."""
        return sum(event.count for event in self.events)

    def low_quality_rows(self) -> int:
        return sum(
            1
            for event in list(self.events) + list(self.no_touch)
            if event.low_quality
        )

    def low_quality_touch_count(self) -> int:
        return sum(event.count for event in self.events if event.low_quality)

    def ignored_rows(self) -> int:
        return len(self.ignored)

    def ignored_touch_count(self) -> int:
        return sum(event.count for event in self.ignored if event.count > 0)


# ---------------------------------------------------------------------------
# the v2 CSV
# ---------------------------------------------------------------------------


class _Column:
    IGNORED = 0
    INDEX = 1
    TYPE = 2
    START = 3
    END = 4
    START_FRAME = 5
    END_FRAME = 6
    FRAME = 7
    COUNT = 8
    FOOT = 9
    PLAYER = 10
    LOW_QUALITY = 11
    EXCLUDE = 12
    NOTE = 13


_LOW_QUALITY_NAMES = frozenset(
    ["low_quality_data", "low_quality", "lowquality", "lowq", "low", "quality",
     "blur", "blurry", "低质量", "模糊", "低质量数据"]
)
_EXCLUDE_NAMES = frozenset(
    ["ignore", "ignored", "exclude", "excluded", "skip", "drop", "无效", "忽略",
     "不计", "剔除"]
)

_COLUMN_NAMES = {
    _Column.INDEX: ("index", "idx", "id", "序号"),
    _Column.TYPE: ("type", "kind", "类型"),
    _Column.START: ("t_start_sec", "t_start", "start_sec", "start", "开始"),
    _Column.END: ("t_end_sec", "t_end", "end_sec", "end", "结束"),
    _Column.START_FRAME: ("t_start_frame", "start_frame", "begin_frame", "frame_start", "开始帧"),
    _Column.END_FRAME: ("t_end_frame", "end_frame", "frame_end", "结束帧"),
    _Column.FRAME: ("frame", "t_frame", "帧"),
    _Column.COUNT: ("count", "n", "times", "次数"),
    _Column.FOOT: ("foot", "side", "脚", "脚别"),
    _Column.PLAYER: ("player", "player_id", "球员"),
    _Column.NOTE: ("note", "remark", "comment", "备注"),
}


def classify_column(name: str) -> int:
    """`ClassifyColumn()`: what a header cell means, case insensitively."""
    key = lower_ascii(trim(name))
    if key in _LOW_QUALITY_NAMES:
        return _Column.LOW_QUALITY
    if key in _EXCLUDE_NAMES:
        return _Column.EXCLUDE
    for column, names in _COLUMN_NAMES.items():
        if key in names:
            return column
    return _Column.IGNORED


def parse_number_hint(comment: str, key: str) -> float:
    """`ParseNumberHint()`: `# fps: 60` / `# point_window_sec: 1.00`."""
    if not comment or comment[0] != "#":
        return 0.0
    body = lower_ascii(comment)
    position = body.find(key)
    if position < 0:
        return 0.0
    rest = comment[position + len(key):]
    if rest and rest[0].isascii() and rest[0].isalpha():
        # "fps" must not match "fpsomething".
        return 0.0
    if lower_ascii(rest).startswith("sec"):
        rest = rest[3:]
    match = re.search(r"[0-9]", rest)
    if not match:
        return 0.0
    value = _atof(rest[match.start():])
    return value if value > 0.0 else 0.0


def parse_touch_csv_text(text: str, source: str, fps_override: float = 0.0) -> GroundTruth:
    """`ParseTouchCsvText()`: the whole v2 notation, comments and all."""
    ground_truth = GroundTruth(
        source=source,
        source_kind="csv",
        default_point_window_sec=0.5,
    )
    if fps_override > 0.0:
        ground_truth.fps = fps_override

    columns: list[int] = []
    have_header = False
    data_row = 0
    fallback_index = 0
    used_indices: set[int] = set()
    warned_frames_ignored = False

    def assign_index(event: GroundTruthEvent, line_number: int, stated: int) -> None:
        nonlocal fallback_index
        if stated > 0 and stated not in used_indices:
            used_indices.add(stated)
            event.index = stated
            return
        candidate = fallback_index + 1
        while candidate in used_indices:
            candidate += 1
        fallback_index = candidate
        used_indices.add(candidate)
        event.index = candidate
        if stated > 0:
            event.warnings.append(
                f"line {line_number}: index {stated} is already used, "
                f"renumbered to {candidate}"
            )

    for line_number, raw in enumerate(text.splitlines(), start=1):
        line = strip_bom(raw) if line_number == 1 else raw
        trimmed = trim(line)
        if not trimmed or trimmed[0] == "#":
            window_sec = parse_number_hint(trimmed, "point_window")
            if window_sec > 0.0:
                ground_truth.default_point_window_sec = window_sec
            if ground_truth.fps <= 0.0:
                fps = parse_number_hint(trimmed, "fps")
                if fps > 0.0:
                    ground_truth.fps = fps
            continue

        fields = split_csv_line(line)
        if not have_header:
            columns = [classify_column(field) for field in fields]
            if _Column.START in columns:
                have_header = True
                continue
            # No header: the fixed eight column layout, deliberately without the
            # flag columns so an old note column stays a note.
            columns = [
                _Column.INDEX,
                _Column.TYPE,
                _Column.START,
                _Column.END,
                _Column.COUNT,
                _Column.FOOT,
                _Column.PLAYER,
                _Column.NOTE,
            ]

        data_row += 1
        event = GroundTruthEvent(raw_line=raw, source_line=line_number, source_row=data_row)
        type_text = ""
        stated_index = 0
        count = 1
        has_start = False
        single_frame = -1
        start_frame = -1
        end_frame = -1
        skipped = False

        for index, field in enumerate(fields):
            if index >= len(columns):
                break
            column = columns[index]
            if column == _Column.IGNORED or not field:
                continue
            if column == _Column.TYPE:
                type_text = field
            elif column == _Column.START_FRAME:
                start_frame = int(_atof(field))
            elif column == _Column.END_FRAME:
                end_frame = int(_atof(field))
            elif column == _Column.FRAME:
                single_frame = int(_atof(field))
            elif column == _Column.COUNT:
                count = int(_atof(field))
            elif column == _Column.FOOT:
                event.foot_hint = normalize_foot_hint(field)
            elif column == _Column.PLAYER:
                event.player_hint = trim(field)
            elif column == _Column.NOTE:
                event.note = trim(field)
            elif column == _Column.INDEX:
                stated_index = int(_atof(field))
            elif column == _Column.START:
                seconds = parse_csv_time(field)
                if seconds is None:
                    ground_truth.warnings.append(
                        f'line {line_number}: cannot read start time "{field}", row skipped'
                    )
                    has_start = False
                    count = -2
                    ground_truth.skipped_rows += 1
                    skipped = True
                else:
                    event.start_sec = seconds
                    event.end_sec = seconds
                    has_start = True
            elif column == _Column.END:
                seconds = parse_csv_time(field)
                if seconds is not None:
                    event.end_sec = seconds
            elif column == _Column.LOW_QUALITY:
                flag = read_flag(field)
                if flag == FlagValue.SET:
                    event.low_quality = True
                elif flag == FlagValue.UNKNOWN:
                    event.warnings.append(
                        f'cannot read low_quality_data "{field}", read as not low quality'
                    )
            elif column == _Column.EXCLUDE:
                flag = read_flag(field)
                if flag == FlagValue.SET:
                    event.ignored = True
                elif flag == FlagValue.UNKNOWN:
                    event.warnings.append(
                        f'cannot read the ignore flag "{field}", read as not ignored'
                    )

        if count == -2 or skipped:
            # The start time was unreadable: the row is dropped, not guessed.
            if skipped and type_text is None:  # pragma: no cover - defensive
                continue
            if count == -2:
                continue
        if not has_start and not (single_frame >= 0 or start_frame >= 0):
            ground_truth.warnings.append(
                f'line {line_number}: no start time, row skipped: "{raw}"'
            )
            ground_truth.skipped_rows += 1
            continue

        has_frames = single_frame >= 0 or start_frame >= 0
        if has_frames:
            if ground_truth.fps <= 0.0:
                event.warnings.append(
                    "frame given but the file declares no fps, using the seconds"
                )
                if not warned_frames_ignored:
                    ground_truth.warnings.append(
                        "frame columns are present but the file declares no fps "
                        "(# fps: 60 or --fps): every frame column is ignored and "
                        "the seconds are used as written"
                    )
                    warned_frames_ignored = True
            else:
                first = single_frame if single_frame >= 0 else start_frame
                last = single_frame if single_frame >= 0 else end_frame
                if last < 0:
                    last = first
                if last < first:
                    first, last = last, first
                event.start_frame = first
                event.end_frame = last
                event.frames_stated = True
                event.start_sec = first / ground_truth.fps
                # The frame columns include both ends.
                event.end_sec = (last + 1) / ground_truth.fps
                has_start = True

        if event.end_sec <= 0.0:
            event.end_sec = event.start_sec
        if ground_truth.fps > 0.0 and event.start_frame < 0:
            event.start_frame = round(event.start_sec * ground_truth.fps)
            event.end_frame = round(event.end_sec * ground_truth.fps)
        if event.end_sec < event.start_sec:
            event.end_sec, event.start_sec = event.start_sec, event.end_sec
            event.warnings.append("end before start, swapped")

        if is_no_touch_type(type_text):
            event.count = 0
            assign_index(event, line_number, stated_index)
            if event.ignored:
                ground_truth.ignored.append(event)
            else:
                ground_truth.no_touch.append(event)
            continue

        if not is_touch_type(type_text):
            event.warnings.append(f'unknown type "{type_text}", read as a touch')
        event.count = count if count >= 1 else 1
        if count == 0:
            event.warnings.append("touch row with count 0, assuming 1")
        assign_index(event, line_number, stated_index)
        if event.ignored:
            ground_truth.ignored.append(event)
        else:
            ground_truth.events.append(event)

    return ground_truth


# ---------------------------------------------------------------------------
# the legacy .label
# ---------------------------------------------------------------------------

_COUNT_PATTERN = re.compile(r"(?:\(|（)\s*([0-9]+)\s*次\s*(?:\)|）)")
_TIME_PATTERN = re.compile(
    r"([0-9]{1,3})\s*([:.：])\s*([0-9]{1,2})(?:\s*([:.：])\s*([0-9]{1,2}))?"
)


def _contains_dash(text: str) -> bool:
    return any(token in text for token in ("-", "~", "到", "—", "～"))


def _foot_hint_of(text: str) -> str:
    if "左右脚" in text:
        return "both"
    if "右脚" in text:
        return "right"
    if "左脚" in text:
        return "left"
    return ""


def _parse_label_line(
    raw_line: str, line_number: int, ground_truth: GroundTruth
) -> None:
    line = trim(raw_line)
    if not line or line[0] == "#":
        return

    count_match = _COUNT_PATTERN.search(line)
    count = 1
    count_end = 0
    if count_match:
        count = max(1, int(_atof(count_match.group(1))))
        count_end = count_match.end()

    tokens: list[tuple[int, int, float, bool]] = []
    for match in _TIME_PATTERN.finditer(line):
        first = int(_atof(match.group(1)))
        second = int(_atof(match.group(3)))
        dotted = match.group(2) == "."
        if match.group(5) is not None:
            third = int(_atof(match.group(5)))
            seconds = first * 3600.0 + second * 60.0 + third
        else:
            seconds = first * 60.0 + second
        tokens.append((match.start(), match.end(), seconds, dotted))

    if not tokens:
        ground_truth.warnings.append(
            f'line {line_number} skipped, no timestamp: "{line}"'
        )
        return

    warnings: list[str] = []
    event = GroundTruthEvent(
        raw_line=raw_line,
        source_line=line_number,
        count=count,
        start_sec=tokens[0][2],
        end_sec=tokens[0][2],
    )
    last_time_end = tokens[0][1]

    if tokens[0][3]:
        warnings.append(f"line {line_number}: dotted time read as mm:ss")

    if len(tokens) > 1:
        between = line[tokens[0][1]:tokens[1][0]]
        if _contains_dash(between):
            event.end_sec = tokens[1][2]
            last_time_end = tokens[1][1]
            if tokens[1][3]:
                warnings.append(
                    f"line {line_number}: dotted interval end read as mm:ss"
                )
            if event.end_sec < event.start_sec:
                event.end_sec, event.start_sec = event.start_sec, event.end_sec
                warnings.append(
                    f"line {line_number}: interval end before start, swapped"
                )
            if event.duration_sec() > 10.0:
                warnings.append(
                    f"line {line_number}: interval longer than 10 s "
                    f"({event.duration_sec()} s), please verify"
                )
        else:
            warnings.append(
                f'line {line_number}: extra timestamp ignored: '
                f'"{line[tokens[1][0]:tokens[1][1]]}"'
            )

    note_begin = max(last_time_end, count_end)
    if note_begin < len(line):
        event.note = trim_note(line[note_begin:])
    event.foot_hint = _foot_hint_of(line)

    if not count_match:
        warnings.append(f"line {line_number}: no (N次) count, assuming 1")

    event.warnings = warnings
    event.index = len(ground_truth.events) + 1
    ground_truth.events.append(event)


def parse_touch_label_text(text: str, source: str) -> GroundTruth:
    """`ParseTouchLabelText()`: a handwritten line is one second of window."""
    ground_truth = GroundTruth(
        source=source,
        source_kind="label",
        default_point_window_sec=1.0,
    )
    for line_number, raw in enumerate(text.splitlines(), start=1):
        line = strip_bom(raw) if line_number == 1 else raw
        _parse_label_line(line, line_number, ground_truth)
    return ground_truth


# ---------------------------------------------------------------------------
# entry points
# ---------------------------------------------------------------------------


def apply_fps(ground_truth: GroundTruth, fps: float) -> None:
    """`ApplyFpsToGroundTruth()`: fill the frame columns seconds imply."""
    if fps <= 0.0:
        return
    for event in list(ground_truth.events) + list(ground_truth.no_touch):
        if event.start_frame < 0:
            event.start_frame = round(event.start_sec * fps)
            event.end_frame = round(event.end_sec * fps)


def parse_annotation_text(text: str, source: str, fps_override: float = 0.0) -> GroundTruth:
    """Parse by extension, like `ParseTouchAnnotationFile()`."""
    lowered = source.lower()
    if lowered.endswith(".csv") or lowered.endswith(".tsv"):
        return parse_touch_csv_text(text, source, fps_override)
    if lowered.endswith(".json"):
        ground_truth = ground_truth_from_json(json.loads(text), source)
        apply_fps(ground_truth, fps_override)
        return ground_truth
    ground_truth = parse_touch_label_text(text, source)
    apply_fps(ground_truth, fps_override)
    return ground_truth


def parse_annotation_file(path: str, fps_override: float = 0.0) -> GroundTruth:
    with open(path, "r", encoding="utf-8") as handle:
        return parse_annotation_text(handle.read(), path, fps_override)


def ground_truth_from_json(root: dict, source: str = "") -> GroundTruth:
    """`ReadGroundTruthJson()`: the `--dump_gt` file read back in."""

    def build(item: dict) -> GroundTruthEvent:
        return GroundTruthEvent(
            index=int(item.get("index", 0)),
            count=int(item.get("count", 1)),
            start_sec=float(item.get("start_sec", 0.0)),
            end_sec=float(item.get("end_sec", 0.0)),
            start_frame=int(item.get("start_frame", -1)),
            end_frame=int(item.get("end_frame", -1)),
            frames_stated=bool(item.get("frames_stated", False)),
            low_quality=bool(item.get("low_quality", False)),
            ignored=bool(item.get("ignored", False)),
            foot_hint=str(item.get("foot_hint", "")),
            player_hint=str(item.get("player_hint", "")),
            note=str(item.get("note", "")),
            raw_line=str(item.get("raw_line", "")),
            source_line=int(item.get("source_line", 0)),
            source_row=int(item.get("source_row", 0)),
            warnings=[str(w) for w in item.get("warnings", [])],
        )

    ground_truth = GroundTruth(
        events=[build(item) for item in root.get("events", [])],
        no_touch=[build(item) for item in root.get("no_touch", [])],
        ignored=[build(item) for item in root.get("ignored", [])],
        source=str(root.get("source", source)),
        source_kind=str(root.get("source_kind", "")),
        fps=float(root.get("fps", 0.0)),
        default_point_window_sec=float(root.get("default_point_window_sec", 0.0)),
        warnings=[str(w) for w in root.get("warnings", [])],
        skipped_rows=int(root.get("totals", {}).get("skipped_rows", 0)),
    )
    return ground_truth


def _event_json(event: GroundTruthEvent) -> dict:
    item: dict = {
        "index": event.index,
        "source_line": event.source_line,
        "start_sec": event.start_sec,
        "end_sec": event.end_sec,
        "frames_stated": event.frames_stated,
        "is_interval": event.is_interval(),
        "count": event.count,
        "foot_hint": event.foot_hint,
        "player_hint": event.player_hint,
        "low_quality": event.low_quality,
        "ignored": event.ignored,
        "source_row": event.source_row,
        "note": event.note,
        "raw_line": event.raw_line,
        "warnings": list(event.warnings),
    }
    if event.frames_stated and event.start_frame >= 0:
        item["start_frame"] = event.start_frame
    if event.frames_stated and event.end_frame >= 0:
        item["end_frame"] = event.end_frame
    return item


def ground_truth_json(ground_truth: GroundTruth) -> dict:
    """`WriteGroundTruthJson()`, as a dict (the CLI writes it out)."""
    return {
        "schema_version": 2,
        "source": ground_truth.source,
        "source_kind": ground_truth.source_kind,
        "default_point_window_sec": ground_truth.default_point_window_sec,
        "fps": ground_truth.fps,
        "generated_by": "touch_eval --dump_gt",
        "totals": {
            "events": len(ground_truth.events),
            "touches": ground_truth.touch_count(),
            "no_touch_spans": len(ground_truth.no_touch),
            "low_quality_rows": ground_truth.low_quality_rows(),
            "low_quality_touches": ground_truth.low_quality_touch_count(),
            "ignored_rows": ground_truth.ignored_rows(),
            "ignored_touches": ground_truth.ignored_touch_count(),
            "skipped_rows": ground_truth.skipped_rows,
        },
        "warnings": list(ground_truth.warnings),
        "events": [_event_json(event) for event in ground_truth.events],
        "no_touch": [_event_json(event) for event in ground_truth.no_touch],
        "ignored": [_event_json(event) for event in ground_truth.ignored],
    }

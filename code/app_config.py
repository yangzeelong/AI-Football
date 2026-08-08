from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from football_vision import Detection


@dataclass(frozen=True)
class BoxFilterConfig:
    min_confidence: float = 0.0
    min_width_px: float = 0.0
    min_height_px: float = 0.0


@dataclass(frozen=True)
class AppConfig:
    person_labels: set[str]
    ball_labels: set[str]
    person_filter: BoxFilterConfig
    ball_filter: BoxFilterConfig

    @classmethod
    def default(cls) -> "AppConfig":
        return cls(
            person_labels={"person"},
            ball_labels={"sports ball", "ball", "football", "soccer ball"},
            person_filter=BoxFilterConfig(
                min_confidence=0.25,
                min_width_px=8.0,
                min_height_px=24.0,
            ),
            ball_filter=BoxFilterConfig(
                min_confidence=0.15,
                min_width_px=3.0,
                min_height_px=3.0,
            ),
        )

    @classmethod
    def from_yaml(cls, config_path: str | Path | None) -> "AppConfig":
        config = cls.default()
        if not config_path:
            return config

        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"App config not found: {path}")

        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError(
                "PyYAML is required for app YAML config. Install PyYAML."
            ) from exc

        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"App config must be a YAML mapping: {path}")

        # 配置采用严格字段校验，字段名拼错时立刻报错，不静默使用默认值。
        _require_exact_keys(raw, {"labels", "filters"}, "app config")
        labels = raw["labels"]
        filters = raw["filters"]
        _require_exact_keys(labels, {"person", "ball"}, "labels")
        _require_exact_keys(filters, {"person", "ball"}, "filters")
        return cls(
            person_labels=_string_set(labels["person"], "labels.person"),
            ball_labels=_string_set(labels["ball"], "labels.ball"),
            person_filter=_box_filter_from_raw(
                filters["person"],
                "filters.person",
            ),
            ball_filter=_box_filter_from_raw(
                filters["ball"],
                "filters.ball",
            ),
        )

    def filter_detections(
        self,
        detections: Sequence[Detection],
    ) -> list[Detection]:
        kept: list[Detection] = []
        for detection in detections:
            # 只对关心的 person/ball 做阈值过滤，其他类别保留给上游 class filter 决定。
            filter_config = self._filter_for_label(detection.label)
            if filter_config is None:
                kept.append(detection)
                continue
            if self.passes_box(detection.label, detection.confidence,
                               detection.bbox):
                kept.append(detection)
        return kept

    def passes_box(self, label: str, confidence: float,
                   bbox: tuple[float, float, float, float]) -> bool:
        filter_config = self._filter_for_label(label)
        if filter_config is None:
            return True
        return _passes_box_filter(confidence, bbox, filter_config)

    def _filter_for_label(self, label: str) -> BoxFilterConfig | None:
        if label in self.person_labels:
            return self.person_filter
        if label in self.ball_labels:
            return self.ball_filter
        return None


def _box_filter_from_raw(raw: dict, section: str) -> BoxFilterConfig:
    _require_exact_keys(raw, {"min_confidence", "min_width_px", "min_height_px"},
                        section)
    return BoxFilterConfig(
        min_confidence=float(raw["min_confidence"]),
        min_width_px=float(raw["min_width_px"]),
        min_height_px=float(raw["min_height_px"]),
    )


def _require_exact_keys(data: dict, expected: set[str], section: str) -> None:
    if not isinstance(data, dict):
        raise ValueError(f"{section} must be a mapping")
    actual = set(data)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        # 同时报告缺失字段和未知字段，便于快速定位 YAML 拼写错误。
        raise ValueError(
            f"Invalid {section} keys: missing={missing} unknown={unknown}")


def _string_set(value, section: str) -> set[str]:
    if not isinstance(value, list) or not all(
            isinstance(item, str) for item in value):
        raise ValueError(f"{section} must be a list of strings")
    return set(value)


def _passes_box_filter(confidence: float,
                       bbox: tuple[float, float, float, float],
                       filter_config: BoxFilterConfig) -> bool:
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    return (
        confidence >= filter_config.min_confidence
        and width >= filter_config.min_width_px
        and height >= filter_config.min_height_px
    )

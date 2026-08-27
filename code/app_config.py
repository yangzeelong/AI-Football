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
class BoxSmoothingConfig:
    enabled: bool = True
    static_alpha: float = 0.15
    moving_alpha: float = 0.45
    static_distance_px: float = 8.0
    ttl_frames: int = 60


@dataclass(frozen=True)
class PoseBoxExpansionConfig:
    enabled: bool = True
    x_pad_ratio: float = 0.15
    y_pad_ratio: float = 0.25
    min_pad_px: float = 12.0


@dataclass(frozen=True)
class DetectionFilterDetail:
    detection: Detection
    failed_confidence: bool
    failed_size: bool


@dataclass(frozen=True)
class DetectionFilterResult:
    kept: list[Detection]
    rejected: list[DetectionFilterDetail]
    confidence_filtered_count: int
    size_filtered_count: int

    @property
    def filtered_count(self) -> int:
        return len(self.rejected)


@dataclass(frozen=True)
class KeypointSmoothingConfig:
    enabled: bool = False
    method: str = "one_euro"
    static_window: int = 8
    static_motion_px: float = 2.5
    deadband_px: float = 1.5
    low_confidence: float = 0.3
    low_confidence_alpha: float = 0.12
    min_cutoff: float = 0.8
    beta: float = 0.03
    d_cutoff: float = 1.0
    static_alpha: float = 0.15
    moving_alpha: float = 0.55
    jitter_alpha: float = 0.08
    max_static_jump_px: float = 8.0
    skeleton_anchor_alpha: float = 0.35
    skeleton_endpoint_alpha: float = 0.12
    skeleton_moving_endpoint_alpha: float = 0.45
    skeleton_fast_endpoint_alpha: float = 0.22
    skeleton_fast_moving_endpoint_alpha: float = 0.62
    skeleton_fast_static_step_px: float = 5.0
    skeleton_max_static_step_px: float = 2.5
    skeleton_limb_tolerance: float = 0.25
    sensitive_keypoints: set[str] = frozenset()


@dataclass(frozen=True)
class AppConfig:
    person_labels: set[str]
    ball_labels: set[str]
    person_filter: BoxFilterConfig
    ball_filter: BoxFilterConfig
    box_smoothing: BoxSmoothingConfig
    pose_box_expansion: PoseBoxExpansionConfig
    keypoint_smoothing: KeypointSmoothingConfig

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
            box_smoothing=BoxSmoothingConfig(),
            pose_box_expansion=PoseBoxExpansionConfig(),
            keypoint_smoothing=KeypointSmoothingConfig(),
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
        _require_exact_keys(raw,
                            {
                                "labels",
                                "filters",
                                "box_smoothing",
                                "pose_box_expansion",
                                "keypoint_smoothing",
                            },
                            "app config")
        labels = raw["labels"]
        filters = raw["filters"]
        box_smoothing = raw["box_smoothing"]
        pose_box_expansion = raw["pose_box_expansion"]
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
            box_smoothing=_box_smoothing_from_raw(
                box_smoothing,
                "box_smoothing",
            ),
            pose_box_expansion=_pose_box_expansion_from_raw(
                pose_box_expansion,
                "pose_box_expansion",
            ),
            keypoint_smoothing=_keypoint_smoothing_from_raw(
                raw["keypoint_smoothing"],
                "keypoint_smoothing",
            ),
        )

    def filter_detections(
        self,
        detections: Sequence[Detection],
    ) -> list[Detection]:
        return self.filter_detections_with_details(detections).kept

    def filter_detections_with_details(
        self,
        detections: Sequence[Detection],
    ) -> DetectionFilterResult:
        kept: list[Detection] = []
        rejected: list[DetectionFilterDetail] = []
        confidence_filtered_count = 0
        size_filtered_count = 0
        for detection in detections:
            # 只对关心的 person/ball 做阈值过滤，其他类别保留给上游 class filter 决定。
            filter_config = self._filter_for_label(detection.label)
            if filter_config is None:
                kept.append(detection)
                continue

            failed_confidence, failed_size = _evaluate_box_filter(
                detection.confidence,
                detection.bbox,
                filter_config,
            )
            if not failed_confidence and not failed_size:
                kept.append(detection)
                continue

            rejected.append(
                DetectionFilterDetail(
                    detection=detection,
                    failed_confidence=failed_confidence,
                    failed_size=failed_size,
                ))
            if failed_confidence:
                confidence_filtered_count += 1
            if failed_size:
                size_filtered_count += 1

        return DetectionFilterResult(
            kept=kept,
            rejected=rejected,
            confidence_filtered_count=confidence_filtered_count,
            size_filtered_count=size_filtered_count,
        )

    def passes_box(self, label: str, confidence: float,
                   bbox: tuple[float, float, float, float]) -> bool:
        filter_config = self._filter_for_label(label)
        if filter_config is None:
            return True
        failed_confidence, failed_size = _evaluate_box_filter(
            confidence,
            bbox,
            filter_config,
        )
        return not failed_confidence and not failed_size

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


def _box_smoothing_from_raw(raw: dict, section: str) -> BoxSmoothingConfig:
    _require_exact_keys(
        raw,
        {"enabled", "static_alpha", "moving_alpha", "static_distance_px", "ttl_frames"},
        section,
    )
    return BoxSmoothingConfig(
        enabled=bool(raw["enabled"]),
        static_alpha=float(raw["static_alpha"]),
        moving_alpha=float(raw["moving_alpha"]),
        static_distance_px=float(raw["static_distance_px"]),
        ttl_frames=int(raw["ttl_frames"]),
    )


def _pose_box_expansion_from_raw(
    raw: dict,
    section: str,
) -> PoseBoxExpansionConfig:
    _require_exact_keys(raw, {"enabled", "x_pad_ratio", "y_pad_ratio", "min_pad_px"},
                        section)
    return PoseBoxExpansionConfig(
        enabled=bool(raw["enabled"]),
        x_pad_ratio=float(raw["x_pad_ratio"]),
        y_pad_ratio=float(raw["y_pad_ratio"]),
        min_pad_px=float(raw["min_pad_px"]),
    )


def _keypoint_smoothing_from_raw(
    raw: dict,
    section: str,
) -> KeypointSmoothingConfig:
    _require_exact_keys(
        raw,
        {
            "enabled",
            "method",
            "static_window",
            "static_motion_px",
            "deadband_px",
            "low_confidence",
            "low_confidence_alpha",
            "min_cutoff",
            "beta",
            "d_cutoff",
            "static_alpha",
            "moving_alpha",
            "jitter_alpha",
            "max_static_jump_px",
            "skeleton_anchor_alpha",
            "skeleton_endpoint_alpha",
            "skeleton_moving_endpoint_alpha",
            "skeleton_fast_endpoint_alpha",
            "skeleton_fast_moving_endpoint_alpha",
            "skeleton_fast_static_step_px",
            "skeleton_max_static_step_px",
            "skeleton_limb_tolerance",
            "sensitive_keypoints",
        },
        section,
    )
    method = str(raw["method"])
    if method not in {"ema", "one_euro", "skeleton"}:
        raise ValueError(
            "keypoint_smoothing.method must be 'ema', 'one_euro', "
            f"or 'skeleton': {method}")
    return KeypointSmoothingConfig(
        enabled=bool(raw["enabled"]),
        method=method,
        static_window=int(raw["static_window"]),
        static_motion_px=float(raw["static_motion_px"]),
        deadband_px=float(raw["deadband_px"]),
        low_confidence=float(raw["low_confidence"]),
        low_confidence_alpha=float(raw["low_confidence_alpha"]),
        min_cutoff=float(raw["min_cutoff"]),
        beta=float(raw["beta"]),
        d_cutoff=float(raw["d_cutoff"]),
        static_alpha=float(raw["static_alpha"]),
        moving_alpha=float(raw["moving_alpha"]),
        jitter_alpha=float(raw["jitter_alpha"]),
        max_static_jump_px=float(raw["max_static_jump_px"]),
        skeleton_anchor_alpha=float(raw["skeleton_anchor_alpha"]),
        skeleton_endpoint_alpha=float(raw["skeleton_endpoint_alpha"]),
        skeleton_moving_endpoint_alpha=float(
            raw["skeleton_moving_endpoint_alpha"]),
        skeleton_fast_endpoint_alpha=float(raw["skeleton_fast_endpoint_alpha"]),
        skeleton_fast_moving_endpoint_alpha=float(
            raw["skeleton_fast_moving_endpoint_alpha"]),
        skeleton_fast_static_step_px=float(raw["skeleton_fast_static_step_px"]),
        skeleton_max_static_step_px=float(raw["skeleton_max_static_step_px"]),
        skeleton_limb_tolerance=float(raw["skeleton_limb_tolerance"]),
        sensitive_keypoints=_string_set(
            raw["sensitive_keypoints"],
            "keypoint_smoothing.sensitive_keypoints",
        ),
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


def _evaluate_box_filter(
    confidence: float,
    bbox: tuple[float, float, float, float],
    filter_config: BoxFilterConfig,
) -> tuple[bool, bool]:
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    failed_confidence = confidence < filter_config.min_confidence
    failed_size = (width < filter_config.min_width_px
                   or height < filter_config.min_height_px)
    return failed_confidence, failed_size

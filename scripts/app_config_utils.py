from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any


POSE_MODEL_PATHS: dict[str, dict[str, str]] = {
    "rtmpose-m": {
        "preset": "rtmpose-m",
        "config_path":
        "models/mmpose/configs/wholebody_2d_keypoint/rtmpose/coco-wholebody/"
        "rtmpose-m_8xb64-270e_coco-wholebody-256x192.py",
        "checkpoint_path":
        "models/mmpose/rtmpose-wholebody/"
        "rtmpose-m_simcc-coco-wholebody_pt-aic-coco_270e-256x192-cd5e845c_20230123.pth",
    },
    "hrnet-w32": {
        "preset": "hrnet-w32",
        "config_path":
        "models/mmpose/configs/wholebody_2d_keypoint/topdown_heatmap/"
        "coco-wholebody/td-hm_hrnet-w32_8xb64-210e_coco-wholebody-256x192.py",
        "checkpoint_path":
        "models/mmpose/hrnet/"
        "hrnet_w32_coco_wholebody_256x192-853765cd_20200918.pth",
    },
    "hrnet-w48-dark": {
        "preset": "hrnet-w48-dark",
        "config_path":
        "models/mmpose/configs/wholebody_2d_keypoint/topdown_heatmap/"
        "coco-wholebody/td-hm_hrnet-w48_dark-8xb32-210e_coco-wholebody-384x288.py",
        "checkpoint_path":
        "models/mmpose/hrnet/"
        "hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth",
    },
}


def load_yaml(path: str | Path) -> dict[str, Any]:
    yaml = _load_yaml()
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML file must be a mapping: {path}")
    return data


def write_yaml(data: dict[str, Any], path: str | Path) -> None:
    yaml = _load_yaml()
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def deep_update(target: dict[str, Any], updates: dict[str, Any]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            deep_update(target[key], value)
        else:
            target[key] = value


def merge_yaml(
    base_path: str | Path,
    output_path: str | Path,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    merged = deepcopy(load_yaml(base_path))
    if overrides:
        deep_update(merged, overrides)
    write_yaml(merged, output_path)
    return merged


def pose_model_config(model_name: str) -> dict[str, str]:
    if model_name not in POSE_MODEL_PATHS:
        raise ValueError(f"Unsupported pose model: {model_name}")
    return dict(POSE_MODEL_PATHS[model_name])


def _load_yaml():
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("PyYAML is required.") from exc
    return yaml

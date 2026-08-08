from __future__ import annotations

from pathlib import Path
import sys
import types
from typing import Sequence

import numpy as np

from football_vision import Detection, VideoFrame
from observations import (
    PROJECT_26_KEYPOINTS,
    WHOLEBODY_TO_PROJECT_26,
    Keypoint2D,
    PersonObservation2D,
    empty_project_26,
)


class MMPoseTopDownEstimator:

    def __init__(
        self,
        config_path: str | Path,
        checkpoint_path: str | Path | None = None,
        device: str = "cuda:0",
    ) -> None:
        _patch_mmcv_lite_ops_for_rtmpose()
        try:
            from mmpose.apis import init_model, inference_topdown
        except ImportError as exc:
            raise RuntimeError(
                "mmpose is required for pose estimation. Install MMPose and pass --pose-config/--pose-checkpoint."
            ) from exc
        from mmengine import Config

        self._inference_topdown = inference_topdown
        config = Config.fromfile(str(config_path))
        _prefer_mmpose_cspnext(config)
        self.model = init_model(
            config,
            str(checkpoint_path) if checkpoint_path else None,
            device=device,
        )

    def estimate(
        self,
        frame: VideoFrame,
        person_detections: Sequence[Detection],
    ) -> list[PersonObservation2D]:
        if not person_detections:
            return []

        bboxes = np.array([detection.bbox for detection in person_detections],
                          dtype=np.float32)
        pose_results = self._inference_topdown(self.model, frame.image, bboxes)

        observations: list[PersonObservation2D] = []
        for detection, pose_result in zip(person_detections, pose_results):
            keypoints, scores = _extract_pose_arrays(pose_result)
            observations.append(
                PersonObservation2D(
                    track_id=detection.track_id,
                    bbox=detection.bbox,
                    confidence=detection.confidence,
                    keypoints=project_wholebody_to_26(keypoints, scores),
                ))
        return observations


def project_wholebody_to_26(keypoints: np.ndarray,
                            scores: np.ndarray) -> list[Keypoint2D]:
    projected = empty_project_26()
    name_to_position = {
        keypoint.name: index
        for index, keypoint in enumerate(projected)
    }

    for name, source_index in WHOLEBODY_TO_PROJECT_26.items():
        if source_index >= len(keypoints):
            continue
        x, y = keypoints[source_index]
        score = float(scores[source_index]) if source_index < len(scores) else 0.0
        projected[name_to_position[name]] = Keypoint2D(
            name=name,
            x=float(x),
            y=float(y),
            confidence=score,
        )

    _set_virtual_midpoint(projected, "neck", "left_shoulder", "right_shoulder")
    _set_virtual_midpoint(projected, "pelvis", "left_hip", "right_hip")
    _set_virtual_midpoint(projected, "thorax", "neck", "pelvis")
    return projected


def _extract_pose_arrays(pose_result) -> tuple[np.ndarray, np.ndarray]:
    pred_instances = pose_result.pred_instances
    keypoints = np.asarray(pred_instances.keypoints)
    scores = np.asarray(pred_instances.keypoint_scores)
    if keypoints.ndim == 3:
        keypoints = keypoints[0]
    if scores.ndim == 2:
        scores = scores[0]
    return keypoints, scores


def _set_virtual_midpoint(projected: list[Keypoint2D], target: str, left: str,
                          right: str) -> None:
    positions = {keypoint.name: index for index, keypoint in enumerate(projected)}
    left_point = projected[positions[left]]
    right_point = projected[positions[right]]
    target_index = positions[target]

    if left_point.x is None or left_point.y is None:
        return
    if right_point.x is None or right_point.y is None:
        return

    confidence = min(left_point.confidence, right_point.confidence)
    projected[target_index] = Keypoint2D(
        name=target,
        x=(left_point.x + right_point.x) / 2.0,
        y=(left_point.y + right_point.y) / 2.0,
        confidence=confidence,
        state="virtual",
    )


def person_detections(detections: Sequence[Detection]) -> list[Detection]:
    return [detection for detection in detections if detection.label == "person"]


def project_26_names() -> tuple[str, ...]:
    return PROJECT_26_KEYPOINTS


def _prefer_mmpose_cspnext(config) -> None:
    backbone = config.get("model", {}).get("backbone", {})
    if backbone.get("type") == "CSPNeXt" and backbone.get("_scope_") == "mmdet":
        backbone.pop("_scope_")


def _patch_mmcv_lite_ops_for_rtmpose() -> None:
    try:
        from mmcv.ops import MultiScaleDeformableAttention  # noqa: F401
        return
    except Exception:
        pass

    class MissingMMCVOp:

        def __init__(self, *_, **__) -> None:
            raise RuntimeError(
                "MultiScaleDeformableAttention requires full mmcv. "
                "RTMPose does not use this op; install full mmcv if you switch to transformer-based heads."
            )

    ops_module = types.ModuleType("mmcv.ops")
    ops_module.__path__ = []
    ops_module.MultiScaleDeformableAttention = MissingMMCVOp
    deform_attn_module = types.ModuleType("mmcv.ops.multi_scale_deform_attn")
    deform_attn_module.MultiScaleDeformableAttention = MissingMMCVOp
    sys.modules["mmcv.ops"] = ops_module
    sys.modules["mmcv.ops.multi_scale_deform_attn"] = deform_attn_module

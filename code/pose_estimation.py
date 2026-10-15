from __future__ import annotations

import os
from pathlib import Path
import sys
import types
from typing import Sequence

import numpy as np
from loguru import logger

from football_vision import Detection, VideoFrame
from observations import (
    PROJECT_26_KEYPOINTS,
    WHOLEBODY_TO_PROJECT_26,
    Keypoint2D,
    PersonObservation2D,
    empty_project_26,
)


class MMPoseTopDownEstimator:
    """Top-down MMPose estimator.

    Two execution paths are available for ``estimate()``:

    * ``mmpose`` (fallback): ``mmpose.apis.inference_topdown``. Correct but
      rebuilds the whole test pipeline and materialises ``PoseDataSample``
      objects on every call, which costs several milliseconds of pure CPU
      time per frame.
    * ``fast`` (default): reuses MMPose's own ``GetBBoxCenterScale`` /
      ``TopdownAffine`` / decoder objects for numerically identical results,
      but skips the dataset pipeline and ``PoseDataSample`` bookkeeping.
      Only the crop batch, one backbone/head forward and one decode remain.
    """

    def __init__(
        self,
        config_path: str | Path,
        checkpoint_path: str | Path | None = None,
        device: str = "cuda:0",
        fast_path: bool = True,
        flip_test: bool | None = None,
    ) -> None:
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"MMPose config not found: {config_path}")
        if checkpoint_path is not None:
            checkpoint_path = Path(checkpoint_path)
            if not checkpoint_path.exists():
                raise FileNotFoundError(
                    f"MMPose checkpoint not found: {checkpoint_path}")

        _patch_mmcv_lite_ops_for_rtmpose()
        try:
            from mmpose.apis import init_model, inference_topdown
        except ImportError as exc:
            raise RuntimeError(
                "mmpose is required for pose estimation. Install MMPose and configure the pose model in config/app.yaml."
            ) from exc
        from mmengine import Config

        self._inference_topdown = inference_topdown
        self.device = device
        config = Config.fromfile(str(config_path))
        _prefer_mmpose_cspnext(config)
        self.model = _init_mmpose_model(
            init_model,
            config,
            checkpoint_path,
            device,
        )
        self.config = config
        self._fast: _FastTopdownPath | None = None
        if fast_path:
            try:
                self._fast = _FastTopdownPath.build(self.model, config, device,
                                                    flip_test=flip_test)
            except Exception as exc:  # pragma: no cover - defensive fallback
                logger.warning(
                    "fast top-down path unavailable, falling back to "
                    "inference_topdown: {}", exc)
                self._fast = None

    @property
    def uses_fast_path(self) -> bool:
        return self._fast is not None

    def estimate(
        self,
        frame: VideoFrame,
        person_detections: Sequence[Detection],
        output_detections: Sequence[Detection] | None = None,
    ) -> list[PersonObservation2D]:
        if not person_detections:
            return []
        if output_detections is None:
            output_detections = person_detections
        if len(output_detections) != len(person_detections):
            raise ValueError(
                "output_detections must have the same length as person_detections"
            )

        if self._fast is not None:
            keypoints, scores = self._fast.infer(
                frame.image,
                [detection.bbox for detection in person_detections],
            )
        else:
            bboxes = np.array(
                [detection.bbox for detection in person_detections],
                dtype=np.float32)
            pose_results = self._inference_topdown(self.model, frame.image,
                                                   bboxes)
            keypoints, scores = [], []
            for pose_result in pose_results:
                kp, sc = _extract_pose_arrays(pose_result)
                keypoints.append(kp)
                scores.append(sc)

        observations: list[PersonObservation2D] = []
        for detection, output_detection, kp, sc in zip(
                person_detections, output_detections, keypoints, scores):
            observations.append(
                PersonObservation2D(
                    track_id=detection.track_id,
                    bbox=output_detection.bbox,
                    confidence=detection.confidence,
                    keypoints=project_wholebody_to_26(kp, sc),
                ))
        return observations


class _FastTopdownPath:
    """Crop + batched forward + single decode, bypassing MMPose's dataset pipeline.

    Reuses MMPose's own ``GetBBoxCenterScale`` / ``TopdownAffine`` / decoder so
    the result stays numerically identical to ``inference_topdown``, including
    the optional horizontal-flip test-time augmentation.
    """

    def __init__(
        self,
        model,
        device: str,
        bbox_to_center_scale,
        affine,
        decoder,
        input_size: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        bgr_to_rgb: bool,
        flip_test: bool,
        flip_indices,
        head_kind: str = "simcc",
        flip_mode: str = "heatmap",
        shift_heatmap: bool = False,
    ) -> None:
        import torch

        self.model = model
        self.device = device
        self.bbox_to_center_scale = bbox_to_center_scale
        self.affine = affine
        self.decoder = decoder
        self.input_size = input_size
        self.mean = mean
        self.std = std
        self.bgr_to_rgb = bgr_to_rgb
        self.flip_test = flip_test
        self.flip_indices = flip_indices
        # simcc: RTMPose 系列，解码器接收 (simcc_x, simcc_y)
        # heatmap: HRNet 系列，解码器接收单张热力图
        self.head_kind = head_kind
        self.flip_mode = flip_mode
        self.shift_heatmap = shift_heatmap
        # 归一化放到 GPU 上做，避免在 CPU 上反复生成 float32 大数组。
        self._mean_t = torch.as_tensor(mean, dtype=torch.float32,
                                       device=device).view(1, 3, 1, 1)
        self._std_t = torch.as_tensor(std, dtype=torch.float32,
                                      device=device).view(1, 3, 1, 1)

    @classmethod
    def build(
        cls,
        model,
        config,
        device: str,
        flip_test: bool | None = None,
    ) -> "_FastTopdownPath":
        import inspect

        from mmpose.datasets.transforms import GetBBoxCenterScale, TopdownAffine

        decoder = getattr(model.head, "decoder", None)
        if decoder is None or not hasattr(decoder, "decode"):
            raise RuntimeError("pose head has no decoder with decode()")

        # SimCC 解码器接收 (simcc_x, simcc_y)，heatmap 解码器接收单张热力图。
        decode_params = len(inspect.signature(decoder.decode).parameters)
        head_kind = "simcc" if decode_params >= 2 else "heatmap"

        head_cfg = config["model"]["head"]
        # RTMPose 把 input_size 写在 head 里；HRNet 只写在顶层 codec 里。
        input_size = head_cfg.get("input_size")
        if input_size is None:
            input_size = config.get("codec", {}).get("input_size")
        if input_size is None:
            input_size = config["model"].get("codec", {}).get("input_size")
        if input_size is None:
            raise RuntimeError("cannot resolve pose input_size from config")
        input_size = tuple(input_size)
        if len(input_size) != 2:
            raise RuntimeError(f"unexpected pose input_size: {input_size}")

        pre_cfg = config["model"]["data_preprocessor"]
        cls._validate_preprocessor(pre_cfg)

        test_cfg = config["model"].get("test_cfg", {}) or {}
        if flip_test is None:
            flip_test = bool(test_cfg.get("flip_test", False))
        flip_indices = None
        if flip_test:
            flip_indices = (model.dataset_meta or {}).get("flip_indices")
            if flip_indices is None:
                raise RuntimeError("flip_test requires dataset_meta.flip_indices")

        import torch
        if not torch.cuda.is_available() and device.startswith("cuda"):
            raise RuntimeError("CUDA unavailable for the fast pose path")

        return cls(
            model=model,
            device=device,
            bbox_to_center_scale=GetBBoxCenterScale(padding=1.25),
            affine=TopdownAffine(input_size=input_size),
            decoder=decoder,
            input_size=np.array(input_size, dtype=np.float32),
            mean=np.array(pre_cfg["mean"], dtype=np.float32),
            std=np.array(pre_cfg["std"], dtype=np.float32),
            bgr_to_rgb=bool(pre_cfg.get("bgr_to_rgb", False)),
            flip_test=flip_test,
            flip_indices=flip_indices,
            head_kind=head_kind,
            flip_mode=str(test_cfg.get("flip_mode", "heatmap")),
            shift_heatmap=bool(test_cfg.get("shift_heatmap", False)),
        )

    @staticmethod
    def _validate_preprocessor(pre_cfg) -> None:
        # 这里只复现 PoseDataPreprocessor 的常见配置；出现额外变换时退回慢路径。
        required = {"mean", "std"}
        if not required.issubset(set(pre_cfg)):
            raise RuntimeError("pose data_preprocessor lacks mean/std")
        if bool(pre_cfg.get("rgb_to_bgr", False)):
            raise RuntimeError("rgb_to_bgr preprocessor is not supported")
        if pre_cfg.get("batch_augments"):
            raise RuntimeError("batch_augments are not supported")

    def infer(
        self,
        image: np.ndarray,
        bboxes: Sequence[tuple[float, float, float, float]],
    ) -> tuple[np.ndarray, np.ndarray]:
        import torch

        boxes = np.asarray(bboxes, dtype=np.float32)
        if boxes.ndim == 1:
            boxes = boxes[None]
        count = boxes.shape[0]
        width, height = int(self.input_size[0]), int(self.input_size[1])
        image_shape = image.shape[:2]

        crops = np.empty((count, height, width, 3), dtype=np.uint8)
        centers = np.empty((count, 2), dtype=np.float32)
        scales = np.empty((count, 2), dtype=np.float32)
        for index in range(count):
            sample = {"bbox": boxes[index:index + 1]}
            sample = self.bbox_to_center_scale.transform(sample)
            sample["img"] = image
            sample["img_shape"] = image_shape
            sample = self.affine.transform(sample)
            crops[index] = sample["img"]
            centers[index] = sample["input_center"]
            scales[index] = sample["input_scale"]

        # 只传 uint8 的裁剪结果（约 1/4 带宽），颜色通道与归一化都在 GPU 上完成。
        tensor = torch.from_numpy(crops).to(self.device,
                                            non_blocking=True).permute(
                                                0, 3, 1, 2)
        if self.bgr_to_rgb:
            tensor = tensor[:, [2, 1, 0]]
        tensor = (tensor.float() - self._mean_t) / self._std_t
        tensor = tensor.contiguous()

        with torch.inference_mode():
            features = self.model.backbone(tensor)
            features_flip = (self.model.backbone(tensor.flip(-1))
                             if self.flip_test else None)
            if self.head_kind == "simcc":
                from mmpose.models.utils.tta import flip_vectors

                simcc_x, simcc_y = self.model.head(features)
                if self.flip_test:
                    flip_x, flip_y = self.model.head(features_flip)
                    flip_x, flip_y = flip_vectors(flip_x,
                                                  flip_y,
                                                  flip_indices=self.flip_indices)
                    simcc_x = (simcc_x + flip_x) * 0.5
                    simcc_y = (simcc_y + flip_y) * 0.5
                keypoints, scores = self.decoder.decode(
                    simcc_x.cpu().numpy(), simcc_y.cpu().numpy())
            else:
                from mmpose.models.utils.tta import flip_heatmaps

                heatmaps = self.model.head(features)
                if self.flip_test:
                    flipped = flip_heatmaps(
                        self.model.head(features_flip),
                        flip_mode=self.flip_mode,
                        flip_indices=self.flip_indices,
                        shift_heatmap=self.shift_heatmap)
                    heatmaps = (heatmaps + flipped) * 0.5
                batch = heatmaps.cpu().numpy()
                kp_list, score_list = [], []
                for index in range(batch.shape[0]):
                    kp, sc = self.decoder.decode(batch[index])
                    kp, sc = np.asarray(kp), np.asarray(sc)
                    kp_list.append(kp[0] if kp.ndim == 3 else kp)
                    score_list.append(sc[0] if sc.ndim == 2 else sc)
                keypoints = np.stack(kp_list)
                scores = np.stack(score_list)
        # 与 TopdownPoseEstimator.add_pred_to_datasample 的反变换保持一致。
        keypoints = (keypoints / self.input_size * scales[:, None, :] +
                     centers[:, None, :] - 0.5 * scales[:, None, :])
        return keypoints, scores


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


def _init_mmpose_model(init_model, config, checkpoint_path: Path | None,
                       device: str):
    """Load legacy local checkpoints under PyTorch's weights-only default.

    MMPose 1.x delegates checkpoint loading to ``torch.load`` without passing
    ``weights_only``. Newer PyTorch versions then reject older trusted MMPose
    checkpoints. Limit the compatibility override to this model-load call.
    """
    if checkpoint_path is None:
        return init_model(config, None, device=device)

    env_name = "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"
    previous = os.environ.get(env_name)
    os.environ[env_name] = "1"
    try:
        return init_model(config, str(checkpoint_path), device=device)
    finally:
        if previous is None:
            os.environ.pop(env_name, None)
        else:
            os.environ[env_name] = previous


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

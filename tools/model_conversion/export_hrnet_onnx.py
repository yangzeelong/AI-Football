#!/usr/bin/env python3
"""Export an MMPose HRNet top-down model to the C++ heatmap contract."""

from __future__ import annotations

import argparse
import sys
import types
from typing import Any
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="MMPose model config")
    parser.add_argument("--checkpoint", required=True, help="MMPose checkpoint")
    parser.add_argument("--output", required=True, help="Output ONNX path")
    parser.add_argument("--input-name", default="images")
    parser.add_argument("--output-name", default="heatmaps")
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=288)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dynamic-batch", action="store_true")
    return parser.parse_args()


def load_model(config_path: Path, checkpoint_path: Path, device: str):
    # MMPose imports its complete dataset/evaluation registry from the public
    # API.  xtcocotools has no usable Python 3.12 wheel, while pycocotools
    # provides the same modules needed during model construction.
    try:
        import pycocotools.coco as coco
        import pycocotools.cocoeval as cocoeval
        import pycocotools.mask as mask

        compat = types.ModuleType("xtcocotools")
        compat.coco = coco
        compat.cocoeval = cocoeval
        compat.mask = mask
        sys.modules.setdefault("xtcocotools", compat)
        sys.modules.setdefault("xtcocotools.coco", coco)
        sys.modules.setdefault("xtcocotools.cocoeval", cocoeval)
        sys.modules.setdefault("xtcocotools.mask", mask)
    except ImportError:
        pass

    # MMPose 1.3 registers optional RTMO components during import.  Those
    # components only need two utilities from MMDetection; HRNet does not use
    # either one during construction or export.
    if "mmdet.utils" not in sys.modules:
        mmdet = types.ModuleType("mmdet")
        mmdet_utils = types.ModuleType("mmdet.utils")
        mmdet_utils.ConfigType = Any
        mmdet_utils.reduce_mean = lambda value: value.mean()
        mmdet.utils = mmdet_utils
        sys.modules["mmdet"] = mmdet
        sys.modules["mmdet.utils"] = mmdet_utils

    # mmcv-lite intentionally omits compiled operators.  MMPose imports the
    # unused EDPose module while registering all heads; a placeholder is
    # sufficient because the HRNet config never constructs this operator.
    if "mmcv.ops" not in sys.modules:
        import torch.nn as nn

        mmcv_ops = types.ModuleType("mmcv.ops")

        class MultiScaleDeformableAttention(nn.Module):
            def forward(self, query, *args, **kwargs):
                return query

        mmcv_ops.MultiScaleDeformableAttention = MultiScaleDeformableAttention
        mmcv_ops.DeformConv2d = MultiScaleDeformableAttention
        mmcv_ops.ModulatedDeformConv2d = MultiScaleDeformableAttention
        sys.modules["mmcv.ops"] = mmcv_ops

    try:
        from mmengine import Config
        from mmpose.apis import init_model
    except ImportError as exc:
        raise RuntimeError(
            "MMPose export requires torch, mmengine and mmpose. "
            "Install the AI-Football requirements first."
        ) from exc

    config = Config.fromfile(str(config_path))
    model = init_model(config, str(checkpoint_path), device=device)
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output)
    if not config_path.is_file():
        raise FileNotFoundError(f"MMPose config not found: {config_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"MMPose checkpoint not found: {checkpoint_path}")

    import torch
    import torch.nn as nn

    model = load_model(config_path, checkpoint_path, args.device)

    class HeatmapWrapper(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, images):
            output = self.inner.forward(images, data_samples=None, mode="tensor")
            if isinstance(output, dict):
                for key in ("heatmaps", "output", "pred_heatmaps"):
                    if key in output:
                        output = output[key]
                        break
            if isinstance(output, (tuple, list)):
                if len(output) != 1:
                    raise RuntimeError("HRNet export returned multiple tensors")
                output = output[0]
            if not hasattr(output, "shape") or len(output.shape) != 4:
                raise RuntimeError(
                    f"Expected rank-4 heatmaps, got {getattr(output, 'shape', None)}"
                )
            return output

    wrapper = HeatmapWrapper(model).eval()
    sample = torch.zeros(
        1, 3, args.height, args.width, dtype=torch.float32, device=args.device)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dynamic_axes = None
    if args.dynamic_batch:
        dynamic_axes = {
            args.input_name: {0: "batch"},
            args.output_name: {0: "batch"},
        }

    with torch.no_grad():
        sample_output = wrapper(sample)
    if tuple(sample_output.shape[2:]) != (args.height // 4, args.width // 4):
        print(
            "warning: heatmap shape is not input/4: "
            f"{tuple(sample_output.shape)}",
            file=sys.stderr,
        )

    torch.onnx.export(
        wrapper,
        sample,
        str(output_path),
        input_names=[args.input_name],
        output_names=[args.output_name],
        dynamic_axes=dynamic_axes,
        opset_version=args.opset,
        do_constant_folding=True,
        training=torch.onnx.TrainingMode.EVAL,
    )
    print(f"exported: {output_path}")
    print(f"input: {args.input_name} {tuple(sample.shape)} float32 NCHW")
    print(f"output: {args.output_name} {tuple(sample_output.shape)} float32")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)

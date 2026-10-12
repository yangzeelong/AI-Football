#!/usr/bin/env python3
"""校验 TensorRT engine 与 PyTorch 的数值一致性。

方法：把**同一张预处理后的张量**分别喂给 PyTorch 与 TRT engine，比较原始输出；
姿态再经过解码，比较关键点在裁剪空间的位移。

判定要点：
  - RF-DETR 原始 query 的 max|Δlogits| 会有 4~5 的量级，但 **fp32 engine 也有同样量级**，
    说明偏差落在未被筛选采用的 query 上，不是 fp16 引入的。真正的判据是
    "分数 > threshold 的 query 类别是否一致" + 类别直方图。
  - HRNet 必须用**真实球员裁剪**。随机噪声输入下热力图接近平坦，argmax 极不稳定，
    会把噪声当成精度损失。

需要 TensorRT 的 Python 绑定（本机在系统 Python 3.12 的 dist-packages 里，
conda 环境 ai-football 没有），所以这个脚本用系统解释器跑：

    /usr/bin/python3.12 tools/model_conversion/validate_engines.py

用法：
    --engine <path> --kind rfdetr --input-size 960
    --engine <path> --kind hrnet --height 384 --width 288
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

try:
    import tensorrt as trt
except ImportError:  # pragma: no cover
    print("TensorRT Python bindings not found. Use /usr/bin/python3.12 on this host.",
          file=sys.stderr)
    raise

ROOT = Path(__file__).resolve().parents[2]              # …/AI-Football
#: Nexusflow 仓库：默认取兄弟目录 <WORKSPACE>/Nexusflow，可用 AI_FOOTBALL_NEXUSFLOW 覆盖
NEXUS_ROOT = Path(os.environ.get("AI_FOOTBALL_NEXUSFLOW")
                  or ROOT.parent / "Nexusflow").expanduser().resolve()
#: 默认校验视频（Nexusflow 仓库里的示例素材）；换机器用 --video 指定
VIDEO = str(NEXUS_ROOT / "data" / "射门1-1080p60.mov")
RF_DETR_WEIGHTS = "models/rfdetr/rf-detr-small.pth"
HRNET_W48_CONFIG = ("models/mmpose/configs/wholebody_2d_keypoint/topdown_heatmap/"
                    "coco-wholebody/td-hm_hrnet-w48_dark-8xb32-210e_coco-wholebody-384x288.py")
HRNET_W48_CKPT = ("models/mmpose/hrnet/"
                  "hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth")
OBS_WITH_PERSONS = "tmp/full_run/rfdetr960_hrnetw32_256_nof_h/observations.jsonl"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
POSE_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
POSE_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)


def load_engine(path: Path):
    with open(path, "rb") as handle:
        engine = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(handle.read())
    return engine, engine.create_execution_context()


def run_engine(engine, ctx, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """按 binding 名绑定地址执行一次。传入张量必须是 contiguous。"""
    names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    for name in names:
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            ctx.set_input_shape(name, tuple(inputs[name].shape))
    buffers = dict(inputs)
    for name in names:
        if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
            buffers[name] = torch.empty(tuple(ctx.get_tensor_shape(name)),
                                        dtype=torch.float32, device="cuda")
        ctx.set_tensor_address(name, buffers[name].data_ptr())
    if not ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream):
        raise RuntimeError("execute_async_v3 failed")
    torch.cuda.synchronize()
    return {k: v for k, v in buffers.items() if k not in inputs}


def letterbox_rgb(image: np.ndarray, target: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(target / width, target / height)
    resized = cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))),
                         interpolation=cv2.INTER_LINEAR)
    canvas = np.full((target, target, 3), 114, dtype=image.dtype)
    px = (target - resized.shape[1]) // 2
    py = (target - resized.shape[0]) // 2
    canvas[py:py + resized.shape[0], px:px + resized.shape[1]] = resized
    return canvas


def read_frames(frame_ids: list[int]) -> dict[int, np.ndarray]:
    cap = cv2.VideoCapture(VIDEO)
    frames = {}
    for fid in frame_ids:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ok, image = cap.read()
        if ok:
            frames[fid] = image
    cap.release()
    return frames


# --------------------------------------------------------------------------- #
# RF-DETR
# --------------------------------------------------------------------------- #

def validate_rfdetr(engine_path: Path, input_size: int, frame_ids: list[int],
                    threshold: float = 0.25) -> None:
    from rfdetr import RFDETRSmall

    model = RFDETRSmall(pretrain_weights=str(ROOT / RF_DETR_WEIGHTS),
                        resolution=input_size, device="cuda")
    model.optimize_for_inference(compile=False, dtype=torch.float16)
    net = model.model.inference_model
    engine, ctx = load_engine(engine_path)
    frames = read_frames(frame_ids)

    for fid, bgr in frames.items():
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        square = letterbox_rgb(rgb, input_size).astype(np.float32) / 255.0
        square = (square - IMAGENET_MEAN) / IMAGENET_STD
        # 必须 contiguous：TRT 通过 data_ptr 直接读内存，非连续张量会读错数据。
        tensor = torch.from_numpy(
            np.ascontiguousarray(square.transpose(2, 0, 1))).unsqueeze(0).contiguous().cuda()

        with torch.inference_mode():
            pt = net(tensor.half())
        pt_logits = pt[1].float()
        out = run_engine(engine, ctx, {"image": tensor})
        tr_logits = out["pred_logits"]

        pt_score, pt_cls = torch.sigmoid(pt_logits).max(-1)
        tr_score, tr_cls = torch.sigmoid(tr_logits).max(-1)
        keep = pt_score[0] > threshold
        agree = (float((pt_cls[0][keep] == tr_cls[0][keep]).float().mean())
                 if keep.any() else 1.0)

        def histogram(classes):
            counts: dict[int, int] = {}
            for c in classes:
                counts[int(c)] = counts.get(int(c), 0) + 1
            return dict(sorted(counts.items()))

        print(f"  frame {fid:4d}  Δboxes={float((pt[0].float() - out['pred_boxes']).abs().max()):.4f}  "
              f"Δlogits={float((pt_logits - tr_logits).abs().max()):.4f}  "
              f"kept={int(keep.sum()):3d}  cls一致率={agree:.4f}")
        print(f"              pytorch histogram={histogram(pt_cls[0][keep])}")
        print(f"              trt     histogram={histogram(tr_cls[0][keep])}")


# --------------------------------------------------------------------------- #
# HRNet
# --------------------------------------------------------------------------- #

def validate_hrnet(engine_path: Path, height: int, width: int,
                   frame_ids: list[int], conf_threshold: float = 0.3) -> None:
    sys.path.insert(0, str(NEXUS_ROOT / "tools" / "model_conversion"))
    from export_hrnet_onnx import load_model
    from mmpose.datasets.transforms import GetBBoxCenterScale, TopdownAffine

    model = load_model(ROOT / HRNET_W48_CONFIG, ROOT / HRNET_W48_CKPT, "cuda:0").half().eval()
    decoder = model.head.decoder
    engine, ctx = load_engine(engine_path)

    # 用真实球员框（来自既有观测），避免噪声输入导致 argmax 不稳。
    boxes_by_frame: dict[int, list] = {}
    obs = ROOT / OBS_WITH_PERSONS
    if obs.exists():
        with obs.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if record.get("type") == "frame" and record["frame_index"] in frame_ids \
                        and record["persons"]:
                    boxes_by_frame[record["frame_index"]] = [p["bbox"] for p in record["persons"]]
    if not boxes_by_frame:
        print(f"  no person boxes found in {OBS_WITH_PERSONS}; cannot run real-crop validation",
              file=sys.stderr)
        return

    b2cs = GetBBoxCenterScale(padding=1.25)
    affine = TopdownAffine(input_size=(width, height))
    frames = read_frames(sorted(boxes_by_frame))
    pairs: list[tuple[float, float, int]] = []

    for fid, bboxes in sorted(boxes_by_frame.items()):
        if fid not in frames:
            continue
        image = frames[fid]
        crops = []
        for bbox in bboxes:
            sample = b2cs.transform({"bbox": np.array([bbox], dtype=np.float32)})
            sample["img"] = image
            sample["img_shape"] = image.shape[:2]
            sample = affine.transform(sample)
            crops.append(sample["img"])
        batch = np.stack(crops).astype(np.float32)
        batch = (batch[..., ::-1] - POSE_MEAN) / POSE_STD
        tensor = torch.from_numpy(
            np.ascontiguousarray(batch).transpose(0, 3, 1, 2)).contiguous().cuda()

        with torch.inference_mode():
            pt_hm = model.head(model.backbone(tensor.half())).float().cpu().numpy()
        tr_hm = run_engine(engine, ctx, {"images": tensor})["heatmaps"].float().cpu().numpy()

        diff = np.abs(pt_hm - tr_hm)
        for index in range(pt_hm.shape[0]):
            kp_pt, sc_pt = decoder.decode(pt_hm[index])
            kp_tr, _ = decoder.decode(tr_hm[index])
            delta = np.linalg.norm(np.asarray(kp_pt).reshape(-1, 2) -
                                   np.asarray(kp_tr).reshape(-1, 2), axis=1)
            scores = np.asarray(sc_pt).reshape(-1)
            pairs.extend((float(delta[i]), float(scores[i]), i) for i in range(delta.size))
        print(f"  frame {fid:4d}  persons={len(bboxes):2d}  "
              f"max|Δheatmap|={float(diff.max()):.6f}  mean|Δ|={float(diff.mean()):.7f}")

    delta_all = np.array([d for d, _, _ in pairs])
    print(f"\n  解码后关键点位移（裁剪空间 {width}x{height}，共 {delta_all.size} 个）:")
    print(f"    mean={delta_all.mean():.4f}  p95={np.percentile(delta_all, 95):.4f}  "
          f"p99={np.percentile(delta_all, 99):.4f}  max={delta_all.max():.4f}  "
          f"逐位相同={float((delta_all < 1e-6).mean()):.4f}")

    used = np.array([d for d, _, i in pairs if i < 23])
    if used.size:
        print(f"    仅项目使用的 23 个关键点（{used.size} 个）: "
              f"mean={used.mean():.4f}  p95={np.percentile(used, 95):.4f}  "
              f"max={used.max():.4f}  逐位相同={float((used < 1e-6).mean()):.4f}")

    print("\n  按 PyTorch 侧置信度分层"
          "（用于区分真正丢精度 与 无信号通道上的 argmax 抖动）:")
    for thr in (0.0, 0.1, conf_threshold, 0.5):
        subset = np.array([d for d, s, _ in pairs if s >= thr])
        if subset.size == 0:
            continue
        print(f"    score>={thr:<4} n={subset.size:<5d} mean={subset.mean():7.4f}  "
              f"p95={np.percentile(subset, 95):7.4f}  max={subset.max():8.4f}  "
              f"逐位相同={float((subset < 1e-6).mean()):.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--kind", choices=["rfdetr", "hrnet"], required=True)
    parser.add_argument("--input-size", type=int, default=960, help="RF-DETR 方形输入边长")
    parser.add_argument("--height", type=int, default=384, help="HRNet 输入高")
    parser.add_argument("--width", type=int, default=288, help="HRNet 输入宽")
    parser.add_argument("--frames", type=int, nargs="+", default=[0, 100, 200, 295])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    engine_path = ROOT / args.engine if not Path(args.engine).is_absolute() else Path(args.engine)
    if not engine_path.is_file():
        print(f"engine not found: {engine_path}", file=sys.stderr)
        return 1

    print(f"engine: {engine_path}")
    print(f"kind:   {args.kind}\n")
    if args.kind == "rfdetr":
        validate_rfdetr(engine_path, args.input_size, args.frames)
    else:
        validate_hrnet(engine_path, args.height, args.width, args.frames)
    return 0


if __name__ == "__main__":
    sys.exit(main())

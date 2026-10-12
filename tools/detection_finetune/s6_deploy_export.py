#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""导出微调权重 → ONNX(960, 动态 batch) → fp16(fp32 I/O) → TRT engine，并与 PyTorch 对拍。

为什么单独一个脚本：`Nexusflow/tools/model_conversion/export_rfdetr_onnx.py` 的产物名
**只由 --resolution 决定**，直接跑会覆盖现有 COCO ONNX/engine。这里强制：
* 导出到独立目录（默认 `AI-Football/models/rfdetr/fine-tune/v1/`），文件名带 `3c`；
* 复用既有 `convert_fp16_onnx.py` 的转换实现（keep_io_types=True，即 fp16 计算 + fp32 I/O）；
* 用 `trtexec` 按 min/opt/max = 1/4/8 建 engine（与既有引擎同一 profile）；
* **数值对拍**：同一张图分别喂 PyTorch 与 ONNXRuntime，比 pred_logits/pred_boxes 的最大绝对
  误差 + argmax 一致率（.pth 与网上 .onnx 不一致是部署事故的高频来源，必须落地成数字）。

只读数据；产物只写到 --out-dir（默认在模型目录，不在 data/ 里）。

    $PY s6_deploy_export.py --checkpoint runs/v1/checkpoint_best_ema.pth
    $PY s6_deploy_export.py --checkpoint ... --skip-engine   # 只出 ONNX + 对拍
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable

import base_layout

HERE = Path(__file__).parent
REPO = HERE.parent.parent.parent                        # .../Work
EXPORT_TOOL = REPO / "Nexusflow" / "tools" / "model_conversion" / "export_rfdetr_onnx.py"
FP16_TOOL = REPO / "AI-Football" / "tools" / "model_conversion" / "convert_fp16_onnx.py"
MODELS_ROOT = REPO / "AI-Football" / "models" / "rfdetr" / "fine-tune"
CLASS_IDS = {"person": 1, "ball": 2, "cone": 3}


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def run(cmd: list[str], **kw: Any) -> subprocess.CompletedProcess:
    print("  $ " + " ".join(cmd))
    return subprocess.run(cmd, text=True, check=False, **kw)


def export_onnx(py: str, checkpoint: Path, resolution: int, out_dir: Path,
                version: str = "v1") -> Path:
    """导出 fp32 ONNX（先落到临时目录，再搬进 out_dir，避免覆盖既有 COCO 产物）。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="rfdetr_3c_") as tmp:
        res = run([py, str(EXPORT_TOOL), "--size", "small", "--checkpoint", str(checkpoint),
                   "--output-dir", tmp, "--resolution", str(resolution), "--dynamic-batch",
                   "--device", "cpu"], capture_output=True)
        if res.returncode != 0:
            print(res.stdout[-2000:])
            print(res.stderr[-2000:])
            raise SystemExit("ONNX 导出失败（见上）")
        # 导出脚本会先写 inference_model.onnx（**未改名**：input/dets/labels），
        # 再写改名后的 rf-detr-<size>-<res>-dynamic.onnx（image/pred_boxes/pred_logits，
        # C++ 就是按这组名字取的）。早先直接 glob()[0] 会按字母序抓到前者 → 名字不符部署契约。
        candidates = sorted(Path(tmp).glob("rf-detr-*.onnx")) or sorted(Path(tmp).glob("*.onnx"))
        if not candidates:
            raise SystemExit("导出脚本没有产出 onnx")
        # 版本名必须从调用方传进来：写死 v1 会让非 v1 版本的 ONNX 名与
        # rollout.py 的 `rf-detr-small-<版本>-960-dynamic.onnx` 契约对不上。
        target = out_dir / ("rf-detr-small-%s-%d-dynamic.onnx" % (version, resolution))
        shutil.move(str(candidates[0]), str(target))
        # 落地即校验命名契约（问题早发现，别等 C++ 侧报"找不到输出")
        import onnx as _onnx
        graph = _onnx.load(str(target), load_external_data=False)
        got_in = [v.name for v in graph.graph.input]
        got_out = [v.name for v in graph.graph.output]
        if got_in != ["image"] or sorted(got_out) != ["pred_boxes", "pred_logits"]:
            raise SystemExit("输出命名不符合部署契约：in=%s out=%s（期望 image / pred_boxes+pred_logits）"
                             % (got_in, got_out))
    print("  → %s（%.1f MB）" % (target, target.stat().st_size / 1e6))
    return target


def to_fp16(src: Path) -> Path:
    """复用既有 convert_fp16_onnx.convert（fp16 计算 + fp32 I/O）。"""
    import importlib.util

    spec = importlib.util.spec_from_file_location("convert_fp16_onnx", FP16_TOOL)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    dst = src.with_name(src.stem + "-fp16-fp32io.onnx")
    if not module.convert(src, dst, keep_io_types=True):
        raise SystemExit("fp16 转换失败：%s" % src)
    print("  → %s（%.1f MB）" % (dst, dst.stat().st_size / 1e6))
    return dst


def build_engine(fp16: Path, resolution: int, out_dir: Path, force: bool, trtexec: str,
                 version: str = "v1", checkpoint: Path | None = None) -> Path:
    engine = out_dir / ("rf-detr-small-%s-%d-fp16-b8-fp32io.engine" % (version, resolution))
    if engine.exists() and not force:
        # 重训后 checkpoint 比 engine 新 → 必须重建。否则会 pin 到**旧权重**的 engine，
        # 而 meta_export.json 里记的是新 checkpoint 的 sha（静默错，且对拍查不出来——
        # 对拍比的是刚导出的 ONNX，不是 engine）。
        if checkpoint is not None and checkpoint.stat().st_mtime > engine.stat().st_mtime:
            print("  重建：checkpoint 比 engine 新（%s > %s），旧 engine 已过期"
                  % (time.ctime(checkpoint.stat().st_mtime),
                     time.ctime(engine.stat().st_mtime)))
        else:
            print("  SKIP: %s 已存在（--force 重建）" % engine.name)
            return engine
    if not Path(trtexec).exists():
        print("  SKIP: 找不到 trtexec（%s），跳过 engine（ONNX 已就绪）" % trtexec)
        return engine
    res = run([trtexec, "--onnx=%s" % fp16, "--saveEngine=%s" % engine, "--skipInference",
               "--builderOptimizationLevel=3",
               "--minShapes=image:1x3x%dx%d" % (resolution, resolution),
               "--optShapes=image:4x3x%dx%d" % (resolution, resolution),
               "--maxShapes=image:8x3x%dx%d" % (resolution, resolution)],
              capture_output=True)
    tail = "\n".join([ln for ln in res.stdout.splitlines()
                      if any(k in ln for k in ("Engine built in", "PASSED", "FAILED", "Error"))][-6:])
    print("  " + (tail or "trtexec 无关键输出"))
    if engine.exists():
        print("  → %s（%.1f MB）" % (engine, engine.stat().st_size / 1e6))
    else:
        print("  !! engine 未生成（trtexec 失败，ONNX 仍可用）")
    return engine


def sample_images(split: str, n: int) -> list[Path]:
    import json as _json

    data = _json.loads((base_layout.DATASET_DIR / split / "_annotations.coco.json").read_text("utf-8"))
    images = sorted(data["images"], key=lambda i: i["file_name"])[:n]
    return [base_layout.DATASET_DIR / split / i["file_name"] for i in images]


def load_model(checkpoint: Path, resolution: int):
    """加载一次模型（对拍多张图时别反复加载）。"""
    from rfdetr import RFDETRSmall

    return RFDETRSmall(pretrain_weights=str(checkpoint), resolution=resolution, device="cuda")


def preprocess(image: Path, resolution: int):
    """与 `RFDETRSmall.predict` 完全一致的预处理：to_tensor → ImageNet normalize → resize。"""
    import torchvision.transforms.functional as F
    from PIL import Image

    with Image.open(image) as fh:
        img = fh.convert("RGB")
        size = (img.size[1], img.size[0])                    # (H, W) 原图尺寸
        blob = F.to_tensor(img)
    blob = F.normalize(blob, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    blob = F.resize(blob, (resolution, resolution)).unsqueeze(0)
    return blob, size


def _as_dict(out: Any) -> dict[str, Any]:
    """postprocess 返回 list（按 batch）或 dict，统一成 batch=1 的 dict。"""
    item = out[0] if isinstance(out, list) else out
    if not isinstance(item, dict):
        raise SystemExit("postprocess 返回类型不认识：%s" % type(out))
    return item


def iou_xyxy(a: Any, b: Any) -> float:
    import numpy as np

    ax1, ay1, ax2, ay2 = (float(v) for v in a[:4])
    bx1, by1, bx2, by2 = (float(v) for v in b[:4])
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def onnx_dets(onnx_path: Path, blob: Any, size: tuple[int, int]) -> tuple[Any, Any, Any]:
    """ONNX 侧最终检测：sigmoid(pred_logits) → argmax（**列下标即 classId，不加 1**）
    → 归一化 cxcywh 还原到**原图像素**（与 `predict()` 同一坐标口径，已实测对齐）。

    classId 口径（实测）：列 0 = no-object（最高 sigmoid 0.002，永不触发），
    列 1 = person、列 2 = ball、列 3 = cone；与 `predict().class_id`（1/2/3）逐框一致，
    也与 C++ `RFDetrDetectorInfer.cpp` 用列下标当 classId 的写法一致。
    早先这里写成 `+1` → 对拍"标签一致率"恒为 0（假失败），已修。"""
    import numpy as np
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    names = [o.name for o in session.get_outputs()]
    outs = session.run(None, {"image": blob.numpy().astype(np.float32)})
    data = dict(zip(names, outs))
    boxes = np.asarray(data["pred_boxes"][0], dtype=np.float64)
    logits = np.asarray(data["pred_logits"][0], dtype=np.float64)
    prob = 1.0 / (1.0 + np.exp(-logits))
    scores = prob.max(axis=1)
    labels = prob.argmax(axis=1)                           # 列下标 = classId（不加 1）
    cx, cy, w, h = boxes.T
    hh, ww = size
    xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    return xyxy * np.array([ww, hh, ww, hh]), scores, labels


def validate(checkpoint: Path, onnx_path: Path, resolution: int, split: str,
             samples: int, thr: float = 0.05) -> dict[str, Any]:
    """逐图逐框比 PyTorch `predict()` vs ONNX 最终输出（同阈值、同坐标口径）。"""
    import numpy as np

    model = load_model(checkpoint, resolution)
    rows = []
    try:
        for image in sample_images(split, samples):
            blob, size = preprocess(image, resolution)
            from PIL import Image

            with Image.open(image) as fh:
                ref = model.predict(fh.convert("RGB"), threshold=thr)
            ref_boxes = np.asarray(ref.xyxy, dtype=np.float64)
            ref_scores = np.asarray(ref.confidence, dtype=np.float64)
            ref_labels = np.asarray(ref.class_id, dtype=np.int64)
            order = np.argsort(-ref_scores)
            ref_boxes, ref_scores, ref_labels = (ref_boxes[order], ref_scores[order],
                                                 ref_labels[order])
            boxes, scores, labels = onnx_dets(onnx_path, blob, size)
            keep = scores >= thr
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            onnx_order = np.argsort(-scores)
            boxes, scores, labels = boxes[onnx_order], scores[onnx_order], labels[onnx_order]

            # 两边都是"同一批候选"，但**排序/截断可能不同** → 用 IoU 贪心匹配逐框对照，
            # 不能按行比（实测行序不同，行对齐会得出 1700px 的假差异）
            used: set[int] = set()
            deltas: list[float] = []
            agree = 0
            for rb, rl in zip(ref_boxes, ref_labels):
                best_i, best_iou = -1, 0.0
                for i, ob in enumerate(boxes):
                    if i in used:
                        continue
                    v = iou_xyxy(rb, ob)
                    if v > best_iou:
                        best_i, best_iou = i, v
                if best_i >= 0 and best_iou >= 0.9:
                    used.add(best_i)
                    deltas.append(float(np.abs(rb - boxes[best_i]).max()))
                    agree += int(labels[best_i] == rl)
            matched = len(deltas)
            box_delta = max(deltas, default=0.0)
            label_agree = (agree / matched) if matched else 1.0
            rows.append({"image": image.name, "n_torch": int(len(ref_scores)),
                         "n_onnx": int(len(scores)), "matched": matched,
                         "recall_matched": round(matched / max(1, len(ref_scores)), 4),
                         "boxes_max_abs_diff_px": round(box_delta, 3),
                         "label_agreement": round(label_agree, 4),
                         "onnx_head_width": int(onnx_logits_shape(onnx_path)[-1] or 0)})
            print("    %-34s 框数 %d/%d  匹配 %d（%.0f%%）  最大框差 %.3fpx  标签一致 %.3f"
                  % (image.name[:34], len(ref_scores), len(scores), matched,
                     100.0 * matched / max(1, len(ref_scores)), box_delta, label_agree))
    finally:
        try:
            import torch
            del model
            torch.cuda.empty_cache()
        except Exception:                                    # noqa: BLE001
            pass
    width = rows[0]["onnx_head_width"] if rows else 0
    return {"samples": rows,
            "boxes_max_abs_diff_px": max((r["boxes_max_abs_diff_px"] for r in rows), default=0.0),
            "label_agreement_min": min((r["label_agreement"] for r in rows), default=1.0),
            "recall_matched_min": min((r["recall_matched"] for r in rows), default=1.0),
            "classes_in_head": width, "channel_contract_ok": width == len(CLASS_IDS) + 1}


def onnx_logits_shape(onnx_path: Path) -> list[int]:
    """用 onnxruntime 读输出形状（动态 batch 时首维是 None，末维是类别数）。"""
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    for out in session.get_outputs():
        if "logit" in out.name:
            return [d if isinstance(d, int) else -1 for d in out.shape]
    return []


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="导出 ONNX/fp16/engine 并与 PyTorch 对拍")
    ap.add_argument("--checkpoint", default="runs/v1/checkpoint_best_ema.pth")
    ap.add_argument("--resolution", type=int, default=960)
    ap.add_argument("--out-dir", default="", help="默认 = models/rfdetr/fine-tune/<checkpoint 所在版本目录名>")
    ap.add_argument("--samples", type=int, default=4, help="对拍取样张数（取 test split 前 N 张）")
    ap.add_argument("--split", default="test")
    ap.add_argument("--skip-engine", action="store_true")
    ap.add_argument("--skip-onnx", action="store_true", help="复用已导出的 ONNX，只对拍/建引擎")
    ap.add_argument("--force", action="store_true", help="已存在也重建 engine")
    a = ap.parse_args(list(argv) if argv is not None else None)

    checkpoint = Path(a.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = HERE / a.checkpoint
    if not checkpoint.exists():
        raise SystemExit("checkpoint 不存在：%s" % checkpoint)
    # 版本即目录：产物落在 models/rfdetr/fine-tune/<版本>/（与 checkpoint 所在目录同名，
    # 保持"一个版本一个目录、一个名字"，不再出现 3c 那种旧名目录）
    version = checkpoint.parent.name
    out_dir = Path(a.out_dir) if a.out_dir else MODELS_ROOT / version
    out_dir.mkdir(parents=True, exist_ok=True)
    print("checkpoint=%s\n  sha256=%s\n  version=%s\n  out_dir=%s"
          % (checkpoint, sha256(checkpoint)[:16], version, out_dir))

    onnx_fp32 = out_dir / ("rf-detr-small-%s-%d-dynamic.onnx" % (version, a.resolution))
    if a.skip_onnx and onnx_fp32.exists():
        print("  SKIP ONNX 导出（复用 %s）" % onnx_fp32.name)
    else:
        print("[1/4] 导出 fp32 ONNX（动态 batch）")
        onnx_fp32 = export_onnx(sys.executable, checkpoint, a.resolution, out_dir, version)

    print("[2/4] 转 fp16（fp32 I/O）")
    onnx_fp16 = to_fp16(onnx_fp32)

    print("[3/4] 建 TRT engine（min/opt/max = 1/4/8）")
    engine = None
    if a.skip_engine:
        print("  SKIP（--skip-engine）")
    else:
        engine = build_engine(onnx_fp16, a.resolution, out_dir, a.force, "/usr/bin/trtexec",
                              version, checkpoint)

    print("[4/4] 与 PyTorch 对拍（%d 张 %s 图）" % (a.samples, a.split))
    result = validate(checkpoint, onnx_fp16, a.resolution, a.split, a.samples)
    meta = {"checkpoint": str(checkpoint), "sha256_16": sha256(checkpoint)[:16],
            "resolution": a.resolution, "class_ids": CLASS_IDS,
            "num_classes_head": result["classes_in_head"],
            "channel_contract_ok": result["channel_contract_ok"],
            "deploy": {"numClasses": len(CLASS_IDS) + 1, "targetClasses": [1, 2, 3],
                       "personClassId": 1, "ballClassId": 2, "coneClassId": 3},
            "artifacts": {"onnx_fp32": str(onnx_fp32), "onnx_fp16": str(onnx_fp16),
                          "engine": str(engine) if engine and engine.exists() else None},
            "validation": {"boxes_max_abs_diff_px": result["boxes_max_abs_diff_px"],
                           "label_agreement_min": result["label_agreement_min"],
                           "per_image": result["samples"]}}
    meta_path = out_dir / "meta_export.json"      # 模型卡是 meta.json，别覆盖它
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")

    # 判据分两级（实测依据）：
    #   硬契约（必须过）：head 宽度 4、标签一致率 >0.98（= classId 映射正确，这是唯一会"静默错"的点）
    #   数值参考（记数不卡）：框差 ≤5px、IoU≥0.9 匹配率 >0.8
    #     —— 实测 fp32 框差 6.96px 反而比 fp16 3.10px 更大，说明这几 px 来自
    #     "torch predict() 的 postprocess" 与"ONNX 图内 postprocess + 本脚本手写解码"两条路径的
    #     固有差异（含 CPU-ONNX 线程非确定性），**不是 fp16 精度损失**，所以不拿它当门槛。
    #     真正的一致性验证在 engine 侧（validate_engines.py）与端到端触球回归。
    ok = (result["channel_contract_ok"] and result["label_agreement_min"] > 0.98
          and result["boxes_max_abs_diff_px"] < 5.0
          and result["recall_matched_min"] > 0.8)
    print("\n对拍结论：head=%d 列（%s）｜匹配率 %.2f｜最大框差 %.3f px｜标签一致率 %.4f → %s"
          % (result["classes_in_head"], "classId 列口径正确" if result["channel_contract_ok"] else "列数异常",
             result["recall_matched_min"], result["boxes_max_abs_diff_px"],
             result["label_agreement_min"], "通过 ✅" if ok else "未通过，需排查 ❌"))
    print("元数据 → %s" % meta_path)
    print("\n部署侧要写进 YAML 的字段：numClasses=%d  targetClasses=[1,2,3]  "
          "personClassId=1 ballClassId=2 coneClassId=3" % (len(CLASS_IDS) + 1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

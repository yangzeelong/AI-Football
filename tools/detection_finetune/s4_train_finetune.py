#!/usr/bin/env python3
"""三分类微调入口：把 train_configs/*.yaml 喂给 rfdetr，带冻结主干与断点续训。

契约要点（读包源码 + 实测确认，踩过的坑写在注释里）：

* `.train(class_names=[...])` **必须显式传**，否则 `main.py:165-167` 不会用它的
  `num_classes`，输出通道数会停在默认 2 类；本脚本直接取 `base_layout.class_names()`。
* 分辨率只改 `resolution`，**不要**动 `positional_encoding_size`：base 权重的位置
  编码是 32²=1025，传 60²=3601 会在 `main.py:130` 的 `load_state_dict(strict=False)`
  上 size mismatch 直接报错（strict=False 只忽略缺/多键，形状不符照样抛）。
* 冻结主干：`main.py:195` 会对参数组再做一次 `requires_grad` 过滤，所以只要在
  `.train()` 之前把 `model.model.backbone`（或 `backbone[0].encoder`）设成
  `requires_grad=False`，它就不会进优化器。冻结后打印可训练参数量做实证。
* roboflow 布局必须有 `train/ valid/ test/` 三目录各自的 `_annotations.coco.json`，
  `run_test` 默认开启。

用法：
    python s4_train_finetune.py --config train_configs/smoke_512.yaml --subset 1 --epochs 1
    python s4_train_finetune.py --config train_configs/v1.yaml --freeze encoder
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import base_layout

TOOL_DIR = Path(__file__).resolve().parent
WEIGHTS = TOOL_DIR.parents[1] / "models/rfdetr/rf-detr-small.pth"


def load_yaml(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # 训练环境自带
        return yaml.safe_load(text) or {}
    except ImportError:                       # 兜底：允许 JSON 内容
        return json.loads(text)


def subset_dataset(src: Path, dst: Path, per_split: int, require_cone: bool = False) -> Path:
    """把每个 split 前 N 张图与标注裁剪成一个小数据集（冒烟用），保证 test/ 存在。

    `require_cone=True` 时**优先选带 cone 标注的图**——否则冒烟集里可能一个 cone 都没有，
    "类别下标正确"这件事根本没被验证到（会陷入假绿）。
    """
    if dst.exists():
        shutil.rmtree(dst)
    cone_id = base_layout.name_to_id()["cone"]
    for split in base_layout.SPLITS:
        (dst / split / "images").mkdir(parents=True, exist_ok=True)
        data = json.loads((src / split / "_annotations.coco.json").read_text(encoding="utf-8"))
        if require_cone:
            with_cone = {a["image_id"] for a in data["annotations"] if a["category_id"] == cone_id}
            head = [img for img in data["images"] if img["id"] in with_cone]
            rest = [img for img in data["images"] if img["id"] not in with_cone]
            images = (head + rest)[:per_split]
        else:
            images = data["images"][:per_split]
        ids = {img["id"] for img in images}
        for img in images:
            src_img = src / split / "images" / Path(img["file_name"]).name
            shutil.copy2(src_img, dst / split / "images" / Path(img["file_name"]).name)
        data["images"] = images
        data["annotations"] = [a for a in data["annotations"] if a["image_id"] in ids]
        counts = Counter(a["category_id"] for a in data["annotations"])
        print("  子集 %-5s 图 %2d | 标注 %s" % (split, len(images),
              {base_layout.id_to_name()[k]: v for k, v in counts.items()}))
        (dst / split / "_annotations.coco.json").write_text(
            json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return dst


def freeze_backbone(model: Any, mode: str) -> int:
    """冻结主干；返回冻结后的可训练参数量（用于实证 freeze 生效）。"""
    import torch
    if mode == "none":
        return sum(p.numel() for p in model.model.model.parameters() if p.requires_grad)
    target = model.model.model.backbone if mode == "backbone" else model.model.model.backbone[0].encoder
    for param in target.parameters():
        param.requires_grad = False
    torch.set_grad_enabled(True)
    return sum(p.numel() for p in model.model.model.parameters() if p.requires_grad)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="三分类微调入口")
    parser.add_argument("--config", required=True, help="train_configs/*.yaml")
    parser.add_argument("--dataset-dir", default=str(base_layout.DATASET_DIR))
    parser.add_argument("--output-dir", default=None, help="默认 <config 里的 output_dir>")
    parser.add_argument("--subset", type=int, default=0, help="每个 split 只用前 N 张（冒烟）")
    parser.add_argument("--subset-require-cone", action="store_true",
                        help="冒烟子集优先挑带 cone 标注的图（否则可能根本没验证到 cone）")
    parser.add_argument("--epochs", type=int, default=0, help="覆盖配置里的 epochs")
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--grad-accum-steps", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.0)
    parser.add_argument("--freeze", choices=("none", "encoder", "backbone"), default="none")
    parser.add_argument("--resume", default=None, help="断点续训的 checkpoint 路径")
    parser.add_argument("--dry-run", action="store_true", help="只装配模型并打印，不训练")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_yaml(Path(args.config) if Path(args.config).is_absolute() else TOOL_DIR / args.config)
    model_kwargs: dict[str, Any] = dict(config.get("model", {}))
    train_kwargs: dict[str, Any] = dict(config.get("train", {}))
    output_dir = Path(args.output_dir or train_kwargs.get("output_dir") or (TOOL_DIR / "runs/smoke"))
    if not output_dir.is_absolute():
        output_dir = TOOL_DIR / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_dir = Path(args.dataset_dir)
    if args.subset:
        dataset_dir = subset_dataset(dataset_dir, Path("/tmp/ds_subset"), args.subset,
                                     require_cone=args.subset_require_cone)
        print("冒烟子集: %s（每 split %d 图）" % (dataset_dir, args.subset))

    for key, value in (("epochs", args.epochs), ("batch_size", args.batch_size),
                       ("grad_accum_steps", args.grad_accum_steps), ("lr", args.lr)):
        if value:
            train_kwargs[key] = value

    import torch
    from rfdetr import RFDETRSmall

    model_kwargs.setdefault("pretrain_weights", str(WEIGHTS))
    model_kwargs["device"] = model_kwargs.get("device", "cuda")
    print("装配模型: %s" % json.dumps({k: v for k, v in model_kwargs.items()
                                      if k != "pretrain_weights"}, ensure_ascii=False))
    model = RFDETRSmall(**model_kwargs)
    trainable = freeze_backbone(model, args.freeze)
    print("冻结模式=%s → 可训练参数 %d（%.1fM）" % (args.freeze, trainable, trainable / 1e6))

    names = base_layout.class_names()
    train_kwargs.update(dataset_file="roboflow", dataset_dir=str(dataset_dir),
                        class_names=names, output_dir=str(output_dir))
    if args.resume:
        train_kwargs["resume"] = args.resume
    print("训练参数: %s" % json.dumps({k: v for k, v in train_kwargs.items() if k != "class_names"},
                                     ensure_ascii=False))
    print("类别: %s（应为 %s）| 输出列数应为 %d" % (names, base_layout.class_names(), base_layout.num_columns()))

    meta = {"started_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "args": vars(args),
            "model": {k: v for k, v in model_kwargs.items() if k != "pretrain_weights"},
            "train": {k: v for k, v in train_kwargs.items() if k != "class_names"},
            "class_names": names, "expected_columns": base_layout.num_columns(),
            "freeze": args.freeze, "trainable_params": trainable}
    (output_dir / "finetune_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                                                   encoding="utf-8")
    if args.dry_run:
        print("dry-run：不训练。meta → %s" % (output_dir / "finetune_meta.json"))
        return 0

    t0 = time.time()
    model.train(**train_kwargs)
    meta["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    meta["elapsed_sec"] = round(time.time() - t0, 1)
    (output_dir / "finetune_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                                                   encoding="utf-8")
    # 训练完扫一遍产物，确认列数是 4（person/ball/cone + 背景）
    cols = None
    for ckpt in sorted(output_dir.glob("*.pth")):
        try:
            state = torch.load(ckpt, map_location="cpu", weights_only=False)
            cols = int(state["model"]["class_embed.bias"].shape[0])
            print("checkpoint %s → class_embed.bias 列数 = %d（期望 %d）"
                  % (ckpt.name, cols, base_layout.num_columns()))
            break
        except Exception as exc:              # noqa: BLE001
            print("读 checkpoint 失败 %s: %s" % (ckpt, exc))
    print("训练完成，用时 %.0fs，产物 → %s" % (time.time() - t0, output_dir))
    return 0 if cols in (None, base_layout.num_columns()) else 1


if __name__ == "__main__":
    sys.exit(main())

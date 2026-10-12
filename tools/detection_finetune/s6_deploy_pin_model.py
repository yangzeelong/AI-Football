#!/usr/bin/env python3
"""把某个版本"上架"为可加载的模型：建权重软链 + 写模型卡 meta.json。

为什么要这一步（实测踩过）：
* Python 管线按 `models.detector.rfdetr_model_dir` 目录找**固定文件名** `rf-detr-small.pth`
  （`football_vision._resolve_weights`），所以每个版本目录里必须有这个名字（软链即可）；
* 模型卡 `meta.json` 是"这个目录到底是什么模型"的唯一说明（含 classId 契约与导出校验结果），
  没有它，几个月后没人知道 `fine-tune/v1-frozen-step2/` 里那堆 onnx 对应哪次训练。

数据来源：`meta_export.json`（`s6_deploy_export.py` 的产物，含 sha256、head 宽度、
对拍结果、engine 路径）。所以先导后上架：

    $PY s6_deploy_export.py --checkpoint runs/<版本>/checkpoint_best_ema.pth
    $PY s6_deploy_pin_model.py --version <版本> [--description "一句话说明"]

幂等：重复跑会刷新模型卡与软链；`--unpin` 只删软链（不动权重与产物）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                                  # AI-Football/
MODELS = ROOT / "models" / "rfdetr" / "fine-tune"
LINK_NAME = "rf-detr-small.pth"                         # 管线按这个文件名加载


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", required=True, help="版本名（= runs/<版本>/ 与 train_configs/<版本>.yaml）")
    ap.add_argument("--description", default="", help="写进模型卡的一句话说明")
    ap.add_argument("--checkpoint", default="", help="默认 runs/<版本>/checkpoint_best_ema.pth")
    ap.add_argument("--unpin", action="store_true", help="只删除权重软链")
    a = ap.parse_args(list(argv) if argv is not None else None)

    ver = a.version.strip()
    ver_dir = MODELS / ver
    checkpoint = Path(a.checkpoint) if a.checkpoint else (HERE / "runs" / ver / "checkpoint_best_ema.pth")
    if not checkpoint.is_absolute():
        checkpoint = (HERE / checkpoint).resolve()
    link = ver_dir / LINK_NAME

    if a.unpin:
        if link.is_symlink() or link.exists():
            link.unlink()
        print("已删除软链 %s" % link)
        return 0

    if not ver_dir.is_dir():
        raise SystemExit("版本目录不存在（先跑 s6_deploy_export.py）：%s" % ver_dir)
    if not checkpoint.is_file():
        raise SystemExit("权重不存在：%s" % checkpoint)
    export_meta_path = ver_dir / "meta_export.json"
    if not export_meta_path.is_file():
        raise SystemExit("缺 meta_export.json（先跑 s6_deploy_export.py）：%s" % export_meta_path)
    exported: dict[str, Any] = json.loads(export_meta_path.read_text(encoding="utf-8"))

    # ① 软链（相对路径，避免搬目录后失效）
    rel = Path("../../../../tools/detection_finetune/runs") / ver / checkpoint.name
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(rel)
    resolved = link.resolve()
    if not resolved.is_file():
        raise SystemExit("软链没指到真实文件：%s → %s" % (link, resolved))

    # ② 模型卡
    artifacts = {k: str(v) for k, v in (exported.get("artifacts") or {}).items() if v}
    artifacts["rfdetr_pth"] = "%s（软链 → %s；Python 管线按此文件名加载）" % (
        str(link.relative_to(ROOT)), str(rel))
    card = {
        "name": ver,
        "description": a.description or "person/ball/cone 三类微调（960 训练分辨率；配置 train_configs/%s.yaml）" % ver,
        "checkpoint": "tools/detection_finetune/runs/%s/%s" % (ver, checkpoint.name),
        "sha256_16": exported.get("sha256_16", ""),
        "resolution": exported.get("resolution", 960),
        "class_ids": exported.get("class_ids", {"person": 1, "ball": 2, "cone": 3}),
        "num_classes_head": exported.get("num_classes_head"),
        "channel_contract_ok": exported.get("channel_contract_ok"),
        "deploy": exported.get("deploy", {"numClasses": 4, "targetClasses": [1, 2, 3],
                                          "personClassId": 1, "ballClassId": 2, "coneClassId": 3}),
        "artifacts": artifacts,
        "validation": exported.get("validation", {}),
        "card_source": "meta_export.json（由 s6_deploy_export.py 生成，本文件由 s6_deploy_pin_model.py 汇总）",
        "load_resolution": exported.get("resolution", 960),
    }
    card_path = ver_dir / "meta.json"
    card_path.write_text(json.dumps(card, ensure_ascii=False, indent=1), encoding="utf-8")
    print("已上架 %s\n  软链 %s → %s\n  模型卡 %s（head=%s 列，对拍框差 %s px，标签一致率 %s）"
          % (ver, link.name, rel, card_path.name, card["num_classes_head"],
             card["validation"].get("boxes_max_abs_diff_px"),
             card["validation"].get("label_agreement_min")))
    print("  engine：%s" % artifacts.get("engine", "（还没建，跑 s6_deploy_export.py 不加 --skip-engine）"))
    return 0


if __name__ == "__main__":
    sys.exit(main())

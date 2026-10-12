#!/usr/bin/env python3
"""把步骤 A 导出的 fp32 ONNX 转成 fp16 ONNX，供 TensorRT 11 编译。

TensorRT 11 删除了 trtexec 的 `--fp16` 开关（strongly-typed），精度只能由图决定，
所以必须先转图。默认输出 **float32 I/O + fp16 内部**，这样 Nexusflow C++ 侧的
GPU 预处理（依赖 float32 输入绑定）和全部 float32 缓冲区都不需要改动。

用法：
    PY=/root/miniconda3/envs/ai-football/bin/python
    $PY tools/model_conversion/convert_fp16_onnx.py            # 全部登记模型，fp32 I/O
    $PY tools/model_conversion/convert_fp16_onnx.py --fp16-io  # 纯 fp16 I/O（仅对比实验）
    $PY tools/model_conversion/convert_fp16_onnx.py --only hrnet-w48-dark
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import onnx
from onnxruntime.transformers.float16 import convert_float_to_float16

ROOT = Path(__file__).resolve().parents[2]

# 对 fp16 数值敏感、且对速度影响有限的算子保持 fp32。
# 注意：ORT 的默认阻断列表里已含 Resize，这里显式列出以免被 op_block_list 覆盖掉。
EXTRA_BLOCK = ["Softmax", "LayerNormalization", "Erf", "ReduceSum", "Sqrt", "Resize"]

# name -> (源 fp32 ONNX, 目标前缀)
# 目标文件为 <前缀>-fp16-fp32io.onnx（或 --fp16-io 时 <前缀>-fp16.onnx）
MODELS: dict[str, tuple[str, str]] = {
    "rfdetr-960": (
        "models/rfdetr/onnx/rf-detr-small-960-dynamic.onnx",
        "models/rfdetr/onnx/rf-detr-small-960-dynamic",
    ),
    "rfdetr-512": (
        "models/rfdetr/onnx/rf-detr-small-512-dynamic.onnx",
        "models/rfdetr/onnx/rf-detr-small-512-dynamic",
    ),
    "hrnet-w48-dark": (
        "models/mmpose/hrnet/hrnet-w48-dark.onnx",
        "models/mmpose/hrnet/hrnet-w48-dark",
    ),
    "hrnet-w32": (
        "models/mmpose/hrnet/hrnet-w32-256x192.onnx",
        "models/mmpose/hrnet/hrnet-w32-256x192",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", action="append", choices=sorted(MODELS),
                        help="只处理指定模型，可重复。默认全部。")
    parser.add_argument("--fp16-io", action="store_true",
                        help="导出纯 fp16 I/O（仅供对比实验，C++ 侧不可用）。")
    # 显式路径模式：给"不在注册表里的新模型"用（例如微调出来的各版本），
    # 免去为每个版本改 MODELS 字典。
    parser.add_argument("--src", help="源 fp32 ONNX 路径（相对仓库根或绝对路径）")
    parser.add_argument("--dst-prefix", help="目标前缀（输出 <前缀>-fp16-fp32io.onnx）")
    return parser.parse_args()


def dedup_cast_outputs(model: onnx.ModelProto) -> int:
    """删除重复的 Cast 节点。

    convert_float_to_float16 对"被多个受保护算子共享的同一个输入张量"会重复插入
    同名 Cast（RF-DETR 图里 15 个、HRNet 图里 25 个），造成 SSA 违规。这些节点输入
    相同、语义等价，保留第一个即可。
    """
    seen: set[str] = set()
    kept: list = []
    dropped = 0
    for node in model.graph.node:
        outputs = [name for name in node.output if name]
        if outputs and all(name in seen for name in outputs):
            dropped += 1
            continue
        for name in outputs:
            if name in seen:
                print(f"   warning: partial duplicate output on {node.op_type}: {name}")
            seen.add(name)
        kept.append(node)
    del model.graph.node[:]
    model.graph.node.extend(kept)
    return dropped


def toposort_model(model: onnx.ModelProto) -> onnx.ModelProto:
    """按依赖关系重排节点，修掉 fp16 转换器留下的乱序。"""
    graph = model.graph
    available = {value.name for value in graph.input}
    available |= {init.name for init in graph.initializer}
    pending = list(graph.node)
    ordered: list = []
    while pending:
        rest = []
        progressed = False
        for node in pending:
            if all(not name or name in available for name in node.input):
                ordered.append(node)
                available.update(node.output)
                progressed = True
            else:
                rest.append(node)
        if not progressed:
            print(f"   warning: {len(rest)} nodes could not be ordered; kept as-is")
            ordered.extend(rest)
            break
        pending = rest
    del graph.node[:]
    graph.node.extend(ordered)
    return model


def convert(src: Path, dst: Path, keep_io_types: bool) -> bool:
    model = onnx.load(str(src))
    converted = convert_float_to_float16(
        model,
        keep_io_types=keep_io_types,
        disable_shape_infer=False,
        op_block_list=EXTRA_BLOCK,
    )
    print(f"   dropped duplicate cast nodes: {dedup_cast_outputs(converted)}")
    converted = toposort_model(converted)
    onnx.checker.check_model(converted, full_check=False)

    # onnx.save 的外部数据是追加写入的：目标文件已存在时 .onnx.data 会越滚越大
    # （实测同一个小模型从 60MB 涨到 178MB）。必须先删干净。
    for stale in (dst, dst.with_name(f"{dst.name}.data")):
        stale.unlink(missing_ok=True)

    onnx.save(converted, str(dst),
              save_as_external_data=True, all_tensors_to_one_file=True,
              location=f"{dst.name}.data", size_threshold=1024)
    return True


def main() -> int:
    args = parse_args()
    if args.src or args.dst_prefix:
        if not (args.src and args.dst_prefix):
            print("--src 与 --dst-prefix 必须成对使用", file=sys.stderr)
            return 2
        jobs = {"(显式路径)": (args.src, args.dst_prefix)}
    else:
        jobs = {n: MODELS[n] for n in (args.only or sorted(MODELS))}
    failures = 0

    for name, (src_rel, prefix) in jobs.items():
        src = Path(src_rel)
        if not src.is_absolute():
            src = ROOT / src
        if not src.is_file():
            print(f"SKIP {name}: missing source {src_rel}", file=sys.stderr)
            failures += 1
            continue

        prefix_path = Path(prefix)
        if not prefix_path.is_absolute():
            prefix_path = ROOT / prefix_path
        suffix = "-fp16.onnx" if args.fp16_io else "-fp16-fp32io.onnx"
        dst = Path(f"{prefix_path}{suffix}")
        print(f"\n=== {name} ===")
        try:
            shown = dst.relative_to(ROOT)
        except ValueError:
            shown = dst
        print(f"   {src_rel} -> {shown}")
        convert(src, dst, keep_io_types=not args.fp16_io)

        saved = onnx.load(str(dst), load_external_data=False)
        for value in list(saved.graph.input) + list(saved.graph.output):
            dims = [(d.dim_param or d.dim_value) for d in value.type.tensor_type.shape.dim]
            dtype = onnx.TensorProto.DataType.Name(value.type.tensor_type.elem_type)
            print(f"   {value.name:12s} {dims} {dtype}")

        size = dst.stat().st_size
        data = dst.with_name(f"{dst.name}.data")
        if data.exists():
            size += data.stat().st_size
        print(f"   saved ({size / 1e6:.1f} MB)")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

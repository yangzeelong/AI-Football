#!/usr/bin/env python3
"""Convert a float32 ONNX graph to float16 so the TensorRT builder emits fp16 kernels.

TensorRT 11.1 removed the legacy `trtexec --fp16` switch, so the precision of a
generated engine is decided by the ONNX graph only. This tool performs that
conversion for the C++ AI-Football pipeline.

`--io-precision` (default `fp32`) is the important knob for the C++ runtime:

* `fp32` keeps the graph inputs/outputs float32 and inserts Cast nodes at the
  boundaries. `inference/IInferenceEngine` only accepts float32 host buffers, so
  this is the mode the SDK engines are built with. It also keeps the CUDA
  preprocessing kernel usable, because it requires a float32 input binding.
* `fp16` converts inputs/outputs as well, which matches the Python AI-Football
  pipeline (it feeds/fetches half tensors directly).

Numerically sensitive operators are kept in float32 to avoid accuracy collapse,
mirroring the Python project's conversion.

Usage:
    python tools/model_conversion/convert_fp16_onnx.py \
      --input  /path/model.onnx \
      --output /path/model-fp16.onnx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import onnx
from onnxruntime.transformers.float16 import convert_float_to_float16

# Keeping these in float32 has negligible runtime cost and protects the graph
# from fp16 accumulation error (Softmax/LayerNorm in RF-DETR's transformer, the
# Resize/upsample paths in HRNet, and the reduction chain in both).
DEFAULT_BLOCK_OPS = ("Softmax", "LayerNormalization", "Erf", "ReduceSum", "Sqrt", "Resize")


def dedup_cast_outputs(model: onnx.ModelProto) -> int:
    """Drop duplicated Cast nodes inserted for a shared, block-listed tensor."""
    seen: set[str] = set()
    kept: list[onnx.NodeProto] = []
    dropped = 0
    for node in model.graph.node:
        outputs = [name for name in node.output if name]
        if outputs and all(name in seen for name in outputs):
            dropped += 1
            continue
        for name in outputs:
            if name in seen:
                print(f"   warning: partial duplicate output on {node.op_type}: {name}",
                      file=sys.stderr)
            seen.add(name)
        kept.append(node)
    del model.graph.node[:]
    model.graph.node.extend(kept)
    return dropped


def toposort_model(model: onnx.ModelProto) -> int:
    """Restore topological order after the converter's Cast insertion."""
    graph = model.graph
    available = {value.name for value in graph.input}
    available |= {init.name for init in graph.initializer}
    pending = list(graph.node)
    ordered: list[onnx.NodeProto] = []
    while pending:
        rest: list[onnx.NodeProto] = []
        progressed = False
        for node in pending:
            if all(not name or name in available for name in node.input):
                ordered.append(node)
                available.update(node.output)
                progressed = True
            else:
                rest.append(node)
        if not progressed:
            ordered.extend(rest)
            break
        pending = rest
    del graph.node[:]
    graph.node.extend(ordered)
    return len(pending)


def convert(input_path: Path, output_path: Path, io_precision: str,
            block_ops: tuple[str, ...]) -> None:
    if not input_path.is_file():
        raise FileNotFoundError(f"input ONNX not found: {input_path}")

    print(f"=== {input_path.name} -> {output_path.name} (io={io_precision}) ===", flush=True)
    model = onnx.load(str(input_path))
    external = any(init.data_location == onnx.TensorProto.EXTERNAL
                   for init in model.graph.initializer)

    converted = convert_float_to_float16(
        model,
        keep_io_types=(io_precision == "fp32"),
        disable_shape_infer=False,
        op_block_list=list(block_ops),
    )
    print(f"   dropped duplicate cast nodes: {dedup_cast_outputs(converted)}")
    unsorted = toposort_model(converted)
    if unsorted:
        print(f"   warning: {unsorted} nodes could not be ordered", file=sys.stderr)
    onnx.checker.check_model(converted, full_check=False)

    # onnx.save appends to an existing external-data file, so clear stale ones.
    for stale in (output_path, output_path.with_name(f"{output_path.name}.data")):
        stale.unlink(missing_ok=True)

    save_kwargs = {}
    if external:
        save_kwargs = dict(
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=f"{output_path.name}.data",
            size_threshold=1024,
        )
    onnx.save(converted, str(output_path), **save_kwargs)

    for value in list(converted.graph.input) + list(converted.graph.output):
        dtype = onnx.TensorProto.DataType.Name(value.type.tensor_type.elem_type)
        dims = [(d.dim_param or d.dim_value) for d in value.type.tensor_type.shape.dim]
        print(f"   {value.name:14s} {dims} {dtype}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, type=Path, help="float32 ONNX graph")
    parser.add_argument("--output", required=True, type=Path, help="float16 ONNX graph")
    parser.add_argument("--io-precision", choices=("fp32", "fp16"), default="fp32",
                        help="graph input/output precision (default: fp32, for the C++ runtime)")
    parser.add_argument("--block-op", action="append", default=None,
                        metavar="OP", help="operator kept in float32 (repeatable)")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    block_ops = tuple(args.block_op) if args.block_op else DEFAULT_BLOCK_OPS
    try:
        convert(args.input, args.output, args.io_precision, block_ops)
    except Exception as error:  # noqa: BLE001 - surface a clean CLI failure
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

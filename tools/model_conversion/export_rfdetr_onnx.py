#!/usr/bin/env python3
"""Export an RF-DETR checkpoint using the installed package exporter."""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx

MODEL_CLASSES = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "base": "RFDETRBase",
    "medium": "RFDETRMedium",
    "large": "RFDETRLarge",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", choices=sorted(MODEL_CLASSES), default="small")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="Square inference resolution. Omit to use the checkpoint default.",
    )
    parser.add_argument(
        "--dynamic-batch",
        action="store_true",
        help="Export a batch-dynamic ONNX graph. TensorRT profile limits are set during engine build.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"RF-DETR checkpoint not found: {checkpoint}")

    try:
        import rfdetr
    except ImportError as exc:
        raise RuntimeError(
            "RF-DETR export requires the rfdetr package. "
            "Install the AI-Football requirements first."
        ) from exc

    if args.dynamic_batch:
        # RF-DETR's projector LayerNorm derives the channel count through
        # x.size(3). The legacy exporter treats that value as symbolic when
        # batch is dynamic, although the layer already stores the same static
        # channel shape in normalized_shape. Patch only the export-time
        # implementation; the installed package and checkpoint are untouched.
        import torch
        import torch.nn.functional as F
        from rfdetr.models.backbone.projector import LayerNorm
        from rfdetr.models import transformer as transformer_module

        def export_layer_norm(self, x):
            x = x.permute(0, 2, 3, 1)
            x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
            return x.permute(0, 3, 1, 2)

        LayerNorm.forward = export_layer_norm

        original_gen_encoder_output_proposals = (
            transformer_module.gen_encoder_output_proposals
        )

        def export_gen_encoder_output_proposals(
            memory, memory_padding_mask, spatial_shapes, unsigmoid=True
        ):
            """Keep the no-mask proposal scale dynamic along the batch axis."""
            if memory_padding_mask is not None:
                return original_gen_encoder_output_proposals(
                    memory, memory_padding_mask, spatial_shapes, unsigmoid
                )

            # The package implementation uses a Python list comprehension over
            # N_, which the legacy exporter evaluates at the example batch size.
            # Derive a zero-cost batch-shaped tensor from memory instead.
            def dynamic_valid_size(spatial_size):
                batch_column = memory[:, :1, 0]
                return batch_column[:, 0] * 0.0 + spatial_size

            N_, _, _ = memory.shape
            proposals = []
            for level, (height, width) in enumerate(spatial_shapes):
                valid_H = dynamic_valid_size(height)
                valid_W = dynamic_valid_size(width)

                grid_y, grid_x = torch.meshgrid(
                    torch.linspace(
                        0,
                        height - 1,
                        height,
                        dtype=torch.float32,
                        device=memory.device,
                    ),
                    torch.linspace(
                        0,
                        width - 1,
                        width,
                        dtype=torch.float32,
                        device=memory.device,
                    ),
                )
                grid = torch.cat(
                    [grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1
                )
                scale = torch.cat(
                    [valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1
                ).view(N_, 1, 1, 2)
                grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
                wh = torch.ones_like(grid) * 0.05 * (2.0 ** level)
                proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
                proposals.append(proposal)

            output_proposals = torch.cat(proposals, 1)
            output_proposals_valid = (
                (output_proposals > 0.01) & (output_proposals < 0.99)
            ).all(-1, keepdim=True)
            if unsigmoid:
                output_proposals = torch.log(
                    output_proposals / (1 - output_proposals)
                )
                output_proposals = output_proposals.masked_fill(
                    ~output_proposals_valid, float("inf")
                )
            else:
                output_proposals = output_proposals.masked_fill(
                    ~output_proposals_valid, float(0)
                )
            output_memory = memory.masked_fill(
                ~output_proposals_valid, float(0)
            )
            return output_memory.to(memory.dtype), output_proposals.to(memory.dtype)

        transformer_module.gen_encoder_output_proposals = (
            export_gen_encoder_output_proposals
        )

    model_class = getattr(rfdetr, MODEL_CLASSES[args.size])
    model_kwargs = {
        "pretrain_weights": str(checkpoint),
        "device": args.device,
    }
    if args.resolution is not None:
        if args.resolution <= 0 or args.resolution % 16 != 0:
            raise ValueError("--resolution must be a positive multiple of 16")
        model_kwargs["resolution"] = args.resolution
    model = model_class(**model_kwargs)
    output_dir.mkdir(parents=True, exist_ok=True)
    export = getattr(model, "export", None)
    if export is None:
        raise RuntimeError("Installed rfdetr package does not expose model.export()")

    # RF-DETR 1.3.0 predates the dynamo exporter used by newer PyTorch.  Its
    # graph is compatible with the legacy exporter and that path also keeps
    # the model's shape-dependent tensor construction traceable.
    import torch.onnx

    original_export = torch.onnx.export

    def legacy_export(*export_args, **export_kwargs):
        export_kwargs.setdefault("dynamo", False)
        if args.dynamic_batch:
            dynamic_axes = dict(export_kwargs.get("dynamic_axes") or {})
            input_names = export_kwargs.get("input_names") or ["input"]
            output_names = export_kwargs.get("output_names") or ["dets", "labels"]
            for name in list(input_names) + list(output_names):
                dynamic_axes.setdefault(name, {})[0] = "batch"
            export_kwargs["dynamic_axes"] = dynamic_axes
        return original_export(*export_args, **export_kwargs)

    torch.onnx.export = legacy_export
    try:
        export(
            output_dir=str(output_dir),
            format="onnx",
            opset_version=args.opset,
            verbose=False,
        )
    finally:
        torch.onnx.export = original_export

    source = output_dir / "inference_model.onnx"
    if not source.is_file():
        raise RuntimeError(f"RF-DETR exporter did not create {source}")

    # Normalize names to the C++ deployment contract. RF-DETR's `dets` output
    # is actually center-format boxes and `labels` is the class-logit tensor.
    graph = onnx.load(str(source))
    rename = {"input": "image", "dets": "pred_boxes", "labels": "pred_logits"}
    for value in list(graph.graph.input) + list(graph.graph.output):
        if value.name in rename:
            value.name = rename[value.name]
    for node in graph.graph.node:
        node.input[:] = [rename.get(name, name) for name in node.input]
        node.output[:] = [rename.get(name, name) for name in node.output]

    resolution = model.model.resolution
    suffix = "-dynamic" if args.dynamic_batch else ""
    target = output_dir / f"rf-detr-{args.size}-{resolution}{suffix}.onnx"
    onnx.save(graph, str(target))
    print(f"RF-DETR export completed: {target}")
    batch_label = "batch" if args.dynamic_batch else "1"
    print(f"input: image [{batch_label},3,{resolution},{resolution}] float32 NCHW")
    print(f"outputs: pred_boxes [{batch_label},N,4], pred_logits [{batch_label},N,91]")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)

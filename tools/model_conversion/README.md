# Model Conversion

This directory contains the reproducible conversion path for the C++
AI-Football pipeline.

## Runtime Contracts

| Model | ONNX input | ONNX output | TensorRT binding names |
| --- | --- | --- | --- |
| RF-DETR small | `image`, `B x 3 x S x S`, float32 NCHW | raw `pred_boxes` + `pred_logits` | input `image`; outputs `pred_boxes` / `pred_logits` |
| HRNet-W48-DARK | `images`, `B x 3 x 384 x 288`, float32 NCHW | `heatmaps`, `[B,133,96,72]` | `images` / `heatmaps` |

The C++ preprocessing matches RF-DETR's Python inference path: RGB values are
scaled to `[0, 1]`, ImageNet-normalized, and resized directly to `512 x 512`
without letterboxing. The exported models must therefore contain the neural
network only. RF-DETR's COCO output indices are `1 = person` and
`37 = sports ball`.

## Prerequisites

The C++ runtime does not require Python. Only model conversion requires the
AI-Football Python dependencies. Run conversion in a dedicated virtual or
conda environment; do not install the requirements into the system Python or
modify the project's official runtime environment:

```bash
python -m pip install -r /home/hx1/yzl/Work/AI-Football/requirements.txt
```

TensorRT and `trtexec` must be installed on the conversion machine as well.
TensorRT 11.1 removed the legacy `--fp16` command-line switch, so the
builder uses the precision encoded by the ONNX graph.

## HRNet-W48-DARK

```bash
python tools/model_conversion/export_hrnet_onnx.py \
  --config /home/hx1/yzl/Work/AI-Football/models/mmpose/configs/wholebody_2d_keypoint/topdown_heatmap/coco-wholebody/td-hm_hrnet-w48_dark-8xb32-210e_coco-wholebody-384x288.py \
  --checkpoint /home/hx1/yzl/Work/AI-Football/models/mmpose/hrnet/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth \
  --output /home/hx1/yzl/Work/AI-Football/models/mmpose/hrnet/hrnet-w48-dark.onnx \
  --height 384 --width 288 --opset 18 --device cuda:0 --dynamic-batch

tools/model_conversion/build_tensorrt_engines.sh hrnet \
  --onnx /home/hx1/yzl/Work/AI-Football/models/mmpose/hrnet/hrnet-w48-dark.onnx \
  --engine /home/hx1/yzl/Work/AI-Football/models/mmpose/hrnet/hrnet-w48-dark.engine \
  --max-batch 16
```

## RF-DETR

The converter supports explicit RF-DETR resolution experiments. For the
current small checkpoint, the default export is 512. To reproduce the 960
experiment used by the resolution comparison workflow, pass
`--resolution 960`; this creates a separate ONNX/engine pair and does not
overwrite the 512 deployment artifacts.

```bash
python tools/model_conversion/export_rfdetr_onnx.py \
  --size small \
  --checkpoint /home/hx1/yzl/Work/AI-Football/models/rfdetr/rf-detr-small.pth \
  --output-dir /home/hx1/yzl/Work/AI-Football/models/rfdetr/onnx \
  --device cpu

tools/model_conversion/build_tensorrt_engines.sh rfdetr \
  --onnx /home/hx1/yzl/Work/AI-Football/models/rfdetr/onnx/rf-detr-small.onnx \
  --engine /home/hx1/yzl/Work/AI-Football/models/rfdetr/rf-detr-small.engine \
  --input-size 512 --max-batch 1
```

For a batch-capable RF-DETR engine, export the batch dimension as dynamic and
set the TensorRT optimization profile limit. The resulting engine accepts any
batch in the profile range `1..4`:

```bash
python tools/model_conversion/export_rfdetr_onnx.py \
  --size small \
  --checkpoint /home/hx1/yzl/Work/AI-Football/models/rfdetr/rf-detr-small.pth \
  --output-dir /home/hx1/yzl/Work/AI-Football/models/rfdetr/onnx \
  --device cpu --resolution 960 --dynamic-batch

tools/model_conversion/build_tensorrt_engines.sh rfdetr \
  --onnx /home/hx1/yzl/Work/AI-Football/models/rfdetr/onnx/rf-detr-small-960-dynamic.onnx \
  --engine /home/hx1/yzl/Work/AI-Football/models/rfdetr/rf-detr-small-960-b4.engine \
  --input-size 960 --max-batch 4
```

The dynamic RF-DETR exporter also replaces two shape constructions from the
installed RF-DETR package that are traced as batch-1 constants by the legacy
TorchScript exporter. This keeps the generated graph valid when TensorRT
executes it at a larger batch.

960 variant:

```bash
python tools/model_conversion/export_rfdetr_onnx.py \
  --size small \
  --checkpoint /home/hx1/yzl/Work/AI-Football/models/rfdetr/rf-detr-small.pth \
  --output-dir /home/hx1/yzl/Work/AI-Football/models/rfdetr/onnx \
  --device cpu --resolution 960

tools/model_conversion/build_tensorrt_engines.sh rfdetr \
  --onnx /home/hx1/yzl/Work/AI-Football/models/rfdetr/onnx/rf-detr-small-960.onnx \
  --engine /home/hx1/yzl/Work/AI-Football/models/rfdetr/rf-detr-small-960.engine \
  --input-size 960 --max-batch 1
```

## Runtime Instances and Batches

`instanceCount` creates independent TensorRT execution contexts, CUDA streams,
and device buffers. Instances can process chunks concurrently, but each
instance also consumes model GPU memory. Start with `1` and increase only
after measuring GPU utilization and memory headroom.

For `RFDetrDetector`, `maxBatchSize` controls the module flush size and the
requested model batch. For `HRNetPoseEstimator`, `maxBatch` is the requested
model batch. Static batch-1 engines are detected at startup and automatically
clamp the effective batch to `1`; the module still chunks larger inputs, so
rows are not silently dropped. Values greater than `1` require engines built
from a dynamic-batch ONNX graph.

The detector copies each batch output tensor from device to host once, then
decodes per-frame slices. This avoids both duplicate results for later batch
items and an unnecessary CUDA synchronization for every frame.

After export, inspect the `trtexec` parser output. The current RF-DETR
export produces two tensors containing raw class logits and normalized
center-format boxes. The C++ decoder also retains compatibility with a single
baked detection tensor containing six values per query.

## C++ Configuration

Point `app/config.yaml` at the generated engines. The checked-in integration
configuration already points to the generated engines under AI-Football:

```yaml
enginePath: models/rfdetr/rf-detr-small.engine
enginePath: models/mmpose/hrnet/hrnet-w48-dark.engine
```

From the Nexusflow repository root, build and run the complete pipeline:

```bash
cmake --build build --parallel 4
build/app/football_server app/config.yaml
```

The sample configuration reads `data/input.mp4` and writes
`output/observations.jsonl`.

The current C++ pose implementation is HRNet heatmap based. The Python
project's `rtmpose-m` preset emits SimCC outputs and is not interchangeable
with this engine without adding a separate SimCC decoder.

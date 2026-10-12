#!/usr/bin/env bash
# 用 trtexec 把 fp16 ONNX 编译成 TensorRT engine。
#
# profile 统一 min=1 / opt=4 / max=8：
#   opt=4 与 C++ 侧 maxBatchSize:4 / batchFrameCount:4 对齐
#   max=8 给 HRNet 的 flip-test 合并（2*B）留空间
# max 必须 >= Nexusflow config.yaml 里配置的批量，否则 setInputShape 会失败。
#
# TRT 11.1 注意：
#   - 没有 --fp16 开关（strongly-typed），精度由图决定，所以必须先跑 convert_fp16_onnx.py
#   - 这里不加 --noTF32（默认已开 TF32，只影响残留 fp32 算子）
#
# 用法：
#   bash tools/model_conversion/build_engines.sh              # 全部
#   bash tools/model_conversion/build_engines.sh hrnet-w32    # 指定模型
#   FORCE=1 bash tools/model_conversion/build_engines.sh      # 覆盖已存在的 engine
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
ROOT="$(pwd)"
TRTEXEC="${TRTEXEC:-/usr/bin/trtexec}"
MIN_BATCH=1
OPT_BATCH=4
MAX_BATCH=8
FORCE="${FORCE:-0}"

# name | onnx | engine | input binding | H | W
MODELS=(
  "rfdetr-960|models/rfdetr/onnx/rf-detr-small-960-dynamic-fp16-fp32io.onnx|models/rfdetr/rf-detr-small-960-fp16-b8-fp32io.engine|image|960|960"
  "rfdetr-512|models/rfdetr/onnx/rf-detr-small-512-dynamic-fp16-fp32io.onnx|models/rfdetr/rf-detr-small-512-fp16-b8-fp32io.engine|image|512|512"
  "hrnet-w48-dark|models/mmpose/hrnet/hrnet-w48-dark-fp16-fp32io.onnx|models/mmpose/hrnet/hrnet-w48-dark-fp16-b8-fp32io.engine|images|384|288"
  "hrnet-w32|models/mmpose/hrnet/hrnet-w32-256x192-fp16-fp32io.onnx|models/mmpose/hrnet/hrnet-w32-256x192-fp16-b8-fp32io.engine|images|256|192"
)

if [ ! -x "$TRTEXEC" ]; then
  echo "trtexec not found at $TRTEXEC (override with TRTEXEC=/path/to/trtexec)" >&2
  exit 1
fi

wanted=("$@")
status=0

for entry in "${MODELS[@]}"; do
  IFS='|' read -r name onnx engine input h w <<<"$entry"
  if [ ${#wanted[@]} -gt 0 ]; then
    match=0
    for want in "${wanted[@]}"; do [ "$want" = "$name" ] && match=1; done
    [ "$match" = "1" ] || continue
  fi

  echo
  echo "########## $name ##########"
  if [ ! -f "$onnx" ]; then
    echo "SKIP: missing $onnx (run convert_fp16_onnx.py first)" >&2
    status=1
    continue
  fi
  if [ -f "$engine" ] && [ "$FORCE" != "1" ]; then
    echo "SKIP: $engine already exists (FORCE=1 to rebuild)"
    continue
  fi

  "$TRTEXEC" \
    --onnx="$onnx" \
    --saveEngine="$engine" \
    --skipInference \
    --builderOptimizationLevel=3 \
    --minShapes="${input}:${MIN_BATCH}x3x${h}x${w}" \
    --optShapes="${input}:${OPT_BATCH}x3x${h}x${w}" \
    --maxShapes="${input}:${MAX_BATCH}x3x${h}x${w}" 2>&1 \
    | grep -E "Engine built in|PASSED|FAILED|Error" || true

  if [ -f "$engine" ]; then
    printf '  -> %s (%.1f MB)\n' "$engine" "$(echo "scale=1; $(stat -c%s "$engine")/1000000" | bc)"
  else
    echo "  -> FAILED" >&2
    status=1
  fi
done

echo
echo "done (root=$ROOT, profile=${MIN_BATCH}/${OPT_BATCH}/${MAX_BATCH})"
exit $status

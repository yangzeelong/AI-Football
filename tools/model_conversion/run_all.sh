#!/usr/bin/env bash
# TensorRT 转换全流程 —— README「快速开始」的可执行版本。
#
# 依次执行：环境自检 -> 导出 fp32 ONNX -> 转 fp16 -> 编译 engine -> 数值校验
#
# 用法：
#   bash tools/model_conversion/run_all.sh                  # 全部模型，全部步骤
#   bash tools/model_conversion/run_all.sh rfdetr-960       # 只跑一个模型
#   STEPS=convert,build bash tools/model_conversion/run_all.sh   # 只跑指定步骤
#   FORCE=1 bash tools/model_conversion/run_all.sh          # 覆盖已有产物
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
ROOT="$(pwd)"

# 根路径推导：约定布局 <WORKSPACE>/{AI-Football,Nexusflow,data}（与 base_layout.py 同一套约定）
WORKSPACE="${AI_FOOTBALL_WORKSPACE:-$(cd "$ROOT/.." && pwd)}"
PY="${PY:-${AI_FOOTBALL_ENV:-/root/miniconda3/envs/ai-football}/bin/python}"
NEXUS="${NEXUS:-${AI_FOOTBALL_NEXUSFLOW:-$WORKSPACE/Nexusflow}}"
# TensorRT 的 Python 绑定在本机只装于系统解释器；没有它则跳过校验步骤。
TRT_PY="${TRT_PY:-/usr/bin/python3.12}"
STEPS="${STEPS:-env,export,convert,build,validate}"
FORCE="${FORCE:-0}"

HRNET_CFG_DIR="models/mmpose/configs/wholebody_2d_keypoint/topdown_heatmap/coco-wholebody"
HRNET_CKPT_DIR="models/mmpose/hrnet"

# name | kind | H | W | fp32 onnx | engine | hrnet config | hrnet ckpt
MODELS=(
  "rfdetr-960|rfdetr|960|960|models/rfdetr/onnx/rf-detr-small-960-dynamic.onnx|models/rfdetr/rf-detr-small-960-fp16-b8-fp32io.engine|-|-"
  "rfdetr-512|rfdetr|512|512|models/rfdetr/onnx/rf-detr-small-512-dynamic.onnx|models/rfdetr/rf-detr-small-512-fp16-b8-fp32io.engine|-|-"
  "hrnet-w48-dark|hrnet|384|288|models/mmpose/hrnet/hrnet-w48-dark.onnx|models/mmpose/hrnet/hrnet-w48-dark-fp16-b8-fp32io.engine|${HRNET_CFG_DIR}/td-hm_hrnet-w48_dark-8xb32-210e_coco-wholebody-384x288.py|${HRNET_CKPT_DIR}/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth"
  "hrnet-w32|hrnet|256|192|models/mmpose/hrnet/hrnet-w32-256x192.onnx|models/mmpose/hrnet/hrnet-w32-256x192-fp16-b8-fp32io.engine|${HRNET_CFG_DIR}/td-hm_hrnet-w32_8xb64-210e_coco-wholebody-256x192.py|${HRNET_CKPT_DIR}/hrnet_w32_coco_wholebody_256x192-853765cd_20200918.pth"
)

wanted=("$@")
status=0
has_step() { [[ ",${STEPS}," == *",$1,"* ]]; }
selected() {
  [ ${#wanted[@]} -eq 0 ] && return 0
  local want
  for want in "${wanted[@]}"; do [ "$want" = "$1" ] && return 0; done
  return 1
}
banner() { printf '\n\033[1m== %s ==\033[0m\n' "$1"; }

# --------------------------------------------------------------------------- #
# 步骤 1：环境自检
# --------------------------------------------------------------------------- #
if has_step env || has_step all; then
  banner "步骤 1/5  环境自检"
  if [ ! -x "$PY" ]; then
    echo "找不到解释器 $PY（用 PY=... 覆盖）" >&2
    exit 1
  fi
  "$PY" - <<'EOF' || exit 1
import sys
import numpy
required = ["onnx", "onnxscript", "onnxruntime", "onnxsim", "onnx_graphsurgeon"]
missing = []
for name in required:
    try:
        __import__(name)
    except ImportError:
        missing.append(name)
print(f"  python  {sys.version.split()[0]}")
print(f"  numpy   {numpy.__version__}")
for name in required:
    mark = "MISSING" if name in missing else "ok"
    print(f"  {name:20s} {mark}")
if missing:
    print("\n缺依赖，先执行：", file=sys.stderr)
    print(f"  {sys.executable} -m pip install {' '.join(m.replace('_', '-') for m in missing)}",
          file=sys.stderr)
    sys.exit(1)
if int(numpy.__version__.split(".")[0]) >= 2:
    print("\n⚠️  numpy >= 2 会破坏 mmpose / ultralytics，必须压回：", file=sys.stderr)
    print(f"  {sys.executable} -m pip install 'numpy<2'", file=sys.stderr)
    sys.exit(1)
EOF
  echo "  环境自检通过"
fi

# --------------------------------------------------------------------------- #
# 步骤 2：导出 fp32 动态 batch ONNX（脚本在 Nexusflow 仓库，这里只做调用）
# --------------------------------------------------------------------------- #
do_export() {
  local name="$1" kind="$2" h="$3" w="$4" onnx="$5" cfg="$7" ckpt="$8"
  [ -f "$onnx" ] && [ "$FORCE" != "1" ] && { echo "  跳过（已存在 $onnx）"; return 0; }
  local rc=0
  if [ "$kind" = "rfdetr" ]; then
    ( cd "$NEXUS" && "$PY" tools/model_conversion/export_rfdetr_onnx.py \
        --size small \
        --checkpoint "$ROOT/models/rfdetr/rf-detr-small.pth" \
        --output-dir "$ROOT/models/rfdetr/onnx" \
        --device cpu --resolution "$h" --dynamic-batch ) 2>&1 | tail -4 || rc=$?
  else
    ( cd "$NEXUS" && "$PY" tools/model_conversion/export_hrnet_onnx.py \
        --config "$ROOT/$cfg" \
        --checkpoint "$ROOT/$ckpt" \
        --output "$ROOT/$onnx" \
        --height "$h" --width "$w" \
        --opset 18 --device cuda:0 --dynamic-batch ) 2>&1 | tail -4 || rc=$?
  fi
  [ -f "$onnx" ] || { echo "  导出失败：$onnx 未生成" >&2; return 1; }
  return $rc
}

# --------------------------------------------------------------------------- #
# 步骤 4：数值校验
# --------------------------------------------------------------------------- #
do_validate() {
  local name="$1" kind="$2" h="$3" w="$4" engine="$6"
  if ! "$TRT_PY" -c "import tensorrt" 2>/dev/null; then
    echo "  跳过（$TRT_PY 没有 tensorrt 绑定，用 TRT_PY=... 指定）"
    return 0
  fi
  if [ "$kind" = "rfdetr" ]; then
    "$TRT_PY" tools/model_conversion/validate_engines.py \
      --engine "$engine" --kind rfdetr --input-size "$h"
  else
    "$TRT_PY" tools/model_conversion/validate_engines.py \
      --engine "$engine" --kind hrnet --height "$h" --width "$w"
  fi
}

# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
has_model_step=0
for step in export convert build validate; do
  has_step "$step" && has_model_step=1
done

for entry in "${MODELS[@]}"; do
  IFS='|' read -r name kind h w onnx engine cfg ckpt <<<"$entry"
  selected "$name" || continue
  [ "$has_model_step" -eq 1 ] && banner "模型 $name  (${kind} ${h}x${w})"

  if has_step export; then
    echo "[2/5] 导出 fp32 ONNX"
    do_export "$name" "$kind" "$h" "$w" "$onnx" "$engine" "$cfg" "$ckpt" || status=1
  fi

  if has_step convert; then
    echo "[3/5] 转 fp16（保留 float32 I/O）"
    "$PY" tools/model_conversion/convert_fp16_onnx.py --only "$name" | grep -vE "Warning|warn" || status=1
  fi

  if has_step build; then
    echo "[4/5] 编译 engine（profile 1/4/8）"
    bash tools/model_conversion/build_engines.sh "$name" || status=1
  fi

  if has_step validate; then
    echo "[5/5] 数值校验（对比 PyTorch）"
    do_validate "$name" "$kind" "$h" "$w" "$onnx" "$engine" || status=1
  fi
done

echo
if [ $status -eq 0 ]; then
  echo "全部步骤完成。"
else
  echo "有步骤失败，请查看上面的输出。" >&2
fi
exit $status

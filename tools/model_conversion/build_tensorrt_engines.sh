#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf '%s\n' \
    "Usage:" \
    "  $0 hrnet --onnx PATH --engine PATH [--max-batch N] [--opt-batch N] [--fp16] [--python PATH]" \
    "  $0 rfdetr --onnx PATH --engine PATH [--input-size N] [--max-batch N] [--opt-batch N] [--fp16] [--python PATH]" \
    "" \
    "  --opt-batch  Batch the optimization profile is tuned for. Defaults to" \
    "               --max-batch. Set it to the batch the pipeline usually" \
    "               aggregates when that is smaller than the profile maximum." \
    "  --fp16     Convert the graph to float16 weights before building. Graph" \
    "             inputs/outputs stay float32, which is what the C++ runtime" \
    "             accepts and what keeps the CUDA preprocess kernel usable." \
    "  --python   Python interpreter with onnx + onnxruntime (required by --fp16)."
}

if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi

kind="$1"
shift
onnx=""
engine=""
max_batch=16
opt_batch=""
input_size=512
fp16=0
python_bin="python3"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --onnx) onnx="$2"; shift 2 ;;
    --engine) engine="$2"; shift 2 ;;
    --max-batch) max_batch="$2"; shift 2 ;;
    --opt-batch) opt_batch="$2"; shift 2 ;;
    --input-size) input_size="$2"; shift 2 ;;
    --fp16) fp16=1; shift ;;
    --python) python_bin="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'unknown argument: %s\n' "$1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$onnx" || -z "$engine" ]]; then
  usage
  exit 2
fi
if [[ ! -f "$onnx" ]]; then
  printf 'ONNX file not found: %s\n' "$onnx" >&2
  exit 1
fi
if ! [[ "$max_batch" =~ ^[1-9][0-9]*$ ]]; then
  printf -- '--max-batch must be a positive integer: %s\n' "$max_batch" >&2
  exit 2
fi
if [[ "$kind" == "rfdetr" ]] && ! [[ "$input_size" =~ ^[1-9][0-9]*$ ]]; then
  printf -- '--input-size must be a positive integer: %s\n' "$input_size" >&2
  exit 2
fi
if [[ -z "$opt_batch" ]]; then
  opt_batch="$max_batch"
elif ! [[ "$opt_batch" =~ ^[1-9][0-9]*$ ]]; then
  printf -- '--opt-batch must be a positive integer: %s\n' "$opt_batch" >&2
  exit 2
elif (( opt_batch > max_batch )); then
  printf -- '--opt-batch (%s) must not exceed --max-batch (%s)\n' "$opt_batch" "$max_batch" >&2
  exit 2
fi

# Only a dynamic batch dimension enables the profile path below. A model with
# dynamic spatial dimensions still needs a separate shape policy.
has_dynamic_batch=0
if "$python_bin" - "$onnx" <<'PY'
import sys
import onnx

model = onnx.load(sys.argv[1])
dynamic = any(
    len(value.type.tensor_type.shape.dim) > 0 and
    (value.type.tensor_type.shape.dim[0].dim_param or
     value.type.tensor_type.shape.dim[0].dim_value < 0)
    for value in model.graph.input
)
raise SystemExit(0 if dynamic else 1)
PY
then
  has_dynamic_batch=1
fi

# TensorRT 11 builds the precision the graph carries, so fp16 has to be baked
# into the ONNX first. The converter keeps graph inputs/outputs float32 because
# inference::IInferenceEngine only accepts float32 host buffers.
if [[ "$fp16" -eq 1 ]]; then
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  fp16_dir="$(mktemp -d -t fp16-onnx-XXXXXX)"
  fp16_onnx="$fp16_dir/model-fp16.onnx"
  trap 'rm -rf "$fp16_dir"' EXIT
  printf 'converting to float16 weights: %s -> %s\n' "$onnx" "$fp16_onnx"
  "$python_bin" "$script_dir/convert_fp16_onnx.py" \
    --input "$onnx" --output "$fp16_onnx" --io-precision fp32
  onnx="$fp16_onnx"
fi

mkdir -p "$(dirname "$engine")"
common=(
  --onnx="$onnx"
  --saveEngine="$engine"
  --skipInference
  --builderOptimizationLevel=3
  --noTF32
)

case "$kind" in
  hrnet)
    if [[ "$has_dynamic_batch" -eq 1 ]]; then
      common+=(
        --minShapes=images:1x3x384x288
        --optShapes=images:"${opt_batch}"x3x384x288
        --maxShapes=images:"${max_batch}"x3x384x288
      )
    fi
    ;;
  rfdetr)
    if [[ "$has_dynamic_batch" -eq 1 ]]; then
      common+=(
        --minShapes=image:1x3x"${input_size}"x"${input_size}"
        --optShapes=image:"${opt_batch}"x3x"${input_size}"x"${input_size}"
        --maxShapes=image:"${max_batch}"x3x"${input_size}"x"${input_size}"
      )
    fi
    ;;
  *)
    printf 'unsupported model kind: %s\n' "$kind" >&2
    usage
    exit 2
    ;;
esac

printf 'building TensorRT engine: %s -> %s\n' "$onnx" "$engine"
trtexec "${common[@]}"
printf 'engine ready: %s\n' "$engine"

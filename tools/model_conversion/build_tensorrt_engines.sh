#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf '%s\n' \
    "Usage:" \
    "  $0 hrnet --onnx PATH --engine PATH [--max-batch N]" \
    "  $0 rfdetr --onnx PATH --engine PATH [--input-size N] [--max-batch N]"
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
input_size=512

while [[ $# -gt 0 ]]; do
  case "$1" in
    --onnx) onnx="$2"; shift 2 ;;
    --engine) engine="$2"; shift 2 ;;
    --max-batch) max_batch="$2"; shift 2 ;;
    --input-size) input_size="$2"; shift 2 ;;
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

# Only a dynamic batch dimension enables the profile path below. A model with
# dynamic spatial dimensions still needs a separate shape policy.
has_dynamic_batch=0
if python3 - "$onnx" <<'PY'
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
        --optShapes=images:4x3x384x288
        --maxShapes=images:"${max_batch}"x3x384x288
      )
    fi
    ;;
  rfdetr)
    if [[ "$has_dynamic_batch" -eq 1 ]]; then
      common+=(
        --minShapes=image:1x3x"${input_size}"x"${input_size}"
        --optShapes=image:1x3x"${input_size}"x"${input_size}"
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

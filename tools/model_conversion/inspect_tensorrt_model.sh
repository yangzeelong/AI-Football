#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf '%s\n' "Usage: $0 --onnx PATH [--engine PATH]"
}

onnx=""
engine=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --onnx) onnx="$2"; shift 2 ;;
    --engine) engine="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'unknown argument: %s\n' "$1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$onnx" && -z "$engine" ]]; then
  usage
  exit 2
fi

if [[ -n "$onnx" ]]; then
  [[ -f "$onnx" ]] || { printf 'ONNX file not found: %s\n' "$onnx" >&2; exit 1; }
  printf '%s\n' "-- ONNX parser inspection --"
  trtexec --onnx="$onnx" --skipInference --verbose 2>&1 \
    | grep -E 'Input|Output|Network Description|Layers|error|Error|ERROR' || true
fi

if [[ -n "$engine" ]]; then
  [[ -f "$engine" ]] || { printf 'engine file not found: %s\n' "$engine" >&2; exit 1; }
  printf '%s\n' "-- TensorRT engine inspection --"
  trtexec --loadEngine="$engine" --skipInference --verbose 2>&1 \
    | grep -E 'Input|Output|Network Description|Layers|error|Error|ERROR' || true
fi

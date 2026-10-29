#!/usr/bin/env bash
# Build the supplied ONNX models on this machine; never download or upload.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$root/env.sh"
kind=${1:-all}
workspace_mb=${SIO_TRT_WORKSPACE_MB:-3072}
optimization_level=${SIO_TRT_OPT_LEVEL:-0}
case "$kind" in disp|raft|all) ;; *) echo 'Usage: bash scripts/build_models.sh [disp|raft|all]' >&2; exit 2;; esac
mkdir -p "$root/models/tensorrt/logs"
build_model() {
  local name=$1 source_model=$2 output_plan=$3
  shift 3
  local output="$root/models/tensorrt/$output_plan"
  local partial="$output.building"
  local log="$root/models/tensorrt/logs/$name-build.log"
  local checksum
  checksum=$(sha256sum "$root/models/$source_model" | cut -d ' ' -f 1)
  local recipe
  recipe=$(printf '%s\n' "$checksum" "$workspace_mb" "$optimization_level" \
    "$(sha256sum "$root/scripts/build_models.sh")" "$(sha256sum "$(command -v trtexec)")" "$@" | sha256sum | cut -d ' ' -f 1)
  if [[ -s "$output" && -f "$output.build-key" && $(cat "$output.build-key") == "$recipe" ]]; then
    echo "Already built: $output (same ONNX SHA256)"
    return
  fi
  echo "Building $source_model -> $output_plan; log: $log"
  trtexec --onnx="$root/models/$source_model" --saveEngine="$partial" \
    --fp16 --inputIOFormats=fp32:chw --outputIOFormats=fp32:chw \
    --memPoolSize="workspace:$workspace_mb" --builderOptimizationLevel="$optimization_level" \
    --timingCacheFile="$root/models/tensorrt/$name.timing-cache" \
    --skipInference --profilingVerbosity=detailed --verbose \
    --exportLayerInfo="$root/models/tensorrt/$name-layers.json" "$@" > "$log" 2>&1 || {
      tail -25 "$log" | cut -c 1-400 >&2
      return 1
    }
  test -s "$partial"
  mv -- "$partial" "$output"
  printf '%s\n' "$checksum" > "$output.source-sha256"
  printf '%s\n' "$recipe" > "$output.build-key"
  echo "Built: $output"
}
if [[ "$kind" == disp || "$kind" == all ]]; then
  disp_source=${SIO_DISP_ONNX:-onnx/disp_cost_volume.onnx}
  if [[ "$disp_source" == onnx/disp_cost_volume.onnx ]]; then
    model_python="$root/build/venv/model-tools/bin/python"
    if [[ ! -x "$model_python" ]]; then
      echo 'Missing build/venv/model-tools; see requirements-model-tools.txt and docs/LOCAL_SETUP.md' >&2
      exit 1
    fi
    "$model_python" "$root/scripts/prepare_disp_onnx.py"
  fi
  disp_shapes=()
  if [[ "$disp_source" == disp.onnx ]]; then
    disp_shapes=(--minShapes=left:1x3x448x896,right:1x3x448x896
      --optShapes=left:1x3x448x896,right:1x3x448x896
      --maxShapes=left:1x3x448x896,right:1x3x448x896)
  fi
  build_model disp "$disp_source" disp_896x448_fp16.plan "${disp_shapes[@]}"
fi
if [[ "$kind" == raft || "$kind" == all ]]; then
  build_model raft raft_large_896x448_iter6.onnx raft_large_896x448_iter6.plan
fi

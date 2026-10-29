#!/usr/bin/env bash
set -eo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$root/env.sh"
set -u
mode=${1:-gpu}
if [[ "$mode" != cpu && "$mode" != gpu ]]; then
  echo "Usage: bash scripts/check_env.sh [cpu|gpu]" >&2
  exit 2
fi
missing=0
check_file() {
  if [[ -f "$2" ]]; then
    printf '[OK] %s: %s\n' "$1" "$2"
  else
    printf '[MISSING] %s: %s\n' "$1" "$2"
    missing=$((missing + 1))
  fi
}

printf 'Workspace: %s\nROS: %s\n' "$SIO_WORKSPACE" "$ROS_DISTRO"
for tool in cmake g++ ninja colcon; do
  if command -v "$tool" >/dev/null; then
    printf '[OK] %s: %s\n' "$tool" "$(command -v "$tool")"
  else
    printf '[MISSING] %s\n' "$tool"
    missing=$((missing + 1))
  fi
done
check_file small_gicp "$SIO_DEPS_PREFIX/include/small_gicp/registration/registration_helper.hpp"
if [[ "$mode" == gpu ]]; then
  if command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; then
    echo '[OK] NVIDIA driver responds'
  else
    echo '[MISSING] Working NVIDIA driver; install system dependencies and reboot'
    missing=$((missing + 1))
  fi
  check_file nvcc "$CUDACXX"
  check_file TensorRT /usr/include/x86_64-linux-gnu/NvInfer.h
  check_file OpenCV-CUDA "$SIO_DEPS_PREFIX/include/opencv4/opencv2/cudaarithm.hpp"
  check_file LibTorch "$SIO_DEPS_PREFIX/libtorch/share/cmake/Torch/TorchConfig.cmake"
  check_file Rasterizer "$root/build/dependencies/build/rasterizer/libCudaRasterizer.a"
fi

# Runtime assets are deliberately separate from compilation prerequisites.
python3 - "$root" <<'PY'
from pathlib import Path
import sys
import yaml
root = Path(sys.argv[1])
config = yaml.safe_load((root / 'project_config.yaml').read_text()) or {}
local = root / 'project_config.local.yaml'
if local.exists():
    config.update(yaml.safe_load(local.read_text()) or {})
for key in ('dataset_dir', 'imu_file', 'plan_disp', 'plan_raft'):
    path = root / config[key]
    exists = path.is_dir() if key == 'dataset_dir' else path.is_file()
    label = 'PRESENT' if exists else 'NOT PROVIDED'
    print(f'[{label}] {key}: {path}')
PY
if (( missing > 0 )); then
  printf '%d compilation/driver prerequisites are missing.\n' "$missing"
  exit 1
fi
echo 'Compilation prerequisites found. Model/data availability is listed separately.'

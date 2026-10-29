#!/usr/bin/env bash
# Run from a local terminal: bash scripts/install_system_deps.sh
set -euo pipefail

mode=${1:-all}
case "$mode" in
  all|--gpu-packages-only|--simulate-gpu) ;;
  *) echo "Usage: bash scripts/install_system_deps.sh [--gpu-packages-only|--simulate-gpu]" >&2; exit 2 ;;
esac
source /etc/os-release
if [[ "$ID" != ubuntu || "$VERSION_ID" != 22.04 || "$(uname -m)" != x86_64 ]]; then
  echo "This installer targets Ubuntu 22.04 x86_64." >&2
  exit 1
fi

trt_version=10.14.1.48-1+cuda12.9
# APT does not automatically choose older transitive dependency versions.
# Explicitly select the complete TensorRT package family, including headers.
trt_packages=(
  libnvinfer-headers-dev libnvinfer-headers-plugin-dev
  libnvinfer10 libnvinfer-dev libnvinfer-lean10 libnvinfer-dispatch10
  libnvinfer-plugin10 libnvinfer-plugin-dev libnvinfer-vc-plugin10
  libnvonnxparsers10 libnvonnxparsers-dev libnvinfer-bin
)
gpu_packages=(cuda-toolkit-12-9)
for package in "${trt_packages[@]}"; do
  gpu_packages+=("$package=$trt_version")
done

# Dependency resolution only: no administrator access or system writes.
if [[ "$mode" == --simulate-gpu ]]; then
  exec apt-get -s --no-remove --no-install-recommends install "${gpu_packages[@]}"
fi
if (( EUID != 0 )); then
  exec sudo bash "$0" "$@"
fi

if [[ "$mode" == all ]]; then
apt-get update
apt-get install -y --no-remove --no-install-recommends \
  build-essential cmake ninja-build git curl ca-certificates unzip pkg-config clangd clang-format \
  ubuntu-drivers-common "linux-headers-$(uname -r)" \
  libeigen3-dev libyaml-cpp-dev libpcl-dev libopencv-dev libgtest-dev \
  libgtk-3-dev libjpeg-dev libpng-dev libtiff-dev libtbb-dev \
  libavcodec-dev libavformat-dev libswscale-dev \
  python3-colcon-common-extensions ros-humble-pcl-conversions \
  ros-humble-ament-cmake-gtest

# Use Ubuntu's recommended driver, including its Secure Boot integration.
if ! nvidia-smi >/dev/null 2>&1; then
  ubuntu-drivers install
fi

# NVIDIA's signed Ubuntu 22.04 repository. Install the toolkit separately
# from the driver so that the toolkit cannot select a different driver branch.
keyring_deb=$(mktemp --suffix=.deb /tmp/sio-cuda-keyring.XXXXXX)
curl -fL --retry 3 \
  https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
  -o "$keyring_deb"
dpkg -i "$keyring_deb"
fi
apt-get update

apt-get install -y --no-remove --no-install-recommends \
  "${gpu_packages[@]}"

echo "System packages installed. Save your work and reboot before GPU verification."
echo "If Secure Boot requests MOK enrollment, complete it during reboot."
echo "Then run, as your normal user: bash scripts/build_deps.sh gpu"

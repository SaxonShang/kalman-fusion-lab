#!/usr/bin/env bash
# Local dependencies only; never run this script with sudo.
set -eo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$root/env.sh"
set -u
mode=${1:-cpu}
if [[ "$mode" != cpu && "$mode" != gpu ]]; then
  echo "Usage: bash scripts/build_deps.sh [cpu|gpu]" >&2
  exit 2
fi
if (( EUID == 0 )); then
  echo "Run as your normal user, without sudo." >&2
  exit 1
fi
jobs=${SIO_BUILD_JOBS:-4}
deps="$root/build/dependencies"
mkdir -p "$deps/src" "$deps/build" "$deps/downloads" "$SIO_DEPS_PREFIX"

fetch_source() {
  local name=$1 url=$2
  if [[ ! -f "$deps/src/$name/README.md" ]]; then
    curl -fL --retry 3 --connect-timeout 30 "$url" -o "$deps/downloads/$name.tar.gz"
    mkdir -p "$deps/src/$name"
    tar -xzf "$deps/downloads/$name.tar.gz" -C "$deps/src/$name" --strip-components=1
  fi
}

fetch_source small_gicp \
  https://github.com/koide3/small_gicp/archive/fa0cfc983417c044b6036f71dfa448038bb9c820.tar.gz
cmake -S "$deps/src/small_gicp" -B "$deps/build/small_gicp" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$SIO_DEPS_PREFIX" \
  -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_WITH_MARCH_NATIVE=OFF
cmake --build "$deps/build/small_gicp" --parallel "$jobs"
cmake --install "$deps/build/small_gicp"

if [[ "$mode" == cpu ]]; then
  echo "small_gicp installed locally. Next: bash scripts/build.sh cpu"
  exit 0
fi

if [[ ! -x "$CUDACXX" || ! -f /usr/include/x86_64-linux-gnu/NvInfer.h ]]; then
  echo "CUDA 12.9 / TensorRT missing. First run bash scripts/install_system_deps.sh" >&2
  exit 1
fi

# Build CUDA OpenCV without replacing the distribution's OpenCV.
fetch_source opencv-4.12.0 https://github.com/opencv/opencv/archive/refs/tags/4.12.0.tar.gz
fetch_source opencv_contrib-4.12.0 https://github.com/opencv/opencv_contrib/archive/refs/tags/4.12.0.tar.gz
cmake -S "$deps/src/opencv-4.12.0" -B "$deps/build/opencv" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$SIO_DEPS_PREFIX" \
  -DCUDA_TOOLKIT_ROOT_DIR="$CUDA_HOME" -DWITH_CUDA=ON \
  -DCUDA_ARCH_BIN=8.9 -DCUDA_ARCH_PTX= -DWITH_CUDNN=OFF -DWITH_CUBLAS=ON \
  -DOPENCV_EXTRA_MODULES_PATH="$deps/src/opencv_contrib-4.12.0/modules" \
  -DBUILD_LIST=cudev,core,imgproc,imgcodecs,highgui,videoio,calib3d,features2d,flann,cudaarithm,cudaimgproc,cudawarping \
  -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF \
  -DBUILD_opencv_python2=OFF -DBUILD_opencv_python3=OFF -DBUILD_JAVA=OFF \
  -DWITH_NVCUVID=OFF -DWITH_NVCUVENC=OFF -DWITH_OPENGL=OFF \
  -DWITH_GTK=ON -DWITH_QT=OFF -DOPENCV_GENERATE_PKGCONFIG=ON
cmake --build "$deps/build/opencv" --parallel "$jobs"
cmake --install "$deps/build/opencv"

# C++11 ABI matches Ubuntu's PCL/OpenCV/ROS libraries.
if [[ ! -f "$SIO_DEPS_PREFIX/libtorch/share/cmake/Torch/TorchConfig.cmake" ]]; then
  libtorch_archive="$deps/downloads/libtorch-2.8.0-cu129.zip"
  if [[ ! -f "$libtorch_archive" ]]; then
    curl -fL --retry 3 --connect-timeout 30 --continue-at - \
      'https://download.pytorch.org/libtorch/cu129/libtorch-shared-with-deps-2.8.0%2Bcu129.zip' \
      -o "$libtorch_archive.part"
    mv -n "$libtorch_archive.part" "$libtorch_archive"
  fi
  unzip -q -o "$libtorch_archive" -d "$SIO_DEPS_PREFIX"
fi

cmake -S "$root/submodules/diff-gaussian-rasterization-w-pose-main" \
  -B "$deps/build/rasterizer" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DCMAKE_CUDA_COMPILER="$CUDACXX" -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build "$deps/build/rasterizer" --parallel "$jobs"
echo "GPU dependencies built. Next: bash scripts/build.sh gpu"

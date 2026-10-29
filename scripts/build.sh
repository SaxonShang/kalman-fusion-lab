#!/usr/bin/env bash
set -eo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$root/env.sh"
set -u
mode=${1:-cpu}
jobs=${SIO_BUILD_JOBS:-4}
args=(-DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON -DSIO_ENABLE_ZED=OFF)
case "$mode" in
  cpu)
    args+=(-DSIO_BUILD_RUNTIME=OFF -DSIO_ENABLE_LIBTORCH=OFF)
    ;;
  gpu)
    for required in \
      "$CUDACXX" \
      "$SIO_DEPS_PREFIX/lib/cmake/opencv4/OpenCVConfig.cmake" \
      "$SIO_DEPS_PREFIX/libtorch/share/cmake/Torch/TorchConfig.cmake" \
      "$root/build/dependencies/build/rasterizer/libCudaRasterizer.a"; do
      if [[ ! -f "$required" ]]; then
        printf 'GPU dependency missing: %s\nRun bash scripts/build_deps.sh gpu first.\n' "$required" >&2
        exit 1
      fi
    done
    args+=(-DSIO_BUILD_RUNTIME=ON -DSIO_ENABLE_LIBTORCH=ON
      "-DCUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME"
      "-DCUDAToolkit_ROOT=$CUDA_HOME"
      "-DOpenCV_DIR=$SIO_DEPS_PREFIX/lib/cmake/opencv4"
      "-DTorch_DIR=$SIO_DEPS_PREFIX/libtorch/share/cmake/Torch"
      "-DSIO_CUDA_RASTERIZER_LIBRARY=$root/build/dependencies/build/rasterizer/libCudaRasterizer.a")
    ;;
  *) echo "Usage: bash scripts/build.sh [cpu|gpu]" >&2; exit 2 ;;
esac

# The repository contains a cache copied from another machine. Keep it intact;
# each local mode gets a fresh, separate build directory.
build_dir="$root/build/local/$mode"
cmake -S "$root" -B "$build_dir" -G Ninja \
  "-DCMAKE_INSTALL_PREFIX=$root/install/local" "${args[@]}"
cmake --build "$build_dir" --parallel "$jobs"
ctest --test-dir "$build_dir" --output-on-failure -R '^sio_unit_tests$'
cmake --install "$build_dir"
ln -sfn "build/local/$mode/compile_commands.json" "$root/compile_commands.json"
echo "Build and unit tests finished. Use: source $root/env.sh"

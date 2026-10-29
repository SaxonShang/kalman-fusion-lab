#!/usr/bin/env bash
# Source this file: source /path/to/SIO_SLAM/env.sh
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Use: source env.sh" >&2
  exit 1
fi

export SIO_WORKSPACE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -f /opt/ros/humble/setup.bash ]]; then
  echo "ROS 2 Humble is missing: /opt/ros/humble/setup.bash" >&2
  return 1
fi
# ROS setup scripts may read unset variables.
_sio_had_nounset=0
if [[ $- == *u* ]]; then
  _sio_had_nounset=1
  set +u
fi
source /opt/ros/humble/setup.bash
if (( _sio_had_nounset )); then set -u; fi
unset _sio_had_nounset

export CUDA_HOME="${SIO_CUDA_ROOT:-/usr/local/cuda-12.9}"
export CUDA_PATH="$CUDA_HOME"
export CUDACXX="$CUDA_HOME/bin/nvcc"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
export SIO_DEPS_PREFIX="$SIO_WORKSPACE/build/dependencies/install"
export CMAKE_PREFIX_PATH="$SIO_DEPS_PREFIX:$SIO_DEPS_PREFIX/libtorch${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
export PATH="$CUDA_HOME/bin:/usr/src/tensorrt/bin${PATH:+:$PATH}"
export LD_LIBRARY_PATH="$SIO_DEPS_PREFIX/lib:$SIO_DEPS_PREFIX/libtorch/lib:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PKG_CONFIG_PATH="$SIO_DEPS_PREFIX/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"

if [[ -d "$SIO_WORKSPACE/install/local/share/ament_index" ]]; then
  export AMENT_PREFIX_PATH="$SIO_WORKSPACE/install/local${AMENT_PREFIX_PATH:+:$AMENT_PREFIX_PATH}"
fi

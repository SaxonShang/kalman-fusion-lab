cmake_minimum_required(VERSION 3.10)
project(sio_slam)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING
      "Choose: Debug Release RelWithDebInfo MinSizeRel" FORCE)
endif()
set(CMAKE_CXX_FLAGS_DEBUG "-g3 -O0 -fno-omit-frame-pointer" CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-g -O2" CACHE STRING "" FORCE)

include(CTest)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(SIO_BUILD_RUNTIME "Build GPU runtime programs and the pipeline smoke test" ON)
option(SIO_ENABLE_LIBTORCH "Enable LibTorch backend optimization" OFF)
option(SIO_BUILD_FUSION_LAB "Build the offline CPU fusion experiment runner" OFF)

# ---- Profiling support ----
option(SIO_ENABLE_PROFILING "Enable frontend profiling instrumentation" OFF)
if(SIO_ENABLE_PROFILING)
  message(STATUS "SIO Profiling: ENABLED")
  add_compile_definitions(SIO_ENABLE_PROFILING)
else()
  message(STATUS "SIO Profiling: DISABLED")
endif()

set(SIO_SLAM_PROJECT_CONFIG_FILE "${CMAKE_CURRENT_SOURCE_DIR}/project_config.yaml")
if(NOT EXISTS "${SIO_SLAM_PROJECT_CONFIG_FILE}")
    message(FATAL_ERROR
        "Missing ${SIO_SLAM_PROJECT_CONFIG_FILE}. Please add the top-level project config.")
endif()

set(SIO_SLAM_LOCAL_PROJECT_CONFIG_FILE
    "${CMAKE_CURRENT_SOURCE_DIR}/project_config.local.yaml")
set(SIO_SLAM_LEGACY_TENSORRT_ROOT
    "/home/czzt/miniconda3/envs/foundation_stereo/TensorRT-10.8.0.43")

function(sio_load_yaml_scalar file_path key out_var)
    if(NOT EXISTS "${file_path}")
        set(${out_var} "" PARENT_SCOPE)
        return()
    endif()

    file(STRINGS "${file_path}" _yaml_lines REGEX "^[ \t]*${key}[ \t]*:")
    set(_value "")
    foreach(_line IN LISTS _yaml_lines)
        if(_line MATCHES "^[ \t]*${key}[ \t]*:[ \t]*(.+)[ \t]*$")
            set(_value "${CMAKE_MATCH_1}")
            string(REGEX REPLACE "[ \t]+#.*$" "" _value "${_value}")
            string(STRIP "${_value}" _value)
            string(REGEX REPLACE "^['\"](.*)['\"]$" "\\1" _value "${_value}")
            break()
        endif()
    endforeach()

    set(${out_var} "${_value}" PARENT_SCOPE)
endfunction()

sio_load_yaml_scalar("${SIO_SLAM_LOCAL_PROJECT_CONFIG_FILE}" "tensorrt_root"
                     SIO_LOCAL_TENSORRT_ROOT)
sio_load_yaml_scalar("${SIO_SLAM_LOCAL_PROJECT_CONFIG_FILE}" "nvinfer_include_dir"
                     SIO_LOCAL_NVINFER_INCLUDE_DIR)
sio_load_yaml_scalar("${SIO_SLAM_LOCAL_PROJECT_CONFIG_FILE}" "nvinfer_library_dir"
                     SIO_LOCAL_NVINFER_LIBRARY_DIR)

if(SIO_LOCAL_TENSORRT_ROOT OR SIO_LOCAL_NVINFER_INCLUDE_DIR OR SIO_LOCAL_NVINFER_LIBRARY_DIR)
    message(STATUS "Using TensorRT hints from project_config.local.yaml")
endif()

# 1. 查找依赖包
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenMP)
if(OpenMP_FOUND)
  set(SIO_OPENMP_LIBS OpenMP::OpenMP_CXX)
else()
  set(SIO_OPENMP_LIBS "")
  message(STATUS "OpenMP not found — ColorPointObserver will run serially")
endif()
find_package(small_gicp QUIET)
# ---- 查找 yaml-cpp（手动方式，因为 find_package 可能找不到配置文件） ----
find_path(YAML_CPP_INCLUDE_DIR
    NAMES yaml-cpp/yaml.h
    PATHS /usr/include /usr/local/include /usr/include/x86_64-linux-gnu
)

find_library(YAML_CPP_LIBRARY
    NAMES yaml-cpp
    PATHS /usr/lib /usr/lib/x86_64-linux-gnu /usr/local/lib
)

if(NOT YAML_CPP_INCLUDE_DIR OR NOT YAML_CPP_LIBRARY)
    message(FATAL_ERROR "yaml-cpp not found. Please install libyaml-cpp-dev.")
endif()

message(STATUS "Found yaml-cpp: include=${YAML_CPP_INCLUDE_DIR}, lib=${YAML_CPP_LIBRARY}")

# 2. 关键：将当前项目的 include 文件夹添加到头文件搜索路径中
# PUBLIC 意味着依赖这个目标的其他目标也能使用这些头文件
include_directories(
    include
    ${PCL_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
)

# 2.1 TensorRT / CUDA 依赖（DynamicCloudProcessor + TrtEngine）
if(SIO_BUILD_RUNTIME)
set(CUDA_TOOLKIT_ROOT_DIR "" CACHE PATH "CUDA toolkit root directory")
find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS
    core imgproc imgcodecs highgui cudaarithm cudaimgproc cudawarping)
set(TENSORRT_ROOT_HINT "" CACHE PATH "TensorRT root path (contains include/ and lib/)")
set(NVINFER_INCLUDE_HINT "" CACHE PATH "Directory containing NvInfer.h")
set(NVINFER_LIBRARY_HINT "" CACHE PATH "Directory containing libnvinfer")

set(_TENSORRT_HINTS
    ${NVINFER_INCLUDE_HINT}
    ${NVINFER_LIBRARY_HINT}
    ${TENSORRT_ROOT_HINT}
    ${SIO_LOCAL_NVINFER_INCLUDE_DIR}
    ${SIO_LOCAL_NVINFER_LIBRARY_DIR}
    ${SIO_LOCAL_TENSORRT_ROOT}
    $ENV{TENSORRT_ROOT}
    $ENV{TRT_ROOT}
    ${SIO_SLAM_LEGACY_TENSORRT_ROOT}
    /usr/local/TensorRT
)

find_path(NVINFER_INCLUDE_DIR
    NAMES NvInfer.h
    HINTS
        ${_TENSORRT_HINTS}
    PATHS
        /usr/include
        /usr/include/x86_64-linux-gnu
        /usr/local/include
        /usr/local/TensorRT/include
    PATH_SUFFIXES
        include
)

find_library(NVINFER_LIBRARY
    NAMES nvinfer
    HINTS
        ${_TENSORRT_HINTS}
    PATHS
        /usr/lib
        /usr/lib/x86_64-linux-gnu
        /usr/local/lib
        /usr/local/lib64
        /usr/local/TensorRT/lib
    PATH_SUFFIXES
        lib
        lib64
)

find_library(NVINFER_PLUGIN_LIBRARY
    NAMES nvinfer_plugin
    HINTS
        ${_TENSORRT_HINTS}
    PATHS
        /usr/lib
        /usr/lib/x86_64-linux-gnu
        /lib
        /lib/x86_64-linux-gnu
        /usr/local/lib
        /usr/local/lib64
        /usr/local/TensorRT/lib
    PATH_SUFFIXES
        lib
        lib64
)

if(CUDAToolkit_FOUND)
    set(CUDART_LINK_TARGET CUDA::cudart)
    set(CUDA_INCLUDE_DIR_FOR_TRT ${CUDAToolkit_INCLUDE_DIRS})
else()
    set(_CUDA_HINTS ${CUDA_TOOLKIT_ROOT_DIR} $ENV{CUDA_ROOT} $ENV{CUDA_PATH})
    find_path(CUDA_RUNTIME_INCLUDE_DIR
        NAMES cuda_runtime_api.h
        HINTS ${_CUDA_HINTS}
        PATHS
            /usr/local/cuda/include
            /usr/local/cuda/targets/x86_64-linux/include
            /usr/include
        PATH_SUFFIXES include
    )

    find_library(CUDART_LIBRARY
        NAMES cudart
        HINTS ${_CUDA_HINTS}
        PATHS
            /usr/local/cuda/lib64
            /usr/local/cuda/targets/x86_64-linux/lib
            /usr/lib/x86_64-linux-gnu
            /usr/lib
        PATH_SUFFIXES lib lib64
    )

    set(CUDART_LINK_TARGET ${CUDART_LIBRARY})
    set(CUDA_INCLUDE_DIR_FOR_TRT ${CUDA_RUNTIME_INCLUDE_DIR})
endif()

if(NOT NVINFER_INCLUDE_DIR)
    message(FATAL_ERROR
        "NvInfer.h not found. Set -DTENSORRT_ROOT_HINT=<TensorRTRoot> or -DNVINFER_INCLUDE_HINT=<include_dir>.")
endif()

if(NOT NVINFER_LIBRARY)
    message(FATAL_ERROR
        "libnvinfer not found. Set -DTENSORRT_ROOT_HINT=<TensorRTRoot> or -DNVINFER_LIBRARY_HINT=<lib_dir>.")
endif()

if(NOT CUDART_LINK_TARGET)
    message(FATAL_ERROR
        "CUDA runtime library (cudart) not found. Please install CUDA toolkit/runtime.")
endif()
endif() # SIO_BUILD_RUNTIME

# Color ICP 依赖 upstream 的 mini-yaml 实现：include/color_icp/yaml.h + src/yaml.cpp
set(COLOR_ICP_YAML_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/include/color_icp/yaml.h")
set(COLOR_ICP_YAML_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/yaml.cpp")

if(NOT EXISTS "${COLOR_ICP_YAML_HEADER}")
    message(FATAL_ERROR
        "Missing ${COLOR_ICP_YAML_HEADER}. Please copy include/color_icp/yaml.h from hanzheteng/color_icp.")
endif()

if(NOT EXISTS "${COLOR_ICP_YAML_SOURCE}")
    message(FATAL_ERROR
        "Missing ${COLOR_ICP_YAML_SOURCE}. Please copy src/yaml.cpp from hanzheteng/color_icp.")
endif()

# 3. 源文件分组：单测避开 TensorRT 运行链路，demo/集成测试使用完整链路
set(SIO_LIGHT_SRC_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/dataset_loader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/feature_processor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/fusion_debug_logger.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/image_preprocessor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/imu_debug_logger.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/imu_integator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/imu_preprocessor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/keyframe_manager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/profiler/simple_profiler.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/state_estimater.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/stereo_gicp.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/modules/registration/src/stereo_gicp.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sio_config.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sio_stereo_geometry.cpp
)

set(SIO_RUNTIME_SRC_FILES
    ${SIO_LIGHT_SRC_FILES}
    ${CMAKE_CURRENT_SOURCE_DIR}/modules/registration/src/color_icp.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/dynamic_cloud_processor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/frontend_debug_visualizer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/yaml.cpp
)

set(SIO_FRONTEND_SRC_FILES
    ${SIO_RUNTIME_SRC_FILES}
    ${CMAKE_CURRENT_SOURCE_DIR}/src/frontend.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/backend.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gaussian_map_manager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pose_3dgs_optimizer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/torch_pose_optimize.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/learning_rate_scheduler_factory.cpp
)

# small_gicp 链接目标兼容：优先 imported target，其次裸目标名，最后库文件
set(SMALL_GICP_LINK_TARGET "")
if(TARGET small_gicp::small_gicp)
    set(SMALL_GICP_LINK_TARGET small_gicp::small_gicp)
elseif(TARGET small_gicp)
    set(SMALL_GICP_LINK_TARGET small_gicp)
else()
    find_library(SMALL_GICP_LIBRARY NAMES small_gicp)
    if(SMALL_GICP_LIBRARY)
        set(SMALL_GICP_LINK_TARGET ${SMALL_GICP_LIBRARY})
    endif()
endif()

if(NOT SMALL_GICP_LINK_TARGET)
    message(FATAL_ERROR
        "small_gicp library not found for linking. Install small_gicp dev package or set library path.")
endif()

# ---- ZED SDK (optional, for ZED 2i live capture) ----
option(SIO_ENABLE_ZED "Enable ZED 2i live capture in DatasetLoader" ON)
set(SIO_ZED_INCLUDE_DIR "" CACHE PATH "ZED SDK include directory")
set(SIO_ZED_LIBRARY_DIR "" CACHE PATH "ZED SDK library directory")

if(SIO_ENABLE_ZED AND SIO_BUILD_RUNTIME)
    find_path(SIO_ZED_INCLUDE_DIR
        NAMES sl/Camera.hpp
        HINTS $ENV{ZED_SDK_ROOT} ${SIO_ZED_INCLUDE_DIR}
        PATHS /usr/local/zed/include /usr/include
        PATH_SUFFIXES include)
    find_library(SIO_ZED_LIBRARY
        NAMES sl_zed
        HINTS $ENV{ZED_SDK_ROOT} ${SIO_ZED_LIBRARY_DIR}
        PATHS /usr/local/zed/lib /usr/lib /usr/lib/x86_64-linux-gnu
        PATH_SUFFIXES lib lib/x86_64-linux-gnu)

    if(SIO_ZED_INCLUDE_DIR AND SIO_ZED_LIBRARY)
        message(STATUS "ZED SDK found: include=${SIO_ZED_INCLUDE_DIR}, lib=${SIO_ZED_LIBRARY}")
        set(SIO_ZED_FOUND TRUE)
    else()
        message(STATUS "ZED SDK not found — DatasetLoader ZED2i mode disabled")
        set(SIO_ZED_FOUND FALSE)
    endif()
else()
    set(SIO_ZED_FOUND FALSE)
endif()

function(sio_apply_zed_support target_name)
    if(SIO_ZED_FOUND)
        target_compile_definitions(${target_name} PRIVATE SIO_ENABLE_ZED)
        target_include_directories(${target_name} PRIVATE
            ${SIO_ZED_INCLUDE_DIR}
            ${CUDA_INCLUDE_DIR_FOR_TRT}
        )
        target_link_libraries(${target_name} ${SIO_ZED_LIBRARY})
    endif()
endfunction()

function(sio_configure_light_target target_name)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${target_name} PRIVATE -Wno-deprecated-declarations)
    endif()

    target_include_directories(${target_name} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/modules/registration/include
        ${CMAKE_CURRENT_SOURCE_DIR}/tests
        ${YAML_CPP_INCLUDE_DIR}
    )

    target_compile_definitions(${target_name} PRIVATE
        SIO_SLAM_PROJECT_CONFIG_PATH="${SIO_SLAM_PROJECT_CONFIG_FILE}"
    )

    target_link_libraries(${target_name}
        ${OpenCV_LIBS}
        ${PCL_LIBRARIES}
        ${SIO_OPENMP_LIBS}
        ${YAML_CPP_LIBRARY}
        ${SMALL_GICP_LINK_TARGET}
    )

    ament_target_dependencies(${target_name}
        rclcpp
        sensor_msgs
        geometry_msgs
        pcl_conversions
    )

    sio_apply_zed_support(${target_name})
endfunction()

function(sio_configure_runtime_target target_name)
    sio_configure_light_target(${target_name})

    target_include_directories(${target_name} PRIVATE
        ${NVINFER_INCLUDE_DIR}
        ${CUDA_INCLUDE_DIR_FOR_TRT}
    )

    target_link_libraries(${target_name}
        ${SMALL_GICP_LINK_TARGET}
        ${NVINFER_LIBRARY}
        ${CUDART_LINK_TARGET}
    )

    if(NVINFER_PLUGIN_LIBRARY)
        target_link_libraries(${target_name} ${NVINFER_PLUGIN_LIBRARY})
    endif()
endfunction()

# 设置 diff-gaussian-rasterization 的根目录（注意路径末尾不要有多余斜杠）
if(SIO_BUILD_RUNTIME)
set(DIFF_GS_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/submodules/diff-gaussian-rasterization-w-pose-main)

# 指定静态库的完整路径
set(SIO_CUDA_RASTERIZER_LIBRARY "${DIFF_GS_ROOT}/build/libCudaRasterizer.a"
    CACHE FILEPATH "Prebuilt CUDA rasterizer static library")
if(NOT EXISTS "${SIO_CUDA_RASTERIZER_LIBRARY}")
    message(FATAL_ERROR
        "CUDA rasterizer missing: ${SIO_CUDA_RASTERIZER_LIBRARY}. Run bash scripts/build_deps.sh gpu first.")
endif()

# 头文件搜索路径（cuda_rasterizer 和 third_party/glm）
set(CUDA_RASTERIZER_INCLUDES
    ${DIFF_GS_ROOT}/cuda_rasterizer
    ${DIFF_GS_ROOT}/third_party/glm
)
# 4. 日常运行入口
add_executable(run_frontend
    ${SIO_FRONTEND_SRC_FILES}
    ${CMAKE_CURRENT_SOURCE_DIR}/apps/run_frontend.cpp
)
target_link_libraries(run_frontend ${SIO_CUDA_RASTERIZER_LIBRARY})
sio_configure_runtime_target(run_frontend)

if(SIO_ENABLE_LIBTORCH)
    find_package(Torch REQUIRED)
    target_compile_definitions(run_frontend PRIVATE SIO_USE_LIBTORCH)
    separate_arguments(SIO_TORCH_CXX_FLAGS NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
    target_compile_options(run_frontend PRIVATE ${SIO_TORCH_CXX_FLAGS})
    target_link_libraries(run_frontend ${TORCH_LIBRARIES})
else()
    message(WARNING "LibTorch is disabled; backend joint optimization is unavailable.")
endif()

# 3. 添加头文件目录
target_include_directories(run_frontend PRIVATE
    ${CUDA_RASTERIZER_INCLUDES}
    # 如果你之前还有别的 include 路径，可以继续加在这里
)

# 4. 强制要求 C++17（CUDA 光栅化器需要）
set_target_properties(run_frontend PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
)

add_executable(run_demo
    ${SIO_RUNTIME_SRC_FILES}
    ${CMAKE_CURRENT_SOURCE_DIR}/apps/run_demo.cpp
)
sio_configure_runtime_target(run_demo)

install(TARGETS 
    run_frontend
    run_demo
    DESTINATION lib/${PROJECT_NAME}
)
add_executable(sio_export_disparity
    ${CMAKE_CURRENT_SOURCE_DIR}/apps/export_disparity.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/dataset_loader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/image_preprocessor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sio_config.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sio_stereo_geometry.cpp
)
sio_configure_runtime_target(sio_export_disparity)
endif() # SIO_BUILD_RUNTIME

if(SIO_BUILD_FUSION_LAB)
    add_executable(sio_fusion_lab
        ${SIO_LIGHT_SRC_FILES}
        ${CMAKE_CURRENT_SOURCE_DIR}/modules/registration/src/color_icp.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/yaml.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/experiments/fusion/src/main.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/experiments/fusion/src/lab_io.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/experiments/fusion/src/lab_prepare.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/experiments/fusion/src/lab_run.cpp
    )
    sio_configure_light_target(sio_fusion_lab)
endif()

# 5. 自动测试：轻依赖单测 + 短集成测试
if(BUILD_TESTING)
    find_package(ament_cmake_gtest REQUIRED)

    ament_add_gtest(sio_unit_tests
        ${SIO_LIGHT_SRC_FILES}
        ${CMAKE_CURRENT_SOURCE_DIR}/modules/registration/src/color_icp.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/modules/registration/tests/test_stereo_gicp.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/unit/test_color_icp_config.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/unit/test_utils.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/unit/test_imu_integrator.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/unit/test_keyframe_manager.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/unit/test_feature_processor.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/unit/test_state_estimator.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/unit/test_image_preprocessor.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/unit/test_stereo_geometry.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/unit/test_stereo_gicp.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/unit/test_geo_flow_error.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/unit/test_imu_preprocessor.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/unit/test_debug_loggers.cpp
    )
    if(TARGET sio_unit_tests)
        sio_configure_light_target(sio_unit_tests)
    endif()

    if(SIO_BUILD_RUNTIME)
    add_executable(pipeline_smoke_test
        ${SIO_RUNTIME_SRC_FILES}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/integration/pipeline_smoke_test.cpp
    )
    sio_configure_runtime_target(pipeline_smoke_test)
    add_test(
        NAME pipeline_smoke_test
        COMMAND pipeline_smoke_test --ros-args -p max_frames:=5 -p run_color_icp:=false
    )
    set_tests_properties(pipeline_smoke_test PROPERTIES TIMEOUT 420)
    endif()
endif()

ament_package()

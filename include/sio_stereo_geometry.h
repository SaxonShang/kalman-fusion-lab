#pragma once

#include "camera_intrinsics.h"

#include <opencv2/core.hpp>

// 双目几何
namespace sio_utils {

// 原始采集分辨率 → TensorRT 推理分辨率（中心裁减 + 缩放）默认是1280x720 → 896x448
struct StereoImageGeometry {
  int source_width = 1280;
  int source_height = 720;
  int inference_width = 896;
  int inference_height = 448;
};
// 中心裁减区域
struct StereoCropRect {
  int x = 0;
  int y = 0;
  int width = 0;
  int height = 0;
};
// 计算中心裁减区域
// 输入：
// - geom：立体几何
// 输出：
// - 中心裁减区域
StereoCropRect ComputeCenterCropRect(const StereoImageGeometry &geom);
// 是否需要立体预处理
// 输入：
// - geom：立体几何
// 输出：
// - 是否需要立体预处理
bool StereoPreprocessRequired(const StereoImageGeometry &geom);
// 调整内参用于裁减
// 输入：
// - source_cam：原始内参
// - geom：立体几何
// 输出：
// - 调整后的内参
CameraIntrinsics AdjustIntrinsicsForCrop(const CameraIntrinsics &source_cam,
                                         const StereoImageGeometry &geom);
// 调整内参用于推理
// 输入：
// - source_cam：原始内参
// - geom：立体几何
// 输出：
// - 调整后的内参
CameraIntrinsics AdjustIntrinsicsForInference(const CameraIntrinsics &source_cam,
                                              const StereoImageGeometry &geom);

// 中心裁减
// 输入：
// - image：输入图像
// - crop：裁减区域
// 输出：
// - 中心裁减后的图像
cv::Mat CenterCrop(const cv::Mat &image, const StereoCropRect &crop);
// 中心裁减并缩放
// 输入：
// - image：输入图像
// - crop：裁减区域
// - out_width：输出宽度
// - out_height：输出高度
// 输出：
// - 中心裁减并缩放后的图像
cv::Mat CenterCropAndResize(const cv::Mat &image, const StereoCropRect &crop, int out_width,
                            int out_height);

// 视差图转深度（米）
// 输入：
// - disp：视差图
// - cam：相机内参
// - min_depth：最小深度
// - max_depth：最大深度
// 输出：
// - 深度图
cv::Mat DisparityToDepthMeters(const cv::Mat &disp, const CameraIntrinsics &cam, float min_depth,
                               float max_depth);
// 推理分辨率视差 → 裁减未缩放分辨率深度（米），供后端 3DGS 与 obs_image 对齐
// 输入：
// - disp_inference：推理视差图
// - inference_cam：推理相机内参
// - geom：立体几何
// - min_depth：最小深度
// - max_depth：最大深度
// 输出：
// - 深度图
cv::Mat DisparityToDepthAtCropResolution(const cv::Mat &disp_inference,
                                         const CameraIntrinsics &inference_cam,
                                         const StereoImageGeometry &geom, float min_depth,
                                         float max_depth);

// 将输入图像缩放到 geom.source_width × source_height（如 1080p → 720p）
bool FitImageToSourceGeometry(cv::Mat &image, const StereoImageGeometry &geom);

} // namespace sio_utils

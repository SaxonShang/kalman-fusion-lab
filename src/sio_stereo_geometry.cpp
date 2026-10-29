#include "sio_stereo_geometry.h"

#include <algorithm>
#include <cmath>
#include <opencv2/imgproc.hpp>

namespace sio_utils {

//根据双目几何计算中心裁减区域
StereoCropRect ComputeCenterCropRect(const StereoImageGeometry &geom) {
  StereoCropRect crop{};
  if (geom.source_width <= 0 || geom.source_height <= 0 || geom.inference_width <= 0
      || geom.inference_height <= 0) {
    return crop;
  }

  const double target_aspect =
      static_cast<double>(geom.inference_width) / static_cast<double>(geom.inference_height);
  const double source_aspect =
      static_cast<double>(geom.source_width) / static_cast<double>(geom.source_height);
//根据目标宽高比计算中心裁减区域
  if (source_aspect > target_aspect) {
    crop.height = geom.source_height;
    crop.width = static_cast<int>(std::lround(static_cast<double>(crop.height) * target_aspect));
    crop.x = std::max(0, (geom.source_width - crop.width) / 2);
    crop.y = 0;
  } else {
    crop.width = geom.source_width;
    crop.height = static_cast<int>(std::lround(static_cast<double>(crop.width) / target_aspect));
    crop.x = 0;
    crop.y = std::max(0, (geom.source_height - crop.height) / 2);
  }
//裁剪区域不能超过源图像大小
  crop.width = std::min(crop.width, geom.source_width - crop.x);
  crop.height = std::min(crop.height, geom.source_height - crop.y);
  return crop;
}
//在中心裁减区域与推理区域不一致时，需要进行立体预处理
bool StereoPreprocessRequired(const StereoImageGeometry &geom) {
  if (geom.source_width <= 0 || geom.source_height <= 0 || geom.inference_width <= 0
      || geom.inference_height <= 0) {
    return false;
  }
  const StereoCropRect crop = ComputeCenterCropRect(geom); // 计算中心裁减区域
  return crop.width > 0 && crop.height > 0
         && (crop.width != geom.inference_width || crop.height != geom.inference_height
             || crop.x != 0 || crop.y != 0 || crop.width != geom.source_width
             || crop.height != geom.source_height);
}

CameraIntrinsics AdjustIntrinsicsForCrop(const CameraIntrinsics &source_cam,
                                         const StereoImageGeometry &geom) {
  if (!StereoPreprocessRequired(geom)) {
    return source_cam;
  }

  const StereoCropRect crop = ComputeCenterCropRect(geom);
  if (crop.width <= 0 || crop.height <= 0) {
    return source_cam;
  }

  CameraIntrinsics cam = source_cam;
  cam.cx -= static_cast<float>(crop.x);
  cam.cy -= static_cast<float>(crop.y);
  return cam;
}

CameraIntrinsics AdjustIntrinsicsForInference(const CameraIntrinsics &source_cam,
                                              const StereoImageGeometry &geom) {
  CameraIntrinsics cam = AdjustIntrinsicsForCrop(source_cam, geom);
  if (geom.inference_width > 0 && geom.inference_height > 0) {
    cam.image_width = geom.inference_width;
    cam.image_height = geom.inference_height;
  }
  if (!StereoPreprocessRequired(geom)) {
    return cam;
  }

  const StereoCropRect crop = ComputeCenterCropRect(geom);
  if (crop.width <= 0 || crop.height <= 0) {
    return cam;
  }

  const float sx =
      static_cast<float>(geom.inference_width) / static_cast<float>(crop.width);
  const float sy =
      static_cast<float>(geom.inference_height) / static_cast<float>(crop.height);
  cam.fx *= sx;
  cam.fy *= sy;
  cam.cx *= sx;
  cam.cy *= sy;
  cam.image_width = geom.inference_width;
  cam.image_height = geom.inference_height;
  return cam;
}

cv::Mat CenterCrop(const cv::Mat &image, const StereoCropRect &crop) {
  if (image.empty() || crop.width <= 0 || crop.height <= 0) {
    return cv::Mat();
  }

  const cv::Rect roi(crop.x, crop.y, crop.width, crop.height);
  if (roi.x < 0 || roi.y < 0 || roi.x + roi.width > image.cols
      || roi.y + roi.height > image.rows) {
    return cv::Mat();
  }

  return image(roi).clone();
}

cv::Mat CenterCropAndResize(const cv::Mat &image, const StereoCropRect &crop, int out_width,
                            int out_height) {
  if (image.empty() || crop.width <= 0 || crop.height <= 0 || out_width <= 0 || out_height <= 0) {
    return cv::Mat();
  }

  const cv::Rect roi(crop.x, crop.y, crop.width, crop.height);
  if (roi.x < 0 || roi.y < 0 || roi.x + roi.width > image.cols
      || roi.y + roi.height > image.rows) {
    return cv::Mat();
  }

  cv::Mat cropped = image(roi);
  cv::Mat resized;
  cv::resize(cropped, resized, cv::Size(out_width, out_height), 0, 0, cv::INTER_LINEAR);
  return resized;
}

cv::Mat DisparityToDepthMeters(const cv::Mat &disp, const CameraIntrinsics &cam, float min_depth,
                               float max_depth) {
  if (disp.empty()) {
    return cv::Mat();
  }

  cv::Mat disp_f32;
  if (disp.type() == CV_32FC1) {
    disp_f32 = disp;
  } else {
    disp.convertTo(disp_f32, CV_32FC1);
  }

  cv::Mat depth = cv::Mat::zeros(disp_f32.size(), CV_32FC1);
  const float fb = std::max(1e-6f, cam.fx * cam.baseline);
  for (int y = 0; y < disp_f32.rows; ++y) {
    const float *dptr = disp_f32.ptr<float>(y);
    float *zptr = depth.ptr<float>(y);
    for (int x = 0; x < disp_f32.cols; ++x) {
      const float d = dptr[x];
      if (!std::isfinite(d) || d <= 1e-6f) {
        continue;
      }
      const float z = fb / d;
      if (std::isfinite(z) && z >= min_depth && z <= max_depth) {
        zptr[x] = z;
      }
    }
  }
  return depth;
}

cv::Mat DisparityToDepthAtCropResolution(const cv::Mat &disp_inference,
                                         const CameraIntrinsics &inference_cam,
                                         const StereoImageGeometry &geom, float min_depth,
                                         float max_depth) {
  const cv::Mat depth_inf =
      DisparityToDepthMeters(disp_inference, inference_cam, min_depth, max_depth);
  if (depth_inf.empty()) {
    return depth_inf;
  }

  const StereoCropRect crop = ComputeCenterCropRect(geom);
  if (crop.width <= 0 || crop.height <= 0) {
    return depth_inf;
  }

  if (depth_inf.cols == crop.width && depth_inf.rows == crop.height) {
    return depth_inf;
  }

  cv::Mat depth_crop;
  cv::resize(depth_inf, depth_crop, cv::Size(crop.width, crop.height), 0, 0, cv::INTER_LINEAR);
  return depth_crop;
}

bool FitImageToSourceGeometry(cv::Mat &image, const StereoImageGeometry &geom) {
  if (image.empty() || geom.source_width <= 0 || geom.source_height <= 0) {
    return false;
  }

  if (image.cols == geom.source_width && image.rows == geom.source_height) {
    return true;
  }

  cv::Mat resized;
  cv::resize(image, resized, cv::Size(geom.source_width, geom.source_height), 0, 0,
             cv::INTER_LINEAR);
  image = std::move(resized);
  return true;
}

} // namespace sio_utils

//  #include "dynamic_cloud_cpp_node/DynamicCloudProcessor.hpp"
#include "sio_frontend_modules.h"
#include "sio_stereo_geometry.h"
#include "internal/geo_flow_error.h"
#include "TrtEngine.hpp"
#include "profiler/simple_profiler.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/imgproc.hpp>
#include <small_gicp/pcl/pcl_point_traits.hpp>
#include <small_gicp/registration/registration.hpp>
#include <small_gicp/registration/registration_helper.hpp>
#include <small_gicp/util/downsampling.hpp>
#include <utility>

namespace {

  inline float NormalizeConfidenceValue(float conf) {
    if (!std::isfinite(conf))
      return 0.1f;
    if (conf > 1.0f) {
      conf /= 255.0f;
    }
    return std::clamp(conf, 0.02f, 1.0f);
  }

  constexpr int kUnreliableBorderMarginPx = 24;
  constexpr int kMinBorderComponentArea = 50;
  // 不可见点着色。动态调试视口为红 (220, 60, 60)，静态保留原色。
  constexpr uint8_t kInvisibleR = 0;
  constexpr uint8_t kInvisibleG = 200;
  constexpr uint8_t kInvisibleB = 255;

  // 去掉贴着顶边或左右边的动态连通域条带；底边保留，过小的连通域整块保留。
  void RemoveUnreliableBorderComponents(cv::Mat &mask) {
    if (mask.empty() || mask.type() != CV_8UC1) {
      return;
    }

    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;
    const int component_count =
      cv::connectedComponentsWithStats(mask, labels, stats, centroids, 8, CV_32S);
    if (component_count <= 1) {
      return;
    }

    const int margin =
      std::min({kUnreliableBorderMarginPx, mask.cols / 4, mask.rows / 4});

    cv::Mat filtered = cv::Mat::zeros(mask.size(), mask.type());
    for (int label = 1; label < component_count; ++label) {
      const int x = stats.at<int>(label, cv::CC_STAT_LEFT);
      const int y = stats.at<int>(label, cv::CC_STAT_TOP);
      const int w = stats.at<int>(label, cv::CC_STAT_WIDTH);
      const int area = stats.at<int>(label, cv::CC_STAT_AREA);

      // Dense optical flow is least reliable where points leave/enter the FOV.
      // Keep bottom-touching components; for top/left/right, drop only a border strip.
      const bool touches_unreliable_border = y == 0 || x == 0 || (x + w) >= mask.cols;
      if (!touches_unreliable_border || area < kMinBorderComponentArea) {
        filtered.setTo(255, labels == label);
        continue;
      }
      cv::Mat comp = (labels == label);
      if (margin > 0) {
        comp.rowRange(0, margin).setTo(0);
        comp.colRange(0, margin).setTo(0);
        comp.colRange(mask.cols - margin, mask.cols).setTo(0);
      }
      filtered.setTo(255, comp);
    }
    mask = std::move(filtered);
  }

  // 动态掩码先做 3×3 腐蚀和 5×5 膨胀，再清理边界上不可靠的动态条带。
  void ApplyDynamicMaskMorphology(cv::Mat &dynamic_mask) {
    cv::Mat kernel3x3 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::Mat kernel5x5 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::erode(dynamic_mask, dynamic_mask, kernel3x3);
    cv::dilate(dynamic_mask, dynamic_mask, kernel5x5);
    RemoveUnreliableBorderComponents(dynamic_mask);
  }

} // namespace

// DynamicCloudProcessor constructor: initialize GPU streams and inference
// engines
DynamicCloudProcessor::DynamicCloudProcessor(const std::string &plan_disp,
                                             const std::string &plan_raft,
                                             const CameraIntrinsics &cam)
  : cam_(cam) {
  // 分别为视差和光流创建独立的执行流
  cudaStreamCreate(&stream_disp_);
  cudaStreamCreate(&stream_raft_);
  algo_disp = std::make_unique<TrtEngine>(plan_disp);
  algo_raft = std::make_unique<TrtEngine>(plan_raft);
  warmup();
}

// DynamicCloudProcessor destructor: cleanup GPU streams
DynamicCloudProcessor::~DynamicCloudProcessor() {
  cudaStreamDestroy(stream_disp_);
  cudaStreamDestroy(stream_raft_);
}

// Warmup disparity and RAFT models with dummy input
void DynamicCloudProcessor::warmup() {
  std::vector<float> dummy_input(1 * 3 * 448 * 896, 0.0f);

  // 1. 预热视差模型使用 stream_disp_
  cudaMemcpy(algo_disp->bindings[0], dummy_input.data(), dummy_input.size() * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(algo_disp->bindings[1], dummy_input.data(), dummy_input.size() * sizeof(float),
             cudaMemcpyHostToDevice);
  algo_disp->infer(stream_disp_);

  // 2. 预热 RAFT 模型使用 stream_raft_
  cudaMemcpy(algo_raft->bindings[0], dummy_input.data(), dummy_input.size() * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(algo_raft->bindings[1], dummy_input.data(), dummy_input.size() * sizeof(float),
             cudaMemcpyHostToDevice);
  algo_raft->infer(stream_raft_);

  // 3. 等待两个并发流都初始化完毕
  cudaStreamSynchronize(stream_disp_);
  cudaStreamSynchronize(stream_raft_);
}

void DynamicCloudProcessor::ConfigureStereoPreprocess(
  const sio_utils::StereoImageGeometry &geom) {
  stereo_geom_ = geom;
  stereo_crop_ = sio_utils::ComputeCenterCropRect(geom);
  stereo_preprocess_enabled_ = sio_utils::StereoPreprocessRequired(geom);
}

cv::Mat DynamicCloudProcessor::preprocessStereoImageCpu(const cv::Mat &cpu_img) const {
  if (cpu_img.empty()) {
    return cv::Mat();
  }

  if (stereo_preprocess_enabled_) {
    // 前端已中心裁减时，此处仅缩放到推理分辨率（供 TensorRT / RAFT）
    if (cpu_img.cols == stereo_crop_.width && cpu_img.rows == stereo_crop_.height) {
      cv::Mat resized;
      cv::resize(cpu_img, resized,
                 cv::Size(stereo_geom_.inference_width, stereo_geom_.inference_height), 0, 0,
                 cv::INTER_LINEAR);
      return resized;
    }
    return sio_utils::CenterCropAndResize(cpu_img, stereo_crop_, stereo_geom_.inference_width,
                                          stereo_geom_.inference_height);
  }

  if (cpu_img.cols == stereo_geom_.inference_width
      && cpu_img.rows == stereo_geom_.inference_height) {
    return cpu_img;
  }

  cv::Mat resized;
  cv::resize(cpu_img, resized,
             cv::Size(stereo_geom_.inference_width, stereo_geom_.inference_height), 0, 0,
             cv::INTER_LINEAR);
  return resized;
}

void DynamicCloudProcessor::uploadDispTensor(const cv::Mat &bgr_inference, float *d_trt_input,
                                           float scale) const {
  const int target_w = stereo_geom_.inference_width;
  const int target_h = stereo_geom_.inference_height;
  cv::cuda::GpuMat d_img_bgr, d_rgb, d_float;
  d_img_bgr.upload(bgr_inference);
  cv::cuda::cvtColor(d_img_bgr, d_rgb, cv::COLOR_BGR2RGB);
  d_rgb.convertTo(d_float, CV_32FC3, scale);
  const int vol = target_w * target_h;
  std::vector<cv::cuda::GpuMat> chw_channels = {
    cv::cuda::GpuMat(target_h, target_w, CV_32FC1, d_trt_input),
    cv::cuda::GpuMat(target_h, target_w, CV_32FC1, d_trt_input + vol),
    cv::cuda::GpuMat(target_h, target_w, CV_32FC1, d_trt_input + vol * 2)};
  cv::cuda::split(d_float, chw_channels);
}

void DynamicCloudProcessor::uploadRaftTensor(const cv::Mat &bgr_inference,
                                             float *d_trt_input) const {
  const int target_w = stereo_geom_.inference_width;
  const int target_h = stereo_geom_.inference_height;
  cv::cuda::GpuMat d_img_bgr, d_rgb, d_float;
  d_img_bgr.upload(bgr_inference);
  cv::cuda::cvtColor(d_img_bgr, d_rgb, cv::COLOR_BGR2RGB);
  d_rgb.convertTo(d_float, CV_32FC3, 2.0f / 255.0f, -1.0f);
  const int vol = target_w * target_h;
  std::vector<cv::cuda::GpuMat> chw_channels = {
    cv::cuda::GpuMat(target_h, target_w, CV_32FC1, d_trt_input),
    cv::cuda::GpuMat(target_h, target_w, CV_32FC1, d_trt_input + vol),
    cv::cuda::GpuMat(target_h, target_w, CV_32FC1, d_trt_input + vol * 2)};
  cv::cuda::split(d_float, chw_channels);
}

// 生成滤波后的视差图
cv::Mat DynamicCloudProcessor::computeFilteredDisparity(const cv::Mat &cpu_left,
                                                        const cv::Mat &cpu_right) {
  const int w = stereo_geom_.inference_width;
  const int h = stereo_geom_.inference_height;
  const int vol = w * h;

  const cv::Mat left_prep = preprocessStereoImageCpu(cpu_left);
  const cv::Mat right_prep = preprocessStereoImageCpu(cpu_right);
  if (left_prep.empty() || right_prep.empty()) {
    return cv::Mat();
  }

  uploadDispTensor(left_prep, (float *)algo_disp->bindings[0], 1.0f);
  uploadDispTensor(right_prep, (float *)algo_disp->bindings[1], 1.0f);

  algo_disp->infer(stream_disp_);
  cudaStreamSynchronize(stream_disp_); // 等待视差流完成

  cv::Mat h_disp_raw(h, w, CV_32FC1);
  cudaMemcpy(h_disp_raw.ptr<float>(), algo_disp->getBindingPtr("disp"), vol * sizeof(float),
             cudaMemcpyDeviceToHost);

  cv::Mat h_disp_filtered(h, w, CV_32FC1);
  filterDisparityMap(h_disp_raw, h_disp_filtered, 3, 0.8f, 5);
  return h_disp_filtered;
}

// Generate point cloud from disparity map with optional confidence values
StereoCloud DynamicCloudProcessor::generatePointCloud(const cv::Mat &rgb_img, const cv::Mat &h_disp,
                                                      const cv::Mat &confidence_map,
                                                      std::vector<float> *point_confidences) {
  const cv::Mat rgb_prep = preprocessStereoImageCpu(rgb_img);
  if (rgb_prep.empty()) {
    return StereoCloud();
  }
  return disparityToPointCloud(rgb_prep, h_disp, dynamic_filter_cfg_.min_depth_m,
                             dynamic_filter_cfg_.max_depth_m, confidence_map, point_confidences);
}

// 用位姿协方差 Σ_f 的马氏门控比较几何光流与 RAFT；无像素、相机后方、出当前图像或被遮挡的点既非静态也非动态。
// 仅左图为空或裁减放不下时返回 false：保留不可见、动态为空、未检验可见点记为静态；成功时静态为空也合法。
bool DynamicCloudProcessor::removeDynamicObjects(
  const cv::Mat &prev_left, const StereoCloud &prev_pcd, const cv::Mat &curr_left,
  Eigen::Matrix4d T_curr_prevkf, const cv::Mat &prev_disp,
  const std::vector<float> *prev_confidences, std::vector<float> *static_confidences,
  std::vector<int> *static_indices, std::vector<int> *dynamic_indices,
  const Eigen::Matrix<double, 6, 6> *pose_cov_cam,
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr *invisible_cloud,
  std::vector<int> *invisible_indices, StereoCloud *static_cloud) {
  const int w = stereo_geom_.inference_width;
  const int h = stereo_geom_.inference_height;

  // T_curr_prevkf: p_curr = T_curr_prevkf * p_prevkf（KF camera → 当前 camera）
  // cam_ 已是推理分辨率内参，与视差/RAFT/点云一致

  auto geo_results =
    generate_geometric_flow_cpp(prev_pcd, T_curr_prevkf.cast<float>(), w, h, cam_);

  // 图像为空或裁减 ROI 放不下时预处理得到空图。几何光流已经标过不可见：
  // 动态为空，未检验的可见点暂作静态。不把整云当静态，也不清掉不可见。
  const cv::Mat prev_prep = preprocessStereoImageCpu(prev_left);
  const cv::Mat curr_prep = preprocessStereoImageCpu(curr_left);
  if (prev_prep.empty() || curr_prep.empty()) {
    StereoCloud static_pcd = extractStaticPointCloud(
      prev_pcd, {}, {}, geo_results.invisible_indices, cv::Mat(), prev_confidences,
      static_confidences, static_indices, dynamic_indices, invisible_cloud, invisible_indices);
    if (static_cloud) {
      *static_cloud = std::move(static_pcd);
    }
    return false;
  }

  uploadRaftTensor(prev_prep, (float *)algo_raft->bindings[0]);
  uploadRaftTensor(curr_prep, (float *)algo_raft->bindings[1]);
  algo_raft->infer(stream_raft_);
  cudaStreamSynchronize(stream_raft_);
  // The supplied iter6 export exposes every iteration; 2206 is the final flow.
  // Single-output exports retain their existing output binding.
  void *flow_output = algo_raft->hasTensor("2206")
    ? algo_raft->getBindingPtr("2206") : algo_raft->bindings[2];
  cv::Mat raft_flow_mat = postprocessRaftFlow(static_cast<float *>(flow_output), w, h);

  cv::Mat dynamic_mask =
    detectDynamicObjectsAdaptive(raft_flow_mat, geo_results.geo_flow, prev_disp, cam_,
                                 pose_cov_cam);
  ApplyDynamicMaskMorphology(dynamic_mask);

  StereoCloud static_pcd = extractStaticPointCloud(
    prev_pcd, geo_results.valid_indices, geo_results.flow_valid, geo_results.invisible_indices,
    dynamic_mask, prev_confidences, static_confidences, static_indices, dynamic_indices,
    invisible_cloud, invisible_indices);
  if (static_cloud) {
    *static_cloud = std::move(static_pcd);
  }
  return true;
}

// Filter disparity map by checking consistency with neighbors
void DynamicCloudProcessor::filterDisparityMap(const cv::Mat &input_disp, cv::Mat &output_disp,
                                               int window_size, float diff_thresh,
                                               int count_thresh) {
  int width = input_disp.cols, height = input_disp.rows;
  output_disp = input_disp.clone();
  int r = window_size / 2;
  for (int y = 0; y < height; ++y) {
    const float *in_row = input_disp.ptr<float>(y);
    float *out_row = output_disp.ptr<float>(y);
    for (int x = 0; x < width; ++x) {
      float d = in_row[x];
      if (d <= 0.0f)
        continue;
      int valid_count = 0;
      float min_bound = d - diff_thresh, max_bound = d + diff_thresh;
      for (int dy = -r; dy <= r; ++dy) {
        for (int dx = -r; dx <= r; ++dx) {
          if (dx == 0 && dy == 0)
            continue;
          int ny = y + dy, nx = x + dx;
          if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
            float n_d = input_disp.ptr<float>(ny)[nx];
            if (n_d > 0.0f && n_d >= min_bound && n_d <= max_bound)
              valid_count++;
          }
        }
      }
      if (valid_count < count_thresh)
        out_row[x] = 0.0f;
    }
  }
}

// 视差反投影为相机系点云，有效像素写入 StereoPoint 的 u、v，has_pixel=1。
// d<=0 或深度超范围不建点，因而既不进静态云也不标动态。
StereoCloud DynamicCloudProcessor::disparityToPointCloud(
  const cv::Mat &rgb_img, const cv::Mat &h_disp, float zmin, float zmax,
  const cv::Mat &confidence_map, std::vector<float> *point_confidences) {
  int width = h_disp.cols, height = h_disp.rows;
  StereoCloud cloud;
  cloud.reserve(static_cast<size_t>(width) * static_cast<size_t>(height));

  cv::Mat confidence_resized;
  if (!confidence_map.empty()) {
    if (confidence_map.size() != h_disp.size()) {
      cv::resize(confidence_map, confidence_resized, h_disp.size(), 0, 0, cv::INTER_LINEAR);
    } else {
      confidence_resized = confidence_map;
    }
    if (confidence_resized.type() != CV_32FC1) {
      confidence_resized.convertTo(confidence_resized, CV_32FC1);
    }
  }

  if (point_confidences) {
    point_confidences->clear();
    point_confidences->reserve(width * height);
  }

  for (int v = 0; v < height; ++v) {
    const float *disp_row = h_disp.ptr<float>(v);
    const float *conf_row = confidence_resized.empty() ? nullptr : confidence_resized.ptr<float>(v);
    for (int u = 0; u < width; ++u) {
      float d = disp_row[u];
      // d<=0：无有效视差（含邻域滤波清零）。不建点，故不进静态云，也不标动态。
      if (d <= 0.0f)
        continue;
      float z = (cam_.fx * cam_.baseline) / d;
      if (z < zmin || z > zmax)
        continue;
      StereoPoint sp{};
      sp.point.x = (u - cam_.cx) * z / cam_.fx;
      sp.point.y = (v - cam_.cy) * z / cam_.fy;
      sp.point.z = z;
      sp.point.data[3] = 1.0f;
      cv::Vec3b color = rgb_img.at<cv::Vec3b>(v, u);
      sp.point.r = color[2];
      sp.point.g = color[1];
      sp.point.b = color[0];
      sp.u = static_cast<std::uint16_t>(u);
      sp.v = static_cast<std::uint16_t>(v);
      sp.has_pixel = 1;
      cloud.push_back(sp);

      if (point_confidences) {
        float conf = 1.0f;
        if (conf_row) {
          conf = NormalizeConfidenceValue(conf_row[u]);
        } else {
          const float d_left = (u > 0) ? disp_row[u - 1] : d;
          const float d_up = (v > 0) ? h_disp.ptr<float>(v - 1)[u] : d;
          const float smooth_penalty = std::abs(d - d_left) + std::abs(d - d_up);
          conf = NormalizeConfidenceValue(1.0f / (1.0f + 0.5f * smooth_penalty));
        }
        point_confidences->push_back(conf);
      }
    }
  }
  return cloud;
}

// 由刚体变换生成投影几何光流，参考像素取自 StereoPoint 的 u、v、has_pixel。
// 无像素、相机后方、出当前图像或被遮挡的点记为不可见，既非静态也非动态。
GeoFlowResult DynamicCloudProcessor::generate_geometric_flow_cpp(
  const StereoCloud &pcd, const Eigen::Matrix4f &T, int width, int height,
  const CameraIntrinsics &camera) {
  GeoFlowResult result;
  result.geo_flow = cv::Mat::zeros(height, width, CV_32FC2);
  result.geo_flow_trans = cv::Mat::zeros(height, width, CV_32FC2);

  // 当前图像 Z-buffer：整数像素 (u', v') 只保留最小当前深度 Z'。
  // 参考像素读 StereoPoint 的 u、v。has_pixel==0 视为不可见。
  cv::Mat depth_buffer(height, width, CV_32FC1, std::numeric_limits<float>::max());
  cv::Mat index_buffer(height, width, CV_32SC1, -1);
  // 光流先记在当前像素槽位上，循环结束后只把获胜点搬到视差图索引。
  cv::Mat slot_flow = cv::Mat::zeros(height, width, CV_32FC2);
  cv::Mat slot_flow_trans = cv::Mat::zeros(height, width, CV_32FC2);

  size_t N = pcd.size();
  float r00 = T(0, 0), r01 = T(0, 1), r02 = T(0, 2), t0 = T(0, 3);
  float r10 = T(1, 0), r11 = T(1, 1), r12 = T(1, 2), t1 = T(1, 3);
  float r20 = T(2, 0), r21 = T(2, 1), r22 = T(2, 2), t2 = T(2, 3);
  result.invisible_indices.reserve(N);
  for (size_t i = 0; i < N; ++i) {
    const StereoPoint &sp = pcd[i];
    const int u_ref = static_cast<int>(sp.u);
    const int v_ref = static_cast<int>(sp.v);
    // has_pixel==0（左右目不重叠 / 立体失败，或点不是视差三角化得到）：不可见
    if (sp.has_pixel == 0 || u_ref < 0 || u_ref >= width || v_ref < 0 || v_ref >= height) {
      result.invisible_indices.push_back(static_cast<int>(i));
      continue;
    }
    const float u0 = static_cast<float>(u_ref);
    const float v0 = static_cast<float>(v_ref);

    float X = sp.point.x, Y = sp.point.y, Z = sp.point.z;
    float X1 = r00 * X + r01 * Y + r02 * Z + t0;
    float Y1 = r10 * X + r11 * Y + r12 * Z + t1;
    float Z1 = r20 * X + r21 * Y + r22 * Z + t2;
    // 当前相机后方：不可见，不占当前像素 Z-buffer，也不进入候选集
    if (Z1 < 1e-4f) {
      result.invisible_indices.push_back(static_cast<int>(i));
      continue;
    }

    float u1 = (X1 * camera.fx) / Z1 + camera.cx;
    float v1 = (Y1 * camera.fy) / Z1 + camera.cy;
    const bool outside_current = u1 < 0.0f || u1 >= static_cast<float>(width) || v1 < 0.0f
                                 || v1 >= static_cast<float>(height);
    // 出当前图像：不可见，不占当前像素 Z-buffer
    if (outside_current) {
      result.invisible_indices.push_back(static_cast<int>(i));
      continue;
    }

    int u1_int = std::round(u1);
    int v1_int = std::round(v1);
    if (u1_int >= width)
      u1_int = width - 1;
    if (v1_int >= height)
      v1_int = height - 1;

    float &depth_cur = depth_buffer.at<float>(v1_int, u1_int);
    int &slot_idx = index_buffer.at<int>(v1_int, u1_int);
    if (Z1 < depth_cur) {
      // 较远点先写入时：把它移入不可见，再用更近点替换该当前像素的光流/索引槽
      if (slot_idx >= 0)
        result.invisible_indices.push_back(slot_idx);
      depth_cur = Z1;
      slot_flow.at<cv::Vec2f>(v1_int, u1_int) = cv::Vec2f(u1 - u0, v1 - v0);
      const float Z1t = Z + t2;
      if (Z1t >= 1e-4f) {
        const float u1t = (X + t0) * camera.fx / Z1t + camera.cx;
        const float v1t = (Y + t1) * camera.fy / Z1t + camera.cy;
        slot_flow_trans.at<cv::Vec2f>(v1_int, u1_int) = cv::Vec2f(u1t - u0, v1t - v0);
      } else {
        slot_flow_trans.at<cv::Vec2f>(v1_int, u1_int) = cv::Vec2f(0.f, 0.f);
      }
      slot_idx = static_cast<int>(i);
    } else {
      result.invisible_indices.push_back(static_cast<int>(i));
    }
  }

  // 只发布当前像素获胜点。光流落到 StereoPoint 的参考像素，与 RAFT / 视差对齐；被替换或被遮挡的点没有样本。
  result.valid_indices.reserve(N);
  result.project_points.reserve(N);
  result.flow_valid.reserve(N);
  for (int v = 0; v < height; ++v) {
    const int *idx_row = index_buffer.ptr<int>(v);
    const cv::Vec2f *flow_row = slot_flow.ptr<cv::Vec2f>(v);
    const cv::Vec2f *flow_t_row = slot_flow_trans.ptr<cv::Vec2f>(v);
    for (int u = 0; u < width; ++u) {
      int pt_idx = idx_row[u];
      if (pt_idx < 0)
        continue;
      const StereoPoint &sp = pcd[static_cast<size_t>(pt_idx)];
      const int u_ref = static_cast<int>(sp.u);
      const int v_ref = static_cast<int>(sp.v);
      if (sp.has_pixel == 0 || u_ref < 0 || u_ref >= width || v_ref < 0 || v_ref >= height) {
        result.invisible_indices.push_back(pt_idx);
        continue;
      }
      result.valid_indices.push_back(pt_idx);
      result.geo_flow.at<cv::Vec2f>(v_ref, u_ref) = flow_row[u];
      result.geo_flow_trans.at<cv::Vec2f>(v_ref, u_ref) = flow_t_row[u];
      result.project_points.emplace_back(static_cast<float>(u_ref), static_cast<float>(v_ref));
      result.flow_valid.emplace_back(u_ref, v_ref);
    }
  }
  return result;
}

// Postprocess RAFT output to optical flow map
cv::Mat DynamicCloudProcessor::postprocessRaftFlow(float *d_trt_out_flow, int width, int height) {
  cv::cuda::GpuMat d_u(height, width, CV_32FC1, d_trt_out_flow);
  cv::cuda::GpuMat d_v(height, width, CV_32FC1, d_trt_out_flow + width * height);
  std::vector<cv::cuda::GpuMat> flow_channels = {d_u, d_v};
  cv::cuda::GpuMat d_flow_merged;
  cv::cuda::merge(flow_channels, d_flow_merged);
  cv::Mat raft_flow_cpu;
  d_flow_merged.download(raft_flow_cpu);
  return raft_flow_cpu;
}

// 自适应动态检测：由位姿协方差传播 Σ_f，对 RAFT 与几何光流的残差做马氏检验，写出动态掩码。
cv::Mat DynamicCloudProcessor::detectDynamicObjectsAdaptive(
  const cv::Mat &raft_flow, const cv::Mat &geo_flow, const cv::Mat &h_disp,
  const CameraIntrinsics &camera, const Eigen::Matrix<double, 6, 6> *pose_cov_cam) {
  int width = h_disp.cols, height = h_disp.rows;
  cv::Mat dynamic_mask = cv::Mat::zeros(height, width, CV_8UC1);
  const Eigen::Matrix<float, 6, 6> pose_cov = sio_geo_flow::MakePoseCovCam(pose_cov_cam);
  const float fx = camera.fx;
  const float fy = camera.fy;
  const float cx = camera.cx;
  const float cy = camera.cy;
  const float baseline = camera.baseline;
  const float z_scale =
    (std::isfinite(baseline) && baseline > 1e-6f && fx > 1e-6f) ? (fx * baseline) : 0.0f;
  // Optional soft depth gate from YAML (in addition to kMinDisp)
  const sio_utils::DisparityDepthBounds disp_bounds = sio_utils::ComputeDisparityDepthBounds(
    fx, baseline, dynamic_filter_cfg_.min_depth_m, dynamic_filter_cfg_.max_depth_m);
  const float skip_sq = std::max(dynamic_filter_cfg_.geo_flow_min_sq, sio_geo_flow::kMinGeoOrRaftSq);

#pragma omp parallel for
  for (int y = 0; y < height; ++y) {
    const cv::Vec2f *r_ptr = raft_flow.ptr<cv::Vec2f>(y);
    const cv::Vec2f *g_ptr = geo_flow.ptr<cv::Vec2f>(y);
    uchar *mask_ptr = dynamic_mask.ptr<uchar>(y);
    const float *disp_row = h_disp.ptr<float>(y);
    const float yn = (static_cast<float>(y) - cy) / fy;
    for (int x = 0; x < width; ++x) {
      const float xn = (static_cast<float>(x) - cx) / fx;
      const float d = disp_row[x];
      const bool disp_ok =
        std::isfinite(d) && d > sio_geo_flow::kMinDisp && sio_utils::IsDisparityInDepthRange(d, disp_bounds);
      const float inv_z = (disp_ok && z_scale > 0.0f) ? (d / z_scale) : 0.0f;
      const sio_geo_flow::FlowJac jac = sio_geo_flow::ComputeFlowJac(xn, yn, fx, fy, inv_z);
      const sio_geo_flow::SigmaF sigma_f =
        sio_geo_flow::AccumulateSigmaFFromPoseCov(jac, pose_cov, inv_z > 0.0f);

      if (!disp_ok)
        continue;

      const float gu = g_ptr[x][0], gv = g_ptr[x][1];
      const float ru = r_ptr[x][0], rv = r_ptr[x][1];
      const float geo_sq = gu * gu + gv * gv;
      const float raft_sq = ru * ru + rv * rv;
      // Skip only when both flows are tiny. Do not skip |geo|≈0 while RAFT is large.
      if (geo_sq <= skip_sq && raft_sq <= skip_sq)
        continue;
      const sio_geo_flow::SigmaF sigma_reg =
        sio_geo_flow::AddIsotropicRidge(sigma_f, sio_geo_flow::kMinFerrPx);
      const float diff_u = ru - gu, diff_v = rv - gv;
      if (sio_geo_flow::MahalanobisSq(diff_u, diff_v, sigma_reg) >= 1.0f)
        mask_ptr[x] = 255;
    }
  }
  return dynamic_mask;
}

// 按动态掩码从参考点云抽出静态点，并可选写出索引与置信度。
// 不可见点（无像素、相机后方、出当前图像或被遮挡）既不进静态云，也不写入动态索引。
StereoCloud DynamicCloudProcessor::extractStaticPointCloud(
  const StereoCloud &original_pcd, const std::vector<int> &valid_indices,
  const std::vector<cv::Point> &pixel_coords, const std::vector<int> &invisible_indices,
  const cv::Mat &dynamic_mask, const std::vector<float> *original_confidences,
  std::vector<float> *static_confidences, std::vector<int> *static_indices,
  std::vector<int> *dynamic_indices,
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr *invisible_cloud,
  std::vector<int> *out_invisible_indices) {
  std::vector<uint8_t> is_static(original_pcd.size(), 1);
  std::vector<uint8_t> is_invisible(original_pcd.size(), 0);
  for (const int idx : invisible_indices) {
    if (idx >= 0 && static_cast<size_t>(idx) < is_invisible.size()) {
      is_invisible[static_cast<size_t>(idx)] = 1;
      is_static[static_cast<size_t>(idx)] = 0;
    }
  }
  int width = dynamic_mask.cols, height = dynamic_mask.rows;
  if (dynamic_indices) {
    dynamic_indices->clear();
    dynamic_indices->reserve(pixel_coords.size());
  }
  for (size_t i = 0; i < pixel_coords.size(); ++i) {
    int u = pixel_coords[i].x, v = pixel_coords[i].y;
    if (u >= 0 && u < width && v >= 0 && v < height && dynamic_mask.at<uchar>(v, u) > 0) {
      const int orig_idx = valid_indices[i];
      if (orig_idx < 0 || static_cast<size_t>(orig_idx) >= is_static.size())
        continue;
      if (is_invisible[static_cast<size_t>(orig_idx)])
        continue;
      is_static[orig_idx] = 0;
      if (dynamic_indices) {
        dynamic_indices->push_back(orig_idx);
      }
    }
  }
  StereoCloud static_pcd;
  static_pcd.reserve(original_pcd.size());
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr invisible_pcd;
  if (invisible_cloud) {
    invisible_pcd.reset(new pcl::PointCloud<pcl::PointXYZRGB>());
    invisible_pcd->points.reserve(original_pcd.size());
  }
  if (static_confidences) {
    static_confidences->clear();
    static_confidences->reserve(original_pcd.size());
  }
  if (static_indices) {
    static_indices->clear();
    static_indices->reserve(original_pcd.size());
  }
  if (out_invisible_indices) {
    out_invisible_indices->clear();
    out_invisible_indices->reserve(original_pcd.size());
  }
  for (size_t i = 0; i < original_pcd.size(); ++i) {
    if (is_invisible[i]) {
      if (out_invisible_indices) {
        out_invisible_indices->push_back(static_cast<int>(i));
      }
      if (invisible_pcd) {
        pcl::PointXYZRGB pt = CopyStereoPointXyzrgb(original_pcd[i]);
        pt.r = kInvisibleR;
        pt.g = kInvisibleG;
        pt.b = kInvisibleB;
        pt.data[3] = 1.0f;
        invisible_pcd->points.push_back(pt);
      }
      continue;
    }
    if (is_static[i]) {
      static_pcd.push_back(original_pcd[i]);
      if (static_indices) {
        static_indices->push_back(static_cast<int>(i));
      }
      if (static_confidences) {
        float conf = 1.0f;
        if (original_confidences && i < original_confidences->size()) {
          conf = NormalizeConfidenceValue((*original_confidences)[i]);
        }
        static_confidences->push_back(conf);
      }
    }
  }
  if (invisible_pcd) {
    invisible_pcd->width = invisible_pcd->points.size();
    invisible_pcd->height = 1;
    invisible_pcd->is_dense = true;
    *invisible_cloud = invisible_pcd;
  }
  return static_pcd;
}

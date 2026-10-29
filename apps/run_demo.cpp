#include "color_icp.h"
#include "dataset_loader.h"
#include "sio_frontend_modules.h"
#include "sio_pipeline_helpers.h"
#include "utils.h"

#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#if __has_include(<cuda_runtime_api.h>)
#include <cuda_runtime_api.h>
#endif

namespace {

using sio_pipeline::BuildCurrentNormals;
using sio_pipeline::BuildFeatureCache;
using sio_pipeline::FeatureCache;
using sio_pipeline::ImuMsg;
using sio_pipeline::IsImuDataContinuous;
using sio_pipeline::PublishCloud;

int ChooseWorkerThreads(int requested_threads) {
  const int hw_threads =
      std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
  if (requested_threads > 0) {
    return std::clamp(requested_threads, 1, hw_threads);
  }

  // 给系统/ROS 留 1-2 个核，防止过度抢占导致实时性抖动。
  return std::clamp(hw_threads - 2, 2, hw_threads);
}

void ConfigureRuntimeThreads(const rclcpp::Node::SharedPtr &node,
                             int worker_threads) {
  node->declare_parameter<int>("opencv_threads", worker_threads);
  const int opencv_threads =
      std::max(1, static_cast<int>(node->get_parameter("opencv_threads").as_int()));

  cv::setUseOptimized(true);
  cv::setNumThreads(opencv_threads);

  const std::string omp_threads = std::to_string(worker_threads);
  setenv("OMP_NUM_THREADS", omp_threads.c_str(), 1);
  setenv("OMP_WAIT_POLICY", "PASSIVE", 1);

  RCLCPP_INFO(node->get_logger(),
              "Thread config: hw=%d worker_threads=%d opencv_threads=%d",
              std::max(1, static_cast<int>(std::thread::hardware_concurrency())),
              worker_threads, opencv_threads);
}

bool ConfigureCudaDevice(const rclcpp::Node::SharedPtr &node) {
#if __has_include(<cuda_runtime_api.h>)
  node->declare_parameter<int>("cuda_device_id", 0);
  int requested_device = node->get_parameter("cuda_device_id").as_int();

  int device_count = 0;
  cudaError_t count_status = cudaGetDeviceCount(&device_count);
  if (count_status != cudaSuccess || device_count <= 0) {
    RCLCPP_ERROR(node->get_logger(), "CUDA device not available: %s",
                 cudaGetErrorString(count_status));
    return false;
  }

  requested_device = std::clamp(requested_device, 0, device_count - 1);
  cudaError_t set_status = cudaSetDevice(requested_device);
  if (set_status != cudaSuccess) {
    RCLCPP_ERROR(node->get_logger(), "Failed to set CUDA device %d: %s",
                 requested_device, cudaGetErrorString(set_status));
    return false;
  }

  // 触发上下文初始化，避免首帧出现明显延迟抖动。
  cudaError_t warmup_status = cudaFree(nullptr);
  if (warmup_status != cudaSuccess) {
    RCLCPP_WARN(node->get_logger(), "CUDA context warmup warning: %s",
                cudaGetErrorString(warmup_status));
  }

  RCLCPP_INFO(node->get_logger(), "CUDA device %d/%d ready", requested_device,
              device_count);
  return true;
#else
  RCLCPP_WARN(node->get_logger(), "CUDA runtime header not found at compile time.");
  return true;
#endif
}

} // namespace

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = rclcpp::Node::make_shared("sio_frontend_run_demo");
  const auto project_cfg = svio_utils::LoadProjectRuntimeConfig();
  const CameraIntrinsics default_cam =
    svio_utils::LoadInferenceCameraIntrinsics(project_cfg.params_file);

  node->declare_parameter<std::string>("img_dir", project_cfg.dataset_dir.string());
  node->declare_parameter<std::string>("imu_file", project_cfg.imu_file.string());
  node->declare_parameter<std::string>("plan_disp", project_cfg.plan_disp.string());
  node->declare_parameter<std::string>("plan_raft", project_cfg.plan_raft.string());
  node->declare_parameter<std::string>("topic", project_cfg.topic);
  node->declare_parameter<std::string>("frame_id", project_cfg.frame_id);
  node->declare_parameter<double>("fx", default_cam.fx);
  node->declare_parameter<double>("fy", default_cam.fy);
  node->declare_parameter<double>("cx", default_cam.cx);
  node->declare_parameter<double>("cy", default_cam.cy);
  node->declare_parameter<double>("baseline", default_cam.baseline);
  node->declare_parameter<int>("imu_init_samples", 200);
  node->declare_parameter<int>("max_frames", 120);
  node->declare_parameter<bool>("run_color_icp", false);
  node->declare_parameter<int>("worker_threads", -1);

  const auto img_dir = node->get_parameter("img_dir").as_string();
  const auto imu_file = node->get_parameter("imu_file").as_string();
  const auto plan_disp = node->get_parameter("plan_disp").as_string();
  const auto plan_raft = node->get_parameter("plan_raft").as_string();
  const auto topic = node->get_parameter("topic").as_string();
  const auto frame_id = node->get_parameter("frame_id").as_string();
  const int imu_init_samples = node->get_parameter("imu_init_samples").as_int();
  const int max_frames = node->get_parameter("max_frames").as_int();
  const bool run_color_icp = node->get_parameter("run_color_icp").as_bool();
  const int worker_threads =
      ChooseWorkerThreads(node->get_parameter("worker_threads").as_int());
  const Eigen::Matrix4d T_imu_cam =
      svio_utils::LoadImuCameraExtrinsic(project_cfg.params_file);

  ConfigureRuntimeThreads(node, worker_threads);
  if (!ConfigureCudaDevice(node)) {
    rclcpp::shutdown();
    return 1;
  }

  if (img_dir.empty() || imu_file.empty()) {
    RCLCPP_ERROR(node->get_logger(),
                 "img_dir and imu_file must be provided through parameters.");
    rclcpp::shutdown();
    return 1;
  }
  if (!std::filesystem::exists(img_dir) || !std::filesystem::exists(imu_file)) {
    RCLCPP_ERROR(node->get_logger(),
                 "Dataset path not found: img_dir=%s imu_file=%s",
                 img_dir.c_str(), imu_file.c_str());
    rclcpp::shutdown();
    return 1;
  }
  if (plan_disp.empty() || plan_raft.empty()) {
    RCLCPP_ERROR(node->get_logger(),
                 "plan_disp and plan_raft must be provided through parameters.");
    rclcpp::shutdown();
    return 1;
  }
  if (!std::filesystem::exists(plan_disp) || !std::filesystem::exists(plan_raft)) {
    RCLCPP_ERROR(node->get_logger(), "Plan file not found: disp=%s raft=%s",
                 plan_disp.c_str(), plan_raft.c_str());
    rclcpp::shutdown();
    return 1;
  }

  CameraIntrinsics cam;
  cam.fx = static_cast<float>(node->get_parameter("fx").as_double());
  cam.fy = static_cast<float>(node->get_parameter("fy").as_double());
  cam.cx = static_cast<float>(node->get_parameter("cx").as_double());
  cam.cy = static_cast<float>(node->get_parameter("cy").as_double());
  cam.baseline = static_cast<float>(node->get_parameter("baseline").as_double());

  const sio_utils::StereoImageGeometry stereo_geom =
    sio_utils::LoadStereoImageGeometry(project_cfg.params_file);
  const std::filesystem::path params_cfg_path =
    sio_utils::LocateParamsFile(project_cfg);

  DatasetLoader loader(img_dir, imu_file);
  ImagePreprocessor image_preprocessor;
  if (!params_cfg_path.empty()) {
    image_preprocessor.LoadConfig(params_cfg_path.string());
  }
  DynamicCloudProcessor dynamic_processor(plan_disp, plan_raft, cam);
  dynamic_processor.ConfigureStereoPreprocess(stereo_geom);
  dynamic_processor.SetDynamicFilterParams(
    sio_utils::LoadDynamicFilterParams(params_cfg_path));
  ImuIntegrator imu_integrator;
  FeatureProcessor feature_processor;
  StateEstimator estimator;
  KeyframeManager kf_manager;
  ColorICP color_icp;

  auto cloud_pub = topic.empty()
                       ? rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr()
                       : node->create_publisher<sensor_msgs::msg::PointCloud2>(topic, 10);

  std::deque<ImuMsg> imu_buffer;
  std::deque<std::shared_ptr<KeyFrameData>> frontend_keyframes;

  ImuStatesGroup imu_states;
  Eigen::Matrix<double, DIM_STATE, DIM_STATE> flow_gate_cov =
      Eigen::Matrix<double, DIM_STATE, DIM_STATE>::Zero();
  bool imu_initialized = false;
  bool has_prev_keyframe = false;
  KeyFrameData prev_frame{};
  StereoCloud prev_stereo;
  KeyFrameData prev_backend_keyframe{};
  FeatureCache target_cache;

  std::size_t sensor_frame_count = 0;
  std::size_t processed_frame_count = 0;
  std::size_t frontend_keyframe_count = 0;
  std::size_t backend_keyframe_count = 0;
  std::size_t eskf_count = 0;
  std::size_t esikf_count = 0;
  double total_process_ms = 0.0;

  while (rclcpp::ok()) {
    if (max_frames > 0 &&
        processed_frame_count >= static_cast<std::size_t>(max_frames)) {
      break;
    }

    auto frame_data = loader.GrabNextFrame();
    if (!frame_data) {
      break;
    }
    ++sensor_frame_count;

    if (!imu_initialized) {
      for (const auto &imu_msg : frame_data->imu) {
        imu_buffer.push_back(std::make_shared<sensor_msgs::msg::Imu>(*imu_msg));
      }

      if (static_cast<int>(imu_buffer.size()) < imu_init_samples) {
        RCLCPP_INFO(node->get_logger(),
                    "[init] frame %zu collected %zu IMU samples, waiting for %d",
                    sensor_frame_count, imu_buffer.size(), imu_init_samples);
        continue;
      }

      if (!IsImuDataContinuous(imu_buffer)) {
        RCLCPP_WARN(node->get_logger(),
                    "[init] IMU samples are not continuous, clearing buffer");
        imu_buffer.clear();
        continue;
      }

      imu_states = imu_integrator.ImuInit(imu_buffer);
      imu_buffer.clear();
      loader.ResetFrameId();
      imu_initialized = true;
      RCLCPP_INFO(node->get_logger(), "IMU initialization complete.");
      continue;
    }

    auto frame_begin = std::chrono::steady_clock::now();

    if (!frame_data->imgl.empty() && !frame_data->imgr.empty()) {
      auto [pre_left, pre_right] =
        image_preprocessor.ProcessStereoAdaptiveLighting(frame_data->imgl, frame_data->imgr);
      if (!pre_left.empty()) {
        frame_data->imgl = pre_left;
      }
      if (!pre_right.empty()) {
        frame_data->imgr = pre_right;
      }
    }

    for (const auto &imu_msg : frame_data->imu) {
      auto imu_copy = std::make_shared<sensor_msgs::msg::Imu>(*imu_msg);
      imu_buffer.push_back(imu_copy);
      kf_manager.AddImuData(imu_copy);
      if (imu_buffer.size() > 600) {
        imu_buffer.pop_front();
      }
    }

    Eigen::Matrix4d T_prevkf_curr = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d T_curr_prevkf = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d T_w_prevkf = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d T_w_prev_cam = Eigen::Matrix4d::Identity();
    ImuStatesGroup imu_propagated = imu_states;

    if (has_prev_keyframe) {
      KeyFrameData propagation_anchor = prev_frame;
      propagation_anchor.timestamp =
          (imu_states.timestamp > 0.0) ? imu_states.timestamp : prev_frame.timestamp;
      propagation_anchor.imu_states = imu_states;

      imu_propagated = imu_integrator.ImuPreintegrationCov(
          propagation_anchor, imu_buffer, frame_data->end_time, &flow_gate_cov);

      T_w_prevkf = svio_utils::ImuStatesToMatrix4d(prev_frame.imu_states);
      const Eigen::Matrix4d T_w_curr_pred =
          svio_utils::ImuStatesToMatrix4d(imu_propagated);
      T_w_prev_cam = svio_utils::WorldCamFromBody(T_w_prevkf, T_imu_cam);
      const Eigen::Matrix4d T_w_curr_pred_cam =
          svio_utils::WorldCamFromBody(T_w_curr_pred, T_imu_cam);
      T_prevkf_curr = svio_utils::RelativeTransform(T_w_prev_cam, T_w_curr_pred_cam);
      T_curr_prevkf = T_prevkf_curr.inverse();
    }

    cv::Mat h_disp;
    cv::Mat confidence_map;
    std::vector<float> current_point_confidences;
    StereoCloud current_stereo;

    std::thread stereo_thread([&]() {
      h_disp = dynamic_processor.computeFilteredDisparity(frame_data->imgl,
                                                          frame_data->imgr);
      current_stereo = dynamic_processor.generatePointCloud(
          frame_data->imgl, h_disp, confidence_map, &current_point_confidences);
    });

    StereoCloud static_prev_stereo;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr invisible_prev_cloud;
    std::vector<float> static_prev_confidences;
    std::thread dynamic_thread;
    bool dynamic_filter_ok = false;
    Eigen::Matrix<double, 6, 6> pose_cov_cam = Eigen::Matrix<double, 6, 6>::Zero();
    const size_t prev_cloud_before = has_prev_keyframe ? prev_stereo.size() : 0;
    if (has_prev_keyframe) {
      pose_cov_cam =
          svio_utils::FlowGatePoseCovInCamera(imu_propagated, flow_gate_cov, T_imu_cam);
      dynamic_thread = std::thread([&]() {
        dynamic_filter_ok = dynamic_processor.removeDynamicObjects(
            prev_frame.image_left, prev_stereo, frame_data->imgl,
            T_curr_prevkf, prev_frame.disp_filtered, &prev_frame.point_confidences,
            &static_prev_confidences, nullptr, nullptr, &pose_cov_cam,
            &invisible_prev_cloud, nullptr, &static_prev_stereo);
      });
    }

    stereo_thread.join();
    if (dynamic_thread.joinable()) {
      dynamic_thread.join();
    }

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr current_cloud = StereoCloudToXyzrgb(current_stereo);
    if (!current_cloud || current_cloud->empty()) {
      RCLCPP_WARN(node->get_logger(), "Frame %zu produced an empty point cloud.",
                  sensor_frame_count);
      continue;
    }

    if (!has_prev_keyframe) {
      prev_frame = kf_manager.CreateFirstKeyFrame(
          frame_data, current_cloud, imu_states, h_disp, current_point_confidences,
          std::vector<Eigen::Matrix3d>{}, confidence_map);
      prev_stereo = current_stereo;
      if (!BuildFeatureCache(prev_frame.pointcloud, feature_processor,
                             target_cache, worker_threads)) {
        RCLCPP_ERROR(node->get_logger(), "Failed to build initial feature cache.");
        rclcpp::shutdown();
        return 2;
      }
      has_prev_keyframe = true;
      flow_gate_cov.setZero();
      imu_states.timestamp = frame_data->end_time;
      RCLCPP_INFO(node->get_logger(),
                  "[keyframe] created initial keyframe %d with %zu points",
                  prev_frame.keyframe_id, prev_frame.pointcloud->size());
      PublishCloud(cloud_pub, prev_frame.pointcloud, node, frame_id);
      continue;
    }

    // 成功且静态非空：写回参考云。成功且静态为空：保留上一静态云，不把动态点并回去。
    // 失败：未检验可见点只给本帧跟踪，不覆盖参考云，不可见不进静态。
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr tracking_cloud = prev_frame.pointcloud;
    std::vector<float> tracking_confidences = prev_frame.point_confidences;
    if (dynamic_filter_ok && !static_prev_stereo.empty()) {
      prev_stereo = std::move(static_prev_stereo);
      prev_frame.pointcloud = StereoCloudToXyzrgb(prev_stereo);
      if (static_prev_confidences.size() == prev_frame.pointcloud->size()) {
        prev_frame.point_confidences = static_prev_confidences;
      }
      tracking_cloud = prev_frame.pointcloud;
      tracking_confidences = prev_frame.point_confidences;
      const size_t prev_cloud_after = prev_frame.pointcloud->size();
      const size_t removed = (prev_cloud_before > prev_cloud_after) ? (prev_cloud_before - prev_cloud_after) : 0;
      const double ratio = (prev_cloud_before > 0) ? (100.0 * removed / prev_cloud_before) : 0.0;
      const size_t invisible_count = invisible_prev_cloud ? invisible_prev_cloud->size() : 0;
      RCLCPP_INFO(node->get_logger(),
                  "[dynamic_filter] frame=%zu before=%zu after=%zu removed=%zu ratio=%.1f%% invisible=%zu",
                  processed_frame_count, prev_cloud_before, prev_cloud_after, removed, ratio,
                  invisible_count);
    } else if (dynamic_filter_ok) {
      const size_t invisible_count = invisible_prev_cloud ? invisible_prev_cloud->size() : 0;
      RCLCPP_INFO(node->get_logger(),
                  "[dynamic_filter] frame=%zu success with empty static; keeping previous static cloud, invisible=%zu",
                  processed_frame_count, invisible_count);
    } else if (!static_prev_stereo.empty()) {
      tracking_cloud = StereoCloudToXyzrgb(static_prev_stereo);
      if (static_prev_confidences.size() == tracking_cloud->size()) {
        tracking_confidences = static_prev_confidences;
      } else {
        tracking_confidences.clear();
      }
      const size_t invisible_count = invisible_prev_cloud ? invisible_prev_cloud->size() : 0;
      RCLCPP_WARN(node->get_logger(),
                  "[dynamic_filter] frame=%zu failed; tracking %zu untested visible points, reference unchanged, invisible=%zu",
                  processed_frame_count, tracking_cloud->size(), invisible_count);
    } else {
      RCLCPP_WARN(node->get_logger(),
                  "[dynamic_filter] frame=%zu failed with no visible points; keeping previous cloud",
                  processed_frame_count);
    }

    const double frame_time_diff = frame_data->end_time - prev_frame.timestamp;
    const bool is_front_keyframe = kf_manager.IsFrontKeyFrame(
        imu_propagated, prev_frame.imu_states, frame_time_diff);

    double gicp_trans_norm = -1.0;
    double color_icp_trans_norm = -1.0;

    if (!is_front_keyframe) {
      const auto gicp_result =
          color_icp.EstimatePoseGICP(tracking_cloud, current_cloud,
                                     T_prevkf_curr);
      gicp_trans_norm = gicp_result.T.block<3, 1>(0, 3).norm();

      const Eigen::Matrix4d T_w_curr_cam_gicp =
          T_w_prev_cam * gicp_result.T.cast<double>();
      const Eigen::Matrix4d T_w_curr_imu_gicp =
          T_w_curr_cam_gicp * T_imu_cam.inverse();

      const Eigen::Matrix<double, 6, 6> R_obs_aligned =
          svio_utils::AlignGicpCovarianceCameraToEskfPose(
              gicp_result.covariance, prev_frame.imu_states.rot_end,
              gicp_result.T.cast<double>(), T_imu_cam);

      imu_states = estimator.ESKF(imu_propagated,
                                  T_w_curr_imu_gicp.cast<float>(),
                                  R_obs_aligned, &flow_gate_cov);
      ++eskf_count;
    }

    if (is_front_keyframe) {
      if (!BuildFeatureCache(tracking_cloud, feature_processor,
                             target_cache, worker_threads)) {
        RCLCPP_ERROR(node->get_logger(),
                     "Failed to build target feature cache for frame %zu.",
                     sensor_frame_count);
        rclcpp::shutdown();
        return 3;
      }

      pcl::PointCloud<ColorICP::PointT>::Ptr source_normals_cloud;
      if (!BuildCurrentNormals(current_cloud, feature_processor,
                               source_normals_cloud)) {
        RCLCPP_ERROR(node->get_logger(),
                     "Failed to estimate normals for current source cloud.");
        rclcpp::shutdown();
        return 4;
      }

      if (run_color_icp && !target_cache.thread_kdtrees.empty() &&
          target_cache.thread_kdtrees[0]) {
        const Eigen::Matrix4d color_icp_transform = color_icp.ColorICPRegistration(
            source_normals_cloud, target_cache.normals_cloud,
            target_cache.color_gradient, target_cache.thread_kdtrees,
            T_prevkf_curr);
        color_icp_trans_norm = color_icp_transform.block<3, 1>(0, 3).norm();
      }

      estimator.ESIKF(current_cloud, target_cache.rgb_kdtree,
                      target_cache.color_gradient, target_cache.normals,
                      imu_propagated, prev_frame.imu_states.rot_end,
                      prev_frame.imu_states.pos_end, current_point_confidences,
                      std::vector<float>{}, 0.5, &flow_gate_cov);
      imu_states = imu_propagated;
      ++esikf_count;

      const auto keyframe_imu_buffer = kf_manager.GetImuBufferForKeyFrame();
      auto new_frontend_kf = kf_manager.CreateFrontendKeyFrame(
          prev_frame, frame_data, keyframe_imu_buffer, current_cloud,
          imu_states, h_disp, current_point_confidences,
          std::vector<Eigen::Matrix3d>{}, confidence_map);
      frontend_keyframes.push_back(new_frontend_kf);
      ++frontend_keyframe_count;

      const double backend_time_diff =
          (prev_backend_keyframe.timestamp > 0.0)
              ? (new_frontend_kf->timestamp - prev_backend_keyframe.timestamp)
              : std::numeric_limits<double>::infinity();
      if (kf_manager.IsBackKeyFrame(*new_frontend_kf, frontend_keyframes.size(),
                                    backend_time_diff)) {
        auto backend_kf = kf_manager.CreateBackendKeyFrame(
            prev_backend_keyframe, new_frontend_kf, keyframe_imu_buffer);
        // 后端关键帧使用经过动态剔除的上一帧点云（滞后一帧以换取动态点滤除）
        backend_kf->pointcloud = tracking_cloud;
        backend_kf->point_confidences = tracking_confidences;
        prev_backend_keyframe = *backend_kf;
        ++backend_keyframe_count;
      }

      prev_frame = *new_frontend_kf;
      prev_stereo = current_stereo;
      flow_gate_cov.setZero();
    }

    PublishCloud(cloud_pub, current_cloud, node, frame_id);
    ++processed_frame_count;

    const auto frame_end = std::chrono::steady_clock::now();
    const double frame_ms =
        std::chrono::duration<double, std::milli>(frame_end - frame_begin).count();
    total_process_ms += frame_ms;

    if (processed_frame_count % 10 == 0) {
      const double avg_ms = total_process_ms / static_cast<double>(processed_frame_count);
      const double fps = (avg_ms > 1e-6) ? (1000.0 / avg_ms) : 0.0;
      RCLCPP_INFO(node->get_logger(),
                  "[perf] frame=%zu avg=%.2f ms fps=%.2f gicp_t=%.3f color_icp_t=%.3f eskf=%zu esikf=%zu",
                  processed_frame_count, avg_ms, fps, gicp_trans_norm,
                  color_icp_trans_norm, eskf_count, esikf_count);
    }

    rclcpp::spin_some(node);
  }

  const double avg_ms =
      processed_frame_count > 0
          ? total_process_ms / static_cast<double>(processed_frame_count)
          : 0.0;
  const double avg_fps = (avg_ms > 1e-6) ? (1000.0 / avg_ms) : 0.0;

  RCLCPP_INFO(node->get_logger(),
              "run_demo done: sensor_frames=%zu processed_frames=%zu front_kfs=%zu back_kfs=%zu eskf=%zu esikf=%zu avg=%.2f ms fps=%.2f",
              sensor_frame_count, processed_frame_count, frontend_keyframe_count,
              backend_keyframe_count, eskf_count, esikf_count, avg_ms, avg_fps);

  rclcpp::shutdown();
  return 0;
}

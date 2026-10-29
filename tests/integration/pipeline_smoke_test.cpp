#include "color_icp.h"
#include "dataset_loader.h"
#include "sio_frontend_modules.h"
#include "sio_pipeline_helpers.h"
#include "utils.h"

#include <Eigen/Dense> // 矩阵运算库
#include <algorithm>
#include <cmath>
#include <deque>
#include <filesystem>
#include <iostream>
#include <limits>
#include <numeric>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <rclcpp/rclcpp.hpp> // ROS2 runtime, publisher/Node/Subscriber

namespace {

using sio_pipeline::BuildCurrentNormals;
using sio_pipeline::BuildFeatureCache;
using sio_pipeline::FeatureCache;
using sio_pipeline::ImuMsg;
using sio_pipeline::IsImuDataContinuous;
using sio_pipeline::PublishCloud;

} // namespace

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = rclcpp::Node::make_shared("sio_pipeline_smoketest");
  const auto project_cfg = sio_utils::LoadProjectRuntimeConfig();
  const CameraIntrinsics default_cam =
    sio_utils::LoadInferenceCameraIntrinsics(project_cfg.params_file);

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
  node->declare_parameter<int>("max_frames", 5);
  node->declare_parameter<bool>("run_color_icp", false);

  const auto img_dir = node->get_parameter("img_dir").as_string();
  const auto imu_file = node->get_parameter("imu_file").as_string();
  const auto plan_disp = node->get_parameter("plan_disp").as_string();
  const auto plan_raft = node->get_parameter("plan_raft").as_string();
  const auto topic = node->get_parameter("topic").as_string();
  const auto frame_id = node->get_parameter("frame_id").as_string();
  const int imu_init_samples = node->get_parameter("imu_init_samples").as_int();
  const int max_frames = node->get_parameter("max_frames").as_int();
  const bool run_color_icp = node->get_parameter("run_color_icp").as_bool();
  const Eigen::Matrix4d T_imu_cam =
  sio_utils::LoadImuCameraExtrinsic(project_cfg.params_file);

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
  DynamicCloudProcessor dynamic_processor(plan_disp, plan_raft, cam); //计算视差、生成点云、去动态物体
  dynamic_processor.ConfigureStereoPreprocess(stereo_geom);
  dynamic_processor.SetDynamicFilterParams(
    sio_utils::LoadDynamicFilterParams(params_cfg_path));
  ImuIntegrator imu_integrator;
  FeatureProcessor feature_processor; //法向量、颜色梯度、协方差、匹配
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

  while (rclcpp::ok()) {
    if (max_frames > 0 &&
        processed_frame_count >= static_cast<std::size_t>(max_frames)) {
      break;
    }

    auto frame_data = loader.GrabNextFrame(); // FrameData
    if (!frame_data) {
      break;
    }

    ++sensor_frame_count;

    if (!imu_initialized) { // 默认收集200  IMU sample
      for (const auto &imu_msg : frame_data->imu) {
        auto imu_copy = std::make_shared<sensor_msgs::msg::Imu>(*imu_msg);
        imu_buffer.push_back(imu_copy);
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

    for (const auto &imu_msg : frame_data->imu) {
      auto imu_copy = std::make_shared<sensor_msgs::msg::Imu>(*imu_msg);
      imu_buffer.push_back(imu_copy);
      kf_manager.AddImuData(imu_copy);
      if (imu_buffer.size() > 500) {
        imu_buffer.pop_front();
      }
    }

    cv::Mat confidence_map;
    std::vector<float> current_point_confidences;

    //双目生成3D点云, 视差图 -> 3D点云
    cv::Mat h_disp = dynamic_processor.computeFilteredDisparity(frame_data->imgl,
                                                                frame_data->imgr);
    StereoCloud current_stereo = dynamic_processor.generatePointCloud(
        frame_data->imgl, h_disp, confidence_map, &current_point_confidences);
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
                             target_cache)) {
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

    // 后续帧处理
    KeyFrameData propagation_anchor = prev_frame;
    propagation_anchor.timestamp = (imu_states.timestamp > 0.0)
                                       ? imu_states.timestamp
                                       : prev_frame.timestamp;
    propagation_anchor.imu_states = imu_states;  //复制上一关键帧，作为 IMU 预积分的起点。

    ImuStatesGroup imu_propagated = imu_integrator.ImuPreintegrationCov(
        propagation_anchor, imu_buffer, frame_data->end_time,
        &flow_gate_cov); //从上一状态开始，根据 IMU 积分，预测当前帧状态。

    const Eigen::Matrix4d T_w_prevkf =
      sio_utils::ImuStatesToMatrix4d(prev_frame.imu_states);
    const Eigen::Matrix4d T_w_curr_pred =
      sio_utils::ImuStatesToMatrix4d(imu_propagated);
    const Eigen::Matrix4d T_w_prev_cam = svio_utils::WorldCamFromBody(T_w_prevkf, T_imu_cam);
    const Eigen::Matrix4d T_w_curr_pred_cam =
        svio_utils::WorldCamFromBody(T_w_curr_pred, T_imu_cam);
    const Eigen::Matrix4d T_prevkf_curr =
        svio_utils::RelativeTransform(T_w_prev_cam, T_w_curr_pred_cam);
    const Eigen::Matrix4d T_curr_prevkf = T_prevkf_curr.inverse();

    std::vector<float> static_prev_confidences;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr invisible_prev_cloud;
    StereoCloud static_prev_stereo;
    const Eigen::Matrix<double, 6, 6> pose_cov_cam =
        svio_utils::FlowGatePoseCovInCamera(imu_propagated, flow_gate_cov, T_imu_cam);
    const bool dynamic_filter_ok = dynamic_processor.removeDynamicObjects(
        prev_frame.image_left, prev_stereo, frame_data->imgl,
        T_curr_prevkf, prev_frame.disp_filtered, &prev_frame.point_confidences,
        &static_prev_confidences, nullptr, nullptr, &pose_cov_cam,
        &invisible_prev_cloud, nullptr, &static_prev_stereo);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr tracking_cloud = prev_frame.pointcloud;
    if (dynamic_filter_ok && !static_prev_stereo.empty()) {
      prev_stereo = std::move(static_prev_stereo);
      prev_frame.pointcloud = StereoCloudToXyzrgb(prev_stereo);
      if (static_prev_confidences.size() == prev_frame.pointcloud->size()) {
        prev_frame.point_confidences = static_prev_confidences;
      }
      tracking_cloud = prev_frame.pointcloud;
    } else if (dynamic_filter_ok) {
      RCLCPP_INFO(node->get_logger(),
                  "Dynamic filter succeeded with an empty static cloud; keeping previous static cloud.");
    } else if (!static_prev_stereo.empty()) {
      tracking_cloud = StereoCloudToXyzrgb(static_prev_stereo);
      RCLCPP_WARN(node->get_logger(),
                  "Dynamic filter failed; tracking untested visible points, reference cloud unchanged.");
    } else {
      RCLCPP_WARN(node->get_logger(),
                  "Dynamic filter failed and no visible points remain; keeping previous keyframe cloud.");
    }
    if (invisible_prev_cloud && !invisible_prev_cloud->empty()) {
      RCLCPP_INFO(node->get_logger(),
                  "[dynamic_filter] invisible=%zu (cyan, neither static nor dynamic)",
                  invisible_prev_cloud->size());
    }

    if (!BuildFeatureCache(tracking_cloud, feature_processor,
                           target_cache)) {
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

    std::vector<int> match_indices;
    feature_processor.FindCorrespondences(current_cloud, target_cache.rgb_kdtree,
                                          match_indices, 0.25f);
    const std::size_t matched_points = std::count_if(
        match_indices.begin(), match_indices.end(), [](int idx) { return idx >= 0; });

    std::vector<Eigen::Matrix3d> geo_covariances;
    std::vector<double> color_variances;
    feature_processor.EstimatePointCovariances(current_cloud,
                                               target_cache.color_gradient,
                                               geo_covariances,
                                               color_variances,
                                               current_point_confidences);

    const auto gicp_result = color_icp.EstimatePoseGICP(tracking_cloud,
                                                        current_cloud,
                                                        T_prevkf_curr);

    Eigen::Matrix4d color_icp_transform = T_prevkf_curr;
    if (run_color_icp && !target_cache.thread_kdtrees.empty() &&
        target_cache.thread_kdtrees[0]) {
      color_icp_transform = color_icp.ColorICPRegistration(
          source_normals_cloud, target_cache.normals_cloud,
          target_cache.color_gradient, target_cache.thread_kdtrees,
          T_prevkf_curr);
    }

    ImuStatesGroup imu_fused = imu_propagated;
    estimator.ESIKF(current_cloud, target_cache.rgb_kdtree,
                    target_cache.color_gradient, target_cache.normals, imu_fused,
                    prev_frame.imu_states.rot_end, prev_frame.imu_states.pos_end,
                    std::vector<float>{}, std::vector<float>{}, 0.5, &flow_gate_cov);
    imu_states = imu_fused;

    const double frame_time_diff = frame_data->end_time - prev_frame.timestamp;
    const bool is_front_keyframe = kf_manager.IsFrontKeyFrame(
        imu_propagated, prev_frame.imu_states, frame_time_diff);

    const double avg_geo_var = geo_covariances.empty()
                                   ? 0.0
                                   : std::accumulate(
                                         geo_covariances.begin(), geo_covariances.end(), 0.0,
                                         [](double sum, const Eigen::Matrix3d &cov) {
                                           return sum + cov.trace() / 3.0;
                                         }) /
                                         static_cast<double>(geo_covariances.size());
    const double avg_color_var = color_variances.empty()
                                     ? 0.0
                                     : std::accumulate(color_variances.begin(),
                                                       color_variances.end(), 0.0) /
                                           static_cast<double>(color_variances.size());

    const double gicp_trans_norm = gicp_result.T.block<3, 1>(0, 3).norm();
    const double color_icp_trans_norm =
      color_icp_transform.block<3, 1>(0, 3).norm();

    RCLCPP_INFO(node->get_logger(),
                "[frame %zu] cloud=%zu matched=%zu avg_geo_var=%.6f avg_color_var=%.6f gicp_t=%.3f color_icp_t=%.3f front_kf=%d",
                sensor_frame_count, current_cloud->size(), matched_points,
          avg_geo_var, avg_color_var, gicp_trans_norm,
          color_icp_trans_norm,
                is_front_keyframe ? 1 : 0);

    if (is_front_keyframe) {
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
        prev_backend_keyframe = *backend_kf;
        ++backend_keyframe_count;
        RCLCPP_INFO(node->get_logger(),
                    "[backend] promoted keyframe %d as backend keyframe %d",
                    new_frontend_kf->keyframe_id, backend_kf->keyframe_id);
      }

      prev_frame = *new_frontend_kf;
      prev_stereo = current_stereo;
      flow_gate_cov.setZero();
      if (!BuildFeatureCache(prev_frame.pointcloud, feature_processor,
                             target_cache)) {
        RCLCPP_ERROR(node->get_logger(),
                     "Failed to refresh feature cache after keyframe update.");
        rclcpp::shutdown();
        return 5;
      }
    }

    PublishCloud(cloud_pub, current_cloud, node, frame_id);
    ++processed_frame_count;
    rclcpp::spin_some(node);
  }

  RCLCPP_INFO(node->get_logger(),
              "Pipeline smoke test complete: sensor_frames=%zu processed_frames=%zu frontend_kfs=%zu backend_kfs=%zu",
              sensor_frame_count, processed_frame_count, frontend_keyframe_count,
              backend_keyframe_count);
  rclcpp::shutdown();
  return 0;
}

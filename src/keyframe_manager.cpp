#include "sio_frontend_modules.h"
#include "utils.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>

// KeyframeManager constructor: load default config file
KeyframeManager::KeyframeManager() {
  const std::filesystem::path default_cfg_path = sio_utils::LocateParamsFile();
  if (!default_cfg_path.empty()) {
    cfg_.params_file_path = default_cfg_path.string();
    LoadConfig(default_cfg_path.string());
  }
}

// KeyframeManager constructor with custom config
KeyframeManager::KeyframeManager(const KeyframeManager::Config &cfg) : cfg_(cfg) {
  if (!cfg_.params_file_path.empty()) {
    LoadConfig(cfg_.params_file_path);
  }
}

// Load and parse configuration from YAML file
bool KeyframeManager::LoadConfig(const std::string &path) {
  if (path.empty() || !std::filesystem::exists(path)) {
    return false;
  }

  const std::filesystem::path cfg_path(path);
  sio_utils::ParseYamlScalarDouble(cfg_path, "front_trans_thresh_m", cfg_.front_trans_thresh_m);
  sio_utils::ParseYamlScalarDouble(cfg_path, "front_angle_thresh_deg", cfg_.front_angle_thresh_deg);
  sio_utils::ParseYamlScalarDouble(cfg_path, "front_time_thresh_s", cfg_.front_time_thresh_s);
  sio_utils::ParseYamlScalarDouble(cfg_path, "speed_ref_mps", cfg_.speed_ref_mps);
  sio_utils::ParseYamlScalarDouble(cfg_path, "adaptive_scale_min", cfg_.adaptive_scale_min);
  sio_utils::ParseYamlScalarDouble(cfg_path, "adaptive_scale_max", cfg_.adaptive_scale_max);
  sio_utils::ParseYamlScalarDouble(cfg_path, "backend_trans_thresh_m", cfg_.backend_trans_thresh_m);
  sio_utils::ParseYamlScalarDouble(cfg_path, "backend_angle_thresh_deg",
                                    cfg_.backend_angle_thresh_deg);
  sio_utils::ParseYamlScalarDouble(cfg_path, "backend_time_thresh_s", cfg_.backend_time_thresh_s);

  double min_frontend = static_cast<double>(cfg_.backend_min_frontend_kfs);
  if (sio_utils::ParseYamlScalarDouble(cfg_path, "backend_min_frontend_kfs", min_frontend)) {
    cfg_.backend_min_frontend_kfs = static_cast<size_t>(std::max(1.0, min_frontend));
  }

  double legacy_trigger = static_cast<double>(cfg_.backend_queue_legacy_trigger);
  if (sio_utils::ParseYamlScalarDouble(cfg_path, "backend_queue_legacy_trigger", legacy_trigger)) {
    cfg_.backend_queue_legacy_trigger = static_cast<size_t>(std::max(1.0, legacy_trigger));
  }

  if (cfg_.adaptive_scale_min > cfg_.adaptive_scale_max) {
    std::swap(cfg_.adaptive_scale_min, cfg_.adaptive_scale_max);
  }
  return true;
}
// Compute geodesic rotation distance between two rotation matrices
double KeyframeManager::RotationAngleBetween(const Eigen::Matrix3d &R_cur,
                                             const Eigen::Matrix3d &R_last) {
  Eigen::Matrix3d dR = R_cur * R_last.transpose();
  const double cos_angle = std::clamp((dR.trace() - 1.0) / 2.0, -1.0, 1.0);
  return std::acos(cos_angle);
}

// Create first keyframe from initial frame data
KeyFrameData KeyframeManager::CreateFirstKeyFrame(
  const std::shared_ptr<FrameData> &frame_data, const pcl::PointCloud<PointTypeRGB>::Ptr &pcd_cur,
  const ImuStatesGroup &imu_states, const cv::Mat &disp_filtered,
  const std::vector<float> &point_confidences,
  const std::vector<Eigen::Matrix3d> &point_covariances, const cv::Mat &confidence_map) {
  KeyFrameData first;

  // 开始时间与结束时间都用 end_time；
  //  如果frame包含同步的imu数据，转移到 CreateFrontendKeyFrame
  //  里，保持一致性，如果没有imu数据，CreateFrontendKeyFrame
  //  imu数据为空，保持兼容性
  if (frame_data) {
    first.last_keyframe_time = frame_data->end_time;
    first.timestamp = frame_data->end_time;
    first.image_left = frame_data->imgl.clone();
    first.image_right = frame_data->imgr.clone();
  }

  // 判断有没有时间跟frame_data.end_time对齐的imu数据；有imu数据加入
  // first.imu_buffer，没有就保持空；如果有但不连续，保持空
  if (frame_data && !frame_data->imu.empty()) {
    for (const auto &imu_msg : frame_data->imu) {
      if (std::abs((imu_msg->header.stamp.sec + imu_msg->header.stamp.nanosec * 1e-9)
                   - frame_data->end_time)
          < 0.001) {
        first.imu_buffer.push_back(std::make_shared<sensor_msgs::msg::Imu>(*imu_msg));
      }
    }
  }

  first.pointcloud = pcd_cur;
  first.imu_states = imu_states;
  first.disp_filtered = disp_filtered;
  first.point_confidences = point_confidences;
  first.point_covariances = point_covariances;
  first.confidence_map = confidence_map.clone();
  first.keyframe_id = frontend_keyframe_id_;
  first.frontend_keyframe_id = frontend_keyframe_id_++;
  first.prev_frontend_keyframe_id = -1;
  first.has_relative_edge = false;
  first.is_backend_keyframe = false;

  prev_frame_ = first;
  return first;
}

// Create frontend keyframe with accumulated IMU data
std::shared_ptr<KeyFrameData> KeyframeManager::CreateFrontendKeyFrame(
  const KeyFrameData &prev_frame, const std::shared_ptr<FrameData> &curr_frame,
  const std::deque<sensor_msgs::msg::Imu::SharedPtr> &imu_buffer,
  pcl::PointCloud<PointTypeRGB>::Ptr pcd_cur, const ImuStatesGroup &imu_states,
  const cv::Mat &disp_filtered, const std::vector<float> &point_confidences,
  const std::vector<Eigen::Matrix3d> &point_covariances, const cv::Mat &confidence_map) {

  auto kf = std::make_shared<KeyFrameData>();
  kf->last_keyframe_time = prev_frame.timestamp;

  if (curr_frame) {
    kf->timestamp = curr_frame->end_time; // IMU 预积分终止时刻，与其余时间锚点一致
    kf->image_left = curr_frame->imgl.clone();
    kf->image_right = curr_frame->imgr.clone();
  }

  kf->pointcloud = pcd_cur;
  kf->imu_states = imu_states;
  kf->imu_buffer = imu_buffer;
  kf->disp_filtered = disp_filtered;
  kf->point_confidences = point_confidences;
  kf->point_covariances = point_covariances;
  kf->confidence_map = confidence_map.clone();

  kf->keyframe_id = frontend_keyframe_id_;
  kf->frontend_keyframe_id = frontend_keyframe_id_++;
  kf->prev_frontend_keyframe_id = prev_frame.frontend_keyframe_id;
  // 记录 ESIKF 后验相对边，供后端绝对修正后沿链复合（不重跑历史 IMU）
  {
    const Eigen::Matrix4d T_w_prev = svio_utils::ImuStatesToMatrix4d(prev_frame.imu_states);
    const Eigen::Matrix4d T_w_curr = svio_utils::ImuStatesToMatrix4d(imu_states);
    kf->relative_T_prev_curr = svio_utils::RelativeTransform(T_w_prev, T_w_curr);
    kf->relative_v_prev_curr_body =
      prev_frame.imu_states.rot_end.transpose() * (imu_states.vel_end - prev_frame.imu_states.vel_end);
    kf->has_relative_edge = true;
  }
  kf->is_backend_keyframe = false;

  prev_frame_ = *kf;

  return kf;
}

// Create backend keyframe for sliding window optimization
std::shared_ptr<KeyFrameData> KeyframeManager::CreateBackendKeyFrame(
  const KeyFrameData &prev_backend_keyframe, std::shared_ptr<KeyFrameData> latest_frontend_kf,
  const std::deque<sensor_msgs::msg::Imu::SharedPtr> &imu_buffer) {

  auto kf = std::make_shared<KeyFrameData>();

  if (latest_frontend_kf) {
    *kf = *latest_frontend_kf; // 拷贝前端关键帧信息（保留 frontend_keyframe_id / 相对边）
  }

  kf->last_keyframe_time = prev_backend_keyframe.timestamp;
  kf->imu_buffer = imu_buffer;
  kf->is_backend_keyframe = true;
  // 后端 ID 覆盖 keyframe_id；frontend_keyframe_id 保持不变供回传匹配
  kf->keyframe_id = backend_keyframe_id_++;

  prev_backend_keyframe_ = *kf;
  return kf;
}
FrontKeyframeDecision KeyframeManager::EvaluateFrontKeyframeDecision(
  const ImuStatesGroup &imu_progated, const ImuStatesGroup &last_fused, double time_diff) const {
  FrontKeyframeDecision decision;
  decision.time_diff = time_diff;
  decision.trans_dist = (imu_progated.pos_end - last_fused.pos_end).norm();
  const double angle_diff = RotationAngleBetween(imu_progated.rot_end, last_fused.rot_end);
  decision.angle_deg = angle_diff * 180.0 / M_PI;
  decision.speed = imu_progated.vel_end.norm();
  decision.trans_trigger = decision.trans_dist > cfg_.front_trans_thresh_m;
  decision.rot_trigger = angle_diff > cfg_.front_angle_thresh_deg * M_PI / 180.0;
  decision.time_trigger = time_diff > cfg_.front_time_thresh_s;
  decision.speed_trigger = decision.speed > cfg_.speed_ref_mps * cfg_.adaptive_scale_max;
  decision.is_keyframe =
    decision.trans_trigger || decision.rot_trigger || decision.time_trigger || decision.speed_trigger;
  return decision;
}

// Check if new frame should be selected as frontend keyframe
bool KeyframeManager::IsFrontKeyFrame(const ImuStatesGroup &imu_progated,
                                      const ImuStatesGroup &last_fused, double time_diff) const {
  return EvaluateFrontKeyframeDecision(imu_progated, last_fused, time_diff).is_keyframe;
}

// Check if frontend keyframe should be selected as backend keyframe
bool KeyframeManager::IsBackKeyFrame(const KeyFrameData &latest_frontend_kf,
                                     size_t frontend_deque_size,
                                     double time_diff_from_last_backend) const {
  // 0) 数量判据：先满足最小前端关键帧数量
  if (frontend_deque_size < cfg_.backend_min_frontend_kfs) {
    return false;
  }

  // 1) 冷启动判据：尚无后端关键帧时沿用队列触发
  if (prev_backend_keyframe_.timestamp <= 0.0) {
    return frontend_deque_size >= cfg_.backend_queue_legacy_trigger;
  }

  // 2) 位移判据（固定阈值）
  const Eigen::Vector3d t_last = prev_backend_keyframe_.imu_states.pos_end;
  const Eigen::Vector3d t_cur = latest_frontend_kf.imu_states.pos_end;
  const double trans_dist = (t_cur - t_last).norm();
  const bool trans_trigger = trans_dist > cfg_.backend_trans_thresh_m;

  // 3) 旋转判据：SO(3) 测地线距离
  const Eigen::Matrix3d R_last = prev_backend_keyframe_.imu_states.rot_end;
  const Eigen::Matrix3d R_cur = latest_frontend_kf.imu_states.rot_end;
  const double angle_diff = RotationAngleBetween(R_cur, R_last);
  const double angle_thresh = cfg_.backend_angle_thresh_deg * M_PI / 180.0;
  const bool rot_trigger = angle_diff > angle_thresh;

  // 4) 时间判据：兜底，防止后端长期不触发
  const bool time_trigger = time_diff_from_last_backend > cfg_.backend_time_thresh_s;

  return trans_trigger || rot_trigger || time_trigger;
}

bool KeyframeManager::IsBackKeyFrame(size_t frontend_deque_size) const {
  return frontend_deque_size >= cfg_.backend_queue_legacy_trigger;
}

// Add single IMU measurement to frontend buffer
void KeyframeManager::AddImuData(const sensor_msgs::msg::Imu::SharedPtr &imu_msg) {
  std::lock_guard<std::mutex> lock(mtx_imu_buffer_);
  front_imu_buffer_.push_back(imu_msg);
}

// Clear frontend IMU buffer
void KeyframeManager::ClearImuBuffer() {
  std::lock_guard<std::mutex> lock(mtx_imu_buffer_);
  front_imu_buffer_.clear();
}

// Get accumulated IMU buffer for keyframe and clear it
std::deque<sensor_msgs::msg::Imu::SharedPtr> KeyframeManager::GetImuBufferForKeyFrame() {
  std::lock_guard<std::mutex> lock(mtx_imu_buffer_);
  std::deque<sensor_msgs::msg::Imu::SharedPtr> temp_buffer = front_imu_buffer_;
  front_imu_buffer_.clear(); // 或者保留最新的一个供下次积分使用
  return temp_buffer;
}

// Extract IMU data within time range [t0, t1]
std::deque<sensor_msgs::msg::Imu::SharedPtr> KeyframeManager::ExtractImuBetween(double t0,
                                                                                double t1) {
  std::lock_guard<std::mutex> lock(mtx_imu_buffer_);
  std::deque<sensor_msgs::msg::Imu::SharedPtr> extracted;

  // 遍历IMU队列，提取 [t0, t1] 范围内的 IMU 数据
  for (const auto &imu : front_imu_buffer_) {
    double current_t = svio_utils::MsgTimeSec(imu);
    if (current_t >= t0 && current_t <= t1) {
      extracted.push_back(imu);
    }
  }

  return extracted;
}

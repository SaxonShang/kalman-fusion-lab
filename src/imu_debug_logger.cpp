#include "imu_debug.h"

#include "utils.h"

#include <Eigen/Dense>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <yaml-cpp/yaml.h>

namespace {

double MsgTimeSecOrZero(const sensor_msgs::msg::Imu::SharedPtr &msg) {
  if (!msg) {
    return 0.0;
  }
  return svio_utils::MsgTimeSec(msg);
}

} // namespace

bool LoadDebugLoggerConfig(const std::string &yaml_path, DebugLoggerConfig &cfg) {
  if (yaml_path.empty()) {
    return false;
  }

  try {
    YAML::Node root = YAML::LoadFile(yaml_path);
    if (!root["debug"]) {
      return false;
    }
    const YAML::Node node = root["debug"];
    if (node["fusion_csv"]) {
      cfg.fusion_csv_path = node["fusion_csv"].as<std::string>();
    }
    if (node["imu_csv"]) {
      cfg.imu_csv_path = node["imu_csv"].as<std::string>();
    }
    cfg.console = node["console"].as<bool>(false);
    cfg.track_console = node["track_console"].as<bool>(false);
    return true;
  } catch (const std::exception &e) {
    std::cerr << "[ImuDebug] Failed to load debug config: " << e.what() << std::endl;
    return false;
  }
}

ImuFrameImuStats ComputeFrameImuStats(
    const std::deque<sensor_msgs::msg::Imu::SharedPtr> &frame_imu) {
  ImuFrameImuStats stats;
  stats.frame_imu_count = static_cast<int>(frame_imu.size());
  if (frame_imu.empty()) {
    return stats;
  }

  std::vector<double> dts;
  std::vector<double> acc_norms;
  std::vector<double> gyro_norms;
  dts.reserve(frame_imu.size());
  acc_norms.reserve(frame_imu.size());
  gyro_norms.reserve(frame_imu.size());

  for (size_t i = 0; i < frame_imu.size(); ++i) {
    const auto &msg = frame_imu[i];
    if (!msg) {
      continue;
    }
    const Eigen::Vector3d acc(msg->linear_acceleration.x,
                              msg->linear_acceleration.y,
                              msg->linear_acceleration.z);
    const Eigen::Vector3d gyro(msg->angular_velocity.x,
                               msg->angular_velocity.y,
                               msg->angular_velocity.z);
    acc_norms.push_back(acc.norm());
    gyro_norms.push_back(gyro.norm());
    if (i > 0) {
      const double dt =
          MsgTimeSecOrZero(frame_imu[i]) - MsgTimeSecOrZero(frame_imu[i - 1]);
      if (dt > 0.0) {
        dts.push_back(dt);
      }
    }
  }

  if (!dts.empty()) {
    stats.imu_dt_min = *std::min_element(dts.begin(), dts.end());
    stats.imu_dt_max = *std::max_element(dts.begin(), dts.end());
    stats.imu_dt_mean =
        std::accumulate(dts.begin(), dts.end(), 0.0) /
        static_cast<double>(dts.size());
  }

  if (!acc_norms.empty()) {
    stats.acc_norm_mean =
        std::accumulate(acc_norms.begin(), acc_norms.end(), 0.0) /
        static_cast<double>(acc_norms.size());
    double acc_var = 0.0;
    for (double v : acc_norms) {
      const double d = v - stats.acc_norm_mean;
      acc_var += d * d;
    }
    stats.acc_norm_std =
        std::sqrt(acc_var / static_cast<double>(acc_norms.size()));
  }

  if (!gyro_norms.empty()) {
    stats.gyro_norm_mean =
        std::accumulate(gyro_norms.begin(), gyro_norms.end(), 0.0) /
        static_cast<double>(gyro_norms.size());
    double gyro_var = 0.0;
    for (double v : gyro_norms) {
      const double d = v - stats.gyro_norm_mean;
      gyro_var += d * d;
    }
    stats.gyro_norm_std =
        std::sqrt(gyro_var / static_cast<double>(gyro_norms.size()));
  }

  return stats;
}

double RotationAngleDegFromDeltaR(const double delta_R[9]) {
  Eigen::Matrix3d dR;
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      dR(r, c) = delta_R[r * 3 + c];
    }
  }
  const double cos_angle =
      std::clamp((dR.trace() - 1.0) / 2.0, -1.0, 1.0);
  return std::acos(cos_angle) * 180.0 / M_PI;
}

ImuKeyframeDebugRecord BuildImuKeyframeDebugRecord(
    int frame_id, double frame_end_ts,
    const std::deque<sensor_msgs::msg::Imu::SharedPtr> &frame_imu,
    const FrontKeyframeDecision &kf, const ImuStatesGroup &prop_state,
    const Pose6D &preint, double gravity_norm, double preint_total_dt, double dt_tail,
    int imu_between_count, double anchor_ts, double prev_kf_ts, double last_imu_ts) {
  ImuKeyframeDebugRecord record;
  record.frame_id = frame_id;
  record.frame_end_ts = frame_end_ts;
  record.frame_imu = ComputeFrameImuStats(frame_imu);
  record.gravity_norm = gravity_norm;
  record.preint_total_dt = preint_total_dt;
  record.dt_tail = dt_tail;
  record.imu_between_count = imu_between_count;
  record.delta_p_norm = std::sqrt(preint.delta_p[0] * preint.delta_p[0] +
                                  preint.delta_p[1] * preint.delta_p[1] +
                                  preint.delta_p[2] * preint.delta_p[2]);
  record.delta_v_norm = std::sqrt(preint.delta_v[0] * preint.delta_v[0] +
                                  preint.delta_v[1] * preint.delta_v[1] +
                                  preint.delta_v[2] * preint.delta_v[2]);
  record.delta_rot_deg = RotationAngleDegFromDeltaR(preint.delta_R);
  record.prop_pos_x = prop_state.pos_end.x();
  record.prop_pos_y = prop_state.pos_end.y();
  record.prop_pos_z = prop_state.pos_end.z();
  record.prop_vel_x = prop_state.vel_end.x();
  record.prop_vel_y = prop_state.vel_end.y();
  record.prop_vel_z = prop_state.vel_end.z();
  record.kf = kf;
  record.anchor_ts = anchor_ts;
  record.prev_kf_ts = prev_kf_ts;
  record.last_imu_ts = last_imu_ts;
  record.anchor_to_end = frame_end_ts - anchor_ts;
  record.last_imu_to_end = frame_end_ts - last_imu_ts;
  return record;
}

void ImuDebugLogger::Open(const std::string &path) {
  if (path.empty()) {
    output_.reset();
    header_written_ = false;
    return;
  }

  const std::filesystem::path out_path(path);
  if (out_path.has_parent_path()) {
    std::filesystem::create_directories(out_path.parent_path());
  }

  output_ = std::make_unique<std::ofstream>(path, std::ios::out | std::ios::trunc);
  header_written_ = false;
  if (!output_ || !(*output_)) {
    output_.reset();
    return;
  }
  WriteHeader();
}

void ImuDebugLogger::WriteHeader() {
  if (!output_ || header_written_) {
    return;
  }
  (*output_) << "frame_id,frame_end_ts,frame_imu_count,imu_dt_min,imu_dt_max,"
                "imu_dt_mean,acc_norm_mean,acc_norm_std,gyro_norm_mean,"
                "gyro_norm_std,gravity_norm,preint_total_dt,dt_tail,"
                "imu_between_count,delta_p_norm,delta_v_norm,delta_rot_deg,"
                "prop_pos_x,prop_pos_y,prop_pos_z,prop_vel_x,prop_vel_y,"
                "prop_vel_z,is_keyframe,trans_dist,angle_deg,speed,time_diff,"
                "trigger_trans,trigger_rot,trigger_time,trigger_speed,"
                "anchor_ts,prev_kf_ts,last_imu_ts,anchor_to_end,last_imu_to_end\n";
  header_written_ = true;
}

void ImuDebugLogger::WriteRecord(const ImuKeyframeDebugRecord &record) {
  if (!output_) {
    return;
  }
  std::lock_guard<std::mutex> lock(mtx_);
  WriteHeader();
  (*output_) << record.frame_id << ',' << record.frame_end_ts << ','
             << record.frame_imu.frame_imu_count << ','
             << record.frame_imu.imu_dt_min << ',' << record.frame_imu.imu_dt_max
             << ',' << record.frame_imu.imu_dt_mean << ','
             << record.frame_imu.acc_norm_mean << ','
             << record.frame_imu.acc_norm_std << ','
             << record.frame_imu.gyro_norm_mean << ','
             << record.frame_imu.gyro_norm_std << ',' << record.gravity_norm
             << ',' << record.preint_total_dt << ',' << record.dt_tail << ','
             << record.imu_between_count << ',' << record.delta_p_norm << ','
             << record.delta_v_norm << ',' << record.delta_rot_deg << ','
             << record.prop_pos_x << ',' << record.prop_pos_y << ','
             << record.prop_pos_z << ',' << record.prop_vel_x << ','
             << record.prop_vel_y << ',' << record.prop_vel_z << ','
             << (record.kf.is_keyframe ? 1 : 0) << ',' << record.kf.trans_dist
             << ',' << record.kf.angle_deg << ',' << record.kf.speed << ','
             << record.kf.time_diff << ','
             << (record.kf.trans_trigger ? 1 : 0) << ','
             << (record.kf.rot_trigger ? 1 : 0) << ','
             << (record.kf.time_trigger ? 1 : 0) << ','
             << (record.kf.speed_trigger ? 1 : 0) << ',' << record.anchor_ts
             << ',' << record.prev_kf_ts << ',' << record.last_imu_ts << ','
             << record.anchor_to_end << ',' << record.last_imu_to_end << '\n';
  output_->flush();
}

std::string ImuDebugLogger::FormatTriggerSummary(
    const FrontKeyframeDecision &decision) {
  std::ostringstream oss;
  bool first = true;
  auto append = [&](const char *name, bool on) {
    if (!on) {
      return;
    }
    if (!first) {
      oss << '+';
    }
    oss << name;
    first = false;
  };
  append("trans", decision.trans_trigger);
  append("rot", decision.rot_trigger);
  append("time", decision.time_trigger);
  append("speed", decision.speed_trigger);
  if (first) {
    return "none";
  }
  return oss.str();
}

std::string ImuDebugLogger::FormatConsoleLine(
    const ImuKeyframeDebugRecord &record) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(3);
  oss << "[imu_debug] frame=" << record.frame_id << " kf="
      << (record.kf.is_keyframe ? 1 : 0) << " trigger="
      << FormatTriggerSummary(record.kf) << " imu_n="
      << record.frame_imu.frame_imu_count << " dt=" << record.preint_total_dt
      << " dt_tail=" << record.dt_tail << " trans=" << record.kf.trans_dist
      << "m rot=" << record.kf.angle_deg << "deg speed=" << record.kf.speed
      << "m/s acc_norm=" << record.frame_imu.acc_norm_mean << " gyro="
      << record.frame_imu.gyro_norm_mean;
  return oss.str();
}

std::string ImuDebugLogger::FormatPreintSummaryLine(int frame_id, const Pose6D &preint,
                                                    bool anchor_is_prev_keyframe,
                                                    int imu_between_count,
                                                    const FrontKeyframeDecision &kf) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(3);
  const double delta_p_norm = std::sqrt(preint.delta_p[0] * preint.delta_p[0] +
                                        preint.delta_p[1] * preint.delta_p[1] +
                                        preint.delta_p[2] * preint.delta_p[2]);
  const double delta_v_norm = std::sqrt(preint.delta_v[0] * preint.delta_v[0] +
                                        preint.delta_v[1] * preint.delta_v[1] +
                                        preint.delta_v[2] * preint.delta_v[2]);
  const double delta_rot_deg = RotationAngleDegFromDeltaR(preint.delta_R);
  oss << "[IMU] frame=" << frame_id << " anchor="
      << (anchor_is_prev_keyframe ? "prev_kf" : "prev_frame") << " dt=" << preint.dt
      << " delta_p=" << delta_p_norm << " delta_v=" << delta_v_norm
      << " delta_rot=" << delta_rot_deg << "deg imu_between=" << imu_between_count
      << " kf=" << (kf.is_keyframe ? 1 : 0);
  if (kf.is_keyframe) {
    oss << " trigger=" << FormatTriggerSummary(kf);
  }
  return oss.str();
}

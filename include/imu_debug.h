#pragma once

#include "data.h"

#include <deque>
#include <fstream>
#include <memory>
#include <mutex>
#include <sensor_msgs/msg/imu.hpp>
#include <string>

struct ImuFrameImuStats {
  int frame_imu_count = 0;
  double imu_dt_min = 0.0;
  double imu_dt_max = 0.0;
  double imu_dt_mean = 0.0;
  double acc_norm_mean = 0.0;
  double acc_norm_std = 0.0;
  double gyro_norm_mean = 0.0;
  double gyro_norm_std = 0.0;
};

struct FrontKeyframeDecision {
  bool is_keyframe = false;
  double trans_dist = 0.0;
  double angle_deg = 0.0;
  double speed = 0.0;
  double time_diff = 0.0;
  bool trans_trigger = false;
  bool rot_trigger = false;
  bool time_trigger = false;
  bool speed_trigger = false;
};

struct ImuKeyframeDebugRecord {
  int frame_id = -1;
  double frame_end_ts = 0.0;
  ImuFrameImuStats frame_imu;
  double gravity_norm = 0.0;
  double preint_total_dt = 0.0;
  double dt_tail = 0.0;
  int imu_between_count = 0;
  double delta_p_norm = 0.0;
  double delta_v_norm = 0.0;
  double delta_rot_deg = 0.0;
  double prop_pos_x = 0.0;
  double prop_pos_y = 0.0;
  double prop_pos_z = 0.0;
  double prop_vel_x = 0.0;
  double prop_vel_y = 0.0;
  double prop_vel_z = 0.0;
  FrontKeyframeDecision kf;
  double anchor_ts = 0.0;
  double prev_kf_ts = 0.0;
  double last_imu_ts = 0.0;
  double anchor_to_end = 0.0;
  double last_imu_to_end = 0.0;
};

struct DebugLoggerConfig {
  std::string fusion_csv_path;
  std::string imu_csv_path;
  bool console = false;
  bool track_console = false;
};

ImuFrameImuStats ComputeFrameImuStats(
    const std::deque<sensor_msgs::msg::Imu::SharedPtr> &frame_imu);

double RotationAngleDegFromDeltaR(const double delta_R[9]);

ImuKeyframeDebugRecord BuildImuKeyframeDebugRecord(
    int frame_id, double frame_end_ts,
    const std::deque<sensor_msgs::msg::Imu::SharedPtr> &frame_imu,
    const FrontKeyframeDecision &kf, const ImuStatesGroup &prop_state,
    const Pose6D &preint, double gravity_norm, double preint_total_dt, double dt_tail,
    int imu_between_count, double anchor_ts, double prev_kf_ts, double last_imu_ts);

bool LoadDebugLoggerConfig(const std::string &yaml_path, DebugLoggerConfig &cfg);

class ImuDebugLogger {
public:
  void Open(const std::string &path);
  void WriteRecord(const ImuKeyframeDebugRecord &record);
  static std::string FormatTriggerSummary(const FrontKeyframeDecision &decision);
  static std::string FormatConsoleLine(const ImuKeyframeDebugRecord &record);
  static std::string FormatPreintSummaryLine(int frame_id, const Pose6D &preint,
                                             bool anchor_is_prev_keyframe, int imu_between_count,
                                             const FrontKeyframeDecision &kf);

private:
  void WriteHeader();

  std::unique_ptr<std::ofstream> output_;
  bool header_written_ = false;
  std::mutex mtx_;
};

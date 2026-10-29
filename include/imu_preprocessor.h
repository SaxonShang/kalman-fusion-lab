#pragma once

/**
 * @file imu_preprocessor.h
 * @brief IMU 低通滤波与野值剔除（可选，供前端预积分前使用）。
 */

#include <Eigen/Core>
#include <deque>
#include <memory>
#include <sensor_msgs/msg/imu.hpp>
#include <string>

struct ImuPreprocessorConfig {
  bool enable = false;
  bool enable_lpf = true;
  bool enable_outlier_reject = true;
  double sample_rate_hz = 400.0;
  double lpf_cutoff_hz = 20.0;
  double gyro_outlier_rad_s = 25.0;
  double acc_outlier_mps2 = 30.0;
  std::string params_file_path;
};

class ImuPreprocessor {
public:
  /** @brief 应用配置并重置内部滤波状态。 */
  void Configure(const ImuPreprocessorConfig &config);
  /** @brief 清空 LPF / 上一有效样本状态。 */
  void Reset();

  /**
   * @brief 处理单条 IMU：可选野值替换 + 低通。
   * @param raw 原始消息
   * @return 处理后消息（可能与输入共享或新建）
   */
  sensor_msgs::msg::Imu::SharedPtr Process(const sensor_msgs::msg::Imu::SharedPtr &raw);

  /**
   * @brief 批量处理 IMU 序列。
   * @param inputs 原始队列
   * @return 处理后队列
   */
  std::deque<sensor_msgs::msg::Imu::SharedPtr>
  ProcessBatch(const std::deque<sensor_msgs::msg::Imu::SharedPtr> &inputs);

  const ImuPreprocessorConfig &GetConfig() const { return config_; }
  /** @brief 当前估计的重力模长 (m/s²)。 */
  double GravityNormEstimate() const { return gravity_norm_est_; }

  /**
   * @brief 从 YAML 加载配置。
   * @param yaml_path 配置路径
   * @return 成功为 true
   */
  bool LoadConfig(const std::string &yaml_path);

private:
  /** @brief 由采样率与截止频率算一阶 LPF α。 */
  double LpfAlpha() const;
  /** @brief 加计/陀螺是否超出野值阈值。 */
  bool IsOutlier(const Eigen::Vector3d &acc, const Eigen::Vector3d &gyro) const;
  /** @brief 用加计模长平滑更新重力估计。 */
  void UpdateGravityEstimate(double acc_norm);

  ImuPreprocessorConfig config_;
  bool has_lpf_state_ = false;
  bool has_last_good_raw_ = false;
  Eigen::Vector3d lpf_acc_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d lpf_gyro_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d last_good_raw_acc_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d last_good_raw_gyro_ = Eigen::Vector3d::Zero();
  double gravity_norm_est_ = 9.81;
  double last_sample_rate_hz_ = 0.0;
};

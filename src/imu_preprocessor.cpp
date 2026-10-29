#include "imu_preprocessor.h"

#include "utils.h"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <yaml-cpp/yaml.h>

namespace {

// 从 ROS Imu 消息提取加速度向量 (m/s²)。
Eigen::Vector3d ImuAcc(const sensor_msgs::msg::Imu::SharedPtr &msg) {
  return Eigen::Vector3d(msg->linear_acceleration.x, msg->linear_acceleration.y,
                         msg->linear_acceleration.z);
}

// 从 ROS Imu 消息提取角速度向量 (rad/s)。
Eigen::Vector3d ImuGyro(const sensor_msgs::msg::Imu::SharedPtr &msg) {
  return Eigen::Vector3d(msg->angular_velocity.x, msg->angular_velocity.y,
                         msg->angular_velocity.z);
}

// 深拷贝 IMU 消息（保留 header 与时间戳）。
sensor_msgs::msg::Imu::SharedPtr CloneImu(
    const sensor_msgs::msg::Imu::SharedPtr &src) {
  return std::make_shared<sensor_msgs::msg::Imu>(*src);
}

// 复用 stamp_src 的 header，写入新的 acc/gyro 量测值。
sensor_msgs::msg::Imu::SharedPtr MakeImuFromVectors(
    const sensor_msgs::msg::Imu::SharedPtr &stamp_src,
    const Eigen::Vector3d &acc, const Eigen::Vector3d &gyro) {
  auto out = CloneImu(stamp_src);
  out->linear_acceleration.x = acc.x();
  out->linear_acceleration.y = acc.y();
  out->linear_acceleration.z = acc.z();
  out->angular_velocity.x = gyro.x();
  out->angular_velocity.y = gyro.y();
  out->angular_velocity.z = gyro.z();
  return out;
}

} // namespace

bool ImuPreprocessor::LoadConfig(const std::string &yaml_path) {
  if (yaml_path.empty()) {
    return false;
  }

  try {
    YAML::Node root = YAML::LoadFile(yaml_path);
    if (!root["imu_preprocessor"]) {
      return false;
    }
    const YAML::Node node = root["imu_preprocessor"];
    ImuPreprocessorConfig cfg;
    cfg.enable = node["enable"].as<bool>(false);
    cfg.enable_lpf = node["enable_lpf"].as<bool>(true);
    cfg.enable_outlier_reject = node["enable_outlier_reject"].as<bool>(true);
    cfg.sample_rate_hz = node["sample_rate_hz"].as<double>(400.0);
    cfg.lpf_cutoff_hz = node["lpf_cutoff_hz"].as<double>(20.0);
    cfg.gyro_outlier_rad_s = node["gyro_outlier_rad_s"].as<double>(25.0);
    cfg.acc_outlier_mps2 = node["acc_outlier_mps2"].as<double>(30.0);
    cfg.params_file_path = yaml_path;
    Configure(cfg);
    return true;
  } catch (const std::exception &e) {
    std::cerr << "[ImuPreprocessor] Failed to load config: " << e.what() << std::endl;
    return false;
  }
}

void ImuPreprocessor::Configure(const ImuPreprocessorConfig &config) {
  config_ = config;
  // 防止非法配置导致 LPF 系数或门限异常。
  config_.sample_rate_hz = std::max(1.0, config_.sample_rate_hz);
  config_.lpf_cutoff_hz = std::max(0.1, config_.lpf_cutoff_hz);
  config_.gyro_outlier_rad_s = std::max(0.0, config_.gyro_outlier_rad_s);
  config_.acc_outlier_mps2 = std::max(0.0, config_.acc_outlier_mps2);
}

void ImuPreprocessor::Reset() {
  has_lpf_state_ = false;
  has_last_good_raw_ = false;
  lpf_acc_.setZero();
  lpf_gyro_.setZero();
  last_good_raw_acc_.setZero();
  last_good_raw_gyro_.setZero();
  gravity_norm_est_ = 9.81;
  last_sample_rate_hz_ = 0.0;
}

double ImuPreprocessor::LpfAlpha() const {
  const double fs = std::max(1.0, config_.sample_rate_hz);
  // 截止频率不超过 Nyquist 的 0.45 倍，避免离散化下混叠。
  const double fc = std::min(config_.lpf_cutoff_hz, 0.45 * fs);
  return 1.0 - std::exp(-2.0 * M_PI * fc / fs);
}

bool ImuPreprocessor::IsOutlier(const Eigen::Vector3d &acc,
                                const Eigen::Vector3d &gyro) const {
  if (!config_.enable_outlier_reject) {
    return false;
  }
  // 陀螺：模长超过门限视为 spike（静止/慢速场景下真实 ω 通常远小于门限）。
  if (config_.gyro_outlier_rad_s > 0.0 &&
      gyro.norm() > config_.gyro_outlier_rad_s) {
    return true;
  }
  // 加速度：模长偏离当前重力估计超过门限（如几十 m/s² 的脉冲）。
  if (config_.acc_outlier_mps2 > 0.0 &&
      std::abs(acc.norm() - gravity_norm_est_) > config_.acc_outlier_mps2) {
    return true;
  }
  return false;
}

void ImuPreprocessor::UpdateGravityEstimate(const double acc_norm) {
  if (acc_norm < 1.0) {
    return;
  }
  // 慢 EMA 跟踪 ‖a‖，避免野值 replace 后 g_est 被单点拉偏。
  constexpr double kGravityEmaAlpha = 0.01;
  gravity_norm_est_ =
      (1.0 - kGravityEmaAlpha) * gravity_norm_est_ + kGravityEmaAlpha * acc_norm;
}

sensor_msgs::msg::Imu::SharedPtr
ImuPreprocessor::Process(const sensor_msgs::msg::Imu::SharedPtr &raw) {
  if (!raw) {
    return nullptr;
  }
  if (!config_.enable) {
    return CloneImu(raw);
  }

  Eigen::Vector3d acc = ImuAcc(raw);
  Eigen::Vector3d gyro = ImuGyro(raw);

  if (IsOutlier(acc, gyro)) {
    // 不删点：用上一合法值 replace，保持 400Hz 时间轴与中值积分 dt 连续。
    if (has_lpf_state_) {
      acc = lpf_acc_;
      gyro = lpf_gyro_;
    } else if (has_last_good_raw_) {
      acc = last_good_raw_acc_;
      gyro = last_good_raw_gyro_;
    }
  } else {
    last_good_raw_acc_ = acc;
    last_good_raw_gyro_ = gyro;
    has_last_good_raw_ = true;
    UpdateGravityEstimate(acc.norm());
  }

  if (config_.enable_lpf) {
    if (!has_lpf_state_) {
      // 首样本：直接初始化 LPF 状态，无历史可滤。
      lpf_acc_ = acc;
      lpf_gyro_ = gyro;
      has_lpf_state_ = true;
    } else {
      // 一阶 IIR：y += α(x - y)
      const double alpha = LpfAlpha();
      lpf_acc_ += alpha * (acc - lpf_acc_);
      lpf_gyro_ += alpha * (gyro - lpf_gyro_);
    }
    acc = lpf_acc_;
    gyro = lpf_gyro_;
  }

  return MakeImuFromVectors(raw, acc, gyro);
}

std::deque<sensor_msgs::msg::Imu::SharedPtr> ImuPreprocessor::ProcessBatch(
    const std::deque<sensor_msgs::msg::Imu::SharedPtr> &inputs) {
  std::deque<sensor_msgs::msg::Imu::SharedPtr> outputs;

  double prev_t = 0.0;
  bool has_prev_t = false;
  for (const auto &raw : inputs) {
    if (raw) {
      const double t = svio_utils::MsgTimeSec(raw);
      if (has_prev_t) {
        const double dt = t - prev_t;
        if (dt > 1e-6) {
          last_sample_rate_hz_ = 1.0 / dt;
          // 用瞬时采样率微调 config_.sample_rate_hz，使 LPF 系数适应实际 dt 抖动。
          config_.sample_rate_hz = std::clamp(
              0.9 * config_.sample_rate_hz + 0.1 * last_sample_rate_hz_, 50.0,
              1000.0);
        }
      }
      prev_t = t;
      has_prev_t = true;
    }
    auto processed = Process(raw);
    if (processed) {
      outputs.push_back(std::move(processed));
    }
  }
  return outputs;
}

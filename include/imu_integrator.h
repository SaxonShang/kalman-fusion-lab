#pragma once

/**
 * @file imu_integrator.h
 * @brief IMU 初始化与预积分（位姿增量 / 状态+协方差传播）。
 */

#include <deque>
#include <sensor_msgs/msg/imu.hpp>

#include "data.h"

class ImuIntegrator {
public:
  /**
   * @brief 静止段 IMU 初始化：估计重力方向、陀螺/加计偏置初值。
   * @param imu_buffer 静止或准静止 IMU 序列
   * @return 初始化后的世界系 ImuStatesGroup
   */
  ImuStatesGroup ImuInit(const std::deque<sensor_msgs::msg::Imu::SharedPtr> &imu_buffer);

  /**
   * @brief 预积分位姿增量 ΔR,Δp,Δv（相对关键帧时刻）。
   * @param keyframe 起点关键帧（含偏置与状态）
   * @param imu_buffer [t_kf, t_end] IMU
   * @param t_end 积分终点时间戳 (s)
   * @return Pose6D 增量；同时缓存到 GetLastPreintegrationIncrement
   */
  Pose6D ImuPreintegrationPose(const KeyFrameData &keyframe,
                                 const std::deque<sensor_msgs::msg::Imu::SharedPtr> &imu_buffer,
                                 double t_end);

  /**
   * @brief 预积分传播完整状态与协方差到 t_end。
   * @param keyframe 起点关键帧
   * @param imu_buffer [t_kf, t_end] IMU
   * @param t_end 终点时间戳 (s)
   * @param flow_gate_cov 非空时用同一次 Φ、Q 传播这份并行协方差，不改返回值里的绝对协方差
   * @return 传播后的 ImuStatesGroup（含 P_pred）
   */
  ImuStatesGroup ImuPreintegrationCov(
    const KeyFrameData &keyframe, const std::deque<sensor_msgs::msg::Imu::SharedPtr> &imu_buffer,
    double t_end,
    Eigen::Matrix<double, DIM_STATE, DIM_STATE> *flow_gate_cov = nullptr);

  /** @brief 最近一次 ImuPreintegrationPose 缓存的增量。 */
  const Pose6D &GetLastPreintegrationIncrement() const { return preint_increment_cache_; }

private:
  Pose6D preint_increment_cache_;
};

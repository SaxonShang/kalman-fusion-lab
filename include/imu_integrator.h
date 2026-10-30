#pragma once

/**
 * @file imu_integrator.h
 * @brief IMU 初始化与预积分（位姿增量 / 状态+协方差传播）。
 */

#include <deque>
#include <sensor_msgs/msg/imu.hpp>

#include "data.h"

/** @brief IMU 连续时间噪声（SI 单位）。默认值即原来写死在积分器里的常数。 */
struct ImuNoiseParams {
  double gyro_noise_density = 1.0e-4;  ///< rad/s/√Hz
  double accel_noise_density = 1.0e-3; ///< m/s²/√Hz
  double gyro_random_walk = 1.0e-5;    ///< rad/s²/√Hz
  double accel_random_walk = 1.0e-4;   ///< m/s³/√Hz
};

/** @brief 静止初始化窗口的统计与检查结果。 */
struct ImuInitReport {
  int samples = 0;
  double duration_s = 0.0;
  double gravity_norm = 0.0;
  double gyro_max_deviation_rad_s = 0.0; ///< 各样本陀螺读数偏离均值的最大值
  Eigen::Vector3d gyro_std_rad_s = Eigen::Vector3d::Zero();
  Eigen::Vector3d accel_std = Eigen::Vector3d::Zero();
  bool static_ok = false; ///< 时长、重力量级、陀螺波动均在阈值内
};

/** @brief 按可观性设置初始协方差用的先验参数（事先设定，不由真值调出）。 */
struct InitialCovarianceParams {
  double accel_bias_sigma = 0.05;         ///< m/s²，加计零偏先验，决定横滚、俯仰不确定度
  double yaw_sigma_rad = 1.0e-4;          ///< 航向由初始化定义，只保留数值下限
  double position_sigma_m = 1.0e-4;       ///< 位置原点由初始化定义
  double velocity_sigma_mps = 0.01;       ///< 静止起步
  double gyro_bias_floor_rad_s = 1.75e-4; ///< 0.01°/s，覆盖温漂等未建模变化
};

class ImuIntegrator {
public:
  /** @brief 设置传播用的连续噪声；不设置时沿用原来的常数。 */
  void SetNoiseParams(const ImuNoiseParams &params) { noise_ = params; }
  const ImuNoiseParams &GetNoiseParams() const { return noise_; }

  /**
   * @brief 静止段 IMU 初始化：估计重力方向、陀螺/加计偏置初值。
   * @param imu_buffer 静止或准静止 IMU 序列
   * @param report 非空时输出窗口统计与静止检查结果（不改变初始化本身）
   * @return 初始化后的世界系 ImuStatesGroup
   */
  ImuStatesGroup ImuInit(const std::deque<sensor_msgs::msg::Imu::SharedPtr> &imu_buffer,
                         ImuInitReport *report = nullptr);

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
   * @param transition 非空时输出本次传播的误差状态转移矩阵 Φ（各步 F_d 按时间顺序左乘），
   *        满足 P_end = Φ P_start Φᵀ + Q（Q 为同一区间从零开始累积的过程噪声）
   * @return 传播后的 ImuStatesGroup（含 P_pred）
   */
  ImuStatesGroup ImuPreintegrationCov(
    const KeyFrameData &keyframe, const std::deque<sensor_msgs::msg::Imu::SharedPtr> &imu_buffer,
    double t_end,
    Eigen::Matrix<double, DIM_STATE, DIM_STATE> *flow_gate_cov = nullptr,
    Eigen::Matrix<double, DIM_STATE, DIM_STATE> *transition = nullptr);

  /** @brief 最近一次 ImuPreintegrationPose 缓存的增量。 */
  const Pose6D &GetLastPreintegrationIncrement() const { return preint_increment_cache_; }

  /**
   * @brief 按可观性设置初始协方差：横滚/俯仰由重力测得（加计零偏决定误差），航向和位置由初始化定义，
   *        速度为静止起步，陀螺零偏取静止样本统计。重力、曝光与预留维沿用默认值。
   * @param state ImuInit 的结果（用其姿态把世界系倾斜/航向不确定度转到机体系）
   * @param report ImuInit 输出的窗口统计
   */
  static Eigen::Matrix<double, DIM_STATE, DIM_STATE> ObservabilityInitialCovariance(
    const ImuStatesGroup &state, const ImuInitReport &report,
    const InitialCovarianceParams &params);

private:
  Pose6D preint_increment_cache_;
  ImuNoiseParams noise_;
};

/**
 * @brief 以锚点位姿为条件的当前状态协方差。
 *
 * 当前状态 x_b = Φ x_a + w，w ~ N(0, Q)。已知锚点位姿 [θ_a, p_a] 时：
 *   P_b|a = Φ (P_a − P_a[:,0:6] P_a[0:6,0:6]⁻¹ P_a[0:6,:]) Φᵀ + Q
 * 锚点速度、零偏等的不确定度保留，并经 Φ 传到当前位姿。P_a[0:6,0:6] 奇异时取伪逆。
 * @param anchor_cov 锚点时刻的完整协方差 P_a
 * @param transition 锚点到当前的 Φ
 * @param process_noise 锚点到当前从零累积的过程噪声 Q
 */
Eigen::Matrix<double, DIM_STATE, DIM_STATE> ConditionOnAnchorPose(
  const Eigen::Matrix<double, DIM_STATE, DIM_STATE> &anchor_cov,
  const Eigen::Matrix<double, DIM_STATE, DIM_STATE> &transition,
  const Eigen::Matrix<double, DIM_STATE, DIM_STATE> &process_noise);

/**
 * @brief 帧对评价参照的雅可比 J（6×6）：评价误差的位姿部分 e = δx_b[0:6] − J δx_a[0:6]。
 *
 * 评价把真值相对运动接到估计锚点上，参照为 x̂_a · (x_a⁻¹ x_b)。旋转为机体系右扰动、位置为世界系，一阶有
 *   e_θ = δθ_b − R̂_bᵀ R̂_a δθ_a
 *   e_p = δp_b − δp_a − [p̂_b − p̂_a]× R̂_a δθ_a
 * 参数均为估计值：锚点与当前帧的先验位姿。
 */
Eigen::Matrix<double, 6, 6> AnchorReferenceJacobian(const Eigen::Matrix3d &R_anchor,
                                                    const Eigen::Vector3d &p_anchor,
                                                    const Eigen::Matrix3d &R_current,
                                                    const Eigen::Vector3d &p_current);

/**
 * @brief 以估计的锚点位姿为参照的当前状态协方差，与帧对评价口径一致。
 *
 * δx_b = Φ δx_a + w，位姿部分换成评价误差 e = δx_b[0:6] − J δx_a[0:6]，其余状态不变，故
 *   P_rel = M P_a Mᵀ + Q，M = Φ − J̃，J̃ 只在位姿行、位姿列等于 J（见 AnchorReferenceJacobian）。
 * 与 ConditionOnAnchorPose 不同，锚点位姿并未已知：锚点速度、零偏保留边际不确定度，锚点姿态误差
 * 经重力和速度造成的位移（Φ_pθ − J_pθ = [v̂_a T + ½ĝT²]× R̂_a）也保留。由全方差公式 P_rel ≥ P_b|a。
 */
Eigen::Matrix<double, DIM_STATE, DIM_STATE> RelativeToAnchorPose(
  const Eigen::Matrix<double, DIM_STATE, DIM_STATE> &anchor_cov,
  const Eigen::Matrix<double, DIM_STATE, DIM_STATE> &transition,
  const Eigen::Matrix<double, DIM_STATE, DIM_STATE> &process_noise,
  const Eigen::Matrix<double, 6, 6> &reference_jacobian);

/**
 * @brief 陀螺刻度误差：旋转方差加上 (scale_sigma · |Δθ|)²，Δθ 为锚点到当前的积分转角。
 *        刻度误差是系统误差，随转角线性增长，白噪声 Q 覆盖不到。
 * @param cov 当前状态协方差（就地修改旋转块）
 * @param scale_sigma 刻度不确定度（例如 0.015 表示 1.5%）
 */
void AddGyroScaleUncertainty(Eigen::Matrix<double, DIM_STATE, DIM_STATE> &cov,
                             const Eigen::Matrix3d &R_anchor, const Eigen::Matrix3d &R_current,
                             double scale_sigma);

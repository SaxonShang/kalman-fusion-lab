#include "imu_integrator.h"
#include "sio_math.h"
#include "profiler/simple_profiler.h"
#include <Eigen/Dense>
#include <cmath>
#include <iostream>

// Static initialization of IMU: estimate gyro bias and gravity from stationary
// sensor data
ImuStatesGroup ImuIntegrator::ImuInit(
  const std::deque<sensor_msgs::msg::Imu::SharedPtr> &imu_buffer) {

  ImuStatesGroup state;

  if (imu_buffer.empty()) {
    std::cerr << "[ImuInit] imu_buffer is empty, returning default state.\n";
    return state;
  }

  // 求加速度计和陀螺仪均値
  Eigen::Vector3d acc_mean = Eigen::Vector3d::Zero();
  Eigen::Vector3d gyro_mean = Eigen::Vector3d::Zero();

  for (const auto &msg : imu_buffer) {
    acc_mean += Eigen::Vector3d(msg->linear_acceleration.x, msg->linear_acceleration.y,
                                msg->linear_acceleration.z);
    gyro_mean +=
      Eigen::Vector3d(msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z);
  }
  acc_mean /= static_cast<double>(imu_buffer.size());
  gyro_mean /= static_cast<double>(imu_buffer.size());

  // 检查重力量级是否合理（静止时 ||acc_mean|| ≈ 9.81 m/s²）
  const double acc_norm = acc_mean.norm();
  if (acc_norm < 8.0 || acc_norm > 11.0) {
    std::cerr << "[ImuInit] abnormal gravity norm: " << acc_norm
              << " m/s^2, device may not be static.\n";
  }

  // 陀螺仪零偏估计（静止时陀螺仪读数即为零偏）
  state.bias_g = gyro_mean;

  // 加速度计零偏：静止时 acc = R^T * g_world + ba，暂时不估计，留相对坐标用
  state.bias_a = Eigen::Vector3d::Zero();

  // 利用均均重力方向构造初始旋转矩阵
  // 约定：世界系 z 轴指向，将加速度计均开方向对齐到 -z_world
  const Eigen::Vector3d g_body = acc_mean.normalized(); // 本体系重力方向
  const Eigen::Vector3d z_world(0.0, 0.0, 1.0);         // 世界系z轴方向向上
  const Eigen::Vector3d axis = g_body.cross(z_world);   // 旋转轴
  const double dot = g_body.dot(z_world);
  const double angle = std::acos(std::max(-1.0, std::min(1.0, dot)));

  if (axis.norm() < 1e-6) {
    // g_body 已与 z_world 对齐，无需旋转
    state.rot_end = Eigen::Matrix3d::Identity();
  } else {
    state.rot_end = Eigen::AngleAxisd(angle, axis.normalized()).toRotationMatrix();
  }

  // 重力向量（世界系，向下为负）
  state.gravity = Eigen::Vector3d(0.0, 0.0, -acc_norm);

  // 初始位置、速度均为零
  state.pos_end = Eigen::Vector3d::Zero();
  state.vel_end = Eigen::Vector3d::Zero();

  // 时间戳取最后一条 IMU 的时间
  const auto &last = imu_buffer.back();
  state.timestamp = last->header.stamp.sec + last->header.stamp.nanosec * 1e-9;

  std::cout << "[ImuInit] done. gyro_bias=" << gyro_mean.transpose()
            << "  gravity_norm=" << acc_norm << "  rot_end=\n"
            << state.rot_end << "\n";

  return state;
}

// Integrate IMU measurements to compute pose increments without covariance
Pose6D ImuIntegrator::ImuPreintegrationPose(
  const KeyFrameData &prev_keyframe, const std::deque<sensor_msgs::msg::Imu::SharedPtr> &imu_buffer,
  double t_end) {

  Pose6D pose_propagated; // 默认构造已将 delta_R
                          // 初始化为单位阵，delta_p/delta_v 为零

  if (imu_buffer.size() < 2) {
    return pose_propagated;
  }

  // 从缓存队列中提取用于预积分的 IMU 数据：
  //   - t < t_start：只保留最后一条作为积分左端点（与第一条 t >= t_start
  //   的数据做中值积分）
  //   - t >= t_start：全部纳入
  const double t_start = prev_keyframe.timestamp;
  sensor_msgs::msg::Imu::SharedPtr imu_before_start = nullptr;
  std::deque<sensor_msgs::msg::Imu::SharedPtr> imu_between;

  for (const auto &msg : imu_buffer) {
    if (!msg)
      continue;
    const double t = sio_utils::MsgTimeSec(msg);
    if (t <= t_start) {
      imu_before_start = msg; // 持续覆盖，循环结束后即为紧邻 t_start 之前的最后一条
    } else {
      imu_between.push_back(msg);
    }
  }

  // 将 t_start 之前的最后一条插到队头，作为中值积分的左端点
  if (imu_before_start) {
    imu_between.push_front(imu_before_start);
  }

  if (imu_between.size() < 2) {
    std::cerr << "[ImuPreintegrationPose] insufficient IMU data after t_start=" << t_start << "\n";
    return pose_propagated;
  }

  // 从上一关键帧状态取积分起始值
  Eigen::Matrix3d R = prev_keyframe.imu_states.rot_end;
  Eigen::Vector3d v = prev_keyframe.imu_states.vel_end;
  Eigen::Vector3d p = prev_keyframe.imu_states.pos_end;

  const Eigen::Vector3d bg = prev_keyframe.imu_states.bias_g;
  const Eigen::Vector3d ba = prev_keyframe.imu_states.bias_a;
  const Eigen::Vector3d g = prev_keyframe.imu_states.gravity;

  const Eigen::Matrix3d R_init = R;
  const Eigen::Vector3d v_init = v;
  const Eigen::Vector3d p_init = p;
  double total_dt = 0.0;

  // 中值积分：利用相邻两条 IMU 的均值角速度/加速度积分
  for (size_t i = 1; i < imu_between.size(); ++i) {

    const auto &imu_prev = imu_between[i - 1];
    const auto &imu_cur = imu_between[i];
    if (!imu_prev || !imu_cur)
      continue;

    const double t_prev_stamp = sio_utils::MsgTimeSec(imu_prev);
    const double t_cur_stamp = sio_utils::MsgTimeSec(imu_cur);

    // 第一段：若左端点 imu_before_start 在 t_start 之前，以 t_start 为积分起点
    const double t_eff_prev = (i == 1 && t_prev_stamp < t_start) ? t_start : t_prev_stamp;
    const double dt = t_cur_stamp - t_eff_prev;
    if (dt <= 0.0)
      continue;

    const Eigen::Vector3d gyr_prev(imu_prev->angular_velocity.x, imu_prev->angular_velocity.y,
                                   imu_prev->angular_velocity.z);
    const Eigen::Vector3d gyr_cur(imu_cur->angular_velocity.x, imu_cur->angular_velocity.y,
                                  imu_cur->angular_velocity.z);
    const Eigen::Vector3d acc_prev(imu_prev->linear_acceleration.x, imu_prev->linear_acceleration.y,
                                   imu_prev->linear_acceleration.z);
    const Eigen::Vector3d acc_cur(imu_cur->linear_acceleration.x, imu_cur->linear_acceleration.y,
                                  imu_cur->linear_acceleration.z);

    // 去零偏，取中值角速度更新旋转
    const Eigen::Vector3d w_mid = 0.5 * ((gyr_prev - bg) + (gyr_cur - bg));
    const Eigen::Matrix3d R_prev = R;
    R = R * sio_utils::ExpSO3(w_mid * dt);

    // 去零偏加速度转到世界系，加重力，取中值更新位置和速度
    const Eigen::Vector3d a_world_prev = R_prev * (acc_prev - ba) + g;
    const Eigen::Vector3d a_world_cur = R * (acc_cur - ba) + g;
    const Eigen::Vector3d a_world_mid = 0.5 * (a_world_prev + a_world_cur);

    p += v * dt + 0.5 * a_world_mid * dt * dt; 
    v += a_world_mid * dt;
    total_dt += dt;
  }

  // 尾段：最后一条 IMU 时间戳 < t_end，用最后一条 IMU 测量值（恒定假设）外推到
  // t_end
  if (!imu_between.empty()) {
    const auto &imu_last = imu_between.back();
    const double t_last = sio_utils::MsgTimeSec(imu_last);
    const double dt_tail = t_end - t_last;
    if (dt_tail > 0.0 && imu_last) {
      const Eigen::Vector3d gyr_last(imu_last->angular_velocity.x, imu_last->angular_velocity.y,
                                     imu_last->angular_velocity.z);
      const Eigen::Vector3d acc_last(imu_last->linear_acceleration.x,
                                     imu_last->linear_acceleration.y,
                                     imu_last->linear_acceleration.z);
      const Eigen::Vector3d w = gyr_last - bg;
      const Eigen::Vector3d a_body = acc_last - ba;
      const Eigen::Matrix3d R_prev = R;
      R = R * sio_utils::ExpSO3(w * dt_tail);
      // 中值：用积分前后旋转各自将同一 body 加速度转到世界系再取均值
      const Eigen::Vector3d a_world_mid = 0.5 * (R_prev * a_body + g + R * a_body + g);
      p += v * dt_tail + 0.5 * a_world_mid * dt_tail * dt_tail;
      v += a_world_mid * dt_tail;
      total_dt += dt_tail;
    }
  }

  // 相对于起始帧的增量（初始 body 系）
  const Eigen::Matrix3d delta_R = R_init.transpose() * R;
  const Eigen::Vector3d delta_p = R_init.transpose() * (p - p_init);
  const Eigen::Vector3d delta_v = R_init.transpose() * (v - v_init);

  pose_propagated.dt = total_dt;
  pose_propagated.timestamp = prev_keyframe.timestamp + total_dt;

  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c)
      pose_propagated.delta_R[r * 3 + c] = delta_R(r, c);

  pose_propagated.delta_p[0] = delta_p.x();
  pose_propagated.delta_p[1] = delta_p.y();
  pose_propagated.delta_p[2] = delta_p.z();

  pose_propagated.delta_v[0] = delta_v.x();
  pose_propagated.delta_v[1] = delta_v.y();
  pose_propagated.delta_v[2] = delta_v.z();

  return pose_propagated;
}

// Integrate IMU with error state propagation and covariance matrix update
ImuStatesGroup ImuIntegrator::ImuPreintegrationCov(
  const KeyFrameData &keyframe, const std::deque<sensor_msgs::msg::Imu::SharedPtr> &imu_buffer,
  double t_end, Eigen::Matrix<double, DIM_STATE, DIM_STATE> *flow_gate_cov) {

  ImuStatesGroup propagated_state = keyframe.imu_states;

  if (imu_buffer.empty()) {
    return propagated_state;
  }

  // 提取上一关键帧到当前帧的 IMU 数据（同 ImuPreintegrationPose 的筛选逻辑）
  const double t_start = keyframe.timestamp;
  sensor_msgs::msg::Imu::SharedPtr imu_before_start = nullptr;
  std::deque<sensor_msgs::msg::Imu::SharedPtr> imu_between;
  for (const auto &msg : imu_buffer) {
    if (!msg)
      continue;
    const double t = sio_utils::MsgTimeSec(msg);
    if (t <= t_start) {
      imu_before_start = msg;
    } else {
      imu_between.push_back(msg);
    }
  }
  if (imu_before_start)
    imu_between.push_front(imu_before_start);

  if (imu_between.size() < 2)
    return propagated_state;

  // 状态取起始値
  Eigen::Matrix3d R = propagated_state.rot_end;
  Eigen::Vector3d v = propagated_state.vel_end;
  Eigen::Vector3d p = propagated_state.pos_end;
  const Eigen::Vector3d bg = propagated_state.bias_g;
  const Eigen::Vector3d ba = propagated_state.bias_a;
  const Eigen::Vector3d g = propagated_state.gravity;

  // 协方差矩阵 P (21×21)，状态顺序:
  // [δθ(0:3), δp(3:6), δv(6:9), δbg(9:12), δba(12:15), δg(15:18), δexpo(18:19),
  // reserved(19:21)]
  Eigen::Matrix<double, 21, 21> P = propagated_state.cov;

  // IMU 噪声密度参数（单位：rad/s/√Hz, m/s²/√Hz, rad/s²/√Hz, m/s³/√Hz）
  constexpr double sigma_g = 1.0e-4;  // 陀螺仪量测噪声
  constexpr double sigma_a = 1.0e-3;  // 加速度计量测噪声
  constexpr double sigma_bg = 1.0e-5; // 陀螺仪偏置随机游走
  constexpr double sigma_ba = 1.0e-4; // 加速度计偏置随机游走

  // 连续时间噪声协方差矩阵 Q_c (12⅗12)
  // 噪声输入顺序: [ng(3), na(3), nbg(3), nba(3)]
  Eigen::Matrix<double, 12, 12> Q_c = Eigen::Matrix<double, 12, 12>::Zero();
  Q_c.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * (sigma_g * sigma_g);
  Q_c.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() * (sigma_a * sigma_a);
  Q_c.block<3, 3>(6, 6) = Eigen::Matrix3d::Identity() * (sigma_bg * sigma_bg);
  Q_c.block<3, 3>(9, 9) = Eigen::Matrix3d::Identity() * (sigma_ba * sigma_ba);

  // 辅助：反对称矩阵
  auto skew = [](const Eigen::Vector3d &u) -> Eigen::Matrix3d {
    Eigen::Matrix3d S;
    S << 0, -u.z(), u.y(), u.z(), 0, -u.x(), -u.y(), u.x(), 0;
    return S;
  };

  for (size_t i = 1; i < imu_between.size(); ++i) {
    const auto &imu_prev = imu_between[i - 1];
    const auto &imu_cur = imu_between[i];
    if (!imu_prev || !imu_cur)
      continue;

    const double t_prev_stamp = sio_utils::MsgTimeSec(imu_prev);
    const double t_cur_stamp = sio_utils::MsgTimeSec(imu_cur);

    // 第一段：若左端点 imu_before_start 在 t_start 之前，以 t_start 为积分起点
    const double t_eff_prev = (i == 1 && t_prev_stamp < t_start) ? t_start : t_prev_stamp;
    const double dt = t_cur_stamp - t_eff_prev;
    if (dt <= 0.0)
      continue;

    const Eigen::Vector3d gyr_prev(imu_prev->angular_velocity.x, imu_prev->angular_velocity.y,
                                   imu_prev->angular_velocity.z);
    const Eigen::Vector3d gyr_cur(imu_cur->angular_velocity.x, imu_cur->angular_velocity.y,
                                  imu_cur->angular_velocity.z);
    const Eigen::Vector3d acc_prev(imu_prev->linear_acceleration.x, imu_prev->linear_acceleration.y,
                                   imu_prev->linear_acceleration.z);
    const Eigen::Vector3d acc_cur(imu_cur->linear_acceleration.x, imu_cur->linear_acceleration.y,
                                  imu_cur->linear_acceleration.z);

    const Eigen::Vector3d w_mid = 0.5 * ((gyr_prev - bg) + (gyr_cur - bg));

    const Eigen::Matrix3d R_prev = R;
    R = R * sio_utils::ExpSO3(w_mid * dt);

    const Eigen::Vector3d a_world_prev = R_prev * (acc_prev - ba) + g;
    const Eigen::Vector3d a_world_cur = R * (acc_cur - ba) + g;
    const Eigen::Vector3d a_world_mid = 0.5 * (a_world_prev + a_world_cur);

    p += v * dt + 0.5 * a_world_mid * dt * dt;
    v += a_world_mid * dt;

    // 约化的 a_mid 用于计算连续误差状态转移矩阵 F 的近似雅可比
    const Eigen::Vector3d a_mid = 0.5 * ((acc_prev - ba) + (acc_cur - ba));

    // 连续误差状态转移矩阵 F (21×21)
    // 状态顺序: [δθ(0:3), δp(3:6), δv(6:9), δbg(9:12), δba(12:15), δg(15:18),
    // δexpo(18:19)] expo 为 1 维标量，对应 index 18；19:21 为预留，F
    // 中对应行列保持零
    Eigen::Matrix<double, 21, 21> F = Eigen::Matrix<double, 21, 21>::Zero();
    // δθ̇ = -[ω]× δθ - δbg
    F.block<3, 3>(0, 0) = -skew(w_mid);
    F.block<3, 3>(0, 9) = -Eigen::Matrix3d::Identity();
    // δṕ = δv
    F.block<3, 3>(3, 6) = Eigen::Matrix3d::Identity();
    // δv̇ = -R[a]× δθ - R δba + δg
    F.block<3, 3>(6, 0) = -R_prev * skew(a_mid);
    F.block<3, 3>(6, 12) = -R_prev;
    F.block<3, 3>(6, 15) = Eigen::Matrix3d::Identity();
    // δbg, δba, δg, δexpo: 随机游走，F 中对应块为零

    // 输入噪声矩阵 G (21⅗12)
    Eigen::Matrix<double, 21, 12> G = Eigen::Matrix<double, 21, 12>::Zero();
    G.block<3, 3>(0, 0) = -Eigen::Matrix3d::Identity(); // δθ ← ng
    G.block<3, 3>(6, 3) = -R_prev;                      // δv ← na
    G.block<3, 3>(9, 6) = Eigen::Matrix3d::Identity();  // δbg ← nbg
    G.block<3, 3>(12, 9) = Eigen::Matrix3d::Identity(); // δba ← nba

    // 离散化（一阶近似）
    const Eigen::Matrix<double, 21, 21> F_d = Eigen::Matrix<double, 21, 21>::Identity() + F * dt;
    const Eigen::Matrix<double, 21, 21> Q_d = G * Q_c * G.transpose() * dt;

    P = F_d * P * F_d.transpose() + Q_d;
    if (flow_gate_cov != nullptr) {
      const Eigen::Matrix<double, 21, 21> updated =
          F_d * (*flow_gate_cov) * F_d.transpose() + Q_d;
      *flow_gate_cov = updated;
    }
  }

  // 最后一段：若最后一条 IMU < t_end，用该 IMU 的测量值外推到 t_end
  if (!imu_between.empty()) {
    const auto &imu_last = imu_between.back();
    const double t_last = sio_utils::MsgTimeSec(imu_last);
    const double dt_tail = t_end - t_last;
    if (dt_tail > 0.0 && imu_last) {
      const Eigen::Vector3d gyr_last(imu_last->angular_velocity.x, imu_last->angular_velocity.y,
                                     imu_last->angular_velocity.z);
      const Eigen::Vector3d acc_last(imu_last->linear_acceleration.x,
                                     imu_last->linear_acceleration.y,
                                     imu_last->linear_acceleration.z);
      const Eigen::Vector3d w = gyr_last - bg;
      const Eigen::Vector3d a_mid = acc_last - ba;
      const Eigen::Matrix3d R_prev = R;
      R = R * sio_utils::ExpSO3(w * dt_tail);
      const Eigen::Vector3d a_world = 0.5 * (R_prev * a_mid + g + R * a_mid + g);

      p += v * dt_tail + 0.5 * a_world * dt_tail * dt_tail;
      v += a_world * dt_tail;

      // 最后一段的协方差传播
      Eigen::Matrix<double, 21, 21> F_tail = Eigen::Matrix<double, 21, 21>::Zero();
      F_tail.block<3, 3>(0, 0) = -skew(w);
      F_tail.block<3, 3>(0, 9) = -Eigen::Matrix3d::Identity();
      F_tail.block<3, 3>(3, 6) = Eigen::Matrix3d::Identity();
      F_tail.block<3, 3>(6, 0) = -R_prev * skew(a_mid);
      F_tail.block<3, 3>(6, 12) = -R_prev;
      F_tail.block<3, 3>(6, 15) = Eigen::Matrix3d::Identity();

      Eigen::Matrix<double, 21, 12> G_tail = Eigen::Matrix<double, 21, 12>::Zero();
      G_tail.block<3, 3>(0, 0) = -Eigen::Matrix3d::Identity();
      G_tail.block<3, 3>(6, 3) = -R_prev;
      G_tail.block<3, 3>(9, 6) = Eigen::Matrix3d::Identity();
      G_tail.block<3, 3>(12, 9) = Eigen::Matrix3d::Identity();

      const auto F_d = Eigen::Matrix<double, 21, 21>::Identity() + F_tail * dt_tail;
      const auto Q_d = G_tail * Q_c * G_tail.transpose() * dt_tail;
      P = F_d * P * F_d.transpose() + Q_d;
      if (flow_gate_cov != nullptr) {
        const Eigen::Matrix<double, 21, 21> updated =
            F_d * (*flow_gate_cov) * F_d.transpose() + Q_d;
        *flow_gate_cov = updated;
      }
    }
  }

  // 写回传播后的状态
  propagated_state.rot_end = R;
  propagated_state.vel_end = v;
  propagated_state.pos_end = p;
  propagated_state.cov = P;
  propagated_state.timestamp = t_end;

  // 计算并缓存预积分增量
  const Eigen::Matrix3d R_init = keyframe.imu_states.rot_end;
  const Eigen::Vector3d p_init = keyframe.imu_states.pos_end;
  const Eigen::Vector3d v_init = keyframe.imu_states.vel_end;

  Eigen::Matrix3d dR = R_init.transpose() * propagated_state.rot_end;
  Eigen::Vector3d dp =
    R_init.transpose() * (propagated_state.pos_end - p_init);
  Eigen::Vector3d dv =
    R_init.transpose() * (propagated_state.vel_end - v_init);

  // 将 Eigen 矩阵复制到 double[9]（按行主序存储）
  for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
          preint_increment_cache_.delta_R[i * 3 + j] = dR(i, j);
      }
      preint_increment_cache_.delta_p[i] = dp(i);
      preint_increment_cache_.delta_v[i] = dv(i);
  }
  preint_increment_cache_.dt = t_end - keyframe.timestamp;
  preint_increment_cache_.timestamp = t_end;

  return propagated_state;
}

#pragma once

#include "data.h"
#include "sio_math.h"

#include <array>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <stdexcept>
#include <string>

// 工具函数
namespace sio_utils {
// 将 ROS2 图像消息转换为 OpenCV 图像
inline void MsgToImg(const sensor_msgs::msg::Image::ConstSharedPtr &msg, cv::Mat &img,
                     double &img_timestamp) {
  if (!msg) {
    img.release();
    return;
  }
  const int width = static_cast<int>(msg->width);
  const int height = static_cast<int>(msg->height);
  if (width <= 0 || height <= 0 || msg->data.empty()) {
    img.release();
    return;
  }

  const std::string &enc = msg->encoding;
  int cv_type = -1;
  int expected_channels = 0;

  if (enc == "bgr8") {
    cv_type = CV_8UC3;
    expected_channels = 3;
  } else if (enc == "rgb8") {
    cv_type = CV_8UC3;
    expected_channels = 3;
  } else if (enc == "mono8") {
    cv_type = CV_8UC1;
    expected_channels = 1;
  } else {
    std::cerr << "[MsgToImg] Unsupported image encoding: " << enc << std::endl;
    img.release();
    return;
  }

  const size_t min_step = static_cast<size_t>(width * expected_channels);
  if (msg->step < min_step) {
    std::cerr << "[MsgToImg] Invalid image step: " << msg->step << ", expected >= " << min_step
              << std::endl;
    img.release();
    return;
  }

  const size_t required = static_cast<size_t>(msg->step) * static_cast<size_t>(height);
  if (msg->data.size() < required) {
    std::cerr << "[MsgToImg] Image data too small: " << msg->data.size()
              << ", required >= " << required << std::endl;
    img.release();
    return;
  }

  const cv::Mat wrapped(height, width, cv_type,
                        const_cast<unsigned char *>(msg->data.data()),
                        static_cast<size_t>(msg->step));

  if (enc == "rgb8") {
    cv::cvtColor(wrapped, img, cv::COLOR_RGB2BGR);
  } else if (enc == "mono8") {
    cv::cvtColor(wrapped, img, cv::COLOR_GRAY2BGR);
  } else {
    img = wrapped.clone();
  }

  img_timestamp = MsgTimeSec(msg);
}

// 计算 SO(3) 对数映射的近似值,将旋转矩阵转换为旋转向量
inline V3D So3LogApprox(const M3D &R) {
  V3D w;
  w << R(2, 1) - R(1, 2), R(0, 2) - R(2, 0), R(1, 0) - R(0, 1);
  return 0.5 * w;
}

// SO(3) 对数映射 θu。So3LogApprox 实际给出 sin(θ)u，只适合小角度。
inline V3D LogSO3(const M3D &R) {
  const Eigen::AngleAxisd angle_axis(R);
  return angle_axis.angle() * angle_axis.axis();
}

// 将 Pose6D 转换为旋转矩阵
inline M3D RotFromPose6D(const Pose6D &pose) {
  M3D R;
  R << pose.delta_R[0], pose.delta_R[1], pose.delta_R[2], pose.delta_R[3], pose.delta_R[4],
    pose.delta_R[5], pose.delta_R[6], pose.delta_R[7], pose.delta_R[8];
  return R;
}

// 将 Pose6D 转换为位置向量
inline V3D PosFromPose6D(const Pose6D &pose) {
  return V3D(pose.delta_p[0], pose.delta_p[1], pose.delta_p[2]);
}

// 将 Pose6D 转换为 4x4 变换矩阵
inline Eigen::Matrix4d TransformVector6dToMatrix4d(const Pose6D &pose) {
  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
  T.block<3, 3>(0, 0) = RotFromPose6D(pose);
  T.block<3, 1>(0, 3) = PosFromPose6D(pose);
  return T;
}

// 将 ImuStatesGroup 转换为 4x4 变换矩阵（T_w_body：p_world = T_w_body * p_body）
inline Eigen::Matrix4d ImuStatesToMatrix4d(const ImuStatesGroup &imu_state) {
  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
  T.block<3, 3>(0, 0) = imu_state.rot_end;
  T.block<3, 1>(0, 3) = imu_state.pos_end;
  return T;
}

// ── 坐标变换约定（全工程统一）────────────────────────────────────────────
// T_a_b：p_a = T_a_b * p_b（下标左 = 目标系，右 = 源系）。
// T_w_x：x 在 world 下的位姿，p_world = T_w_x * p_x（与 ImuStatesToMatrix4d 一致）。
// T_body_cam / T_imu_cam：p_body = T_body_cam * p_cam（params.yaml: T_imu_cam）。
// small_gicp T_target_source 等价于 T_kf_curr：p_kf = T_kf_curr * p_curr。

// 已知两帧世界位姿，求相对变换 T_a_b = T_w_a^{-1} * T_w_b
inline Eigen::Matrix4d RelativeTransform(const Eigen::Matrix4d &T_w_a,
                                         const Eigen::Matrix4d &T_w_b) {
  return T_w_a.inverse() * T_w_b;
}

// T_w_cam = T_w_body * T_body_cam（p_world = T_w_cam * p_cam）
inline Eigen::Matrix4d WorldCamFromBody(const Eigen::Matrix4d &T_w_body,
                                        const Eigen::Matrix4d &T_body_cam) {
  return T_w_body * T_body_cam;
}

// Camera-frame pose covariance 6×6: [δω; δt], including P_θp, in-block
// off-diagonals, and IMU–cam lever arm. Maps mixed error (body δθ, world δp):
//   δω_cam = R_icᵀ δθ
//   δt_cam = R_wcᵀ δp − R_icᵀ [t_ic]× δθ
// Transforms whatever 6×6 is in imu.cov. The flow gate passes the
// keyframe-relative parallel covariance; this function does not subtract P_k.
inline Eigen::Matrix<double, 6, 6>
ImuStatePoseCovInCamera(const ImuStatesGroup &imu,
                        const Eigen::Matrix4d &T_imu_cam) {
  const Eigen::Matrix3d R_ic = T_imu_cam.block<3, 3>(0, 0);
  const Eigen::Vector3d t_ic = T_imu_cam.block<3, 1>(0, 3);
  const Eigen::Matrix3d R_w_cam = imu.rot_end * R_ic;
  Eigen::Matrix<double, 6, 6> A = Eigen::Matrix<double, 6, 6>::Zero();
  A.block<3, 3>(0, 0) = R_ic.transpose();
  A.block<3, 3>(3, 0) = -R_ic.transpose() * SkewSymmetric(t_ic);
  A.block<3, 3>(3, 3) = R_w_cam.transpose();
  const Eigen::Matrix<double, 6, 6> P_body = imu.cov.block<6, 6>(0, 0);
  const Eigen::Matrix<double, 6, 6> P_cam = A * P_body * A.transpose();
  return 0.5 * (P_cam + P_cam.transpose());
}

// Flow-gate P_cam: same A as ImuStatePoseCovInCamera, but the 6×6 is the
// keyframe-relative parallel covariance. Nominal rotation still comes from imu_mean.
inline Eigen::Matrix<double, 6, 6> FlowGatePoseCovInCamera(
    const ImuStatesGroup &imu_mean,
    const Eigen::Matrix<double, DIM_STATE, DIM_STATE> &flow_gate_cov,
    const Eigen::Matrix4d &T_imu_cam) {
  ImuStatesGroup gate_state = imu_mean;
  gate_state.cov = flow_gate_cov;
  return ImuStatePoseCovInCamera(gate_state, T_imu_cam);
}

// Diagonal std of ImuStatePoseCovInCamera: [ωx,ωy,ωz, tx,ty,tz].
inline Eigen::Matrix<double, 6, 1>
ImuStatePoseSigmaInCamera(const ImuStatesGroup &imu,
                          const Eigen::Matrix4d &T_imu_cam) {
  const Eigen::Matrix<double, 6, 6> P_cam =
      ImuStatePoseCovInCamera(imu, T_imu_cam);
  Eigen::Matrix<double, 6, 1> sigma = Eigen::Matrix<double, 6, 1>::Zero();
  for (int i = 0; i < 6; ++i) {
    sigma(i) = std::sqrt(std::max(P_cam(i, i), 0.0));
  }
  return sigma;
}

// small_gicp 相对位姿 T_target_source 的 se(3) 右扰动协方差 → ESKF 混合坐标
// [δθ_body, δp_world] 的 6×6 观测噪声。T_w_keyframe 为 anchor 世界旋转 R_wk。
inline Eigen::Matrix<double, 6, 6> BuildGicpToEskfPoseJacobian(
    const Eigen::Matrix3d &R_w_keyframe,
    const Eigen::Matrix4d &T_target_source) {
  const Eigen::Matrix3d R_kc = T_target_source.block<3, 3>(0, 0);
  Eigen::Matrix<double, 6, 6> J = Eigen::Matrix<double, 6, 6>::Zero();
  J.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
  // T * Exp([w,rho]): rotation error is w in current body, while
  // additive world position changes only by R_w_keyframe * R_kc * rho.
  J.block<3, 3>(3, 3) = R_w_keyframe * R_kc;
  return J;
}

// Stereo-GICP supplies target/body left perturbations, whereas ESKF uses
// [rotation in current body, additive position in world]. Keep cross terms.
inline Eigen::Matrix<double, 6, 6> AlignStereoGicpCovarianceToEskfPose(
    const Eigen::Matrix<double, 6, 6> &cov_body,
    const Eigen::Matrix3d &R_w_keyframe,
    const Eigen::Matrix4d &T_target_source) {
  Eigen::Matrix<double, 6, 6> J = Eigen::Matrix<double, 6, 6>::Zero();
  J.block<3, 3>(0, 0) = T_target_source.block<3, 3>(0, 0).transpose();
  // Exp([w,rho]) * T: delta_t_target = -[t]x*w + rho.
  J.block<3, 3>(3, 0) = -R_w_keyframe *
    sio_utils::SkewSymmetric(T_target_source.block<3, 1>(0, 3));
  J.block<3, 3>(3, 3) = R_w_keyframe;
  const Eigen::Matrix<double, 6, 6> aligned = J * cov_body * J.transpose();
  return 0.5 * (aligned + aligned.transpose());
}

inline Eigen::Matrix<double, 6, 6> AlignGicpCovarianceToEskfPose(
    const Eigen::Matrix<double, 6, 6> &gicp_cov,
    const Eigen::Matrix3d &R_w_keyframe,
    const Eigen::Matrix4d &T_target_source) {
  const Eigen::Matrix<double, 6, 6> J =
      BuildGicpToEskfPoseJacobian(R_w_keyframe, T_target_source);
  Eigen::Matrix<double, 6, 6> aligned = J * gicp_cov * J.transpose();
  return 0.5 * (aligned + aligned.transpose());
}

// 相机系 GICP：先用 T_imu_cam 将 se(3) 协方差共轭到 IMU 体系，再对齐到 ESKF。
inline Eigen::Matrix<double, 6, 6> AlignGicpCovarianceCameraToEskfPose(
    const Eigen::Matrix<double, 6, 6> &gicp_cov,
    const Eigen::Matrix3d &R_w_keyframe,
    const Eigen::Matrix4d &T_gicp_target_source,
    const Eigen::Matrix4d &T_imu_cam) {
  const Eigen::Matrix4d T_body_rel =
      T_imu_cam * T_gicp_target_source * T_imu_cam.inverse();
  const Eigen::Matrix<double, 6, 6> Ad_imu_cam = sio_utils::Se3Adjoint(
      T_imu_cam.block<3, 3>(0, 0), T_imu_cam.block<3, 1>(0, 3));
  const Eigen::Matrix<double, 6, 6> gicp_cov_body =
      Ad_imu_cam * gicp_cov * Ad_imu_cam.transpose();
  return AlignGicpCovarianceToEskfPose(gicp_cov_body, R_w_keyframe, T_body_rel);
}

} // namespace sio_utils

#include "sio_config.h"

namespace svio_utils = sio_utils;

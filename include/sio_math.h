#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <memory>

namespace sio_utils {

template <typename MsgT> inline double MsgTimeSec(const MsgT &msg) {
  if (!msg) {
    return 0.0;
  }
  return static_cast<double>(msg->header.stamp.sec) +
         static_cast<double>(msg->header.stamp.nanosec) * 1e-9;
}
// 计算 SO(3) 的指数映射，将旋转向量转换为旋转矩阵
inline Eigen::Matrix3d ExpSO3(const Eigen::Vector3d &w) {
  const double theta = w.norm();
  if (theta < 1e-12) {
    return Eigen::Matrix3d::Identity();
  }
  return Eigen::AngleAxisd(theta, w / theta).toRotationMatrix();
}

inline Eigen::Matrix3d SkewSymmetric(const Eigen::Vector3d &v) {
  Eigen::Matrix3d S;
  S << 0.0, -v.z(), v.y(), v.z(), 0.0, -v.x(), -v.y(), v.x(), 0.0;
  return S;
}

// se3 右扰动（rotation-first）的 Adjoint，与 small_gicp 一致。
inline Eigen::Matrix<double, 6, 6> Se3Adjoint(const Eigen::Matrix3d &R,
                                              const Eigen::Vector3d &t) {
  Eigen::Matrix<double, 6, 6> Ad = Eigen::Matrix<double, 6, 6>::Zero();
  Ad.block<3, 3>(0, 0) = R;
  Ad.block<3, 3>(3, 0) = SkewSymmetric(t) * R;
  Ad.block<3, 3>(3, 3) = R;
  return Ad;
}

} // namespace sio_utils

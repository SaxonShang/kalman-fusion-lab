#pragma once

#include <Eigen/Dense>
#include <deque>
#include <memory>
#include <sensor_msgs/msg/imu.hpp>

namespace sio_test {

inline sensor_msgs::msg::Imu::SharedPtr MakeImu(
    double t_sec, const Eigen::Vector3d &acc,
    const Eigen::Vector3d &gyro_rad_s) {
  auto msg = std::make_shared<sensor_msgs::msg::Imu>();
  msg->header.stamp.sec = static_cast<int32_t>(t_sec);
  msg->header.stamp.nanosec =
      static_cast<uint32_t>((t_sec - msg->header.stamp.sec) * 1.0e9);
  msg->linear_acceleration.x = acc.x();
  msg->linear_acceleration.y = acc.y();
  msg->linear_acceleration.z = acc.z();
  msg->angular_velocity.x = gyro_rad_s.x();
  msg->angular_velocity.y = gyro_rad_s.y();
  msg->angular_velocity.z = gyro_rad_s.z();
  return msg;
}

inline std::deque<sensor_msgs::msg::Imu::SharedPtr>
MakeImuSequence(double t0, int count, double hz, const Eigen::Vector3d &acc,
                const Eigen::Vector3d &gyro_rad_s) {
  std::deque<sensor_msgs::msg::Imu::SharedPtr> out;
  const double dt = 1.0 / hz;
  for (int i = 0; i < count; ++i) {
    out.push_back(MakeImu(t0 + static_cast<double>(i) * dt, acc, gyro_rad_s));
  }
  return out;
}

inline std::deque<sensor_msgs::msg::Imu::SharedPtr>
MakeStaticImu(double t0, int count, double hz,
              const Eigen::Vector3d &acc = Eigen::Vector3d(0.0, 0.0, 9.81),
              const Eigen::Vector3d &gyro = Eigen::Vector3d::Zero()) {
  return MakeImuSequence(t0, count, hz, acc, gyro);
}

} // namespace sio_test

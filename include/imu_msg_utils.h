#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <memory>
#include <sensor_msgs/msg/imu.hpp>

inline double ImuMsgTimeSec(const sensor_msgs::msg::Imu::SharedPtr &imu_msg) {
  return static_cast<double>(imu_msg->header.stamp.sec) +
         static_cast<double>(imu_msg->header.stamp.nanosec) * 1e-9;
}

inline sensor_msgs::msg::Imu::SharedPtr MakeImuMsg(double time_sec, double ax, double ay, double az,
                                                   double gx_rad, double gy_rad, double gz_rad) {
  auto imu_msg = std::make_shared<sensor_msgs::msg::Imu>();
  const int64_t sec = static_cast<int64_t>(std::floor(time_sec));
  const int64_t nsec =
    static_cast<int64_t>(std::llround((time_sec - static_cast<double>(sec)) * 1e9));
  imu_msg->header.stamp.sec = static_cast<int32_t>(sec);
  imu_msg->header.stamp.nanosec = static_cast<uint32_t>(nsec);
  imu_msg->header.frame_id = "imu_link";
  imu_msg->linear_acceleration.x = ax;
  imu_msg->linear_acceleration.y = ay;
  imu_msg->linear_acceleration.z = az;
  imu_msg->angular_velocity.x = gx_rad;
  imu_msg->angular_velocity.y = gy_rad;
  imu_msg->angular_velocity.z = gz_rad;

  std::fill(imu_msg->linear_acceleration_covariance.begin(),
            imu_msg->linear_acceleration_covariance.end(), 0.0);
  imu_msg->linear_acceleration_covariance[0] = 0.01;
  imu_msg->linear_acceleration_covariance[4] = 0.01;
  imu_msg->linear_acceleration_covariance[8] = 0.01;
  return imu_msg;
}

inline std::deque<sensor_msgs::msg::Imu::SharedPtr>
PackImuBetween(const std::deque<sensor_msgs::msg::Imu::SharedPtr> &imu_snapshot,
               double start_time_sec, double end_time_sec) {
  std::deque<sensor_msgs::msg::Imu::SharedPtr> packed;
  for (const auto &imu_msg : imu_snapshot) {
    if (!imu_msg) {
      continue;
    }
    const double imu_time = ImuMsgTimeSec(imu_msg);
    if (imu_time > start_time_sec && imu_time <= end_time_sec) {
      packed.push_back(imu_msg);
    }
  }
  return packed;
}

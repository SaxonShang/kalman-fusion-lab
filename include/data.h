#pragma once

#include "camera_intrinsics.h"
#include <sensor_msgs/msg/imu.hpp>

#include <Eigen/Core>
#include <chrono>
#include <cstdint>
#include <deque>
#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <vector>
#include <deque>

#ifndef DIM_STATE
#  define DIM_STATE 21 // 典型值：rot(3), pos(3), vel(3), bg(3), ba(3), g(3), expo(1)
#endif
#define MD(a, b) Eigen::Matrix<double, (a), (b)>

using M3D = Eigen::Matrix3d;
using V3D = Eigen::Vector3d;
using PointTypeRGB = pcl::PointXYZRGB;
using PointTypeNormal = pcl::PointXYZRGBNormal;

// 视差三角化点。参考像素在 u、v 上，has_pixel 为 1 表示来自视差图。
// 刚体变换只改 point.x/y/z；u、v、has_pixel 不动，point 的 w 保持 1。
struct StereoPoint {
  pcl::PointXYZRGB point;  // x,y,z,rgb; w stays 1
  std::uint16_t u;
  std::uint16_t v;
  std::uint8_t has_pixel;  // 1 means u,v come from the disparity map
};
using StereoCloud = std::vector<StereoPoint>;

// 拷贝 StereoPoint 的 XYZRGB，并把 w 置为 1；u、v、has_pixel 不写入 PCL 点。
inline pcl::PointXYZRGB CopyStereoPointXyzrgb(const StereoPoint &sp) {
  pcl::PointXYZRGB pt = sp.point;
  pt.data[3] = 1.0f;
  return pt;
}

// 把整幅 StereoCloud 打包为 XYZRGB 点云，参考像素不带出。
inline pcl::PointCloud<pcl::PointXYZRGB>::Ptr StereoCloudToXyzrgb(const StereoCloud &cloud) {
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr out(new pcl::PointCloud<pcl::PointXYZRGB>());
  out->points.reserve(cloud.size());
  for (const auto &sp : cloud) {
    out->points.push_back(CopyStereoPointXyzrgb(sp));
  }
  out->width = static_cast<uint32_t>(out->points.size());
  out->height = 1;
  out->is_dense = true;
  return out;
}

// 按下标从 StereoCloud 打包 XYZRGB 点云，越界下标跳过。
inline pcl::PointCloud<pcl::PointXYZRGB>::Ptr StereoCloudToXyzrgbByIndices(
  const StereoCloud &cloud, const std::vector<int> &indices) {
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr out(new pcl::PointCloud<pcl::PointXYZRGB>());
  out->points.reserve(indices.size());
  for (const int idx : indices) {
    if (idx >= 0 && static_cast<size_t>(idx) < cloud.size()) {
      out->points.push_back(CopyStereoPointXyzrgb(cloud[static_cast<size_t>(idx)]));
    }
  }
  out->width = static_cast<uint32_t>(out->points.size());
  out->height = 1;
  out->is_dense = true;
  return out;
}

struct GeoFlowResult {
  cv::Mat geo_flow;
  cv::Mat geo_flow_trans; // 仅平移诱导的几何光流（诊断用）
  std::vector<int> valid_indices;
  // 无有效视差索引、出当前图像、相机后方或当前视图被更近表面遮挡：非动态、非静态，单独输出为不可见点云
  std::vector<int> invisible_indices;
  std::vector<cv::Point2f> project_points;
  std::vector<cv::Point> flow_valid;
};

// 采集的原始传感器数据结构体
struct FrameData {
  double st_time;  // 帧开始时间戳（秒）
  double end_time; // 帧结束时间戳（秒）
  std::deque<sensor_msgs::msg::Imu::SharedPtr> imu;
  cv::Mat imgl;
  cv::Mat imgr;
  int frame_id;
  FrameData() {
    st_time = 0.0;
    end_time = 0.0;
    frame_id = -1;
  }
};
struct FrontFrameData {
  double last_timestamp;
  double timestamp;
  cv::Mat left_img;
  cv::Mat right_img;
  std::deque<sensor_msgs::msg::Imu::SharedPtr> imu; // 修复 std::
  pcl::PointCloud<PointTypeRGB>::Ptr pcd;
  cv::Mat disp_filtered;
  cv::Mat confidence_map;
  std::vector<float> point_confidences;
  std::vector<Eigen::Matrix3d> point_covariances;
};
struct ImuData {
  long long timestamp; // 使用 long long 存储微秒级时间戳
  double ax;
  double ay;
  double az;
  double gx;
  double gy;
  double gz;
};

// ── 结构体1：IMU 运动学传递结果（普通帧，轻量级） ──────────────────────
// 用途：在相邻两帧之间通过 IMU 积分推算位姿增量，
//       为 ColorICP / ESKF 提供初值，帧处理完成后即可丢弃
// 特点：无雅可比，无协方差，仅保存增量本身
struct ImuKinematics {
  double timestamp; // 当前帧时间戳（秒）
  double dt;        // 积分时间间隔（秒）

  double delta_p[3]; // 位移增量（本体坐标系）
  double delta_v[3]; // 速度增量（本体坐标系）
  double delta_R[9]; // 旋转增量（本体坐标系，行优先 3x3）

  ImuKinematics() {
    timestamp = 0.0;
    dt = 0.0;
    std::fill(delta_p, delta_p + 3, 0.0);
    std::fill(delta_v, delta_v + 3, 0.0);
    std::fill(delta_R, delta_R + 9, 0.0);
    delta_R[0] = delta_R[4] = delta_R[8] = 1.0;
  }
};

// ── 结构体2：IMU 预积分结果（关键帧，完整版） ──────────────────────────
// 用途：持久化存入关键帧，供后端图优化构建预积分残差
// 特点：包含偏置快照、噪声协方差、偏置修正雅可比
// bias 更新后可通过雅可比一阶线性修正，无需重新积分：
//   ΔR_new ≈ ΔR · Exp(J_R_bg · δbg)
//   Δv_new ≈ Δv + J_v_bg · δbg + J_v_ba · δba
//   Δp_new ≈ Δp + J_p_bg · δbg + J_p_ba · δba
struct ImuPreintegration {
  double timestamp; // 起始关键帧时间戳（秒）
  double dt;        // 两关键帧之间的总积分时长（秒）

  // ── 预积分增量 ────────────────────────────────────────────────
  double delta_p[3]; // 位移增量（本体坐标系）
  double delta_v[3]; // 速度增量（本体坐标系）
  double delta_R[9]; // 旋转增量（本体坐标系，行优先 3x3）

  // ── 线性化点处的 bias 快照 ────────────────────────────────────
  double bias_gyr[3]; // 陀螺仪偏置 bg（线性化时刻）
  double bias_acc[3]; // 加速度计偏置 ba（线性化时刻）

  // ── 预积分噪声协方差（9x9：ΔR, Δv, Δp）─────────────────────
  double cov[81]; // 行优先展开，用于后端残差权重

  // ── 偏置修正雅可比（3x3，行优先展开） ─────────────────────────
  double J_R_bg[9]; // ∂ΔR / ∂bg
  double J_v_bg[9]; // ∂Δv / ∂bg
  double J_v_ba[9]; // ∂Δv / ∂ba
  double J_p_bg[9]; // ∂Δp / ∂bg
  double J_p_ba[9]; // ∂Δp / ∂ba

  ImuPreintegration() {
    timestamp = 0.0;
    dt = 0.0;
    std::fill(delta_p, delta_p + 3, 0.0);
    std::fill(delta_v, delta_v + 3, 0.0);
    std::fill(delta_R, delta_R + 9, 0.0);
    delta_R[0] = delta_R[4] = delta_R[8] = 1.0;
    std::fill(bias_gyr, bias_gyr + 3, 0.0);
    std::fill(bias_acc, bias_acc + 3, 0.0);
    std::fill(cov, cov + 81, 0.0);
    std::fill(J_R_bg, J_R_bg + 9, 0.0);
    std::fill(J_v_bg, J_v_bg + 9, 0.0);
    std::fill(J_v_ba, J_v_ba + 9, 0.0);
    std::fill(J_p_bg, J_p_bg + 9, 0.0);
    std::fill(J_p_ba, J_p_ba + 9, 0.0);
  }
};

// 向后兼容别名（旧代码中 Pose6D 的引用可逐步替换）
using Pose6D = ImuKinematics;

// 卡尔曼滤波计算后的本体（IMU）位姿状态
// 状态向量维度 DIM_STATE=21: rot(3), pos(3), vel(3), bg(3), ba(3), g(3),
// expo(3)
struct ImuStatesGroup {
  // ── 时间信息 ───────────────────────────────────────────
  double timestamp; // 当前状态对应的时间戳（秒）

  // ── 本体位姿（全局坐标系，由滤波器更新输出） ─────────────
  // ImuStatesToMatrix4d → T_w_body，p_world = T_w_body * p_body
  M3D rot_end; // 旋转矩阵 R_wb（世界系←本体系）
  V3D pos_end; // 位置 p_wb（世界系下本体原点坐标）
  V3D vel_end; // 速度 v_wb（世界系下本体线速度）

  // ── IMU 传感器偏置 ──────────────────────────────────────
  V3D bias_g; // 陀螺仪零偏 bg
  V3D bias_a; // 加速度计零偏 ba

  // ── 环境/传感器辅助量 ──────────────────────────────────
  V3D gravity;          // 重力向量（世界系，自动估计）
  double inv_expo_time; // 曝光时间倒数（用于事件/图像同步补偿）

  // ── 滤波协方差矩阵 ──────────────────────────────────────
  MD(DIM_STATE, DIM_STATE) cov; // P 矩阵，维度 DIM_STATE × DIM_STATE

  // 默认构造
  ImuStatesGroup() {
    timestamp = 0.0;
    rot_end = M3D::Identity();
    pos_end = V3D::Zero();
    vel_end = V3D::Zero();
    bias_g = V3D::Zero();
    bias_a = V3D::Zero();
    gravity = V3D::Zero();
    inv_expo_time = 1.0;
    // 初始协方差：默认较小正定值，偏置部分更小
    cov = MD(DIM_STATE, DIM_STATE)::Identity() * 0.01;
    cov.block<3, 3>(6, 6) = M3D::Identity() * 0.00001;   // velocity, all axes
    cov.block<3, 3>(9, 9) = M3D::Identity() * 0.00001;   // gyro bias
    cov.block<3, 3>(12, 12) = M3D::Identity() * 0.00001; // accel bias
    cov.block<3, 3>(15, 15) = M3D::Identity() * 0.00001; // gravity
    cov(18, 18) = 0.00001;                            // exposure
  }

  // 拷贝构造
  ImuStatesGroup(const ImuStatesGroup &b) = default;
};

// 关键帧数据结构体，包含图像、点云、IMU数据/本体位姿状态等
struct KeyFrameData {
  double last_keyframe_time; // 上一个关键帧的时间戳（秒）
  double timestamp;          // ✅ 改为 double (秒)

  // 点云与图像数据
  pcl::PointCloud<PointTypeRGB>::Ptr pointcloud;
  cv::Mat image_left;
  cv::Mat image_right;

  // 位姿信息
  ImuStatesGroup imu_states;

  // IMU 预积分数据（两个关键帧之间）
  std::deque<sensor_msgs::msg::Imu::SharedPtr> imu_buffer;

  // 视差图（用于下一帧动态目标剔除的深度锚点）
  cv::Mat disp_filtered;

  // 稠密双目网络输出的观测置信度
  cv::Mat confidence_map;
  std::vector<float> point_confidences;
  std::vector<Eigen::Matrix3d> point_covariances;

  // 其他辅助信息
  int keyframe_id = -1;
  // 前端稳定 ID：CreateBackendKeyFrame 会覆盖 keyframe_id，回传匹配必须用本字段。
  int frontend_keyframe_id = -1;
  int prev_frontend_keyframe_id = -1;
  // 与上一前端关键帧之间的后验相对边（ESIKF 融合后）：T_prev_curr = T_w_prev^{-1} T_w_curr
  Eigen::Matrix4d relative_T_prev_curr = Eigen::Matrix4d::Identity();
  // 速度相对量（上一关键帧机体系）：v_rel^b = R_w_prev^T (v_curr - v_prev)
  Eigen::Vector3d relative_v_prev_curr_body = Eigen::Vector3d::Zero();
  bool has_relative_edge = false;
  bool is_backend_keyframe = true;
};

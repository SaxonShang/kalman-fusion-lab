#pragma once
// 运行集成测试的时候, 不用重复写内容

#include "color_icp.h"
#include "sio_frontend_modules.h"
#include "utils.h"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <deque>
#include <limits>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include <pcl/search/kdtree.h>
#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

// 辅助函数
namespace sio_pipeline {

using ImuMsg = sensor_msgs::msg::Imu::SharedPtr;
// 判断IMU数据是否连续
inline bool IsImuDataContinuous(const std::deque<ImuMsg> &imu_buffer) {
  constexpr double kMaxGapSec = 0.01;
  for (std::size_t i = 1; i < imu_buffer.size(); ++i) {
    const double prev_t = svio_utils::MsgTimeSec(imu_buffer[i - 1]);
    const double curr_t = svio_utils::MsgTimeSec(imu_buffer[i]);
    if (curr_t - prev_t > kMaxGapSec) {
      return false;
    }
  }
  return true;
}

// 体素键
struct VoxelKey {
  int x = 0;
  int y = 0;
  int z = 0;

  bool operator==(const VoxelKey &other) const {
    return x == other.x && y == other.y && z == other.z;
  }
};

// 体素键哈希，用于将体素键转换为哈希值
struct VoxelKeyHash {
  std::size_t operator()(const VoxelKey &key) const {
    const std::size_t h1 = std::hash<int>{}(key.x);
    const std::size_t h2 = std::hash<int>{}(key.y);
    const std::size_t h3 = std::hash<int>{}(key.z);
    return h1 ^ (h2 << 1) ^ (h3 << 2); // 异或运算
  }
};
// 构建索引对齐的目标点云
// 输入：
// - cloud：输入点云
// - point_confidences：点云置信度
// - voxel_leaf：体素叶大小
// - out_cloud：输出点云
// - out_confidences：输出置信度
// - out_source_indices：输出源索引
// - point_covariances：点云协方差
// - out_covariances：输出协方差
// 输出：
// - 是否成功
inline bool BuildIndexAlignedTargetCloud(
    const pcl::PointCloud<PointTypeRGB>::Ptr &cloud,
    const std::vector<float> &point_confidences, float voxel_leaf,
    pcl::PointCloud<PointTypeRGB>::Ptr &out_cloud,
    std::vector<float> &out_confidences,
    std::vector<size_t> &out_source_indices,
    const std::vector<Eigen::Matrix3d> *point_covariances = nullptr,
    std::vector<Eigen::Matrix3d> *out_covariances = nullptr) {
  out_cloud.reset(new pcl::PointCloud<PointTypeRGB>());
  out_confidences.clear();
  out_source_indices.clear();
  if (out_covariances) {
    out_covariances->clear();
  }

  if (!cloud || cloud->empty() || !(voxel_leaf > 0.0f)) {
    return false;
  }

  std::unordered_set<VoxelKey, VoxelKeyHash> seen_voxels;
  seen_voxels.reserve(cloud->size());
  out_cloud->points.reserve(cloud->size());
  out_confidences.reserve(cloud->size());
  out_source_indices.reserve(cloud->size());
  if (out_covariances) {
    out_covariances->reserve(cloud->size());
  }

  for (size_t i = 0; i < cloud->size(); ++i) {
    const auto &pt = cloud->points[i];
    if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) {
      continue;
    }

    const VoxelKey key{
        static_cast<int>(std::floor(static_cast<double>(pt.x) / voxel_leaf)),
        static_cast<int>(std::floor(static_cast<double>(pt.y) / voxel_leaf)),
        static_cast<int>(std::floor(static_cast<double>(pt.z) / voxel_leaf))};
    if (!seen_voxels.insert(key).second) {
      continue;
    }

    out_cloud->points.push_back(pt);
    out_confidences.push_back(i < point_confidences.size() ? point_confidences[i] : 1.0f);
    out_source_indices.push_back(i);
    if (out_covariances) {
      if (point_covariances && i < point_covariances->size()) {
        out_covariances->push_back((*point_covariances)[i]);
      } else {
        out_covariances->push_back(Eigen::Matrix3d::Identity() * 1e-4);
      }
    }
  }

  out_cloud->width = static_cast<uint32_t>(out_cloud->points.size());
  out_cloud->height = 1;
  out_cloud->is_dense = cloud->is_dense;
  const bool cov_ok = !out_covariances || out_covariances->size() == out_cloud->size();
  return !out_cloud->empty() && out_confidences.size() == out_cloud->size()
         && out_source_indices.size() == out_cloud->size() && cov_ok;
}

// 特征缓存
struct FeatureCache {
  pcl::PointCloud<ColorICP::PointT>::Ptr normals_cloud;
  std::vector<Eigen::Vector3d> color_gradient;
  std::vector<Eigen::Vector3d> normals;
  pcl::KdTreeFLANN<PointTypeRGB>::Ptr rgb_kdtree;
  std::vector<pcl::search::KdTree<ColorICP::PointT>::Ptr> thread_kdtrees; // 线程KD树
}; 

// 构建特征缓存
// 输入：
// - cloud：输入点云
// - feature_proc：特征处理器
// - cache：特征缓存
// - thread_count：线程数
// 输出：
// - 是否成功
inline bool BuildFeatureCache(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud,
    FeatureProcessor &feature_proc, FeatureCache &cache,
    int thread_count = static_cast<int>(std::thread::hardware_concurrency())) {
  if (!cloud || cloud->empty()) {
    return false;
  }

  cache.normals_cloud.reset(new pcl::PointCloud<ColorICP::PointT>());
  feature_proc.EstimateNormals(cloud, cache.normals_cloud);
  if (!cache.normals_cloud || cache.normals_cloud->empty()) {
    return false;
  }

  cache.color_gradient.clear();
  feature_proc.PrepareColorGradient(cache.normals_cloud, cache.color_gradient);
  if (cache.color_gradient.size() != cache.normals_cloud->size()) {
    return false;
  }

  cache.normals.clear();
  cache.normals.reserve(cache.normals_cloud->size());
  for (const auto &pt : cache.normals_cloud->points) {
    cache.normals.emplace_back(pt.normal_x, pt.normal_y, pt.normal_z);
  }

  if (!cache.rgb_kdtree) {
    cache.rgb_kdtree.reset(new pcl::KdTreeFLANN<PointTypeRGB>());
  }
  cache.rgb_kdtree->setInputCloud(cloud);

  const int safe_threads = std::max(1, thread_count);
  cache.thread_kdtrees.resize(safe_threads);
  for (int i = 0; i < safe_threads; ++i) {
    if (!cache.thread_kdtrees[i]) {
      cache.thread_kdtrees[i].reset(new pcl::search::KdTree<ColorICP::PointT>());
    }
    cache.thread_kdtrees[i]->setInputCloud(cache.normals_cloud);
  }

  return true;
}
// 构建当前帧的法向量
// 输入：
// - cloud：输入点云
// - feature_proc：特征处理器
// - normals_cloud：输出法向量点云
// 输出：
// - 是否成功
inline bool BuildCurrentNormals(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud,
    FeatureProcessor &feature_proc,
    pcl::PointCloud<ColorICP::PointT>::Ptr &normals_cloud) {
  if (!cloud || cloud->empty()) {
    return false;
  }

  feature_proc.EstimateNormals(cloud, normals_cloud);
  return normals_cloud && !normals_cloud->empty();
}
//
// 发布点云
// 输入：
// - pub：发布器
// - cloud：输入点云
// - node：节点
// - frame_id：帧ID
// 输出：
// - 是否成功
inline void PublishCloud(
    const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr &pub,
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud,
    const rclcpp::Node::SharedPtr &node, const std::string &frame_id) {
  if (!pub || !cloud || cloud->empty()) {
    return;
  }

  sensor_msgs::msg::PointCloud2 msg;
  pcl::toROSMsg(*cloud, msg);
  msg.header.frame_id = frame_id;
  msg.header.stamp = node->get_clock()->now();
  pub->publish(msg);
}

} // namespace sio_pipeline

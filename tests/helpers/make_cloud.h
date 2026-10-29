#pragma once

#include "data.h"

#include <Eigen/Dense>
#include <pcl/common/transforms.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace sio_test {

inline PointTypeRGB MakePoint(float x, float y, float z, uint8_t r = 128,
                              uint8_t g = 128, uint8_t b = 128) {
  PointTypeRGB pt;
  pt.x = x;
  pt.y = y;
  pt.z = z;
  pt.r = r;
  pt.g = g;
  pt.b = b;
  return pt;
}

inline pcl::PointCloud<PointTypeRGB>::Ptr MakeLineCloud(int count,
                                                        float spacing = 0.05f) {
  auto cloud = pcl::PointCloud<PointTypeRGB>::Ptr(
      new pcl::PointCloud<PointTypeRGB>());
  cloud->reserve(static_cast<std::size_t>(count));
  for (int i = 0; i < count; ++i) {
    const auto v = static_cast<float>(i);
    cloud->push_back(MakePoint(v * spacing, 0.1f * std::sin(v), 1.0f,
                               static_cast<uint8_t>(20 + i),
                               static_cast<uint8_t>(80 + i),
                               static_cast<uint8_t>(120 + i)));
  }
  cloud->width = cloud->size();
  cloud->height = 1;
  cloud->is_dense = true;
  return cloud;
}

inline pcl::PointCloud<PointTypeRGB>::Ptr MakeGridCloud(int width, int height,
                                                        float spacing = 0.04f,
                                                        float z = 1.0f) {
  auto cloud = pcl::PointCloud<PointTypeRGB>::Ptr(
      new pcl::PointCloud<PointTypeRGB>());
  cloud->reserve(static_cast<std::size_t>(width * height));
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      cloud->push_back(MakePoint(static_cast<float>(x) * spacing,
                                 static_cast<float>(y) * spacing, z,
                                 static_cast<uint8_t>(x * 7),
                                 static_cast<uint8_t>(y * 9), 180));
    }
  }
  cloud->width = cloud->size();
  cloud->height = 1;
  cloud->is_dense = true;
  return cloud;
}

inline pcl::PointCloud<PointTypeRGB>::Ptr TransformCloud(
    const pcl::PointCloud<PointTypeRGB>::Ptr &cloud, const Eigen::Matrix4f &T) {
  auto out = pcl::PointCloud<PointTypeRGB>::Ptr(
      new pcl::PointCloud<PointTypeRGB>());
  pcl::transformPointCloud(*cloud, *out, T);
  return out;
}

} // namespace sio_test

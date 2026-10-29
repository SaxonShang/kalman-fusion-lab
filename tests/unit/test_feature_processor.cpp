#include "helpers/assert_eigen.h"
#include "helpers/make_cloud.h"

#include "sio_frontend_modules.h"

#include <Eigen/Dense>
#include <cmath>
#include <gtest/gtest.h>
#include <pcl/kdtree/kdtree_flann.h>

TEST(FeatureProcessorTest, FindsNearestCorrespondencesWithinThreshold) {
  FeatureProcessor processor;
  const auto target = sio_test::MakeGridCloud(6, 4);
  const auto source = sio_test::MakeGridCloud(6, 4);
  for (auto &pt : source->points) {
    pt.x += 0.005f;
  }
  auto kdtree = pcl::KdTreeFLANN<PointTypeRGB>::Ptr(
      new pcl::KdTreeFLANN<PointTypeRGB>());
  kdtree->setInputCloud(target);

  std::vector<int> matches;
  processor.FindCorrespondences(source, kdtree, matches, 0.01f);

  ASSERT_EQ(matches.size(), source->size());
  for (int idx : matches) {
    EXPECT_GE(idx, 0);
  }
}

TEST(FeatureProcessorTest, RejectsCorrespondencesBeyondThreshold) {
  FeatureProcessor processor;
  const auto target = sio_test::MakeGridCloud(4, 4);
  const auto source = sio_test::MakeGridCloud(4, 4);
  for (auto &pt : source->points) {
    pt.x += 10.0f;
  }
  auto kdtree = pcl::KdTreeFLANN<PointTypeRGB>::Ptr(
      new pcl::KdTreeFLANN<PointTypeRGB>());
  kdtree->setInputCloud(target);

  std::vector<int> matches;
  processor.FindCorrespondences(source, kdtree, matches, 0.01f);

  ASSERT_EQ(matches.size(), source->size());
  for (int idx : matches) {
    EXPECT_EQ(idx, -1);
  }
}

TEST(FeatureProcessorTest, EstimatesFiniteCovariances) {
  FeatureProcessor processor;
  const auto cloud = sio_test::MakeGridCloud(8, 5);
  std::vector<Eigen::Vector3d> gradients(cloud->size(), Eigen::Vector3d::Ones());
  std::vector<Eigen::Matrix3d> geo_covariances;
  std::vector<double> color_variances;
  std::vector<float> confidences(cloud->size(), 0.75f);

  processor.EstimatePointCovariances(cloud, gradients, geo_covariances,
                                     color_variances, confidences);

  ASSERT_EQ(geo_covariances.size(), cloud->size());
  ASSERT_EQ(color_variances.size(), cloud->size());
  for (std::size_t i = 0; i < geo_covariances.size(); ++i) {
    sio_test::ExpectMatrixFinite(geo_covariances[i]);
    sio_test::ExpectSymmetric(geo_covariances[i], 1e-9);
    EXPECT_GT(color_variances[i], 0.0);
    EXPECT_TRUE(std::isfinite(color_variances[i]));
  }
}

TEST(FeatureProcessorTest, PreparesColorGradientForEveryPoint) {
  FeatureProcessor processor;
  const auto cloud = sio_test::MakeGridCloud(8, 5);
  std::vector<Eigen::Vector3d> gradients;

  processor.PrepareColorGradient(cloud, gradients);

  ASSERT_EQ(gradients.size(), cloud->size());
  for (const auto &gradient : gradients) {
    sio_test::ExpectMatrixFinite(gradient);
  }
}

#include "helpers/assert_eigen.h"
#include "helpers/make_cloud.h"

#include "stereo_gicp.h"

#include <gtest/gtest.h>
#include <limits>

TEST(StereoObservationCompatTest, SearchRadiusGrowsAtNearRange) {
  EXPECT_EQ(sio_stereo_gicp::SearchRadiusPx(2.0, 500.0), 15);
  EXPECT_EQ(sio_stereo_gicp::SearchRadiusPx(10.0, 500.0), 5);
  EXPECT_EQ(sio_stereo_gicp::SearchRadiusPx(50.0, 500.0), sio_stereo_gicp::kSearchRadiusMinPx);
}

TEST(StereoObservationCompatTest, HuberWeightDownweightsLargeMahalanobis) {
  EXPECT_DOUBLE_EQ(sio_stereo_gicp::HuberWeight(0.0), 1.0);
  EXPECT_DOUBLE_EQ(sio_stereo_gicp::HuberWeight(1.0), 1.0);
  const double delta = sio_stereo_gicp::kHuberDelta;
  const double mahalanobis_sq = (2.0 * delta) * (2.0 * delta);
  EXPECT_NEAR(sio_stereo_gicp::HuberWeight(mahalanobis_sq), 0.5, 1e-12);
}

TEST(StereoObservationCompatTest, CovarianceHasLargerDepthVariance) {
  const CameraIntrinsics camera{500.0f, 500.0f, 320.0f, 240.0f, 0.12f, 640, 480};
  const Eigen::Matrix3d sigma =
    sio_stereo_gicp::PointCovarianceFromXyz(Eigen::Vector3d(0.0, 0.0, 5.0), camera);
  EXPECT_GT(sigma(2, 2), sigma(0, 0));
  EXPECT_GT(sigma(2, 2), sigma(1, 1));
  sio_test::ExpectSymmetric(sigma, 1e-12);
}

TEST(StereoObservationCompatTest, ProjectiveAssociateFindsSameRayTarget) {
  const CameraIntrinsics camera{500.0f, 500.0f, 320.0f, 240.0f, 0.12f, 640, 480};
  auto source = pcl::PointCloud<PointTypeRGB>::Ptr(new pcl::PointCloud<PointTypeRGB>());
  source->push_back(sio_test::MakePoint(0.0f, 0.0f, 2.0f));
  source->width = 1;
  source->height = 1;

  auto target = pcl::PointCloud<PointTypeRGB>::Ptr(new pcl::PointCloud<PointTypeRGB>());
  target->push_back(sio_test::MakePoint(0.0f, 0.0f, 2.05f));
  target->width = 1;
  target->height = 1;

  const auto splat = sio_stereo_gicp::SplatTarget(*target, camera, 640, 480, 2);
  const auto matches = sio_stereo_gicp::ProjectiveAssociate(
    *source, Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero(), splat.depth, camera,
    splat.index);

  ASSERT_EQ(matches.size(), 1u);
  EXPECT_EQ(matches.front().src_index, 0);
  EXPECT_NEAR(matches.front().q.z(), 2.05, 1e-3);
  const Eigen::Vector3d residual = matches.front().p_prime - matches.front().q;
  EXPECT_NEAR(residual.x(), 0.0, 1e-4);
  EXPECT_NEAR(residual.y(), 0.0, 1e-4);
  EXPECT_NEAR(residual.z(), -0.05, 1e-3);
}

TEST(StereoObservationCompatTest, ProjectiveAssociateKeepsPixelOffset) {
  const CameraIntrinsics camera{500.0f, 500.0f, 320.0f, 240.0f, 0.12f, 640, 480};
  // 落到像素中心旁 0.4 px，q 取像素中心光线，横向残差约 0.4 * z / fx。
  const float z = 2.0f;
  const float x = static_cast<float>((10.4 - 0.0) * z / 500.0);
  auto source = pcl::PointCloud<PointTypeRGB>::Ptr(new pcl::PointCloud<PointTypeRGB>());
  source->push_back(sio_test::MakePoint(x, 0.0f, z));
  source->width = 1;
  source->height = 1;

  cv::Mat depth(480, 640, CV_32FC1, cv::Scalar(std::numeric_limits<float>::quiet_NaN()));
  depth.at<float>(240, 330) = z;

  const auto matches = sio_stereo_gicp::ProjectiveAssociate(
    *source, Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero(), depth, camera);

  ASSERT_EQ(matches.size(), 1u);
  EXPECT_NEAR(matches.front().q.x(), 10.0 * z / 500.0, 1e-4);
  EXPECT_NEAR(matches.front().q.y(), 0.0, 1e-6);
  const Eigen::Vector3d residual = matches.front().p_prime - matches.front().q;
  EXPECT_NEAR(residual.x(), 0.4 * z / 500.0, 1e-4);
  EXPECT_NEAR(residual.y(), 0.0, 1e-6);
  EXPECT_NEAR(residual.z(), 0.0, 1e-6);
}

TEST(StereoObservationCompatTest, CameraPoseToBodyIsIdentityWhenExtrinsicIsIdentity) {
  Eigen::Matrix4d T_cam = Eigen::Matrix4d::Identity();
  T_cam(0, 3) = 0.1;
  const Eigen::Matrix4d T_body =
    sio_stereo_gicp::CameraPoseToBody(T_cam, Eigen::Matrix4d::Identity());
  sio_test::ExpectMatrixNear(T_body, T_cam, 1e-12);
}

#include "registration/stereo_gicp.h"

#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <opencv2/core.hpp>
#include <vector>

namespace {

sio_stereo_gicp::Calib MakeCalib() {
  sio_stereo_gicp::Calib calib;
  calib.fx = 500.0;
  calib.fy = 500.0;
  calib.cx = 320.0;
  calib.cy = 240.0;
  calib.baseline_m = 0.12;
  calib.width = 640;
  calib.height = 480;
  return calib;
}

Eigen::Matrix4d MakeYawTx(double yaw_rad, double tx) {
  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
  T.block<3, 3>(0, 0) =
      Eigen::AngleAxisd(yaw_rad, Eigen::Vector3d::UnitY()).toRotationMatrix();
  T(0, 3) = tx;
  return T;
}

double RotationAngleDeg(const Eigen::Matrix3d &R) {
  const double c = std::clamp((R.trace() - 1.0) / 2.0, -1.0, 1.0);
  return std::acos(c) * 180.0 / M_PI;
}

} // namespace

TEST(StereoGicpTest, CovarianceMatchesAnalyticAtOpticalCenter) {
  const auto calib = MakeCalib();
  sio_stereo_gicp::Settings settings;
  settings.eps_m2 = 0.0;
  const Eigen::Vector3d p(0.0, 0.0, 4.0);
  const Eigen::Matrix3d sigma =
      sio_stereo_gicp::PointCovarianceFromXyz(p, calib, settings);

  const double dz_dd = -(4.0 * 4.0) / (calib.fx * calib.baseline_m);
  const double var_z = dz_dd * dz_dd * settings.sigma_d * settings.sigma_d;
  const double var_x =
      std::pow(4.0 / calib.fx, 2) * settings.sigma_u * settings.sigma_u;
  const double var_y =
      std::pow(4.0 / calib.fy, 2) * settings.sigma_v * settings.sigma_v;

  EXPECT_NEAR(sigma(0, 0), var_x, 1e-12);
  EXPECT_NEAR(sigma(1, 1), var_y, 1e-12);
  EXPECT_NEAR(sigma(2, 2), var_z, 1e-12);
  EXPECT_NEAR(sigma(0, 2), 0.0, 1e-12);
  EXPECT_NEAR(sigma(1, 2), 0.0, 1e-12);
  EXPECT_GT(sigma(2, 2), sigma(0, 0));
}

TEST(StereoGicpTest, OffAxisHasXzCouplingAndSigmaZGrowsWithZ) {
  const auto calib = MakeCalib();
  sio_stereo_gicp::Settings settings;
  settings.eps_m2 = 0.0;
  const Eigen::Vector3d near_p(1.0, -1.0, 3.0);
  const Eigen::Vector3d far_p(4.0, -4.0, 12.0);
  const Eigen::Matrix3d near_s =
      sio_stereo_gicp::PointCovarianceFromXyz(near_p, calib, settings);
  const Eigen::Matrix3d far_s =
      sio_stereo_gicp::PointCovarianceFromXyz(far_p, calib, settings);

  const double dz_dd = -(3.0 * 3.0) / (calib.fx * calib.baseline_m);
  const double cov_xz =
      (near_p.x() / near_p.z()) * dz_dd * dz_dd * settings.sigma_d * settings.sigma_d;
  EXPECT_NEAR(near_s(0, 2), cov_xz, 1e-10);
  EXPECT_NE(near_s(0, 2), 0.0);
  EXPECT_GT(far_s(2, 2), near_s(2, 2) * 10.0);
}

TEST(StereoGicpTest, DepthFromDisparityInvertsTriangulation) {
  const auto calib = MakeCalib();
  cv::Mat disp(2, 2, CV_32FC1, cv::Scalar(0.0f));
  const double z = 6.0;
  disp.at<float>(1, 1) = static_cast<float>(calib.fx * calib.baseline_m / z);
  const cv::Mat depth = sio_stereo_gicp::DepthFromDisparity(disp, calib, 0.5, 15.0);
  EXPECT_TRUE(std::isnan(depth.at<double>(0, 0)));
  EXPECT_NEAR(depth.at<double>(1, 1), z, 1e-6);
}

TEST(StereoGicpTest, ProjectiveAssociateIdentityKeepsRay) {
  const auto calib = MakeCalib();
  sio_stereo_gicp::Settings settings;
  settings.search_radius = 1;
  settings.zmin = 0.5;
  settings.zmax = 15.0;

  std::vector<Eigen::Vector3d> pts;
  for (int y = -4; y <= 4; ++y) {
    for (int x = -4; x <= 4; ++x) {
      pts.emplace_back(0.05 * x, 0.05 * y, 3.0);
    }
  }
  const cv::Mat depth =
      sio_stereo_gicp::SplatDepth(pts, calib, calib.width, calib.height, 0.5,
                                  15.0, 1);
  const auto assoc = sio_stereo_gicp::ProjectiveAssociate(
      pts, Eigen::Matrix4d::Identity(), depth, calib, settings);
  ASSERT_GT(assoc.src_idx.size(), 20u);
  for (std::size_t i = 0; i < assoc.src_idx.size(); ++i) {
    const Eigen::Vector3d &p = pts[static_cast<std::size_t>(assoc.src_idx[i])];
    const Eigen::Vector3d &q = assoc.q_xyz[i];
    const Eigen::Vector3d cross = p.cross(q);
    EXPECT_LT(cross.norm(), 1e-6);
    EXPECT_NEAR(q.z(), 3.0, 0.05);
  }
}

TEST(StereoGicpTest, RecoversSmallYawFromImuLikeInit) {
  const auto calib = MakeCalib();
  sio_stereo_gicp::Settings settings;
  settings.voxel_size = 0.0;
  settings.max_iteration = 25;
  settings.num_threads = 1;
  settings.zmin = 0.5;
  settings.zmax = 15.0;

  std::vector<Eigen::Vector3d> target_pts;
  for (int y = -12; y <= 12; ++y) {
    for (int x = -16; x <= 16; ++x) {
      target_pts.emplace_back(0.08 * x, 0.08 * y, 3.2);
    }
  }
  // 只验旋转精修：单平面上 yaw 与 tx 简并，平移不是这个模型的强观测量。
  const Eigen::Matrix4d T_true = MakeYawTx(0.8 * M_PI / 180.0, 0.0);
  std::vector<Eigen::Vector3d> source_pts;
  source_pts.reserve(target_pts.size());
  const Eigen::Matrix4d T_inv = T_true.inverse();
  for (const auto &q : target_pts) {
    const Eigen::Vector3d p =
        T_inv.block<3, 3>(0, 0) * q + T_inv.block<3, 1>(0, 3);
    source_pts.push_back(p);
  }
  const cv::Mat depth = sio_stereo_gicp::SplatDepth(
      target_pts, calib, calib.width, calib.height, 0.5, 15.0, 1);

  const Eigen::Matrix4d T_init = MakeYawTx(1.05 * M_PI / 180.0, 0.0);
  const auto result =
      sio_stereo_gicp::AlignProjective(source_pts, depth, calib, T_init, settings);

  ASSERT_TRUE(result.converged) << result.status;
  EXPECT_GT(result.fitness, 0.5);
  const Eigen::Matrix3d dR =
      T_true.block<3, 3>(0, 0).transpose() * result.T_target_source.block<3, 3>(0, 0);
  const double init_err = RotationAngleDeg(
      T_true.block<3, 3>(0, 0).transpose() * T_init.block<3, 3>(0, 0));
  EXPECT_LT(RotationAngleDeg(dR), 0.12);
  EXPECT_LT(RotationAngleDeg(dR), init_err * 0.5);
  EXPECT_LT((result.T_target_source.block<3, 1>(0, 3).norm()), 0.03);
}

TEST(StereoGicpTest, IdentityInitOnMatchingCloudStaysNearIdentity) {
  const auto calib = MakeCalib();
  sio_stereo_gicp::Settings settings;
  settings.voxel_size = 0.0;
  settings.max_iteration = 10;
  settings.num_threads = 1;
  std::vector<Eigen::Vector3d> pts;
  for (int y = -8; y <= 8; ++y) {
    for (int x = -8; x <= 8; ++x) {
      pts.emplace_back(0.06 * x, 0.06 * y, 2.8);
    }
  }
  const cv::Mat depth = sio_stereo_gicp::SplatDepth(
      pts, calib, calib.width, calib.height, 0.5, 15.0, 1);
  const auto result = sio_stereo_gicp::AlignProjective(
      pts, depth, calib, Eigen::Matrix4d::Identity(), settings);
  ASSERT_TRUE(result.converged) << result.status;
  EXPECT_LT((RotationAngleDeg(result.T_target_source.block<3, 3>(0, 0))), 0.05);
  EXPECT_LT((result.T_target_source.block<3, 1>(0, 3).norm()), 0.01);
}

TEST(StereoGicpTest, BodyCameraConjugationRoundtrip) {
  Eigen::Matrix4d T_imu_cam = Eigen::Matrix4d::Identity();
  T_imu_cam(0, 3) = 0.10;
  T_imu_cam(2, 3) = 0.05;
  const Eigen::Matrix4d T_cam = MakeYawTx(0.03, 0.04);
  const Eigen::Matrix4d T_body =
      sio_stereo_gicp::BodyFromCamera(T_cam, T_imu_cam);
  const Eigen::Matrix4d T_cam_back =
      sio_stereo_gicp::CameraFromBody(T_body, T_imu_cam);
  EXPECT_NEAR((T_cam_back - T_cam).norm(), 0.0, 1e-12);

  const Eigen::Matrix<double, 6, 6> cov_cam =
      Eigen::Matrix<double, 6, 6>::Identity();
  const auto cov_body =
      sio_stereo_gicp::CovarianceBodyFromCamera(cov_cam, T_imu_cam);
  EXPECT_TRUE(cov_body.allFinite());
  EXPECT_GT(cov_body.determinant(), 0.0);
}

TEST(StereoGicpTest, EmptyInputFailsGracefully) {
  const auto calib = MakeCalib();
  sio_stereo_gicp::Settings settings;
  const auto result = sio_stereo_gicp::AlignProjective(
      {}, cv::Mat(), calib, Eigen::Matrix4d::Identity(), settings);
  EXPECT_FALSE(result.converged);
  EXPECT_EQ(result.status, "empty_input");
}

TEST(StereoGicpTest, PrefersSameRayOverNearbySurface) {
  const auto calib = MakeCalib();
  sio_stereo_gicp::Settings settings;
  settings.search_radius = 1;

  const Eigen::Vector3d wall(0.0, 0.0, 8.0);
  std::vector<Eigen::Vector3d> src = {wall};
  cv::Mat depth(calib.height, calib.width, CV_64FC1,
                cv::Scalar(std::numeric_limits<double>::quiet_NaN()));
  const int u = static_cast<int>(std::lround(calib.cx));
  const int v = static_cast<int>(std::lround(calib.cy));
  depth.at<double>(v, u) = 8.30;
  depth.at<double>(v, u + 3) = 7.95;

  const auto assoc = sio_stereo_gicp::ProjectiveAssociate(
      src, Eigen::Matrix4d::Identity(), depth, calib, settings);
  ASSERT_EQ(assoc.src_idx.size(), 1u);
  EXPECT_NEAR(assoc.z_j[0], 8.30, 1e-9);
  const Eigen::Vector3d cross = wall.cross(assoc.q_xyz[0]);
  EXPECT_LT(cross.norm(), 1e-6);
}

TEST(StereoGicpTest, VoxelRepresentativeKeepsMeasuredPointAndItsCovariance) {
  // 首个体素质心为0.4，最近实测点为0.3；同时验证无效点不会产生体素。
  std::vector<Eigen::Vector3d> points{{0.1,0.1,2.1},{0.3,0.1,2.1},{0.8,0.1,2.1},
      {1.2,0.1,2.1},{std::numeric_limits<double>::quiet_NaN(),0.,2.1}};
  std::vector<Eigen::Matrix3d> covariance;
  for(int i=0;i<5;++i)covariance.push_back((i+1)*Eigen::Matrix3d::Identity());
  std::vector<Eigen::Vector3d> output;
  std::vector<Eigen::Matrix3d> output_covariance;
  sio_stereo_gicp::VoxelDownsample(points,covariance,1.0,&output,&output_covariance,true);
  ASSERT_EQ(output.size(),2u);
  for(size_t i=0;i<output.size();++i) {
    const size_t expected=output[i].x()<1.0 ? 1u : 3u;
    EXPECT_TRUE(output[i].isApprox(points[expected],1e-14));
    EXPECT_TRUE(output_covariance[i].isApprox(covariance[expected],1e-14));
  }
  // 默认路径仍返回质心及原来的平均协方差。
  sio_stereo_gicp::VoxelDownsample(points,covariance,1.0,&output,&output_covariance);
  for(size_t i=0;i<output.size();++i)if(output[i].x()<1.0) {
    EXPECT_NEAR(output[i].x(),0.4,1e-14);
    EXPECT_TRUE(output_covariance[i].isApprox(2*Eigen::Matrix3d::Identity(),1e-14));
  }
}

TEST(StereoGicpTest, VoxelRepresentativeTieAndEmptyInputAreDeterministic) {
  std::vector<Eigen::Vector3d> points{{0.25,0.0,2.0},{0.75,0.0,2.0}},output;
  std::vector<Eigen::Matrix3d> cov;
  sio_stereo_gicp::VoxelDownsample(points,{},1.0,&output,&cov,true);
  ASSERT_EQ(output.size(),1u);
  EXPECT_TRUE(output[0].isApprox(points[0],1e-14));
  EXPECT_TRUE(cov[0].isZero());
  sio_stereo_gicp::VoxelDownsample({}, {},1.0,&output,&cov,true);
  EXPECT_TRUE(output.empty());EXPECT_TRUE(cov.empty());
}

TEST(StereoGicpTest, PointPlaneEmptyInputPreservesPredictionAndFails) {
  const auto calib = MakeCalib();
  sio_stereo_gicp::Settings settings;
  settings.point_plane = true;
  const Eigen::Matrix4d initial = MakeYawTx(0.03, 0.04);
  const auto result = sio_stereo_gicp::AlignProjective(
      {}, cv::Mat(), calib, initial, settings);
  EXPECT_FALSE(result.converged);
  EXPECT_EQ(result.status, "empty_input");
  EXPECT_TRUE(result.T_target_source.isApprox(initial, 1e-14));
}

TEST(StereoGicpTest, PointPlaneBudgetAndMissingSupportRemainFailures) {
  auto calib = MakeCalib();
  calib.fx = calib.fy = 184.0;
  calib.cx = calib.cy = 16.0;
  calib.width = calib.height = 33;
  std::vector<Eigen::Vector3d> points;
  for (int y = 4; y < 29; ++y)
    for (int x = 4; x < 29; ++x)
      points.emplace_back((x - 16) * 2.0 / 184, (y - 16) * 2.0 / 184, 2.0);
  sio_stereo_gicp::Settings settings;
  settings.point_plane = settings.disparity_normals = true;
  settings.voxel_size = 0.0;
  settings.max_iteration = 0;
  const Eigen::Matrix4d initial = MakeYawTx(0.01, 0.02);
  const cv::Mat flat(33, 33, CV_64F, cv::Scalar(2.0));
  const auto exhausted = sio_stereo_gicp::AlignProjective(points, flat, calib, initial, settings);
  EXPECT_FALSE(exhausted.converged);
  EXPECT_EQ(exhausted.status, "max_iter");
  EXPECT_TRUE(exhausted.T_target_source.isApprox(initial, 1e-14));
  // 缺少目标支持时不得借原始像素、填洞或初值伪造有效收敛。
  settings.max_iteration = 20;
  const cv::Mat missing(33, 33, CV_64F,
                        cv::Scalar(std::numeric_limits<double>::quiet_NaN()));
  const auto unsupported = sio_stereo_gicp::AlignProjective(points, missing, calib, initial, settings);
  EXPECT_FALSE(unsupported.converged);
  EXPECT_EQ(unsupported.status, "too_few_correspondences");
  EXPECT_TRUE(unsupported.T_target_source.isApprox(initial, 1e-14));
}

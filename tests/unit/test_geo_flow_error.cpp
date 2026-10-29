#include "helpers/assert_eigen.h"

#include "internal/geo_flow_error.h"
#include "utils.h"

#include <Eigen/Dense>
#include <cmath>
#include <gtest/gtest.h>

TEST(GeoFlowErrorTest, MissingPoseSigmaUsesTransFloorAndZeroRotation) {
  const sio_geo_flow::DofSigma s = sio_geo_flow::MakeDofSigma(nullptr);

  EXPECT_FLOAT_EQ(s.wx, 0.0f);
  EXPECT_FLOAT_EQ(s.wy, 0.0f);
  EXPECT_FLOAT_EQ(s.wz, 0.0f);
  EXPECT_FLOAT_EQ(s.tx, sio_geo_flow::kMinSigmaTransM);
  EXPECT_FLOAT_EQ(s.ty, sio_geo_flow::kMinSigmaTransM);
  EXPECT_FLOAT_EQ(s.tz, sio_geo_flow::kMinSigmaTransM);
}

TEST(GeoFlowErrorTest, PoseSigmaSetsRotationAndFloorsTranslation) {
  Eigen::Matrix<double, 6, 1> pose_sigma;
  pose_sigma << 0.001, 0.1, 0.0, 0.001, 0.5, 0.0;
  const sio_geo_flow::DofSigma s = sio_geo_flow::MakeDofSigma(&pose_sigma);

  EXPECT_FLOAT_EQ(s.wx, 0.001f);
  EXPECT_FLOAT_EQ(s.wy, 0.1f);
  EXPECT_FLOAT_EQ(s.wz, 0.0f);
  EXPECT_FLOAT_EQ(s.tx, sio_geo_flow::kMinSigmaTransM);
  EXPECT_NEAR(s.ty, 0.5f, 1e-6f);
  EXPECT_FLOAT_EQ(s.tz, sio_geo_flow::kMinSigmaTransM);
}

TEST(GeoFlowErrorTest, YawUncertaintyIsMostlyHorizontalAtCenter) {
  const sio_geo_flow::FlowJac j =
    sio_geo_flow::ComputeFlowJac(0.0f, 0.0f, 370.0f, 370.0f, 0.0f);
  sio_geo_flow::DofSigma sigma;
  sigma.wx = sigma.wy = sigma.wz = 0.0f;
  sigma.wy = 0.01f;
  const sio_geo_flow::SigmaF s = sio_geo_flow::AccumulateSigmaF(j, sigma, false);
  EXPECT_GT(s.suu, 10.0f);
  EXPECT_NEAR(s.svv, 0.0f, 1e-4f);
}

TEST(GeoFlowErrorTest, TranslationInflatesNearMoreThanFar) {
  sio_geo_flow::DofSigma sigma;
  sigma.wx = sigma.wy = sigma.wz = 0.0f;
  sigma.tx = 0.02f;
  const sio_geo_flow::FlowJac near_j =
    sio_geo_flow::ComputeFlowJac(0.0f, 0.0f, 370.0f, 370.0f, 1.0f / 1.5f);
  const sio_geo_flow::FlowJac far_j =
    sio_geo_flow::ComputeFlowJac(0.0f, 0.0f, 370.0f, 370.0f, 1.0f / 8.0f);
  const float near_t =
    sio_geo_flow::TraceSqrt(sio_geo_flow::AccumulateSigmaF(near_j, sigma, true));
  const float far_t =
    sio_geo_flow::TraceSqrt(sio_geo_flow::AccumulateSigmaF(far_j, sigma, true));
  EXPECT_GT(near_t, 2.5f * far_t);
}

TEST(GeoFlowErrorTest, MahalanobisIsAnisotropicForYaw) {
  const sio_geo_flow::FlowJac j =
    sio_geo_flow::ComputeFlowJac(0.0f, 0.0f, 370.0f, 370.0f, 0.0f);
  sio_geo_flow::DofSigma sigma;
  sigma.wy = 0.01f;
  sio_geo_flow::SigmaF s = sio_geo_flow::AccumulateSigmaF(j, sigma, false);
  s = sio_geo_flow::AddIsotropicRidge(s, sio_geo_flow::kMinFerrPx);
  const float horiz = sio_geo_flow::MahalanobisSq(4.0f, 0.0f, s);
  const float vert = sio_geo_flow::MahalanobisSq(0.0f, 4.0f, s);
  EXPECT_GT(vert, horiz);
}

TEST(GeoFlowErrorTest, MissingPoseCovUsesTransFloorAndZeroRotation) {
  const Eigen::Matrix<float, 6, 6> P = sio_geo_flow::MakePoseCovCam(nullptr);
  const float t2 =
    sio_geo_flow::kMinSigmaTransM * sio_geo_flow::kMinSigmaTransM;

  EXPECT_NEAR(P(0, 0), 0.0f, 1e-8f);
  EXPECT_NEAR(P(1, 1), 0.0f, 1e-8f);
  EXPECT_NEAR(P(2, 2), 0.0f, 1e-8f);
  EXPECT_NEAR(P(3, 3), t2, 1e-8f);
  EXPECT_NEAR(P(4, 4), t2, 1e-8f);
  EXPECT_NEAR(P(5, 5), t2, 1e-8f);
}

TEST(GeoFlowErrorTest, PoseCovFloorsTranslationDiagAndKeepsCrossTerm) {
  Eigen::Matrix<double, 6, 6> cov = Eigen::Matrix<double, 6, 6>::Zero();
  cov(1, 1) = 0.01;
  cov(3, 3) = 1e-8;
  cov(1, 4) = 0.002;
  cov(4, 1) = 0.002;
  const Eigen::Matrix<float, 6, 6> P = sio_geo_flow::MakePoseCovCam(&cov);
  const float t2 =
    sio_geo_flow::kMinSigmaTransM * sio_geo_flow::kMinSigmaTransM;

  EXPECT_NEAR(P(1, 1), 0.01f, 1e-6f);
  EXPECT_NEAR(P(3, 3), t2, 1e-6f);
  EXPECT_NEAR(P(1, 4), 0.002f, 1e-6f);
  EXPECT_NEAR(P(4, 1), 0.002f, 1e-6f);
}

TEST(GeoFlowErrorTest, PoseCrossTermTiltsFlowEllipseOffAxes) {
  const sio_geo_flow::FlowJac j =
    sio_geo_flow::ComputeFlowJac(0.0f, 0.0f, 370.0f, 370.0f, 1.0f / 2.0f);
  sio_geo_flow::DofSigma sigma;
  sigma.wy = 0.01f;
  sigma.ty = 0.02f;
  const sio_geo_flow::SigmaF axis =
    sio_geo_flow::AccumulateSigmaF(j, sigma, true);

  Eigen::Matrix<float, 6, 6> P = Eigen::Matrix<float, 6, 6>::Zero();
  P(1, 1) = 0.01f * 0.01f;
  P(4, 4) = 0.02f * 0.02f;
  P(1, 4) = 0.9f * 0.01f * 0.02f;
  P(4, 1) = P(1, 4);
  const sio_geo_flow::SigmaF full =
    sio_geo_flow::AccumulateSigmaFFromPoseCov(j, P, true);

  EXPECT_NEAR(axis.suv, 0.0f, 1e-4f);
  EXPECT_GT(std::abs(full.suv), 1.0f);
}

TEST(UtilsTest, ImuStatePoseCovInCameraKeepsRotationPositionCross) {
  ImuStatesGroup imu;
  imu.rot_end = Eigen::Matrix3d::Identity();
  imu.cov.setZero();
  imu.cov(2, 2) = 0.04;           // yaw
  imu.cov(3, 3) = 0.01;           // world px
  imu.cov(2, 3) = 0.015;          // yaw–px
  imu.cov(3, 2) = 0.015;
  Eigen::Matrix4d T_imu_cam = Eigen::Matrix4d::Identity();

  const Eigen::Matrix<double, 6, 6> P_cam =
    svio_utils::ImuStatePoseCovInCamera(imu, T_imu_cam);
  EXPECT_NEAR(P_cam(2, 2), 0.04, 1e-12);
  EXPECT_NEAR(P_cam(3, 3), 0.01, 1e-12);
  EXPECT_NEAR(P_cam(2, 3), 0.015, 1e-12);
  EXPECT_NEAR(P_cam(3, 2), 0.015, 1e-12);
}

TEST(UtilsTest, ImuStatePoseCovInCameraLeverArmCouplesYawToCameraTranslation) {
  ImuStatesGroup imu;
  imu.rot_end = Eigen::Matrix3d::Identity();
  imu.cov.setZero();
  imu.cov(2, 2) = 0.04; // yaw variance
  Eigen::Matrix4d T_imu_cam = Eigen::Matrix4d::Identity();
  T_imu_cam(0, 3) = 0.1; // t_ic along body X

  const Eigen::Matrix<double, 6, 6> P_cam =
    svio_utils::ImuStatePoseCovInCamera(imu, T_imu_cam);
  // δt = −t × δθ: yaw δψ maps to camera +t_y = 0.1 δψ
  EXPECT_NEAR(P_cam(2, 2), 0.04, 1e-12);
  EXPECT_NEAR(P_cam(4, 4), 0.1 * 0.1 * 0.04, 1e-12);
  EXPECT_NEAR(P_cam(2, 4), 0.1 * 0.04, 1e-12);
  EXPECT_NEAR(P_cam(4, 2), 0.1 * 0.04, 1e-12);
}

TEST(UtilsTest, ImuStatePoseSigmaInCameraRotatesBodyAxes) {
  ImuStatesGroup imu;
  imu.rot_end = Eigen::Matrix3d::Identity();
  imu.cov.setZero();
  imu.cov(0, 0) = 0.04; // 0.2 rad about body X
  Eigen::Matrix4d T_imu_cam = Eigen::Matrix4d::Identity();
  T_imu_cam.block<3, 3>(0, 0) =
    Eigen::AngleAxisd(M_PI / 2.0, Eigen::Vector3d::UnitZ()).toRotationMatrix();

  const Eigen::Matrix<double, 6, 1> sigma =
    svio_utils::ImuStatePoseSigmaInCamera(imu, T_imu_cam);
  EXPECT_NEAR(sigma(0), 0.0, 1e-9);
  EXPECT_NEAR(sigma(1), 0.2, 1e-9);
  EXPECT_NEAR(sigma(2), 0.0, 1e-9);
}

TEST(DynamicFilterDepthBoundsTest, DisparityBoundsMatch30mMaxDepth) {
  constexpr float kFx = 373.2715f;
  sio_utils::DynamicFilterParams cfg;
  cfg.min_depth_m = 0.1f;
  cfg.max_depth_m = 30.0f;

  const auto bounds =
    sio_utils::ComputeDisparityDepthBounds(kFx, 0.119879f, cfg.min_depth_m, cfg.max_depth_m);

  EXPECT_NEAR(bounds.min_disparity, kFx * 0.119879f / 30.0f, 0.05f);
  EXPECT_NEAR(bounds.max_disparity, kFx * 0.119879f / 0.1f, 1.0f);
  EXPECT_FALSE(sio_utils::IsDisparityInDepthRange(0.5f, bounds));
  EXPECT_TRUE(sio_utils::IsDisparityInDepthRange(bounds.min_disparity, bounds));
}

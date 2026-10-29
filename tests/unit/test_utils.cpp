#include "helpers/assert_eigen.h"
#include "helpers/make_imu.h"

#include "utils.h"
#include "registration/stereo_gicp.h"

#include <Eigen/Dense>
#include <cmath>
#include <gtest/gtest.h>

namespace {
Eigen::Matrix<double, 6, 6> NumericalPoseJacobian(
    const Eigen::Matrix3d &anchor_rotation, const Eigen::Matrix4d &relative, bool left) {
  Eigen::Matrix4d anchor = Eigen::Matrix4d::Identity();
  anchor.block<3, 3>(0, 0) = anchor_rotation;
  anchor.block<3, 1>(0, 3) = Eigen::Vector3d(.3, -.7, .2);
  const Eigen::Matrix4d nominal = anchor * relative;
  constexpr double step = 1e-6;
  Eigen::Matrix<double, 6, 6> jacobian;
  for (int axis = 0; axis < 6; ++axis) {
    Eigen::Matrix<double, 6, 1> delta = Eigen::Matrix<double, 6, 1>::Zero();
    delta(axis) = step;
    const Eigen::Matrix4d perturbation = sio_stereo_gicp::Se3Exp(delta);
    const Eigen::Matrix4d changed = anchor * (left ? perturbation * relative : relative * perturbation);
    jacobian.col(axis).head<3>() = sio_utils::So3LogApprox(
      nominal.block<3, 3>(0, 0).transpose() * changed.block<3, 3>(0, 0)) / step;
    jacobian.col(axis).tail<3>() = (changed.block<3, 1>(0, 3) - nominal.block<3, 1>(0, 3)) / step;
  }
  return jacobian;
}
} // namespace

TEST(UtilsTest, StereoCovarianceAlignmentMatchesLeftPerturbation) {
  Eigen::Matrix<double, 6, 6> a = Eigen::Matrix<double, 6, 6>::Identity();
  a.diagonal() << 1, 2, 3, 4, 5, 6;
  a(0, 3) = .4; a(2, 5) = -.2;
  const Eigen::Matrix<double, 6, 6> covariance = a * a.transpose();
  const auto rotation = sio_utils::ExpSO3(Eigen::Vector3d(.2, -.3, .1));
  Eigen::Matrix4d relative = Eigen::Matrix4d::Identity();
  relative.block<3, 3>(0, 0) = sio_utils::ExpSO3(Eigen::Vector3d(-.1, .4, .3));
  relative.block<3, 1>(0, 3) = Eigen::Vector3d(.4, -.2, .1);
  const auto aligned = sio_utils::AlignStereoGicpCovarianceToEskfPose(covariance, rotation, relative);
  sio_test::ExpectSymmetric(aligned, 1e-12);
  const auto numerical = NumericalPoseJacobian(rotation, relative, true);
  sio_test::ExpectMatrixNear(aligned, numerical * covariance * numerical.transpose(), 1e-5);
  EXPECT_GT((aligned.block<3, 3>(0, 3).norm()), 0.0);
  EXPECT_GT((aligned - covariance).norm(), 0.1);
}

TEST(UtilsTest, MsgTimeSecUsesRosHeaderStamp) {
  const auto imu = sio_test::MakeImu(12.345, Eigen::Vector3d::Zero(),
                                    Eigen::Vector3d::Zero());

  EXPECT_NEAR(svio_utils::MsgTimeSec(imu), 12.345, 1e-9);
}

TEST(UtilsTest, ExpSO3ReturnsIdentityForTinyRotation) {
  const Eigen::Matrix3d R = svio_utils::ExpSO3(Eigen::Vector3d::Zero());

  sio_test::ExpectMatrixNear(R, Eigen::Matrix3d::Identity(), 1e-12);
}

TEST(UtilsTest, ExpSO3RotatesAroundZAxis) {
  const Eigen::Matrix3d R =
      svio_utils::ExpSO3(Eigen::Vector3d(0.0, 0.0, M_PI / 2.0));
  const Eigen::Vector3d x_axis = R * Eigen::Vector3d::UnitX();

  EXPECT_NEAR(x_axis.x(), 0.0, 1e-9);
  EXPECT_NEAR(x_axis.y(), 1.0, 1e-9);
  EXPECT_NEAR(x_axis.z(), 0.0, 1e-9);
}

TEST(UtilsTest, So3LogApproxMatchesSmallAngle) {
  const Eigen::Vector3d w(1e-4, -2e-4, 3e-4);
  const Eigen::Vector3d approx = svio_utils::So3LogApprox(svio_utils::ExpSO3(w));

  sio_test::ExpectMatrixNear(approx, w, 1e-8);
}

TEST(UtilsTest, ImuStatesToMatrix4dCopiesPose) {
  ImuStatesGroup state;
  state.rot_end = svio_utils::ExpSO3(Eigen::Vector3d(0.1, 0.2, -0.1));
  state.pos_end = Eigen::Vector3d(1.0, 2.0, 3.0);

  const Eigen::Matrix4d T = svio_utils::ImuStatesToMatrix4d(state);

  sio_test::ExpectMatrixNear(T.block<3, 3>(0, 0), state.rot_end, 1e-12);
  sio_test::ExpectMatrixNear(T.block<3, 1>(0, 3), state.pos_end, 1e-12);
  EXPECT_DOUBLE_EQ(T(3, 3), 1.0);
}

TEST(UtilsTest, AlignGicpCovarianceUsesFullPoseJacobian) {
  const Eigen::Matrix<double, 6, 6> gicp_cov =
      Eigen::Matrix<double, 6, 6>::Identity() * 0.01;
  const Eigen::Matrix3d R_wk = svio_utils::ExpSO3(Eigen::Vector3d(0.0, 0.0, 0.3));
  Eigen::Matrix4d T_kc = Eigen::Matrix4d::Identity();
  T_kc.block<3, 3>(0, 0) = svio_utils::ExpSO3(Eigen::Vector3d(0.1, -0.2, 0.05));
  T_kc.block<3, 1>(0, 3) = Eigen::Vector3d(0.4, -0.1, 0.2);

  const Eigen::Matrix<double, 6, 6> J =
      svio_utils::BuildGicpToEskfPoseJacobian(R_wk, T_kc);
  sio_test::ExpectMatrixNear(J, NumericalPoseJacobian(R_wk, T_kc, false), 1e-7);
  const Eigen::Matrix<double, 6, 6> expected = J * gicp_cov * J.transpose();
  const Eigen::Matrix<double, 6, 6> aligned =
      svio_utils::AlignGicpCovarianceToEskfPose(gicp_cov, R_wk, T_kc);

  sio_test::ExpectMatrixNear(aligned, expected, 1e-12);
  sio_test::ExpectSymmetric(aligned, 1e-12);
  const double det_rot = J.block<3, 3>(0, 0).determinant();
  EXPECT_NEAR(det_rot, 1.0, 1e-9);
  sio_test::ExpectMatrixNear(J.block<3, 3>(3, 3), R_wk * T_kc.block<3, 3>(0, 0), 1e-12);
}

TEST(UtilsTest, RightGicpRotationDoesNotMoveSourceOrigin) {
  Eigen::Matrix<double, 6, 6> gicp_cov = Eigen::Matrix<double, 6, 6>::Zero();
  gicp_cov.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * 0.02;
  const Eigen::Matrix3d R_wk = Eigen::Matrix3d::Identity();
  Eigen::Matrix4d T_kc = Eigen::Matrix4d::Identity();
  T_kc.block<3, 1>(0, 3) = Eigen::Vector3d(1.0, 0.0, 0.0);

  const Eigen::Matrix<double, 6, 6> aligned =
      svio_utils::AlignGicpCovarianceToEskfPose(gicp_cov, R_wk, T_kc);

  const double rot_to_trans_coupling = aligned.block<3, 3>(3, 0).norm();
  EXPECT_DOUBLE_EQ(rot_to_trans_coupling, 0.0);
  EXPECT_DOUBLE_EQ((aligned.bottomRightCorner<3, 3>().norm()), 0.0);
  sio_test::ExpectSymmetric(aligned, 1e-12);
}

TEST(UtilsTest, CurrPrevkfCamTransformMapsStaticPoint) {
  Eigen::Matrix4d T_w_kf = Eigen::Matrix4d::Identity();
  Eigen::Matrix4d T_w_curr = Eigen::Matrix4d::Identity();
  T_w_curr.block<3, 1>(0, 3) = Eigen::Vector3d(1.0, 0.0, 0.0);
  const Eigen::Matrix4d T_imu_cam = Eigen::Matrix4d::Identity();

  const Eigen::Matrix4d T_w_kf_cam = svio_utils::WorldCamFromBody(T_w_kf, T_imu_cam);
  const Eigen::Matrix4d T_w_curr_cam = svio_utils::WorldCamFromBody(T_w_curr, T_imu_cam);
  const Eigen::Matrix4d T_prevkf_curr_cam =
      svio_utils::RelativeTransform(T_w_kf_cam, T_w_curr_cam);
  const Eigen::Matrix4d T_curr_prevkf_cam = T_prevkf_curr_cam.inverse();

  const Eigen::Vector4d p_kf_cam(0.0, 0.0, 10.0, 1.0);
  const Eigen::Vector4d p_curr_cam = T_curr_prevkf_cam * p_kf_cam;

  sio_test::ExpectMatrixNear(p_curr_cam.head<3>(), Eigen::Vector3d(-1.0, 0.0, 10.0), 1e-9);

  const Eigen::Vector4d p_world = T_w_kf_cam * p_kf_cam;
  const Eigen::Vector4d p_curr_from_world = T_w_curr_cam.inverse() * p_world;
  sio_test::ExpectMatrixNear(p_curr_cam.head<3>(), p_curr_from_world.head<3>(), 1e-9);
}

TEST(UtilsTest, RelativeTransformMatchesManualComposition) {
  Eigen::Matrix4d T_w_a = Eigen::Matrix4d::Identity();
  T_w_a.block<3, 1>(0, 3) = Eigen::Vector3d(1.0, 0.0, 0.0);
  Eigen::Matrix4d T_w_b = Eigen::Matrix4d::Identity();
  T_w_b.block<3, 1>(0, 3) = Eigen::Vector3d(3.0, 0.5, 0.0);

  const Eigen::Matrix4d T_a_b = svio_utils::RelativeTransform(T_w_a, T_w_b);
  const Eigen::Vector4d p_b(0.0, 1.0, 2.0, 1.0);
  const Eigen::Vector4d p_a_manual = T_w_a.inverse() * T_w_b * p_b;
  sio_test::ExpectMatrixNear(T_a_b * p_b, p_a_manual, 1e-9);
}

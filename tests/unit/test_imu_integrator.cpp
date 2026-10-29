#include "helpers/assert_eigen.h"
#include "helpers/make_imu.h"

#include "imu_integrator.h"
#include "utils.h"

#include <Eigen/Dense>
#include <cmath>
#include <deque>
#include <gtest/gtest.h>

TEST(ImuIntegratorTest, DefaultPriorIsIsotropicWithinPhysicalStateBlocks) {
  const ImuStatesGroup state;
  // An isotropic default must not select a privileged velocity or bias axis.
  // A cyclic permutation changes no information about an uncalibrated state.
  Eigen::Matrix<double, DIM_STATE, DIM_STATE> axes =
    Eigen::Matrix<double, DIM_STATE, DIM_STATE>::Identity();
  Eigen::Matrix3d cycle;
  cycle << 0, 1, 0, 0, 0, 1, 1, 0, 0;
  for (int start : {0, 3, 6, 9, 12, 15}) axes.block<3, 3>(start, start) = cycle;
  sio_test::ExpectMatrixNear(state.cov, axes * state.cov * axes.transpose(), 1e-15);
}

TEST(ImuIntegratorTest, EmptyInitReturnsDefaultState) {
  ImuIntegrator integrator;
  const std::deque<sensor_msgs::msg::Imu::SharedPtr> empty;

  const ImuStatesGroup state = integrator.ImuInit(empty);

  EXPECT_DOUBLE_EQ(state.timestamp, 0.0);
  sio_test::ExpectMatrixNear(state.rot_end, Eigen::Matrix3d::Identity(), 1e-12);
  sio_test::ExpectMatrixNear(state.pos_end, Eigen::Vector3d::Zero(), 1e-12);
}

TEST(ImuIntegratorTest, StaticInitEstimatesGravityAndGyroBias) {
  ImuIntegrator integrator;
  const auto imu = sio_test::MakeStaticImu(
      1.0, 100, 100.0, Eigen::Vector3d(0.0, 0.0, 9.81),
      Eigen::Vector3d(0.01, -0.02, 0.03));

  const ImuStatesGroup state = integrator.ImuInit(imu);

  EXPECT_NEAR(state.gravity.norm(), 9.81, 1e-9);
  sio_test::ExpectMatrixNear(state.bias_g, Eigen::Vector3d(0.01, -0.02, 0.03),
                             1e-12);
  EXPECT_NEAR(state.timestamp, 1.99, 1e-9);
}

TEST(ImuIntegratorTest, StaticPreintegrationStaysNearRest) {
  ImuIntegrator integrator;
  KeyFrameData keyframe;
  keyframe.timestamp = 0.0;
  keyframe.imu_states.gravity = Eigen::Vector3d(0.0, 0.0, -9.81);
  const auto imu = sio_test::MakeStaticImu(0.0, 101, 100.0);

  const ImuStatesGroup state =
      integrator.ImuPreintegrationCov(keyframe, imu, 1.0);

  EXPECT_NEAR(state.timestamp, 1.0, 1e-12);
  sio_test::ExpectMatrixNear(state.pos_end, Eigen::Vector3d::Zero(), 1e-6);
  sio_test::ExpectMatrixNear(state.vel_end, Eigen::Vector3d::Zero(), 1e-6);
  sio_test::ExpectMatrixFinite(state.cov);
}

TEST(ImuIntegratorTest, PreintegrationIncrementIsInAnchorBodyFrame) {
  ImuIntegrator integrator;
  KeyFrameData keyframe;
  keyframe.timestamp = 0.0;
  keyframe.imu_states.gravity = Eigen::Vector3d(0.0, 0.0, -9.81);
  keyframe.imu_states.rot_end =
    Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  keyframe.imu_states.pos_end = Eigen::Vector3d(1.0, -0.5, 0.2);
  keyframe.imu_states.vel_end = Eigen::Vector3d(0.2, 0.1, 0.0);
  const auto imu = sio_test::MakeStaticImu(0.0, 101, 100.0);

  const ImuStatesGroup state =
    integrator.ImuPreintegrationCov(keyframe, imu, 1.0);
  const Pose6D &preint = integrator.GetLastPreintegrationIncrement();

  Eigen::Matrix3d dR = Eigen::Matrix3d::Identity();
  dR << preint.delta_R[0], preint.delta_R[1], preint.delta_R[2], preint.delta_R[3],
    preint.delta_R[4], preint.delta_R[5], preint.delta_R[6], preint.delta_R[7],
    preint.delta_R[8];
  const Eigen::Vector3d dp(preint.delta_p[0], preint.delta_p[1], preint.delta_p[2]);
  const Eigen::Vector3d dv(preint.delta_v[0], preint.delta_v[1], preint.delta_v[2]);

  ImuStatesGroup predicted = keyframe.imu_states;
  predicted.rot_end = keyframe.imu_states.rot_end * dR;
  predicted.pos_end = keyframe.imu_states.pos_end + keyframe.imu_states.rot_end * dp;
  predicted.vel_end = keyframe.imu_states.vel_end + keyframe.imu_states.rot_end * dv;

  sio_test::ExpectMatrixNear(state.rot_end, predicted.rot_end, 1e-9);
  sio_test::ExpectMatrixNear(state.pos_end, predicted.pos_end, 1e-9);
  sio_test::ExpectMatrixNear(state.vel_end, predicted.vel_end, 1e-9);
}

TEST(ImuIntegratorTest, ParallelCovFollowsSameTransitionFromItsOwnPrior) {
  ImuIntegrator integrator;
  KeyFrameData keyframe;
  keyframe.timestamp = 0.0;
  keyframe.imu_states.gravity = Eigen::Vector3d(0.0, 0.0, -9.81);
  keyframe.imu_states.cov.setIdentity();
  keyframe.imu_states.cov *= 0.2;
  const auto imu = sio_test::MakeStaticImu(0.0, 21, 100.0);

  Eigen::Matrix<double, DIM_STATE, DIM_STATE> parallel =
      Eigen::Matrix<double, DIM_STATE, DIM_STATE>::Zero();
  const ImuStatesGroup state =
      integrator.ImuPreintegrationCov(keyframe, imu, 0.2, &parallel);

  EXPECT_NEAR(keyframe.imu_states.cov(0, 0), 0.2, 1e-15);
  EXPECT_GT((state.cov - parallel).norm(), 1e-8);
  EXPECT_GT(parallel.norm(), 0.0);

  ImuIntegrator integrator_same;
  Eigen::Matrix<double, DIM_STATE, DIM_STATE> parallel_same = keyframe.imu_states.cov;
  const ImuStatesGroup state_same =
      integrator_same.ImuPreintegrationCov(keyframe, imu, 0.2, &parallel_same);
  sio_test::ExpectMatrixNear(parallel_same, state_same.cov, 1e-9);
  sio_test::ExpectMatrixNear(state_same.cov, state.cov, 1e-9);
}

TEST(ImuIntegratorTest, ConstantYawRateIntegratesRotation) {
  ImuIntegrator integrator;
  KeyFrameData keyframe;
  keyframe.timestamp = 0.0;
  keyframe.imu_states.gravity = Eigen::Vector3d::Zero();
  const auto imu = sio_test::MakeImuSequence(
      0.0, 101, 100.0, Eigen::Vector3d::Zero(), Eigen::Vector3d(0.0, 0.0, 1.0));

  const ImuStatesGroup state =
      integrator.ImuPreintegrationCov(keyframe, imu, 1.0);
  const Eigen::Vector3d x_axis = state.rot_end * Eigen::Vector3d::UnitX();

  EXPECT_NEAR(std::atan2(x_axis.y(), x_axis.x()), 1.0, 1e-3);
  sio_test::ExpectMatrixFinite(state.cov);
}

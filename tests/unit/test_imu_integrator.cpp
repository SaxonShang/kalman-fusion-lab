#include "helpers/assert_eigen.h"
#include "helpers/make_imu.h"

#include "imu_integrator.h"
#include "sio_math.h"
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

namespace {
using Mat21 = Eigen::Matrix<double, DIM_STATE, DIM_STATE>;

// 确定性的对称正定矩阵，用于协方差代数测试
Mat21 DeterministicSpd(double seed, double scale) {
  Mat21 a;
  for (int r = 0; r < DIM_STATE; ++r)
    for (int c = 0; c < DIM_STATE; ++c) a(r, c) = std::sin(seed + 0.7 * r + 1.3 * c);
  return scale * (a * a.transpose()) + 1e-4 * Mat21::Identity();
}

Mat21 DeterministicTransition(double seed) {
  Mat21 phi = Mat21::Identity();
  for (int r = 0; r < DIM_STATE; ++r)
    for (int c = 0; c < DIM_STATE; ++c) phi(r, c) += 0.1 * std::cos(seed + 0.9 * r - 0.4 * c);
  return phi;
}
} // namespace

TEST(ImuIntegratorTest, TransitionReproducesCovariancePropagation) {
  ImuIntegrator integrator;
  KeyFrameData keyframe;
  keyframe.timestamp = 0.0;
  keyframe.imu_states.gravity = Eigen::Vector3d(0.0, 0.0, -9.81);
  keyframe.imu_states.vel_end = Eigen::Vector3d(0.3, 0.0, 0.0);
  keyframe.imu_states.cov = DeterministicSpd(0.5, 1e-3);
  const auto imu = sio_test::MakeImuSequence(
    0.0, 41, 200.0, Eigen::Vector3d(0.5, 0.1, 9.81), Eigen::Vector3d(0.2, -0.1, 0.5));

  Mat21 noise = Mat21::Zero();
  Mat21 transition;
  // 0.203 s 让最后一段外推也参与 Φ
  const ImuStatesGroup state =
    integrator.ImuPreintegrationCov(keyframe, imu, 0.203, &noise, &transition);
  const Mat21 expected =
    transition * keyframe.imu_states.cov * transition.transpose() + noise;
  sio_test::ExpectMatrixNear(state.cov, expected, 1e-10);
  EXPECT_GT((transition - Mat21::Identity()).norm(), 1e-3);
}

TEST(ImuIntegratorTest, AnchorConditionedCovarianceMatchesGaussianConditioning) {
  const Mat21 anchor = DeterministicSpd(0.3, 1e-3);
  const Mat21 noise = DeterministicSpd(1.1, 1e-4);
  const Mat21 phi = DeterministicTransition(2.0);

  // 联合高斯 [x_a; x_b]，x_b = Φ x_a + w，再以 x_a 的前 6 维（位姿）为条件
  Eigen::MatrixXd joint(2 * DIM_STATE, 2 * DIM_STATE);
  joint.topLeftCorner(DIM_STATE, DIM_STATE) = anchor;
  joint.topRightCorner(DIM_STATE, DIM_STATE) = anchor * phi.transpose();
  joint.bottomLeftCorner(DIM_STATE, DIM_STATE) = phi * anchor;
  joint.bottomRightCorner(DIM_STATE, DIM_STATE) = phi * anchor * phi.transpose() + noise;
  const Eigen::MatrixXd s_bb = joint.bottomRightCorner(DIM_STATE, DIM_STATE);
  const Eigen::MatrixXd s_ba = joint.block(DIM_STATE, 0, DIM_STATE, 6);
  const Eigen::MatrixXd s_aa = joint.block(0, 0, 6, 6);
  const Eigen::MatrixXd expected = s_bb - s_ba * s_aa.ldlt().solve(s_ba.transpose());

  const Mat21 actual = ConditionOnAnchorPose(anchor, phi, noise);
  sio_test::ExpectMatrixNear(actual, expected, 1e-10);
  sio_test::ExpectSymmetric(actual, 1e-14);
}

TEST(ImuIntegratorTest, AnchorConditionedPoseUncertaintyVanishesWithoutMotion) {
  const Mat21 anchor = DeterministicSpd(0.7, 1e-3);
  const Mat21 conditioned = ConditionOnAnchorPose(anchor, Mat21::Identity(), Mat21::Zero());
  // 没有传播时当前位姿就是锚点位姿：位姿块为 0，速度等仍保留剩余不确定度
  sio_test::ExpectMatrixNear(conditioned.block<6, 6>(0, 0), Eigen::Matrix<double, 6, 6>::Zero(),
                             1e-12);
  EXPECT_GT((conditioned.block<3, 3>(6, 6).trace()), 1e-6);
}

TEST(ImuIntegratorTest, AnchorReferenceJacobianMatchesEvaluationReference) {
  // 真值位姿与小的估计误差；估计 = 真值 ⊟ 误差：R̂ = R Exp(−δθ)，p̂ = p − δp
  const Eigen::Matrix3d R_a = sio_utils::ExpSO3(Eigen::Vector3d(0.3, -0.2, 0.5));
  const Eigen::Vector3d p_a(0.4, -0.1, 1.2);
  const Eigen::Matrix3d R_b = R_a * sio_utils::ExpSO3(Eigen::Vector3d(0.05, 0.1, -0.08));
  const Eigen::Vector3d p_b = p_a + Eigen::Vector3d(0.3, 0.2, -0.1);
  Eigen::Matrix<double, 6, 1> da, db;
  da << 2e-5, -1e-5, 3e-5, 1e-5, 2e-5, -1e-5;
  db << -1e-5, 2e-5, 1e-5, 3e-5, -2e-5, 1e-5;
  const Eigen::Matrix3d Rh_a = R_a * sio_utils::ExpSO3(-da.head<3>());
  const Eigen::Vector3d ph_a = p_a - da.tail<3>();
  const Eigen::Matrix3d Rh_b = R_b * sio_utils::ExpSO3(-db.head<3>());
  const Eigen::Vector3d ph_b = p_b - db.tail<3>();

  // 评价参照 x̂_a · (x_a⁻¹ x_b)，误差 = 参照 ⊟ 当前估计（与 fusion_report.py 相同）
  const Eigen::Matrix3d R_ref = Rh_a * R_a.transpose() * R_b;
  const Eigen::Vector3d p_ref = Rh_a * R_a.transpose() * (p_b - p_a) + ph_a;
  Eigen::Matrix<double, 6, 1> error;
  error.head<3>() = sio_utils::LogSO3(Rh_b.transpose() * R_ref);
  error.tail<3>() = p_ref - ph_b;

  const Eigen::Matrix<double, 6, 6> J = AnchorReferenceJacobian(Rh_a, ph_a, Rh_b, ph_b);
  // 一阶相符，余项为误差的二次量（约 1e-9）；漏掉 J_pθ 杠杆项时差约 1e-5
  sio_test::ExpectMatrixNear(error, db - J * da, 1e-8);
  EXPECT_GT((J.block<3, 3>(3, 0)).norm(), 0.1);
}

TEST(ImuIntegratorTest, AnchorRelativeCovarianceMatchesRelativeErrorCovariance) {
  const Mat21 anchor = DeterministicSpd(0.3, 1e-3);
  const Mat21 noise = DeterministicSpd(1.1, 1e-4);
  const Mat21 phi = DeterministicTransition(2.0);
  const Eigen::Matrix<double, 6, 6> J = AnchorReferenceJacobian(
    sio_utils::ExpSO3(Eigen::Vector3d(0.1, 0.2, -0.3)), Eigen::Vector3d(0.2, 0.0, 1.0),
    sio_utils::ExpSO3(Eigen::Vector3d(0.15, 0.1, -0.2)), Eigen::Vector3d(0.5, 0.1, 0.8));

  // e = x_b − J̃ x_a[0:6]，J̃ = [J; 0]：只有位姿行减去评价参照带走的锚点位姿误差
  Eigen::Matrix<double, DIM_STATE, 6> J_full = Eigen::Matrix<double, DIM_STATE, 6>::Zero();
  J_full.topRows<6>() = J;
  const Eigen::MatrixXd s_bb = phi * anchor * phi.transpose() + noise;
  const Eigen::MatrixXd s_ba = phi * anchor.leftCols<6>();
  const Eigen::MatrixXd s_aa = anchor.topLeftCorner<6, 6>();
  const Eigen::MatrixXd expected =
    s_bb - s_ba * J_full.transpose() - J_full * s_ba.transpose() + J_full * s_aa * J_full.transpose();

  const Mat21 actual = RelativeToAnchorPose(anchor, phi, noise, J);
  sio_test::ExpectMatrixNear(actual, expected, 1e-10);
  sio_test::ExpectSymmetric(actual, 1e-14);
}

TEST(ImuIntegratorTest, AnchorRelativeCovarianceBoundsConditionedCovariance) {
  const Mat21 noise = DeterministicSpd(1.1, 1e-4);
  const Mat21 phi = DeterministicTransition(2.0);
  const Eigen::Matrix<double, 6, 6> J = AnchorReferenceJacobian(
    Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero(),
    sio_utils::ExpSO3(Eigen::Vector3d(0.0, 0.1, 0.2)), Eigen::Vector3d(0.3, -0.2, 0.1));

  // 全方差公式：P_rel = P_b|a + Var(E[e | 锚点位姿]) ≥ P_b|a；锚点位姿与其余状态相关时严格更大
  const Mat21 correlated = DeterministicSpd(0.3, 1e-3);
  const Mat21 difference =
    RelativeToAnchorPose(correlated, phi, noise, J) - ConditionOnAnchorPose(correlated, phi, noise);
  const Eigen::SelfAdjointEigenSolver<Mat21> eig(0.5 * (difference + difference.transpose()));
  EXPECT_GT(eig.eigenvalues().minCoeff(), -1e-12);
  EXPECT_GT(difference.trace(), 1e-6);

  // 锚点位姿与其余状态不相关、且参照恰好带走 Φ 中锚点位姿的全部影响时，两者相同
  Mat21 independent = correlated;
  independent.block<6, DIM_STATE - 6>(0, 6).setZero();
  independent.block<DIM_STATE - 6, 6>(6, 0).setZero();
  Mat21 pose_only = phi;
  pose_only.block<DIM_STATE - 6, 6>(6, 0).setZero();
  sio_test::ExpectMatrixNear(
    RelativeToAnchorPose(independent, pose_only, noise, pose_only.topLeftCorner<6, 6>()),
    ConditionOnAnchorPose(independent, pose_only, noise), 1e-12);
}

TEST(ImuIntegratorTest, GyroScaleUncertaintyGrowsWithRotationAngle) {
  Mat21 cov = Mat21::Zero();
  const Eigen::Matrix3d R_current =
    Eigen::AngleAxisd(0.2, Eigen::Vector3d::UnitY()).toRotationMatrix();
  AddGyroScaleUncertainty(cov, Eigen::Matrix3d::Identity(), R_current, 0.015);
  const double expected = std::pow(0.015 * 0.2, 2);
  sio_test::ExpectMatrixNear(cov.block<3, 3>(0, 0), Eigen::Matrix3d::Identity() * expected, 1e-15);
  EXPECT_DOUBLE_EQ((cov.block<18, 18>(3, 3).norm()), 0.0);

  Mat21 unchanged = Mat21::Zero();
  AddGyroScaleUncertainty(unchanged, Eigen::Matrix3d::Identity(), Eigen::Matrix3d::Identity(), 0.015);
  AddGyroScaleUncertainty(unchanged, Eigen::Matrix3d::Identity(), R_current, 0.0);
  EXPECT_DOUBLE_EQ(unchanged.norm(), 0.0);
}

TEST(ImuIntegratorTest, NoiseParamsDriveRotationVariance) {
  for (const double sigma : {1e-3, 2e-3}) {
    ImuIntegrator integrator;
    ImuNoiseParams noise;
    noise.gyro_noise_density = sigma;
    noise.gyro_random_walk = 1e-12; // 去掉零偏随机游走对姿态方差的贡献
    integrator.SetNoiseParams(noise);
    KeyFrameData keyframe;
    keyframe.timestamp = 0.0;
    keyframe.imu_states.gravity = Eigen::Vector3d(0.0, 0.0, -9.81);
    keyframe.imu_states.cov.setZero();
    const auto imu = sio_test::MakeStaticImu(0.0, 101, 100.0);
    const ImuStatesGroup state = integrator.ImuPreintegrationCov(keyframe, imu, 1.0);
    // 静止、零初始协方差：姿态方差为陀螺白噪声的积分 σ² T
    EXPECT_NEAR(state.cov(0, 0), sigma * sigma, 1e-3 * sigma * sigma);
    EXPECT_NEAR(state.cov(2, 2), sigma * sigma, 1e-3 * sigma * sigma);
  }
}

TEST(ImuIntegratorTest, InitHandlesUpsideDownStaticSensor) {
  ImuIntegrator integrator;
  const Eigen::Vector3d upside_down(0.0, 0.0, -9.81);
  const auto imu = sio_test::MakeStaticImu(1.0, 100, 100.0, upside_down);
  ImuInitReport report;
  const ImuStatesGroup state = integrator.ImuInit(imu, &report);
  // 比力方向必须映到世界系 +z，重力补偿后加速度为零
  sio_test::ExpectMatrixNear(state.rot_end * upside_down, Eigen::Vector3d(0.0, 0.0, 9.81), 1e-9);
  sio_test::ExpectMatrixNear(state.rot_end * upside_down + state.gravity, Eigen::Vector3d::Zero(),
                             1e-9);
  EXPECT_TRUE(report.static_ok);
  EXPECT_EQ(report.samples, 100);
}

TEST(ImuIntegratorTest, InitReportFlagsRotatingWindow) {
  ImuIntegrator integrator;
  std::deque<sensor_msgs::msg::Imu::SharedPtr> imu;
  for (int i = 0; i < 100; ++i) {
    const double rate = (i % 2 == 0) ? 0.2 : -0.2;
    imu.push_back(sio_test::MakeImu(1.0 + 0.01 * i, Eigen::Vector3d(0.0, 0.0, 9.81),
                                    Eigen::Vector3d(rate, 0.0, 0.0)));
  }
  ImuInitReport report;
  integrator.ImuInit(imu, &report);
  EXPECT_FALSE(report.static_ok);
  EXPECT_NEAR(report.gyro_max_deviation_rad_s, 0.2, 1e-9);
}

TEST(ImuIntegratorTest, InitObservabilityCovarianceSeparatesTiltAndYaw) {
  ImuStatesGroup state;
  state.rot_end = (Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitZ()) *
                   Eigen::AngleAxisd(0.4, Eigen::Vector3d::UnitX())).toRotationMatrix();
  ImuInitReport report;
  report.samples = 400;
  report.gravity_norm = 9.81;
  report.gyro_std_rad_s = Eigen::Vector3d::Constant(1e-3);
  report.accel_std = Eigen::Vector3d::Constant(0.02);
  const InitialCovarianceParams params;
  const Mat21 cov = ImuIntegrator::ObservabilityInitialCovariance(state, report, params);

  const double accel_mean_noise = report.accel_std.norm() / std::sqrt(3.0 * report.samples);
  const double tilt_var = (params.accel_bias_sigma * params.accel_bias_sigma
                           + accel_mean_noise * accel_mean_noise) / (9.81 * 9.81);
  // 机体系姿态协方差转回世界系后：x、y 为倾斜，z 为航向
  const Eigen::Matrix3d world =
    state.rot_end * cov.block<3, 3>(0, 0) * state.rot_end.transpose();
  const Eigen::Vector3d expected_diag(tilt_var, tilt_var,
                                      params.yaw_sigma_rad * params.yaw_sigma_rad);
  sio_test::ExpectMatrixNear(world, Eigen::Matrix3d(expected_diag.asDiagonal()), 1e-12);
  sio_test::ExpectMatrixNear(cov.block<3, 3>(3, 3),
    Eigen::Matrix3d::Identity() * params.position_sigma_m * params.position_sigma_m, 1e-15);
  sio_test::ExpectMatrixNear(cov.block<3, 3>(6, 6),
    Eigen::Matrix3d::Identity() * params.velocity_sigma_mps * params.velocity_sigma_mps, 1e-15);
  const double bias_var = 1e-6 / 400.0 + params.gyro_bias_floor_rad_s * params.gyro_bias_floor_rad_s;
  EXPECT_NEAR(cov(9, 9), bias_var, 1e-15);
  sio_test::ExpectSymmetric(cov, 1e-15);
  EXPECT_GT(Eigen::SelfAdjointEigenSolver<Mat21>(cov).eigenvalues().minCoeff(), 0.0);
}

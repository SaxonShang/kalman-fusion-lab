#include "helpers/assert_eigen.h"
#include "helpers/make_cloud.h"

#include "sio_frontend_modules.h"
#include "utils.h"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <utility>
#include <vector>
#include <pcl/kdtree/kdtree_flann.h>

TEST(StateEstimatorTest, BuildsFinitePointObservationFactor) {
  StateEstimator estimator;
  StateEstimator::PointObservationFactor factor;
  const PointTypeRGB source = sio_test::MakePoint(0.01f, 0.0f, 1.0f, 80, 90, 100);
  const PointTypeRGB target = sio_test::MakePoint(0.0f, 0.0f, 1.0f, 82, 91, 101);

  const bool ok = estimator.BuildPointObservationFactor(
    source, target, Eigen::Vector3d(0.2, 0.1, 0.0), Eigen::Vector3d::UnitZ(),
    Eigen::Vector3d::Zero(), Eigen::Matrix3d::Identity(), 0.9, 0.8, factor);

  EXPECT_TRUE(ok);
  EXPECT_TRUE(factor.has_stereo_geo);
  EXPECT_TRUE(factor.r_geo_vec.allFinite());
  EXPECT_TRUE(factor.J_geo_mat.allFinite());
  EXPECT_TRUE(factor.W_geo.allFinite());
  EXPECT_TRUE(std::isfinite(factor.r_geo));
  EXPECT_TRUE(std::isfinite(factor.r_color));
  sio_test::ExpectMatrixFinite(factor.J_color);
  EXPECT_GT(factor.inv_geo_noise, 0.0);
  EXPECT_GT(factor.inv_color_noise, 0.0);
}

TEST(StateEstimatorTest, RejectsInvalidObservationFactors) {
  StateEstimator estimator;
  StateEstimator::PointObservationFactor factor;
  PointTypeRGB source = sio_test::MakePoint(0.0f, 0.0f, 1.0f);
  const PointTypeRGB target = sio_test::MakePoint(0.0f, 0.0f, 1.0f);

  source.x = std::numeric_limits<float>::quiet_NaN();
  EXPECT_FALSE(estimator.BuildPointObservationFactor(
    source, target, Eigen::Vector3d::UnitX(), Eigen::Vector3d::UnitZ(), Eigen::Vector3d::Zero(),
    Eigen::Matrix3d::Identity(), 1.0, 1.0, factor));

  // 零法向：几何项仍可建立，光度项跳过
  source = sio_test::MakePoint(0.0f, 0.0f, 1.0f);
  EXPECT_TRUE(estimator.BuildPointObservationFactor(
    source, target, Eigen::Vector3d::UnitX(), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
    Eigen::Matrix3d::Identity(), 1.0, 1.0, factor));
  EXPECT_TRUE(factor.has_stereo_geo);
  EXPECT_EQ(factor.inv_color_noise, 0.0);
}

TEST(StateEstimatorTest, BuildObservationSubcloudsFiltersDistanceAndNormals) {
  StateEstimator estimator;
  auto target = pcl::PointCloud<PointTypeRGB>::Ptr(new pcl::PointCloud<PointTypeRGB>());
  target->push_back(sio_test::MakePoint(0.0f, 0.0f, 1.0f));
  target->push_back(sio_test::MakePoint(1.0f, 0.0f, 1.0f));
  target->width = target->size();
  target->height = 1;
  auto source = pcl::PointCloud<PointTypeRGB>::Ptr(new pcl::PointCloud<PointTypeRGB>());
  source->push_back(sio_test::MakePoint(0.01f, 0.0f, 1.0f));
  source->push_back(sio_test::MakePoint(1.01f, 0.0f, 1.0f));
  source->width = source->size();
  source->height = 1;

  auto kdtree = pcl::KdTreeFLANN<PointTypeRGB>::Ptr(new pcl::KdTreeFLANN<PointTypeRGB>());
  kdtree->setInputCloud(target);
  const std::vector<Eigen::Vector3d> gradients(2, Eigen::Vector3d::UnitX());
  const std::vector<Eigen::Vector3d> normals = {Eigen::Vector3d::UnitZ(), Eigen::Vector3d::Zero()};
  pcl::PointCloud<PointTypeRGB>::Ptr source_corr;
  pcl::PointCloud<PointTypeRGB>::Ptr target_corr;
  std::vector<Eigen::Vector3d> corr_gradients;
  std::vector<Eigen::Vector3d> corr_normals;
  const std::vector<float> source_confs = {0.9f, 0.8f};
  const std::vector<float> target_confs = {0.95f, 1.0f};
  std::vector<float> source_corr_conf;
  std::vector<float> target_corr_conf;

  estimator.BuildObservationSubclouds(source, kdtree, gradients, normals, source_confs,
                                        target_confs, source_corr, target_corr, corr_gradients,
                                        corr_normals, source_corr_conf, target_corr_conf, 0.01f);

  ASSERT_TRUE(source_corr);
  ASSERT_TRUE(target_corr);
  EXPECT_EQ(source_corr->size(), 1u);
  EXPECT_EQ(target_corr->size(), 1u);
  ASSERT_EQ(corr_normals.size(), 1u);
  sio_test::ExpectMatrixNear(corr_normals.front(), Eigen::Vector3d::UnitZ(), 1e-12);
  ASSERT_EQ(source_corr_conf.size(), 1u);
  EXPECT_NEAR(source_corr_conf.front(), 0.9f, 1e-6f);
}

TEST(StateEstimatorTest, EskfProducesFiniteSymmetricCovariance) {
  StateEstimator estimator;
  ImuStatesGroup propagated;
  propagated.cov = Eigen::Matrix<double, DIM_STATE, DIM_STATE>::Identity() * 0.1;
  propagated.gravity = Eigen::Vector3d(0.0, 0.0, -9.81);
  Eigen::Matrix4f observed = Eigen::Matrix4f::Identity();
  observed(0, 3) = 0.2f;
  const Eigen::Matrix<double, 6, 6> R_obs =
    Eigen::Matrix<double, 6, 6>::Identity() * 0.01;

  const ImuStatesGroup fused = estimator.ESKF(propagated, observed, R_obs);

  sio_test::ExpectMatrixFinite(fused.rot_end);
  sio_test::ExpectMatrixFinite(fused.pos_end);
  sio_test::ExpectMatrixFinite(fused.vel_end);
  sio_test::ExpectMatrixFinite(fused.cov);
  sio_test::ExpectSymmetric(fused.cov, 1e-9);
}

TEST(StateEstimatorTest, EskfAngularGainIsIndependentOfUncorrelatedPositionVariance) {
  StateEstimator estimator;
  ImuStatesGroup local, uncertain_position;
  local.cov.setIdentity(); local.cov *= 1e-6;
  local.cov.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() * 1e-2;
  uncertain_position = local;
  uncertain_position.cov.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() * 1e12;
  Eigen::Matrix4f observation = Eigen::Matrix4f::Identity();
  observation.block<3, 3>(0, 0) =
    svio_utils::ExpSO3(Eigen::Vector3d(.01, -.005, .002)).cast<float>();
  Eigen::Matrix<double, 6, 6> noise = Eigen::Matrix<double, 6, 6>::Identity() * 1e-8;
  noise.bottomRightCorner<3, 3>() *= 1e4;
  ESKFDebugSummary a, b;
  const auto first = estimator.ESKF(local, observation, noise, nullptr, &a);
  const auto second = estimator.ESKF(uncertain_position, observation, noise, nullptr, &b);
  const double expected = 1e-6 / (1e-6 + 1e-8);
  EXPECT_NEAR(a.gain(0, 0), expected, 1e-12);
  EXPECT_NEAR(b.gain(0, 0), expected, 1e-12);
  sio_test::ExpectMatrixNear(first.rot_end, second.rot_end, 1e-12);
  sio_test::ExpectSymmetric(second.cov, 1e-12);
}

TEST(StateEstimatorTest, EskfHandlesCorrelatedInnovationAtFloatingPointResolution) {
  StateEstimator estimator;
  ImuStatesGroup prior;
  prior.cov.setIdentity();
  prior.cov(0, 1) = prior.cov(1, 0) = 1.0;
  // R is positive, but adding it to the unit-scale rank-one block loses
  // the 1e-20 diagonal increment in double precision.
  const Eigen::Matrix<double, 6, 6> noise = Eigen::Matrix<double, 6, 6>::Identity() * 1e-20;
  Eigen::Matrix4f observation = Eigen::Matrix4f::Identity();
  observation.block<3, 3>(0, 0) =
    svio_utils::ExpSO3(Eigen::Vector3d(.01, -.01, 0)).cast<float>();
  ESKFDebugSummary debug;
  const auto fused = estimator.ESKF(prior, observation, noise, nullptr, &debug);
  EXPECT_GT(debug.damping, 0.0);
  EXPECT_LT(debug.damping, 1e-6);
  EXPECT_TRUE(debug.gain.allFinite());
  EXPECT_TRUE(std::isfinite(debug.nis_regularized));
  sio_test::ExpectSymmetric(fused.cov, 1e-12);
  EXPECT_GT(fused.cov.selfadjointView<Eigen::Lower>().eigenvalues().minCoeff(), -1e-12);
}

TEST(StateEstimatorTest, EskfGainRespondsToObservationUncertainty) {
  StateEstimator estimator;
  ImuStatesGroup prior;
  prior.cov.setIdentity();
  prior.cov *= 0.1;
  Eigen::Matrix4f observed = Eigen::Matrix4f::Identity();
  observed.block<3, 3>(0, 0) = svio_utils::ExpSO3(Eigen::Vector3d(0.02, -0.01, 0.03)).cast<float>();
  observed(0, 3) = 0.2f;
  const auto original_cov = prior.cov;
  ESKFDebugSummary low_debug, high_debug;
  const auto low = estimator.ESKF(prior, observed,
    Eigen::Matrix<double, 6, 6>::Identity() * 1e-4, nullptr, &low_debug);
  const auto high = estimator.ESKF(prior, observed,
    Eigen::Matrix<double, 6, 6>::Identity() * 1.0, nullptr, &high_debug);
  EXPECT_GT(low.pos_end.x(), high.pos_end.x());
  EXPECT_GT(low_debug.gain.topRows<6>().trace(), high_debug.gain.topRows<6>().trace());
  EXPECT_GT(low_debug.nis_regularized, high_debug.nis_regularized);
  EXPECT_LT(low.cov.trace(), high.cov.trace());
  sio_test::ExpectMatrixNear(prior.cov, original_cov, 0.0);
  const auto without_debug = estimator.ESKF(prior, observed,
    Eigen::Matrix<double, 6, 6>::Identity() * 1e-4);
  sio_test::ExpectMatrixNear(without_debug.pos_end, low.pos_end, 0.0);
  sio_test::ExpectMatrixNear(without_debug.cov, low.cov, 0.0);
}

TEST(StateEstimatorTest, EsikfUpdatesGeometryWithoutGicpPoseObservation) {
  StateEstimator estimator;
  const auto target = sio_test::MakeGridCloud(20, 20);
  auto tree = std::make_shared<pcl::KdTreeFLANN<PointTypeRGB>>();
  tree->setInputCloud(target);
  Eigen::Matrix4f offset = Eigen::Matrix4f::Identity();
  offset(2, 3) = -0.02f;
  const auto source = sio_test::TransformCloud(target, offset);
  std::vector<Eigen::Vector3d> normals(target->size(), Eigen::Vector3d::UnitZ());
  std::vector<Eigen::Vector3d> gradients(target->size(), Eigen::Vector3d::Zero());
  ImuStatesGroup state;
  ESIKFDebugSummary debug;
  estimator.ESIKF(source, tree, gradients, normals, state, Eigen::Matrix3d::Identity(),
    Eigen::Vector3d::Zero(), {}, {}, 1.0, nullptr, Eigen::Matrix4d::Identity(), &debug);
  ASSERT_GT(debug.num_iters, 0);
  ASSERT_FALSE(debug.iterations.empty());
  EXPECT_EQ(debug.iterations.size(), static_cast<std::size_t>(debug.num_iters));
  EXPECT_GT(state.pos_end.z(), 0.005);
  EXPECT_NEAR(state.pos_end.z(), 0.02, 0.015);
  EXPECT_GE(debug.final_corr, 20u);
  sio_test::ExpectMatrixFinite(debug.iterations.back().information);
  sio_test::ExpectMatrixFinite(debug.iterations.back().observation_covariance);
  sio_test::ExpectSymmetric(state.cov, 1e-9);
}

TEST(StateEstimatorTest, EskfUsesAxisSpecificObservationUncertainty) {
  StateEstimator estimator;
  ImuStatesGroup prior;
  prior.cov.setIdentity();
  prior.cov *= 0.01;
  Eigen::Matrix4f observed = Eigen::Matrix4f::Identity();
  observed(0, 3) = observed(1, 3) = 0.1f;
  Eigen::Matrix<double, 6, 6> noise = Eigen::Matrix<double, 6, 6>::Identity();
  noise(3, 3) = 1e-6;
  const auto fused = estimator.ESKF(prior, observed, noise);
  EXPECT_GT(fused.pos_end.x(), 0.09);
  EXPECT_LT(fused.pos_end.y(), 0.002);
  sio_test::ExpectSymmetric(fused.cov, 1e-12);
  EXPECT_GT(fused.cov.selfadjointView<Eigen::Lower>().eigenvalues().minCoeff(), 0.0);
}

TEST(StateEstimatorTest, EskfJosephOnParallelUsesMainGain) {
  StateEstimator estimator;
  ImuStatesGroup propagated;
  propagated.cov.setIdentity();
  propagated.cov *= 0.1;
  propagated.gravity = Eigen::Vector3d(0.0, 0.0, -9.81);
  Eigen::Matrix4f observed = Eigen::Matrix4f::Identity();
  observed(0, 3) = 0.2f;
  const Eigen::Matrix<double, 6, 6> R_obs =
      Eigen::Matrix<double, 6, 6>::Identity() * 0.01;

  Eigen::Matrix<double, DIM_STATE, DIM_STATE> parallel =
      Eigen::Matrix<double, DIM_STATE, DIM_STATE>::Zero();
  const auto parallel_before = parallel;
  const auto main_before = propagated.cov;

  const ImuStatesGroup fused = estimator.ESKF(propagated, observed, R_obs, &parallel);

  sio_test::ExpectMatrixNear(propagated.cov, main_before, 1e-12);

  Eigen::Matrix<double, 6, DIM_STATE> H = Eigen::Matrix<double, 6, DIM_STATE>::Zero();
  H.block<3, 3>(0, 0).setIdentity();
  H.block<3, 3>(3, 3).setIdentity();
  const Eigen::Matrix<double, 6, 6> S = H * main_before * H.transpose() + R_obs;
  const Eigen::Matrix<double, 6, 6> S_inv =
      S.ldlt().solve(Eigen::Matrix<double, 6, 6>::Identity());
  const Eigen::Matrix<double, DIM_STATE, 6> K = main_before * H.transpose() * S_inv;
  const Eigen::Matrix<double, DIM_STATE, DIM_STATE> I_KH =
      Eigen::Matrix<double, DIM_STATE, DIM_STATE>::Identity() - K * H;
  const Eigen::Matrix<double, DIM_STATE, DIM_STATE> expected =
      I_KH * parallel_before * I_KH.transpose() + K * R_obs * K.transpose();

  sio_test::ExpectMatrixNear(parallel, expected, 1e-9);
  sio_test::ExpectMatrixFinite(fused.cov);
  EXPECT_GT((parallel - fused.cov).norm(), 1e-6);
}

namespace {
// 位置、速度相关的先验；其余块为默认量级。
ImuStatesGroup MakeCorrelatedPrior() {
  ImuStatesGroup prior;
  prior.cov = Eigen::Matrix<double, DIM_STATE, DIM_STATE>::Identity() * 1e-2;
  prior.cov.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() * 4e-4;
  prior.cov.block<3, 3>(6, 6) = Eigen::Matrix3d::Identity() * 1e-3;
  prior.cov.block<3, 3>(3, 6) = Eigen::Matrix3d::Identity() * 2e-4;
  prior.cov.block<3, 3>(6, 3) = Eigen::Matrix3d::Identity() * 2e-4;
  prior.pos_end = Eigen::Vector3d(0.1, -0.2, 0.3);
  prior.vel_end = Eigen::Vector3d(0.5, 0.0, 0.0);
  return prior;
}

// 线性情形的虚拟观测：从线性化点 x_j 直接走到观测位姿 (R_z, p_z) 的高斯牛顿步。
Eigen::Matrix<double, 6, 1> StepToPose(const ImuStatesGroup &x, const Eigen::Matrix3d &R_z,
                                       const Eigen::Vector3d &p_z) {
  Eigen::Matrix<double, 6, 1> r;
  r.head<3>() = sio_utils::LogSO3(x.rot_end.transpose() * R_z);
  r.tail<3>() = p_z - x.pos_end;
  return r;
}

// 一次闭式卡尔曼更新 d = K (z ⊟ x_pred) 后的位置与速度。
std::pair<Eigen::Vector3d, Eigen::Vector3d> ClosedFormPositionVelocity(
  const ImuStatesGroup &prior, const Eigen::Vector3d &p_z, const Eigen::Matrix<double, 6, 6> &R) {
  Eigen::Matrix<double, 6, DIM_STATE> H = Eigen::Matrix<double, 6, DIM_STATE>::Zero();
  H.block<3, 3>(0, 0).setIdentity();
  H.block<3, 3>(3, 3).setIdentity();
  const Eigen::Matrix<double, 6, 6> S = H * prior.cov * H.transpose() + R;
  const Eigen::Matrix<double, DIM_STATE, 6> K = prior.cov * H.transpose() * S.inverse();
  Eigen::Matrix<double, 6, 1> innovation = Eigen::Matrix<double, 6, 1>::Zero();
  innovation.tail<3>() = p_z - prior.pos_end;
  const Eigen::Matrix<double, DIM_STATE, 1> d = K * innovation;
  return {prior.pos_end + d.segment<3>(3), prior.vel_end + d.segment<3>(6)};
}
} // namespace

TEST(StateEstimatorTest, EsikfPriorResidualStepMatchesClosedFormUpdate) {
  const ImuStatesGroup prior = MakeCorrelatedPrior();
  const Eigen::Vector3d p_z = prior.pos_end + Eigen::Vector3d(0.02, -0.01, 0.005);
  const Eigen::Matrix<double, 6, 6> R = Eigen::Matrix<double, 6, 6>::Identity() * 1e-4;
  const auto expected = ClosedFormPositionVelocity(prior, p_z, R);

  ImuStatesGroup x = prior;
  for (int j = 0; j < 5; ++j) {
    const EsikfStepResult step =
      EsikfIterationStep(prior, x, StepToPose(x, prior.rot_end, p_z), R, true, false);
    ASSERT_TRUE(step.ok);
    x = step.state;
  }
  // 线性观测下重复迭代停在闭式解，不向纯观测漂移；速度经交叉协方差一并修正。
  sio_test::ExpectMatrixNear(x.pos_end, expected.first, 1e-12);
  sio_test::ExpectMatrixNear(x.vel_end, expected.second, 1e-12);
  sio_test::ExpectMatrixNear(x.rot_end, prior.rot_end, 1e-12);
}

TEST(StateEstimatorTest, EsikfLegacyStepDriftsToObservation) {
  const ImuStatesGroup prior = MakeCorrelatedPrior();
  const Eigen::Vector3d p_z = prior.pos_end + Eigen::Vector3d(0.02, -0.01, 0.005);
  const Eigen::Matrix<double, 6, 6> R = Eigen::Matrix<double, 6, 6>::Identity() * 1e-4;
  const auto expected = ClosedFormPositionVelocity(prior, p_z, R);

  ImuStatesGroup x = prior;
  const EsikfStepResult first =
    EsikfIterationStep(prior, x, StepToPose(x, prior.rot_end, p_z), R, false, false);
  ASSERT_TRUE(first.ok);
  // 第一轮与闭式解相同
  sio_test::ExpectMatrixNear(first.state.pos_end, expected.first, 1e-12);
  for (int j = 0; j < 60; ++j) {
    const EsikfStepResult step =
      EsikfIterationStep(prior, x, StepToPose(x, prior.rot_end, p_z), R, false, false);
    ASSERT_TRUE(step.ok);
    x = step.state;
  }
  // 原实现重复注入 K r_v，最终位置等于观测，先验被丢掉
  sio_test::ExpectMatrixNear(x.pos_end, p_z, 1e-6);
  EXPECT_GT((x.pos_end - expected.first).norm(), 1e-3);
}

TEST(StateEstimatorTest, EsikfObservationOnlyStepTakesVirtualPose) {
  const ImuStatesGroup prior = MakeCorrelatedPrior();
  Eigen::Matrix<double, 6, 1> r_v;
  r_v << 0.01, -0.02, 0.005, 0.03, 0.0, -0.01;
  const EsikfStepResult step = EsikfIterationStep(
    prior, prior, r_v, Eigen::Matrix<double, 6, 6>::Identity() * 1e-4, false, true);
  ASSERT_TRUE(step.ok);
  sio_test::ExpectMatrixNear(step.gain.topRows<6>(), Eigen::Matrix<double, 6, 6>::Identity(), 0.0);
  sio_test::ExpectMatrixNear(step.state.pos_end, prior.pos_end + r_v.tail<3>(), 1e-15);
  sio_test::ExpectMatrixNear(step.state.vel_end, prior.vel_end, 0.0);
  sio_test::ExpectMatrixNear(sio_utils::LogSO3(prior.rot_end.transpose() * step.state.rot_end),
                             r_v.head<3>(), 1e-12);
}

TEST(StateEstimatorTest, EsikfFrozenRotationKeepsPriorRotationOnly) {
  ImuStatesGroup prior = MakeCorrelatedPrior();
  // 姿态与位置相关：只量位置也会经 P_θp 改动姿态，冻结旋转时不应改动
  prior.cov.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity() * 1e-4;
  prior.cov.block<3, 3>(3, 0) = Eigen::Matrix3d::Identity() * 1e-4;
  Eigen::Matrix<double, 6, 1> r_v;
  r_v << 0.0, 0.0, 0.0, 0.02, -0.01, 0.005;
  const Eigen::Matrix<double, 6, 6> R = Eigen::Matrix<double, 6, 6>::Identity() * 1e-4;

  const EsikfStepResult full = EsikfIterationStep(prior, prior, r_v, R, true, false);
  const EsikfStepResult frozen = EsikfIterationStep(prior, prior, r_v, R, true, false, false);
  ASSERT_TRUE(full.ok);
  ASSERT_TRUE(frozen.ok);
  EXPECT_GT(sio_utils::LogSO3(prior.rot_end.transpose() * full.state.rot_end).norm(), 1e-6);
  sio_test::ExpectMatrixNear(frozen.state.rot_end, prior.rot_end, 0.0);
  sio_test::ExpectMatrixNear(frozen.gain.topRows<3>(), Eigen::Matrix<double, 3, 6>::Zero(), 0.0);
  // 其余行仍是最优增益，位置、速度与正常更新相同
  sio_test::ExpectMatrixNear(frozen.gain.bottomRows<DIM_STATE - 3>(),
                             full.gain.bottomRows<DIM_STATE - 3>(), 0.0);
  sio_test::ExpectMatrixNear(frozen.state.pos_end, full.state.pos_end, 1e-15);
  sio_test::ExpectMatrixNear(frozen.state.vel_end, full.state.vel_end, 1e-15);
}

TEST(StateEstimatorTest, EsikfKeepsPriorWhenNoCorrespondences) {
  StateEstimator estimator;
  const auto target = sio_test::MakeGridCloud(20, 20);
  auto tree = std::make_shared<pcl::KdTreeFLANN<PointTypeRGB>>();
  tree->setInputCloud(target);
  Eigen::Matrix4f far = Eigen::Matrix4f::Identity();
  far(2, 3) = 5.0f;
  const auto source = sio_test::TransformCloud(target, far);
  std::vector<Eigen::Vector3d> normals(target->size(), Eigen::Vector3d::UnitZ());
  std::vector<Eigen::Vector3d> gradients(target->size(), Eigen::Vector3d::Zero());
  ImuStatesGroup state;
  state.pos_end = Eigen::Vector3d(0.1, 0.2, 0.3);
  const ImuStatesGroup before = state;
  ESIKFDebugSummary debug;
  estimator.ESIKF(source, tree, gradients, normals, state, Eigen::Matrix3d::Identity(),
    Eigen::Vector3d::Zero(), {}, {}, 1.0, nullptr, Eigen::Matrix4d::Identity(), &debug);
  EXPECT_FALSE(debug.updated);
  EXPECT_EQ(debug.status, "too_few_correspondences");
  sio_test::ExpectMatrixNear(state.pos_end, before.pos_end, 0.0);
  sio_test::ExpectMatrixNear(state.rot_end, before.rot_end, 0.0);
  sio_test::ExpectMatrixNear(state.cov, before.cov, 0.0);
}

TEST(StateEstimatorTest, EsikfObservationScaleInflatesVirtualCovariance) {
  const auto target = sio_test::MakeGridCloud(20, 20);
  auto tree = std::make_shared<pcl::KdTreeFLANN<PointTypeRGB>>();
  tree->setInputCloud(target);
  Eigen::Matrix4f offset = Eigen::Matrix4f::Identity();
  offset(2, 3) = -0.02f;
  const auto source = sio_test::TransformCloud(target, offset);
  std::vector<Eigen::Vector3d> normals(target->size(), Eigen::Vector3d::UnitZ());
  std::vector<Eigen::Vector3d> gradients(target->size(), Eigen::Vector3d::Zero());

  auto first_observation = [&](double scale_rot, double scale_trans) {
    StateEstimator estimator;
    ESIKFOptions options;
    options.observation_scale_rotation = scale_rot;
    options.observation_scale_translation = scale_trans;
    estimator.SetESIKFOptions(options);
    ImuStatesGroup state;
    ESIKFDebugSummary debug;
    estimator.ESIKF(source, tree, gradients, normals, state, Eigen::Matrix3d::Identity(),
      Eigen::Vector3d::Zero(), {}, {}, 1.0, nullptr, Eigen::Matrix4d::Identity(), &debug);
    EXPECT_FALSE(debug.iterations.empty());
    return debug.iterations.empty() ? Eigen::Matrix<double, 6, 6>::Zero().eval()
                                    : debug.iterations.front().observation_covariance;
  };
  const Eigen::Matrix<double, 6, 6> base = first_observation(1.0, 1.0);
  const Eigen::Matrix<double, 6, 6> scaled = first_observation(4.0, 9.0);
  Eigen::Matrix<double, 6, 1> d;
  d << 2, 2, 2, 3, 3, 3;
  const Eigen::Matrix<double, 6, 6> expected = d.asDiagonal() * base * d.asDiagonal();
  EXPECT_LT((scaled - expected).norm(), 1e-9 * expected.norm());
}

TEST(StateEstimatorTest, EsikfPriorResidualKeepsPriorWeight) {
  const auto target = sio_test::MakeGridCloud(20, 20);
  auto tree = std::make_shared<pcl::KdTreeFLANN<PointTypeRGB>>();
  tree->setInputCloud(target);
  Eigen::Matrix4f offset = Eigen::Matrix4f::Identity();
  offset(2, 3) = -0.02f;
  const auto source = sio_test::TransformCloud(target, offset);
  std::vector<Eigen::Vector3d> normals(target->size(), Eigen::Vector3d::UnitZ());
  std::vector<Eigen::Vector3d> gradients(target->size(), Eigen::Vector3d::Zero());

  // 旋转几乎已知，位置先验方差与点云给出的位置观测方差同量级（K≈0.5）。
  auto run = [&](bool prior_residual) {
    StateEstimator estimator;
    ESIKFOptions options;
    options.prior_residual = prior_residual;
    estimator.SetESIKFOptions(options);
    ImuStatesGroup state;
    state.cov.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * 1e-12;
    state.cov.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() * 5e-7;
    ESIKFDebugSummary debug;
    estimator.ESIKF(source, tree, gradients, normals, state, Eigen::Matrix3d::Identity(),
      Eigen::Vector3d::Zero(), {}, {}, 1.0, nullptr, Eigen::Matrix4d::Identity(), &debug);
    EXPECT_TRUE(debug.updated);
    return state.pos_end.z();
  };
  const double with_prior = run(true);
  const double legacy = run(false);
  // 观测要求移动约 0.02 m；保留先验时停在两者之间，原实现逐轮推向观测。
  EXPECT_GT(with_prior, 0.006);
  EXPECT_LT(with_prior, 0.014);
  EXPECT_GT(legacy, 0.016);
}

TEST(StateEstimatorTest, ObservationFactorRayResidualIgnoresLateralError) {
  StateEstimator estimator;
  ESIKFOptions options;
  options.geometry_residual = ESIKFOptions::GeometryResidual::Ray;
  estimator.SetESIKFOptions(options);
  estimator.SetRayOriginInTarget(Eigen::Vector3d::Zero());
  const PointTypeRGB target = sio_test::MakePoint(0.0f, 0.0f, 1.0f);
  StateEstimator::PointObservationFactor lateral, along;
  ASSERT_TRUE(estimator.BuildPointObservationFactor(
    sio_test::MakePoint(0.01f, 0.0f, 1.0f), target, Eigen::Vector3d::Zero(),
    Eigen::Vector3d::UnitZ(), Eigen::Vector3d::Zero(), Eigen::Matrix3d::Identity(), 1.0, 1.0,
    lateral));
  ASSERT_TRUE(estimator.BuildPointObservationFactor(
    sio_test::MakePoint(0.0f, 0.0f, 1.01f), target, Eigen::Vector3d::Zero(),
    Eigen::Vector3d::UnitZ(), Eigen::Vector3d::Zero(), Eigen::Matrix3d::Identity(), 1.0, 1.0,
    along));
  // 垂直于视线的误差不产生约束，信息矩阵为沿视线的秩一矩阵
  EXPECT_NEAR(lateral.r_geo_vec.dot(lateral.W_geo * lateral.r_geo_vec), 0.0, 1e-12);
  EXPECT_GT(along.r_geo_vec.dot(along.W_geo * along.r_geo_vec), 0.1);
  sio_test::ExpectMatrixNear(lateral.W_geo * Eigen::Vector3d::UnitX(), Eigen::Vector3d::Zero(), 1e-9);
}

TEST(StateEstimatorTest, ObservationFactorPointToPlaneUsesNormal) {
  StateEstimator estimator;
  ESIKFOptions options;
  options.geometry_residual = ESIKFOptions::GeometryResidual::PointToPlane;
  estimator.SetESIKFOptions(options);
  const PointTypeRGB target = sio_test::MakePoint(0.0f, 0.0f, 1.0f);
  StateEstimator::PointObservationFactor in_plane, normal_offset, no_normal;
  ASSERT_TRUE(estimator.BuildPointObservationFactor(
    sio_test::MakePoint(0.0f, 0.0f, 1.01f), target, Eigen::Vector3d::Zero(),
    Eigen::Vector3d::UnitX(), Eigen::Vector3d::Zero(), Eigen::Matrix3d::Identity(), 1.0, 1.0,
    in_plane));
  ASSERT_TRUE(estimator.BuildPointObservationFactor(
    sio_test::MakePoint(0.01f, 0.0f, 1.0f), target, Eigen::Vector3d::Zero(),
    Eigen::Vector3d::UnitX(), Eigen::Vector3d::Zero(), Eigen::Matrix3d::Identity(), 1.0, 1.0,
    normal_offset));
  EXPECT_NEAR(in_plane.r_geo_vec.dot(in_plane.W_geo * in_plane.r_geo_vec), 0.0, 1e-12);
  EXPECT_GT(normal_offset.r_geo_vec.dot(normal_offset.W_geo * normal_offset.r_geo_vec), 0.1);
  EXPECT_FALSE(estimator.BuildPointObservationFactor(
    sio_test::MakePoint(0.0f, 0.0f, 1.01f), target, Eigen::Vector3d::Zero(),
    Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), Eigen::Matrix3d::Identity(), 1.0, 1.0,
    no_normal));
}

TEST(StateEstimatorTest, ObservationFactorUniformLuminanceMatchesGradient) {
  const PointTypeRGB source = sio_test::MakePoint(0.0f, 0.0f, 1.0f, 255, 0, 0);
  const PointTypeRGB target = sio_test::MakePoint(0.0f, 0.0f, 1.0f, 0, 0, 0);
  auto color_residual = [&](bool uniform) {
    StateEstimator estimator;
    ESIKFOptions options;
    options.uniform_luminance = uniform;
    estimator.SetESIKFOptions(options);
    StateEstimator::PointObservationFactor factor;
    EXPECT_TRUE(estimator.BuildPointObservationFactor(
      source, target, Eigen::Vector3d::Zero(), Eigen::Vector3d::UnitZ(), Eigen::Vector3d::Zero(),
      Eigen::Matrix3d::Identity(), 1.0, 1.0, factor));
    return factor.r_color;
  };
  EXPECT_NEAR(color_residual(false), -0.2126, 1e-9);
  EXPECT_NEAR(color_residual(true), -1.0 / 3.0, 1e-9);
}

TEST(StateEstimatorTest, ObservationAttributeGateDropsFarProperties) {
  CameraIntrinsics cam{100.0f, 100.0f, 50.0f, 50.0f, 0.12f, 100, 100};
  const cv::Mat depth(100, 100, CV_32FC1, cv::Scalar(1.0f));
  auto source = pcl::PointCloud<PointTypeRGB>::Ptr(new pcl::PointCloud<PointTypeRGB>());
  source->push_back(sio_test::MakePoint(0.0f, 0.0f, 1.0f));
  auto target = pcl::PointCloud<PointTypeRGB>::Ptr(new pcl::PointCloud<PointTypeRGB>());
  target->push_back(sio_test::MakePoint(2.0f, 2.0f, 3.0f));
  auto tree = std::make_shared<pcl::KdTreeFLANN<PointTypeRGB>>();
  tree->setInputCloud(target);
  const std::vector<Eigen::Vector3d> gradients{Eigen::Vector3d::UnitX()};
  const std::vector<Eigen::Vector3d> normals{Eigen::Vector3d::UnitZ()};

  auto matched_normal = [&](double gate) {
    StateEstimator estimator;
    ESIKFOptions options;
    options.attribute_max_distance_m = gate;
    estimator.SetESIKFOptions(options);
    pcl::PointCloud<PointTypeRGB>::Ptr source_corr, target_corr;
    std::vector<Eigen::Vector3d> corr_gradients, corr_normals;
    std::vector<float> source_conf, target_conf;
    estimator.BuildProjectiveObservationSubclouds(
      source, tree, gradients, normals, {}, {}, depth, cam, Eigen::Matrix4d::Identity(),
      source_corr, target_corr, corr_gradients, corr_normals, source_conf, target_conf);
    EXPECT_EQ(corr_normals.size(), 1u);
    return corr_normals.empty() ? Eigen::Vector3d::Zero().eval() : corr_normals.front();
  };
  // 原实现：几何目标在 (0,0,1)，仍借用 3.5 m 外稀疏点的法向
  EXPECT_NEAR(matched_normal(0.0).norm(), 1.0, 1e-9);
  // 距离门限：属性置零，光度项与点到面残差都不会使用它
  EXPECT_NEAR(matched_normal(0.05).norm(), 0.0, 0.0);
}

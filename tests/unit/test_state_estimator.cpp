#include "helpers/assert_eigen.h"
#include "helpers/make_cloud.h"

#include "sio_frontend_modules.h"
#include "utils.h"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
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

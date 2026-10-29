#include "helpers/assert_eigen.h"
#include "helpers/make_cloud.h"
#include "helpers/make_frame.h"
#include "helpers/make_imu.h"

#include "sio_frontend_modules.h"
#include "utils.h"

#include <Eigen/Dense>
#include <cmath>
#include <gtest/gtest.h>

namespace {

KeyframeManager MakeManager() {
  KeyframeManager::Config cfg;
  cfg.params_file_path.clear();
  cfg.front_trans_thresh_m = 0.4;
  cfg.front_angle_thresh_deg = 10.0;
  cfg.front_time_thresh_s = 0.6;
  cfg.backend_min_frontend_kfs = 3;
  cfg.backend_queue_legacy_trigger = 5;
  return KeyframeManager(cfg);
}

} // namespace

TEST(KeyframeManagerTest, FrontKeyframeTriggersOnMotionRotationOrTime) {
  const auto manager = MakeManager();
  ImuStatesGroup last;
  ImuStatesGroup current;

  current.pos_end = Eigen::Vector3d(0.41, 0.0, 0.0);
  EXPECT_TRUE(manager.IsFrontKeyFrame(current, last, 0.1));

  current = ImuStatesGroup();
  current.rot_end = svio_utils::ExpSO3(Eigen::Vector3d(0.0, 0.0, M_PI / 9.0));
  EXPECT_TRUE(manager.IsFrontKeyFrame(current, last, 0.1));

  current = ImuStatesGroup();
  EXPECT_TRUE(manager.IsFrontKeyFrame(current, last, 0.61));
  EXPECT_FALSE(manager.IsFrontKeyFrame(current, last, 0.1));
}

TEST(KeyframeManagerTest, BackendKeyframeLegacyQueueThresholdWorks) {
  const auto manager = MakeManager();

  EXPECT_FALSE(manager.IsBackKeyFrame(4));
  EXPECT_TRUE(manager.IsBackKeyFrame(5));
}

TEST(KeyframeManagerTest, BackendPromotionCopiesFrontendFrame) {
  auto manager = MakeManager();
  auto latest = std::make_shared<KeyFrameData>();
  latest->timestamp = 2.0;
  latest->keyframe_id = 7;
  latest->pointcloud = sio_test::MakeLineCloud(4);
  latest->imu_states.pos_end = Eigen::Vector3d(1.0, 2.0, 3.0);
  KeyFrameData prev_backend;
  prev_backend.timestamp = 0.5;
  const auto imu = sio_test::MakeStaticImu(1.0, 3, 100.0);

  const auto backend = manager.CreateBackendKeyFrame(prev_backend, latest, imu);

  ASSERT_TRUE(backend);
  EXPECT_TRUE(backend->is_backend_keyframe);
  EXPECT_EQ(backend->keyframe_id, 0);
  EXPECT_DOUBLE_EQ(backend->last_keyframe_time, 0.5);
  EXPECT_EQ(backend->imu_buffer.size(), imu.size());
  ASSERT_TRUE(backend->pointcloud);
  EXPECT_EQ(backend->pointcloud->size(), latest->pointcloud->size());
}

TEST(KeyframeManagerTest, ImuBufferAddExtractAndClear) {
  auto manager = MakeManager();
  manager.AddImuData(sio_test::MakeImu(1.0, Eigen::Vector3d::Zero(),
                                       Eigen::Vector3d::Zero()));
  manager.AddImuData(sio_test::MakeImu(1.1, Eigen::Vector3d::Zero(),
                                       Eigen::Vector3d::Zero()));
  manager.AddImuData(sio_test::MakeImu(1.2, Eigen::Vector3d::Zero(),
                                       Eigen::Vector3d::Zero()));

  const auto extracted = manager.ExtractImuBetween(1.05, 1.15);
  ASSERT_EQ(extracted.size(), 1u);
  EXPECT_NEAR(svio_utils::MsgTimeSec(extracted.front()), 1.1, 1e-12);

  const auto all = manager.GetImuBufferForKeyFrame();
  EXPECT_EQ(all.size(), 3u);
  EXPECT_TRUE(manager.GetImuBufferForKeyFrame().empty());
}

TEST(KeyframeManagerTest, CreatesKeyframesWithCopiedFrameData) {
  auto manager = MakeManager();
  const auto frame0 = sio_test::MakeFrame(1.0, 0);
  const auto cloud0 = sio_test::MakeLineCloud(5);
  ImuStatesGroup state0;
  state0.pos_end = Eigen::Vector3d(0.0, 0.0, 0.0);
  cv::Mat disp0(2, 3, CV_32FC1, cv::Scalar(4.0f));
  cv::Mat conf0(2, 3, CV_32FC1, cv::Scalar(0.8f));
  const std::vector<float> point_conf{0.5f, 0.6f};

  const KeyFrameData first = manager.CreateFirstKeyFrame(
      frame0, cloud0, state0, disp0, point_conf,
      std::vector<Eigen::Matrix3d>{}, conf0);

  EXPECT_EQ(first.keyframe_id, 0);
  EXPECT_EQ(first.frontend_keyframe_id, 0);
  EXPECT_FALSE(first.is_backend_keyframe);
  EXPECT_DOUBLE_EQ(first.timestamp, 1.0);
  EXPECT_EQ(first.pointcloud, cloud0);
  EXPECT_EQ(first.point_confidences, point_conf);
  EXPECT_FALSE(first.image_left.empty());
  EXPECT_FALSE(first.confidence_map.empty());

  const auto frame1 = sio_test::MakeFrame(1.5, 1);
  const auto cloud1 = sio_test::MakeLineCloud(6);
  ImuStatesGroup state1 = state0;
  state1.pos_end = Eigen::Vector3d(1.0, 0.0, 0.0);
  state1.vel_end = Eigen::Vector3d(0.5, 0.0, 0.0);
  const auto frontend = manager.CreateFrontendKeyFrame(
      first, frame1, sio_test::MakeStaticImu(1.1, 2, 100.0), cloud1, state1,
      disp0, point_conf, std::vector<Eigen::Matrix3d>{}, conf0);

  ASSERT_TRUE(frontend);
  EXPECT_EQ(frontend->keyframe_id, 1);
  EXPECT_EQ(frontend->frontend_keyframe_id, 1);
  EXPECT_EQ(frontend->prev_frontend_keyframe_id, 0);
  EXPECT_TRUE(frontend->has_relative_edge);
  EXPECT_NEAR(frontend->relative_T_prev_curr(0, 3), 1.0, 1e-9);
  EXPECT_DOUBLE_EQ(frontend->last_keyframe_time, first.timestamp);
  EXPECT_DOUBLE_EQ(frontend->timestamp, 1.5);
  EXPECT_EQ(frontend->pointcloud, cloud1);

  const auto backend = manager.CreateBackendKeyFrame(first, frontend,
                                                     sio_test::MakeStaticImu(1.1, 2, 100.0));
  ASSERT_TRUE(backend);
  EXPECT_EQ(backend->frontend_keyframe_id, 1);
  EXPECT_NE(backend->keyframe_id, frontend->keyframe_id);
  EXPECT_TRUE(backend->is_backend_keyframe);
}

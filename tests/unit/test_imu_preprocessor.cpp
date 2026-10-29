#include <gtest/gtest.h>

#include "imu_msg_utils.h"
#include "imu_preprocessor.h"

TEST(ImuPreprocessorTest, DisabledReturnsClone) {
  ImuPreprocessor preprocessor;
  ImuPreprocessorConfig cfg;
  cfg.enable = false;
  preprocessor.Configure(cfg);

  auto raw = MakeImuMsg(1.0, 0.0, 0.0, 9.81, 0.1, 0.0, 0.0);
  auto out = preprocessor.Process(raw);
  EXPECT_TRUE(out);
  EXPECT_NEAR(out->linear_acceleration.z, 9.81, 1e-6);
  EXPECT_NEAR(out->angular_velocity.x, 0.1, 1e-6);
}

TEST(ImuPreprocessorTest, OutlierReplaceKeepsSampleRate) {
  ImuPreprocessor preprocessor;
  ImuPreprocessorConfig cfg;
  cfg.enable = true;
  cfg.enable_lpf = false;
  cfg.enable_outlier_reject = true;
  cfg.gyro_outlier_rad_s = 1.0;
  preprocessor.Configure(cfg);

  auto good = MakeImuMsg(1.0, 0.0, 0.0, 9.81, 0.1, 0.0, 0.0);
  auto spike = MakeImuMsg(1.0025, 0.0, 0.0, 9.81, 10.0, 0.0, 0.0);
  preprocessor.Process(good);
  auto out = preprocessor.Process(spike);
  EXPECT_TRUE(out);
  EXPECT_NEAR(out->angular_velocity.x, 0.1, 1e-3);
}

TEST(ImuPreprocessorTest, ProcessBatchPreservesCount) {
  ImuPreprocessor preprocessor;
  ImuPreprocessorConfig cfg;
  cfg.enable = true;
  preprocessor.Configure(cfg);

  std::deque<sensor_msgs::msg::Imu::SharedPtr> inputs;
  for (int i = 0; i < 5; ++i) {
    inputs.push_back(MakeImuMsg(1.0 + 0.0025 * i, 0.0, 0.0, 9.81, 0.0, 0.0, 0.0));
  }
  const auto outputs = preprocessor.ProcessBatch(inputs);
  EXPECT_EQ(outputs.size(), inputs.size());
}

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>

#include "fusion_debug.h"
#include "imu_debug.h"
#include "imu_msg_utils.h"

TEST(DebugLoggerTest, ImuDebugWritesCsvHeaderAndRow) {
  const std::filesystem::path path =
    std::filesystem::temp_directory_path() / "sio_imu_debug_test.csv";
  std::filesystem::remove(path);

  ImuDebugLogger logger;
  logger.Open(path.string());

  FrontKeyframeDecision kf;
  kf.is_keyframe = true;
  kf.trans_trigger = true;
  kf.trans_dist = 0.5;
  auto imu = MakeImuMsg(1.0, 0.0, 0.0, 9.81, 0.0, 0.0, 0.0);
  std::deque<sensor_msgs::msg::Imu::SharedPtr> frame_imu{imu};
  Pose6D preint;
  ImuStatesGroup prop;
  const auto record = BuildImuKeyframeDebugRecord(
    3, 1.5, frame_imu, kf, prop, preint, 9.81, 0.02, 0.001, 1, 1.0, 0.5, 1.49);
  logger.WriteRecord(record);

  std::ifstream ifs(path);
  std::string header;
  std::string row;
  std::getline(ifs, header);
  std::getline(ifs, row);
  EXPECT_FALSE(header.empty());
  EXPECT_FALSE(row.empty());
  EXPECT_NE(header.find("frame_id"), std::string::npos);
  EXPECT_NE(row.find("3"), std::string::npos);
  std::filesystem::remove(path);
}

TEST(DebugLoggerTest, FusionDebugFormatsConsoleLine) {
  VisualFusionDebugRecord record;
  record.frame_id = 7;
  record.is_keyframe = false;
  record.source_cloud_size = 1200;
  record.target_cloud_size = 1100;
  record.gicp_inliers = 80;
  const std::string line = FusionDebugLogger::FormatConsoleLine(record);
  EXPECT_NE(line.find("frame=7"), std::string::npos);
  EXPECT_NE(line.find("gicp_inl=80"), std::string::npos);
}

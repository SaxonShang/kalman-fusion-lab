#pragma once

#include "data.h"
#include "sio_stereo_geometry.h"

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#if defined(SIO_ENABLE_ZED)
#include <sl/Camera.hpp>
#endif

enum class DatasetLoaderSource {
  FILE,
  ZED2I,
};

struct Zed2iOptions {
  int camera_fps = 30;
  enum class Resolution { HD720, HD1080, HD2K, VGA };
  Resolution resolution = Resolution::HD720;
  int serial_number = 0; // 0 表示任意已连接设备
  std::size_t max_imu_cache = 4000;
  std::size_t min_imu_per_interval = 25;
};

class DatasetLoader {
public:
  DatasetLoader(const std::string &img_dir, const std::string &imu_file);
#if defined(SIO_ENABLE_ZED)
  explicit DatasetLoader(const Zed2iOptions &opts);
#endif
  ~DatasetLoader();

  DatasetLoader(const DatasetLoader &) = delete;
  DatasetLoader &operator=(const DatasetLoader &) = delete;

  std::shared_ptr<FrameData> GrabNextFrame();

  // 中心裁减配置（默认从 params.yaml 加载）；ZED / 文件模式均在读入后立即裁减
  void ConfigureStereoPreprocess(const sio_utils::StereoImageGeometry &geom);

  void Stop();

  DatasetLoaderSource GetSource() const { return source_; }

  // IMU 初始化完成后调用，将 frame_id 重置为 0
  void ResetFrameId() { current_frame_id_ = 0; }

private:
  struct RecordedFrameEntry {
    int record_frame_id = 0;
    double st_time = 0.0;
    double end_time = 0.0;
    std::string left_path;
    std::string right_path;
    int expected_imu_count = 0;
  };

  void InitStereoPreprocessFromParams();
  void ApplyStereoCenterCrop(cv::Mat &left, cv::Mat &right) const;
  bool LoadZedRecordFrames(const std::filesystem::path &frames_csv);
  bool LoadZedRecordImu(const std::string &imu_file);

  std::shared_ptr<FrameData> GrabNextFrameFromFiles();
  std::shared_ptr<FrameData> GrabNextFrameFromZedRecord();
#if defined(SIO_ENABLE_ZED)
  void StartZedCapture(const Zed2iOptions &opts);
  void ImageCaptureLoop();
  void ImuCaptureLoop();
  std::shared_ptr<FrameData> GrabNextFrameFromZed();

  struct StereoFrameSample {
    cv::Mat left;
    cv::Mat right;
    double timestamp_sec = 0.0;
    uint64_t timestamp_ns = 0;
  };

  std::unique_ptr<sl::Camera> zed_;
  std::thread image_thread_;
  std::thread imu_thread_;
  std::atomic<bool> stop_{false};
  std::atomic<bool> zed_ready_{false};

  std::mutex image_mutex_;
  std::condition_variable image_cv_;
  StereoFrameSample latest_image_;
  bool image_updated_ = false;

  std::mutex imu_mutex_;
  std::deque<sensor_msgs::msg::Imu::SharedPtr> imu_cache_;
  std::size_t max_imu_cache_ = 4000;
  std::size_t min_imu_per_interval_ = 25;

  double last_image_time_sec_ = 0.0;
  bool has_last_image_time_ = false;
  std::size_t zed_frame_index_ = 0;
#endif

  DatasetLoaderSource source_ = DatasetLoaderSource::FILE;

  std::vector<std::string> leftimg_files_;
  std::vector<std::string> rightimg_files_;
  std::ifstream infile_;

  bool use_zed_record_format_ = false;
  std::string dataset_root_;
  std::vector<RecordedFrameEntry> recorded_frames_;
  std::vector<std::vector<sensor_msgs::msg::Imu::SharedPtr>> imu_samples_by_frame_id_;

  std::size_t frame_count_ = 0;
  std::size_t current_index_ = 0;
  int current_frame_id_ = 0;

  std::string pending_line_;
  bool has_pending_line_ = false;

  sio_utils::StereoImageGeometry stereo_geom_{};
  sio_utils::StereoCropRect stereo_crop_{};
  bool stereo_crop_enabled_ = false;
};

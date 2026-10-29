#include "dataset_loader.h"
#include "imu_msg_utils.h"
#include "sio_config.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <sstream>

namespace {

constexpr double kDegToRad = M_PI / 180.0;

std::vector<std::string> GetFilesInDirectory(const std::string &directory) {
  std::vector<std::string> files;
  for (const auto &entry : std::filesystem::directory_iterator(directory)) {
    files.push_back(entry.path().string());
  }
  std::sort(files.begin(), files.end());
  return files;
}

long long ExtractTimestampFromFilename(const std::string &path) {
  const auto slash = path.find_last_of('/');
  const auto dot = path.find_last_of('.');
  const auto start = (slash == std::string::npos) ? 0 : slash + 1;
  const auto len = (dot == std::string::npos || dot <= start) ? std::string::npos : dot - start;
  return std::stoll(path.substr(start, len));
}

#if defined(SIO_ENABLE_ZED)
cv::Mat ZedMatToBgr(const sl::Mat &zed_mat) {
  if (zed_mat.getWidth() <= 0 || zed_mat.getHeight() <= 0) {
    return cv::Mat();
  }

  const int width = zed_mat.getWidth();
  const int height = zed_mat.getHeight();
  const sl::MAT_TYPE mat_type = zed_mat.getDataType();

  if (mat_type == sl::MAT_TYPE::U8_C4) {
    cv::Mat bgra(height, width, CV_8UC4, zed_mat.getPtr<sl::uchar1>(sl::MEM::CPU));
    cv::Mat bgr;
    cv::cvtColor(bgra, bgr, cv::COLOR_BGRA2BGR);
    return bgr;
  }

  if (mat_type == sl::MAT_TYPE::U8_C3) {
    cv::Mat bgr(height, width, CV_8UC3, zed_mat.getPtr<sl::uchar1>(sl::MEM::CPU));
    return bgr.clone();
  }

  std::cerr << "[DatasetLoader] Unsupported ZED image type\n";
  return cv::Mat();
}

sl::RESOLUTION ToSlResolution(Zed2iOptions::Resolution resolution) {
  switch (resolution) {
  case Zed2iOptions::Resolution::HD2K:
    return sl::RESOLUTION::HD2K;
  case Zed2iOptions::Resolution::HD1080:
    return sl::RESOLUTION::HD1080;
  case Zed2iOptions::Resolution::VGA:
    return sl::RESOLUTION::VGA;
  case Zed2iOptions::Resolution::HD720:
  default:
    return sl::RESOLUTION::HD720;
  }
}

sensor_msgs::msg::Imu::SharedPtr MakeImuMsgFromZed(const sl::SensorsData::IMUData &imu_data) {
  const double time_sec = imu_data.timestamp.getNanoseconds() * 1e-9;
  return MakeImuMsg(time_sec, imu_data.linear_acceleration.x, imu_data.linear_acceleration.y,
                    imu_data.linear_acceleration.z, imu_data.angular_velocity.x * kDegToRad,
                    imu_data.angular_velocity.y * kDegToRad,
                    imu_data.angular_velocity.z * kDegToRad);
}
#endif

} // namespace

void DatasetLoader::ConfigureStereoPreprocess(const sio_utils::StereoImageGeometry &geom) {
  stereo_geom_ = geom;
  stereo_crop_ = sio_utils::ComputeCenterCropRect(geom);
  stereo_crop_enabled_ = sio_utils::StereoPreprocessRequired(geom);
}

void DatasetLoader::InitStereoPreprocessFromParams() {
  const std::filesystem::path params_file = sio_utils::LocateParamsFile();
  if (params_file.empty()) {
    return;
  }
  ConfigureStereoPreprocess(sio_utils::LoadStereoImageGeometry(params_file));
}

void DatasetLoader::ApplyStereoCenterCrop(cv::Mat &left, cv::Mat &right) const {
  if (!stereo_crop_enabled_) {
    return;
  }

  sio_utils::FitImageToSourceGeometry(left, stereo_geom_);
  sio_utils::FitImageToSourceGeometry(right, stereo_geom_);

  if (!left.empty()) {
    const cv::Mat cropped_l = sio_utils::CenterCrop(left, stereo_crop_);
    if (!cropped_l.empty()) {
      left = cropped_l;
    }
  }
  if (!right.empty()) {
    const cv::Mat cropped_r = sio_utils::CenterCrop(right, stereo_crop_);
    if (!cropped_r.empty()) {
      right = cropped_r;
    }
  }
}

DatasetLoader::DatasetLoader(const std::string &img_dir, const std::string &imu_file) {
  InitStereoPreprocessFromParams();
  source_ = DatasetLoaderSource::FILE;

  dataset_root_ = img_dir;
  if (!dataset_root_.empty() && dataset_root_.back() != '/') {
    dataset_root_.push_back('/');
  }

  const std::filesystem::path frames_csv = std::filesystem::path(img_dir) / "frames.csv";
  if (std::filesystem::exists(frames_csv)) {
    use_zed_record_format_ = true;
    if (!LoadZedRecordFrames(frames_csv) || !LoadZedRecordImu(imu_file)) {
      frame_count_ = 0;
    }
    return;
  }

  leftimg_files_ = GetFilesInDirectory(img_dir + "left/");
  rightimg_files_ = GetFilesInDirectory(img_dir + "right/");

  if (leftimg_files_.empty() || rightimg_files_.empty()) {
    std::cerr << "[DatasetLoader] image directory is empty\n";
    return;
  }

  frame_count_ = std::min(leftimg_files_.size(), rightimg_files_.size());

  infile_.open(imu_file);
  if (!infile_.is_open()) {
    std::cerr << "[DatasetLoader] failed to open imu file: " << imu_file << "\n";
    return;
  }

  std::string line;
  std::getline(infile_, line); // 跳过表头
}

bool DatasetLoader::LoadZedRecordFrames(const std::filesystem::path &frames_csv) {
  std::ifstream in(frames_csv);
  if (!in.is_open()) {
    std::cerr << "[DatasetLoader] failed to open frames.csv: " << frames_csv << "\n";
    return false;
  }

  std::string line;
  if (!std::getline(in, line)) {
    std::cerr << "[DatasetLoader] frames.csv is empty\n";
    return false;
  }

  recorded_frames_.clear();
  while (std::getline(in, line)) {
    if (line.empty()) {
      continue;
    }

    std::stringstream ss(line);
    std::string token;
    std::vector<std::string> values;
    while (std::getline(ss, token, ',')) {
      values.push_back(token);
    }
    if (values.size() < 6) {
      continue;
    }

    RecordedFrameEntry entry;
    entry.record_frame_id = std::stoi(values[0]);
    entry.st_time = std::stod(values[1]);
    entry.end_time = std::stod(values[2]);
    entry.left_path = values[3];
    entry.right_path = values[4];
    entry.expected_imu_count = std::stoi(values[5]);
    recorded_frames_.push_back(entry);
  }

  frame_count_ = recorded_frames_.size();
  if (frame_count_ == 0) {
    std::cerr << "[DatasetLoader] no frames parsed from frames.csv\n";
    return false;
  }

  std::cout << "[DatasetLoader] Loaded " << frame_count_ << " frames from frames.csv\n";
  return true;
}

bool DatasetLoader::LoadZedRecordImu(const std::string &imu_file) {
  std::ifstream in(imu_file);
  if (!in.is_open()) {
    std::cerr << "[DatasetLoader] failed to open imu file: " << imu_file << "\n";
    return false;
  }

  imu_samples_by_frame_id_.clear();
  int max_frame_id = -1;
  std::string line;
  if (!std::getline(in, line)) {
    std::cerr << "[DatasetLoader] imu file is empty\n";
    return false;
  }

  while (std::getline(in, line)) {
    if (line.empty()) {
      continue;
    }

    std::stringstream ss(line);
    std::string token;
    std::vector<std::string> values;
    while (std::getline(ss, token, ',')) {
      values.push_back(token);
    }
    if (values.size() < 8) {
      continue;
    }

    const int frame_id = std::stoi(values[0]);
    const double timestamp_sec = std::stod(values[1]);
    const double ax = std::stod(values[3]);
    const double ay = std::stod(values[4]);
    const double az = std::stod(values[5]);
    const double gx = std::stod(values[6]);
    const double gy = std::stod(values[7]);
    const double gz = std::stod(values[8]);

    max_frame_id = std::max(max_frame_id, frame_id);
    if (frame_id >= static_cast<int>(imu_samples_by_frame_id_.size())) {
      imu_samples_by_frame_id_.resize(static_cast<size_t>(frame_id) + 1);
    }
    imu_samples_by_frame_id_[static_cast<size_t>(frame_id)].push_back(
      MakeImuMsg(timestamp_sec, ax, ay, az, gx * kDegToRad, gy * kDegToRad, gz * kDegToRad));
  }

  if (max_frame_id < 0) {
    std::cerr << "[DatasetLoader] no IMU samples parsed from " << imu_file << "\n";
    return false;
  }

  std::size_t total_imu = 0;
  for (const auto &samples : imu_samples_by_frame_id_) {
    total_imu += samples.size();
  }
  std::cout << "[DatasetLoader] Loaded " << total_imu << " IMU samples for "
            << (max_frame_id + 1) << " frame ids\n";
  return true;
}

#if defined(SIO_ENABLE_ZED)
DatasetLoader::DatasetLoader(const Zed2iOptions &opts) {
  InitStereoPreprocessFromParams();
  StartZedCapture(opts);
}
#endif

DatasetLoader::~DatasetLoader() {
  Stop();
  if (infile_.is_open()) {
    infile_.close();
  }
}

void DatasetLoader::Stop() {
#if defined(SIO_ENABLE_ZED)
  if (source_ != DatasetLoaderSource::ZED2I) {
    return;
  }

  stop_.store(true);
  image_cv_.notify_all();

  if (image_thread_.joinable()) {
    image_thread_.join();
  }
  if (imu_thread_.joinable()) {
    imu_thread_.join();
  }

  if (zed_) {
    zed_->close();
    zed_.reset();
  }
#endif
}

std::shared_ptr<FrameData> DatasetLoader::GrabNextFrame() {
  if (source_ == DatasetLoaderSource::FILE) {
    return GrabNextFrameFromFiles();
  }
#if defined(SIO_ENABLE_ZED)
  return GrabNextFrameFromZed();
#else
  return nullptr;
#endif
}

std::shared_ptr<FrameData> DatasetLoader::GrabNextFrameFromFiles() {
  if (use_zed_record_format_) {
    return GrabNextFrameFromZedRecord();
  }

  if (current_index_ >= frame_count_) {
    return nullptr;
  }

  const size_t i = current_index_;
  cv::Mat img_l = cv::imread(leftimg_files_[i], cv::IMREAD_COLOR);
  cv::Mat img_r = cv::imread(rightimg_files_[i], cv::IMREAD_COLOR);

  if (img_l.empty() || img_r.empty()) {
    std::cerr << "[DatasetLoader] failed to read image pair: " << i << "\n";
    current_index_++;
    return nullptr;
  }

  ApplyStereoCenterCrop(img_l, img_r);

  const long long img_l_timestamp = ExtractTimestampFromFilename(leftimg_files_[i]);
  const long long img_r_timestamp = ExtractTimestampFromFilename(rightimg_files_[i]);

  if (img_l_timestamp != img_r_timestamp) {
    std::cerr << "[DatasetLoader] timestamp mismatch at frame " << i << "\n";
    current_index_++;
    return nullptr;
  }

  const double last_time =
    i > 0 ? ExtractTimestampFromFilename(leftimg_files_[i - 1]) / 1000000.0 : 0.0;
  const double current_time = img_l_timestamp / 1000000.0;

  std::deque<sensor_msgs::msg::Imu::SharedPtr> imu_buffer;
  int imu_data_count = 0;
  std::string line;

  while (true) {
    if (!has_pending_line_) {
      if (!std::getline(infile_, line)) {
        break;
      }
    } else {
      line = pending_line_;
      has_pending_line_ = false;
    }

    if (line.empty()) {
      continue;
    }

    std::stringstream ss(line);
    std::string token;
    std::vector<std::string> values;
    while (std::getline(ss, token, ',')) {
      values.push_back(token);
    }

    if (values.size() < 9) {
      continue;
    }

    const long long img_ts = std::stoll(values[0]);
    const long long imu_ts = std::stoll(values[2]);
    const double ax = std::stod(values[3]);
    const double ay = std::stod(values[4]);
    const double az = std::stod(values[5]);
    const double gx = std::stod(values[6]);
    const double gy = std::stod(values[7]);
    const double gz = std::stod(values[8]);

    if (img_ts < img_l_timestamp) {
      continue;
    }

    if (img_ts > img_l_timestamp) {
      pending_line_ = line;
      has_pending_line_ = true;
      break;
    }

    auto imu_msg = MakeImuMsg(imu_ts / 1000000.0, ax, ay, az, gx * kDegToRad, gy * kDegToRad,
                              gz * kDegToRad);
    imu_buffer.push_back(imu_msg);
    ++imu_data_count;
  }

  if (i > 0 && imu_data_count < 25) {
    std::cerr << "[DatasetLoader] imu data too few at frame " << i << "\n";
    current_index_++;
    return nullptr;
  }

  auto frame_data = std::make_shared<FrameData>();
  frame_data->st_time = last_time;
  frame_data->end_time = current_time;
  frame_data->imgl = img_l;
  frame_data->imgr = img_r;
  frame_data->imu = imu_buffer;
  frame_data->frame_id = current_frame_id_++;

  std::cout << "[DatasetLoader] Grabbed frame " << i << ", imu count = " << imu_data_count << "\n";
  current_index_++;
  return frame_data;
}

std::shared_ptr<FrameData> DatasetLoader::GrabNextFrameFromZedRecord() {
  if (current_index_ >= frame_count_) {
    return nullptr;
  }

  const size_t i = current_index_;
  const RecordedFrameEntry &entry = recorded_frames_[i];
  const std::string left_path = dataset_root_ + entry.left_path;
  const std::string right_path = dataset_root_ + entry.right_path;

  cv::Mat img_l = cv::imread(left_path, cv::IMREAD_COLOR);
  cv::Mat img_r = cv::imread(right_path, cv::IMREAD_COLOR);
  if (img_l.empty() || img_r.empty()) {
    std::cerr << "[DatasetLoader] failed to read image pair: " << i << " (" << left_path << ")\n";
    current_index_++;
    return nullptr;
  }

  ApplyStereoCenterCrop(img_l, img_r);

  std::deque<sensor_msgs::msg::Imu::SharedPtr> imu_buffer;
  const int frame_id = entry.record_frame_id;
  if (frame_id >= 0 && frame_id < static_cast<int>(imu_samples_by_frame_id_.size())) {
    for (const auto &imu_msg : imu_samples_by_frame_id_[static_cast<size_t>(frame_id)]) {
      imu_buffer.push_back(imu_msg);
    }
  }

  const int imu_data_count = static_cast<int>(imu_buffer.size());
  if (i > 0 && entry.expected_imu_count > 0
      && imu_data_count < std::min(25, entry.expected_imu_count)) {
    std::cerr << "[DatasetLoader] imu data too few at frame " << i << " (got " << imu_data_count
              << ", expected " << entry.expected_imu_count << ")\n";
    current_index_++;
    return nullptr;
  }

  auto frame_data = std::make_shared<FrameData>();
  frame_data->st_time = entry.st_time;
  frame_data->end_time = entry.end_time;
  frame_data->imgl = img_l;
  frame_data->imgr = img_r;
  frame_data->imu = imu_buffer;
  frame_data->frame_id = current_frame_id_++;

  std::cout << "[DatasetLoader] Grabbed frame " << i << ", imu count = " << imu_data_count << "\n";
  current_index_++;
  return frame_data;
}
// 启动 ZED 相机捕获
#if defined(SIO_ENABLE_ZED)
void DatasetLoader::StartZedCapture(const Zed2iOptions &opts) {
  source_ = DatasetLoaderSource::ZED2I;
  max_imu_cache_ = opts.max_imu_cache;
  min_imu_per_interval_ = opts.min_imu_per_interval;

  zed_ = std::make_unique<sl::Camera>();

  sl::InitParameters init_params;
  init_params.camera_fps = opts.camera_fps;
  init_params.camera_resolution = ToSlResolution(opts.resolution);
  init_params.depth_mode = sl::DEPTH_MODE::NONE;
  init_params.sensors_required = true;
  init_params.coordinate_units = sl::UNIT::METER;
  init_params.coordinate_system = sl::COORDINATE_SYSTEM::RIGHT_HANDED_Z_UP;
  if (opts.serial_number > 0) {
    init_params.input.setFromSerialNumber(opts.serial_number);
  }

  const sl::ERROR_CODE open_err = zed_->open(init_params);
  if (open_err != sl::ERROR_CODE::SUCCESS) {
    std::cerr << "[DatasetLoader] Failed to open ZED 2i: " << sl::toString(open_err) << "\n";
    zed_.reset();
    return;
  }

  zed_ready_.store(true);
  stop_.store(false);

  imu_thread_ = std::thread(&DatasetLoader::ImuCaptureLoop, this);
  image_thread_ = std::thread(&DatasetLoader::ImageCaptureLoop, this);

  std::cout << "[DatasetLoader] ZED 2i capture started (fps=" << opts.camera_fps << ")\n";
}

void DatasetLoader::ImageCaptureLoop() {
  sl::RuntimeParameters runtime_params;
  runtime_params.confidence_threshold = 50;

  while (!stop_.load()) {
    if (!zed_) {
      break;
    }

    const sl::ERROR_CODE grab_err = zed_->grab(runtime_params);
    if (grab_err != sl::ERROR_CODE::SUCCESS) {
      if (stop_.load()) {
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      continue;
    }

    sl::Mat left_zed;
    sl::Mat right_zed;
    zed_->retrieveImage(left_zed, sl::VIEW::LEFT, sl::MEM::CPU);
    zed_->retrieveImage(right_zed, sl::VIEW::RIGHT, sl::MEM::CPU);

    cv::Mat left_bgr = ZedMatToBgr(left_zed);
    cv::Mat right_bgr = ZedMatToBgr(right_zed);
    if (left_bgr.empty() || right_bgr.empty()) {
      continue;
    }

    ApplyStereoCenterCrop(left_bgr, right_bgr);

    const sl::Timestamp image_ts = zed_->getTimestamp(sl::TIME_REFERENCE::IMAGE);
    const uint64_t timestamp_ns = image_ts.getNanoseconds();
    if (timestamp_ns == 0) {
      continue;
    }

    {
      std::lock_guard<std::mutex> lock(image_mutex_);
      latest_image_.left = std::move(left_bgr);
      latest_image_.right = std::move(right_bgr);
      latest_image_.timestamp_sec = timestamp_ns * 1e-9;
      latest_image_.timestamp_ns = timestamp_ns;
      image_updated_ = true;
    }
    image_cv_.notify_all();
  }
}

void DatasetLoader::ImuCaptureLoop() {
  sl::SensorsData sensors_data;
  uint64_t last_imu_timestamp_ns = 0;

  while (!stop_.load()) {
    if (!zed_) {
      break;
    }

    const sl::ERROR_CODE sensor_err =
      zed_->getSensorsData(sensors_data, sl::TIME_REFERENCE::CURRENT);
    if (sensor_err != sl::ERROR_CODE::SUCCESS) {
      std::this_thread::sleep_for(std::chrono::microseconds(500));
      continue;
    }

    if (!sensors_data.imu.is_available) {
      std::this_thread::sleep_for(std::chrono::microseconds(500));
      continue;
    }

    const uint64_t imu_timestamp_ns = sensors_data.imu.timestamp.getNanoseconds();
    if (imu_timestamp_ns == 0 || imu_timestamp_ns == last_imu_timestamp_ns) {
      std::this_thread::sleep_for(std::chrono::microseconds(500));
      continue;
    }
    last_imu_timestamp_ns = imu_timestamp_ns;

    auto imu_msg = MakeImuMsgFromZed(sensors_data.imu);
    {
      std::lock_guard<std::mutex> lock(imu_mutex_);
      imu_cache_.push_back(imu_msg);
      while (imu_cache_.size() > max_imu_cache_) {
        imu_cache_.pop_front();
      }
    }
  }
}
// 从 ZED 相机中获取下一帧数据
std::shared_ptr<FrameData> DatasetLoader::GrabNextFrameFromZed() {
  if (!zed_ready_.load() || !zed_) {
    return nullptr;
  }

  StereoFrameSample frame_sample;
  {
    std::unique_lock<std::mutex> lock(image_mutex_);
    image_cv_.wait(lock, [this]() {
      if (stop_.load()) {
        return true;
      }
      if (!image_updated_) {
        return false;
      }
      if (!has_last_image_time_) {
        return true;
      }
      return latest_image_.timestamp_sec > last_image_time_sec_;
    });

    if (stop_.load() && !image_updated_) {
      return nullptr;
    }

    frame_sample.left = latest_image_.left.clone();
    frame_sample.right = latest_image_.right.clone();
    frame_sample.timestamp_sec = latest_image_.timestamp_sec;
    frame_sample.timestamp_ns = latest_image_.timestamp_ns;
    image_updated_ = false;
  }

  if (frame_sample.left.empty() || frame_sample.right.empty()) {
    return nullptr;
  }
  // 从 ZED 相机中获取最新的 IMU 缓存数据
  std::deque<sensor_msgs::msg::Imu::SharedPtr> imu_snapshot;
  {
    std::lock_guard<std::mutex> lock(imu_mutex_);
    imu_snapshot = imu_cache_;
  }

  const double start_time_sec = has_last_image_time_ ? last_image_time_sec_ : 0.0;
  const double end_time_sec = frame_sample.timestamp_sec;
  std::deque<sensor_msgs::msg::Imu::SharedPtr> imu_packed =
    PackImuBetween(imu_snapshot, start_time_sec, end_time_sec);

  if (has_last_image_time_ && imu_packed.size() < min_imu_per_interval_) {
    std::cerr << "[DatasetLoader] imu data too few between frames at zed frame "
              << zed_frame_index_ << " (count=" << imu_packed.size() << ")\n";
    last_image_time_sec_ = end_time_sec;
    ++zed_frame_index_;
    return nullptr;
  }

  auto frame_data = std::make_shared<FrameData>();
  frame_data->st_time = start_time_sec;
  frame_data->end_time = end_time_sec;
  frame_data->imgl = std::move(frame_sample.left);
  frame_data->imgr = std::move(frame_sample.right);
  frame_data->imu = std::move(imu_packed);
  frame_data->frame_id = current_frame_id_++;

  last_image_time_sec_ = end_time_sec;
  has_last_image_time_ = true;

  std::cout << "[DatasetLoader] Grabbed ZED frame " << zed_frame_index_
            << ", imu count = " << frame_data->imu.size() << ", t=" << end_time_sec << "\n";
  ++zed_frame_index_;
  return frame_data;
}
#endif

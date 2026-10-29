#include "lab.h"
#include "dataset_loader.h"
#include "imu_preprocessor.h"
#include "stereo_gicp.h"
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>

namespace fusion_lab {
namespace {
std::vector<int> SourceIds(const fs::path &dataset) {
  std::ifstream in(dataset / "frames.csv");
  Require(bool(in), "frames.csv is required; run prepare_fr5_dataset.py first");
  std::string line;
  std::getline(in, line);
  std::vector<int> ids;
  while (std::getline(in, line)) {
    std::stringstream stream(line);
    std::vector<std::string> columns;
    std::string field;
    while (std::getline(stream, field, ',')) columns.push_back(field);
    Require(columns.size() >= 7, "frames.csv must contain source_frame_id");
    ids.push_back(std::stoi(columns[6]));
  }
  return ids;
}

cv::Mat Disparity(const YAML::Node &depth, const cv::Mat &left, const cv::Mat &right,
                  double time) {
  cv::Mat disparity;
  const auto provider = depth["provider"].as<std::string>();
  if (provider == "disparity_cache") {
    const auto stem = std::to_string(std::llround(time * 1000));
    auto file = fs::path(depth["disparity_dir"].as<std::string>()) / (stem + ".yml.gz");
    if (!fs::exists(file)) file.replace_filename(stem + ".yml");
    cv::FileStorage input(file.string(), cv::FileStorage::READ);
    Require(input.isOpened(), "Model disparity not found: " + file.string());
    input["disparity"] >> disparity;
    Require(disparity.channels() == 1 &&
            (disparity.depth() == CV_32F || disparity.depth() == CV_64F),
            "Cached disparity must be floating-point pixels, not depth or fixed-point /16");
    disparity.convertTo(disparity, CV_32F);
  } else {
    Require(provider == "sgbm", "Unknown disparity provider: " + provider);
    const int n = depth["sgbm_num_disparities"].as<int>();
    const int block = depth["sgbm_block_size"].as<int>();
    Require(n > 0 && n % 16 == 0 && n < left.cols && block >= 3 && block % 2 == 1,
            "Invalid SGBM disparity count or block size");
    const auto matcher = cv::StereoSGBM::create(0, n, block, 8 * block * block,
        32 * block * block, 1, 31, 10, 100, 2, cv::StereoSGBM::MODE_SGBM_3WAY);
    cv::Mat gray_left, gray_right, fixed;
    cv::cvtColor(left, gray_left, cv::COLOR_BGR2GRAY);
    cv::cvtColor(right, gray_right, cv::COLOR_BGR2GRAY);
    matcher->compute(gray_left, gray_right, fixed);
    fixed.convertTo(disparity, CV_32F, 1.0 / 16.0);
  }
  Require(disparity.size() == left.size(), "Disparity must match inference image geometry");
  return disparity;
}
} // namespace

void Prepare(const YAML::Node &cfg, const fs::path &output) {
  const fs::path params = cfg["params_file"].as<std::string>();
  const auto geom = sio_utils::LoadStereoImageGeometry(params);
  const auto cam = sio_utils::LoadInferenceCameraIntrinsics(params);
  const auto ids = SourceIds(cfg["dataset_dir"].as<std::string>());
  Require(sio_stereo_gicp::CameraIsValid(cam), "Invalid inference intrinsics");
  std::set<int> wanted;
  for (const auto &pair : cfg["pairs"]) {
    const int a = pair["anchor"].as<int>(), b = pair["current"].as<int>();
    Require(a < b, "Pair anchor must precede current frame");
    wanted.insert(a); wanted.insert(b);
  }
  Require(!wanted.empty(), "No frame pairs configured");
  for (int id : wanted)
    Require(std::find(ids.begin(), ids.end(), id) != ids.end(),
            "Source frame id is not in prepared recording: " + std::to_string(id));
  const auto depth_cfg = cfg["depth"];
  const float zmin = depth_cfg["min_m"].as<float>(), zmax = depth_cfg["max_m"].as<float>();
  const int stride = depth_cfg["pixel_stride"].as<int>();
  Require(zmin > 0 && zmax > zmin && stride > 0, "Invalid depth limits or pixel stride");
  DatasetLoader loader(cfg["dataset_dir"].as<std::string>(), cfg["imu_file"].as<std::string>());
  loader.ConfigureStereoPreprocess(geom);
  ImagePreprocessor image_preprocessor;
  image_preprocessor.LoadConfig(params.string());
  ImuPreprocessor imu_preprocessor;
  imu_preprocessor.LoadConfig(params.string());
  ImuIntegrator integrator;
  ImuStatesGroup state;
  std::deque<sensor_msgs::msg::Imu::SharedPtr> imu;
  bool initialized = false;
  std::size_t index = 0, cached = 0;
  double previous_imu_time = 0;
  while (index < ids.size() && ids[index] <= *wanted.rbegin()) {
    auto input = loader.GrabNextFrame();
    Require(bool(input), "DatasetLoader rejected frame " + std::to_string(index));
    const int source_id = ids[index++];
    for (const auto &sample : input->imu) {
      auto processed = imu_preprocessor.Process(sample);
      if (!processed) continue;
      const double time = processed->header.stamp.sec + processed->header.stamp.nanosec * 1e-9;
      Require(!previous_imu_time || (time > previous_imu_time && time - previous_imu_time <= .010001),
              "IMU discontinuity during preparation");
      previous_imu_time = time;
      imu.push_back(processed);
    }
    if (!initialized) {
      if (imu.size() < 200) continue;
      state = integrator.ImuInit(imu);
      // Like the runtime frontend, start the next interval at the image boundary.
      state.timestamp = input->end_time;
      initialized = true;
    } else {
      KeyFrameData anchor;
      anchor.imu_states = state;
      anchor.timestamp = state.timestamp;
      state = integrator.ImuPreintegrationCov(anchor, imu, input->end_time);
    }
    Require(HealthyState(state), "Invalid pure-IMU prior at source frame " + std::to_string(source_id));
    if (wanted.count(source_id)) {
      auto stereo = image_preprocessor.ProcessStereoAdaptiveLighting(input->imgl, input->imgr);
      cv::Mat left, right;
      cv::resize(stereo.first, left, cv::Size(geom.inference_width, geom.inference_height));
      cv::resize(stereo.second, right, left.size());
      const auto disparity = Disparity(depth_cfg, left, right, input->end_time);
      FrozenFrame frame;
      frame.id = source_id; frame.time = input->end_time; frame.prior = state;
      frame.disparity = disparity;
      frame.depth = sio_stereo_gicp::DepthFromDisparity(disparity, cam, zmin, zmax);
      for (int v = 0; v < left.rows; v += stride) {
        for (int u = 0; u < left.cols; u += stride) {
          const float z = frame.depth.at<float>(v, u);
          if (!std::isfinite(z) || z < zmin || z > zmax) continue;
          const auto color = left.at<cv::Vec3b>(v, u);
          PointTypeRGB p;
          p.x = (u - cam.cx) * z / cam.fx; p.y = (v - cam.cy) * z / cam.fy; p.z = z;
          p.r = color[2]; p.g = color[1]; p.b = color[0];
          frame.camera_cloud->push_back(p);
        }
      }
      Require(frame.camera_cloud->size() >= 100, "Too few valid stereo points");
      WriteFrame(output, frame);
      ++cached;
      std::cout << "[freeze] source_frame=" << source_id << " points="
                << frame.camera_cloud->size() << '\n';
    }
    while (imu.size() > 1) imu.pop_front(); // Keep the last sample across frame boundaries.
  }
  Require(cached == wanted.size(), "Some selected frames precede IMU initialization");
  std::ofstream summary(output / "inputs.yaml");
  summary << "format_version: 2\nframes: " << cached
          << "\nprior: pure_imu_dead_reckoning\ngroundtruth_used: false\n"
          << "depth_provider: " << depth_cfg["provider"].as<std::string>() << '\n';
}
} // namespace fusion_lab

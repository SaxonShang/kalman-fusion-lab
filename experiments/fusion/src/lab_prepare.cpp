#include "lab.h"
#include "dataset_loader.h"
#include "imu_integrator.h"
#include "imu_preprocessor.h"
#include "stereo_gicp.h"
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
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
ImuNoiseParams NoiseFromConfig(const YAML::Node &prior, bool has_prior) {
  ImuNoiseParams noise; // legacy constants of the core integrator
  if (!has_prior || !prior["imu_noise"]) return noise;
  const YAML::Node node = prior["imu_noise"];
  if (node.IsScalar()) {
    Require(node.as<std::string>() == "legacy", "prior.imu_noise must be 'legacy' or a map");
    return noise;
  }
  noise.gyro_noise_density = node["gyro_noise_density"].as<double>();
  noise.accel_noise_density = node["accel_noise_density"].as<double>();
  noise.gyro_random_walk = node["gyro_random_walk"].as<double>();
  noise.accel_random_walk = node["accel_random_walk"].as<double>();
  Require(noise.gyro_noise_density > 0 && noise.accel_noise_density > 0 &&
          noise.gyro_random_walk > 0 && noise.accel_random_walk > 0,
          "IMU noise densities must be positive (SI units)");
  return noise;
}

// Optional offline gyro correction ω' = M ω (prior.gyro_correction.matrix, 9 values row-major).
std::optional<Eigen::Matrix3d> GyroCorrectionFromConfig(const YAML::Node &prior, bool has_prior) {
  if (!has_prior || !prior["gyro_correction"]) return std::nullopt;
  const YAML::Node values = prior["gyro_correction"]["matrix"];
  Require(values.IsSequence() && values.size() == 9, "prior.gyro_correction.matrix needs 9 values");
  Eigen::Matrix3d correction;
  for (int i = 0; i < 9; ++i) correction(i / 3, i % 3) = values[i].as<double>();
  // A calibration of scale and axis alignment stays close to identity.
  Require(correction.allFinite() && correction.determinant() > 0 &&
          (correction - Eigen::Matrix3d::Identity()).norm() < 0.1,
          "prior.gyro_correction.matrix must be finite and close to identity");
  return correction;
}

InitialCovarianceParams CovarianceParamsFromConfig(const YAML::Node &prior, bool has_prior) {
  InitialCovarianceParams params;
  if (!has_prior || !prior["initial_covariance_params"]) return params;
  const YAML::Node node = prior["initial_covariance_params"];
  params.accel_bias_sigma = node["accel_bias_sigma"].as<double>(params.accel_bias_sigma);
  params.yaw_sigma_rad = node["yaw_sigma_rad"].as<double>(params.yaw_sigma_rad);
  params.position_sigma_m = node["position_sigma_m"].as<double>(params.position_sigma_m);
  params.velocity_sigma_mps = node["velocity_sigma_mps"].as<double>(params.velocity_sigma_mps);
  params.gyro_bias_floor_rad_s =
    node["gyro_bias_floor_rad_s"].as<double>(params.gyro_bias_floor_rad_s);
  Require(params.accel_bias_sigma > 0 && params.yaw_sigma_rad > 0 && params.position_sigma_m > 0 &&
          params.velocity_sigma_mps > 0 && params.gyro_bias_floor_rad_s > 0,
          "Initial covariance standard deviations must be positive");
  return params;
}

// Anchor-to-current propagation for the anchor-conditioned prior (format_version 3).
struct AnchorTrack {
  Eigen::Matrix<double, DIM_STATE, DIM_STATE> anchor_cov =
    Eigen::Matrix<double, DIM_STATE, DIM_STATE>::Zero();
  Eigen::Matrix<double, DIM_STATE, DIM_STATE> transition =
    Eigen::Matrix<double, DIM_STATE, DIM_STATE>::Identity();
  Eigen::Matrix<double, DIM_STATE, DIM_STATE> noise =
    Eigen::Matrix<double, DIM_STATE, DIM_STATE>::Zero();
};
} // namespace

void Prepare(const YAML::Node &cfg, const fs::path &output) {
  const fs::path params = cfg["params_file"].as<std::string>();
  const auto geom = sio_utils::LoadStereoImageGeometry(params);
  const auto cam = sio_utils::LoadInferenceCameraIntrinsics(params);
  const auto ids = SourceIds(cfg["dataset_dir"].as<std::string>());
  Require(sio_stereo_gicp::CameraIsValid(cam), "Invalid inference intrinsics");
  std::set<int> wanted, anchors;
  std::multimap<int, int> anchors_by_current;
  for (const auto &pair : cfg["pairs"]) {
    const int a = pair["anchor"].as<int>(), b = pair["current"].as<int>();
    Require(a < b, "Pair anchor must precede current frame");
    wanted.insert(a); wanted.insert(b);
    anchors.insert(a);
    anchors_by_current.emplace(b, a);
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
  const YAML::Node prior_cfg = cfg["prior"];
  const bool has_prior = prior_cfg.IsDefined() && prior_cfg.IsMap();
  integrator.SetNoiseParams(NoiseFromConfig(prior_cfg, has_prior));
  const auto gyro_correction = GyroCorrectionFromConfig(prior_cfg, has_prior);
  const auto covariance_params = CovarianceParamsFromConfig(prior_cfg, has_prior);
  const std::string initial_covariance = has_prior && prior_cfg["initial_covariance"]
    ? prior_cfg["initial_covariance"].as<std::string>() : "legacy";
  Require(initial_covariance == "legacy" || initial_covariance == "observability",
          "prior.initial_covariance must be legacy or observability");
  const bool require_static = has_prior && prior_cfg["require_static_init"]
    ? prior_cfg["require_static_init"].as<bool>() : false;
  ImuInitReport init_report;
  std::map<int, AnchorTrack> tracks;
  std::size_t conditionals = 0;
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
      if (gyro_correction) ApplyGyroCorrection(*gyro_correction, *processed);
      const double time = processed->header.stamp.sec + processed->header.stamp.nanosec * 1e-9;
      Require(!previous_imu_time || (time > previous_imu_time && time - previous_imu_time <= .010001),
              "IMU discontinuity during preparation");
      previous_imu_time = time;
      imu.push_back(processed);
    }
    if (!initialized) {
      if (imu.size() < 200) continue;
      state = integrator.ImuInit(imu, &init_report);
      Require(!require_static || init_report.static_ok,
              "IMU initialization window is not static (gyro deviation " +
              std::to_string(init_report.gyro_max_deviation_rad_s) + " rad/s, |g| " +
              std::to_string(init_report.gravity_norm) + ")");
      if (initial_covariance == "observability")
        state.cov = ImuIntegrator::ObservabilityInitialCovariance(state, init_report, covariance_params);
      // Like the runtime frontend, start the next interval at the image boundary.
      state.timestamp = input->end_time;
      initialized = true;
    } else {
      KeyFrameData anchor;
      anchor.imu_states = state;
      anchor.timestamp = state.timestamp;
      Eigen::Matrix<double, DIM_STATE, DIM_STATE> interval_noise =
        Eigen::Matrix<double, DIM_STATE, DIM_STATE>::Zero();
      Eigen::Matrix<double, DIM_STATE, DIM_STATE> interval_transition =
        Eigen::Matrix<double, DIM_STATE, DIM_STATE>::Identity();
      state = integrator.ImuPreintegrationCov(anchor, imu, input->end_time, &interval_noise,
                                              &interval_transition);
      // Extend every open anchor: Phi <- Phi_k Phi, Q <- Phi_k Q Phi_k^T + Q_k.
      for (auto &entry : tracks) {
        AnchorTrack &track = entry.second;
        const Eigen::Matrix<double, DIM_STATE, DIM_STATE> transition =
          interval_transition * track.transition;
        const Eigen::Matrix<double, DIM_STATE, DIM_STATE> noise =
          interval_transition * track.noise * interval_transition.transpose() + interval_noise;
        track.transition = transition;
        track.noise = 0.5 * (noise + noise.transpose());
      }
    }
    Require(HealthyState(state), "Invalid pure-IMU prior at source frame " + std::to_string(source_id));
    if (anchors.count(source_id)) {
      AnchorTrack track;
      track.anchor_cov = state.cov;
      tracks[source_id] = track;
    }
    const auto paired = anchors_by_current.equal_range(source_id);
    for (auto it = paired.first; it != paired.second; ++it) {
      const auto track = tracks.find(it->second);
      Require(track != tracks.end(), "Anchor frame " + std::to_string(it->second) +
              " precedes IMU initialization");
      AnchorConditional conditional;
      conditional.anchor_id = it->second;
      conditional.current_id = source_id;
      conditional.anchor_covariance = track->second.anchor_cov;
      conditional.transition = track->second.transition;
      conditional.process_noise = track->second.noise;
      conditional.covariance = ConditionOnAnchorPose(
        track->second.anchor_cov, track->second.transition, track->second.noise);
      Require(conditional.covariance.allFinite(), "Invalid anchor-conditioned prior");
      WriteConditional(output, conditional);
      ++conditionals;
    }
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
  Require(conditionals == anchors_by_current.size(), "Missing anchor-conditioned priors");
  const auto &noise = integrator.GetNoiseParams();
  std::ofstream summary(output / "inputs.yaml");
  summary << "format_version: 3\nframes: " << cached << "\nanchor_conditionals: " << conditionals
          << "\nprior: pure_imu_dead_reckoning\ngroundtruth_used: false\n"
          << "depth_provider: " << depth_cfg["provider"].as<std::string>() << '\n'
          << "initial_covariance: " << initial_covariance << '\n'
          << "imu_noise: {gyro_noise_density: " << noise.gyro_noise_density
          << ", accel_noise_density: " << noise.accel_noise_density
          << ", gyro_random_walk: " << noise.gyro_random_walk
          << ", accel_random_walk: " << noise.accel_random_walk << "}\n"
          << "gyro_correction: " << (gyro_correction ? "applied" : "none") << '\n'
          << "init_window: {samples: " << init_report.samples
          << ", duration_s: " << init_report.duration_s
          << ", gravity_norm: " << init_report.gravity_norm
          << ", gyro_max_deviation_rad_s: " << init_report.gyro_max_deviation_rad_s
          << ", static_ok: " << (init_report.static_ok ? "true" : "false") << "}\n";
}
} // namespace fusion_lab

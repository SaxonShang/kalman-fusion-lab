#include "lab.h"
#include "color_icp.h"
#include "sio_pipeline_helpers.h"
#include "utils.h"
#include <chrono>
#include <iostream>
#include <sstream>

namespace fusion_lab {
namespace {
using Clock = std::chrono::steady_clock;
double Milliseconds(const Clock::time_point &start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

Record Base(const YAML::Node &pair, const FrozenFrame &anchor, const FrozenFrame &current) {
  Record r;
  r.name = pair["name"].as<std::string>();
  r.anchor_id = anchor.id; r.current_id = current.id;
  r.anchor_time = anchor.time; r.current_time = current.time;
  r.anchor = anchor.prior; r.prior = current.prior; r.result = current.prior;
  return r;
}

cv::FileStorage Output(const YAML::Node &cfg, const fs::path &file) {
  Require(!fs::exists(file), "Result already exists; choose a new output directory");
  cv::FileStorage out(file.string(), cv::FileStorage::WRITE);
  Require(out.isOpened(), "Cannot write experiment result");
  out << "format_version" << 1 << "cache_key" << cfg["cache_key"].as<std::string>()
      << "depth_provider" << cfg["depth"]["provider"].as<std::string>();
  if (cfg["registration_upstream"])
    out << "registration_commit" << cfg["registration_upstream"]["commit"].as<std::string>();
  return out;
}
} // namespace

void RunGicp(const YAML::Node &cfg, const fs::path &output) {
  const fs::path params = cfg["params_file"].as<std::string>();
  const fs::path cache = cfg["cache_dir"].as<std::string>();
  const auto cam = sio_utils::LoadInferenceCameraIntrinsics(params);
  const auto tbc = sio_utils::LoadImuCameraExtrinsic(params);
  ColorICP registration(params.string());
  const auto settings = cfg["gicp"];
  registration.ConfigureGICP(settings["voxel_m"].as<double>(),
                              settings["max_correspondence_m"].as<double>());
  const YAML::Node solver = settings["stereo_solver"] ? settings["stereo_solver"]
                                                     : YAML::Node(YAML::NodeType::Map);
  registration.ConfigureStereoGicpSolver(solver["gn_rot_eps"].as<double>(1e-6),
    solver["gn_trans_eps"].as<double>(1e-6), solver["cov_floor_trans_m"].as<double>(0),
    solver["cov_floor_rot_deg"].as<double>(0));
  auto out = Output(cfg, output / "observations.yml.gz");
  out << "records" << "[";
  for (const auto &pair : cfg["pairs"]) {
    const auto anchor = ReadFrame(cache, pair["anchor"].as<int>());
    const auto current = ReadFrame(cache, pair["current"].as<int>());
    const Eigen::Matrix4d relative_prior = Pose(anchor.prior).inverse() * Pose(current.prior);
    const auto anchor_body = ToBody(anchor, tbc), current_body = ToBody(current, tbc);
    for (const auto &method_node : settings["methods"]) {
      const auto method = method_node.as<std::string>();
      auto r = Base(pair, anchor, current);
      r.method = method; r.variant = "registration";
      ColorICP::GICPResult result;
      const auto start = Clock::now();
      if (method == "stereo") {
        registration.SetRegistrationType(ColorICP::RegistrationType::STEREO_GICP);
        cv::Mat disparity = anchor.disparity;
        if (disparity.empty()) {
          disparity.create(anchor.depth.size(), CV_32F);
          for (int v = 0; v < disparity.rows; ++v)
            for (int u = 0; u < disparity.cols; ++u)
              disparity.at<float>(v, u) = cam.fx * cam.baseline / anchor.depth.at<float>(v, u);
        }
        result = registration.EstimatePoseStereoGICP(current.camera_cloud, disparity,
          anchor.camera_cloud, cam, tbc, tbc.inverse() * relative_prior * tbc);
        r.observation_covariance = sio_utils::AlignStereoGicpCovarianceToEskfPose(
          result.covariance, anchor.prior.rot_end, result.T.cast<double>());
      } else {
        Require(method == "small_gicp", "Unknown registration method: " + method);
        registration.SetRegistrationType(ColorICP::RegistrationType::GICP);
        result = registration.EstimatePoseGICP(anchor_body, current_body, relative_prior);
        r.observation_covariance = sio_utils::AlignGicpCovarianceToEskfPose(
          result.covariance, anchor.prior.rot_end, result.T.cast<double>());
      }
      r.elapsed_ms = Milliseconds(start);
      const Eigen::Matrix4d observation = Pose(anchor.prior) * result.T.cast<double>();
      r.result.rot_end = observation.block<3, 3>(0, 0);
      r.result.pos_end = observation.block<3, 1>(0, 3);
      r.covariance_native = result.covariance;
      r.covariance_from_hessian = result.covariance_from_hessian;
      r.inliers = result.num_inliers; r.iterations = result.num_iterations;
      r.converged = result.converged;
      r.solver_status = result.status;
      r.covariance_convention = result.covariance_convention ==
        ColorICP::CovarianceConvention::LeftTarget ? "left_target_body" : "right_source_body";
      r.fitness = result.fitness;
      r.registration_rmse = result.rmse;
      r.gicp_trace = result.stereo_trace;
      r.status = "eligible";
      if (r.inliers < settings["min_inliers"].as<int>()) r.status = "too_few_inliers";
      else if (!observation.allFinite()) r.status = "invalid_pose";
      else if (!PositiveCovariance(r.observation_covariance)) r.status = "invalid_covariance";
      else if (!r.covariance_from_hessian) r.status = "fallback_covariance";
      else if (settings["require_convergence"].as<bool>() && !r.converged)
        r.status = "not_converged";
      r.valid = r.status == "eligible";
      WriteRecord(out, r);
      std::cout << "[gicp] " << r.name << ' ' << method << ' ' << r.status
                << " inliers=" << r.inliers << " converged=" << r.converged << '\n';
    }
  }
  out << "]";
}

void RunEskf(const YAML::Node &cfg, const fs::path &output) {
  cv::FileStorage observations(cfg["observations"].as<std::string>(), cv::FileStorage::READ);
  Require(observations.isOpened(), "Frozen GICP observations not found");
  Require(static_cast<std::string>(observations["cache_key"]) == cfg["cache_key"].as<std::string>(),
          "Observation/input cache mismatch");
  auto out = Output(cfg, output / "eskf.yml.gz");
  out << "records" << "[";
  StateEstimator estimator;
  std::vector<std::pair<std::string, double>> variants;
  for (const auto &scale_node : cfg["eskf"]["covariance_scales"]) {
    const auto scale = scale_node.as<double>();
    Require(std::isfinite(scale) && scale > 0, "ESKF covariance scales must be positive");
    std::ostringstream name; name << "R_x" << scale;
    variants.emplace_back(name.str(), scale);
  }
  variants.emplace_back("R_fixed", -1);
  for (const auto &node : observations["records"]) {
    const auto observation = ReadRecord(node);
    for (const auto &variant : variants) {
      auto r = observation;
      r.variant = variant.first; r.result = r.prior; r.elapsed_ms = 0;
      const double prior_scale = cfg["eskf"]["prior_covariance_scale"].as<double>(1.0);
      Require(std::isfinite(prior_scale) && prior_scale > 0, "Prior covariance scale must be positive");
      r.prior.cov *= prior_scale;
      r.result = r.prior;
      if (variant.second > 0) {
        r.observation_covariance *= variant.second;
      } else {
        const double rot = cfg["eskf"]["fixed_rotation_sigma_deg"].as<double>() * M_PI / 180;
        const double pos = cfg["eskf"]["fixed_position_sigma_m"].as<double>();
        Require(rot > 0 && pos > 0, "Fixed observation standard deviations must be positive");
        r.observation_covariance.setZero();
        r.observation_covariance.diagonal().head<3>().setConstant(rot * rot);
        r.observation_covariance.diagonal().tail<3>().setConstant(pos * pos);
      }
      if (observation.valid) {
        const auto start = Clock::now();
        r.result = estimator.ESKF(r.prior, Pose(observation.result).cast<float>(),
                                  r.observation_covariance, nullptr, &r.eskf_debug);
        r.elapsed_ms = Milliseconds(start);
        r.valid = HealthyState(r.result);
        r.status = r.valid ? "updated" : "invalid_posterior";
      } else {
        r.valid = false; r.status = "skipped_" + observation.status;
      }
      WriteRecord(out, r);
    }
  }
  out << "]";
}

void RunEsikf(const YAML::Node &cfg, const fs::path &output) {
  const fs::path params = cfg["params_file"].as<std::string>();
  const fs::path cache = cfg["cache_dir"].as<std::string>();
  const auto cam = sio_utils::LoadInferenceCameraIntrinsics(params);
  const auto tbc = sio_utils::LoadImuCameraExtrinsic(params);
  const auto settings = cfg["esikf"];
  FeatureProcessor features;
  features.SetCameraIntrinsics(cam);
  features.SetBodyFromCameraExtrinsic(tbc.block<3, 3>(0, 0), tbc.block<3, 1>(0, 3));
  StateEstimator estimator;
  auto out = Output(cfg, output / "esikf.yml.gz");
  out << "records" << "[";
  for (const auto &pair : cfg["pairs"]) {
    const auto anchor = ReadFrame(cache, pair["anchor"].as<int>());
    const auto current = ReadFrame(cache, pair["current"].as<int>());
    const auto source_body = ToBody(current, tbc);
    const auto target_body_full = ToBody(anchor, tbc);
    std::vector<Eigen::Matrix3d> source_cov, target_cov_full, target_cov;
    features.ComputePointCovariancesBody(current.camera_cloud, {}, tbc.block<3, 3>(0, 0), source_cov);
    features.ComputePointCovariancesBody(anchor.camera_cloud, {}, tbc.block<3, 3>(0, 0), target_cov_full);
    auto target = pcl::PointCloud<PointTypeRGB>::Ptr(new pcl::PointCloud<PointTypeRGB>());
    std::vector<float> target_conf;
    std::vector<std::size_t> indices;
    Require(sio_pipeline::BuildIndexAlignedTargetCloud(target_body_full, {},
      settings["voxel_m"].as<float>(), target, target_conf, indices, &target_cov_full, &target_cov),
      "Could not build ESIKF target cloud");
    auto tree = std::make_shared<pcl::KdTreeFLANN<PointTypeRGB>>();
    tree->setInputCloud(target);
    pcl::PointCloud<PointTypeNormal>::Ptr with_normals(new pcl::PointCloud<PointTypeNormal>());
    features.EstimateNormals(target, with_normals);
    std::vector<Eigen::Vector3d> gradients, normals;
    features.PrepareColorGradient(with_normals, gradients);
    for (const auto &p : *with_normals) normals.emplace_back(p.normal_x, p.normal_y, p.normal_z);
    for (const auto &weight_node : settings["geometry_weights"]) {
      const auto weight = weight_node.as<double>();
      Require(std::isfinite(weight) && weight >= 0 && weight <= 1, "Invalid ESIKF geometry weight");
      auto r = Base(pair, anchor, current);
      r.method = "esikf";
      const double prior_scale = settings["prior_covariance_scale"].as<double>(1.0);
      Require(std::isfinite(prior_scale) && prior_scale > 0, "Prior covariance scale must be positive");
      r.prior.cov *= prior_scale;
      r.result = r.prior;
      std::ostringstream name; name << "geometry_weight_" << weight; r.variant = name.str();
      const auto start = Clock::now();
      estimator.ESIKF(source_body, tree, gradients, normals, r.result,
        anchor.prior.rot_end, anchor.prior.pos_end, {}, target_conf, weight,
        nullptr, tbc, &r.esikf_debug, settings["voxel_m"].as<double>(), source_cov, target_cov,
        settings["max_correspondence_m"].as<double>(), anchor.depth, cam);
      r.elapsed_ms = Milliseconds(start);
      r.inliers = static_cast<int>(r.esikf_debug.final_corr);
      r.iterations = r.esikf_debug.num_iters; r.converged = r.esikf_debug.converged;
      if (!r.esikf_debug.iterations.empty()) {
        const auto &last = r.esikf_debug.iterations.back();
        r.observation_covariance = last.observation_covariance;
        r.covariance_native = last.observation_covariance;
        r.covariance_from_hessian = true;
      }
      r.valid = r.iterations > 0 && !r.esikf_debug.skipped && HealthyState(r.result) &&
                PositiveCovariance(r.observation_covariance);
      r.status = r.valid ? (r.converged ? "updated" : "updated_not_converged") : "no_valid_update";
      WriteRecord(out, r);
      std::cout << "[esikf] " << r.name << ' ' << r.variant << ' ' << r.status << '\n';
    }
  }
  out << "]";
}
} // namespace fusion_lab

#include "lab.h"
#include "color_icp.h"
#include "imu_integrator.h"
#include "sio_pipeline_helpers.h"
#include "utils.h"
#include <chrono>
#include <cmath>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

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

namespace {
// One ESIKF configuration compared on identical inputs. Defaults reproduce the original filter.
struct EsikfVariant {
  std::string name;
  double geometry_weight = 1.0;
  std::string prior_covariance = "absolute"; // absolute | anchor_conditioned | anchor_relative
  double prior_covariance_scale = 1.0;
  double gyro_scale_sigma = 0.0;
  ESIKFOptions options;
};

std::vector<EsikfVariant> EsikfVariants(const YAML::Node &settings) {
  std::vector<EsikfVariant> variants;
  const double default_scale = settings["prior_covariance_scale"].as<double>(1.0);
  if (!settings["variants"]) {
    // Older configurations: one original-filter variant per geometry weight, same names as before.
    for (const auto &weight_node : settings["geometry_weights"]) {
      EsikfVariant variant;
      variant.geometry_weight = weight_node.as<double>();
      variant.prior_covariance_scale = default_scale;
      std::ostringstream name; name << "geometry_weight_" << variant.geometry_weight;
      variant.name = name.str();
      variants.push_back(variant);
    }
  } else {
    for (const auto &node : settings["variants"]) {
      EsikfVariant variant;
      variant.name = node["name"].as<std::string>();
      variant.geometry_weight = node["geometry_weight"].as<double>(1.0);
      variant.prior_covariance = node["prior_covariance"].as<std::string>("absolute");
      variant.prior_covariance_scale = node["prior_covariance_scale"].as<double>(default_scale);
      variant.gyro_scale_sigma = node["gyro_scale_sigma"].as<double>(0.0);
      auto &o = variant.options;
      o.prior_residual = node["prior_residual"].as<bool>(false);
      o.observation_only = node["observation_only"].as<bool>(false);
      o.stop_rotation_rad = node["stop_rotation_rad"].as<double>(0.0);
      o.stop_translation_m = node["stop_translation_m"].as<double>(0.0);
      o.max_iterations = node["max_iterations"].as<int>(8);
      o.observation_scale_rotation = node["observation_scale_rotation"].as<double>(1.0);
      o.observation_scale_translation = node["observation_scale_translation"].as<double>(1.0);
      o.uniform_luminance = node["uniform_luminance"].as<bool>(false);
      o.attribute_max_distance_m = node["attribute_max_distance_m"].as<double>(0.0);
      o.update_rotation = node["update_rotation"].as<bool>(true);
      const auto residual = node["geometry_residual"].as<std::string>("point_to_point");
      if (residual == "ray") {
        o.geometry_residual = ESIKFOptions::GeometryResidual::Ray;
      } else if (residual == "point_to_plane") {
        o.geometry_residual = ESIKFOptions::GeometryResidual::PointToPlane;
      } else {
        Require(residual == "point_to_point",
                "geometry_residual must be point_to_point, ray or point_to_plane");
      }
      Require(variant.prior_covariance == "absolute" ||
              variant.prior_covariance == "anchor_conditioned" ||
              variant.prior_covariance == "anchor_relative",
              "prior_covariance must be absolute, anchor_conditioned or anchor_relative");
      Require(std::isfinite(variant.prior_covariance_scale) && variant.prior_covariance_scale > 0,
              "Prior covariance scale must be positive");
      Require(variant.gyro_scale_sigma >= 0, "gyro_scale_sigma must be non-negative");
      Require(o.max_iterations > 0, "max_iterations must be positive");
      Require(o.observation_scale_rotation > 0 && o.observation_scale_translation > 0,
              "Observation scales must be positive");
      Require(o.stop_rotation_rad >= 0 && o.stop_translation_m >= 0 &&
              o.attribute_max_distance_m >= 0, "Thresholds must be non-negative");
      variants.push_back(variant);
    }
  }
  std::set<std::string> names;
  for (const auto &variant : variants) {
    Require(std::isfinite(variant.geometry_weight) && variant.geometry_weight >= 0 &&
            variant.geometry_weight <= 1, "Invalid ESIKF geometry weight");
    Require(names.insert(variant.name).second, "Duplicate ESIKF variant name: " + variant.name);
  }
  Require(!variants.empty(), "No ESIKF variants configured");
  return variants;
}
} // namespace

void RunEsikf(const YAML::Node &cfg, const fs::path &output) {
  const fs::path params = cfg["params_file"].as<std::string>();
  const fs::path cache = cfg["cache_dir"].as<std::string>();
  const auto cam = sio_utils::LoadInferenceCameraIntrinsics(params);
  const auto tbc = sio_utils::LoadImuCameraExtrinsic(params);
  const auto settings = cfg["esikf"];
  const auto variants = EsikfVariants(settings);
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
    for (const auto &variant : variants) {
      auto r = Base(pair, anchor, current);
      r.method = "esikf";
      r.variant = variant.name;
      if (variant.prior_covariance == "anchor_conditioned") {
        // Same mean; covariance of the current state given the anchor pose (see Prepare).
        r.prior.cov = ReadConditional(cache, anchor.id, current.id).covariance;
      } else if (variant.prior_covariance == "anchor_relative") {
        // Same mean; covariance of the error against the evaluation reference, which places the
        // true relative motion on the estimated anchor pose (see fusion_report.py).
        const auto conditional = ReadConditional(cache, anchor.id, current.id);
        r.prior.cov = RelativeToAnchorPose(
          conditional.anchor_covariance, conditional.transition, conditional.process_noise,
          AnchorReferenceJacobian(anchor.prior.rot_end, anchor.prior.pos_end,
                                  current.prior.rot_end, current.prior.pos_end));
      }
      r.prior.cov *= variant.prior_covariance_scale;
      AddGyroScaleUncertainty(r.prior.cov, anchor.prior.rot_end, current.prior.rot_end,
                              variant.gyro_scale_sigma);
      r.result = r.prior;
      estimator.SetESIKFOptions(variant.options);
      const auto start = Clock::now();
      estimator.ESIKF(source_body, tree, gradients, normals, r.result,
        anchor.prior.rot_end, anchor.prior.pos_end, {}, target_conf, variant.geometry_weight,
        nullptr, tbc, &r.esikf_debug, settings["voxel_m"].as<double>(), source_cov, target_cov,
        settings["max_correspondence_m"].as<double>(), anchor.depth, cam);
      r.elapsed_ms = Milliseconds(start);
      r.inliers = static_cast<int>(r.esikf_debug.final_corr);
      r.iterations = r.esikf_debug.num_iters; r.converged = r.esikf_debug.converged;
      r.solver_status = r.esikf_debug.status;
      if (!r.esikf_debug.iterations.empty()) {
        const auto &last = r.esikf_debug.iterations.back();
        r.observation_covariance = last.observation_covariance;
        r.covariance_native = last.observation_covariance;
        r.covariance_from_hessian = true;
      }
      r.valid = r.esikf_debug.updated && HealthyState(r.result) &&
                PositiveCovariance(r.observation_covariance);
      r.status = r.valid ? (r.converged ? "updated" : "updated_not_converged") : "no_valid_update";
      WriteRecord(out, r);
      std::cout << "[esikf] " << r.name << ' ' << r.variant << ' ' << r.status
                << " (" << r.solver_status << ")\n";
    }
  }
  out << "]";
}
} // namespace fusion_lab

#include "lab.h"
#include <Eigen/Eigenvalues>
#include <pcl/common/transforms.h>
#include <stdexcept>

namespace fusion_lab {
void Require(bool value, const std::string &message) {
  if (!value) throw std::runtime_error(message);
}

Eigen::Matrix4d Pose(const ImuStatesGroup &state) {
  Eigen::Matrix4d result = Eigen::Matrix4d::Identity();
  result.block<3, 3>(0, 0) = state.rot_end;
  result.block<3, 1>(0, 3) = state.pos_end;
  return result;
}

bool PositiveCovariance(const Mat6 &value) {
  if (!value.allFinite() || !value.isApprox(value.transpose(), 1e-7)) return false;
  const Eigen::SelfAdjointEigenSolver<Mat6> eig(value);
  return eig.info() == Eigen::Success && eig.eigenvalues().minCoeff() > 0;
}

bool HealthyState(const ImuStatesGroup &state) {
  if (!state.rot_end.allFinite() || !state.pos_end.allFinite() || !state.vel_end.allFinite() ||
      !state.bias_g.allFinite() || !state.bias_a.allFinite() || !state.gravity.allFinite() ||
      !state.cov.allFinite() || !state.cov.isApprox(state.cov.transpose(), 1e-7)) return false;
  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, DIM_STATE, DIM_STATE>> eig(state.cov);
  return eig.info() == Eigen::Success && eig.eigenvalues().minCoeff() >= -1e-9;
}

void WriteState(cv::FileStorage &out, const ImuStatesGroup &state) {
  out << "{" << "timestamp" << state.timestamp << "pose" << Cv(Pose(state))
      << "velocity" << Cv(state.vel_end) << "bias_g" << Cv(state.bias_g)
      << "bias_a" << Cv(state.bias_a) << "gravity" << Cv(state.gravity)
      << "covariance" << Cv(state.cov) << "inv_expo_time" << state.inv_expo_time << "}";
}

ImuStatesGroup ReadState(const cv::FileNode &node) {
  ImuStatesGroup state;
  const auto pose = EigenRead<4, 4>(node["pose"]);
  state.timestamp = static_cast<double>(node["timestamp"]);
  state.rot_end = pose.block<3, 3>(0, 0);
  state.pos_end = pose.block<3, 1>(0, 3);
  state.vel_end = EigenRead<3, 1>(node["velocity"]);
  state.bias_g = EigenRead<3, 1>(node["bias_g"]);
  state.bias_a = EigenRead<3, 1>(node["bias_a"]);
  state.gravity = EigenRead<3, 1>(node["gravity"]);
  state.cov = EigenRead<DIM_STATE, DIM_STATE>(node["covariance"]);
  state.inv_expo_time = static_cast<double>(node["inv_expo_time"]);
  return state;
}

void WriteFrame(const fs::path &cache, const FrozenFrame &frame) {
  cv::FileStorage out((cache / ("frame_" + std::to_string(frame.id) + ".yml.gz")).string(),
                       cv::FileStorage::WRITE);
  Require(out.isOpened(), "Cannot create frozen frame");
  cv::Mat points(frame.camera_cloud->size(), 6, CV_32F);
  for (int i = 0; i < points.rows; ++i) {
    const auto &p = (*frame.camera_cloud)[i];
    float *row = points.ptr<float>(i);
    row[0] = p.x; row[1] = p.y; row[2] = p.z;
    row[3] = p.r; row[4] = p.g; row[5] = p.b;
  }
  out << "source_frame_id" << frame.id << "time" << frame.time
      << "points_xyzrgb" << points << "depth_m" << frame.depth
      << "disparity_pixels" << frame.disparity << "prior";
  WriteState(out, frame.prior);
}

FrozenFrame ReadFrame(const fs::path &cache, int id) {
  const auto file = cache / ("frame_" + std::to_string(id) + ".yml.gz");
  cv::FileStorage in(file.string(), cv::FileStorage::READ);
  Require(in.isOpened(), "Frozen input not found: " + file.string());
  FrozenFrame frame;
  frame.id = static_cast<int>(in["source_frame_id"]);
  Require(frame.id == id, "Frozen frame id mismatch");
  frame.time = static_cast<double>(in["time"]);
  frame.prior = ReadState(in["prior"]);
  in["depth_m"] >> frame.depth;
  if (!in["disparity_pixels"].empty()) in["disparity_pixels"] >> frame.disparity;
  cv::Mat points;
  in["points_xyzrgb"] >> points;
  Require(points.cols == 6 && points.type() == CV_32F, "Invalid cached cloud");
  frame.camera_cloud->reserve(points.rows);
  for (int i = 0; i < points.rows; ++i) {
    const auto *row = points.ptr<float>(i);
    PointTypeRGB p;
    p.x = row[0]; p.y = row[1]; p.z = row[2];
    p.r = static_cast<uint8_t>(row[3]); p.g = static_cast<uint8_t>(row[4]);
    p.b = static_cast<uint8_t>(row[5]);
    frame.camera_cloud->push_back(p);
  }
  return frame;
}

pcl::PointCloud<PointTypeRGB>::Ptr ToBody(const FrozenFrame &frame, const Eigen::Matrix4d &t) {
  auto cloud = pcl::PointCloud<PointTypeRGB>::Ptr(new pcl::PointCloud<PointTypeRGB>());
  pcl::transformPointCloud(*frame.camera_cloud, *cloud, t.cast<float>());
  return cloud;
}

void WriteRecord(cv::FileStorage &out, const Record &r) {
  out << "{" << "case" << r.name << "method" << r.method << "variant" << r.variant
      << "status" << r.status << "anchor_id" << r.anchor_id << "current_id" << r.current_id
      << "anchor_time" << r.anchor_time << "current_time" << r.current_time
      << "elapsed_ms" << r.elapsed_ms << "inliers" << r.inliers << "iterations" << r.iterations
      << "converged" << int(r.converged) << "valid" << int(r.valid)
      << "solver_status" << r.solver_status << "covariance_convention" << r.covariance_convention
      << "fitness" << r.fitness << "registration_rmse" << r.registration_rmse
      << "covariance_from_hessian" << int(r.covariance_from_hessian)
      << "covariance_native" << Cv(r.covariance_native)
      << "observation_covariance" << Cv(r.observation_covariance) << "anchor";
  WriteState(out, r.anchor);
  out << "prior"; WriteState(out, r.prior);
  out << "result"; WriteState(out, r.result);
  if (r.variant != "registration" && r.method != "esikf") {
    out << "innovation" << Cv(r.eskf_debug.innovation)
        << "innovation_covariance" << Cv(r.eskf_debug.innovation_covariance)
        << "innovation_solver" << "diagonal_whitening"
        << "innovation_scale" << Cv(r.eskf_debug.innovation_scale)
        << "innovation_covariance_normalized" << Cv(r.eskf_debug.normalized_innovation_covariance)
        << "gain" << Cv(r.eskf_debug.gain) << "correction" << Cv(r.eskf_debug.correction)
        << "damping" << r.eskf_debug.damping << "nis_regularized" << r.eskf_debug.nis_regularized;
  }
  out << "gicp_iterations" << "[";
  for (const auto &it : r.gicp_trace) {
    out << "{" << "iteration" << it.iteration << "inliers" << it.inliers
        << "fitness" << it.fitness << "mahalanobis_cost" << it.mahalanobis_cost
        << "rmse" << it.rmse << "drot_rad" << it.drot_rad << "dtrans_m" << it.dtrans_m
        << "accepted" << int(it.accepted) << "normalized_hessian_min" << it.normalized_hessian_min
        << "normalized_hessian_condition" << it.normalized_hessian_condition
        << "observable_dofs" << it.observable_dofs << "}";
  }
  out << "]" << "esikf_iterations" << "[";
  for (const auto &it : r.esikf_debug.iterations) {
    out << "{" << "iteration" << it.iteration << "correspondences" << int(it.correspondences)
        << "information" << Cv(it.information) << "R" << Cv(it.observation_covariance)
        << "pose_gain" << Cv(it.pose_gain) << "virtual_residual" << Cv(it.virtual_residual)
        << "pose_correction" << Cv(it.pose_correction) << "}";
  }
  out << "]" << "}";
}

Record ReadRecord(const cv::FileNode &n) {
  Record r;
  r.name = static_cast<std::string>(n["case"]);
  r.method = static_cast<std::string>(n["method"]);
  r.status = static_cast<std::string>(n["status"]);
  if (!n["solver_status"].empty()) r.solver_status = static_cast<std::string>(n["solver_status"]);
  if (!n["covariance_convention"].empty()) r.covariance_convention = static_cast<std::string>(n["covariance_convention"]);
  if (!n["fitness"].empty()) r.fitness = static_cast<double>(n["fitness"]);
  if (!n["registration_rmse"].empty()) r.registration_rmse = static_cast<double>(n["registration_rmse"]);
  r.anchor_id = static_cast<int>(n["anchor_id"]);
  r.current_id = static_cast<int>(n["current_id"]);
  r.anchor_time = static_cast<double>(n["anchor_time"]);
  r.current_time = static_cast<double>(n["current_time"]);
  r.valid = static_cast<int>(n["valid"]) != 0;
  r.converged = static_cast<int>(n["converged"]) != 0;
  r.inliers = static_cast<int>(n["inliers"]);
  r.iterations = static_cast<int>(n["iterations"]);
  r.covariance_from_hessian = static_cast<int>(n["covariance_from_hessian"]) != 0;
  r.covariance_native = EigenRead<6, 6>(n["covariance_native"]);
  r.observation_covariance = EigenRead<6, 6>(n["observation_covariance"]);
  r.anchor = ReadState(n["anchor"]);
  r.prior = ReadState(n["prior"]);
  r.result = ReadState(n["result"]);
  return r;
}
} // namespace fusion_lab

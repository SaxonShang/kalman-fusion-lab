#pragma once

#include "sio_frontend_modules.h"
#include <opencv2/core.hpp>
#include <yaml-cpp/yaml.h>
#include <filesystem>
#include <string>

namespace fusion_lab {
namespace fs = std::filesystem;
using Mat6 = Eigen::Matrix<double, 6, 6>;

struct FrozenFrame {
  int id = -1;
  double time = 0;
  ImuStatesGroup prior;
  pcl::PointCloud<PointTypeRGB>::Ptr camera_cloud{new pcl::PointCloud<PointTypeRGB>()};
  cv::Mat depth;
  cv::Mat disparity;
};

struct Record {
  std::string name, method, variant, status, solver_status, covariance_convention;
  int anchor_id = -1, current_id = -1;
  double anchor_time = 0, current_time = 0, elapsed_ms = 0;
  double fitness = 0, registration_rmse = 0;
  int inliers = 0, iterations = 0;
  bool converged = false, valid = false, covariance_from_hessian = false;
  ImuStatesGroup anchor, prior, result;
  Mat6 covariance_native = Mat6::Zero();
  Mat6 observation_covariance = Mat6::Zero();
  ESKFDebugSummary eskf_debug;
  ESIKFDebugSummary esikf_debug;
  std::vector<sio_stereo_gicp::IterationTrace> gicp_trace;
};

void Require(bool value, const std::string &message);
Eigen::Matrix4d Pose(const ImuStatesGroup &state);
template <typename Derived> cv::Mat Cv(const Eigen::MatrixBase<Derived> &value) {
  cv::Mat out(value.rows(), value.cols(), CV_64F);
  for (int r = 0; r < out.rows; ++r)
    for (int c = 0; c < out.cols; ++c) out.at<double>(r, c) = value(r, c);
  return out;
}
template <int Rows, int Cols> Eigen::Matrix<double, Rows, Cols> EigenRead(const cv::FileNode &node) {
  cv::Mat value;
  node >> value;
  Require(value.rows == Rows && value.cols == Cols && value.type() == CV_64F,
          "Invalid cached matrix shape or scalar type");
  Eigen::Matrix<double, Rows, Cols> result;
  for (int r = 0; r < Rows; ++r)
    for (int c = 0; c < Cols; ++c) result(r, c) = value.at<double>(r, c);
  return result;
}
bool PositiveCovariance(const Mat6 &value);
bool HealthyState(const ImuStatesGroup &state);
void WriteState(cv::FileStorage &out, const ImuStatesGroup &state);
ImuStatesGroup ReadState(const cv::FileNode &node);
void WriteFrame(const fs::path &cache, const FrozenFrame &frame);
FrozenFrame ReadFrame(const fs::path &cache, int id);
void WriteRecord(cv::FileStorage &out, const Record &record);
Record ReadRecord(const cv::FileNode &node);
pcl::PointCloud<PointTypeRGB>::Ptr ToBody(const FrozenFrame &frame, const Eigen::Matrix4d &t);
void Prepare(const YAML::Node &config, const fs::path &output);
void RunGicp(const YAML::Node &config, const fs::path &output);
void RunEskf(const YAML::Node &config, const fs::path &output);
void RunEsikf(const YAML::Node &config, const fs::path &output);
} // namespace fusion_lab

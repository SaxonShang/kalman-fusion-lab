#pragma once

#include "data.h"

#include <Eigen/Core>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

struct CorrespondenceFunnelStats {
  std::size_t source_N = 0;
  std::size_t target_N = 0;
  std::size_t knn_hit = 0;
  std::size_t dist_pass = 0;
  std::size_t normal_pass = 0;
};

struct ESKFDebugSummary {
  Eigen::Matrix<double, 6, 1> innovation = Eigen::Matrix<double, 6, 1>::Zero();
  Eigen::Matrix<double, 6, 6> observation_covariance = Eigen::Matrix<double, 6, 6>::Zero();
  Eigen::Matrix<double, 6, 6> innovation_covariance = Eigen::Matrix<double, 6, 6>::Zero();
  Eigen::Matrix<double, 6, 1> innovation_scale = Eigen::Matrix<double, 6, 1>::Ones();
  Eigen::Matrix<double, 6, 6> normalized_innovation_covariance = Eigen::Matrix<double, 6, 6>::Zero();
  Eigen::Matrix<double, DIM_STATE, 6> gain = Eigen::Matrix<double, DIM_STATE, 6>::Zero();
  Eigen::Matrix<double, DIM_STATE, 1> correction = Eigen::Matrix<double, DIM_STATE, 1>::Zero();
  double damping = 0.0; // Dimensionless relative jitter after diagonal whitening.
  double nis_regularized = 0.0;
};

struct ESIKFIterationDebug {
  int iteration = 0;
  std::size_t correspondences = 0;
  Eigen::Matrix<double, 6, 6> information = Eigen::Matrix<double, 6, 6>::Zero();
  Eigen::Matrix<double, 6, 6> observation_covariance = Eigen::Matrix<double, 6, 6>::Zero();
  Eigen::Matrix<double, 6, 6> pose_gain = Eigen::Matrix<double, 6, 6>::Zero();
  Eigen::Matrix<double, 6, 1> virtual_residual = Eigen::Matrix<double, 6, 1>::Zero();
  Eigen::Matrix<double, 6, 1> pose_correction = Eigen::Matrix<double, 6, 1>::Zero();
};

struct ESIKFDebugSummary {
  bool skipped = false;
  bool converged = false;
  int num_iters = 0;
  std::size_t final_corr = 0;
  std::size_t source_N = 0;
  std::size_t target_N = 0;
  std::size_t knn_hit = 0;
  std::size_t dist_pass = 0;
  double max_corr_m = 0.0;
  double pose_delta_m = 0.0;
  double pose_delta_deg = 0.0;
  double virtual_residual_norm = 0.0;
  // Filled only when a summary is requested. One record per completed iteration.
  std::vector<ESIKFIterationDebug> iterations;
};

struct VisualFusionDebugRecord {
  int frame_id = -1;
  double frame_end_ts = 0.0;
  bool is_keyframe = false;
  size_t target_cloud_size = 0;
  size_t source_cloud_size = 0;
  double dynamic_removed_ratio = 0.0;
  double pred_pos_x = 0.0;
  double pred_pos_y = 0.0;
  double pred_pos_z = 0.0;
  double pred_rot_deg = 0.0;
  double fused_pos_x = 0.0;
  double fused_pos_y = 0.0;
  double fused_pos_z = 0.0;
  double fused_rot_deg = 0.0;
  int gicp_inliers = 0;
  int gicp_iters = 0;
  bool gicp_converged = false;
  int gicp_ds_target = 0;
  int gicp_ds_source = 0;
  bool gicp_target_prep_cached = false;
  double gicp_corr_trans_m = 0.0;
  double gicp_corr_rot_deg = 0.0;
  double gicp_cov_trace = 0.0;
  double eskf_innov_trans_m = 0.0;
  double eskf_innov_rot_deg = 0.0;
  double eskf_fused_trans_m = 0.0;
  double eskf_fused_rot_deg = 0.0;
  bool esikf_skipped = false;
  bool esikf_converged = false;
  int esikf_iters = 0;
  double esikf_final_corr = 0.0;
  double esikf_pose_delta_m = 0.0;
  double esikf_pose_delta_deg = 0.0;
  size_t target_cache_size = 0;
};

double RotationAngleDegFromMatrix3d(const Eigen::Matrix3d &R);

void ComputeRelativeTransformDelta(const Eigen::Matrix4d &T_init,
                                   const Eigen::Matrix4d &T_result, double *trans_m,
                                   double *rot_deg);

void ComputeImuStatePoseDelta(const ImuStatesGroup &before, const ImuStatesGroup &after,
                              double *trans_m, double *rot_deg);

void FillImuPriorFields(const ImuStatesGroup &imu, const Eigen::Matrix4d &T_prevkf_curr,
                        VisualFusionDebugRecord *record);

void FillImuFusedFields(const ImuStatesGroup &imu, const Eigen::Matrix4d &T_prevkf_curr,
                        VisualFusionDebugRecord *record);

// T_kf_curr：上一关键帧 body → 当前帧 body（GICP / IMU 先验共用）
std::string FormatMatrix4dPose(const Eigen::Matrix4d &T);

// 6×6 协方差，行主序逐行输出
std::string FormatCovariance6x6(const Eigen::Matrix<double, 6, 6> &cov);

// IMU 误差状态 P 的位姿块 P[0:6,0:6]（δθ, δp）
Eigen::Matrix<double, 6, 6> ExtractImuPoseCovariance(const ImuStatesGroup &imu);

void LogPoseFusionDebug(int frame_id, const Eigen::Matrix4d &T_prior_kf_curr,
                        const Eigen::Matrix4d &T_gicp_kf_curr,
                        const ImuStatesGroup &imu_prior,
                        const Eigen::Matrix<double, 6, 6> &gicp_cov);

class FusionDebugLogger {
public:
  void Open(const std::string &path);
  void WriteRecord(const VisualFusionDebugRecord &record);
  static std::string FormatConsoleLine(const VisualFusionDebugRecord &record);

private:
  void WriteHeader();

  std::unique_ptr<std::ofstream> output_;
  bool header_written_ = false;
  std::mutex mtx_;
};

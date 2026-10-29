#include "fusion_debug.h"

#include "utils.h"

#include <cmath>
#include <filesystem>
#include <iomanip>
#include <sstream>

double RotationAngleDegFromMatrix3d(const Eigen::Matrix3d &R) {
  const double cos_angle =
      std::clamp((R.trace() - 1.0) / 2.0, -1.0, 1.0);
  return std::acos(cos_angle) * 180.0 / M_PI;
}

void ComputeRelativeTransformDelta(const Eigen::Matrix4d &T_init,
                                   const Eigen::Matrix4d &T_result,
                                   double *trans_m, double *rot_deg) {
  if (!trans_m && !rot_deg) {
    return;
  }

  const Eigen::Matrix3d R_init = T_init.block<3, 3>(0, 0);
  const Eigen::Matrix3d R_result = T_result.block<3, 3>(0, 0);
  const Eigen::Matrix3d dR = R_init.transpose() * R_result;
  const Eigen::Vector3d delta_t =
      T_result.block<3, 1>(0, 3) - T_init.block<3, 1>(0, 3);

  if (trans_m) {
    *trans_m = delta_t.norm();
  }
  if (rot_deg) {
    *rot_deg = RotationAngleDegFromMatrix3d(dR);
  }
}

void ComputeImuStatePoseDelta(const ImuStatesGroup &before,
                              const ImuStatesGroup &after, double *trans_m,
                              double *rot_deg) {
  if (trans_m) {
    *trans_m = (after.pos_end - before.pos_end).norm();
  }
  if (rot_deg) {
    const Eigen::Matrix3d dR = before.rot_end.transpose() * after.rot_end;
    *rot_deg = RotationAngleDegFromMatrix3d(dR);
  }
}

void FillImuPriorFields(const ImuStatesGroup &imu,
                        const Eigen::Matrix4d &T_prevkf_curr,
                        VisualFusionDebugRecord *record) {
  if (!record) {
    return;
  }
  record->pred_pos_x = imu.pos_end.x();
  record->pred_pos_y = imu.pos_end.y();
  record->pred_pos_z = imu.pos_end.z();
  record->pred_rot_deg =
      RotationAngleDegFromMatrix3d(T_prevkf_curr.block<3, 3>(0, 0));
}

void FillImuFusedFields(const ImuStatesGroup &imu,
                        const Eigen::Matrix4d &T_prevkf_curr,
                        VisualFusionDebugRecord *record) {
  if (!record) {
    return;
  }
  record->fused_pos_x = imu.pos_end.x();
  record->fused_pos_y = imu.pos_end.y();
  record->fused_pos_z = imu.pos_end.z();
  record->fused_rot_deg =
      RotationAngleDegFromMatrix3d(T_prevkf_curr.block<3, 3>(0, 0));
}

std::string FormatMatrix4dPose(const Eigen::Matrix4d &T) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(6);
  const Eigen::Vector3d t = T.block<3, 1>(0, 3);
  const Eigen::Matrix3d R = T.block<3, 3>(0, 0);
  oss << "t=[" << t.x() << "," << t.y() << "," << t.z() << "] "
      << "R=[" << R(0, 0) << "," << R(0, 1) << "," << R(0, 2) << ";"
      << R(1, 0) << "," << R(1, 1) << "," << R(1, 2) << ";"
      << R(2, 0) << "," << R(2, 1) << "," << R(2, 2) << "]";
  return oss.str();
}

std::string FormatCovariance6x6(const Eigen::Matrix<double, 6, 6> &cov) {
  std::ostringstream oss;
  oss << std::scientific << std::setprecision(4);
  for (int r = 0; r < 6; ++r) {
    if (r > 0) {
      oss << ';';
    }
    oss << '[';
    for (int c = 0; c < 6; ++c) {
      if (c > 0) {
        oss << ',';
      }
      oss << cov(r, c);
    }
    oss << ']';
  }
  return oss.str();
}

Eigen::Matrix<double, 6, 6> ExtractImuPoseCovariance(const ImuStatesGroup &imu) {
  return imu.cov.block<6, 6>(0, 0);
}

void LogPoseFusionDebug(int frame_id, const Eigen::Matrix4d &T_prior_kf_curr,
                        const Eigen::Matrix4d &T_gicp_kf_curr,
                        const ImuStatesGroup &imu_prior,
                        const Eigen::Matrix<double, 6, 6> &gicp_cov) {
  const Eigen::Matrix<double, 6, 6> imu_pose_cov = ExtractImuPoseCovariance(imu_prior);
  std::cout << "[PoseFusion] frame=" << frame_id << " T_prior_kf_curr "
            << FormatMatrix4dPose(T_prior_kf_curr) << '\n'
            << "[PoseFusion] frame=" << frame_id << " T_gicp_kf_curr "
            << FormatMatrix4dPose(T_gicp_kf_curr) << '\n'
            << "[PoseFusion] frame=" << frame_id << " imu_pose_cov_6x6 "
            << FormatCovariance6x6(imu_pose_cov) << '\n'
            << "[PoseFusion] frame=" << frame_id << " gicp_cov_6x6 "
            << FormatCovariance6x6(gicp_cov) << '\n';
}

void FusionDebugLogger::Open(const std::string &path) {
  if (path.empty()) {
    output_.reset();
    header_written_ = false;
    return;
  }

  const std::filesystem::path out_path(path);
  if (out_path.has_parent_path()) {
    std::filesystem::create_directories(out_path.parent_path());
  }

  output_ =
      std::make_unique<std::ofstream>(path, std::ios::out | std::ios::trunc);
  header_written_ = false;
  if (!output_ || !(*output_)) {
    output_.reset();
    return;
  }
  WriteHeader();
}

void FusionDebugLogger::WriteHeader() {
  if (!output_ || header_written_) {
    return;
  }
  (*output_) << "frame_id,frame_end_ts,is_keyframe,target_cloud_size,"
                "source_cloud_size,dynamic_removed_ratio,pred_pos_x,pred_pos_y,"
                "pred_pos_z,pred_rot_deg,fused_pos_x,fused_pos_y,fused_pos_z,"
                "fused_rot_deg,gicp_inliers,gicp_iters,gicp_converged,"
                "gicp_ds_target,gicp_ds_source,gicp_target_prep_cached,"
                "gicp_corr_trans_m,gicp_corr_rot_deg,gicp_cov_trace,"
                "eskf_innov_trans_m,eskf_innov_rot_deg,eskf_fused_trans_m,"
                "eskf_fused_rot_deg,esikf_skipped,esikf_converged,esikf_iters,"
                "esikf_final_corr,esikf_pose_delta_m,esikf_pose_delta_deg,"
                "target_cache_size\n";
  header_written_ = true;
}

void FusionDebugLogger::WriteRecord(const VisualFusionDebugRecord &record) {
  if (!output_) {
    return;
  }
  std::lock_guard<std::mutex> lock(mtx_);
  WriteHeader();
  (*output_) << record.frame_id << ',' << record.frame_end_ts << ','
             << (record.is_keyframe ? 1 : 0) << ','
             << record.target_cloud_size << ',' << record.source_cloud_size
             << ',' << record.dynamic_removed_ratio << ','
             << record.pred_pos_x << ',' << record.pred_pos_y << ','
             << record.pred_pos_z << ',' << record.pred_rot_deg << ','
             << record.fused_pos_x << ',' << record.fused_pos_y << ','
             << record.fused_pos_z << ',' << record.fused_rot_deg << ','
             << record.gicp_inliers << ',' << record.gicp_iters << ','
             << (record.gicp_converged ? 1 : 0) << ','
             << record.gicp_ds_target << ',' << record.gicp_ds_source << ','
             << (record.gicp_target_prep_cached ? 1 : 0) << ','
             << record.gicp_corr_trans_m << ',' << record.gicp_corr_rot_deg
             << ',' << record.gicp_cov_trace << ','
             << record.eskf_innov_trans_m << ',' << record.eskf_innov_rot_deg
             << ',' << record.eskf_fused_trans_m << ','
             << record.eskf_fused_rot_deg << ','
             << (record.esikf_skipped ? 1 : 0) << ','
             << (record.esikf_converged ? 1 : 0) << ',' << record.esikf_iters
             << ',' << record.esikf_final_corr << ','
             << record.esikf_pose_delta_m << ',' << record.esikf_pose_delta_deg
             << ',' << record.target_cache_size << '\n';
  output_->flush();
}

std::string FusionDebugLogger::FormatConsoleLine(
    const VisualFusionDebugRecord &record) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(3);
  oss << "[fusion_debug] frame=" << record.frame_id
      << " kf=" << (record.is_keyframe ? 1 : 0) << " src="
      << record.source_cloud_size << " tgt=" << record.target_cloud_size;
  if (record.is_keyframe) {
    oss << " esikf_skip=" << (record.esikf_skipped ? 1 : 0)
        << " esikf_conv=" << (record.esikf_converged ? 1 : 0)
        << " corr=" << record.esikf_final_corr
        << " delta=" << record.esikf_pose_delta_m << "m/"
        << record.esikf_pose_delta_deg << "deg";
  } else {
    oss << " gicp_inl=" << record.gicp_inliers
        << " gicp_ds=" << record.gicp_ds_target << "/"
        << record.gicp_ds_source
        << " gicp_tgt_cache=" << (record.gicp_target_prep_cached ? 1 : 0)
        << " gicp_corr=" << record.gicp_corr_trans_m << "m/"
        << record.gicp_corr_rot_deg << "deg eskf_innov="
        << record.eskf_innov_trans_m << "m/" << record.eskf_innov_rot_deg
        << "deg";
  }
  return oss.str();
}

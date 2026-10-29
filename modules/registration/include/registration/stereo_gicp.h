#pragma once

// 投影关联 Stereo-GICP：对应按目标像素光线配对，代价用双目三角化协方差。
// 在左目相机系求解 T_target_source，再按需要变到 body 给 ESKF。

#include "data.h"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/core.hpp>
#include <string>
#include <utility>
#include <vector>

namespace sio_stereo_gicp {

struct Calib {
  double fx = 1.0;
  double fy = 1.0;
  double cx = 0.0;
  double cy = 0.0;
  double baseline_m = 0.12;
  int width = 0;
  int height = 0;
};

struct Settings {
  double voxel_size = 0.05;
  bool source_voxel_representative = false; // 可选：每体素取最接近质心的实测点，保留原始像素射线
  double sigma_u = 0.5;
  double sigma_v = 0.5;
  double sigma_d = 0.3;
  double eps_m2 = 0.004 * 0.004; // (4 mm)^2 对角脊，保证 C 可逆
  double zmin = 0.5;
  double zmax = 15.0;
  int max_iteration = 20;
  int search_radius = 1; // 1 → 3×3 窗
  bool disparity_normals = false; // 7x7视差平面线性拟合，分离滤波实现
  double plane_gate_sigma = 0.0; // 0为固定距离消融，正值启用法向标准差门控
  double grazing_cos = 0.0; // 掠射面法向可靠性门槛
  bool point_plane = false; // 实验：单点投影和法向标量残差，默认关闭
  bool joint_geometry = false; // 实验入口，默认保留已有路径
  bool stable_normals = false; // 3x3与7x7法向一致性筛选
  double capture_distance = 0.10; // 投影关联的物理捕获半径，米
  int capture_pixel_cap = 15; // 像素窗半径上限
  double measurement_scale = 1.0; // 0仅作几何消融，1加入完整双目协方差
  double geometry_tangent_sigma = 0.10; // 局部切向关联不确定性，米
  double robust_delta = 0.0; // 白化残差Huber门槛，0关闭
  double normal_cos_gate = 0.0; // 源/目标法向一致性，0关闭
  double z_gate_sigma = 3.0;
  double edge_sigma = 3.0;
  int min_correspondences = 20;
  double gn_rot_eps = 1e-6;
  double gn_trans_eps = 1e-6;
  int num_threads = 1;
};

struct IterationTrace {
  int iteration = 0;
  int inliers = 0;
  double fitness = 0.0;
  double mahalanobis_cost = 0.0;
  double rmse = 0.0;
  double drot_rad = 0.0;
  double dtrans_m = 0.0;
  bool accepted = false;
  double normalized_hessian_min = -1.0; // 对角归一化后的无量纲Hessian诊断
  double normalized_hessian_condition = -1.0;
  int observable_dofs = -1; // 相对特征值1e-8门槛，仅诊断，不代替真实精度验收
};

struct Result {
  Eigen::Matrix4d T_target_source = Eigen::Matrix4d::Identity();
  Eigen::Matrix<double, 6, 6> covariance =
      Eigen::Matrix<double, 6, 6>::Identity() * 1e-2;
  int num_inliers = 0;
  int num_iterations = 0;
  bool converged = false;
  bool covariance_from_hessian = false; // Local diagnostic; does not change the solver.
  double fitness = 0.0;
  double rmse = 0.0;
  double error = 0.0; // 马氏代价 Σ eᵀ W e
  int downsampled_source_points = 0;
  int downsampled_target_points = 0;
  double preprocess_ms = 0.0;
  double align_ms = 0.0;
  std::string status = "ok";
  std::vector<IterationTrace> trace;
};

inline Calib CalibFromCamera(const CameraIntrinsics &cam) {
  Calib calib;
  calib.fx = static_cast<double>(cam.fx);
  calib.fy = static_cast<double>(cam.fy);
  calib.cx = static_cast<double>(cam.cx);
  calib.cy = static_cast<double>(cam.cy);
  calib.baseline_m = static_cast<double>(cam.baseline);
  calib.width = cam.image_width;
  calib.height = cam.image_height;
  return calib;
}

Eigen::Matrix3d Skew(const Eigen::Vector3d &v);
Eigen::Matrix4d Se3Exp(const Eigen::Matrix<double, 6, 1> &xi);

// 一阶立体协方差 Σ = J diag(σ_u²,σ_v²,σ_d²) Jᵀ + εI。
Eigen::Matrix3d PointCovarianceFromXyz(const Eigen::Vector3d &p,
                                       const Calib &calib,
                                       const Settings &settings);

double SigmaZ(double z, const Calib &calib, double sigma_d);

cv::Mat DepthFromDisparity(const cv::Mat &disp, const Calib &calib,
                           double zmin, double zmax);

cv::Mat SplatDepth(const std::vector<Eigen::Vector3d> &points,
                   const Calib &calib, int width, int height, double zmin,
                   double zmax, int max_fill_px = 0);

void VoxelDownsample(const std::vector<Eigen::Vector3d> &points,
                     const std::vector<Eigen::Matrix3d> &covs, double voxel,
                     std::vector<Eigen::Vector3d> *out_points,
                     std::vector<Eigen::Matrix3d> *out_covs,
                     bool use_representative = false);

struct Association {
  std::vector<int> src_idx;
  std::vector<Eigen::Vector3d> q_xyz;
  std::vector<Eigen::Vector2d> uv;
  std::vector<double> z_j;
};

Association ProjectiveAssociate(const std::vector<Eigen::Vector3d> &source_pts,
                                const Eigen::Matrix4d &transform,
                                const cv::Mat &depth, const Calib &calib,
                                const Settings &settings);

Result AlignProjective(const std::vector<Eigen::Vector3d> &source_pts,
                       const cv::Mat &target_depth, const Calib &calib,
                       const Eigen::Matrix4d &init, const Settings &settings);

// T_body = T_imu_cam * T_cam * T_imu_cam^{-1}
Eigen::Matrix4d BodyFromCamera(const Eigen::Matrix4d &T_cam,
                               const Eigen::Matrix4d &T_imu_cam);
Eigen::Matrix4d CameraFromBody(const Eigen::Matrix4d &T_body,
                               const Eigen::Matrix4d &T_imu_cam);
Eigen::Matrix<double, 6, 6> AdjointSE3(const Eigen::Matrix4d &T);
Eigen::Matrix<double, 6, 6>
CovarianceBodyFromCamera(const Eigen::Matrix<double, 6, 6> &cov_cam,
                         const Eigen::Matrix4d &T_imu_cam);

} // namespace sio_stereo_gicp

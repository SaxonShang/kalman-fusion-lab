#include "stereo_gicp.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <pcl/filters/voxel_grid.h>
#include <queue>
#include <utility>

#include <Eigen/Cholesky>
#include <Eigen/Geometry>

namespace sio_stereo_gicp {
namespace {

constexpr float kMinFrontZ = 1e-6f;

/** @brief 点坐标是否均为有限值。 */
bool FinitePoint(const PointTypeRGB &pt) {
  return std::isfinite(pt.x) && std::isfinite(pt.y) && std::isfinite(pt.z);
}

/**
 * @brief 深度图空洞最近邻填充（同步复制 index）。
 * @param depth,index splat 深度与源点下标图
 * @param max_fill_px 最大填充曼哈顿半径 (px)
 */
void FillDepthHoles(cv::Mat &depth, cv::Mat &index, int max_fill_px) {
  if (max_fill_px <= 0 || depth.empty()) {
    return;
  }

  const int height = depth.rows;
  const int width = depth.cols;
  cv::Mat dist(height, width, CV_32SC1, cv::Scalar(std::numeric_limits<int>::max()));
  std::queue<std::pair<int, int>> queue;

  for (int y = 0; y < height; ++y) {
    const float *depth_row = depth.ptr<float>(y);
    int *dist_row = dist.ptr<int>(y);
    for (int x = 0; x < width; ++x) {
      if (std::isfinite(depth_row[x])) {
        dist_row[x] = 0;
        queue.emplace(x, y);
      }
    }
  }

  const int dx[4] = {1, -1, 0, 0};
  const int dy[4] = {0, 0, 1, -1};
  while (!queue.empty()) {
    const auto [x, y] = queue.front();
    queue.pop();
    const int cur = dist.at<int>(y, x);
    if (cur >= max_fill_px) {
      continue;
    }
    const float z = depth.at<float>(y, x);
    const int idx = index.empty() ? -1 : index.at<int>(y, x);
    for (int k = 0; k < 4; ++k) {
      const int nx = x + dx[k];
      const int ny = y + dy[k];
      if (nx < 0 || ny < 0 || nx >= width || ny >= height) {
        continue;
      }
      int &nd = dist.at<int>(ny, nx);
      if (nd <= cur + 1) {
        continue;
      }
      nd = cur + 1;
      depth.at<float>(ny, nx) = z;
      if (!index.empty()) {
        index.at<int>(ny, nx) = idx;
      }
      queue.emplace(nx, ny);
    }
  }
}

} // namespace

bool CameraIsValid(const CameraIntrinsics &camera) {
  return std::isfinite(camera.fx) && camera.fx > 1e-6f && std::isfinite(camera.fy)
         && camera.fy > 1e-6f && std::isfinite(camera.baseline) && camera.baseline > 1e-6f
         && std::isfinite(camera.cx) && std::isfinite(camera.cy);
}

int SearchRadiusPx(double z, double fx, double radius_m) {
  if (!(z > 1e-3) || !(fx > 1e-3) || !(radius_m > 0.0)) {
    return kSearchRadiusMinPx;
  }
  const int radius = static_cast<int>(std::ceil(fx * radius_m / z));
  return std::clamp(radius, kSearchRadiusMinPx, kSearchRadiusMaxPx);
}

double HuberWeight(double mahalanobis_sq, double delta) {
  if (!std::isfinite(mahalanobis_sq) || mahalanobis_sq < 0.0 || !(delta > 0.0)) {
    return 0.0;
  }
  const double thresh = delta * delta;
  if (mahalanobis_sq <= thresh) {
    return 1.0;
  }
  return delta / std::sqrt(mahalanobis_sq);
}

double SigmaZ(double z, const CameraIntrinsics &camera, double sigma_d) {
  if (!CameraIsValid(camera) || !std::isfinite(z)) {
    return std::numeric_limits<double>::infinity();
  }
  return std::abs((z * z)
                  / (static_cast<double>(camera.fx) * static_cast<double>(camera.baseline)))
         * sigma_d;
}

Eigen::Matrix3d PointCovarianceFromXyz(const Eigen::Vector3d &p, const CameraIntrinsics &camera,
                                       double sigma_u, double sigma_v, double sigma_d,
                                       double eps) {
  if (!p.allFinite() || !CameraIsValid(camera) || std::abs(p.z()) < 1e-9) {
    return eps * Eigen::Matrix3d::Identity();
  }

  const double fx = static_cast<double>(camera.fx);
  const double fy = static_cast<double>(camera.fy);
  const double baseline = static_cast<double>(camera.baseline);
  const double z = p.z();
  const double dz_dd = -(z * z) / (fx * baseline);

  Eigen::Matrix3d jac = Eigen::Matrix3d::Zero();
  jac(0, 0) = z / fx;
  jac(1, 1) = z / fy;
  jac(0, 2) = (p.x() / z) * dz_dd;
  jac(1, 2) = (p.y() / z) * dz_dd;
  jac(2, 2) = dz_dd;

  Eigen::Matrix3d meas = Eigen::Matrix3d::Zero();
  meas(0, 0) = sigma_u * sigma_u;
  meas(1, 1) = sigma_v * sigma_v;
  meas(2, 2) = sigma_d * sigma_d;
  Eigen::Matrix3d sigma = jac * meas * jac.transpose();
  sigma += eps * Eigen::Matrix3d::Identity();
  return 0.5 * (sigma + sigma.transpose());
}

cv::Mat DepthFromDisparity(const cv::Mat &disparity, const CameraIntrinsics &camera, float zmin,
                           float zmax) {
  cv::Mat depth;
  if (disparity.empty() || !CameraIsValid(camera)) {
    return depth;
  }

  cv::Mat disp_f;
  if (disparity.type() == CV_32FC1) {
    disp_f = disparity;
  } else {
    disparity.convertTo(disp_f, CV_32FC1);
  }

  depth.create(disp_f.size(), CV_32FC1);
  depth.setTo(std::numeric_limits<float>::quiet_NaN());
  const float fb = camera.fx * camera.baseline;
  for (int y = 0; y < disp_f.rows; ++y) {
    const float *disp_row = disp_f.ptr<float>(y);
    float *depth_row = depth.ptr<float>(y);
    for (int x = 0; x < disp_f.cols; ++x) {
      const float d = disp_row[x];
      if (!std::isfinite(d) || d <= 1e-6f) {
        continue;
      }
      const float z = fb / d;
      if (std::isfinite(z) && z >= zmin && z <= zmax) {
        depth_row[x] = z;
      }
    }
  }
  return depth;
}

SplatMaps SplatTarget(const pcl::PointCloud<PointTypeRGB> &cloud, const CameraIntrinsics &camera,
                      int width, int height, int fill_px, float zmin, float zmax) {
  SplatMaps maps;
  if (width <= 0 || height <= 0 || !CameraIsValid(camera)) {
    return maps;
  }

  maps.depth =
    cv::Mat(height, width, CV_32FC1, cv::Scalar(std::numeric_limits<float>::quiet_NaN()));
  maps.index = cv::Mat(height, width, CV_32SC1, cv::Scalar(-1));

  for (size_t i = 0; i < cloud.size(); ++i) {
    const auto &pt = cloud.points[i];
    if (!FinitePoint(pt) || pt.z < zmin || pt.z > zmax) {
      continue;
    }
    const float u = camera.fx * pt.x / pt.z + camera.cx;
    const float v = camera.fy * pt.y / pt.z + camera.cy;
    const int ui = static_cast<int>(std::lround(u));
    const int vi = static_cast<int>(std::lround(v));
    if (ui < 0 || vi < 0 || ui >= width || vi >= height) {
      continue;
    }
    float &z_ref = maps.depth.at<float>(vi, ui);
    if (!std::isfinite(z_ref) || pt.z < z_ref) {
      z_ref = pt.z;
      maps.index.at<int>(vi, ui) = static_cast<int>(i);
    }
  }

  FillDepthHoles(maps.depth, maps.index, fill_px);
  return maps;
}

std::vector<ProjectiveMatch> ProjectiveAssociate(
  const pcl::PointCloud<PointTypeRGB> &source_cam, const Eigen::Matrix3d &R_target_source,
  const Eigen::Vector3d &t_target_source, const cv::Mat &depth, const CameraIntrinsics &camera,
  const cv::Mat &splat_index, int search_radius, float z_gate_sigma, float edge_sigma,
  float zmin, float zmax) {
  std::vector<ProjectiveMatch> matches;
  if (depth.empty() || depth.type() != CV_32FC1 || !CameraIsValid(camera) || source_cam.empty()) {
    return matches;
  }

  const int height = depth.rows;
  const int width = depth.cols;
  const bool adaptive_radius = search_radius < 0;
  const int max_radius = adaptive_radius ? kSearchRadiusMaxPx : std::max(0, search_radius);
  const int n_off = (2 * max_radius + 1) * (2 * max_radius + 1);
  matches.reserve(source_cam.size());

  std::vector<std::pair<int, int>> offsets;
  offsets.reserve(static_cast<size_t>(n_off));
  int center_k = 0;
  for (int dv = -max_radius; dv <= max_radius; ++dv) {
    for (int du = -max_radius; du <= max_radius; ++du) {
      if (du == 0 && dv == 0) {
        center_k = static_cast<int>(offsets.size());
      }
      offsets.emplace_back(du, dv);
    }
  }

  const bool has_index = !splat_index.empty() && splat_index.size() == depth.size()
                         && splat_index.type() == CV_32SC1;

  for (size_t i = 0; i < source_cam.size(); ++i) {
    const auto &src = source_cam.points[i];
    if (!FinitePoint(src)) {
      continue;
    }
    const Eigen::Vector3d p_src(src.x, src.y, src.z);
    const Eigen::Vector3d p_t = R_target_source * p_src + t_target_source;
    if (!p_t.allFinite() || p_t.z() <= kMinFrontZ) {
      continue;
    }

    const double u = camera.fx * p_t.x() / p_t.z() + camera.cx;
    const double v = camera.fy * p_t.y() / p_t.z() + camera.cy;
    if (!std::isfinite(u) || !std::isfinite(v)) {
      continue;
    }
    const int ui = static_cast<int>(std::lround(u));
    const int vi = static_cast<int>(std::lround(v));
    const int radius = adaptive_radius ? SearchRadiusPx(p_t.z(), camera.fx) : max_radius;

    std::vector<double> z_stack(static_cast<size_t>(n_off),
                                std::numeric_limits<double>::quiet_NaN());
    int n_valid = 0;
    int n_edge = 0;
    double z_hi = -std::numeric_limits<double>::infinity();
    double z_lo = std::numeric_limits<double>::infinity();
    double z_sum = 0.0;
    for (int k = 0; k < n_off; ++k) {
      const int du = offsets[static_cast<size_t>(k)].first;
      const int dv = offsets[static_cast<size_t>(k)].second;
      if (std::max(std::abs(du), std::abs(dv)) > radius) {
        continue;
      }
      const int uu = ui + du;
      const int vv = vi + dv;
      if (uu < 0 || vv < 0 || uu >= width || vv >= height) {
        continue;
      }
      const float z_pix = depth.at<float>(vv, uu);
      if (!std::isfinite(z_pix) || z_pix < zmin || z_pix > zmax) {
        continue;
      }
      z_stack[static_cast<size_t>(k)] = static_cast<double>(z_pix);
      ++n_valid;
      if (std::max(std::abs(du), std::abs(dv)) <= kEdgeRadiusPx) {
        ++n_edge;
        z_hi = std::max(z_hi, static_cast<double>(z_pix));
        z_lo = std::min(z_lo, static_cast<double>(z_pix));
        z_sum += z_pix;
      }
    }
    if (n_valid <= 0) {
      continue;
    }

    const double z_ref = z_sum / static_cast<double>(std::max(n_edge, 1));
    const double sig_ref = std::max(SigmaZ(z_ref, camera), 1e-6);
    if (n_edge > 1 && (z_hi - z_lo) > static_cast<double>(edge_sigma) * sig_ref) {
      continue;
    }

    int best = -1;
    double best_diff = std::numeric_limits<double>::infinity();
    for (int k = 0; k < n_off; ++k) {
      const double z_k = z_stack[static_cast<size_t>(k)];
      if (!std::isfinite(z_k)) {
        continue;
      }
      const double diff = std::abs(z_k - p_t.z());
      if (diff < best_diff) {
        best_diff = diff;
        best = k;
      }
    }

    const double center_z = z_stack[static_cast<size_t>(center_k)];
    if (std::isfinite(center_z)) {
      const double center_sig = std::max(SigmaZ(center_z, camera), 1e-6);
      if (std::abs(p_t.z() - center_z) <= static_cast<double>(z_gate_sigma) * center_sig) {
        best = center_k;
      }
    }
    if (best < 0) {
      continue;
    }

    const double z_j = z_stack[static_cast<size_t>(best)];
    const double gate_sig = std::max(SigmaZ(z_j, camera), 1e-6);
    if (std::abs(p_t.z() - z_j) > static_cast<double>(z_gate_sigma) * gate_sig) {
      continue;
    }

    const int bu = ui + offsets[static_cast<size_t>(best)].first;
    const int bv = vi + offsets[static_cast<size_t>(best)].second;

    ProjectiveMatch match;
    match.src_index = static_cast<int>(i);
    match.p_prime = p_t;
    // q 落在命中像素的光线上，横向残差保留投影差。
    match.uv = Eigen::Vector2d(static_cast<double>(bu), static_cast<double>(bv));
    match.z_j = z_j;
    match.q = Eigen::Vector3d((static_cast<double>(bu) - camera.cx) / camera.fx * z_j,
                              (static_cast<double>(bv) - camera.cy) / camera.fy * z_j, z_j);
    if (has_index && bu >= 0 && bv >= 0 && bu < width && bv < height) {
      match.splat_index = splat_index.at<int>(bv, bu);
    }
    matches.push_back(match);
  }

  return matches;
}

namespace {

/** @brief 向量反对称矩阵 [v]_×。 */
Eigen::Matrix3d Skew(const Eigen::Vector3d &v) {
  Eigen::Matrix3d s;
  s << 0.0, -v.z(), v.y(), v.z(), 0.0, -v.x(), -v.y(), v.x(), 0.0;
  return s;
}

/** @brief SO(3) 指数映射 Exp([w]_×)。 */
Eigen::Matrix3d ExpSO3(const Eigen::Vector3d &w) {
  const double theta = w.norm();
  if (theta < 1e-12) {
    return Eigen::Matrix3d::Identity();
  }
  return Eigen::AngleAxisd(theta, w / theta).toRotationMatrix();
}

/**
 * @brief SE(3) 伴随 Ad_T，用于协方差从相机系推到机体。
 * @param T 4×4 刚体变换
 */
Eigen::Matrix<double, 6, 6> AdjointSE3(const Eigen::Matrix4d &T) {
  const Eigen::Matrix3d R = T.block<3, 3>(0, 0);
  const Eigen::Vector3d t = T.block<3, 1>(0, 3);
  Eigen::Matrix<double, 6, 6> Ad = Eigen::Matrix<double, 6, 6>::Zero();
  Ad.block<3, 3>(0, 0) = R;
  Ad.block<3, 3>(3, 3) = R;
  Ad.block<3, 3>(3, 0) = Skew(t) * R;
  return Ad;
}

/**
 * @brief 体素下采样；叶长≤0 或点数过少时原样返回。
 * @param leaf 叶长 (m)
 */
pcl::PointCloud<PointTypeRGB> VoxelDownsample(const pcl::PointCloud<PointTypeRGB> &cloud,
                                              float leaf) {
  if (leaf <= 0.0f || cloud.size() < 32) {
    return cloud;
  }
  pcl::PointCloud<PointTypeRGB>::Ptr in(new pcl::PointCloud<PointTypeRGB>(cloud));
  pcl::PointCloud<PointTypeRGB> out;
  pcl::VoxelGrid<PointTypeRGB> voxel;
  voxel.setInputCloud(in);
  voxel.setLeafSize(leaf, leaf, leaf);
  voxel.filter(out);
  return out;
}

} // namespace

Eigen::Matrix4d CameraPoseToBody(const Eigen::Matrix4d &T_cam_target_source,
                                 const Eigen::Matrix4d &T_imu_cam) {
  return T_imu_cam * T_cam_target_source * T_imu_cam.inverse();
}

Eigen::Matrix<double, 6, 6> CameraPoseCovToBody(const Eigen::Matrix<double, 6, 6> &cov_cam,
                                                const Eigen::Matrix4d &T_imu_cam) {
  const Eigen::Matrix<double, 6, 6> Ad = AdjointSE3(T_imu_cam);
  Eigen::Matrix<double, 6, 6> cov = Ad * cov_cam * Ad.transpose();
  return 0.5 * (cov + cov.transpose());
}

/**
 * @brief 独立投影 Stereo-GICP：外层重关联 + 内层左扰动 GN（Huber）。
 *
 * 有 target_cam 时只从该点云 splat 深度做投影关联；target_depth 仅作画布尺寸或无点云时的回退。
 * 流程：下采样源云 → 准备目标深度 → 外层迭代（关联 → 累加 H/g → 左乘更新 T）→ 用 H⁻¹ 作协方差。
 */
AlignResult AlignProjective(const pcl::PointCloud<PointTypeRGB> &source_cam,
                            const CameraIntrinsics &camera,
                            const Eigen::Matrix4d &T_target_source_init,
                            const cv::Mat &target_depth,
                            const pcl::PointCloud<PointTypeRGB> *target_cam, int max_iterations,
                            float voxel_leaf) {
  AlignResult out;
  out.T_target_source = T_target_source_init;
  if (!CameraIsValid(camera) || source_cam.empty()) {
    return out;
  }

  // 源点体素下采样，控制关联与 GN 规模。
  const pcl::PointCloud<PointTypeRGB> source = VoxelDownsample(source_cam, voxel_leaf);
  if (source.empty()) {
    return out;
  }

  // 有目标点云时只从它 splat 深度，避免整图视差把动态区域又配回来。
  cv::Mat depth;
  cv::Mat splat_index;
  const bool have_target_cloud = target_cam != nullptr && !target_cam->empty();
  if (have_target_cloud) {
    int width = camera.image_width > 0
                  ? camera.image_width
                  : std::max(1, static_cast<int>(std::ceil(2.0f * camera.cx + 2.0f)));
    int height = camera.image_height > 0
                   ? camera.image_height
                   : std::max(1, static_cast<int>(std::ceil(2.0f * camera.cy + 2.0f)));
    if (!target_depth.empty()) {
      width = target_depth.cols;
      height = target_depth.rows;
    }
    // fill ≈ 0.05 m 在约 3 m 处的像素半径，用于 splat 空洞近邻填充。
    const int fill =
      std::max(1, static_cast<int>(std::ceil(0.05 * static_cast<double>(camera.fx) / 3.0)));
    const SplatMaps splat = SplatTarget(*target_cam, camera, width, height, fill);
    depth = splat.depth;
    splat_index = splat.index;
  } else if (!target_depth.empty() && target_depth.type() == CV_32FC1) {
    depth = target_depth;
  } else {
    return out;
  }

  constexpr int kMinCorrespondences = 20;
  const int iters = std::max(1, max_iterations);
  Eigen::Matrix3d R = out.T_target_source.block<3, 3>(0, 0);
  Eigen::Vector3d t = out.T_target_source.block<3, 1>(0, 3);
  Eigen::Matrix<double, 6, 6> H_last = Eigen::Matrix<double, 6, 6>::Zero();

  // 外层：当前 (R,t) 下重关联，再做一轮左扰动 GN。
  for (int iter = 0; iter < iters; ++iter) {
    const std::vector<ProjectiveMatch> matches =
      ProjectiveAssociate(source, R, t, depth, camera, splat_index);
    if (static_cast<int>(matches.size()) < kMinCorrespondences) {
      break;
    }

    Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
    Eigen::Matrix<double, 6, 1> g = Eigen::Matrix<double, 6, 1>::Zero();
    double cost = 0.0;
    const int match_count = static_cast<int>(matches.size());

    // 内层：对每对对应累加 Huber 加权的法方程 H ξ = -g。
#ifdef _OPENMP
#  pragma omp parallel
    {
      Eigen::Matrix<double, 6, 6> H_thread = Eigen::Matrix<double, 6, 6>::Zero();
      Eigen::Matrix<double, 6, 1> g_thread = Eigen::Matrix<double, 6, 1>::Zero();
      double cost_thread = 0.0;
#  pragma omp for nowait
      for (int i = 0; i < match_count; ++i) {
#else
    for (int i = 0; i < match_count; ++i) {
#endif
        const auto &match = matches[static_cast<size_t>(i)];
        if (match.src_index < 0 || match.src_index >= static_cast<int>(source.size())) {
          continue;
        }
        const auto &src = source.points[static_cast<size_t>(match.src_index)];
        const Eigen::Vector3d p_src(src.x, src.y, src.z);
        const Eigen::Vector3d e = match.p_prime - match.q; // 目标系点到点残差
        // Σ = R Σ_src Rᵀ + Σ_tgt；W = Σ⁻¹ 白化。
        const Eigen::Matrix3d cov =
          R * PointCovarianceFromXyz(p_src, camera) * R.transpose()
          + PointCovarianceFromXyz(match.q, camera, kMatchSigmaUPx, kMatchSigmaVPx, kSigmaDPx);
        const Eigen::LDLT<Eigen::Matrix3d> cov_ldlt(cov);
        if (cov_ldlt.info() != Eigen::Success) {
          continue;
        }
        const Eigen::Matrix3d W = cov_ldlt.solve(Eigen::Matrix3d::Identity());
        if (!W.allFinite()) {
          continue;
        }
        const double mahalanobis_sq = e.dot(W * e);
        const double robust_w = HuberWeight(mahalanobis_sq);
        if (!(robust_w > 0.0)) {
          continue;
        }
        // 左扰动：δp' ≈ [-p'^] ξ_ω + ξ_v。
        Eigen::Matrix<double, 3, 6> J = Eigen::Matrix<double, 3, 6>::Zero();
        J.leftCols<3>() = -Skew(match.p_prime);
        J.rightCols<3>() = Eigen::Matrix3d::Identity();
        const double robust_cost = (mahalanobis_sq <= kHuberDelta * kHuberDelta)
                                     ? mahalanobis_sq
                                     : (2.0 * kHuberDelta * std::sqrt(mahalanobis_sq)
                                        - kHuberDelta * kHuberDelta);
#ifdef _OPENMP
        H_thread.noalias() += robust_w * J.transpose() * W * J;
        g_thread.noalias() += robust_w * J.transpose() * W * e;
        cost_thread += robust_cost;
#else
        H.noalias() += robust_w * J.transpose() * W * J;
        g.noalias() += robust_w * J.transpose() * W * e;
        cost += robust_cost;
#endif
#ifdef _OPENMP
      }
#  pragma omp critical
      {
        H += H_thread;
        g += g_thread;
        cost += cost_thread;
      }
    }
#else
    }
#endif

    if (!H.allFinite() || !g.allFinite()) {
      break;
    }
    // 对称化 + 迹比例阻尼，解 ξ，左乘更新 T ← exp(ξ^) T。
    H_last = 0.5 * (H + H.transpose());
    const double damping = 1e-6 * std::max(1.0, H_last.trace() / 6.0);
    const Eigen::Matrix<double, 6, 6> H_reg =
      H_last + damping * Eigen::Matrix<double, 6, 6>::Identity();
    const Eigen::LDLT<Eigen::Matrix<double, 6, 6>> H_ldlt(H_reg);
    if (H_ldlt.info() != Eigen::Success) {
      break;
    }
    const Eigen::Matrix<double, 6, 1> dx = -H_ldlt.solve(g);
    if (!dx.allFinite()) {
      break;
    }

    const Eigen::Matrix3d dR = ExpSO3(dx.head<3>());
    R = dR * R;
    t = dR * t + dx.tail<3>();
    out.num_inliers = match_count;
    out.num_iterations = iter + 1;
    out.error = cost;
    if (dx.head<3>().norm() < 1e-4 && dx.tail<3>().norm() < 1e-4) {
      out.converged = true;
      break;
    }
  }

  out.T_target_source = Eigen::Matrix4d::Identity();
  out.T_target_source.block<3, 3>(0, 0) = R;
  out.T_target_source.block<3, 1>(0, 3) = t;
  // 末轮信息矩阵 H⁻¹ 作为左扰动协方差（相机目标系）。
  if (H_last.allFinite() && H_last.squaredNorm() > 0.0) {
    const Eigen::LDLT<Eigen::Matrix<double, 6, 6>> H_ldlt(H_last);
    if (H_ldlt.info() == Eigen::Success) {
      out.covariance = H_ldlt.solve(Eigen::Matrix<double, 6, 6>::Identity());
      out.covariance = 0.5 * (out.covariance + out.covariance.transpose());
      out.covariance_from_hessian = true;
    }
  }
  return out;
}

} // namespace sio_stereo_gicp

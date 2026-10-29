#include "stereo_gicp.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <utility>

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

Eigen::Matrix4d CameraPoseToBody(const Eigen::Matrix4d &pose,
                                 const Eigen::Matrix4d &extrinsic) {
  return BodyFromCamera(pose, extrinsic);
}

Eigen::Matrix<double, 6, 6> CameraPoseCovToBody(
    const Eigen::Matrix<double, 6, 6> &covariance,
    const Eigen::Matrix4d &extrinsic) {
  return CovarianceBodyFromCamera(covariance, extrinsic);
}
} // namespace sio_stereo_gicp

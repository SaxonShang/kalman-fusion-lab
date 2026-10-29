#include "registration/stereo_gicp.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <utility>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <opencv2/imgproc.hpp>
#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>

namespace sio_stereo_gicp {
namespace {

constexpr int kMinCorrespondencesFloor = 6;

Eigen::Matrix3d Invert3x3(const Eigen::Matrix3d &a, bool *ok) {
  const double det = a.determinant();
  if (!std::isfinite(det) || std::abs(det) < 1e-18) {
    if (ok) {
      *ok = false;
    }
    return Eigen::Matrix3d::Identity();
  }
  const Eigen::Matrix3d inv = a.inverse();
  if (!inv.allFinite()) {
    if (ok) {
      *ok = false;
    }
    return Eigen::Matrix3d::Identity();
  }
  if (ok) {
    *ok = true;
  }
  return inv;
}

int VoxelCoord(double x, double voxel) {
  return static_cast<int>(std::floor(x / voxel));
}

struct VoxelKey {
  int x = 0;
  int y = 0;
  int z = 0;
  bool operator==(const VoxelKey &o) const {
    return x == o.x && y == o.y && z == o.z;
  }
};

struct VoxelKeyHash {
  std::size_t operator()(const VoxelKey &k) const {
    std::size_t h = 0;
    const auto mix = [&h](int v) {
      h ^= static_cast<std::size_t>(v) + 0x9e3779b9 + (h << 6) + (h >> 2);
    };
    mix(k.x);
    mix(k.y);
    mix(k.z);
    return h;
  }
};

struct VoxelAcc {
  Eigen::Vector3d sum_p = Eigen::Vector3d::Zero();
  Eigen::Matrix3d sum_c = Eigen::Matrix3d::Zero();
  int n = 0;
  std::size_t representative_index = 0;
  double representative_distance = std::numeric_limits<double>::infinity();
};

} // namespace

Eigen::Matrix3d Skew(const Eigen::Vector3d &v) {
  Eigen::Matrix3d s;
  s << 0.0, -v.z(), v.y(), v.z(), 0.0, -v.x(), -v.y(), v.x(), 0.0;
  return s;
}

Eigen::Matrix4d Se3Exp(const Eigen::Matrix<double, 6, 1> &xi) {
  const Eigen::Vector3d w = xi.head<3>();
  const Eigen::Vector3d t = xi.tail<3>();
  const double theta = w.norm();
  Eigen::Matrix3d rot;
  Eigen::Matrix3d v_mat;
  if (theta < 1e-12) {
    rot = Eigen::Matrix3d::Identity() + Skew(w);
    v_mat = Eigen::Matrix3d::Identity();
  } else {
    const Eigen::Vector3d k = w / theta;
    const Eigen::Matrix3d kx = Skew(k);
    rot = Eigen::Matrix3d::Identity() + std::sin(theta) * kx +
          (1.0 - std::cos(theta)) * (kx * kx);
    v_mat = Eigen::Matrix3d::Identity() +
            (1.0 - std::cos(theta)) / theta * kx +
            (theta - std::sin(theta)) / theta * (kx * kx);
  }
  Eigen::Matrix4d out = Eigen::Matrix4d::Identity();
  out.block<3, 3>(0, 0) = rot;
  out.block<3, 1>(0, 3) = v_mat * t;
  return out;
}

double SigmaZ(double z, const Calib &calib, double sigma_d) {
  const double denom = calib.fx * calib.baseline_m;
  if (!(std::isfinite(denom) && std::abs(denom) > 1e-12)) {
    return std::numeric_limits<double>::infinity();
  }
  return std::abs((z * z) / denom) * sigma_d;
}

Eigen::Matrix3d PointCovarianceFromXyz(const Eigen::Vector3d &p,
                                       const Calib &calib,
                                       const Settings &settings) {
  const double x = p.x();
  const double y = p.y();
  const double z = p.z();
  const double z_safe = (std::abs(z) < 1e-9) ? 1e-9 : z;
  const double dz_dd = -(z * z) / (calib.fx * calib.baseline_m);

  Eigen::Matrix3d j = Eigen::Matrix3d::Zero();
  j(0, 0) = z / calib.fx;
  j(1, 1) = z / calib.fy;
  j(0, 2) = (x / z_safe) * dz_dd;
  j(1, 2) = (y / z_safe) * dz_dd;
  j(2, 2) = dz_dd;

  Eigen::Matrix3d meas = Eigen::Matrix3d::Zero();
  meas(0, 0) = settings.sigma_u * settings.sigma_u;
  meas(1, 1) = settings.sigma_v * settings.sigma_v;
  meas(2, 2) = settings.sigma_d * settings.sigma_d;

  Eigen::Matrix3d sigma = j * meas * j.transpose();
  sigma += Eigen::Matrix3d::Identity() * settings.eps_m2;
  return 0.5 * (sigma + sigma.transpose());
}

cv::Mat DepthFromDisparity(const cv::Mat &disp, const Calib &calib,
                           double zmin, double zmax) {
  cv::Mat depth(disp.size(), CV_64FC1, cv::Scalar(std::numeric_limits<double>::quiet_NaN()));
  if (disp.empty()) {
    return depth;
  }
  const double fb = calib.fx * calib.baseline_m;
  for (int v = 0; v < disp.rows; ++v) {
    double *zrow = depth.ptr<double>(v);
    for (int u = 0; u < disp.cols; ++u) {
      double d = 0.0;
      if (disp.type() == CV_32FC1) {
        d = static_cast<double>(disp.at<float>(v, u));
      } else if (disp.type() == CV_64FC1) {
        d = disp.at<double>(v, u);
      } else {
        d = static_cast<double>(disp.at<float>(v, u));
      }
      if (!(d > 1e-6)) {
        continue;
      }
      const double z = fb / d;
      if (std::isfinite(z) && z >= zmin && z <= zmax) {
        zrow[u] = z;
      }
    }
  }
  return depth;
}

cv::Mat SplatDepth(const std::vector<Eigen::Vector3d> &points,
                   const Calib &calib, int width, int height, double zmin,
                   double zmax, int max_fill_px) {
  cv::Mat depth(height, width, CV_64FC1,
                cv::Scalar(std::numeric_limits<double>::quiet_NaN()));
  if (points.empty() || width <= 0 || height <= 0) {
    return depth;
  }

  struct Hit {
    int u;
    int v;
    double z;
  };
  std::vector<Hit> hits;
  hits.reserve(points.size());
  for (const auto &p : points) {
    if (!(p.z() > 1e-6) || !p.allFinite()) {
      continue;
    }
    if (p.z() < zmin || p.z() > zmax) {
      continue;
    }
    const double u = calib.fx * p.x() / p.z() + calib.cx;
    const double v = calib.fy * p.y() / p.z() + calib.cy;
    const int ui = static_cast<int>(std::lround(u));
    const int vi = static_cast<int>(std::lround(v));
    if (ui < 0 || ui >= width || vi < 0 || vi >= height) {
      continue;
    }
    hits.push_back({ui, vi, p.z()});
  }
  std::sort(hits.begin(), hits.end(),
            [](const Hit &a, const Hit &b) { return a.z > b.z; });
  for (const auto &h : hits) {
    depth.at<double>(h.v, h.u) = h.z;
  }

  if (max_fill_px > 0) {
    cv::Mat filled = depth.clone();
    const int r = max_fill_px;
    for (int v = 0; v < height; ++v) {
      for (int u = 0; u < width; ++u) {
        if (std::isfinite(depth.at<double>(v, u))) {
          continue;
        }
        double best_z = std::numeric_limits<double>::quiet_NaN();
        int best_d2 = r * r + 1;
        for (int dv = -r; dv <= r; ++dv) {
          const int vv = v + dv;
          if (vv < 0 || vv >= height) {
            continue;
          }
          for (int du = -r; du <= r; ++du) {
            const int uu = u + du;
            if (uu < 0 || uu >= width) {
              continue;
            }
            const double z = depth.at<double>(vv, uu);
            if (!std::isfinite(z)) {
              continue;
            }
            const int d2 = du * du + dv * dv;
            if (d2 < best_d2) {
              best_d2 = d2;
              best_z = z;
            }
          }
        }
        if (std::isfinite(best_z)) {
          filled.at<double>(v, u) = best_z;
        }
      }
    }
    return filled;
  }
  return depth;
}

void VoxelDownsample(const std::vector<Eigen::Vector3d> &points,
                     const std::vector<Eigen::Matrix3d> &covs, double voxel,
                     std::vector<Eigen::Vector3d> *out_points,
                     std::vector<Eigen::Matrix3d> *out_covs,
                     bool use_representative) {
  out_points->clear();
  out_covs->clear();
  if (points.empty()) {
    return;
  }
  if (voxel <= 0.0) {
    *out_points = points;
    *out_covs = covs;
    return;
  }

  std::unordered_map<VoxelKey, VoxelAcc, VoxelKeyHash> acc;
  acc.reserve(points.size() / 4 + 1);
  for (std::size_t i = 0; i < points.size(); ++i) {
    const auto &p = points[i];
    if (!p.allFinite()) {
      continue;
    }
    VoxelKey key{VoxelCoord(p.x(), voxel), VoxelCoord(p.y(), voxel),
                 VoxelCoord(p.z(), voxel)};
    VoxelAcc &a = acc[key];
    a.sum_p += p;
    if (i < covs.size()) {
      a.sum_c += covs[i];
    } else {
      a.sum_c += Eigen::Matrix3d::Zero();
    }
    a.n += 1;
  }
  if (use_representative) {
    // 质心可能不位于任何实测像素射线上。第二遍选择同体素内距质心
    // 最近的原始点；等距离时保留输入中先出现者，结果可复现。
    for (std::size_t i = 0; i < points.size(); ++i) {
      const auto &p = points[i];
      if (!p.allFinite()) continue;
      const VoxelKey key{VoxelCoord(p.x(), voxel), VoxelCoord(p.y(), voxel),
                         VoxelCoord(p.z(), voxel)};
      VoxelAcc &a = acc.at(key);
      const double distance = (p - a.sum_p / a.n).squaredNorm();
      if (distance < a.representative_distance) {
        a.representative_distance = distance;
        a.representative_index = i;
      }
    }
  }
  out_points->reserve(acc.size());
  out_covs->reserve(acc.size());
  for (const auto &kv : acc) {
    const VoxelAcc &a = kv.second;
    if (a.n <= 0) {
      continue;
    }
    if (use_representative) {
      out_points->push_back(points[a.representative_index]);
      // 选择实测点时协方差也随该点选择，不能继续返回体素平均协方差。
      out_covs->push_back(a.representative_index < covs.size()
                             ? covs[a.representative_index]
                             : Eigen::Matrix3d::Zero());
    } else {
      out_points->push_back(a.sum_p / static_cast<double>(a.n));
      out_covs->push_back(a.sum_c / static_cast<double>(a.n));
    }
  }
}

Association ProjectiveAssociate(const std::vector<Eigen::Vector3d> &source_pts,
                                const Eigen::Matrix4d &transform,
                                const cv::Mat &depth, const Calib &calib,
                                const Settings &settings) {
  Association out;
  if (source_pts.empty() || depth.empty() || depth.type() != CV_64FC1) {
    return out;
  }
  const int height = depth.rows;
  const int width = depth.cols;
  const int radius = std::max(0, settings.search_radius);
  const int n_off = (2 * radius + 1) * (2 * radius + 1);
  const Eigen::Matrix3d rot = transform.block<3, 3>(0, 0);
  const Eigen::Vector3d trans = transform.block<3, 1>(0, 3);
  const int n = static_cast<int>(source_pts.size());

  std::vector<int> keep_idx;
  keep_idx.reserve(static_cast<std::size_t>(n) / 2);
  std::vector<Eigen::Vector3d> q_keep;
  std::vector<Eigen::Vector2d> uv_keep;
  std::vector<double> z_keep;
  q_keep.reserve(keep_idx.capacity());
  uv_keep.reserve(keep_idx.capacity());
  z_keep.reserve(keep_idx.capacity());

  std::vector<std::pair<int, int>> offsets;
  offsets.reserve(static_cast<std::size_t>(n_off));
  int center_k = 0;
  for (int dv = -radius; dv <= radius; ++dv) {
    for (int du = -radius; du <= radius; ++du) {
      if (du == 0 && dv == 0) {
        center_k = static_cast<int>(offsets.size());
      }
      offsets.emplace_back(du, dv);
    }
  }

#ifdef _OPENMP
  const int assoc_threads = std::max(1, settings.num_threads);
#endif

#ifdef _OPENMP
#pragma omp parallel num_threads(assoc_threads)
#endif
  {
    std::vector<int> th_keep;
    std::vector<Eigen::Vector3d> th_q;
    std::vector<Eigen::Vector2d> th_uv;
    std::vector<double> th_z;
    std::vector<double> z_stack(static_cast<std::size_t>(n_off),
                                std::numeric_limits<double>::quiet_NaN());
#ifdef _OPENMP
#pragma omp for nowait schedule(static)
#endif
    for (int i = 0; i < n; ++i) {
      const Eigen::Vector3d p_t = rot * source_pts[static_cast<std::size_t>(i)] + trans;
      if (!(p_t.z() > 1e-6) || !p_t.allFinite()) {
        continue;
      }
      const double u = calib.fx * p_t.x() / p_t.z() + calib.cx;
      const double v = calib.fy * p_t.y() / p_t.z() + calib.cy;
      if (!std::isfinite(u) || !std::isfinite(v)) {
        continue;
      }
      const int ui = static_cast<int>(std::lround(u));
      const int vi = static_cast<int>(std::lround(v));

      std::fill(z_stack.begin(), z_stack.end(),
                std::numeric_limits<double>::quiet_NaN());
      int n_valid = 0;
      double z_hi = -std::numeric_limits<double>::infinity();
      double z_lo = std::numeric_limits<double>::infinity();
      double z_sum = 0.0;
      for (int k = 0; k < n_off; ++k) {
        const int uu = ui + offsets[static_cast<std::size_t>(k)].first;
        const int vv = vi + offsets[static_cast<std::size_t>(k)].second;
        if (uu < 0 || uu >= width || vv < 0 || vv >= height) {
          continue;
        }
        const double z_pix = depth.at<double>(vv, uu);
        if (!std::isfinite(z_pix) || z_pix < settings.zmin ||
            z_pix > settings.zmax) {
          continue;
        }
        z_stack[static_cast<std::size_t>(k)] = z_pix;
        n_valid += 1;
        z_hi = std::max(z_hi, z_pix);
        z_lo = std::min(z_lo, z_pix);
        z_sum += z_pix;
      }
      if (n_valid <= 0) {
        continue;
      }
      const double z_ref = z_sum / static_cast<double>(n_valid);
      const double sig =
          std::max(SigmaZ(z_ref, calib, settings.sigma_d), 1e-6);
      if (n_valid > 1 && (z_hi - z_lo) > settings.edge_sigma * sig) {
        continue;
      }

      int best = 0;
      double best_diff = std::numeric_limits<double>::infinity();
      for (int k = 0; k < n_off; ++k) {
        const double z_pix = z_stack[static_cast<std::size_t>(k)];
        if (!std::isfinite(z_pix)) {
          continue;
        }
        const double diff = std::abs(z_pix - p_t.z());
        if (diff < best_diff) {
          best_diff = diff;
          best = k;
        }
      }
      const double center_z = z_stack[static_cast<std::size_t>(center_k)];
      if (std::isfinite(center_z)) {
        const double center_sig =
            std::max(SigmaZ(center_z, calib, settings.sigma_d), 1e-6);
        if (std::abs(p_t.z() - center_z) <= settings.z_gate_sigma * center_sig) {
          best = center_k;
        }
      }
      const double z_j = z_stack[static_cast<std::size_t>(best)];
      if (!std::isfinite(z_j)) {
        continue;
      }
      const double gate_sig =
          std::max(SigmaZ(z_j, calib, settings.sigma_d), 1e-6);
      if (std::abs(p_t.z() - z_j) > settings.z_gate_sigma * gate_sig) {
        continue;
      }

      // 必须沿连续 (u,v) 反投影，避免整数像素中心引入假横向残差。
      const Eigen::Vector3d q((u - calib.cx) / calib.fx * z_j,
                              (v - calib.cy) / calib.fy * z_j, z_j);
      th_keep.push_back(i);
      th_q.push_back(q);
      th_uv.emplace_back(u, v);
      th_z.push_back(z_j);
    }
#ifdef _OPENMP
#pragma omp critical
#endif
    {
      keep_idx.insert(keep_idx.end(), th_keep.begin(), th_keep.end());
      q_keep.insert(q_keep.end(), th_q.begin(), th_q.end());
      uv_keep.insert(uv_keep.end(), th_uv.begin(), th_uv.end());
      z_keep.insert(z_keep.end(), th_z.begin(), th_z.end());
    }
  }

  out.src_idx = std::move(keep_idx);
  out.q_xyz = std::move(q_keep);
  out.uv = std::move(uv_keep);
  out.z_j = std::move(z_keep);
  return out;
}

namespace {

void LeftJacAndResidual(const std::vector<Eigen::Vector3d> &source_pts,
                        const std::vector<int> &src_idx,
                        const std::vector<Eigen::Vector3d> &q_xyz,
                        const Eigen::Matrix4d &transform,
                        std::vector<Eigen::Vector3d> *residual,
                        std::vector<Eigen::Matrix<double, 3, 6>> *jac) {
  const Eigen::Matrix3d rot = transform.block<3, 3>(0, 0);
  const Eigen::Vector3d trans = transform.block<3, 1>(0, 3);
  const int k = static_cast<int>(src_idx.size());
  residual->resize(static_cast<std::size_t>(k));
  jac->resize(static_cast<std::size_t>(k));
  for (int i = 0; i < k; ++i) {
    const Eigen::Vector3d p_t =
        rot * source_pts[static_cast<std::size_t>(src_idx[static_cast<std::size_t>(i)])] +
        trans;
    (*residual)[static_cast<std::size_t>(i)] = p_t - q_xyz[static_cast<std::size_t>(i)];
    Eigen::Matrix<double, 3, 6> j = Eigen::Matrix<double, 3, 6>::Zero();
    j(0, 1) = p_t.z();
    j(0, 2) = -p_t.y();
    j(1, 0) = -p_t.z();
    j(1, 2) = p_t.x();
    j(2, 0) = p_t.y();
    j(2, 1) = -p_t.x();
    j(0, 3) = 1.0;
    j(1, 4) = 1.0;
    j(2, 5) = 1.0;
    (*jac)[static_cast<std::size_t>(i)] = j;
  }
}

bool InvertCovariances(const std::vector<Eigen::Matrix3d> &cov,
                       std::vector<Eigen::Matrix3d> *weight) {
  weight->resize(cov.size());
  bool all_ok = true;
  for (std::size_t i = 0; i < cov.size(); ++i) {
    bool inv_ok = false;
    (*weight)[i] = Invert3x3(cov[i], &inv_ok);
    all_ok = all_ok && inv_ok;
  }
  return all_ok;
}

double MahalanobisCost(const std::vector<Eigen::Vector3d> &residual,
                       const std::vector<Eigen::Matrix3d> &weight) {
  double cost = 0.0;
  const int k = static_cast<int>(residual.size());
  for (int i = 0; i < k; ++i) {
    const Eigen::Vector3d &e = residual[static_cast<std::size_t>(i)];
    cost += e.dot(weight[static_cast<std::size_t>(i)] * e);
  }
  return cost;
}

bool MahalanobisSystem(const std::vector<Eigen::Vector3d> &residual,
                       const std::vector<Eigen::Matrix<double, 3, 6>> &jac,
                       const std::vector<Eigen::Matrix3d> &weight,
                       Eigen::Matrix<double, 6, 6> *hess,
                       Eigen::Matrix<double, 6, 1> *grad, double *cost) {
  *hess = Eigen::Matrix<double, 6, 6>::Zero();
  *grad = Eigen::Matrix<double, 6, 1>::Zero();
  *cost = 0.0;
  const int k = static_cast<int>(residual.size());
  for (int i = 0; i < k; ++i) {
    const Eigen::Matrix3d &w = weight[static_cast<std::size_t>(i)];
    const Eigen::Matrix<double, 3, 6> &j = jac[static_cast<std::size_t>(i)];
    const Eigen::Vector3d &e = residual[static_cast<std::size_t>(i)];
    const Eigen::Vector3d we = w * e;
    *hess += j.transpose() * w * j;
    *grad += j.transpose() * we;
    *cost += e.dot(we);
  }
  *hess = 0.5 * (*hess + hess->transpose());
  return hess->allFinite() && grad->allFinite() && std::isfinite(*cost);
}

Eigen::Matrix<double, 6, 6>
PoseCovarianceFromHessian(const Eigen::Matrix<double, 6, 6> &hess) {
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> eig(
      0.5 * (hess + hess.transpose()));
  Eigen::Matrix<double, 6, 1> evals = eig.eigenvalues();
  for (int i = 0; i < 6; ++i) {
    evals(i) = std::max(evals(i), 1e-12);
  }
  Eigen::Matrix<double, 6, 6> cov =
      eig.eigenvectors() * evals.cwiseInverse().asDiagonal() *
      eig.eigenvectors().transpose();
  return 0.5 * (cov + cov.transpose());
}

} // namespace

namespace {
// 在固定像素邻域拟合局部表面；先做三维距离与平面性筛选，避免直接跨深度边缘求差分。
std::vector<Eigen::Vector3d> SurfaceNormals(const cv::Mat &depth, const Calib &k, bool stable=false) {
  std::vector<Eigen::Vector3d> normals(depth.total(),Eigen::Vector3d::Zero());
  auto point = [&](int u,int v) -> Eigen::Vector3d {
    const double z=depth.at<double>(v,u);
    return {(u-k.cx)*z/k.fx,(v-k.cy)*z/k.fy,z};
  };
  for(int v=3;v<depth.rows-3;++v) for(int u=3;u<depth.cols-3;++u) {
    const auto center=point(u,v);
    if(!center.allFinite() || center.z()<0.5 || center.z()>15.0) continue;
    Eigen::Vector3d sum=Eigen::Vector3d::Zero();Eigen::Matrix3d square=Eigen::Matrix3d::Zero();int count=0;
    for(int dv=-3;dv<=3;++dv) for(int du=-3;du<=3;++du) {
      const auto p=point(u+du,v+dv);
      if(!p.allFinite() || p.z()<0.5 || (p-center).squaredNorm()>0.15*0.15)continue;
      sum+=p;square+=p*p.transpose();++count;
    }
    if(count<8)continue;
    const Eigen::Vector3d mean=sum/count;
    const Eigen::Matrix3d cov=square/count-mean*mean.transpose();
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(cov);
    if(eig.info()!=Eigen::Success || eig.eigenvalues()[1]<1e-8 || eig.eigenvalues()[0]>0.15*eig.eigenvalues().sum())continue;
    if(stable) {
      Eigen::Vector3d local_sum=Eigen::Vector3d::Zero();Eigen::Matrix3d local_square=Eigen::Matrix3d::Zero();int local_count=0;
      for(int dy=-1;dy<=1;++dy)for(int dx=-1;dx<=1;++dx) {
        const auto q=point(u+dx,v+dy);
        if(!q.allFinite() || q.z()<0.5 || (q-center).squaredNorm()>0.15*0.15)continue;
        local_sum+=q;local_square+=q*q.transpose();++local_count;
      }
      if(local_count<5)continue;
      const Eigen::Vector3d m=local_sum/local_count;
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> small(local_square/local_count-m*m.transpose());
      if(small.info()!=Eigen::Success || small.eigenvalues()[1]<1e-8 ||
          std::abs(small.eigenvectors().col(0).dot(eig.eigenvectors().col(0)))<0.9659258262890683)continue;
    }
    normals[v*depth.cols+u]=eig.eigenvectors().col(0);
  }
  return normals;
}
// 平面逆深度（视差）在像素上是仿射函数。完整7x7支持的最小二乘
// 通过可分离滤波一次算好，避免每像素反复做PCA；不跨无效深度填洞。
std::vector<Eigen::Vector3d> DisparityNormals(const cv::Mat &depth,const Calib &k,const Settings &s) {
  cv::Mat disp=cv::Mat::zeros(depth.size(),CV_64F),valid=cv::Mat::zeros(depth.size(),CV_64F);
  for(int y=0;y<depth.rows;++y)for(int x=0;x<depth.cols;++x) {
    const double z=depth.at<double>(y,x);
    if(std::isfinite(z) && z>=s.zmin && z<=s.zmax) {disp.at<double>(y,x)=k.fx*k.baseline_m/z;valid.at<double>(y,x)=1.;}
  }
  cv::Mat avg=cv::Mat::ones(7,1,CV_64F)/7.,derivative(7,1,CV_64F);
  for(int i=0;i<7;++i)derivative.at<double>(i)=(i-3)/28.;
  cv::Mat mean,a,b,square,support;
  cv::sepFilter2D(disp,mean,-1,avg,avg);
  cv::sepFilter2D(disp,a,-1,derivative,avg);
  cv::sepFilter2D(disp,b,-1,avg,derivative);
  cv::boxFilter(disp.mul(disp),square,-1,cv::Size(7,7));
  cv::boxFilter(valid,support,-1,cv::Size(7,7));
  std::vector<Eigen::Vector3d> normals(depth.total(),Eigen::Vector3d::Zero());
  for(int y=3;y<depth.rows-3;++y)for(int x=3;x<depth.cols-3;++x) {
    if(support.at<double>(y,x)<.999999)continue;
    const double d=mean.at<double>(y,x),du=a.at<double>(y,x),dv=b.at<double>(y,x);
    // 去除斜率解释的方差后检查平面拟合残差，拒绝明显跨表面支持。
    const double noise=std::max(0.,square.at<double>(y,x)-d*d-4*(du*du+dv*dv));
    if(noise>0.5*0.5)continue;
    Eigen::Vector3d n(k.fx*du,k.fy*dv,d-du*(x-k.cx)-dv*(y-k.cy));
    if(n.allFinite() && n.norm()>1e-10) {
      normals[y*depth.cols+x]=n.normalized();
    }
  }
  return normals;
}
Eigen::Vector3d NormalAt(const Eigen::Vector3d &p,const std::vector<Eigen::Vector3d>& normals,const cv::Mat &depth,const Calib &k) {
  if(p.z()<=0 || normals.empty())return Eigen::Vector3d::Zero();
  const int u=std::lround(k.fx*p.x()/p.z()+k.cx),v=std::lround(k.fy*p.y()/p.z()+k.cy);
  if(u<0 || v<0 || u>=depth.cols || v>=depth.rows)return Eigen::Vector3d::Zero();
  return normals[v*depth.cols+u];
}
Eigen::Matrix3d SurfaceCovariance(const Eigen::Vector3d &n,double sigma) {
  if(n.squaredNorm()<0.5)return sigma*sigma*Eigen::Matrix3d::Identity();
  return sigma*sigma*(Eigen::Matrix3d::Identity()-n*n.transpose())+0.003*0.003*n*n.transpose();
}
// 以10cm物理搜索范围换算像素窗，保留投影关联；用真实命中像素三维点计算距离。
Association AssociateSurfaces(const std::vector<Eigen::Vector3d> &source,const Eigen::Matrix4d &T,
    const cv::Mat &depth,const Calib &k,const Settings &s,
    const std::vector<Eigen::Vector3d>& target_normals,const std::vector<Eigen::Vector3d>& source_normals) {
  Association a;const Eigen::Matrix3d R=T.block<3,3>(0,0);const Eigen::Vector3d t=T.block<3,1>(0,3);
  for(size_t i=0;i<source.size();++i) {
    const Eigen::Vector3d p=R*source[i]+t;if(!p.allFinite() || p.z()<s.zmin)continue;
    const int u=std::lround(k.fx*p.x()/p.z()+k.cx),v=std::lround(k.fy*p.y()/p.z()+k.cy);
    const int radius=s.point_plane ? 0 : std::clamp(static_cast<int>(std::ceil(k.fx*s.capture_distance/p.z())),2,s.capture_pixel_cap);
    double best=s.capture_distance*s.capture_distance;int bu=-1,bv=-1;Eigen::Vector3d bq;
    const Eigen::Vector3d ns=R*source_normals[i];
    for(int y=std::max(0,v-radius);y<=std::min(depth.rows-1,v+radius);++y)
      for(int x=std::max(0,u-radius);x<=std::min(depth.cols-1,u+radius);++x) {
        const double z=depth.at<double>(y,x);if(!std::isfinite(z)||z<s.zmin||z>s.zmax)continue;
        const auto &n=target_normals[y*depth.cols+x];if(n.squaredNorm()<0.5)continue;
        if(s.normal_cos_gate>0 && ns.squaredNorm()>0.5 && std::abs(ns.dot(n))<s.normal_cos_gate)continue;
        const Eigen::Vector3d q((x-k.cx)*z/k.fx,(y-k.cy)*z/k.fy,z);
        const double d=(p-q).squaredNorm();
        if(s.point_plane && s.grazing_cos>0 && std::abs(n.dot(q.normalized()))<s.grazing_cos)continue;
        if(s.point_plane && s.plane_gate_sigma>0) {
          // 门控固定使用法向测量标准差，与代价加权消融独立。
          // 不在大量候选中挑最小噪声残差。
          const Eigen::Matrix3d C=R*PointCovarianceFromXyz(source[i],k,s)*R.transpose()+PointCovarianceFromXyz(q,k,s);
          const double variance=std::max(s.eps_m2,s.eps_m2+n.dot(C*n));
          if(std::abs(n.dot(p-q))>s.plane_gate_sigma*std::sqrt(variance))continue;
          best=std::numeric_limits<double>::infinity();
        }
        if(d<best){best=d;bu=x;bv=y;bq=q;}
      }
    if(bu<0)continue;
    a.src_idx.push_back(i);a.q_xyz.push_back(bq);a.uv.emplace_back(bu,bv);a.z_j.push_back(bq.z());
  }
  return a;
}
} // namespace

Result AlignProjective(const std::vector<Eigen::Vector3d> &source_pts_in,
                       const cv::Mat &target_depth_in, const Calib &calib,
                       const Eigen::Matrix4d &init, const Settings &settings) {
  cv::Mat target_depth=target_depth_in;
  Result result;
  if(settings.point_plane)result.T_target_source=init;
  const auto t_pre0 = std::chrono::steady_clock::now();

  if (source_pts_in.size() <
          static_cast<std::size_t>(settings.min_correspondences) ||
      target_depth.empty()) {
    result.status = "empty_input";
    return result;
  }

  // 先体素再算 Σ：前端稠密点约 40 万，体素后才是配准规模。
  std::vector<Eigen::Matrix3d> empty_covs;
  std::vector<Eigen::Vector3d> source_pts;
  std::vector<Eigen::Matrix3d> ignored_covs;
  VoxelDownsample(source_pts_in, empty_covs, settings.voxel_size, &source_pts,
                  &ignored_covs, settings.source_voxel_representative);
  std::vector<Eigen::Matrix3d> cov_src(source_pts.size());
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(std::max(1, settings.num_threads))
#endif
  for (int i = 0; i < static_cast<int>(source_pts.size()); ++i) {
    cov_src[static_cast<std::size_t>(i)] = PointCovarianceFromXyz(
        source_pts[static_cast<std::size_t>(i)], calib, settings);
  }
  result.downsampled_source_points = static_cast<int>(source_pts.size());
  result.downsampled_target_points = target_depth.rows * target_depth.cols;
  if (source_pts.size() <
      static_cast<std::size_t>(settings.min_correspondences)) {
    result.status = "too_few_source";
    return result;
  }

  std::vector<Eigen::Vector3d> target_normals,source_normals(source_pts.size(),Eigen::Vector3d::Zero());
  if(settings.joint_geometry || settings.point_plane) {
    if(settings.disparity_normals) {
      target_normals=DisparityNormals(target_depth,calib,settings);
    } else target_normals=SurfaceNormals(target_depth,calib,settings.stable_normals);
    if(!settings.point_plane) {
    const cv::Mat source_depth=SplatDepth(source_pts_in,calib,target_depth.cols,target_depth.rows,settings.zmin,settings.zmax,0);
    const auto dense_source_normals=SurfaceNormals(source_depth,calib,settings.stable_normals);
    for(size_t i=0;i<source_pts.size();++i)source_normals[i]=NormalAt(source_pts[i],dense_source_normals,source_depth,calib);
    }
  }
  const auto t_pre1 = std::chrono::steady_clock::now();
  result.preprocess_ms =
      std::chrono::duration<double, std::milli>(t_pre1 - t_pre0).count();

  const auto t_al0 = std::chrono::steady_clock::now();
  Eigen::Matrix4d transform = init;
  double damp = 1e-3;
  const int n_src = static_cast<int>(source_pts.size());
  const int min_corr =
      std::max(kMinCorrespondencesFloor, settings.min_correspondences);

  auto associate = [&](const Eigen::Matrix4d &T) {
    return (settings.joint_geometry || settings.point_plane) ? AssociateSurfaces(source_pts,T,target_depth,calib,settings,target_normals,source_normals) :
        ProjectiveAssociate(source_pts,T,target_depth,calib,settings);
  };
  for (int iter = 0; iter < settings.max_iteration; ++iter) {
    const Association assoc =
        associate(transform);
    const int n_in = static_cast<int>(assoc.src_idx.size());
    if (n_in < min_corr) {
      result.status = "too_few_correspondences";
      result.num_inliers = n_in;
      result.num_iterations = iter;
      break;
    }
    result.fitness = static_cast<double>(n_in) / static_cast<double>(n_src);
    result.num_inliers = n_in;

    const Eigen::Matrix3d rot = transform.block<3, 3>(0, 0);
    std::vector<Eigen::Matrix3d> cov(static_cast<std::size_t>(n_in));
    for (int i = 0; i < n_in; ++i) {
      const int si = assoc.src_idx[static_cast<std::size_t>(i)];
      const Eigen::Matrix3d cov_i =
          rot * cov_src[static_cast<std::size_t>(si)] * rot.transpose();
      const Eigen::Vector3d q = assoc.q_xyz[static_cast<std::size_t>(i)];
      const Eigen::Matrix3d cov_j = PointCovarianceFromXyz(q, calib, settings);
      cov[static_cast<std::size_t>(i)] = cov_i + cov_j;
      if(settings.joint_geometry && !settings.point_plane) {
        const Eigen::Vector3d nt=NormalAt(q,target_normals,target_depth,calib);
        const Eigen::Vector3d ns=source_normals[static_cast<std::size_t>(si)];
        cov[static_cast<std::size_t>(i)] = settings.measurement_scale*(cov_i+cov_j) +
            rot*SurfaceCovariance(ns,settings.geometry_tangent_sigma)*rot.transpose()+
            SurfaceCovariance(nt,settings.geometry_tangent_sigma);
      }
    }
    std::vector<Eigen::Matrix3d> weight;
    if(settings.point_plane) {
      // 标量点面代价等价于秩一信息矩阵；切向不施加伪约束。
      // 当前外层的法向和测量方差冻结，LM试探不重新改变权重。
      weight.resize(n_in);
      for(int i=0;i<n_in;++i) {
        const Eigen::Vector3d n=NormalAt(assoc.q_xyz[i],target_normals,target_depth,calib);
        const double variance=std::max(settings.eps_m2, settings.eps_m2+
            settings.measurement_scale*n.dot(cov[i]*n));
        weight[i]=n*n.transpose()/variance;
      }
    } else if (!InvertCovariances(cov, &weight)) {
      result.status = "singular_system";
      result.num_iterations = iter + 1;
      break;
    }

    std::vector<Eigen::Vector3d> residual;
    std::vector<Eigen::Matrix<double, 3, 6>> jac;
    LeftJacAndResidual(source_pts, assoc.src_idx, assoc.q_xyz, transform,
                       &residual, &jac);
    // 固定对应关系内进行IRLS：Huber降权进入H/g及本次LM比较，下一轮重新计算。
    if((settings.joint_geometry || settings.point_plane) && settings.robust_delta>0) {
      for(size_t i=0;i<residual.size();++i) {
        const double whitened=std::sqrt(std::max(0.0,residual[i].dot(weight[i]*residual[i])));
        if(whitened>settings.robust_delta)weight[i]*=settings.robust_delta/whitened;
      }
    }
    double sse = 0.0;
    for (const auto &e : residual) {
      sse += e.squaredNorm();
    }
    result.rmse = std::sqrt(sse / static_cast<double>(n_in));

    Eigen::Matrix<double, 6, 6> hess;
    Eigen::Matrix<double, 6, 1> grad;
    double cost = 0.0;
    if (!MahalanobisSystem(residual, jac, weight, &hess, &grad, &cost)) {
      result.status = "singular_system";
      result.num_iterations = iter + 1;
      break;
    }
    result.error = cost;
    result.covariance = PoseCovarianceFromHessian(hess);
    result.covariance_from_hessian = true;

    double scale = hess.diagonal().mean();
    if (!std::isfinite(scale) || scale <= 0.0) {
      scale = 1.0;
    }

    bool accepted = false;
    Eigen::Matrix<double, 6, 1> dxi = Eigen::Matrix<double, 6, 1>::Zero();
    for (int attempt = 0; attempt < 8; ++attempt) {
      Eigen::Matrix<double, 6, 6> lhs =
          hess + (damp * scale) * Eigen::Matrix<double, 6, 6>::Identity();
      Eigen::LDLT<Eigen::Matrix<double, 6, 6>> ldlt(lhs);
      if (ldlt.info() != Eigen::Success) {
        damp *= 4.0;
        continue;
      }
      dxi = -ldlt.solve(grad);
      if (!dxi.allFinite()) {
        damp *= 4.0;
        continue;
      }
      const Eigen::Matrix4d trial = Se3Exp(dxi) * transform;
      std::vector<Eigen::Vector3d> residual_t;
      std::vector<Eigen::Matrix<double, 3, 6>> jac_unused;
      LeftJacAndResidual(source_pts, assoc.src_idx, assoc.q_xyz, trial,
                         &residual_t, &jac_unused);
      const double cost_t = MahalanobisCost(residual_t, weight);
      if (!std::isfinite(cost_t) || cost_t >= cost) {
        damp *= 4.0;
        continue;
      }
      transform = trial;
      damp = std::max(damp * 0.3, 1e-8);
      accepted = true;
      break;
    }

    IterationTrace tr;
    if(settings.point_plane) {
      // 先按对角归一化再看特征值，避免直接混合米和弧度的单位尺度。
      const Eigen::Matrix<double,6,1> scale_diag=hess.diagonal().cwiseMax(1e-20).cwiseSqrt().cwiseInverse();
      const Eigen::Matrix<double,6,6> normalized=scale_diag.asDiagonal()*hess*scale_diag.asDiagonal();
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double,6,6>> eig(normalized);
      if(eig.info()==Eigen::Success) {
        const auto ev=eig.eigenvalues();
        tr.normalized_hessian_min=ev[0];
        tr.normalized_hessian_condition=ev[5]/std::max(1e-20,ev[0]);
        tr.observable_dofs=(ev.array()>std::max(1e-12,ev[5]*1e-8)).count();
      }
    }
    tr.iteration = iter;
    tr.inliers = n_in;
    tr.fitness = result.fitness;
    tr.mahalanobis_cost = result.error;
    tr.rmse = result.rmse;
    tr.drot_rad = dxi.head<3>().norm();
    tr.dtrans_m = dxi.tail<3>().norm();
    tr.accepted = accepted;
    result.trace.push_back(tr);
    result.num_iterations = iter + 1;
    result.T_target_source = transform;

    if (!accepted) {
      // 旧路径保留历史行为；点面候选不能把LM拒绝当作收敛证据。
      result.converged = !settings.point_plane && result.num_inliers >= min_corr;
      result.status = result.converged ? "stopped_at_minimum" : "lm_rejected";
      break;
    }
    if (tr.drot_rad < settings.gn_rot_eps &&
        tr.dtrans_m < settings.gn_trans_eps) {
      const Association assoc_final =
          associate(transform);
      result.num_inliers = static_cast<int>(assoc_final.src_idx.size());
      if (result.num_inliers >= min_corr) {
        std::vector<Eigen::Vector3d> residual_f;
        std::vector<Eigen::Matrix<double, 3, 6>> jac_f;
        LeftJacAndResidual(source_pts, assoc_final.src_idx, assoc_final.q_xyz,
                           transform, &residual_f, &jac_f);
        double sse_f = 0.0;
        for (const auto &e : residual_f) {
          sse_f += e.squaredNorm();
        }
        result.rmse =
            std::sqrt(sse_f / static_cast<double>(result.num_inliers));
        result.fitness = static_cast<double>(result.num_inliers) /
                         static_cast<double>(n_src);
      }
      result.converged = !settings.point_plane || result.num_inliers >= min_corr;
      result.status = result.converged ? "converged" : "too_few_correspondences";
      break;
    }
  }

  if (!result.converged && result.status == "ok") {
    result.status = (result.num_iterations >= settings.max_iteration)
                        ? "max_iter"
                        : result.status;
    result.converged = !settings.point_plane && result.num_inliers >= min_corr;
  }
  result.T_target_source = transform;
  const auto t_al1 = std::chrono::steady_clock::now();
  result.align_ms =
      std::chrono::duration<double, std::milli>(t_al1 - t_al0).count();
  return result;
}

Eigen::Matrix4d BodyFromCamera(const Eigen::Matrix4d &T_cam,
                               const Eigen::Matrix4d &T_imu_cam) {
  return T_imu_cam * T_cam * T_imu_cam.inverse();
}

Eigen::Matrix4d CameraFromBody(const Eigen::Matrix4d &T_body,
                               const Eigen::Matrix4d &T_imu_cam) {
  return T_imu_cam.inverse() * T_body * T_imu_cam;
}

Eigen::Matrix<double, 6, 6> AdjointSE3(const Eigen::Matrix4d &T) {
  const Eigen::Matrix3d r = T.block<3, 3>(0, 0);
  const Eigen::Vector3d t = T.block<3, 1>(0, 3);
  Eigen::Matrix<double, 6, 6> ad = Eigen::Matrix<double, 6, 6>::Zero();
  ad.block<3, 3>(0, 0) = r;
  ad.block<3, 3>(3, 3) = r;
  ad.block<3, 3>(3, 0) = Skew(t) * r;
  return ad;
}

Eigen::Matrix<double, 6, 6>
CovarianceBodyFromCamera(const Eigen::Matrix<double, 6, 6> &cov_cam,
                         const Eigen::Matrix4d &T_imu_cam) {
  const Eigen::Matrix<double, 6, 6> ad = AdjointSE3(T_imu_cam);
  Eigen::Matrix<double, 6, 6> cov = ad * cov_cam * ad.transpose();
  return 0.5 * (cov + cov.transpose());
}

} // namespace sio_stereo_gicp

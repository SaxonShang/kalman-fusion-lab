#include "sio_frontend_modules.h"

#include <Eigen/Dense>
#include <algorithm>
#include <numeric>
#include <pcl/features/normal_3d.h>
#include <pcl/search/kdtree.h>
#include <thread>
#include <pcl/common/io.h> 

#ifdef _OPENMP
#  include <omp.h>
#endif

// 利用PCL提供的NormalEstimation计算点云法向量
void FeatureProcessor::EstimateNormals(const pcl::PointCloud<PointTypeRGB>::Ptr &cloud,
                                       pcl::PointCloud<PointTypeNormal>::Ptr &cloud_out) {
  if (!cloud || cloud->empty())
    return;

  if (!cloud_out) {
    cloud_out.reset(new pcl::PointCloud<PointTypeNormal>());
  }

  pcl::NormalEstimation<PointTypeRGB, PointTypeNormal> ne;
  ne.setInputCloud(cloud);

  pcl::search::KdTree<PointTypeRGB>::Ptr tree(new pcl::search::KdTree<PointTypeRGB>());
  ne.setSearchMethod(tree);

  // 计算法向量：设置搜索半径或 K 近邻
  ne.setRadiusSearch(0.05); // 根据实际场景调整
  ne.compute(*cloud_out);

  // 将原始点云的 XYZRGB 拷贝到含有 Normal 的点云中
  pcl::copyPointCloud(*cloud, *cloud_out);
}

// 简化版（PointTypeRGB，无法向）：k 近邻空间差分
void FeatureProcessor::PrepareColorGradient(const pcl::PointCloud<PointTypeRGB>::Ptr &target,
                                            std::vector<Eigen::Vector3d> &target_color_gradient) {
  if (!target || target->empty())
    return;

  const int cloud_size = static_cast<int>(target->points.size());
  target_color_gradient.assign(cloud_size, Eigen::Vector3d::Zero());

  pcl::KdTreeFLANN<PointTypeRGB> kdtree;
  kdtree.setInputCloud(target);
  constexpr int K = 6;
  std::vector<int> nn_idx(K);
  std::vector<float> nn_sq(K);

  // 亮度权重（NTSC）
  const Eigen::Vector3d w(0.299, 0.587, 0.114);

  for (int i = 0; i < cloud_size; ++i) {
    if (kdtree.nearestKSearch(target->points[i], K, nn_idx, nn_sq) < 2)
      continue;
    const Eigen::Vector3d pi(target->points[i].x, target->points[i].y, target->points[i].z);
    const double Ii =
      Eigen::Vector3d(target->points[i].r, target->points[i].g, target->points[i].b).dot(w) / 255.0;
    Eigen::Vector3d grad = Eigen::Vector3d::Zero();
    double weight_sum = 0.0;
    for (int j = 1; j < static_cast<int>(nn_idx.size()); ++j) {
      const int nb = nn_idx[j];
      const Eigen::Vector3d pj(target->points[nb].x, target->points[nb].y, target->points[nb].z);
      const double Ij =
        Eigen::Vector3d(target->points[nb].r, target->points[nb].g, target->points[nb].b).dot(w)
        / 255.0;
      const double dist2 = static_cast<double>(nn_sq[j]);
      if (dist2 < 1e-8)
        continue;
      grad += (Ij - Ii) / dist2 * (pj - pi);
      weight_sum += 1.0 / dist2;
    }
    if (weight_sum > 0.0)
      target_color_gradient[i] = grad / weight_sum;
  }
}

// 完整版（PointTypeNormal，含法向）：Open3D Colored ICP 切平面最小二乘梯度

void FeatureProcessor::PrepareColorGradient(const pcl::PointCloud<PointTypeNormal>::Ptr &target,
                                            std::vector<Eigen::Vector3d> &target_color_gradient) {
  if (!target || target->empty())
    return;

  const int cloud_size = static_cast<int>(target->points.size());
  target_color_gradient.assign(cloud_size, Eigen::Vector3d::Zero());

  // 亮度权重（均匀）
  const Eigen::Vector3d w = Eigen::Vector3d::Ones() / 3.0;

  // 预计算各点亮度
  std::vector<double> intensity(cloud_size);
  for (int i = 0; i < cloud_size; ++i)
    intensity[i] =
      Eigen::Vector3d(target->points[i].r, target->points[i].g, target->points[i].b).dot(w) / 255.0;

  // 每线程独立 KD-Tree
  const int num_threads = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
  std::vector<pcl::search::KdTree<PointTypeNormal>::Ptr> kdtrees(num_threads);
  for (int t = 0; t < num_threads; ++t) {
    kdtrees[t].reset(new pcl::search::KdTree<PointTypeNormal>());
    kdtrees[t]->setInputCloud(target);
  }

  constexpr double search_radius = 0.05;
  constexpr int min_neighbors = 5;
  constexpr int max_neighbors = 15;

#ifdef _OPENMP
#  pragma omp parallel for num_threads(num_threads) schedule(static)
#endif
  for (int i = 0; i < cloud_size; ++i) {
    int tid = 0;
#ifdef _OPENMP
    tid = omp_get_thread_num();
#endif
    const Eigen::Vector3d p = target->points[i].getVector3fMap().cast<double>();
    const Eigen::Vector3d np = target->points[i].getNormalVector3fMap().cast<double>();
    const double ip = intensity[i];

    std::vector<int> nn_idx;
    std::vector<float> nn_sq;
    kdtrees[tid]->radiusSearch(target->points[i], search_radius, nn_idx, nn_sq,
                               static_cast<unsigned int>(max_neighbors));
    const int nn_size = static_cast<int>(nn_idx.size());
    if (nn_size < min_neighbors)
      continue;

    Eigen::Matrix3d M = Eigen::Matrix3d::Zero();
    Eigen::Vector3d rhs = Eigen::Vector3d::Zero();
    for (int k = 1; k < nn_size; ++k) {
      const Eigen::Vector3d pp = target->points[nn_idx[k]].getVector3fMap().cast<double>();
      const double ipp = intensity[nn_idx[k]];
      // 投影到切平面
      const Eigen::Vector3d v = (pp - (pp - p).dot(np) * np) - p;
      M.noalias() += v * v.transpose();
      rhs.noalias() += v * (ipp - ip); // 亮度差作为残差，乘以切平面投影向量作为雅可比
    }
    const double ww = static_cast<double>(nn_size - 1);
    M.noalias() += (ww * ww) * (np * np.transpose());
    target_color_gradient[i] = M.ldlt().solve(rhs);
  }
}

// 估计点云几何和颜色协方差
void FeatureProcessor::EstimatePointCovariances(
  const pcl::PointCloud<PointTypeRGB>::Ptr &cloud,
  const std::vector<Eigen::Vector3d> &target_color_gradient,
  std::vector<Eigen::Matrix3d> &geo_covariances, std::vector<double> &color_variances,
  const std::vector<float> &point_confidences) const {

  geo_covariances.clear();
  color_variances.clear();
  if (!cloud || cloud->empty()) {
    return;
  }

  ComputePointCovariancesBody(cloud, point_confidences, Eigen::Matrix3d::Identity(),
                              geo_covariances);

  color_variances.resize(cloud->size(), 1.0);
  for (size_t i = 0; i < cloud->size(); ++i) {
    if (i < target_color_gradient.size()) {
      const Eigen::Vector3d &grad = target_color_gradient[i];
      const double grad_norm = grad.norm();
      if (grad_norm > 1e-3 && i < geo_covariances.size()) {
        color_variances[i] = grad.dot(geo_covariances[i] * grad) / (grad_norm * grad_norm);
        continue;
      }
    }
    if (i < geo_covariances.size()) {
      color_variances[i] = geo_covariances[i].trace() / 3.0;
    }
  }
}

// 供给源点云在目标点云KDTree中查找最近邻点阀值匹配
void FeatureProcessor::FindCorrespondences(const pcl::PointCloud<PointTypeRGB>::Ptr &source,
                                           const pcl::KdTreeFLANN<PointTypeRGB>::Ptr &target_kdtree,
                                           std::vector<int> &match_indices, float max_dist_sq,
                                           CorrespondenceFunnelStats *funnel_out) {
  if (!source || !target_kdtree)
    return;

  match_indices.assign(source->points.size(), -1);
  CorrespondenceFunnelStats funnel;
  funnel.source_N = source->points.size();
  const auto target_cloud = target_kdtree->getInputCloud();
  funnel.target_N = target_cloud ? target_cloud->size() : 0;

  int actual_threads = 1;
#ifdef _OPENMP
  actual_threads = std::max(1, omp_get_max_threads());
#endif

#ifdef _OPENMP
  std::vector<std::size_t> knn_hits(actual_threads, 0);
  std::vector<std::size_t> dist_passes(actual_threads, 0);
#  pragma omp parallel num_threads(actual_threads)
#endif
  {
    std::vector<int> pointIdxNKNSearch(1);
    std::vector<float> pointNKNSquaredDistance(1);

#ifdef _OPENMP
    const int thread_id = omp_get_thread_num();
#  pragma omp for schedule(static) // 每线程独立计算最近邻点，避免数据竞争
#endif
    for (size_t i = 0; i < source->points.size(); ++i) {
      if (target_kdtree->nearestKSearch(source->points[i], 1, pointIdxNKNSearch,
                                        pointNKNSquaredDistance)
          > 0) {
#ifdef _OPENMP
        ++knn_hits[thread_id];
#else
        ++funnel.knn_hit;
#endif
        if (pointNKNSquaredDistance[0] < max_dist_sq) {
          match_indices[i] = pointIdxNKNSearch[0];
#ifdef _OPENMP
          ++dist_passes[thread_id];
#else
          ++funnel.dist_pass;
#endif
        }
      }
    }
  }

#ifdef _OPENMP
  funnel.knn_hit = std::accumulate(knn_hits.begin(), knn_hits.end(), std::size_t{0});
  funnel.dist_pass = std::accumulate(dist_passes.begin(), dist_passes.end(), std::size_t{0});
#endif

  if (funnel_out) {
    *funnel_out = funnel;
  }
}

void FeatureProcessor::ComputePointCovariancesBody(
  const pcl::PointCloud<PointTypeRGB>::Ptr &cloud_cam, const std::vector<float> &confidences,
  const Eigen::Matrix3d &R_body_cam, std::vector<Eigen::Matrix3d> &out_covariances) const {

  out_covariances.clear();
  if (!cloud_cam)
    return;
  const int cloud_size = static_cast<int>(cloud_cam->size());
  out_covariances.resize(cloud_size);

  int actual_threads = 1;
#ifdef _OPENMP
  actual_threads = std::max(1, omp_get_max_threads()); // 根据系统硬件并行度自动调整线程数
#endif

#ifdef _OPENMP
#  pragma omp parallel for num_threads(actual_threads)                                             \
    schedule(static) // 每线程独立计算点云协方差，避免数据竞争
#endif
  for (int i = 0; i < cloud_size; ++i) {
    const auto &pt = cloud_cam->points[i];
    const float conf = (i < static_cast<int>(confidences.size())) ? confidences[i] : 1.0f;
    double local_trace_mean = 0.0;

    // 相机坐标系下计算协方差
    Eigen::Matrix3d cov_cam =
      ComputeCameraPointCovariance(Eigen::Vector3d(pt.x, pt.y, pt.z), pt.z, conf, local_trace_mean);

    // 转换到本体坐标系
    out_covariances[i] = R_body_cam * cov_cam * R_body_cam.transpose();
  }
}

Eigen::Matrix3d FeatureProcessor::ComputeCameraPointCovariance(const Eigen::Vector3d &pc, double z,
                                                               double confidence,
                                                               double &out_trace_mean) const {

  constexpr double kMinPointVar = 1e-4; // 最小位置方差下限（m²）
  const double src_conf = std::clamp(confidence, 1e-3, 1.0);
  const double fxb =
    std::max(1e-6, static_cast<double>(cam_.fx) * static_cast<double>(cam_.baseline));

  const double fx = std::max(1e-6, static_cast<double>(cam_.fx));
  const double fy = std::max(1e-6, static_cast<double>(cam_.fy));

  // ── 像素空间视差方差 ──
  const double sigma_u2 = disp_var_base_ / src_conf;
  const double sigma_v2 = disp_var_base_ / src_conf;

  // ── 深度噪声 ──
  const double ratio = (z * z) / fxb; // z²/(f_x·b)，单位 m/px
  const double sigma_z2 = kMinPointVar + disp_var_base_ * ratio * ratio / src_conf;

  // ── 3D各向异性传播 ──
  const double zx = pc.x() / std::max(z, 1e-6);
  const double zy = pc.y() / std::max(z, 1e-6);

  // 相机系下各向异性位置方差（由像素噪声与深度噪声共同传播）
  const double sigma_x2 = kMinPointVar + (z * z / (fx * fx)) * sigma_u2 + (zx * zx) * sigma_z2;
  const double sigma_y2 = kMinPointVar + (z * z / (fy * fy)) * sigma_v2 + (zy * zy) * sigma_z2;

  // ── 构建相机系对角协方差矩阵 ──
  Eigen::Matrix3d cov_cam = Eigen::Matrix3d::Zero();
  cov_cam(0, 0) = sigma_x2;
  cov_cam(1, 1) = sigma_y2;
  cov_cam(2, 2) = sigma_z2;

  // 返回迹的平均值用于标量近似fallback
  out_trace_mean = (sigma_x2 + sigma_y2 + sigma_z2) / 3.0;

  return cov_cam;
}

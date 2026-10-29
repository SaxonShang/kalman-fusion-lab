// Color ICP Registration
// Hanzhe Teng, Feb 2022
// 实现 Colored ICP（ICCV 2017）和 GICP（small_gicp 库）两种配准方法

#include "registration/color_icp.h"
#include "registration/stereo_gicp.h"
#include "sio_pipeline_helpers.h"
#include "utils.h"
#include "profiler/simple_profiler.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
// Ubuntu/OpenCV and PCL share a FLANN macro; instantiate PCL with std::map.
#ifdef USE_UNORDERED_MAP
#undef USE_UNORDERED_MAP
#endif
#define USE_UNORDERED_MAP 0
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/search/impl/kdtree.hpp>
#include <pcl/common/transforms.h>
#include <small_gicp/pcl/pcl_point_traits.hpp>
#include <small_gicp/points/eigen.hpp>
#include <small_gicp/registration/registration.hpp>
#include <small_gicp/registration/registration_helper.hpp>
#include <small_gicp/util/downsampling.hpp>
#include <small_gicp/util/downsampling_omp.hpp>
#include <small_gicp/util/normal_estimation_omp.hpp>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <yaml-cpp/yaml.h>

ColorICP::ColorICP(const std::string &params_file)
    // ── 初始化列表：设置默认值 ──
    : source_cloud_(
          new pcl::PointCloud<PointT>), // 源点云（预留，ColorICPRegistration
                                        // 内部不直接使用）
      target_cloud_(new pcl::PointCloud<PointT>),   // 目标点云（同上）
      transformation_(Eigen::Matrix4d::Identity()), // 累计变换矩阵
      rgb_to_intensity_weight_(Eigen::Vector3d::Ones() /
                               3.0), // RGB→亮度权重，默认均匀
      num_threads_(1), gradient_min_neighbors_(5), // 颜色梯度估计最少邻域点数
      gradient_max_neighbors_(24), // 颜色梯度估计最大邻域点数
      nn_search_epsilon_(0.01),    // KD-Tree 近似搜索容差
      verbose_iteration_(false),   // 是否输出每轮迭代信息
      kdtree_(new pcl::search::KdTree<
              PointT, pcl::KdTreeFLANN<PointT, flann::L2_Simple<float>>>) {

  // 从项目运行配置读取参数文件路径，避免 VS Code / 终端 cwd 不同时找错配置。
  const auto path = params_file.empty() ? svio_utils::LocateParamsFile()
                                        : std::filesystem::path(params_file);
  const YAML::Node root = path.empty() ? YAML::Node(YAML::NodeType::Map)
                                       : YAML::LoadFile(path.string());
  params_ = YAML::Clone(root);
  const auto section = root["color_icp"];
  if (section && section.IsMap()) {
    for (const auto &entry : section)
      params_[entry.first.as<std::string>()] = YAML::Clone(entry.second);
  }

  // RGB→亮度转换权重：支持 NTSC / sRGB 标准
  const std::string color_weight = params_["color_weight"].as<std::string>("uniform");
  if (color_weight == "ntsc")
    rgb_to_intensity_weight_ =
        Eigen::Vector3d(0.299, 0.587, 0.114); // YIQ/YUV/NTSC 标准
  else if (color_weight == "srgb")
    rgb_to_intensity_weight_ =
        Eigen::Vector3d(0.2126, 0.7152, 0.0722); // sRGB/Rec709 标准
  // 否则保持默认均匀权重 (1/3, 1/3, 1/3)

  // 多线程与梯度估计参数
  num_threads_ = std::max(1, params_["num_threads"].as<int>(1));
  gradient_min_neighbors_ =
      std::max(5, params_["gradient_min_neighbors"].as<int>(5));
  gradient_max_neighbors_ = std::max(
      gradient_min_neighbors_, params_["gradient_max_neighbors"].as<int>(32));
  nn_search_epsilon_ = params_["nn_search_epsilon"].as<double>(0.0);
  verbose_iteration_ = params_["verbose_iteration"].as<bool>(false);

  // 【优化】缓存配准参数到成员变量，避免每次调用 ColorICPRegistration /
  // 每次迭代都重新解析 YAML 字符串
  max_iterations_ = params_["icp_max_iterations"].as<int>(20); // 最大迭代轮数
  transformation_epsilon_ = params_["icp_transformation_epsilon"].as<double>(
      1e-6); // 收敛阈值（平移平方 + 旋转角度）
  max_corres_dist_ =
      params_["icp_max_corres_dist"].as<double>(0.5); // 最近邻对应点最大距离
  color_icp_lambda_ = // 几何项/光度项权重（0=纯光度，1=纯几何）
      std::max(0.0, std::min(1.0, params_["color_icp_lambda"].as<double>(0.5)));
  gicp_voxel_size_ =
      std::max(0.01, params_["gicp_voxel_size"].as<double>(0.05));
  gicp_max_correspondence_distance_ = std::max(
      gicp_voxel_size_,
      params_["gicp_max_correspondence_distance"].as<double>(
          2.0 * gicp_voxel_size_));

  const std::string reg_type =
      params_["registration_type"].as<std::string>("gicp");
  SetRegistrationType(reg_type);
}

void ColorICP::SetRegistrationType(RegistrationType type) {
  registration_type_ = type;
  // 缓存中的法向/协方差依赖于配准目标函数，切换后禁止复用。
  InvalidateGicpTargetPrepCache();
}

void ColorICP::SetRegistrationType(const std::string &type_str) {
  std::string s = type_str;
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  if (s == "icp" || s == "p2p" || s == "point_to_point") {
    SetRegistrationType(RegistrationType::ICP);
  } else if (s == "plane_icp" || s == "p2pl" || s == "point_to_plane") {
    SetRegistrationType(RegistrationType::PLANE_ICP);
  } else if (s == "stereo_gicp" || s == "gicp_stereo" || s == "stereo") {
    SetRegistrationType(RegistrationType::STEREO_GICP);
  } else {
    SetRegistrationType(RegistrationType::GICP);
  }
}

std::string ColorICP::GetRegistrationTypeString() const {
  switch (registration_type_) {
  case RegistrationType::ICP:
    return "ICP";
  case RegistrationType::PLANE_ICP:
    return "PLANE_ICP";
  case RegistrationType::STEREO_GICP:
    return "STEREO_GICP";
  case RegistrationType::GICP:
  default:
    return "GICP";
  }
}

void ColorICP::ConfigureGICP(double voxel_size,
                             double max_correspondence_distance) {
  gicp_voxel_size_ = std::max(0.01, voxel_size);
  gicp_max_correspondence_distance_ =
      std::max(gicp_voxel_size_, max_correspondence_distance);
  InvalidateGicpTargetPrepCache();
}

void ColorICP::ConfigureStereoGicpSolver(double gn_rot_eps, double gn_trans_eps,
                                         double cov_floor_trans_m,
                                         double cov_floor_rot_deg) {
  stereo_gn_rot_eps_ = std::max(0.0, gn_rot_eps);
  stereo_gn_trans_eps_ = std::max(0.0, gn_trans_eps);
  stereo_cov_floor_trans_m_ = std::max(0.0, cov_floor_trans_m);
  stereo_cov_floor_rot_deg_ = std::max(0.0, cov_floor_rot_deg);
}

void ColorICP::InvalidateGicpTargetPrepCache() {
  gicp_keyframe_prep_ = GicpKeyframePrep{};
}

namespace {

void PclRgbCloudToEigen4f(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud,
                          std::vector<Eigen::Vector4f> &out) {
  if (!cloud) {
    out.clear();
    return;
  }
  out.clear();
  out.reserve(cloud->size());
  for (const auto &p : cloud->points) {
    out.emplace_back(p.x, p.y, p.z, 1.0f);
  }
}

small_gicp::PointCloud::Ptr PclRgbCloudToSmallGicp(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud) {
  auto out = std::make_shared<small_gicp::PointCloud>();
  if (!cloud || cloud->empty()) {
    return out;
  }
  out->resize(cloud->size());
  for (size_t i = 0; i < cloud->size(); ++i) {
    const auto &p = cloud->points[i];
    out->point(i) << static_cast<double>(p.x), static_cast<double>(p.y),
        static_cast<double>(p.z), 1.0;
  }
  return out;
}

small_gicp::PointCloud::Ptr FilterGicpCloudByStaticIndices(
    const small_gicp::PointCloud &full_cloud,
    const std::vector<size_t> &source_indices,
    const std::vector<int> &static_indices) {
  if (full_cloud.empty() || source_indices.empty() || static_indices.empty()) {
    return nullptr;
  }

  const int max_index = *std::max_element(static_indices.begin(), static_indices.end());
  if (max_index < 0) {
    return nullptr;
  }

  std::vector<uint8_t> is_static(static_cast<size_t>(max_index) + 1, 0);
  for (const int idx : static_indices) {
    if (idx >= 0 && idx < static_cast<int>(is_static.size())) {
      is_static[static_cast<size_t>(idx)] = 1;
    }
  }

  const size_t cache_size = std::min(full_cloud.size(), source_indices.size());
  auto filtered = std::make_shared<small_gicp::PointCloud>();
  filtered->points.reserve(cache_size);
  const bool has_normals = !full_cloud.normals.empty();
  const bool has_covs = !full_cloud.covs.empty();
  if (has_normals) filtered->normals.reserve(cache_size);
  if (has_covs) filtered->covs.reserve(cache_size);

  for (size_t i = 0; i < cache_size; ++i) {
    const size_t src_idx = source_indices[i];
    if (src_idx >= is_static.size() || !is_static[src_idx]) {
      continue;
    }
    filtered->points.push_back(full_cloud.points[i]);
    if (has_normals) filtered->normals.push_back(full_cloud.normals[i]);
    if (has_covs) filtered->covs.push_back(full_cloud.covs[i]);
  }

  return filtered->empty() ? nullptr : filtered;
}

// 与 target 关键帧 prep / ESIKF 一致：先 PCL 索引对齐体素，再按算法需求估计法向/协方差。
small_gicp::PointCloud::Ptr PrepareGicpSourceCloud(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &source_cloud,
    double voxel_size, int num_threads,
    ColorICP::RegistrationType reg_type) {
  if (!source_cloud || source_cloud->empty()) {
    return nullptr;
  }

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_for_gicp = source_cloud;
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr downsampled;
  std::vector<float> downsampled_confidences;
  std::vector<size_t> unused_indices;
  const std::vector<float> empty_confidences;
  constexpr size_t kDenseCloudThreshold = 5000;
  if (source_cloud->size() > kDenseCloudThreshold && voxel_size > 0.0) {
    if (!sio_pipeline::BuildIndexAlignedTargetCloud(
            source_cloud, empty_confidences,
            static_cast<float>(voxel_size), downsampled,
            downsampled_confidences, unused_indices) ||
        !downsampled || downsampled->empty()) {
      return nullptr;
    }
    cloud_for_gicp = downsampled;
  }

  auto sg_cloud = PclRgbCloudToSmallGicp(cloud_for_gicp);
  if (!sg_cloud || sg_cloud->empty()) {
    return nullptr;
  }

  if (reg_type == ColorICP::RegistrationType::GICP) {
    auto kdtree =
        std::make_shared<small_gicp::KdTree<small_gicp::PointCloud>>(sg_cloud);
    small_gicp::estimate_normals_covariances_omp(*sg_cloud, *kdtree, 10,
                                                 num_threads);
  } else if (reg_type == ColorICP::RegistrationType::PLANE_ICP) {
    auto kdtree =
        std::make_shared<small_gicp::KdTree<small_gicp::PointCloud>>(sg_cloud);
    small_gicp::estimate_normals_omp(*sg_cloud, *kdtree, 10, num_threads);
  }
  // 纯点到点 ICP 不需要估计法向与协方差
  return sg_cloud;
}

} // namespace
// ── 构建配准关键帧目标预处理 ────────────────────────────────────────────
// 输入 cache_cloud: 缓存的目标点云
//   source_indices: 源点云索引（用于筛选缓存点云）
//   keyframe_ts: 关键帧时间戳
// 输出 gicp_keyframe_prep_: 关键帧目标预处理缓存
// 返回 true 成功，false 失败
bool ColorICP::BuildGicpKeyframeTargetPrep(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cache_cloud,
    const std::vector<size_t> &source_indices, double keyframe_ts) {
  InvalidateGicpTargetPrepCache();
  if (!cache_cloud || cache_cloud->empty() ||
      source_indices.size() != cache_cloud->size()) {
    return false;
  }

  auto sg_cloud = PclRgbCloudToSmallGicp(cache_cloud);
  if (!sg_cloud || sg_cloud->empty()) {
    return false;
  }

  auto kdtree = std::make_shared<small_gicp::KdTree<small_gicp::PointCloud>>(sg_cloud);
  if (registration_type_ == RegistrationType::GICP) {
    small_gicp::estimate_normals_covariances_omp(*sg_cloud, *kdtree, 10,
                                                 num_threads_);
  } else if (registration_type_ == RegistrationType::PLANE_ICP) {
    small_gicp::estimate_normals_omp(*sg_cloud, *kdtree, 10, num_threads_);
  }
  // 纯点到点 ICP 仅构建 KD-Tree，跳过法向和协方差估计

  gicp_keyframe_prep_.cloud = sg_cloud;
  gicp_keyframe_prep_.kdtree = kdtree;
  gicp_keyframe_prep_.source_indices = source_indices;
  gicp_keyframe_prep_.keyframe_ts = keyframe_ts;
  gicp_keyframe_prep_.voxel_size = gicp_voxel_size_;
  gicp_keyframe_prep_.valid = true;
  return true;
}
// ── 对齐 GICP 目标点云与源点云 ────────────────────────────────────────────
// 输入 target_cloud_prep: 目标点云预处理（体素化+KD-Tree）
//   target_kdtree: 目标点云 KD-Tree
//   source_cloud: 源点云
//   init_guess: 初始猜测变换
//   target_prep_from_keyframe_cache: 是否从关键帧体素缓存预处理
// 输出 out: GICP 配准结果
// 返回 GICP 配准结果
ColorICP::GICPResult ColorICP::AlignGICP(
    const small_gicp::PointCloud::Ptr &target_cloud_prep,
    const std::shared_ptr<small_gicp::KdTree<small_gicp::PointCloud>>
        &target_kdtree,
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr source_cloud,
    const Eigen::Matrix4d &init_guess, bool target_prep_from_keyframe_cache
    SIO_PROFILER_PARAMS_IMPL) {
  GICPResult out;
  out.T = Eigen::Matrix4f::Identity();
  out.covariance = Eigen::Matrix<double, 6, 6>::Identity() * 1e-2;
  out.target_prep_cached = target_prep_from_keyframe_cache;
  out.registration_type = GetRegistrationTypeString();
  out.downsampled_target_points =
      target_cloud_prep ? static_cast<int>(target_cloud_prep->size()) : 0;

  if (registration_type_ == RegistrationType::STEREO_GICP) {
    out.status = "use_EstimatePoseStereoGICP";
    return out;
  }

  if (!target_cloud_prep || target_cloud_prep->empty() || !target_kdtree ||
      !source_cloud || source_cloud->empty()) {
    return out;
  }

  const auto t_pre0 = std::chrono::steady_clock::now();
  small_gicp::PointCloud::Ptr source_cloud_prep = [&]() {
    SIO_PROFILE((*sio_prof), sio_frame_id, "gicp", "preprocess_source");
    return PrepareGicpSourceCloud(source_cloud, gicp_voxel_size_, num_threads_,
                                  registration_type_);
  }();
  out.preprocess_ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - t_pre0)
                          .count();
  out.downsampled_source_points =
      source_cloud_prep ? static_cast<int>(source_cloud_prep->size()) : 0;

  if (!source_cloud_prep) {
    return out;
  }

  small_gicp::RegistrationSetting setting;
  if (registration_type_ == RegistrationType::ICP) {
    setting.type = small_gicp::RegistrationSetting::ICP;
  } else if (registration_type_ == RegistrationType::PLANE_ICP) {
    setting.type = small_gicp::RegistrationSetting::PLANE_ICP;
  } else {
    setting.type = small_gicp::RegistrationSetting::GICP;
  }
  setting.max_correspondence_distance = gicp_max_correspondence_distance_;
  setting.max_iterations = max_iterations_;
  setting.num_threads = num_threads_;

  Eigen::Isometry3d init_iso;
  init_iso.matrix() = init_guess;

  const auto t_al0 = std::chrono::steady_clock::now();
  auto result = [&]() {
    SIO_PROFILE((*sio_prof), sio_frame_id, "gicp", "align");
    return small_gicp::align(*target_cloud_prep, *source_cloud_prep,
                           *target_kdtree, init_iso, setting);
  }();
  out.align_ms = std::chrono::duration<double, std::milli>(
                     std::chrono::steady_clock::now() - t_al0)
                     .count();

  out.T = result.T_target_source.matrix().cast<float>();
  out.num_inliers = static_cast<int>(result.num_inliers);
  out.num_iterations = static_cast<int>(result.iterations + 1);
  out.converged = result.converged;
  out.error = result.error;
  if (out.downsampled_source_points > 0) {
    out.fitness = static_cast<double>(out.num_inliers) /
                  static_cast<double>(out.downsampled_source_points);
  }

  {
    SIO_PROFILE((*sio_prof), sio_frame_id, "gicp", "covariance");
    const double det = result.H.determinant();
    if (std::isfinite(det) && std::abs(det) > 1e-12) {
      out.covariance = result.H.inverse();
      out.covariance = 0.5 * (out.covariance + out.covariance.transpose());
      out.covariance_from_hessian = true;
    } else {
      const double scale = (result.num_inliers > 0)
                               ? 1.0 / static_cast<double>(result.num_inliers)
                               : 1e-2;
      out.covariance = Eigen::Matrix<double, 6, 6>::Identity() * scale;
    }
  }

  return out;
}

// ── 6 维李代数增量 → 4x4 SE(3) 齐次变换矩阵 ─────────────────────────────────
// 输入 input = [α, β, γ, tx, ty, tz]
//   α, β, γ: 绕 X/Y/Z 轴的旋转角（欧拉角 XYZ 顺序）
//   tx, ty, tz: 平移分量
Eigen::Matrix4d
ColorICP::TransformVector6dToMatrix4d(const Eigen::Vector6d &input) {
  Eigen::Matrix4d output = Eigen::Matrix4d::Identity();
  output.block<3, 3>(0, 0) =
      (Eigen::AngleAxisd(input(2), Eigen::Vector3d::UnitZ()) *
       Eigen::AngleAxisd(input(1), Eigen::Vector3d::UnitY()) *
       Eigen::AngleAxisd(input(0), Eigen::Vector3d::UnitX()))
          .matrix();
  output.block<3, 1>(0, 3) = input.block<3, 1>(3, 0);
  return output;
}

// ── 主接口 1：Colored ICP 配准 ──────────────────────────────────────────────
// 论文：Colored Point Cloud Registration Revisited, ICCV 2017
//
// 算法流程（每轮迭代）：
//   对 source 施加当前估计变换 → 对每个变换后的源点在 target 中找最近邻
//   → 对命中的对应点对同时计算几何残差（point-to-plane）和光度残差（颜色差）
//   → 用 Gauss-Newton 求解 6 维位姿增量 → 更新累计变换 → 判断收敛
//
// 【优化要点】
//   1. NN 搜索与 GN 累加融合为单次遍历，消除中间子点云拷贝
//   2. 配准参数从构造函数缓存，消除每轮 YAML 解析
//   3. 所有缓冲区在迭代外层预分配，消除每轮堆分配
Eigen::Matrix4d ColorICP::ColorICPRegistration(
    const PointCloudPtr &source, const PointCloudPtr &target,
    const std::vector<Eigen::Vector3d> &target_color_gradient,
    const std::vector<pcl::search::KdTree<PointT>::Ptr> &thread_kdtrees,
    Eigen::Matrix4d guess_transformation) {
  if (!source || source->empty() || !target || target->empty() ||
      thread_kdtrees.empty())
    return guess_transformation;

  // ── 从缓存的成员变量读取参数（构造时已从 YAML 解析） ──
  const double rotation_epsilon = 1.0 - transformation_epsilon_;
  const double dist_sq_thresh = max_corres_dist_ * max_corres_dist_;
  const double sqrt_lambda = std::sqrt(color_icp_lambda_);
  const double sqrt_1m_lambda = std::sqrt(1.0 - color_icp_lambda_);
  const int cloud_size = static_cast<int>(source->size());
  const int actual_threads = static_cast<int>(thread_kdtrees.size());

  Eigen::Matrix4d final_transformation = guess_transformation;

  // ── 预分配变换后源点云缓冲区（每轮迭代复用） ──
  pcl::PointCloud<PointT> source_trans;
  source_trans.points.resize(source->size());
  // ══════════════════════════════════════════════════════════════════════════
  // 迭代优化主循环：每轮融合 NN 搜索 + Gauss-Newton 累加为单次遍历
  // ══════════════════════════════════════════════════════════════════════════
  for (int iter = 0; iter < max_iterations_; ++iter) {
    // 用当前累计变换将 source 变换到估计坐标系
    pcl::transformPointCloud<PointT>(*source, source_trans,
                                     final_transformation);

    // ── per-thread Gauss-Newton 累加器（每轮置零，线程独立无竞争） ──
    // JTJ_G / JTr_G: 几何项（point-to-plane）的法方程累加
    // JTJ_C / JTr_C: 光度项（photometric）的法方程累加
    // corres_local:  每线程有效对应数（用于判断本轮是否有足够对应）
    std::vector<Eigen::Matrix6d> JTJ_G_local(actual_threads,
                                             Eigen::Matrix6d::Zero());
    std::vector<Eigen::Vector6d> JTr_G_local(actual_threads,
                                             Eigen::Vector6d::Zero());
    std::vector<Eigen::Matrix6d> JTJ_C_local(actual_threads,
                                             Eigen::Matrix6d::Zero());
    std::vector<Eigen::Vector6d> JTr_C_local(actual_threads,
                                             Eigen::Vector6d::Zero());
    std::vector<int> corres_local(actual_threads, 0);

#ifdef _OPENMP
#pragma omp parallel num_threads(actual_threads)
#endif
    {
      int tid = 0;
#ifdef _OPENMP
      tid = omp_get_thread_num();
#endif
      // 线程私有的 NN 搜索缓冲区（parallel 区域外分配一次，内层循环复用）
      std::vector<int> nn_idx(1);
      std::vector<float> nn_sq(1);

#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
      for (int i = 0; i < cloud_size; ++i) {

        // ── Step 1: 在 target 中搜索源点 i 的最近邻 ──
        thread_kdtrees[tid]->nearestKSearch(source_trans.points[i], 1, nn_idx,
                                            nn_sq);
        if (nn_sq[0] > dist_sq_thresh) // 距离超阈值，丢弃此对应
          continue;
        const int j = nn_idx[0]; // target 中最近邻点索引
        ++corres_local[tid];     // 统计有效对应数

        // ── Step 2: 就地读取源/目标点数据（零拷贝，不再拷贝到子点云） ──
        // q: 变换后的源点坐标   cq/iq: 源点颜色/亮度
        const Eigen::Vector3d q =
            source_trans.points[i].getVector3fMap().cast<double>();
        const Eigen::Vector3d cq =
            source_trans.points[i].getRGBVector3i().cast<double>();
        const double iq = cq.dot(rgb_to_intensity_weight_) / 255.0;

        // p: 目标对应点坐标   np: 目标点法向   cp/ip: 目标点颜色/亮度
        // dp: 目标点颜色梯度（由 PrepareColorGradient 预计算）
        const Eigen::Vector3d p =
            target->points[j].getVector3fMap().cast<double>();
        const Eigen::Vector3d np =
            target->points[j].getNormalVector3fMap().cast<double>();
        const Eigen::Vector3d cp =
            target->points[j].getRGBVector3i().cast<double>();
        const double ip = cp.dot(rgb_to_intensity_weight_) / 255.0;
        const Eigen::Vector3d &dp = target_color_gradient[j];

        // ── Step 3: 几何项（point-to-plane） ──
        // 残差: r_geo = (q - p) · n_p              (论文 Eq.19)
        // 雅可比: J_geo = [q × n_p ; n_p]^T        (论文 Eq.30)
        const double r_geo = (q - p).dot(np);
        Eigen::Vector6d J_geo;
        J_geo.head<3>() = q.cross(np); // 对旋转分量的导数
        J_geo.tail<3>() = np;          // 对平移分量的导数
        JTJ_G_local[tid].noalias() += J_geo * J_geo.transpose(); // J^T J 累加
        JTr_G_local[tid].noalias() += J_geo * r_geo;             // J^T r 累加

        // ── Step 4: 光度项（photometric） ──
        // dpM = dp - (dp · np) np  等价于 dp^T (I - np np^T)，但避免了 3×3
        // 矩阵乘 r_color = I(q_proj) - I(q)，其中 q_proj 是 q 在切平面上的投影
        const Eigen::Vector3d dpM = dp - dp.dot(np) * np;
        const double r_color = ip + dp.dot(q - (q - p).dot(np) * np - p) - iq;
        Eigen::Vector6d J_color;
        J_color.head<3>() =
            q.cross(dpM); // 光度雅可比对旋转分量  (论文 Eq.28-29)
        J_color.tail<3>() = dpM; // 光度雅可比对平移分量
        JTJ_C_local[tid].noalias() += J_color * J_color.transpose();
        JTr_C_local[tid].noalias() += J_color * r_color;
      }
    }

    // ── 归约：合并各线程的法方程，按 lambda 权重融合几何项与光度项 ──
    // JTJ = √λ · JTJ_G + √(1-λ) · JTJ_C  （与 Open3D 公式一致）
    Eigen::Matrix6d JTJ = Eigen::Matrix6d::Zero();
    Eigen::Vector6d JTr = Eigen::Vector6d::Zero();
    int total_corres = 0;
    for (int tid = 0; tid < actual_threads; ++tid) {
      JTJ.noalias() +=
          sqrt_lambda * JTJ_G_local[tid] + sqrt_1m_lambda * JTJ_C_local[tid];
      JTr.noalias() +=
          sqrt_lambda * JTr_G_local[tid] + sqrt_1m_lambda * JTr_C_local[tid];
      total_corres += corres_local[tid];
    }

    // 无有效对应或数值异常时提前退出
    if (total_corres == 0 || !JTJ.allFinite() || !JTr.allFinite())
      break;

    // ── 求解 6 维位姿增量 X = [α,β,γ,tx,ty,tz] ──
    // 线性系统: JTJ · X = -JTr（Gauss-Newton 正规方程）
    const Eigen::Vector6d X = JTJ.ldlt().solve(-JTr);
    const Eigen::Matrix4d dT = TransformVector6dToMatrix4d(X); // 增量变换矩阵
    final_transformation = dT * final_transformation; // 左乘更新累计变换

    // ── 收敛判断 ──
    // 旋转：cos(angle) = (tr(dT_rot) - 1) / 2，越接近 1 增量越小
    // 平移：增量平移的平方和
    const double cos_angle = 0.5 * (dT(0, 0) + dT(1, 1) + dT(2, 2) - 1.0);
    const double trans_sqr =
        dT(0, 3) * dT(0, 3) + dT(1, 3) * dT(1, 3) + dT(2, 3) * dT(2, 3);
    if (cos_angle >= rotation_epsilon && trans_sqr <= transformation_epsilon_)
      break; // 旋转和平移增量均足够小，认为已收敛
  }

  return final_transformation; // 返回 source → target 的最终 4×4 刚体变换
}

ColorICP::GICPResult
ColorICP::EstimatePoseGICPWithCacheFilter(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr source_cloud,
    const Eigen::Matrix4d &init_guess, double target_keyframe_ts,
    const std::vector<int> &static_indices,
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr fallback_target_cloud
    SIO_PROFILER_PARAMS_IMPL) {
  if (!source_cloud || source_cloud->empty()) {
    return GICPResult{};
  }

  const bool keyframe_prep_ready =
      gicp_keyframe_prep_.valid && gicp_keyframe_prep_.cloud &&
      gicp_keyframe_prep_.kdtree &&
      gicp_keyframe_prep_.keyframe_ts == target_keyframe_ts &&
      gicp_keyframe_prep_.voxel_size == gicp_voxel_size_ &&
      gicp_keyframe_prep_.source_indices.size() ==
          gicp_keyframe_prep_.cloud->size();

  if (keyframe_prep_ready && !static_indices.empty()) {
    small_gicp::PointCloud::Ptr target_cloud_prep;
    std::shared_ptr<small_gicp::KdTree<small_gicp::PointCloud>> target_kdtree;

    target_cloud_prep = [&]() {
      SIO_PROFILE((*sio_prof), sio_frame_id, "gicp", "filter_target_prep");
      return FilterGicpCloudByStaticIndices(
          *gicp_keyframe_prep_.cloud, gicp_keyframe_prep_.source_indices,
          static_indices);
    }();

    if (target_cloud_prep) {
      if (target_cloud_prep->size() == gicp_keyframe_prep_.cloud->size()) {
        target_kdtree = gicp_keyframe_prep_.kdtree;
      } else {
        target_kdtree = [&]() {
          SIO_PROFILE((*sio_prof), sio_frame_id, "gicp", "build_target_kdtree");
          return std::make_shared<small_gicp::KdTree<small_gicp::PointCloud>>(
              target_cloud_prep);
        }();
      }

      return AlignGICP(target_cloud_prep, target_kdtree, source_cloud,
                       init_guess, true SIO_PROFILER_CALL(sio_prof, sio_frame_id));
    }
  }

  return EstimatePoseGICP(fallback_target_cloud, source_cloud, init_guess,
                          target_keyframe_ts
                          SIO_PROFILER_CALL(sio_prof, sio_frame_id));
}

ColorICP::GICPResult
ColorICP::EstimatePoseGICP(pcl::PointCloud<pcl::PointXYZRGB>::Ptr target_cloud,
                           pcl::PointCloud<pcl::PointXYZRGB>::Ptr source_cloud,
                           const Eigen::Matrix4d &init_guess,
                           double target_keyframe_ts
                           SIO_PROFILER_PARAMS_IMPL) {
  (void)target_keyframe_ts;

  if (!target_cloud || target_cloud->empty() || !source_cloud ||
      source_cloud->empty()) {
    return GICPResult{};
  }

  const double voxel_size = gicp_voxel_size_;
  std::vector<Eigen::Vector4f> target_pts;
  {
    SIO_PROFILE((*sio_prof), sio_frame_id, "gicp", "pcl_to_eigen_target");
    PclRgbCloudToEigen4f(target_cloud, target_pts);
  }
  auto target_prep = [&]() {
    SIO_PROFILE((*sio_prof), sio_frame_id, "gicp", "preprocess_target");
    if (registration_type_ == RegistrationType::ICP) {
      auto target_raw =
          small_gicp::voxelgrid_sampling_omp<std::vector<Eigen::Vector4f>,
                                             small_gicp::PointCloud>(
              target_pts, voxel_size, num_threads_);
      auto target_kdtree =
          std::make_shared<small_gicp::KdTree<small_gicp::PointCloud>>(target_raw);
      return std::make_pair(target_raw, target_kdtree);
    } else if (registration_type_ == RegistrationType::PLANE_ICP) {
      auto target_raw =
          small_gicp::voxelgrid_sampling_omp<std::vector<Eigen::Vector4f>,
                                             small_gicp::PointCloud>(
              target_pts, voxel_size, num_threads_);
      auto target_kdtree =
          std::make_shared<small_gicp::KdTree<small_gicp::PointCloud>>(target_raw);
      small_gicp::estimate_normals_omp(*target_raw, *target_kdtree, 10,
                                       num_threads_);
      return std::make_pair(target_raw, target_kdtree);
    }
    return small_gicp::preprocess_points(target_pts, voxel_size, 10,
                                         num_threads_);
  }();

  return AlignGICP(target_prep.first, target_prep.second, source_cloud,
                   init_guess, false SIO_PROFILER_CALL(sio_prof, sio_frame_id));
}

ColorICP::GICPResult ColorICP::EstimatePoseStereoGICP(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr source_cloud_camera,
    const cv::Mat &target_disp,
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr target_cloud_camera,
    const CameraIntrinsics &model_camera, const Eigen::Matrix4d &T_imu_cam,
    const Eigen::Matrix4d &T_curr_cam_to_prevkf_cam
    SIO_PROFILER_PARAMS_IMPL) {
  GICPResult out;
  out.T = Eigen::Matrix4f::Identity();
  out.covariance = Eigen::Matrix<double, 6, 6>::Identity() * 1e-2;
  out.registration_type = "STEREO_GICP";
  out.covariance_convention = CovarianceConvention::LeftTarget;

  if (!source_cloud_camera || source_cloud_camera->empty()) {
    out.status = "empty_source";
    return out;
  }

  sio_stereo_gicp::Calib calib = sio_stereo_gicp::CalibFromCamera(model_camera);
  sio_stereo_gicp::Settings settings;
  settings.voxel_size = gicp_voxel_size_;
  settings.source_voxel_representative =
      params_["stereo_source_voxel_representative"].as<bool>(false);
  settings.max_iteration = max_iterations_;
  settings.num_threads = num_threads_;
  settings.gn_rot_eps = stereo_gn_rot_eps_;
  settings.gn_trans_eps = stereo_gn_trans_eps_;
  settings.disparity_normals = params_["stereo_disparity_normals"].as<bool>(false);
  settings.plane_gate_sigma = std::max(0.0, params_["stereo_plane_gate_sigma"].as<double>(0.0));
  settings.grazing_cos = std::clamp(params_["stereo_grazing_cos"].as<double>(0.0),0.0,1.0);
  settings.point_plane = params_["stereo_point_plane"].as<bool>(false);
  settings.joint_geometry = params_["stereo_joint_geometry"].as<bool>(false);
  settings.stable_normals = params_["stereo_stable_normals"].as<bool>(false);
  settings.capture_distance = std::clamp(params_["stereo_capture_distance"].as<double>(0.10),0.01,0.50);
  settings.capture_pixel_cap = std::clamp(params_["stereo_capture_pixel_cap"].as<int>(15),2,40);
  settings.measurement_scale = std::max(0.0, params_["stereo_measurement_scale"].as<double>(1.0));
  settings.geometry_tangent_sigma = std::max(0.001, params_["stereo_geometry_tangent_sigma"].as<double>(0.10));
  settings.robust_delta = std::max(0.0, params_["stereo_robust_delta"].as<double>(0.0));
  settings.normal_cos_gate = std::clamp(params_["stereo_normal_cos_gate"].as<double>(0.0),0.0,1.0);

  cv::Mat depth;
  const auto t_depth0 = std::chrono::steady_clock::now();
  {
    SIO_PROFILE((*sio_prof), sio_frame_id, "gicp", "stereo_depth");
    if (!target_disp.empty()) {
      depth = sio_stereo_gicp::DepthFromDisparity(target_disp, calib,
                                                  settings.zmin, settings.zmax);
    }
    if(settings.point_plane && target_cloud_camera && !depth.empty()) {
      // 候选严格尊重动态过滤后的目标支持；禁止原视差把已删除像素复活。
      // 这里只取实际像素支持，不膨胀也不填洞；空点云对应空支持。
      cv::Mat support=cv::Mat::zeros(depth.size(),CV_8U);
      for(const auto &p:target_cloud_camera->points) {
        if(!std::isfinite(p.x)||!std::isfinite(p.y)||!std::isfinite(p.z)||p.z<=0)continue;
        const int u=std::lround(calib.fx*p.x/p.z+calib.cx),v=std::lround(calib.fy*p.y/p.z+calib.cy);
        if(u>=0 && v>=0 && u<depth.cols && v<depth.rows) {
          const double z=depth.at<double>(v,u);
          if(std::isfinite(z) && std::abs(z-p.z)<=std::max(1e-4,1e-4*z))support.at<unsigned char>(v,u)=1;
        }
      }
      depth.setTo(std::numeric_limits<double>::quiet_NaN(),support==0);
    }
    bool has_depth = false;
    if (!depth.empty()) {
      for (int v = 0; v < depth.rows && !has_depth; ++v) {
        const double *row = depth.ptr<double>(v);
        for (int u = 0; u < depth.cols; ++u) {
          if (std::isfinite(row[u])) {
            has_depth = true;
            break;
          }
        }
      }
    }
    if (!has_depth && target_cloud_camera && !target_cloud_camera->empty()) {
      std::vector<Eigen::Vector3d> tgt_pts;
      tgt_pts.reserve(target_cloud_camera->size());
      for (const auto &pt : target_cloud_camera->points) {
        if (std::isfinite(pt.x) && std::isfinite(pt.y) && std::isfinite(pt.z)) {
          tgt_pts.emplace_back(pt.x, pt.y, pt.z);
        }
      }
      int width = calib.width > 0 ? calib.width : 896;
      int height = calib.height > 0 ? calib.height : 448;
      if (!target_disp.empty()) {
        width = target_disp.cols;
        height = target_disp.rows;
      }
      const int fill_px =
          std::max(1, static_cast<int>(std::ceil(
                          settings.voxel_size * calib.fx /
                          std::max(1.0, 3.0))));
      depth = sio_stereo_gicp::SplatDepth(tgt_pts, calib, width, height,
                                          settings.zmin, settings.zmax, settings.point_plane ? 0 : fill_px);
    }
  }

  if (depth.empty()) {
    out.status = "empty_target_depth";
    return out;
  }
  if (calib.width <= 0) {
    calib.width = depth.cols;
  }
  if (calib.height <= 0) {
    calib.height = depth.rows;
  }

  std::vector<Eigen::Vector3d> src_pts;
  src_pts.reserve(source_cloud_camera->size());
  for (const auto &pt : source_cloud_camera->points) {
    if (std::isfinite(pt.x) && std::isfinite(pt.y) && std::isfinite(pt.z) &&
        pt.z >= static_cast<float>(settings.zmin) &&
        pt.z <= static_cast<float>(settings.zmax)) {
      src_pts.emplace_back(pt.x, pt.y, pt.z);
    }
  }

  const double depth_ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - t_depth0)
                              .count();

  sio_stereo_gicp::Result solved;
  {
    SIO_PROFILE((*sio_prof), sio_frame_id, "gicp", "align");
    solved = sio_stereo_gicp::AlignProjective(
        src_pts, depth, calib, T_curr_cam_to_prevkf_cam, settings);
  }

  const Eigen::Matrix4d T_body =
      sio_stereo_gicp::BodyFromCamera(solved.T_target_source, T_imu_cam);
  out.T = T_body.cast<float>();
  out.covariance =
      sio_stereo_gicp::CovarianceBodyFromCamera(solved.covariance, T_imu_cam);
  if (stereo_cov_floor_trans_m_ > 0.0 || stereo_cov_floor_rot_deg_ > 0.0) {
    const double rot_var = std::pow(
        stereo_cov_floor_rot_deg_ * M_PI / 180.0, 2.0);
    const double trans_var = std::pow(stereo_cov_floor_trans_m_, 2.0);
    for (int i = 0; i < 3; ++i) {
      if (stereo_cov_floor_rot_deg_ > 0.0) {
        out.covariance(i, i) = std::max(out.covariance(i, i), rot_var);
      }
      if (stereo_cov_floor_trans_m_ > 0.0) {
        out.covariance(i + 3, i + 3) =
            std::max(out.covariance(i + 3, i + 3), trans_var);
      }
    }
  }
  out.num_inliers = solved.num_inliers;
  out.num_iterations = solved.num_iterations;
  out.converged = solved.converged;
  out.error = solved.error;
  out.rmse = solved.rmse;
  out.fitness = solved.fitness;
  out.preprocess_ms = solved.preprocess_ms + depth_ms;
  out.align_ms = solved.align_ms;
  out.downsampled_source_points = solved.downsampled_source_points;
  out.downsampled_target_points = solved.downsampled_target_points;
  out.status = solved.status;
  out.stereo_trace = solved.trace;
  out.covariance_from_hessian = solved.covariance_from_hessian;
  out.registration_type = "STEREO_GICP";
  return out;
}

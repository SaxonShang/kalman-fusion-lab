// Color ICP Registration
// Hanzhe Teng, Feb 2022
// 实现 Colored ICP（ICCV 2017）、Stereo-GICP（投影关联 + 立体 Σ）和 small_gicp 欧氏退路
#include "color_icp.h"

#include "sio_frontend_modules.h"
#include "stereo_gicp.h"

// OpenCV（经 data.h 等）会先包含 opencv2/flann，并把 USE_UNORDERED_MAP 设为 1；
// 系统 FLANN 对 LSH 的序列化只特化了 std::map，unordered_map 会编译失败。
// 在实例化 PCL/FLANN 模板之前强制使用 std::map。
#ifdef USE_UNORDERED_MAP
#undef USE_UNORDERED_MAP
#endif
#define USE_UNORDERED_MAP 0

#include <algorithm>
#include <atomic>
#include <chrono>
#include <pcl/common/transforms.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp> // needed for L1 distance
#include <pcl/search/impl/kdtree.hpp>       // needed for L2 distance
#include <small_gicp/pcl/pcl_point_traits.hpp>
#include <small_gicp/registration/registration.hpp>
#include <small_gicp/registration/registration_helper.hpp>
#include <small_gicp/util/downsampling.hpp>
#include <small_gicp/util/normal_estimation_omp.hpp>

#include "sio_pipeline_helpers.h"

#ifdef _OPENMP
#  include <omp.h>
#endif

#include <yaml-cpp/yaml.h>
#include "sio_config.h"

namespace {

YAML::Node ColorIcpParamsNode(const YAML::Node &root) {
  YAML::Node section = root["color_icp"];
  return section && section.IsMap() ? section : root;
}

} // namespace

// ColorICP构造函数：初始化配置参数
ColorICP::ColorICP(const std::string &params_file)
  // ── 初始化列表：设置默认值 ──
  : source_cloud_(new pcl::PointCloud<PointT>)
  , // 源点云（预留，ColorICPRegistration
    // 内部不直接使用）
  target_cloud_(new pcl::PointCloud<PointT>)
  , // 目标点云（同上）
  transformation_(Eigen::Matrix4d::Identity())
  , // 累计变换矩阵
  rgb_to_intensity_weight_(Eigen::Vector3d::Ones() / 3.0)
  , // RGB→亮度权重，默认均匀
  num_threads_(1)
  , gradient_min_neighbors_(5)
  , // 颜色梯度估计最少邻域点数
  gradient_max_neighbors_(24)
  , // 颜色梯度估计最大邻域点数
  nn_search_epsilon_(0.01)
  , // KD-Tree 近似搜索容差
  verbose_iteration_(false)
  , // 是否输出每轮迭代信息
  kdtree_(new pcl::search::KdTree<PointT, pcl::KdTreeFLANN<PointT, flann::L2_Simple<float>>>) {

  // 从 params.yaml 的 color_icp 节读取参数（路径与 project_config.params_file 一致）
  const std::filesystem::path params_path = params_file.empty()
    ? sio_utils::LocateParamsFile() : std::filesystem::path(params_file);
  const YAML::Node root = params_path.empty()
    ? YAML::Node(YAML::NodeType::Map) : YAML::LoadFile(params_path.string());
  YAML::Node cfg = ColorIcpParamsNode(root);

  // RGB→亮度转换权重：支持 NTSC / sRGB 标准
  const std::string color_weight = cfg["color_weight"].as<std::string>("uniform");
  if (color_weight == "ntsc")
    rgb_to_intensity_weight_ = Eigen::Vector3d(0.299, 0.587, 0.114); // YIQ/YUV/NTSC 标准
  else if (color_weight == "srgb")
    rgb_to_intensity_weight_ = Eigen::Vector3d(0.2126, 0.7152, 0.0722); // sRGB/Rec709 标准
  // 否则保持默认均匀权重 (1/3, 1/3, 1/3)

  // 多线程与梯度估计参数
  num_threads_ = std::max(1, cfg["num_threads"].as<int>(1));
  gradient_min_neighbors_ = std::max(5, cfg["gradient_min_neighbors"].as<int>(5));
  gradient_max_neighbors_ =
    std::max(gradient_min_neighbors_, cfg["gradient_max_neighbors"].as<int>(32));
  nn_search_epsilon_ = cfg["nn_search_epsilon"].as<double>(0.0);
  verbose_iteration_ = cfg["verbose_iteration"].as<bool>(false);

  // 【优化】缓存配准参数到成员变量，避免每次调用 ColorICPRegistration /
  // 每次迭代都重新解析 YAML 字符串
  max_iterations_ = cfg["icp_max_iterations"].as<int>(20); // 最大迭代轮数
  transformation_epsilon_ =
    cfg["icp_transformation_epsilon"].as<double>(1e-6); // 收敛阈值（平移平方 + 旋转角度）
  max_corres_dist_ = cfg["icp_max_corres_dist"].as<double>(0.5); // 最近邻对应点最大距离
  color_icp_lambda_ = // 几何项/光度项权重（0=纯光度，1=纯几何）
    std::max(0.0, std::min(1.0, cfg["color_icp_lambda"].as<double>(0.5)));
  gicp_voxel_size_ = std::max(0.01, cfg["gicp_voxel_size"].as<double>(0.05));
  gicp_max_correspondence_distance_ = std::max(
    gicp_voxel_size_, cfg["gicp_max_correspondence_distance"].as<double>(2.0 * gicp_voxel_size_));
}

void ColorICP::ConfigureGICP(double voxel_size, double max_correspondence_distance) {
  gicp_voxel_size_ = std::max(0.01, voxel_size);
  gicp_max_correspondence_distance_ = std::max(gicp_voxel_size_, max_correspondence_distance);
  InvalidateGicpTargetPrepCache();
}
// gicp目标缓存清理函数
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
  const small_gicp::PointCloud &full_cloud, const std::vector<size_t> &source_indices,
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
  filtered->normals.reserve(cache_size);
  filtered->covs.reserve(cache_size);

  for (size_t i = 0; i < cache_size; ++i) {
    const size_t src_idx = source_indices[i];
    if (src_idx >= is_static.size() || !is_static[src_idx]) {
      continue;
    }
    filtered->points.push_back(full_cloud.points[i]);
    filtered->normals.push_back(full_cloud.normals[i]);
    filtered->covs.push_back(full_cloud.covs[i]);
  }

  return filtered->empty() ? nullptr : filtered;
}

small_gicp::PointCloud::Ptr PrepareGicpSourceCloud(
  const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &source_cloud, double voxel_size,
  int num_threads) {
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
          source_cloud, empty_confidences, static_cast<float>(voxel_size), downsampled,
          downsampled_confidences, unused_indices)
        || !downsampled || downsampled->empty()) {
      return nullptr;
    }
    cloud_for_gicp = downsampled;
  }

  auto sg_cloud = PclRgbCloudToSmallGicp(cloud_for_gicp);
  if (!sg_cloud || sg_cloud->empty()) {
    return nullptr;
  }

  auto kdtree = std::make_shared<small_gicp::KdTree<small_gicp::PointCloud>>(sg_cloud);
  small_gicp::estimate_normals_covariances_omp(*sg_cloud, *kdtree, 10, num_threads);
  return sg_cloud;
}

} // namespace

bool ColorICP::BuildGicpKeyframeTargetPrep(
  const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cache_cloud,
  const std::vector<size_t> &source_indices, double keyframe_ts) {
  InvalidateGicpTargetPrepCache();
  if (!cache_cloud || cache_cloud->empty() || source_indices.size() != cache_cloud->size()) {
    return false;
  }

  auto sg_cloud = PclRgbCloudToSmallGicp(cache_cloud); // 将pcl点云转换为small_gicp点云
  if (!sg_cloud || sg_cloud->empty()) {
    return false;
  }

  auto kdtree = std::make_shared<small_gicp::KdTree<small_gicp::PointCloud>>(sg_cloud);
  small_gicp::estimate_normals_covariances_omp(*sg_cloud, *kdtree, 10, num_threads_);

  gicp_keyframe_prep_.cloud = sg_cloud;
  gicp_keyframe_prep_.kdtree = kdtree;
  gicp_keyframe_prep_.source_indices = source_indices;
  gicp_keyframe_prep_.keyframe_ts = keyframe_ts;
  gicp_keyframe_prep_.voxel_size = gicp_voxel_size_;
  gicp_keyframe_prep_.valid = true;
  return true;
}

ColorICP::GICPResult ColorICP::AlignGICP(
  const small_gicp::PointCloud::Ptr &target_cloud_prep,
  const std::shared_ptr<small_gicp::KdTree<small_gicp::PointCloud>> &target_kdtree,
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr source_cloud, const Eigen::Matrix4d &init_guess,
  bool target_prep_from_keyframe_cache SIO_PROFILER_PARAMS_IMPL) {
  GICPResult out;
  out.T = Eigen::Matrix4f::Identity();
  out.covariance = Eigen::Matrix<double, 6, 6>::Identity() * 1e-2;
  out.target_prep_cached = target_prep_from_keyframe_cache;
  out.downsample_target = target_cloud_prep ? static_cast<int>(target_cloud_prep->size()) : 0;

  if (!target_cloud_prep || target_cloud_prep->empty() || !target_kdtree || !source_cloud
      || source_cloud->empty()) {
    return out;
  }

  small_gicp::PointCloud::Ptr source_cloud_prep = [&]() {
    SIO_PROFILE((*sio_prof), sio_frame_id, "gicp", "preprocess_source");
    return PrepareGicpSourceCloud(source_cloud, gicp_voxel_size_, num_threads_);
  }();
  out.downsample_source = source_cloud_prep ? static_cast<int>(source_cloud_prep->size()) : 0;

  if (!source_cloud_prep) {
    return out;
  }

  small_gicp::RegistrationSetting setting;
  setting.type = small_gicp::RegistrationSetting::GICP;
  setting.max_correspondence_distance = gicp_max_correspondence_distance_;
  setting.max_iterations = max_iterations_;
  setting.num_threads = num_threads_;

  Eigen::Isometry3d init_iso;
  init_iso.matrix() = init_guess;

  auto result = [&]() {
    SIO_PROFILE((*sio_prof), sio_frame_id, "gicp", "align");
    return small_gicp::align(*target_cloud_prep, *source_cloud_prep, *target_kdtree, init_iso,
                             setting);
  }();

  out.T = result.T_target_source.matrix().cast<float>();
  out.num_inliers = static_cast<int>(result.num_inliers);
  out.iterations = static_cast<int>(result.iterations + 1);
  out.converged = result.converged;

  {
    SIO_PROFILE((*sio_prof), sio_frame_id, "gicp", "covariance");
    const double det = result.H.determinant();
    if (std::isfinite(det) && std::abs(det) > 1e-12) {
      out.covariance = result.H.inverse();
      out.covariance = 0.5 * (out.covariance + out.covariance.transpose());
      out.covariance_from_hessian = true;
    } else {
      const double scale =
        (result.num_inliers > 0) ? 1.0 / static_cast<double>(result.num_inliers) : 1e-2;
      out.covariance = Eigen::Matrix<double, 6, 6>::Identity() * scale;
    }
  }

  return out;
}

ColorICP::GICPResult ColorICP::EstimatePoseGICPWithCacheFilter(
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr source_cloud, const Eigen::Matrix4d &init_guess,
  double target_keyframe_ts, const std::vector<int> &static_indices,
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr fallback_target_cloud SIO_PROFILER_PARAMS_IMPL) {
  if (!source_cloud || source_cloud->empty()) {
    return GICPResult{};
  }

  const bool keyframe_prep_ready =
    gicp_keyframe_prep_.valid && gicp_keyframe_prep_.cloud && gicp_keyframe_prep_.kdtree
    && gicp_keyframe_prep_.keyframe_ts == target_keyframe_ts
    && gicp_keyframe_prep_.voxel_size == gicp_voxel_size_
    && gicp_keyframe_prep_.source_indices.size() == gicp_keyframe_prep_.cloud->size();

  if (keyframe_prep_ready && !static_indices.empty()) {
    small_gicp::PointCloud::Ptr target_cloud_prep;
    std::shared_ptr<small_gicp::KdTree<small_gicp::PointCloud>> target_kdtree;

    target_cloud_prep = [&]() {
      SIO_PROFILE((*sio_prof), sio_frame_id, "gicp", "filter_target_prep");
      return FilterGicpCloudByStaticIndices(*gicp_keyframe_prep_.cloud,
                                            gicp_keyframe_prep_.source_indices, static_indices);
    }();

    if (target_cloud_prep) {
      if (target_cloud_prep->size() == gicp_keyframe_prep_.cloud->size()) {
        target_kdtree = gicp_keyframe_prep_.kdtree;
      } else {
        target_kdtree = [&]() {
          SIO_PROFILE((*sio_prof), sio_frame_id, "gicp", "build_target_kdtree");
          return std::make_shared<small_gicp::KdTree<small_gicp::PointCloud>>(target_cloud_prep);
        }();
      }

      return AlignGICP(target_cloud_prep, target_kdtree, source_cloud, init_guess, true
                       SIO_PROFILER_CALL(sio_prof, sio_frame_id));
    }
  }

  return EstimatePoseGICP(fallback_target_cloud, source_cloud, init_guess, target_keyframe_ts
                          SIO_PROFILER_CALL(sio_prof, sio_frame_id));
}

ColorICP::GICPResult ColorICP::EstimatePoseGICP(
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr target_cloud,
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr source_cloud, const Eigen::Matrix4d &init_guess,
  double target_keyframe_ts SIO_PROFILER_PARAMS_IMPL) {
  (void)target_keyframe_ts;

  if (!target_cloud || target_cloud->empty() || !source_cloud || source_cloud->empty()) {
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
    return small_gicp::preprocess_points(target_pts, voxel_size, 10, num_threads_);
  }();

  return AlignGICP(target_prep.first, target_prep.second, source_cloud, init_guess, false
                   SIO_PROFILER_CALL(sio_prof, sio_frame_id));
}

ColorICP::GICPResult ColorICP::EstimatePoseStereoGICP(
  const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &source_cam,
  const Eigen::Matrix4d &T_curr_cam_to_prevkf_cam, const CameraIntrinsics &camera,
  const cv::Mat &target_depth, const Eigen::Matrix4d &T_imu_cam,
  const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &target_cam SIO_PROFILER_PARAMS_IMPL) {
  GICPResult out;
  out.T = Eigen::Matrix4f::Identity();
  out.covariance = Eigen::Matrix<double, 6, 6>::Identity() * 1e-2;
  if (!source_cam || source_cam->empty() || !sio_stereo_gicp::CameraIsValid(camera)) {
    return out;
  }

  sio_stereo_gicp::AlignResult aligned;
  {
    SIO_PROFILE((*sio_prof), sio_frame_id, "gicp", "stereo_projective");
    aligned = sio_stereo_gicp::AlignProjective(
      *source_cam, camera, T_curr_cam_to_prevkf_cam, target_depth,
      target_cam ? target_cam.get() : nullptr, max_iterations_,
      static_cast<float>(gicp_voxel_size_));
  }

  const Eigen::Matrix4d T_body =
    sio_stereo_gicp::CameraPoseToBody(aligned.T_target_source, T_imu_cam);
  out.T = T_body.cast<float>();
  out.covariance = sio_stereo_gicp::CameraPoseCovToBody(aligned.covariance, T_imu_cam);
  out.num_inliers = aligned.num_inliers;
  out.iterations = aligned.num_iterations;
  out.converged = aligned.converged;
  out.error = aligned.error;
  out.covariance_from_hessian = aligned.covariance_from_hessian;
  out.downsample_source = aligned.num_inliers;
  out.downsample_target = target_cam ? static_cast<int>(target_cam->size()) : 0;
  return out;
}

// 将6维李代数增量转换为4x4 SE(3)齐次变换矩阵
Eigen::Matrix4d ColorICP::TransformVector6dToMatrix4d(const Eigen::Vector6d &input) {
  Eigen::Matrix4d output = Eigen::Matrix4d::Identity();
  output.block<3, 3>(0, 0) = (Eigen::AngleAxisd(input(2), Eigen::Vector3d::UnitZ())
                              * Eigen::AngleAxisd(input(1), Eigen::Vector3d::UnitY())
                              * Eigen::AngleAxisd(input(0), Eigen::Vector3d::UnitX()))
                               .matrix();
  output.block<3, 1>(0, 3) = input.block<3, 1>(3, 0);
  return output;
}

// 执行Colored ICP配准，融合几何和光度信息进行位姿估计
Eigen::Matrix4d ColorICP::ColorICPRegistration(
  const PointCloudPtr &source, const PointCloudPtr &target,
  const std::vector<Eigen::Vector3d> &target_color_gradient,
  const pcl::search::KdTree<PointT> &target_kdtree,
  Eigen::Matrix4d guess_transformation) {
  if (!source || source->empty() || !target || target->empty())
    return guess_transformation;

  // ── 从缓存的成员变量读取参数（构造时已从 YAML 解析） ──
  const double rotation_epsilon = 1.0 - transformation_epsilon_;
  const double dist_sq_thresh = max_corres_dist_ * max_corres_dist_;
  const double sqrt_lambda = std::sqrt(color_icp_lambda_);
  const double sqrt_1m_lambda = std::sqrt(1.0 - color_icp_lambda_);
  const int cloud_size = static_cast<int>(source->size());
  const int actual_threads = num_threads_;

  Eigen::Matrix4d final_transformation = guess_transformation;

  // ── 预分配变换后源点云缓冲区（每轮迭代复用） ──
  pcl::PointCloud<PointT> source_trans;
  source_trans.points.resize(source->size());
  // ══════════════════════════════════════════════════════════════════════════
  // 迭代优化主循环：每轮融合 NN 搜索 + Gauss-Newton 累加为单次遍历
  // ══════════════════════════════════════════════════════════════════════════
  for (int iter = 0; iter < max_iterations_; ++iter) {
    // 用当前累计变换将 source 变换到估计坐标系
    pcl::transformPointCloud<PointT>(*source, source_trans, final_transformation);

    // ── per-thread Gauss-Newton 累加器（每轮置零，线程独立无竞争） ──
    // JTJ_G / JTr_G: 几何项（point-to-plane）的法方程累加
    // JTJ_C / JTr_C: 光度项（photometric）的法方程累加
    // corres_local:  每线程有效对应数（用于判断本轮是否有足够对应）
    std::vector<Eigen::Matrix6d> JTJ_G_local(actual_threads, Eigen::Matrix6d::Zero());
    std::vector<Eigen::Vector6d> JTr_G_local(actual_threads, Eigen::Vector6d::Zero());
    std::vector<Eigen::Matrix6d> JTJ_C_local(actual_threads, Eigen::Matrix6d::Zero());
    std::vector<Eigen::Vector6d> JTr_C_local(actual_threads, Eigen::Vector6d::Zero());
    std::vector<int> corres_local(actual_threads, 0);

#ifdef _OPENMP
#  pragma omp parallel num_threads(actual_threads)
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
#  pragma omp for schedule(static)
#endif
      for (int i = 0; i < cloud_size; ++i) {

        // ── Step 1: 在 target 中搜索源点 i 的最近邻 ──
        target_kdtree.nearestKSearch(source_trans.points[i], 1, nn_idx, nn_sq);
        if (nn_sq[0] > dist_sq_thresh) // 距离超阈值，丢弃此对应
          continue;
        const int j = nn_idx[0]; // target 中最近邻点索引
        ++corres_local[tid];     // 统计有效对应数

        // ── Step 2: 就地读取源/目标点数据（零拷贝，不再拷贝到子点云） ──
        // q: 变换后的源点坐标   cq/iq: 源点颜色/亮度
        const Eigen::Vector3d q = source_trans.points[i].getVector3fMap().cast<double>();
        const Eigen::Vector3d cq = source_trans.points[i].getRGBVector3i().cast<double>();
        const double iq = cq.dot(rgb_to_intensity_weight_) / 255.0;

        // p: 目标对应点坐标   np: 目标点法向   cp/ip: 目标点颜色/亮度
        // dp: 目标点颜色梯度（由 PrepareColorGradient 预计算）
        const Eigen::Vector3d p = target->points[j].getVector3fMap().cast<double>();
        const Eigen::Vector3d np = target->points[j].getNormalVector3fMap().cast<double>();
        const Eigen::Vector3d cp = target->points[j].getRGBVector3i().cast<double>();
        const double ip = cp.dot(rgb_to_intensity_weight_) / 255.0;
        const Eigen::Vector3d &dp = target_color_gradient[j];

        // ── Step 3: 几何项（point-to-plane） ──
        // 残差: r_geo = (q - p) · n_p              (论文 Eq.19)
        // 雅可比: J_geo = [q × n_p ; n_p]^T        (论文 Eq.30)
        const double r_geo = (q - p).dot(np);
        Eigen::Vector6d J_geo;
        J_geo.head<3>() = q.cross(np);                           // 对旋转分量的导数
        J_geo.tail<3>() = np;                                    // 对平移分量的导数
        JTJ_G_local[tid].noalias() += J_geo * J_geo.transpose(); // J^T J 累加
        JTr_G_local[tid].noalias() += J_geo * r_geo;             // J^T r 累加

        // ── Step 4: 光度项（photometric） ──
        // dpM = dp - (dp · np) np  等价于 dp^T (I - np np^T)，但避免了 3×3
        // 矩阵乘 r_color = I(q_proj) - I(q)，其中 q_proj 是 q 在切平面上的投影
        const Eigen::Vector3d dpM = dp - dp.dot(np) * np;
        const double r_color = ip + dp.dot(q - (q - p).dot(np) * np - p) - iq;
        Eigen::Vector6d J_color;
        J_color.head<3>() = q.cross(dpM); // 光度雅可比对旋转分量  (论文 Eq.28-29)
        J_color.tail<3>() = dpM;          // 光度雅可比对平移分量
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
      JTJ.noalias() += sqrt_lambda * JTJ_G_local[tid] + sqrt_1m_lambda * JTJ_C_local[tid];
      JTr.noalias() += sqrt_lambda * JTr_G_local[tid] + sqrt_1m_lambda * JTr_C_local[tid];
      total_corres += corres_local[tid];
    }

    // 无有效对应或数值异常时提前退出
    if (total_corres == 0 || !JTJ.allFinite() || !JTr.allFinite())
      break;

    // ── 求解 6 维位姿增量 X = [α,β,γ,tx,ty,tz] ──
    // 线性系统: JTJ · X = -JTr（Gauss-Newton 正规方程）
    const Eigen::Vector6d X = JTJ.ldlt().solve(-JTr);
    const Eigen::Matrix4d dT = TransformVector6dToMatrix4d(X); // 增量变换矩阵
    final_transformation = dT * final_transformation;          // 左乘更新累计变换

    // ── 收敛判断 ──
    // 旋转：cos(angle) = (tr(dT_rot) - 1) / 2，越接近 1 增量越小
    // 平移：增量平移的平方和
    const double cos_angle = 0.5 * (dT(0, 0) + dT(1, 1) + dT(2, 2) - 1.0);
    const double trans_sqr = dT(0, 3) * dT(0, 3) + dT(1, 3) * dT(1, 3) + dT(2, 3) * dT(2, 3);
    if (cos_angle >= rotation_epsilon && trans_sqr <= transformation_epsilon_)
      break; // 旋转和平移增量均足够小，认为已收敛
  }

  return final_transformation; // 返回 source → target 的最终 4×4 刚体变换
}

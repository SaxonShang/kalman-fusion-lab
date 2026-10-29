#pragma once

// Compatibility helpers for ESIKF point observations and image preprocessing.
// Standalone registration uses the pinned robotic-front module below.
#include "registration/stereo_gicp.h"

#include "data.h"

#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <pcl/point_cloud.h>
#include <vector>

namespace sio_stereo_gicp {

// ---------------------------------------------------------------------------
// 测量与门控常数（与 docs/SCI_投影StereoGICP.md 一致）
// ---------------------------------------------------------------------------
constexpr double kSigmaUPx = 0.5;   // 源点三角化：u 标准差 (px)
constexpr double kSigmaVPx = 0.5;   // 源点三角化：v 标准差 (px)
constexpr double kSigmaDPx = 0.3;   // 视差标准差 (px)
// 最近像素取整的最大偏差是 0.5 px，目标点横向噪声按这个标准差取。
constexpr double kMatchSigmaUPx = 0.5;
constexpr double kMatchSigmaVPx = 0.5;
constexpr double kEpsM2 = 0.004 * 0.004; // 点协方差数值正则 (m²)
constexpr float kZGateSigma = 3.0f;      // |p'_z - Z| 门控倍数
constexpr float kEdgeSigma = 3.0f;       // 近邻深度跳变门控倍数
// 搜索半径初值：与 GICP 对应距离一致，再按 z 换成像素并夹紧。
constexpr double kInitSearchRadiusM = 0.10;
constexpr int kSearchRadiusMinPx = 2;
constexpr int kSearchRadiusMaxPx = 15;
constexpr int kEdgeRadiusPx = 1; // 边缘跳变只看该切比雪夫半径
// 白化残差范数超过该阈值后，Huber 将权重降为 delta / sqrt(e^T W e)。
constexpr double kHuberDelta = 1.345;
constexpr float kZMin = 0.5f;
constexpr float kZMax = 15.0f;

/** 目标点云溅射到图像平面后的深度图与源点索引图。 */
struct SplatMaps {
  cv::Mat depth; ///< CV_32FC1，无效为 NaN
  cv::Mat index; ///< CV_32SC1，对应 cloud 下标，无效为 -1
};

/** 一次投影关联的结果：源下标、目标点 q、命中像素等。 */
struct ProjectiveMatch {
  int src_index = -1;     ///< 源点云下标
  int splat_index = -1;   ///< splat 索引图上的目标点下标（可无）
  Eigen::Vector3d p_prime = Eigen::Vector3d::Zero(); ///< 变换后源点（目标相机系，m）
  Eigen::Vector3d q = Eigen::Vector3d::Zero();       ///< 命中像素中心光线上的目标点（m）
  Eigen::Vector2d uv = Eigen::Vector2d::Zero();      ///< 命中整数像素 (b_u, b_v)
  double z_j = 0.0;                                  ///< 命中深度 (m)
};

/**
 * @brief 检查双目内参是否可用于三角化 / 投影。
 * @param camera 焦距、主点、基线等
 * @return fx,fy,baseline 有限且为正，且 cx,cy 有限时为 true
 */
bool CameraIsValid(const CameraIntrinsics &camera);

/**
 * @brief 立体模型下深度标准差 σ_Z ≈ (Z²/(fx·B))·σ_d。
 * @param z 深度 (m)
 * @param camera 内参（需有效 fx、baseline）
 * @param sigma_d 视差标准差 (px)
 * @return σ_Z (m)；输入非法时为 +∞
 */
double SigmaZ(double z, const CameraIntrinsics &camera, double sigma_d = kSigmaDPx);

/**
 * @brief 把空间搜索半径换成像素切比雪夫半径并夹紧。
 * @param z 深度 (m)，实现里常用 p'_z
 * @param fx 水平焦距 (px)
 * @param radius_m 空间半径 (m)，默认 0.10
 * @return 像素半径 ∈ [kSearchRadiusMinPx, kSearchRadiusMaxPx]
 */
int SearchRadiusPx(double z, double fx, double radius_m = kInitSearchRadiusM);

/**
 * @brief Huber 权重：mahalanobis_sq = eᵀ W e。
 * @param mahalanobis_sq 白化残差平方
 * @param delta 核宽，默认 1.345（作用在 √s 上）
 * @return 核内 1，核外 delta/√s；非法输入为 0
 */
double HuberWeight(double mahalanobis_sq, double delta = kHuberDelta);

/**
 * @brief 由相机系点经 J_uvd 传播完整 3×3 立体协方差。
 * @param p 左目系三维点 (m)
 * @param camera 内参
 * @param sigma_u,sigma_v,sigma_d 测量标准差 (px)；目标点横向用 kMatchSigma*
 * @param eps 对角正则 (m²)
 * @return Σ (m²)，对称化
 */
Eigen::Matrix3d PointCovarianceFromXyz(const Eigen::Vector3d &p, const CameraIntrinsics &camera,
                                       double sigma_u = kSigmaUPx, double sigma_v = kSigmaVPx,
                                       double sigma_d = kSigmaDPx, double eps = kEpsM2);

/**
 * @brief 视差图转深度图：Z = fx·B/d，量程外置 NaN。
 * @param disparity 视差 (px)，CV_32F 或可转换类型
 * @param camera 内参
 * @param zmin,zmax 有效深度 (m)
 * @return CV_32FC1 深度图；失败时为空 Mat
 */
cv::Mat DepthFromDisparity(const cv::Mat &disparity, const CameraIntrinsics &camera,
                           float zmin = kZMin, float zmax = kZMax);

/**
 * @brief 将目标点云溅射到图像：每像素保留最近深度，可选孔洞填充。
 * @param cloud 目标相机系点云
 * @param camera 内参
 * @param width,height 画布尺寸
 * @param fill_px 孔洞填充最大像素距离；≤0 不填充
 * @param zmin,zmax 溅射深度范围 (m)
 * @return depth + index 图
 */
SplatMaps SplatTarget(const pcl::PointCloud<PointTypeRGB> &cloud, const CameraIntrinsics &camera,
                      int width, int height, int fill_px = 0, float zmin = kZMin,
                      float zmax = kZMax);

/**
 * @brief 投影关联：自适应半径搜索，目标点取命中像素中心光线。
 * @param source_cam 源点云（源相机系）
 * @param R_target_source,t_target_source 当前相对位姿（目标←源）
 * @param depth 目标深度图 CV_32FC1
 * @param camera 内参
 * @param splat_index 可选索引图，用于回填 splat_index
 * @param search_radius 像素半径；小于 0 时按 SearchRadiusPx 自适应
 * @param z_gate_sigma,edge_sigma 深度 / 边缘门控倍数
 * @param zmin,zmax 深度量程 (m)
 * @return 有效对应列表
 */
std::vector<ProjectiveMatch> ProjectiveAssociate(
  const pcl::PointCloud<PointTypeRGB> &source_cam, const Eigen::Matrix3d &R_target_source,
  const Eigen::Vector3d &t_target_source, const cv::Mat &depth, const CameraIntrinsics &camera,
  const cv::Mat &splat_index = cv::Mat(), int search_radius = -1,
  float z_gate_sigma = kZGateSigma, float edge_sigma = kEdgeSigma, float zmin = kZMin,
  float zmax = kZMax);

/**
 * @brief 相机系相对位姿共轭到机体系：T_body = T_imu_cam · T_cam · T_imu_cam⁻¹。
 * @param T_cam_target_source 相机系目标←源
 * @param T_imu_cam 外参 相机→IMU/body
 */
Eigen::Matrix4d CameraPoseToBody(const Eigen::Matrix4d &T_cam_target_source,
                                 const Eigen::Matrix4d &T_imu_cam);

/**
 * @brief 左扰动协方差从相机系伴随变换到机体系：Ad · Σ · Adᵀ。
 * @param cov_cam 相机系 6×6（旋转在前）
 * @param T_imu_cam 外参
 * @return 机体系协方差（对称化）
 */
Eigen::Matrix<double, 6, 6> CameraPoseCovToBody(const Eigen::Matrix<double, 6, 6> &cov_cam,
                                                const Eigen::Matrix4d &T_imu_cam);

} // namespace sio_stereo_gicp

#include "sio_backend_modules.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
#include <limits>
#include <rasterizer.h>

namespace {
  inline void BuildViewMatrix(const ImuStatesGroup &pose, float *view4x4) {
    const Eigen::Matrix3d R_cw = pose.rot_end.transpose();
    const Eigen::Vector3d t_cw = -R_cw * pose.pos_end;

    view4x4[0] = static_cast<float>(R_cw(0, 0));
    view4x4[1] = static_cast<float>(R_cw(0, 1));
    view4x4[2] = static_cast<float>(R_cw(0, 2));
    view4x4[3] = static_cast<float>(t_cw(0));
    view4x4[4] = static_cast<float>(R_cw(1, 0));
    view4x4[5] = static_cast<float>(R_cw(1, 1));
    view4x4[6] = static_cast<float>(R_cw(1, 2));
    view4x4[7] = static_cast<float>(t_cw(1));
    view4x4[8] = static_cast<float>(R_cw(2, 0));
    view4x4[9] = static_cast<float>(R_cw(2, 1));
    view4x4[10] = static_cast<float>(R_cw(2, 2));
    view4x4[11] = static_cast<float>(t_cw(2));
    view4x4[12] = 0.0f;
    view4x4[13] = 0.0f;
    view4x4[14] = 0.0f;
    view4x4[15] = 1.0f;
  }

  inline void BuildProjMatrix(float near_z, float far_z, float tan_fovx, float tan_fovy,
                              float *proj4x4) {
    std::fill(proj4x4, proj4x4 + 16, 0.0f);
    proj4x4[0] = 1.0f / tan_fovx;
    proj4x4[5] = 1.0f / tan_fovy;
    proj4x4[10] = -(far_z + near_z) / (far_z - near_z);
    proj4x4[11] = -(2.0f * far_z * near_z) / (far_z - near_z);
    proj4x4[14] = -1.0f;
  }
} // namespace

// 设置高斯地图快照用于后续渲染和梯度计算
void Pose3DGSOptimizer::SetMapSnapshot(const std::vector<Gaussian3D> &gaussians) {
  gaussians_ = gaussians;
}

// 从向量的指数映射计算SO(3)旋转矩阵
Eigen::Matrix3d Pose3DGSOptimizer::ExpSO3(const Eigen::Vector3d &w) {
  const double theta = w.norm();
  if (theta < 1e-12) {
    return Eigen::Matrix3d::Identity();
  }
  return Eigen::AngleAxisd(theta, w / theta).toRotationMatrix();
}

// 根据se(3)增量更新位姿和旋转
void Pose3DGSOptimizer::UpdatePose(ImuStatesGroup &pose, const Eigen::Matrix<double, 6, 1> &delta) {
  const Eigen::Vector3d d_rot = delta.head<3>();
  const Eigen::Vector3d d_pos = delta.tail<3>();
  pose.rot_end = pose.rot_end * ExpSO3(d_rot);
  pose.pos_end += d_pos;
}

// 使用CUDA渲染高斯地图为BGR图像
cv::Mat Pose3DGSOptimizer::RenderBGR(const ImuStatesGroup &pose, const CameraIntrinsics &cam,
                                     int width, int height,
                                     const Pose3DGSOptimizeOptions &options) const {
  cv::Mat render = cv::Mat::zeros(height, width, CV_8UC3);
  if (gaussians_.empty() || width <= 0 || height <= 0) {
    return render;
  }

  const int P = static_cast<int>(gaussians_.size());
  std::vector<float> means3D(3 * P, 0.0f);
  std::vector<float> colors(3 * P, 0.0f);
  std::vector<float> opacities(P, 1.0f);
  std::vector<float> scales(3 * P, 0.01f);
  std::vector<float> rotations(4 * P, 0.0f);
  for (int i = 0; i < P; ++i) {
    const auto &g = gaussians_[i];
    means3D[3 * i + 0] = g.mean.x();
    means3D[3 * i + 1] = g.mean.y();
    means3D[3 * i + 2] = g.mean.z();
    colors[3 * i + 0] = std::clamp(g.color_feature.x(), 0.0f, 1.0f);
    colors[3 * i + 1] = std::clamp(g.color_feature.y(), 0.0f, 1.0f);
    colors[3 * i + 2] = std::clamp(g.color_feature.z(), 0.0f, 1.0f);
    opacities[i] = std::clamp(g.opacity, 0.0f, 1.0f);

    const Eigen::Vector3f s = g.scale.cwiseMax(1e-4f);
    scales[3 * i + 0] = s.x();
    scales[3 * i + 1] = s.y();
    scales[3 * i + 2] = s.z();

    Eigen::Quaternionf q = g.rotation;
    if (!std::isfinite(q.norm()) || q.norm() < 1e-8f) {
      q = Eigen::Quaternionf::Identity();
    }
    q.normalize();
    rotations[4 * i + 0] = q.w();
    rotations[4 * i + 1] = q.x();
    rotations[4 * i + 2] = q.y();
    rotations[4 * i + 3] = q.z();
  }

  float tan_fovx = static_cast<float>(width) / (2.0f * cam.fx);
  float tan_fovy = static_cast<float>(height) / (2.0f * cam.fy);
  tan_fovx = std::max(tan_fovx, 1e-4f);
  tan_fovy = std::max(tan_fovy, 1e-4f);

  std::vector<float> viewmatrix(16, 0.0f);
  std::vector<float> projmatrix(16, 0.0f);
  BuildViewMatrix(pose, viewmatrix.data());
  BuildProjMatrix(options.min_depth, options.max_depth, tan_fovx, tan_fovy, projmatrix.data());
  const std::vector<float> cam_pos = {static_cast<float>(pose.pos_end.x()),
                                      static_cast<float>(pose.pos_end.y()),
                                      static_cast<float>(pose.pos_end.z())};
  const std::vector<float> background = {0.0f, 0.0f, 0.0f};

  auto alloc_and_copy_to_device = [](const std::vector<float> &host) -> float * {
    if (host.empty()) {
      return nullptr;
    }
    float *ptr = nullptr;
    cudaMalloc(reinterpret_cast<void **>(&ptr), host.size() * sizeof(float));
    cudaMemcpy(ptr, host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice);
    return ptr;
  };

  float *d_means3D = alloc_and_copy_to_device(means3D);
  float *d_colors = alloc_and_copy_to_device(colors);
  float *d_opacities = alloc_and_copy_to_device(opacities);
  float *d_scales = alloc_and_copy_to_device(scales);
  float *d_rotations = alloc_and_copy_to_device(rotations);
  float *d_viewmatrix = alloc_and_copy_to_device(viewmatrix);
  float *d_projmatrix = alloc_and_copy_to_device(projmatrix);
  float *d_cam_pos = alloc_and_copy_to_device(cam_pos);
  float *d_background = alloc_and_copy_to_device(background);

  float *d_out_color = nullptr;
  float *d_out_depth = nullptr;
  float *d_out_opacity = nullptr;
  cudaMalloc(reinterpret_cast<void **>(&d_out_color),
             static_cast<size_t>(3 * width * height) * sizeof(float));
  cudaMalloc(reinterpret_cast<void **>(&d_out_depth),
             static_cast<size_t>(width * height) * sizeof(float));
  cudaMalloc(reinterpret_cast<void **>(&d_out_opacity),
             static_cast<size_t>(width * height) * sizeof(float));

  char *geom_buffer = nullptr;
  size_t geom_size = 0;
  char *binning_buffer = nullptr;
  size_t binning_size = 0;
  char *image_buffer = nullptr;
  size_t image_size = 0;

  auto geom_alloc = [&geom_buffer, &geom_size](size_t n) -> char * {
    if (n > geom_size) {
      if (geom_buffer) {
        cudaFree(geom_buffer);
      }
      cudaMalloc(reinterpret_cast<void **>(&geom_buffer), n);
      geom_size = n;
    }
    return geom_buffer;
  };

  auto binning_alloc = [&binning_buffer, &binning_size](size_t n) -> char * {
    if (n > binning_size) {
      if (binning_buffer) {
        cudaFree(binning_buffer);
      }
      cudaMalloc(reinterpret_cast<void **>(&binning_buffer), n);
      binning_size = n;
    }
    return binning_buffer;
  };

  auto image_alloc = [&image_buffer, &image_size](size_t n) -> char * {
    if (n > image_size) {
      if (image_buffer) {
        cudaFree(image_buffer);
      }
      cudaMalloc(reinterpret_cast<void **>(&image_buffer), n);
      image_size = n;
    }
    return image_buffer;
  };

  (void)CudaRasterizer::Rasterizer::forward(
    geom_alloc, binning_alloc, image_alloc, P,
    /*D=*/0, /*M=*/0, d_background, width, height, d_means3D,
    /*shs=*/nullptr, d_colors, d_opacities, d_scales,
    /*scale_modifier=*/1.0f, d_rotations,
    /*cov3D_precomp=*/nullptr, d_viewmatrix, d_projmatrix, d_cam_pos, tan_fovx, tan_fovy,
    /*prefiltered=*/false, d_out_color, d_out_depth, d_out_opacity,
    /*radii=*/nullptr, /*n_touched=*/nullptr,
    /*debug=*/false);

  std::vector<float> out_color(3 * width * height, 0.0f);
  cudaMemcpy(out_color.data(), d_out_color, static_cast<size_t>(3 * width * height) * sizeof(float),
             cudaMemcpyDeviceToHost);

  for (int y = 0; y < height; ++y) {
    cv::Vec3b *ptr = render.ptr<cv::Vec3b>(y);
    for (int x = 0; x < width; ++x) {
      const int idx = y * width + x;
      const float r = std::clamp(out_color[idx + 0 * width * height], 0.0f, 1.0f);
      const float g = std::clamp(out_color[idx + 1 * width * height], 0.0f, 1.0f);
      const float b = std::clamp(out_color[idx + 2 * width * height], 0.0f, 1.0f);
      ptr[x] =
        cv::Vec3b(static_cast<unsigned char>(b * 255.0f), static_cast<unsigned char>(g * 255.0f),
                  static_cast<unsigned char>(r * 255.0f));
    }
  }

  if (geom_buffer) {
    cudaFree(geom_buffer);
  }
  if (binning_buffer) {
    cudaFree(binning_buffer);
  }
  if (image_buffer) {
    cudaFree(image_buffer);
  }

  cudaFree(d_means3D);
  cudaFree(d_colors);
  cudaFree(d_opacities);
  cudaFree(d_scales);
  cudaFree(d_rotations);
  cudaFree(d_viewmatrix);
  cudaFree(d_projmatrix);
  cudaFree(d_cam_pos);
  cudaFree(d_background);
  cudaFree(d_out_color);
  cudaFree(d_out_depth);
  cudaFree(d_out_opacity);
  return render;
}

// 渲染图像并计算位姿梯度、光度损失和深度损失。
//
// 该函数是后端位姿优化的核心入口，整体流程如下：
// 1) 将当前高斯地图与相机位姿打包为 CUDA 栅格化输入，并执行前向渲染；
// 2) 计算渲染图与观测图的光度残差（可选叠加深度残差），构造像素级损失梯度；
// 3) 调用 rasterizer backward，把像素梯度反传到高斯参数与相机位姿扰动（tau）；
// 4) 汇总 tau 梯度并转换到本工程使用的位姿梯度顺序 [theta, rho]；
// 5) 回填渲染结果、损失、有效性标记与每高斯 3D 均值梯度范数（用于 densify 统计）。
//
// 输入约束：
// - observed_bgr 不能为空，且当前 gaussians_ 不为空；
// - observed_depth/depth_confidence 仅在 depth_loss_weight>0 且尺寸匹配时参与损失。
//
// 输出约定：
// - result->valid=true 表示本次渲染与梯度计算有效；
// - result->grad_pose 为优化器可直接使用的位姿梯度；
// - 返回 false 表示输入无效或渲染结果不可用于优化（如无有效像素）。
bool Pose3DGSOptimizer::RenderAndPoseGradients(
  const cv::Mat &observed_bgr, const ImuStatesGroup &pose, const CameraIntrinsics &cam,
  const Pose3DGSOptimizeOptions &options, const cv::Mat *observed_depth,
  const cv::Mat *depth_confidence, double depth_loss_weight,
  Pose3DGSRenderGradResult *result) const {
  if (!result) {
    return false;
  }
  *result = Pose3DGSRenderGradResult();

  if (observed_bgr.empty() || gaussians_.empty()) {
    return false;
  }

  const int width = observed_bgr.cols;
  const int height = observed_bgr.rows;
  if (width <= 0 || height <= 0) {
    return false;
  }

  const int P = static_cast<int>(gaussians_.size()); // 获取高斯数量
  const int HW = width * height;                     // 像素总数

  // 准备高斯参数与相机参数的CUDA输入格式
  std::vector<float> means3D(3 * P, 0.0f);
  std::vector<float> colors(3 * P, 0.0f);
  std::vector<float> opacities(P, 1.0f);
  std::vector<float> scales(3 * P, 0.01f);
  std::vector<float> rotations(4 * P, 0.0f);

  // 将每个高斯的参数打包到连续的数组中，准备传输到CUDA设备。对于旋转，使用四元数格式以避免奇异性，并确保其归一化。
  for (int i = 0; i < P; ++i) {
    const auto &g = gaussians_[i];
    means3D[3 * i + 0] = g.mean.x();
    means3D[3 * i + 1] = g.mean.y();
    means3D[3 * i + 2] = g.mean.z();
    colors[3 * i + 0] = std::clamp(g.color_feature.x(), 0.0f, 1.0f);
    colors[3 * i + 1] = std::clamp(g.color_feature.y(), 0.0f, 1.0f);
    colors[3 * i + 2] = std::clamp(g.color_feature.z(), 0.0f, 1.0f);
    opacities[i] = std::clamp(g.opacity, 0.0f, 1.0f);

    const Eigen::Vector3f s = g.scale.cwiseMax(1e-4f);
    scales[3 * i + 0] = s.x();
    scales[3 * i + 1] = s.y();
    scales[3 * i + 2] = s.z();

    Eigen::Quaternionf q = g.rotation;
    if (!std::isfinite(q.norm()) || q.norm() < 1e-8f) {
      q = Eigen::Quaternionf::Identity();
    }
    q.normalize();
    rotations[4 * i + 0] = q.w();
    rotations[4 * i + 1] = q.x();
    rotations[4 * i + 2] = q.y();
    rotations[4 * i + 3] = q.z();
  }


  // 计算相机的水平和垂直视场切线值，作为投影矩阵构建的输入。这里使用图像尺寸和相机内参计算，并确保数值稳定性。
  float tan_fovx = static_cast<float>(width) / (2.0f * cam.fx);
  float tan_fovy = static_cast<float>(height) / (2.0f * cam.fy);
  tan_fovx = std::max(tan_fovx, 1e-4f);
  tan_fovy = std::max(tan_fovy, 1e-4f);

  // 构建视图矩阵和投影矩阵，并准备相机位置与背景颜色的CUDA输入。视图矩阵由相机位姿转换得到，投影矩阵由相机内参和选项中的深度范围计算得到。
  std::vector<float> viewmatrix(16, 0.0f); // 4x4视图矩阵，行主序存储
  std::vector<float> projmatrix(16, 0.0f); // 4x4投影矩阵，行主序存储
  std::vector<float> projmatrix_raw(
    16, 0.0f); // 这里保留一份原始投影矩阵的副本，以防后续优化过程中需要使用未修改的版本。
  BuildViewMatrix(pose, viewmatrix.data());
  BuildProjMatrix(options.min_depth, options.max_depth, tan_fovx, tan_fovy, projmatrix.data());
  projmatrix_raw = projmatrix;

  // 相机位置用于某些渲染效果（如基于视角的衰减），背景颜色用于渲染时填充未被高斯覆盖的像素。这些参数也需要传输到CUDA设备。
  const std::vector<float> cam_pos = {static_cast<float>(pose.pos_end.x()),
                                      static_cast<float>(pose.pos_end.y()),
                                      static_cast<float>(pose.pos_end.z())};
  const std::vector<float> background = {0.0f, 0.0f, 0.0f};

  // 辅助函数：分配CUDA内存并将数据从主机复制到设备。对于空输入，返回nullptr以节省资源。
  auto alloc_and_copy_to_device = [](const std::vector<float> &host) -> float * {
    if (host.empty()) {
      return nullptr;
    }
    float *ptr = nullptr;
    cudaMalloc(reinterpret_cast<void **>(&ptr), host.size() * sizeof(float));
    cudaMemcpy(ptr, host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice);
    return ptr;
  };
  // 将准备好的高斯参数、相机参数和渲染配置复制到CUDA设备内存，为后续的前向渲染做好准备。
  float *d_means3D = alloc_and_copy_to_device(means3D);
  float *d_colors = alloc_and_copy_to_device(colors);
  float *d_opacities = alloc_and_copy_to_device(opacities);
  float *d_scales = alloc_and_copy_to_device(scales);
  float *d_rotations = alloc_and_copy_to_device(rotations);
  float *d_viewmatrix = alloc_and_copy_to_device(viewmatrix);
  float *d_projmatrix = alloc_and_copy_to_device(projmatrix);
  float *d_projmatrix_raw = alloc_and_copy_to_device(projmatrix_raw);
  float *d_cam_pos = alloc_and_copy_to_device(cam_pos);
  float *d_background = alloc_and_copy_to_device(background);

  // 为渲染输出分配CUDA内存，包括颜色、深度和不透明度。这些输出将被复制回主机用于损失计算和结果展示。
  float *d_out_color = nullptr;
  float *d_out_depth = nullptr;
  float *d_out_opacity = nullptr;
  // 输出缓冲区的大小取决于图像的像素数量和每个像素的通道数。颜色输出需要3个通道（BGR），深度和不透明度各需要1个通道。
  cudaMalloc(reinterpret_cast<void **>(&d_out_color), static_cast<size_t>(3 * HW) * sizeof(float));
  cudaMalloc(reinterpret_cast<void **>(&d_out_depth), static_cast<size_t>(HW) * sizeof(float));
  cudaMalloc(reinterpret_cast<void **>(&d_out_opacity), static_cast<size_t>(HW) * sizeof(float));

  int *d_radii =
    nullptr; // 栅格化过程中每个像素对应的高斯半径，用于后续梯度计算和优化统计。虽然当前版本的渲染前向函数不使用该参数，但预先分配以备后续功能扩展。
  cudaMalloc(reinterpret_cast<void **>(&d_radii), static_cast<size_t>(P) * sizeof(int));

  // 辅助函数：根据需要动态分配或重用CUDA缓冲区，以满足栅格化过程中对几何数据、分割数据和图像数据的临时存储需求。这种策略可以减少频繁的内存分配和释放，提高性能。
  char *geom_buffer = nullptr; // 用于存储几何相关的临时数据，如顶点信息、变换矩阵等。
  size_t geom_size = 0;
  char *binning_buffer = nullptr; // 用于存储分割相关的临时数据，如栅格化索引等。
  size_t binning_size = 0;
  char *image_buffer = nullptr; // 用于存储图像相关的临时数据，如像素值、深度信息等。
  size_t image_size = 0;

  // 这些分配函数会在每次调用时检查请求的大小是否超过当前缓冲区的容量，如果超过则释放旧缓冲区并分配一个新的缓冲区。这样可以确保在渲染过程中有足够的内存来存储必要的数据，同时避免不必要的内存分配和释放操作。
  auto geom_alloc = [&geom_buffer, &geom_size](size_t n) -> char * {
    if (n > geom_size) {
      if (geom_buffer) {
        cudaFree(geom_buffer);
      }
      cudaMalloc(reinterpret_cast<void **>(&geom_buffer), n);
      geom_size = n;
    }
    return geom_buffer;
  };

  auto binning_alloc = [&binning_buffer, &binning_size](size_t n) -> char * {
    if (n > binning_size) {
      if (binning_buffer) {
        cudaFree(binning_buffer);
      }
      cudaMalloc(reinterpret_cast<void **>(&binning_buffer), n);
      binning_size = n;
    }
    return binning_buffer;
  };

  auto image_alloc = [&image_buffer, &image_size](size_t n) -> char * {
    if (n > image_size) {
      if (image_buffer) {
        cudaFree(image_buffer);
      }
      cudaMalloc(reinterpret_cast<void **>(&image_buffer), n);
      image_size = n;
    }
    return image_buffer;
  };

  // 调用CUDA栅格化器的前向函数执行渲染。该函数将高斯参数、相机参数和渲染配置作为输入，输出渲染结果（颜色、深度、不透明度）到预先分配的CUDA缓冲区中。渲染过程中会使用提供的分配函数来获取必要的临时缓冲区，以支持栅格化算法的执行。
  const int R = CudaRasterizer::Rasterizer::forward(
    geom_alloc, binning_alloc, image_alloc, P,
    /*D=*/0, /*M=*/0, d_background, width, height, d_means3D,
    /*shs=*/nullptr, d_colors, d_opacities, d_scales,
    /*scale_modifier=*/1.0f, d_rotations,
    /*cov3D_precomp=*/nullptr, d_viewmatrix, d_projmatrix, d_cam_pos, tan_fovx, tan_fovy,
    /*prefiltered=*/false, d_out_color, d_out_depth, d_out_opacity, d_radii,
    /*n_touched=*/nullptr,
    /*debug=*/false);
  // 栅格化器的返回值R表示渲染过程中有效像素的数量。如果R为0，表示没有像素被高斯覆盖，渲染结果无效，此时可以直接返回以避免后续的损失计算和梯度反传。
  if (R <= 0) {
    return false;
  }
  // 从CUDA设备复制渲染结果回主机内存，以便进行损失计算和结果展示。颜色输出需要3个通道（BGR），深度和不透明度各需要1个通道。复制完成后，渲染结果将存储在out_color、out_depth和out_opacity中。
  std::vector<float> out_color(3 * HW, 0.0f);
  std::vector<float> out_depth(HW, 0.0f);
  std::vector<float> out_opacity(HW, 0.0f);
  cudaMemcpy(out_color.data(), d_out_color, static_cast<size_t>(3 * HW) * sizeof(float),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(out_depth.data(), d_out_depth, static_cast<size_t>(HW) * sizeof(float),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(out_opacity.data(), d_out_opacity, static_cast<size_t>(HW) * sizeof(float),
             cudaMemcpyDeviceToHost);

  // 初始化渲染结果的输出结构，包括渲染图像、深度图、损失值和有效像素数量。渲染图像和深度图被初始化为全零，以便后续填充渲染结果。
  result->rendered_bgr = cv::Mat::zeros(height, width, CV_8UC3);
  result->rendered_depth = cv::Mat::zeros(height, width, CV_32FC1);

  cv::Mat
    observed; // 将输入的观测图像转换为BGR格式，以确保与渲染结果的颜色通道顺序一致。对于单通道图像，使用OpenCV的cvtColor函数将其转换为三通道BGR格式；对于已经是三通道的图像，直接使用原图。这样可以简化后续的损失计算过程。
  if (observed_bgr.channels() == 1) {
    cv::cvtColor(observed_bgr, observed, cv::COLOR_GRAY2BGR);
  } else {
    observed = observed_bgr;
  }

  // 初始化像素级损失梯度和统计变量，用于后续的损失计算和有效像素统计。dL_dpix存储每个像素的颜色损失梯度，dL_dpix_depth存储每个像素的深度损失梯度。loss和depth_loss分别累计颜色损失和深度损失，valid_pix统计有效像素的数量。
  std::vector<float> dL_dpix(3 * HW, 0.0f);
  std::vector<float> dL_dpix_depth(HW, 0.0f);
  double loss = 0.0;
  double depth_loss = 0.0;
  int valid_pix = 0;

  // 如果启用深度损失且观测深度图有效，将其转换为32位浮点格式以便后续计算。对于深度置信度图，如果存在且有效，也将其转换为32位浮点格式，并将值缩放到[0,
  // 1]范围内。这样可以确保在计算深度损失时使用正确的数据类型和数值范围。
  cv::Mat depth_obs_f32;
  cv::Mat conf_obs_f32;
  const bool use_depth_loss = (depth_loss_weight > 0.0) && observed_depth
                              && !observed_depth->empty() && observed_depth->rows == height
                              && observed_depth->cols == width;
  if (use_depth_loss) {
    if (observed_depth->type() == CV_32FC1) {
      depth_obs_f32 = *observed_depth;
    } else {
      observed_depth->convertTo(depth_obs_f32, CV_32FC1);
    }
    if (depth_confidence && !depth_confidence->empty() && depth_confidence->rows == height
        && depth_confidence->cols == width) {
      if (depth_confidence->type() == CV_32FC1) {
        conf_obs_f32 = *depth_confidence;
      } else {
        depth_confidence->convertTo(conf_obs_f32, CV_32FC1, 1.0 / 255.0);
      }
    }
  }

  // 遍历渲染结果的每个像素，计算颜色损失和深度损失，并构造像素级的损失梯度。对于每个有效像素（由不透明度判断），计算渲染颜色与观测颜色之间的差异，并累积颜色损失。同时，如果启用深度损失且深度值有效，计算渲染深度与观测深度之间的差异，并根据置信度加权累积深度损失。最终，统计有效像素的数量以便后续归一化。
  double depth_weight_sum = 0.0; // 累积深度损失的权重总和，用于后续计算加权平均深度损失。
  // 遍历每个像素，计算渲染结果与观测结果之间的差异，并根据不透明度判断有效像素。对于有效像素，计算颜色损失并累积，同时如果启用深度损失且深度值有效，计算深度损失并根据置信度加权累积。最终统计有效像素数量以便后续归一化。
  for (int y = 0; y < height; ++y) {
    cv::Vec3b *ren_ptr = result->rendered_bgr.ptr<cv::Vec3b>(y);
    float *dep_ptr = result->rendered_depth.ptr<float>(y);
    const cv::Vec3b *obs_ptr = observed.ptr<cv::Vec3b>(y);
    for (int x = 0; x < width; ++x) {
      const int idx = y * width + x;

      const float r = std::clamp(out_color[idx + 0 * HW], 0.0f, 1.0f);
      const float g = std::clamp(out_color[idx + 1 * HW], 0.0f, 1.0f);
      const float b = std::clamp(out_color[idx + 2 * HW], 0.0f, 1.0f);

      ren_ptr[x] =
        cv::Vec3b(static_cast<unsigned char>(b * 255.0f), static_cast<unsigned char>(g * 255.0f),
                  static_cast<unsigned char>(r * 255.0f));
      dep_ptr[x] = out_depth[idx];

      if (out_opacity[idx] <= 1e-6f) {
        continue;
      }

      const float obs_r = static_cast<float>(obs_ptr[x][2]) / 255.0f;
      const float obs_g = static_cast<float>(obs_ptr[x][1]) / 255.0f;
      const float obs_b = static_cast<float>(obs_ptr[x][0]) / 255.0f;

      const float dr = r - obs_r;
      const float dg = g - obs_g;
      const float db = b - obs_b;

      loss += static_cast<double>(dr * dr + dg * dg + db * db);
      ++valid_pix;

      if (use_depth_loss) {
        const float z_pred = out_depth[idx];
        const float z_obs = depth_obs_f32.at<float>(y, x);
        if (std::isfinite(z_pred) && std::isfinite(z_obs) && z_pred > 1e-4f && z_obs > 1e-4f) {
          float conf = 1.0f;
          if (!conf_obs_f32.empty()) {
            conf = std::clamp(conf_obs_f32.at<float>(y, x), 0.0f, 1.0f);
          }
          if (conf > 1e-6f) {
            const double r = static_cast<double>(z_pred - z_obs);
            depth_loss += static_cast<double>(conf) * r * r;
            depth_weight_sum += static_cast<double>(conf);
          }
        }
      }
    }
  }

  // 如果没有有效像素或者渲染结果无效（如深度值异常），则释放CUDA资源并返回false，表示本次渲染和梯度计算不可用于优化。
  if (valid_pix == 0 || R <= 0) {
    if (geom_buffer) {
      cudaFree(geom_buffer);
    }
    if (binning_buffer) {
      cudaFree(binning_buffer);
    }
    if (image_buffer) {
      cudaFree(image_buffer);
    }
    cudaFree(d_means3D);
    cudaFree(d_colors);
    cudaFree(d_opacities);
    cudaFree(d_scales);
    cudaFree(d_rotations);
    cudaFree(d_viewmatrix);
    cudaFree(d_projmatrix);
    cudaFree(d_projmatrix_raw);
    cudaFree(d_cam_pos);
    cudaFree(d_background);
    cudaFree(d_out_color);
    cudaFree(d_out_depth);
    cudaFree(d_out_opacity);
    cudaFree(d_radii);
    return false;
  }

  // 归一化颜色损失和深度损失的梯度，以便后续反向传播使用。
  const float inv_norm = 1.0f / static_cast<float>(valid_pix);
  const double inv_depth_norm = (depth_weight_sum > 1e-12) ? (1.0 / depth_weight_sum) : 0.0;
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const int idx = y * width + x;
      if (out_opacity[idx] <= 1e-6f) {
        continue;
      }

      const cv::Vec3b &o = observed.at<cv::Vec3b>(y, x);
      const float obs_r = static_cast<float>(o[2]) / 255.0f;
      const float obs_g = static_cast<float>(o[1]) / 255.0f;
      const float obs_b = static_cast<float>(o[0]) / 255.0f;

      dL_dpix[idx + 0 * HW] = (out_color[idx + 0 * HW] - obs_r) * inv_norm;
      dL_dpix[idx + 1 * HW] = (out_color[idx + 1 * HW] - obs_g) * inv_norm;
      dL_dpix[idx + 2 * HW] = (out_color[idx + 2 * HW] - obs_b) * inv_norm;

      if (use_depth_loss && inv_depth_norm > 0.0) {
        const float z_pred = out_depth[idx];
        const float z_obs = depth_obs_f32.at<float>(y, x);
        if (std::isfinite(z_pred) && std::isfinite(z_obs) && z_pred > 1e-4f && z_obs > 1e-4f) {
          float conf = 1.0f;
          if (!conf_obs_f32.empty()) {
            conf = std::clamp(conf_obs_f32.at<float>(y, x), 0.0f, 1.0f);
          }
          if (conf > 1e-6f) {
            const double r = static_cast<double>(z_pred - z_obs);
            dL_dpix_depth[idx] = static_cast<float>(
              depth_loss_weight * 2.0 * static_cast<double>(conf) * r * inv_depth_norm);
          }
        }
      }
    }
  }

  // 将像素级损失梯度复制到CUDA设备，以便在反向传播中使用。
  float *d_dL_dpix = nullptr;
  float *d_dL_dpix_depth = nullptr;
  cudaMalloc(reinterpret_cast<void **>(&d_dL_dpix), static_cast<size_t>(3 * HW) * sizeof(float));
  cudaMalloc(reinterpret_cast<void **>(&d_dL_dpix_depth), static_cast<size_t>(HW) * sizeof(float));
  cudaMemcpy(d_dL_dpix, dL_dpix.data(), static_cast<size_t>(3 * HW) * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_dL_dpix_depth, dL_dpix_depth.data(), static_cast<size_t>(HW) * sizeof(float),
             cudaMemcpyHostToDevice);

  // 为反向传播分配CUDA内存，用于存储高斯参数和相机位姿扰动的梯度。根据栅格化器的反向函数接口，需要为每个高斯参数（如3D均值、颜色、尺度、旋转等）以及相机位姿扰动（tau）分配对应的梯度缓冲区。这些缓冲区将被反向函数填充，以便后续从设备复制回主机进行结果分析和优化更新。
  float *d_dmean2D = nullptr;
  float *d_dconic = nullptr;
  float *d_dopacity = nullptr;
  float *d_dcolor = nullptr;
  float *d_ddepth = nullptr;
  float *d_dmean3D = nullptr;
  float *d_dcov3D = nullptr;
  float *d_dsh = nullptr;
  float *d_dscale = nullptr;
  float *d_drot = nullptr;
  float *d_dtau = nullptr;

  cudaMalloc(reinterpret_cast<void **>(&d_dmean2D), static_cast<size_t>(3 * P) * sizeof(float));
  cudaMalloc(reinterpret_cast<void **>(&d_dconic), static_cast<size_t>(4 * P) * sizeof(float));
  cudaMalloc(reinterpret_cast<void **>(&d_dopacity), static_cast<size_t>(P) * sizeof(float));
  cudaMalloc(reinterpret_cast<void **>(&d_dcolor), static_cast<size_t>(3 * P) * sizeof(float));
  cudaMalloc(reinterpret_cast<void **>(&d_ddepth), static_cast<size_t>(P) * sizeof(float));
  cudaMalloc(reinterpret_cast<void **>(&d_dmean3D), static_cast<size_t>(3 * P) * sizeof(float));
  cudaMalloc(reinterpret_cast<void **>(&d_dcov3D), static_cast<size_t>(6 * P) * sizeof(float));
  cudaMalloc(reinterpret_cast<void **>(&d_dsh), sizeof(float));
  cudaMalloc(reinterpret_cast<void **>(&d_dscale), static_cast<size_t>(3 * P) * sizeof(float));
  cudaMalloc(reinterpret_cast<void **>(&d_drot), static_cast<size_t>(4 * P) * sizeof(float));
  cudaMalloc(reinterpret_cast<void **>(&d_dtau), static_cast<size_t>(6 * P) * sizeof(float));

  // 调用CUDA栅格化器的反向函数执行梯度计算。该函数将前向渲染的输入参数、渲染结果、像素级损失梯度以及预先分配的梯度缓冲区作为输入，计算每个高斯参数和相机位姿扰动的梯度，并将结果存储在对应的CUDA缓冲区中。反向函数内部会使用前向渲染过程中分配的临时缓冲区来支持梯度计算的执行。
  CudaRasterizer::Rasterizer::backward(
    P, /*D=*/0, /*M=*/0, R, d_background, width, height, d_means3D,
    /*shs=*/nullptr, d_colors, d_scales, /*scale_modifier=*/1.0f, d_rotations,
    /*cov3D_precomp=*/nullptr, d_viewmatrix, d_projmatrix, d_projmatrix_raw, d_cam_pos, tan_fovx,
    tan_fovy, d_radii, geom_buffer, binning_buffer, image_buffer, d_dL_dpix, d_dL_dpix_depth,
    d_dmean2D, d_dconic, d_dopacity, d_dcolor, d_ddepth, d_dmean3D, d_dcov3D, d_dsh, d_dscale,
    d_drot, d_dtau,
    /*debug=*/false);

  // 从CUDA设备复制梯度结果回主机内存，以便进行结果分析和优化更新。对于相机位姿扰动（tau）的梯度，复制到h_dtau中；对于高斯3D均值的梯度，复制到h_dmean3d中。后续将从这些梯度中提取相机位姿的更新方向，并计算每个高斯3D均值梯度的范数用于统计分析。
  std::vector<float> h_dtau(
    6 * P, 0.0f); // 存储相机位姿扰动（tau）的梯度，包含平移和旋转部分，每个高斯对应6个梯度值。
  std::vector<float> h_dmean3d(3 * P, 0.0f); // 存储高斯3D均值的梯度，每个高斯对应3个梯度值。
  cudaMemcpy(h_dtau.data(), d_dtau, static_cast<size_t>(6 * P) * sizeof(float),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(h_dmean3d.data(), d_dmean3D, static_cast<size_t>(3 * P) * sizeof(float),
             cudaMemcpyDeviceToHost);

  Eigen::Matrix<double, 6, 1> grad_tau = Eigen::Matrix<double, 6, 1>::
    Zero(); // 累积相机位姿扰动（tau）的梯度，用于后续计算优化更新的方向。由于每个高斯都会对tau产生梯度贡献，因此需要遍历所有高斯的tau梯度并进行累积。最终，grad_tau将包含所有高斯对相机位姿扰动的总梯度贡献。
  for (int i = 0; i < P; ++i) {
    for (int k = 0; k < 6; ++k) {
      grad_tau[k] += static_cast<double>(h_dtau[6 * i + k]);
    }
  }

  // diff 模块返回顺序是 [rho(平移), theta(旋转)]，本项目使用 [theta, rho]。
  result->grad_pose.head<3>() = grad_tau.segment<3>(3);
  result->grad_pose.tail<3>() = grad_tau.segment<3>(0);
  result->gaussian_mean3d_grad_norm.assign(static_cast<size_t>(P), 0.0f);
  for (int i = 0; i < P; ++i) {
    const float gx = h_dmean3d[3 * i + 0];
    const float gy = h_dmean3d[3 * i + 1];
    const float gz = h_dmean3d[3 * i + 2];
    const float gn = std::sqrt(gx * gx + gy * gy + gz * gz);
    result->gaussian_mean3d_grad_norm[static_cast<size_t>(i)] = std::isfinite(gn) ? gn : 0.0f;
  }
  result->photometric_loss = loss / static_cast<double>(valid_pix);
  result->depth_loss = (depth_weight_sum > 1e-12) ? (depth_loss / depth_weight_sum) : 0.0;
  result->rendered_count = R;
  result->valid = true;

  if (geom_buffer) {
    cudaFree(geom_buffer);
  }
  if (binning_buffer) {
    cudaFree(binning_buffer);
  }
  if (image_buffer) {
    cudaFree(image_buffer);
  }
  cudaFree(d_means3D);
  cudaFree(d_colors);
  cudaFree(d_opacities);
  cudaFree(d_scales);
  cudaFree(d_rotations);
  cudaFree(d_viewmatrix);
  cudaFree(d_projmatrix);
  cudaFree(d_projmatrix_raw);
  cudaFree(d_cam_pos);
  cudaFree(d_background);
  cudaFree(d_out_color);
  cudaFree(d_out_depth);
  cudaFree(d_out_opacity);
  cudaFree(d_radii);
  cudaFree(d_dL_dpix);
  cudaFree(d_dL_dpix_depth);
  cudaFree(d_dmean2D);
  cudaFree(d_dconic);
  cudaFree(d_dopacity);
  cudaFree(d_dcolor);
  cudaFree(d_ddepth);
  cudaFree(d_dmean3D);
  cudaFree(d_dcov3D);
  cudaFree(d_dsh);
  cudaFree(d_dscale);
  cudaFree(d_drot);
  cudaFree(d_dtau);
  return true;
}

// 计算渲染RGB与观察RGB间的光度误差
double Pose3DGSOptimizer::ComputePhotometricCost(const cv::Mat &observed_bgr,
                                                 const ImuStatesGroup &pose,
                                                 const CameraIntrinsics &cam,
                                                 const Pose3DGSOptimizeOptions &options) const {
  if (observed_bgr.empty()) {
    return 1e12;
  }

  cv::Mat observed;
  if (observed_bgr.channels() == 1) {
    cv::cvtColor(observed_bgr, observed, cv::COLOR_GRAY2BGR);
  } else {
    observed = observed_bgr;
  }

  cv::Mat render = RenderBGR(pose, cam, observed.cols, observed.rows, options);

  double total = 0.0;
  int count = 0;
  for (int y = 0; y < observed.rows; ++y) {
    const cv::Vec3b *obs_ptr = observed.ptr<cv::Vec3b>(y);
    const cv::Vec3b *ren_ptr = render.ptr<cv::Vec3b>(y);
    for (int x = 0; x < observed.cols; ++x) {
      const cv::Vec3b &r = ren_ptr[x];
      if (r[0] == 0 && r[1] == 0 && r[2] == 0) {
        continue;
      }
      const cv::Vec3b &o = obs_ptr[x];
      const double db = static_cast<double>(r[0]) - static_cast<double>(o[0]);
      const double dg = static_cast<double>(r[1]) - static_cast<double>(o[1]);
      const double dr = static_cast<double>(r[2]) - static_cast<double>(o[2]);
      total += db * db + dg * dg + dr * dr;
      ++count;
    }
  }

  if (count == 0) {
    return 1e12;
  }
  return total / static_cast<double>(count);
}

// 通过梯度下降优化位姿使其匹配观察图像
bool Pose3DGSOptimizer::OptimizePose(const cv::Mat &observed_bgr, ImuStatesGroup &pose,
                                     const CameraIntrinsics &cam,
                                     const Pose3DGSOptimizeOptions &options,
                                     Pose3DGSOptimizeStats *stats) const {
  if (gaussians_.empty() || observed_bgr.empty()) {
    return false;
  }

  ImuStatesGroup cur = pose;
  double cur_cost = ComputePhotometricCost(observed_bgr, cur, cam, options);

  Pose3DGSOptimizeStats local_stats;
  local_stats.initial_cost = cur_cost;

  for (int iter = 0; iter < options.max_iterations; ++iter) {
    Eigen::Matrix<double, 6, 1> grad = Eigen::Matrix<double, 6, 1>::Zero();

    Pose3DGSRenderGradResult render_grad;
    if (!RenderAndPoseGradients(observed_bgr, cur, cam, options,
                                /*observed_depth=*/nullptr,
                                /*depth_confidence=*/nullptr,
                                /*depth_loss_weight=*/0.0, &render_grad)
        || !render_grad.valid) {
      break;
    }
    cur_cost = render_grad.photometric_loss;
    grad = render_grad.grad_pose;

    const double grad_norm = grad.norm();
    local_stats.iterations = iter + 1;
    if (grad_norm < options.grad_stop_thr) {
      local_stats.converged = true;
      break;
    }

    Eigen::Matrix<double, 6, 1> delta = -options.learning_rate * grad;
    delta.head<3>() = delta.head<3>().cwiseMax(-0.05).cwiseMin(0.05);
    delta.tail<3>() = delta.tail<3>().cwiseMax(-0.10).cwiseMin(0.10);

    ImuStatesGroup candidate = cur;
    UpdatePose(candidate, delta);
    const double candidate_cost = ComputePhotometricCost(observed_bgr, candidate, cam, options);

    if (candidate_cost < cur_cost) {
      cur = candidate;
      cur_cost = candidate_cost;
    } else {
      // 简单回溯，避免梯度过大导致发散
      ImuStatesGroup small_step = cur;
      UpdatePose(small_step, 0.3 * delta);
      const double small_cost = ComputePhotometricCost(observed_bgr, small_step, cam, options);
      if (small_cost < cur_cost) {
        cur = small_step;
        cur_cost = small_cost;
      } else {
        break;
      }
    }
  }

  local_stats.final_cost = cur_cost;
  if (stats) {
    *stats = local_stats;
  }

  const bool improved = local_stats.final_cost < local_stats.initial_cost;
  if (improved) {
    pose = cur;
  }
  return improved;
}

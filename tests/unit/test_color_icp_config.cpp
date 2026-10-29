#include "color_icp.h"
#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>
#include <filesystem>
#include <fstream>
#include <unistd.h>

namespace {
class TemporaryConfig {
public:
  explicit TemporaryConfig(const std::string &content) {
    std::string pattern = (std::filesystem::temp_directory_path() / "sio_color_icp_XXXXXX").string();
    const int descriptor = mkstemp(pattern.data());
    if (descriptor < 0) throw std::runtime_error("Cannot create configuration fixture");
    close(descriptor);
    path = pattern;
    std::ofstream(path) << content;
  }
  ~TemporaryConfig() { std::error_code error; std::filesystem::remove(path, error); }
  std::filesystem::path path;
};
} // namespace

TEST(ColorIcpConfigTest, LoadsStandardYamlBlockSequencesFromExplicitFile) {
  TemporaryConfig config(
    "camera intrinsic:\n  distortion_coefficients:\n  - 0.0\n  - 0.0\n"
    "T_imu_cam:\n  rotation:\n  - 0.0\n  - 0.0\n  - 0.0\n"
    "color_icp:\n  color_weight: srgb\n  num_threads: 1\n");
  EXPECT_NO_THROW({ ColorICP matcher(config.path.string()); });
}

TEST(ColorIcpConfigTest, UsesMainGicpDefaultAndAcceptsExplicitRegistrationType) {
  TemporaryConfig config("color_icp:\n  registration_type: gicp\n  num_threads: 1\n");
  ColorICP matcher(config.path.string());
  EXPECT_EQ(matcher.GetRegistrationType(), ColorICP::RegistrationType::GICP);
  matcher.SetRegistrationType("icp");
  EXPECT_EQ(matcher.GetRegistrationType(), ColorICP::RegistrationType::ICP);
  matcher.SetRegistrationType("plane_icp");
  EXPECT_EQ(matcher.GetRegistrationType(), ColorICP::RegistrationType::PLANE_ICP);
  matcher.SetRegistrationType("stereo");
  EXPECT_EQ(matcher.GetRegistrationType(), ColorICP::RegistrationType::STEREO_GICP);
}

TEST(ColorIcpConfigTest, InvalidInputsReturnDeterministicCovarianceAndConvention) {
  TemporaryConfig config("color_icp: {registration_type: gicp}\n");
  ColorICP matcher(config.path.string());
  auto empty = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>());
  const auto gicp = matcher.EstimatePoseGICP(empty, empty);
  EXPECT_FALSE(gicp.covariance_from_hessian);
  EXPECT_TRUE(gicp.covariance.allFinite());
  EXPECT_EQ(gicp.covariance_convention, ColorICP::CovarianceConvention::RightSource);
  const CameraIntrinsics camera{500, 500, 320, 240, .12f, 640, 480};
  const auto stereo = matcher.EstimatePoseStereoGICP(empty, {}, empty, camera,
    Eigen::Matrix4d::Identity(), Eigen::Matrix4d::Identity());
  EXPECT_FALSE(stereo.covariance_from_hessian);
  EXPECT_EQ(stereo.status, "empty_source");
  EXPECT_EQ(stereo.covariance_convention, ColorICP::CovarianceConvention::LeftTarget);
}

TEST(ColorIcpConfigTest, RetainsLegacyFlatConfigurationSupport) {
  TemporaryConfig config("color_weight: uniform\nnum_threads: 1\nicp_max_iterations: 5\n");
  EXPECT_NO_THROW({ ColorICP matcher(config.path.string()); });
}

TEST(ColorIcpConfigTest, MissingExplicitConfigurationDoesNotUseAnotherDataset) {
  TemporaryConfig config("color_icp: {}\n");
  std::filesystem::remove(config.path);
  EXPECT_THROW({ ColorICP matcher(config.path.string()); }, YAML::BadFile);
}

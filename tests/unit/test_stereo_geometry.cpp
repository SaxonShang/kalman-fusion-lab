#include "sio_stereo_geometry.h"

#include <gtest/gtest.h>

TEST(StereoGeometryTest, CenterCropRectFor1280x720To896x448) {
  sio_utils::StereoImageGeometry geom;
  geom.source_width = 1280;
  geom.source_height = 720;
  geom.inference_width = 896;
  geom.inference_height = 448;

  const sio_utils::StereoCropRect crop = sio_utils::ComputeCenterCropRect(geom);
  EXPECT_EQ(crop.x, 0);
  EXPECT_EQ(crop.y, 40);
  EXPECT_EQ(crop.width, 1280);
  EXPECT_EQ(crop.height, 640);
}

TEST(StereoGeometryTest, AdjustIntrinsicsForCropOnlyOffsetsPrincipalPoint) {
  sio_utils::StereoImageGeometry geom;
  geom.source_width = 1280;
  geom.source_height = 720;
  geom.inference_width = 896;
  geom.inference_height = 448;

  CameraIntrinsics source{};
  source.fx = 533.245f;
  source.fy = 533.26f;
  source.cx = 647.175f;
  source.cy = 370.99f;
  source.baseline = 0.119879f;

  const CameraIntrinsics cropped = sio_utils::AdjustIntrinsicsForCrop(source, geom);

  EXPECT_FLOAT_EQ(cropped.fx, source.fx);
  EXPECT_FLOAT_EQ(cropped.fy, source.fy);
  EXPECT_FLOAT_EQ(cropped.cx, source.cx);
  EXPECT_NEAR(cropped.cy, 330.99f, 1e-3f);
  EXPECT_FLOAT_EQ(cropped.baseline, source.baseline);
}

TEST(StereoGeometryTest, CenterCropOutputSize) {
  cv::Mat image(720, 1280, CV_8UC3, cv::Scalar(10, 20, 30));
  sio_utils::StereoCropRect crop{0, 40, 1280, 640};
  const cv::Mat out = sio_utils::CenterCrop(image, crop);
  ASSERT_FALSE(out.empty());
  EXPECT_EQ(out.cols, 1280);
  EXPECT_EQ(out.rows, 640);
}

TEST(StereoGeometryTest, DisparityToDepthAtCropResolutionMatchesUpscaledDepth) {
  sio_utils::StereoImageGeometry geom;
  geom.source_width = 1280;
  geom.source_height = 720;
  geom.inference_width = 896;
  geom.inference_height = 448;

  CameraIntrinsics inference =
    sio_utils::AdjustIntrinsicsForInference(
      CameraIntrinsics{533.245f, 533.26f, 647.175f, 370.99f, 0.119879f}, geom);

  cv::Mat disp(448, 896, CV_32FC1, cv::Scalar(40.0f));
  const cv::Mat depth_crop =
    sio_utils::DisparityToDepthAtCropResolution(disp, inference, geom, 0.1f, 80.0f);
  ASSERT_FALSE(depth_crop.empty());
  EXPECT_EQ(depth_crop.cols, 1280);
  EXPECT_EQ(depth_crop.rows, 640);
}

TEST(StereoGeometryTest, AdjustIntrinsicsMatchesInferenceResolution) {
  sio_utils::StereoImageGeometry geom;
  geom.source_width = 1280;
  geom.source_height = 720;
  geom.inference_width = 896;
  geom.inference_height = 448;

  CameraIntrinsics source{};
  source.fx = 533.245f;
  source.fy = 533.26f;
  source.cx = 647.175f;
  source.cy = 370.99f;
  source.baseline = 0.119879f;

  const CameraIntrinsics inference =
    sio_utils::AdjustIntrinsicsForInference(source, geom);

  EXPECT_NEAR(inference.fx, 373.2715f, 1e-3f);
  EXPECT_NEAR(inference.fy, 373.282f, 1e-3f);
  EXPECT_NEAR(inference.cx, 453.0225f, 1e-3f);
  EXPECT_NEAR(inference.cy, 231.693f, 1e-3f);
  EXPECT_FLOAT_EQ(inference.baseline, source.baseline);
}

TEST(StereoGeometryTest, CenterCropAndResizeOutputSize) {
  cv::Mat image(720, 1280, CV_8UC3, cv::Scalar(10, 20, 30));
  sio_utils::StereoCropRect crop{0, 40, 1280, 640};
  const cv::Mat out = sio_utils::CenterCropAndResize(image, crop, 896, 448);
  ASSERT_FALSE(out.empty());
  EXPECT_EQ(out.cols, 896);
  EXPECT_EQ(out.rows, 448);
}

TEST(StereoGeometryTest, FitImageToSourceGeometryDownscales1080pTo720p) {
  sio_utils::StereoImageGeometry geom;
  geom.source_width = 1280;
  geom.source_height = 720;

  cv::Mat image(1080, 1920, CV_8UC3, cv::Scalar(1, 2, 3));
  ASSERT_TRUE(sio_utils::FitImageToSourceGeometry(image, geom));
  EXPECT_EQ(image.cols, 1280);
  EXPECT_EQ(image.rows, 720);
}

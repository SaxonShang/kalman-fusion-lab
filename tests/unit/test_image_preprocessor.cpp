#include <gtest/gtest.h>
#include <opencv2/imgproc.hpp>
#include "image_preprocessor.h"

class ImagePreprocessorTest : public ::testing::Test {
protected:
  ImagePreprocessor preprocessor;
  cv::Mat test_image;
};

TEST_F(ImagePreprocessorTest, DefaultConstruction) {
  ImagePreprocessor p;
  EXPECT_TRUE(p.GetConfig().image_preprocess.enable);
  EXPECT_DOUBLE_EQ(p.GetConfig().image_preprocess.target_luminance, 70.0);
  EXPECT_DOUBLE_EQ(p.GetConfig().image_preprocess.saturation_gain, 1.5);
}

TEST_F(ImagePreprocessorTest, EmptyInputHandling) {
  cv::Mat empty;
  cv::Mat output = preprocessor.ProcessAdaptiveLighting(empty);
  EXPECT_TRUE(output.empty());
}

TEST_F(ImagePreprocessorTest, DisabledPreprocessPassesThrough) {
  ImagePreprocessor::Config cfg;
  cfg.image_preprocess.enable = false;
  ImagePreprocessor p(cfg);
  cv::Mat input(60, 80, CV_8UC3, cv::Scalar(30, 40, 20));
  const cv::Mat out = p.ProcessAdaptiveLighting(input);
  ASSERT_FALSE(out.empty());
  EXPECT_EQ(cv::norm(input, out, cv::NORM_INF), 0);
}

TEST_F(ImagePreprocessorTest, ConfigValidationOddDenoiseKernel) {
  ImagePreprocessor::Config cfg;
  cfg.image_preprocess.denoise_ksize = 4;
  ImagePreprocessor p(cfg);
  EXPECT_EQ(p.GetConfig().image_preprocess.denoise_ksize % 2, 1);
}

static double MeanYChannel(const cv::Mat &bgr) {
  cv::Mat ycrcb;
  cv::cvtColor(bgr, ycrcb, cv::COLOR_BGR2YCrCb);
  std::vector<cv::Mat> ch;
  cv::split(ycrcb, ch);
  return cv::mean(ch[0])[0];
}

TEST_F(ImagePreprocessorTest, AdaptiveLightingBrightensDarkImageOnYChannel) {
  ImagePreprocessor::Config cfg;
  cfg.image_preprocess.saturation_enable = false;
  ImagePreprocessor p(cfg);
  cv::Mat dark(120, 160, CV_8UC3, cv::Scalar(20, 25, 15));
  const double y_before = MeanYChannel(dark);
  cv::Mat out = p.ProcessAdaptiveLighting(dark);
  ASSERT_FALSE(out.empty());
  EXPECT_GT(MeanYChannel(out), y_before);

  cv::Mat ycrcb_in, ycrcb_out;
  cv::cvtColor(dark, ycrcb_in, cv::COLOR_BGR2YCrCb);
  cv::cvtColor(out, ycrcb_out, cv::COLOR_BGR2YCrCb);
  std::vector<cv::Mat> in_ch, out_ch;
  cv::split(ycrcb_in, in_ch);
  cv::split(ycrcb_out, out_ch);
  EXPECT_LT(cv::norm(in_ch[1], out_ch[1], cv::NORM_INF), 2.0);
  EXPECT_LT(cv::norm(in_ch[2], out_ch[2], cv::NORM_INF), 2.0);
}

TEST_F(ImagePreprocessorTest, AdaptiveSaturationScalesWithYGain) {
  ImagePreprocessor::Config mild_cfg;
  mild_cfg.image_preprocess.saturation_gain = 1.5;
  mild_cfg.image_preprocess.gain_max = 1.8;
  mild_cfg.image_preprocess.target_luminance = 19.5;
  ImagePreprocessor mild(mild_cfg);

  ImagePreprocessor::Config strong_cfg = mild_cfg;
  strong_cfg.image_preprocess.target_luminance = 70.0;
  ImagePreprocessor strong(strong_cfg);

  cv::Mat colored(120, 160, CV_8UC3, cv::Scalar(15, 80, 40));
  const cv::Mat mild_out = mild.ProcessAdaptiveLighting(colored);
  const cv::Mat strong_out = strong.ProcessAdaptiveLighting(colored);
  ASSERT_FALSE(mild_out.empty());
  ASSERT_FALSE(strong_out.empty());

  auto mean_cr_dev = [](const cv::Mat &bgr) {
    cv::Mat ycrcb;
    cv::cvtColor(bgr, ycrcb, cv::COLOR_BGR2YCrCb);
    std::vector<cv::Mat> ch;
    cv::split(ycrcb, ch);
    cv::Mat cr_f;
    ch[1].convertTo(cr_f, CV_32F);
    return cv::mean(cv::abs(cr_f - 128.0f))[0];
  };

  EXPECT_GT(mean_cr_dev(strong_out), mean_cr_dev(mild_out));
}

TEST_F(ImagePreprocessorTest, SaturationDoesNotChangeYChannel) {
  ImagePreprocessor::Config with_sat_cfg;
  with_sat_cfg.image_preprocess.saturation_enable = true;
  with_sat_cfg.image_preprocess.saturation_gain = 1.5;
  ImagePreprocessor with_sat(with_sat_cfg);

  ImagePreprocessor::Config no_sat_cfg = with_sat_cfg;
  no_sat_cfg.image_preprocess.saturation_enable = false;
  ImagePreprocessor no_sat(no_sat_cfg);

  cv::Mat dark(120, 160, CV_8UC3, cv::Scalar(15, 80, 40));
  const cv::Mat out_with = with_sat.ProcessAdaptiveLighting(dark);
  const cv::Mat out_without = no_sat.ProcessAdaptiveLighting(dark);
  ASSERT_FALSE(out_with.empty());
  ASSERT_FALSE(out_without.empty());

  cv::Mat ycrcb_with, ycrcb_without;
  cv::cvtColor(out_with, ycrcb_with, cv::COLOR_BGR2YCrCb);
  cv::cvtColor(out_without, ycrcb_without, cv::COLOR_BGR2YCrCb);
  std::vector<cv::Mat> ch_with, ch_without;
  cv::split(ycrcb_with, ch_with);
  cv::split(ycrcb_without, ch_without);
  EXPECT_EQ(cv::norm(ch_with[0], ch_without[0], cv::NORM_INF), 0);
  EXPECT_GT(cv::norm(ch_with[1], ch_without[1], cv::NORM_INF), 0);
}

TEST_F(ImagePreprocessorTest, AdaptiveLightingCompressesBrightHighlights) {
  ImagePreprocessor p;
  cv::Mat bright(120, 160, CV_8UC3, cv::Scalar(250, 245, 240));
  cv::Mat out = p.ProcessAdaptiveLighting(bright);
  ASSERT_FALSE(out.empty());
  cv::Mat ycrcb;
  cv::cvtColor(out, ycrcb, cv::COLOR_BGR2YCrCb);
  std::vector<cv::Mat> ch;
  cv::split(ycrcb, ch);
  double max_val = 0.0;
  cv::minMaxLoc(ch[0], nullptr, &max_val);
  EXPECT_LT(max_val, 250.0);
}

TEST_F(ImagePreprocessorTest, StereoAdaptiveLightingPreservesDimensions) {
  ImagePreprocessor p;
  cv::Mat left(100, 120, CV_8UC3, cv::Scalar(30, 35, 25));
  cv::Mat right(100, 120, CV_8UC3, cv::Scalar(35, 30, 28));
  auto [out_left, out_right] = p.ProcessStereoAdaptiveLighting(left, right);
  EXPECT_EQ(out_left.size(), left.size());
  EXPECT_EQ(out_right.size(), right.size());
  EXPECT_GT(MeanYChannel(out_left), MeanYChannel(left));
  EXPECT_GT(MeanYChannel(out_right), MeanYChannel(right));
}

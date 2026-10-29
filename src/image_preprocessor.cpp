#include "image_preprocessor.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <yaml-cpp/yaml.h>

namespace {

using ImagePreprocessConfig = ImagePreprocessor::ImagePreprocessConfig;

enum class BrightnessClass { kDark, kNormal, kBright };

BrightnessClass ClassifyBrightness(double raw_median, double highlight_p99,
                                 const ImagePreprocessConfig &cfg) {
  if (raw_median > cfg.bright_threshold || highlight_p99 >= cfg.highlight_ceiling) {
    return BrightnessClass::kBright;
  }
  if (raw_median < cfg.dark_threshold) {
    return BrightnessClass::kDark;
  }
  return BrightnessClass::kNormal;
}

struct LuminanceStats {
  double raw_median = 0.0;
  double midtone_median = 0.0;
  double highlight = 0.0;
};

double Percentile(const std::vector<uchar> &values, double pct) {
  if (values.empty()) {
    return 0.0;
  }
  std::vector<uchar> sorted = values;
  std::sort(sorted.begin(), sorted.end());
  const double idx = (pct / 100.0) * static_cast<double>(sorted.size() - 1);
  const size_t lo = static_cast<size_t>(std::floor(idx));
  const size_t hi = static_cast<size_t>(std::ceil(idx));
  if (lo == hi) {
    return static_cast<double>(sorted[lo]);
  }
  const double t = idx - static_cast<double>(lo);
  return sorted[lo] * (1.0 - t) + sorted[hi] * t;
}

void CollectYChannelValues(const cv::Mat &y, std::vector<uchar> &all,
                           std::vector<uchar> &midtone, const ImagePreprocessConfig &cfg) {
  all.clear();
  midtone.clear();
  all.reserve(static_cast<size_t>(y.rows * y.cols));
  midtone.reserve(static_cast<size_t>(y.rows * y.cols));

  const int lo = static_cast<int>(cfg.shadow_exclude_below);
  const int hi = static_cast<int>(cfg.highlight_exclude_above);
  for (int r = 0; r < y.rows; ++r) {
    const uchar *row = y.ptr<uchar>(r);
    for (int c = 0; c < y.cols; ++c) {
      const uchar v = row[c];
      all.push_back(v);
      if (v >= lo && v <= hi) {
        midtone.push_back(v);
      }
    }
  }
}

LuminanceStats ComputeLuminanceStats(const cv::Mat &image_bgr, const ImagePreprocessConfig &cfg) {
  LuminanceStats stats;
  cv::Mat ycrcb;
  cv::cvtColor(image_bgr, ycrcb, cv::COLOR_BGR2YCrCb);
  std::vector<cv::Mat> channels;
  cv::split(ycrcb, channels);
  const cv::Mat &y = channels[0];

  std::vector<uchar> all_y;
  std::vector<uchar> mid_y;
  CollectYChannelValues(y, all_y, mid_y, cfg);

  const std::vector<uchar> &stat_y =
    mid_y.size() >= all_y.size() * 5 / 100 ? mid_y : all_y;
  stats.raw_median = Percentile(all_y, cfg.luminance_percentile);
  stats.midtone_median = Percentile(stat_y, cfg.luminance_percentile);
  stats.highlight = Percentile(all_y, cfg.highlight_percentile);
  return stats;
}

LuminanceStats ComputeLuminanceStatsGray(const cv::Mat &gray, const ImagePreprocessConfig &cfg) {
  LuminanceStats stats;
  std::vector<uchar> all_y;
  std::vector<uchar> mid_y;
  CollectYChannelValues(gray, all_y, mid_y, cfg);

  const std::vector<uchar> &stat_y =
    mid_y.size() >= all_y.size() * 5 / 100 ? mid_y : all_y;
  stats.raw_median = Percentile(all_y, cfg.luminance_percentile);
  stats.midtone_median = Percentile(stat_y, cfg.luminance_percentile);
  stats.highlight = Percentile(all_y, cfg.highlight_percentile);
  return stats;
}

std::pair<double, double> ComputeAdaptiveGain(double luminance, double highlight_lum,
                                                    const ImagePreprocessConfig &cfg) {
  double gain = 1.0;
  double highlight_cap = -1.0;

  if (luminance < cfg.target_luminance) {
    gain = cfg.target_luminance / std::max(luminance, 1.0);
    gain = std::min(gain, cfg.gain_max);
  }

  if (cfg.highlight_aware && highlight_lum > 1.0) {
    highlight_cap = cfg.highlight_ceiling / highlight_lum;
    if (highlight_cap < 1.0) {
      gain = std::min(gain, 1.0);
    } else {
      gain = std::min(gain, highlight_cap);
    }
  }
  return {gain, highlight_cap};
}

double ComputeAutoGamma(double midtone, const ImagePreprocessConfig &cfg) {
  const double lo = cfg.gamma_midtone_dark;
  const double hi = cfg.target_luminance;
  if (midtone >= hi) {
    return 1.0;
  }
  if (midtone <= lo) {
    return cfg.gamma_min;
  }
  const double t = (midtone - lo) / (hi - lo);
  return cfg.gamma_min + t * (1.0 - cfg.gamma_min);
}

cv::Mat ApplyLightDenoise(const cv::Mat &image, const ImagePreprocessConfig &cfg) {
  int ksize = cfg.denoise_ksize;
  if (ksize % 2 == 0) {
    ++ksize;
  }
  cv::Mat denoised;
  cv::GaussianBlur(image, denoised, cv::Size(ksize, ksize), cfg.denoise_sigma);
  return denoised;
}

cv::Mat ApplyLinearBrightness(const cv::Mat &image, double gain) {
  if (gain == 1.0) {
    return image;
  }
  cv::Mat adjusted;
  image.convertTo(adjusted, CV_32F);
  adjusted *= static_cast<float>(gain);
  cv::min(adjusted, 255.0f, adjusted);
  adjusted.convertTo(adjusted, CV_8U);
  return adjusted;
}

cv::Mat ApplyGammaCorrection(const cv::Mat &image, double gamma) {
  if (gamma == 1.0) {
    return image;
  }
  uchar lut[256];
  for (int i = 0; i < 256; ++i) {
    lut[i] = static_cast<uchar>(std::clamp(std::pow(i / 255.0, gamma) * 255.0, 0.0, 255.0));
  }
  cv::Mat out;
  cv::LUT(image, cv::Mat(1, 256, CV_8U, lut), out);
  return out;
}

cv::Mat ApplyHighlightRolloff(const cv::Mat &y, const ImagePreprocessConfig &cfg) {
  uchar lut[256];
  const int knee = cfg.highlight_knee;
  for (int i = 0; i < 256; ++i) {
    if (i <= knee) {
      lut[i] = static_cast<uchar>(i);
    } else {
      const int compressed =
        knee + static_cast<int>(cfg.highlight_slope * static_cast<double>(i - knee));
      lut[i] = static_cast<uchar>(std::min(255, compressed));
    }
  }
  cv::Mat out;
  cv::LUT(y, cv::Mat(1, 256, CV_8U, lut), out);
  return out;
}

cv::Mat ProcessYChannel(const cv::Mat &y, const ImagePreprocessConfig &cfg, BrightnessClass bc,
                        double gain, double gamma_midtone, double luminance) {
  if (bc == BrightnessClass::kBright) {
    cv::Mat out = ApplyHighlightRolloff(y, cfg);
    if (cfg.bright_gamma > 1.0) {
      out = ApplyGammaCorrection(out, cfg.bright_gamma);
    }
    return out;
  }

  if (bc != BrightnessClass::kDark || gain <= 1.0) {
    return y;
  }

  cv::Mat out = y;
  if (cfg.denoise) {
    out = ApplyLightDenoise(out, cfg);
  }
  out = ApplyLinearBrightness(out, gain);

  if (cfg.auto_gamma) {
    const double midtone = gamma_midtone >= 0.0 ? gamma_midtone : luminance;
    const double gamma = ComputeAutoGamma(midtone, cfg);
    if (gamma != 1.0) {
      out = ApplyGammaCorrection(out, gamma);
    }
  }
  return out;
}

void ApplyChromaSaturationGain(cv::Mat &cr, cv::Mat &cb, double gain) {
  if (gain == 1.0) {
    return;
  }
  cv::Mat cr_f, cb_f;
  cr.convertTo(cr_f, CV_32F);
  cb.convertTo(cb_f, CV_32F);
  cr_f = 128.0f + static_cast<float>(gain) * (cr_f - 128.0f);
  cb_f = 128.0f + static_cast<float>(gain) * (cb_f - 128.0f);
  cv::min(cr_f, 255.0f, cr_f);
  cv::max(cr_f, 0.0f, cr_f);
  cv::min(cb_f, 255.0f, cb_f);
  cv::max(cb_f, 0.0f, cb_f);
  cr_f.convertTo(cr, CV_8U);
  cb_f.convertTo(cb, CV_8U);
}

double ComputeAdaptiveSaturationGain(double y_gain, const ImagePreprocessConfig &cfg) {
  if (y_gain <= 1.0 || cfg.saturation_gain <= 1.0) {
    return 1.0;
  }
  const double span = std::max(cfg.gain_max - 1.0, 1e-6);
  const double t = std::clamp((y_gain - 1.0) / span, 0.0, 1.0);
  return 1.0 + (cfg.saturation_gain - 1.0) * t;
}

bool ShouldApplySaturation(const ImagePreprocessConfig &cfg, BrightnessClass bc, double gain) {
  if (!cfg.saturation_enable || cfg.saturation_gain == 1.0) {
    return false;
  }
  if (cfg.saturation_only_dark) {
    return bc == BrightnessClass::kDark && gain > 1.0;
  }
  return true;
}

cv::Mat ApplyPreprocess(const cv::Mat &input_image, const ImagePreprocessConfig &cfg,
                                 double preset_gain = -1.0, double gamma_midtone = -1.0,
                                 BrightnessClass preset_bc = BrightnessClass::kNormal,
                                 bool has_preset_bc = false) {
  if (input_image.empty()) {
    return cv::Mat();
  }

  const bool is_color = input_image.channels() == 3;
  const LuminanceStats stats =
    is_color ? ComputeLuminanceStats(input_image, cfg)
             : ComputeLuminanceStatsGray(input_image, cfg);

  const double luminance = stats.midtone_median;
  const double highlight = stats.highlight;
  const BrightnessClass bc =
    has_preset_bc ? preset_bc
                  : ClassifyBrightness(stats.raw_median, highlight, cfg);

  double gain = preset_gain;
  if (gain < 0.0) {
    gain = bc == BrightnessClass::kBright
             ? 1.0
             : ComputeAdaptiveGain(luminance, highlight, cfg).first;
  }

  if (!is_color) {
    return ProcessYChannel(input_image, cfg, bc, gain, gamma_midtone, luminance);
  }

  cv::Mat ycrcb;
  cv::cvtColor(input_image, ycrcb, cv::COLOR_BGR2YCrCb);
  std::vector<cv::Mat> channels;
  cv::split(ycrcb, channels);
  channels[0] = ProcessYChannel(channels[0], cfg, bc, gain, gamma_midtone, luminance);
  if (ShouldApplySaturation(cfg, bc, gain)) {
    const double sat_gain = ComputeAdaptiveSaturationGain(gain, cfg);
    ApplyChromaSaturationGain(channels[1], channels[2], sat_gain);
  }
  cv::merge(channels, ycrcb);
  cv::Mat out;
  cv::cvtColor(ycrcb, out, cv::COLOR_YCrCb2BGR);
  return out;
}

}  // namespace

ImagePreprocessor::ImagePreprocessor() {
  // 使用默认配置
}

ImagePreprocessor::ImagePreprocessor(const Config &cfg) : cfg_(cfg) {
  ValidateConfig();
}

bool ImagePreprocessor::LoadConfig(const std::string &config_path) {
  if (config_path.empty()) {
    std::cerr << "[ImagePreprocessor] Config path is empty" << std::endl;
    return false;
  }

  if (config_path.find(".yaml") != std::string::npos || 
      config_path.find(".yml") != std::string::npos) {
    return LoadConfigFromYAML(config_path);
  }

  std::cerr << "[ImagePreprocessor] Unsupported config format: " << config_path << std::endl;
  return false;
}

bool ImagePreprocessor::LoadConfigFromYAML(const std::string &yaml_path) {
  try {
    YAML::Node config = YAML::LoadFile(yaml_path);

    if (!config["image_preprocessor"]) {
      std::cerr << "[ImagePreprocessor] 'image_preprocessor' section not found in "
                << yaml_path << std::endl;
      return false;
    }

    const auto node = config["image_preprocessor"];
    if (!node["image_preprocess"]) {
      std::cerr << "[ImagePreprocessor] 'image_preprocessor.image_preprocess' not found in "
                << yaml_path << std::endl;
      return false;
    }

    const auto section = node["image_preprocess"];

    auto &pc = cfg_.image_preprocess;
    pc.enable = section["enable"].as<bool>(pc.enable);
    pc.dark_threshold = section["dark_threshold"].as<double>(pc.dark_threshold);
    pc.bright_threshold = section["bright_threshold"].as<double>(pc.bright_threshold);
    pc.luminance_percentile = section["luminance_percentile"].as<double>(pc.luminance_percentile);
    pc.target_luminance = section["target_luminance"].as<double>(pc.target_luminance);
    pc.gain_max = section["gain_max"].as<double>(pc.gain_max);
    pc.denoise = section["denoise"].as<bool>(pc.denoise);
    pc.denoise_ksize = section["denoise_ksize"].as<int>(pc.denoise_ksize);
    pc.denoise_sigma = section["denoise_sigma"].as<double>(pc.denoise_sigma);
    pc.auto_gamma = section["auto_gamma"].as<bool>(pc.auto_gamma);
    pc.gamma_min = section["gamma_min"].as<double>(pc.gamma_min);
    pc.gamma_midtone_dark = section["gamma_midtone_dark"].as<double>(pc.gamma_midtone_dark);
    pc.highlight_aware = section["highlight_aware"].as<bool>(pc.highlight_aware);
    pc.shadow_exclude_below = section["shadow_exclude_below"].as<double>(pc.shadow_exclude_below);
    pc.highlight_exclude_above =
      section["highlight_exclude_above"].as<double>(pc.highlight_exclude_above);
    pc.highlight_ceiling = section["highlight_ceiling"].as<double>(pc.highlight_ceiling);
    pc.highlight_percentile = section["highlight_percentile"].as<double>(pc.highlight_percentile);
    pc.highlight_knee = section["highlight_knee"].as<int>(pc.highlight_knee);
    pc.highlight_slope = section["highlight_slope"].as<double>(pc.highlight_slope);
    pc.bright_gamma = section["bright_gamma"].as<double>(pc.bright_gamma);
    pc.saturation_enable = section["saturation_enable"].as<bool>(pc.saturation_enable);
    pc.saturation_gain = section["saturation_gain"].as<double>(pc.saturation_gain);
    pc.saturation_only_dark = section["saturation_only_dark"].as<bool>(pc.saturation_only_dark);

    cfg_.params_file_path = yaml_path;
    std::cout << "[ImagePreprocessor] Image preprocess config loaded from " << yaml_path
              << std::endl;
    return ValidateConfig();

  } catch (const std::exception &e) {
    std::cerr << "[ImagePreprocessor] Failed to load config: " << e.what() << std::endl;
    return false;
  }
}

bool ImagePreprocessor::ValidateConfig() {
  if (cfg_.image_preprocess.denoise_ksize % 2 == 0) {
    cfg_.image_preprocess.denoise_ksize++;
  }
  return true;
}

cv::Mat ImagePreprocessor::ProcessAdaptiveLighting(const cv::Mat &input_image) {
  if (input_image.empty()) {
    std::cerr << "[ImagePreprocessor] Input image is empty" << std::endl;
    return cv::Mat();
  }
  if (!cfg_.image_preprocess.enable) {
    return input_image;
  }
  return ApplyPreprocess(input_image, cfg_.image_preprocess);
}

std::pair<cv::Mat, cv::Mat> ImagePreprocessor::ProcessStereoAdaptiveLighting(
  const cv::Mat &left, const cv::Mat &right) {
  if (left.empty() || right.empty()) {
    std::cerr << "[ImagePreprocessor] Stereo input image is empty" << std::endl;
    return {left, right};
  }
  if (!cfg_.image_preprocess.enable) {
    return {left, right};
  }

  const ImagePreprocessConfig &cfg = cfg_.image_preprocess;
  const LuminanceStats left_stats = left.channels() == 3 ? ComputeLuminanceStats(left, cfg)
                                                         : ComputeLuminanceStatsGray(left, cfg);
  const LuminanceStats right_stats = right.channels() == 3 ? ComputeLuminanceStats(right, cfg)
                                                           : ComputeLuminanceStatsGray(right, cfg);

  const double raw_median_avg = (left_stats.raw_median + right_stats.raw_median) / 2.0;
  const double midtone_avg = (left_stats.midtone_median + right_stats.midtone_median) / 2.0;
  const double highlight = std::max(left_stats.highlight, right_stats.highlight);
  const BrightnessClass bc = ClassifyBrightness(raw_median_avg, highlight, cfg);
  const double shared_gain =
    bc == BrightnessClass::kBright
      ? 1.0
      : ComputeAdaptiveGain(midtone_avg, highlight, cfg).first;

  cv::Mat left_out =
    ApplyPreprocess(left, cfg, shared_gain, midtone_avg, bc, true);
  cv::Mat right_out =
    ApplyPreprocess(right, cfg, shared_gain, midtone_avg, bc, true);
  return {left_out, right_out};
}

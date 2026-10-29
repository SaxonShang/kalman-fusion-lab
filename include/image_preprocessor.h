#pragma once

/**
 * @file image_preprocessor.h
 * @brief 双目图像自适应光照预处理（Y 通道增强，左右共用增益/gamma）。
 */

#include <opencv2/core.hpp>
#include <string>
#include <utility>
#include <vector>

class ImagePreprocessor {
public:
  struct ImagePreprocessConfig {
    bool enable = true;
    double dark_threshold = 70.0;
    double bright_threshold = 170.0;
    double luminance_percentile = 50.0;
    double target_luminance = 70.0;
    double gain_max = 1.8;
    bool denoise = true;
    int denoise_ksize = 3;
    double denoise_sigma = 0.8;
    bool auto_gamma = true;
    double gamma_min = 0.55;
    double gamma_midtone_dark = 15.0;
    bool highlight_aware = true;
    double shadow_exclude_below = 10.0;
    double highlight_exclude_above = 220.0;
    double highlight_ceiling = 245.0;
    double highlight_percentile = 99.0;
    int highlight_knee = 220;
    double highlight_slope = 0.45;
    double bright_gamma = 1.15;
    bool saturation_enable = true;
    double saturation_gain = 1.5;   ///< gain_max 时 Cr/Cb 增益上限
    bool saturation_only_dark = true;
  };

  struct Config {
    ImagePreprocessConfig image_preprocess;
    std::string params_file_path;
  };

  ImagePreprocessor();
  /** @param cfg 预处理配置 */
  explicit ImagePreprocessor(const Config &cfg);

  /**
   * @brief 从配置文件加载参数。
   * @param config_path YAML 路径
   * @return 加载并校验成功为 true
   */
  bool LoadConfig(const std::string &config_path);

  const Config &GetConfig() const { return cfg_; }
  void SetConfig(const Config &cfg) { cfg_ = cfg; }

  /**
   * @brief 单目自适应光照增强。
   * @param input_image BGR 输入
   * @return 增强后 BGR
   */
  cv::Mat ProcessAdaptiveLighting(const cv::Mat &input_image);

  /**
   * @brief 双目预处理：左右共用 gain/gamma，保持相对光度一致。
   * @param left,right BGR 左右图
   * @return (left_out, right_out)
   */
  std::pair<cv::Mat, cv::Mat> ProcessStereoAdaptiveLighting(const cv::Mat &left,
                                                            const cv::Mat &right);

private:
  Config cfg_;

  /** @brief 校验配置范围合法性。 */
  bool ValidateConfig();
  /** @brief 从 YAML 解析 Config。 */
  bool LoadConfigFromYAML(const std::string &yaml_path);
};

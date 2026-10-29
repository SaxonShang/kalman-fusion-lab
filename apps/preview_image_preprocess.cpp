#include "image_preprocessor.h"

#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <string>

static double MeanHsvSaturation(const cv::Mat &bgr) {
  cv::Mat hsv;
  cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
  std::vector<cv::Mat> ch;
  cv::split(hsv, ch);
  return cv::mean(ch[1])[0];
}

static cv::Mat MakeCompareCanvas(const cv::Mat &a, const cv::Mat &b, const cv::Mat &c,
                                 const std::string &la, const std::string &lb,
                                 const std::string &lc) {
  const int gap = 10;
  cv::Mat canvas(a.rows, a.cols * 3 + gap * 2, CV_8UC3, cv::Scalar(0, 0, 0));
  a.copyTo(canvas(cv::Rect(0, 0, a.cols, a.rows)));
  b.copyTo(canvas(cv::Rect(a.cols + gap, 0, b.cols, b.rows)));
  c.copyTo(canvas(cv::Rect(a.cols * 2 + gap * 2, 0, c.cols, c.rows)));
  cv::putText(canvas, la, cv::Point(10, 28), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 220, 0),
              2);
  cv::putText(canvas, lb, cv::Point(a.cols + gap + 10, 28), cv::FONT_HERSHEY_SIMPLEX, 0.7,
              cv::Scalar(0, 220, 0), 2);
  cv::putText(canvas, lc, cv::Point(a.cols * 2 + gap * 2 + 10, 28), cv::FONT_HERSHEY_SIMPLEX, 0.7,
              cv::Scalar(0, 220, 0), 2);
  return canvas;
}

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "Usage: preview_image_preprocess <input_bgr> <output_compare> [params.yaml]\n";
    return 1;
  }

  const std::string input_path = argv[1];
  const std::string output_path = argv[2];
  const std::string params_path = argc >= 4 ? argv[3] : "config/params.yaml";

  cv::Mat input = cv::imread(input_path, cv::IMREAD_COLOR);
  if (input.empty()) {
    std::cerr << "Failed to read: " << input_path << std::endl;
    return 1;
  }

  ImagePreprocessor with_sat;
  if (!with_sat.LoadConfig(params_path)) {
    std::cerr << "Using default image preprocess config\n";
  }

  ImagePreprocessor::Config no_sat_cfg = with_sat.GetConfig();
  no_sat_cfg.image_preprocess.saturation_enable = false;
  ImagePreprocessor without_sat(no_sat_cfg);

  cv::Mat out_sat = with_sat.ProcessAdaptiveLighting(input);
  cv::Mat out_no_sat = without_sat.ProcessAdaptiveLighting(input);
  if (out_sat.empty() || out_no_sat.empty()) {
    std::cerr << "Preprocessing failed\n";
    return 1;
  }

  const auto &pc = with_sat.GetConfig().image_preprocess;
  cv::Mat canvas = MakeCompareCanvas(input, out_no_sat, out_sat, "original", "Y only",
                                     "Y + adaptive sat");

  if (!cv::imwrite(output_path, canvas)) {
    std::cerr << "Failed to write: " << output_path << std::endl;
    return 1;
  }

  std::cout << "HSV S mean: " << MeanHsvSaturation(input) << " -> "
            << MeanHsvSaturation(out_no_sat) << " (Y only) -> " << MeanHsvSaturation(out_sat)
            << " (Y+sat)\n";
  std::cout << "saturation_gain@max=" << pc.saturation_gain
            << " gain_max=" << pc.gain_max
            << " only_dark=" << (pc.saturation_only_dark ? "true" : "false") << std::endl;
  std::cout << "saved: " << output_path << std::endl;
  return 0;
}

#pragma once

#include "data.h"

#include <memory>
#include <opencv2/core.hpp>

namespace sio_test {

inline std::shared_ptr<FrameData> MakeFrame(double t, int frame_id = 0,
                                            int width = 8, int height = 6) {
  auto frame = std::make_shared<FrameData>();
  frame->st_time = t;
  frame->end_time = t;
  frame->frame_id = frame_id;
  frame->imgl = cv::Mat(height, width, CV_8UC3, cv::Scalar(10, 20, 30));
  frame->imgr = cv::Mat(height, width, CV_8UC3, cv::Scalar(30, 20, 10));
  return frame;
}

} // namespace sio_test

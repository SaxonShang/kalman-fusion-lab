#pragma once

struct CameraIntrinsics {
  float fx, fy, cx, cy, baseline;
  int image_width = 0;
  int image_height = 0;
};

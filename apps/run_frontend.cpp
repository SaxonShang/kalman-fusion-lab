#include "frontend.h"

#include <exception>
#include <iostream>
#include <rclcpp/rclcpp.hpp>

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);

  int exit_code = 0;
  try {
    SIOFrontend frontend(nullptr);
    frontend.run();
  } catch (const std::exception &e) {
    std::cerr << "run_frontend failed: " << e.what() << "\n";
    exit_code = 1;
  }

  if (rclcpp::ok()) {
    rclcpp::shutdown();
  }
  return exit_code;
}

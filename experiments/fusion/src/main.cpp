#include "lab.h"
#include <iostream>
#ifdef _OPENMP
#include <omp.h>
#endif

int main(int argc, char **argv) {
  try {
    fusion_lab::Require(argc == 4,
      "Usage: sio_fusion_lab prepare|gicp|eskf|esikf CONFIG.yaml OUTPUT_DIR");
    const std::string mode = argv[1];
    const auto config = YAML::LoadFile(argv[2]);
    const int threads = config["threads"].as<int>(4);
    fusion_lab::Require(threads > 0, "threads must be positive");
    cv::setNumThreads(threads);
#ifdef _OPENMP
    omp_set_num_threads(threads);
#endif
    const std::filesystem::path output = std::string(argv[3]) == "@config"
      ? config["output_dir"].as<std::string>() : argv[3];
    std::filesystem::create_directories(output);
    if (mode == "prepare") fusion_lab::Prepare(config, output);
    else if (mode == "gicp") fusion_lab::RunGicp(config, output);
    else if (mode == "eskf") fusion_lab::RunEskf(config, output);
    else if (mode == "esikf") fusion_lab::RunEsikf(config, output);
    else throw std::runtime_error("Unknown experiment mode: " + mode);
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "[fusion_lab] " << e.what() << '\n';
    return 1;
  }
}

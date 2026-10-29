// Export model observations once; the CPU fusion lab only consumes the cache.
#include "TrtEngine.hpp"
#include "dataset_loader.h"
#include "image_preprocessor.h"
#include "sio_config.h"
#include "sio_stereo_geometry.h"
#include <yaml-cpp/yaml.h>
#include <opencv2/imgproc.hpp>
#include <fstream>
#include <set>
#include <sstream>
#include <chrono>
#include <cmath>
#include <algorithm>

namespace {
void Require(bool ok, const std::string &message) {
  if (!ok) throw std::runtime_error(message);
}
void Upload(TrtEngine &engine, const std::string &name, const cv::Mat &bgr) {
  cv::Mat rgb, fp;
  cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
  rgb.convertTo(fp, CV_32FC3); // Graph contains normalization: RGB 0..255.
  std::vector<cv::Mat> planes;
  cv::split(fp, planes);
  auto *device = static_cast<float *>(engine.getBindingPtr(name));
  for (int channel = 0; channel < 3; ++channel)
    TrtEngine::CheckCuda(cudaMemcpy(device + channel * bgr.total(), planes[channel].ptr<float>(),
      bgr.total() * sizeof(float), cudaMemcpyHostToDevice), "upload image");
}
void Inspect(TrtEngine &engine) {
  for (const auto &name : engine.tensor_names) {
    const auto dims = engine.context->getTensorShape(name.c_str());
    std::cout << (engine.engine->getTensorIOMode(name.c_str()) == nvinfer1::TensorIOMode::kINPUT
      ? "INPUT " : "OUTPUT ") << name << " fp32";
    for (int d = 0; d < dims.nbDims; ++d) std::cout << ' ' << dims.d[d];
    std::cout << '\n';
  }
}
} // namespace

int main(int argc, char **argv) {
  try {
    Require(argc == 3 || argc == 5,
      "Usage: sio_export_disparity inspect|smoke-flow PLAN | export PLAN CONFIG OUTPUT");
    const std::string mode = argv[1];
    TrtEngine engine(argv[2]);
    Inspect(engine);
    if (mode == "inspect") return 0;
    if (mode == "smoke-flow") {
      constexpr size_t pixels = 448 * 896;
      std::vector<float> input(pixels * 3, -1.f), output(pixels * 2);
      for (const auto &name : {"img1", "img2"})
        TrtEngine::CheckCuda(cudaMemcpy(engine.getBindingPtr(name), input.data(),
          input.size() * sizeof(float), cudaMemcpyHostToDevice), "upload RAFT test");
      Require(engine.hasTensor("2206"), "This iter6 export must expose final flow 2206");
      engine.infer(nullptr);
      TrtEngine::CheckCuda(cudaDeviceSynchronize(), "RAFT synchronize");
      TrtEngine::CheckCuda(cudaMemcpy(output.data(), engine.getBindingPtr("2206"),
        output.size() * sizeof(float), cudaMemcpyDeviceToHost), "download flow");
      double max_abs = 0;
      for (float v : output) {
        Require(std::isfinite(v), "Non-finite RAFT output");
        max_abs = std::max(max_abs, std::abs(double(v)));
      }
      std::cout << "Final flow finite; identical black images max_abs_px=" << max_abs << '\n';
      return 0;
    }
    Require(mode == "export" && argc == 5, "Unknown mode or argument count");
    const auto cfg = YAML::LoadFile(argv[3]);
    const std::filesystem::path output = argv[4];
    std::filesystem::create_directories(output);
    const auto params = cfg["params_file"].as<std::string>();
    const auto geom = sio_utils::LoadStereoImageGeometry(params);
    Require(geom.inference_width == 896 && geom.inference_height == 448,
      "Provided model requires 896x448 inference geometry");
    for (const auto &name : {"left", "right"}) {
      const auto dims = engine.context->getTensorShape(name);
      Require(dims.nbDims == 4 && dims.d[0] == 1 && dims.d[1] == 3 &&
        dims.d[2] == 448 && dims.d[3] == 896, "Unexpected model input shape");
    }
    const auto dims = engine.context->getTensorShape("disp");
    Require(dims.nbDims == 4 && dims.d[0] == 1 && dims.d[1] == 1 &&
      dims.d[2] == 448 && dims.d[3] == 896, "Unexpected disparity output shape");
    std::set<int> wanted;
    for (const auto &pair : cfg["pairs"]) {
      wanted.insert(pair["anchor"].as<int>()); wanted.insert(pair["current"].as<int>());
    }
    Require(!wanted.empty(), "No frame pairs configured");
    std::ifstream frames(std::filesystem::path(cfg["dataset_dir"].as<std::string>()) / "frames.csv");
    Require(bool(frames), "Cannot open frames.csv");
    std::string line;
    std::getline(frames, line);
    std::vector<int> ids;
    while (std::getline(frames, line)) ids.push_back(std::stoi(line.substr(line.rfind(',') + 1)));
    DatasetLoader loader(cfg["dataset_dir"].as<std::string>(), cfg["imu_file"].as<std::string>());
    loader.ConfigureStereoPreprocess(geom);
    ImagePreprocessor lighting;
    Require(lighting.LoadConfig(params), "Invalid image preprocessing config");
    cv::setNumThreads(4);
    std::size_t saved = 0;
    for (std::size_t index = 0; index < ids.size() && ids[index] <= *wanted.rbegin(); ++index) {
      const auto frame = loader.GrabNextFrame();
      Require(bool(frame), "DatasetLoader rejected frame");
      if (!wanted.count(ids[index])) continue;
      auto stereo = lighting.ProcessStereoAdaptiveLighting(frame->imgl, frame->imgr);
      cv::Mat left, right;
      cv::resize(stereo.first, left, cv::Size(896, 448));
      cv::resize(stereo.second, right, left.size());
      Upload(engine, "left", left); Upload(engine, "right", right);
      const auto start = std::chrono::steady_clock::now();
      engine.infer(nullptr);
      TrtEngine::CheckCuda(cudaDeviceSynchronize(), "disparity synchronize");
      const double inference_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
      cv::Mat disparity(448, 896, CV_32FC1);
      TrtEngine::CheckCuda(cudaMemcpy(disparity.ptr<float>(), engine.getBindingPtr("disp"),
        disparity.total() * sizeof(float), cudaMemcpyDeviceToHost), "download disparity");
      size_t positive = 0;
      for (size_t i = 0; i < disparity.total(); ++i) {
        const float value = disparity.ptr<float>()[i];
        Require(std::isfinite(value), "Non-finite disparity; inspect FP16 precision before use");
        positive += value > 0;
      }
      Require(positive > 100, "Too few positive disparity values");
      const auto path = output / (std::to_string(std::llround(frame->end_time * 1000)) + ".yml.gz");
      Require(!std::filesystem::exists(path), "Disparity already exists: " + path.string());
      cv::FileStorage storage(path.string(), cv::FileStorage::WRITE);
      Require(storage.isOpened(), "Cannot write disparity cache");
      storage << "disparity" << disparity << "source_frame_id" << ids[index];
      double lo, hi; cv::minMaxLoc(disparity, &lo, &hi);
      std::cout << "[disparity] frame=" << ids[index] << " min=" << lo << " max=" << hi
        << " positive=" << positive << " inference_ms=" << inference_ms
        << " export_ms=" << std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - start).count() << '\n';
      ++saved;
    }
    Require(saved == wanted.size(), "Missing selected source frame ids");
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "[model] " << error.what() << '\n'; return 1;
  }
}

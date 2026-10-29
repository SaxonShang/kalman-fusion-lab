#pragma once

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <fstream>
#include <iostream>
#include <string>
#include <stdexcept>
#include <algorithm>
#include <vector>

class Logger : public nvinfer1::ILogger {
  void log(nvinfer1::ILogger::Severity severity, const char *msg) noexcept override {
    if (severity <= nvinfer1::ILogger::Severity::kWARNING)
      std::cout << "[TRT] " << msg << std::endl;
  }
};

class TrtEngine {
public:
  nvinfer1::IRuntime *runtime = nullptr;
  nvinfer1::ICudaEngine *engine = nullptr;
  nvinfer1::IExecutionContext *context = nullptr;

  std::vector<void *> bindings;
  std::vector<std::string> tensor_names;
  inline static Logger gLogger;

  static void CheckCuda(cudaError_t code, const char *operation) {
    if (code != cudaSuccess)
      throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(code));
  }

  explicit TrtEngine(const std::string &plan_file) {
    std::ifstream file(plan_file, std::ios::binary);
    if (!file.good()) {
      throw std::runtime_error("Cannot open TensorRT engine: " + plan_file);
    }

    file.seekg(0, file.end);
    size_t size = static_cast<size_t>(file.tellg());
    file.seekg(0, file.beg);
    std::vector<char> trt_model_stream(size);
    if (size == 0 || !file.read(trt_model_stream.data(), static_cast<std::streamsize>(size)))
      throw std::runtime_error("Empty or incomplete TensorRT engine: " + plan_file);
    file.close();

    try {
      runtime = nvinfer1::createInferRuntime(gLogger);
      if (!runtime) throw std::runtime_error("TensorRT runtime creation failed");
      engine = runtime->deserializeCudaEngine(trt_model_stream.data(), size);
      if (!engine) throw std::runtime_error("TensorRT deserialization failed: " + plan_file);
      context = engine->createExecutionContext();
      if (!context) throw std::runtime_error("TensorRT execution context creation failed");

    int num_tensors = engine->getNbIOTensors();
    bindings.resize(static_cast<size_t>(num_tensors), nullptr);
    tensor_names.resize(static_cast<size_t>(num_tensors));

    // Resolve dynamic batch dimensions before sizing any output buffer.
    for (int i = 0; i < num_tensors; ++i) {
      const char *name = engine->getIOTensorName(i);
      if (engine->getTensorIOMode(name) != nvinfer1::TensorIOMode::kINPUT) continue;
      auto dims = engine->getTensorShape(name);
      bool dynamic = false;
      for (int d = 0; d < dims.nbDims; ++d) dynamic |= dims.d[d] < 0;
      if (dynamic) dims = engine->getProfileShape(name, 0, nvinfer1::OptProfileSelector::kOPT);
      if (!context->setInputShape(name, dims))
        throw std::runtime_error(std::string("Invalid input profile: ") + name);
    }

    for (int i = 0; i < num_tensors; ++i) {
      const char *name = engine->getIOTensorName(i);
      tensor_names[static_cast<size_t>(i)] = name;
      if (engine->getTensorDataType(name) != nvinfer1::DataType::kFLOAT)
        throw std::runtime_error(std::string("Expected FP32 tensor I/O: ") + name);
      nvinfer1::Dims dims = context->getTensorShape(name);
      if (dims.nbDims <= 0) throw std::runtime_error("Unresolved tensor shape");

      size_t vol = 1;
      for (int d = 0; d < dims.nbDims; ++d) {
        if (dims.d[d] <= 0) throw std::runtime_error(std::string("Unresolved dimension: ") + name);
        vol *= static_cast<size_t>(dims.d[d]);
      }

      CheckCuda(cudaMalloc(&bindings[static_cast<size_t>(i)], vol * sizeof(float)), "cudaMalloc");
      if (!context->setTensorAddress(name, bindings[static_cast<size_t>(i)]))
        throw std::runtime_error(std::string("Cannot bind tensor: ") + name);
    }
    } catch (...) { Release(); throw; }
  }

  TrtEngine(const TrtEngine &) = delete;
  TrtEngine &operator=(const TrtEngine &) = delete;
  ~TrtEngine() { Release(); }

  void Release() noexcept {
    for (void *ptr : bindings)
      cudaFree(ptr);
    delete context;
    delete engine;
    delete runtime;
    bindings.clear(); context = nullptr; engine = nullptr; runtime = nullptr;
  }

  void infer(cudaStream_t stream) {
    if (!context->enqueueV3(stream)) throw std::runtime_error("TensorRT enqueueV3 failed");
  }

  bool hasTensor(const std::string &name) const {
    return std::find(tensor_names.begin(), tensor_names.end(), name) != tensor_names.end();
  }

  void *getBindingPtr(const std::string &name) {
    for (size_t i = 0; i < tensor_names.size(); ++i) {
      if (tensor_names[i] == name)
        return bindings[i];
    }
    throw std::runtime_error("Tensor not found: " + name);
  }
};

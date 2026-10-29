#include "profiler/simple_profiler.h"

#ifdef SIO_ENABLE_PROFILING

#include <filesystem>
#include <iomanip>
#include <sstream>
#include <thread>

namespace sio {

namespace {
std::string SerializeEvent(const ProfileEvent &e) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(4);
  oss << "{"
      << R"("run_id":")" << e.run_id << "\","
      << R"("frame_id":)" << e.frame_id << ","
      << R"("module":")" << e.module << "\","
      << R"("stage":")" << e.stage << "\","
      << R"("kind":")" << e.kind << "\","
      << R"("thread_id":)" << e.thread_id << ","
      << R"("duration_ms":)" << e.duration_ms << ","
      << R"("points_in":)" << e.points_in << ","
      << R"("points_out":)" << e.points_out << ","
      << R"("correspondences":)" << e.correspondences << ","
      << R"("iterations":)" << e.iterations << ","
      << R"("dynamic_ratio":)" << e.dynamic_ratio << ","
      << R"("bytes":)" << e.bytes << ","
      << R"("direction":")" << e.direction << "\","
      << R"("valid_projected":)" << e.valid_projected << ","
      << R"("error":")" << e.error << "\""
      << "}";
  return oss.str();
}
} // namespace

Profiler::Profiler(const std::string &run_id) : run_id_(run_id) {
  std::filesystem::path run_dir = std::filesystem::path("runs") / run_id;
  std::filesystem::create_directories(run_dir);
  output_path_ = (run_dir / "trace.jsonl").string();
  output_.open(output_path_, std::ios::out | std::ios::trunc);
  WriteHeader();
}

Profiler::~Profiler() { Flush(); }

void Profiler::WriteHeader() {
  auto now = std::chrono::system_clock::now();
  auto time_t_now = std::chrono::system_clock::to_time_t(now);
  output_ << "# SIO-SLAM Frontend Profiling Trace\n";
  output_ << "# run_id: " << run_id_ << "\n";
  output_ << "# created: " << std::ctime(&time_t_now); // ctime includes \n
  output_ << "# format: one JSON object per line (JSONL)\n";
  output_ << "# ---\n";
  output_.flush();
}

void Profiler::RecordEvent(const ProfileEvent &event) {
  std::lock_guard<std::mutex> lock(mtx_);
  output_ << SerializeEvent(event) << "\n";
  ++count_;
  if (count_ % kFlushEvery == 0) {
    output_.flush();
  }
}

void Profiler::Flush() {
  std::lock_guard<std::mutex> lock(mtx_);
  if (output_.is_open()) {
    output_.flush();
  }
}

double Profiler::NowMs() noexcept {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

uint64_t Profiler::ThisThreadId() noexcept {
  return std::hash<std::thread::id>{}(std::this_thread::get_id());
}

} // namespace sio

#endif // SIO_ENABLE_PROFILING

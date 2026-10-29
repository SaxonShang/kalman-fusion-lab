#pragma once

#ifdef SIO_ENABLE_PROFILING

#include <atomic>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>

namespace sio {

struct ProfileEvent {
  std::string run_id;
  int frame_id = -1;
  const char *module = "";
  const char *stage = "";
  const char *kind = "cpu";
  uint64_t thread_id = 0;
  double duration_ms = 0.0;

  int points_in = 0;
  int points_out = 0;
  int correspondences = 0;
  int iterations = 0;
  double dynamic_ratio = 0.0;
  int64_t bytes = 0;
  const char *direction = "none";
  int valid_projected = 0;
  const char *error = "";

  ProfileEvent WithMeta(int pi = 0, int po = 0, int corr = 0, int iters = 0,
                        double dr = 0.0, int64_t b = 0,
                        const char *dir = "none", int vp = 0,
                        const char *err = "") const {
    ProfileEvent e = *this;
    e.points_in = pi;
    e.points_out = po;
    e.correspondences = corr;
    e.iterations = iters;
    e.dynamic_ratio = dr;
    e.bytes = b;
    e.direction = dir;
    e.valid_projected = vp;
    e.error = err;
    return e;
  }
};

class Profiler {
public:
  explicit Profiler(const std::string &run_id);
  ~Profiler();

  void RecordEvent(const ProfileEvent &event);
  void Flush();

  static double NowMs() noexcept;
  static uint64_t ThisThreadId() noexcept;

  const std::string &run_id() const { return run_id_; }

private:
  void WriteHeader();

  std::string run_id_;
  std::string output_path_;
  std::ofstream output_;
  std::mutex mtx_;
  int count_ = 0;
  static constexpr int kFlushEvery = 128;
};

class ScopedTimer {
public:
  ScopedTimer(Profiler *prof, int frame_id, const char *module,
              const char *stage) noexcept
      : prof_(prof), frame_id_(frame_id), module_(module), stage_(stage),
        start_ms_(prof ? Profiler::NowMs() : 0.0) {}

  ~ScopedTimer() {
    if (prof_) {
      ProfileEvent evt;
      evt.run_id = prof_->run_id();
      evt.frame_id = frame_id_;
      evt.module = module_;
      evt.stage = stage_;
      evt.thread_id = Profiler::ThisThreadId();
      evt.duration_ms = Profiler::NowMs() - start_ms_;
      prof_->RecordEvent(evt);
    }
  }

  ScopedTimer(const ScopedTimer &) = delete;
  ScopedTimer &operator=(const ScopedTimer &) = delete;

private:
  Profiler *prof_;
  int frame_id_;
  const char *module_;
  const char *stage_;
  double start_ms_;
};

} // namespace sio

#define SIO_PROFILE_SCOPE_IMPL(prof, fid, mod, stg, line)                      \
  ::sio::ScopedTimer _sio_st_##line(&(prof), fid, mod, stg)

#define SIO_PROFILE(prof, fid, mod, stg)                                       \
  SIO_PROFILE_SCOPE_IMPL(prof, fid, mod, stg, __LINE__)

#define SIO_RECORD_EVENT(prof, fid, mod, stg, dur_ms)                          \
  do {                                                                          \
    ::sio::ProfileEvent _sio_evt_;                                              \
    _sio_evt_.run_id = (prof).run_id();                                         \
    _sio_evt_.frame_id = (fid);                                                 \
    _sio_evt_.module = (mod);                                                   \
    _sio_evt_.stage = (stg);                                                    \
    _sio_evt_.thread_id = ::sio::Profiler::ThisThreadId();                      \
    _sio_evt_.duration_ms = (dur_ms);                                           \
    (prof).RecordEvent(_sio_evt_);                                              \
  } while (0)

#define SIO_RECORD_META(prof, fid, mod, stg, dur_ms, ...)                      \
  do {                                                                          \
    ::sio::ProfileEvent _sio_evt_;                                              \
    _sio_evt_.run_id = (prof).run_id();                                         \
    _sio_evt_.frame_id = (fid);                                                 \
    _sio_evt_.module = (mod);                                                   \
    _sio_evt_.stage = (stg);                                                    \
    _sio_evt_.thread_id = ::sio::Profiler::ThisThreadId();                      \
    _sio_evt_.duration_ms = (dur_ms);                                           \
    _sio_evt_ = _sio_evt_.WithMeta(__VA_ARGS__);                               \
    (prof).RecordEvent(_sio_evt_);                                              \
  } while (0)

// Optional profiler params for function signatures / call sites
// Use SIO_PROFILER_PARAMS in header declarations (with defaults)
// Use SIO_PROFILER_PARAMS_IMPL in .cpp definitions (no defaults, C++ rule)
#define SIO_PROFILER_PARAMS                                                     \
  , ::sio::Profiler *sio_prof = nullptr, int sio_frame_id = -1
#define SIO_PROFILER_PARAMS_IMPL                                                \
  , ::sio::Profiler *sio_prof, int sio_frame_id
#define SIO_PROFILER_CALL(prof, fid) , (prof), (fid)

#else // !SIO_ENABLE_PROFILING

#define SIO_PROFILE(prof, fid, mod, stg) ((void)0)
#define SIO_RECORD_EVENT(prof, fid, mod, stg, dur_ms) ((void)0)
#define SIO_RECORD_META(prof, fid, mod, stg, dur_ms, ...) ((void)0)
#define SIO_PROFILER_PARAMS
#define SIO_PROFILER_PARAMS_IMPL
#define SIO_PROFILER_CALL(prof, fid)

#endif

#include "sio_backend_modules.h"

double ConstantLRScheduler::GetLearningRate(int iteration, double base_lr) const {
  (void)iteration;
  return base_lr;
}

std::string ConstantLRScheduler::GetName() const {
  return "constant";
}

LinearDecayScheduler::LinearDecayScheduler(int max_iterations) : max_iterations_(max_iterations) {
}

double LinearDecayScheduler::GetLearningRate(int iteration, double base_lr) const {
  if (max_iterations_ <= 0)
    return base_lr;
  double progress = static_cast<double>(iteration) / max_iterations_;
  progress = std::min(progress, 1.0);
  return base_lr * (1.0 - progress);
}

std::string LinearDecayScheduler::GetName() const {
  return "linear_decay(max=" + std::to_string(max_iterations_) + ")";
}

ExponentialDecayScheduler::ExponentialDecayScheduler(double decay_rate, int decay_steps)
  : decay_rate_(decay_rate), decay_steps_(decay_steps) {
}

double ExponentialDecayScheduler::GetLearningRate(int iteration, double base_lr) const {
  if (decay_steps_ <= 0)
    return base_lr;
  int steps = iteration / decay_steps_;
  return base_lr * std::pow(decay_rate_, steps);
}

std::string ExponentialDecayScheduler::GetName() const {
  return "exponential_decay(rate=" + std::to_string(decay_rate_) + ")";
}

StepDecayScheduler::StepDecayScheduler(const std::vector<int> &milestones,
                                       const std::vector<double> &decay_rates)
  : milestones_(milestones), decay_rates_(decay_rates) {
  // 确保 milestone 递增
  for (size_t i = 1; i < milestones_.size(); ++i) {
    if (milestones_[i] <= milestones_[i - 1]) {
      milestones_[i] = milestones_[i - 1] + 1;
    }
  }
}

double StepDecayScheduler::GetLearningRate(int iteration, double base_lr) const {
  double lr = base_lr;
  for (size_t i = 0; i < milestones_.size(); ++i) {
    if (iteration >= milestones_[i]) {
      lr *= decay_rates_[i];
    }
  }
  return lr;
}

std::string StepDecayScheduler::GetName() const {
  return "step_decay";
}

CosineAnnealingScheduler::CosineAnnealingScheduler(int max_iterations, double min_lr_ratio)
  : max_iterations_(max_iterations), min_lr_ratio_(min_lr_ratio) {
}

double CosineAnnealingScheduler::GetLearningRate(int iteration, double base_lr) const {
  if (max_iterations_ <= 0)
    return base_lr;
  double t = static_cast<double>(iteration);
  double T = static_cast<double>(max_iterations_);
  t = std::min(t, T);

  double lr_min = base_lr * min_lr_ratio_;
  double lr_max = base_lr;
  double progress = t / T;
  double cosine_decay = (1.0 + std::cos(M_PI * progress)) / 2.0;
  return lr_min + (lr_max - lr_min) * cosine_decay;
}

std::string CosineAnnealingScheduler::GetName() const {
  return "cosine_annealing(max=" + std::to_string(max_iterations_) + ")";
}

MultiStageScheduler::MultiStageScheduler(const std::vector<int> &stages,
                                         const std::vector<double> &lr_ratios)
  : stages_(stages), lr_ratios_(lr_ratios) {
  // stages 和 lr_ratios 长度应一致
  if (stages_.size() != lr_ratios_.size()) {
    if (lr_ratios_.size() < stages_.size()) {
      lr_ratios_.resize(stages_.size(), lr_ratios_.back());
    } else {
      stages_.resize(lr_ratios_.size(), stages_.back() + 1);
    }
  }
  // 确保 stage 递增
  for (size_t i = 1; i < stages_.size(); ++i) {
    if (stages_[i] <= stages_[i - 1]) {
      stages_[i] = stages_[i - 1] + 1;
    }
  }
}

double MultiStageScheduler::GetLearningRate(int iteration, double base_lr) const {
  double ratio = 1.0;
  for (size_t i = 0; i < stages_.size(); ++i) {
    if (iteration >= stages_[i]) {
      ratio = lr_ratios_[i];
    }
  }
  return base_lr * ratio;
}

std::string MultiStageScheduler::GetName() const {
  return "multistage";
}

PolynomialDecayScheduler::PolynomialDecayScheduler(int max_iterations, double power)
  : max_iterations_(max_iterations), power_(power) {
}

double PolynomialDecayScheduler::GetLearningRate(int iteration, double base_lr) const {
  if (max_iterations_ <= 0)
    return base_lr;
  double progress = static_cast<double>(iteration) / max_iterations_;
  progress = std::min(progress, 1.0);
  return base_lr * std::pow(1.0 - progress, power_);
}

std::string PolynomialDecayScheduler::GetName() const {
  return "polynomial_decay(power=" + std::to_string(power_) + ")";
}

// ----------------------------------------------------------------------------
// LearningRateSchedulerFactory Implementation
// ----------------------------------------------------------------------------

std::unique_ptr<LearningRateScheduler> LearningRateSchedulerFactory::Create(
  const std::string &config, int max_iterations) {
  // 辅助：分割字符串
  auto split = [](const std::string &s, char delim) {
    std::vector<std::string> tokens;
    std::string tok;
    for (char c : s) {
      if (c == delim) {
        if (!tok.empty()) {
          tokens.push_back(tok);
          tok.clear();
        }
      } else {
        tok += c;
      }
    }
    if (!tok.empty())
      tokens.push_back(tok);
    return tokens;
  };
  // 辅助：从 "key=value" 对列表中取值（返回 default 若不存在）
  auto get_param = [&](const std::vector<std::string> &parts, const std::string &key,
                       const std::string &def) -> std::string {
    for (const auto &p : parts) {
      auto kv = split(p, '=');
      if (kv.size() == 2 && kv[0] == key)
        return kv[1];
    }
    return def;
  };

  auto parts = split(config, ':');
  if (parts.empty())
    return std::make_unique<ConstantLRScheduler>();

  const std::string &type = parts[0];

  if (type == "constant") {
    return std::make_unique<ConstantLRScheduler>();

  } else if (type == "linear_decay") {
    int mi = std::stoi(get_param(parts, "max_iterations", std::to_string(max_iterations)));
    return std::make_unique<LinearDecayScheduler>(mi);

  } else if (type == "exponential_decay") {
    double rate = std::stod(get_param(parts, "decay_rate", "0.9"));
    int steps = std::stoi(get_param(parts, "decay_steps", "100"));
    return std::make_unique<ExponentialDecayScheduler>(rate, steps);

  } else if (type == "step_decay") {
    // milestones=100,200,300  decay_rates=0.1,0.01,0.001
    auto ms_str = split(get_param(parts, "milestones", ""), ',');
    auto dr_str = split(get_param(parts, "decay_rates", ""), ',');
    std::vector<int> ms;
    std::vector<double> dr;
    for (auto &s : ms_str)
      if (!s.empty())
        ms.push_back(std::stoi(s));
    for (auto &s : dr_str)
      if (!s.empty())
        dr.push_back(std::stod(s));
    if (ms.empty()) {
      ms = {max_iterations / 2};
      dr = {0.1};
    }
    return std::make_unique<StepDecayScheduler>(ms, dr);

  } else if (type == "cosine_annealing") {
    int mi = std::stoi(get_param(parts, "max_iterations", std::to_string(max_iterations)));
    double ratio = std::stod(get_param(parts, "min_lr_ratio", "0.0"));
    return std::make_unique<CosineAnnealingScheduler>(mi, ratio);

  } else if (type == "polynomial_decay") {
    int mi = std::stoi(get_param(parts, "max_iterations", std::to_string(max_iterations)));
    double power = std::stod(get_param(parts, "power", "1.0"));
    return std::make_unique<PolynomialDecayScheduler>(mi, power);

  } else if (type == "multistage") {
    auto st_str = split(get_param(parts, "stages", ""), ',');
    auto rt_str = split(get_param(parts, "ratios", ""), ',');
    std::vector<int> st;
    std::vector<double> rt;
    for (auto &s : st_str)
      if (!s.empty())
        st.push_back(std::stoi(s));
    for (auto &s : rt_str)
      if (!s.empty())
        rt.push_back(std::stod(s));
    if (st.empty()) {
      st = {0};
      rt = {1.0};
    }
    return std::make_unique<MultiStageScheduler>(st, rt);
  }

  // 未知类型回退到恒定学习率
  return std::make_unique<ConstantLRScheduler>();
}

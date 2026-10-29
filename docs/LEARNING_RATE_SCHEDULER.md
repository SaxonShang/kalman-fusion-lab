# 学习率调度器使用指南

## 概述

SIO_SLAM 现已支持**动态学习率调度**功能，允许在优化过程中根据迭代轮次自动调整学习率。这对于提高收敛速度和优化质量非常重要。

## 核心概念

- **基础学习率**（Base Learning Rate）：初始学习率值，定义在 `BackendParams` 中
  - `imu_bias_gyr_lr`: IMU 陀螺仪偏置学习率（默认 0.15）
  - `imu_bias_acc_lr`: IMU 加速度计偏置学习率（默认 0.10）
  - `gs_pose_lr`: 高斯球位姿学习率（默认 0.08）
  - `gs_opacity_lr`: 高斯球参数学习率（默认 0.20）

- **调度策略**（Scheduler Strategy）：定义如何随迭代轮次调整学习率
  - 在 `BackendParams` 中配置四个独立的调度器

## 支持的调度策略

### 1. **constant** - 恒定学习率（默认）
```yaml
pose_lr_scheduler: "constant"
```
学习率保持不变，不进行调度。

### 2. **linear_decay** - 线性衰减
```yaml
pose_lr_scheduler: "linear_decay:max_iterations=1000"
```
公式：`lr(t) = base_lr × (1 - t/max_iterations)`

适用场景：
- 需要逐步平稳降低学习率
- 总迭代次数已知

### 3. **step_decay** - 分段衰减
```yaml
pose_lr_scheduler: "step_decay:milestones=100,200,300:decay_rates=0.1,0.01,0.001"
```
在指定迭代数处降低学习率。

迭代进度：
- 0-99：`base_lr × 1.0`
- 100-199：`base_lr × 0.1`
- 200-299：`base_lr × 0.001`
- 300+：`base_lr × 0.00001`

适用场景：
- 明确的优化阶段划分
- 前期快速优化，后期精细微调

### 4. **exponential_decay** - 指数衰减
```yaml
pose_lr_scheduler: "exponential_decay:decay_rate=0.5:decay_steps=100"
```
公式：`lr(t) = base_lr × decay_rate^(⌊t/decay_steps⌋)`

特点：
- 前期衰减快，后期衰减慢
- 自适应的衰减节奏

适用场景：
- 需要快速初期收敛
- 避免过度优化

### 5. **cosine_annealing** - 余弦退火
```yaml
pose_lr_scheduler: "cosine_annealing:max_iterations=1000:min_lr_ratio=0.1"
```
公式：`lr(t) = lr_min + (lr_max - lr_min) × (1 + cos(π×t/T))/2`

特点：
- 平滑的 S 形衰减曲线
- 前期保持较高学习率，后期快速衰减

适用场景：
- 深度学习优化（特别推荐）
- 需要平衡收敛速度和精度

### 6. **polynomial_decay** - 多项式衰减
```yaml
pose_lr_scheduler: "polynomial_decay:max_iterations=1000:power=2.0"
```
公式：`lr(t) = base_lr × (1 - t/max_iterations)^power`

特点：
- `power=1.0` 时为线性衰减
- `power>1` 时衰减更快
- `power<1` 时衰减更慢

适用场景：
- 需要灵活控制衰减曲线形状的场景

### 7. **multistage** - 多阶段
```yaml
gs_param_lr_scheduler: "multistage:stages=100,200:ratios=1.0,0.5"
```
不同阶段使用完全独立的学习率倍率。

阶段划分：
- 0-99：`base_lr × 1.0`
- 100-199：`base_lr × 0.5`
- 200+：`base_lr × 0.5`

适用场景：
- 有明确的优化阶段（init → main → refine）
- 需要对不同阶段进行差异化优化

## 配置方法

在 `BackendParams` 中设置四个独立的调度器：

```cpp
struct BackendParams {
  // ... 其他参数 ...
  
  std::string pose_lr_scheduler = "constant";     // 位姿学习率调度
  std::string bias_lr_scheduler = "constant";     // 偏置学习率调度
  std::string gs_pose_lr_scheduler = "constant";  // 高斯位姿学习率调度
  std::string gs_param_lr_scheduler = "constant"; // 高斯参数学习率调度
};
```

## 推荐配置方案

### 方案 1：快速收敛（适合初期调试）
```yaml
pose_lr_scheduler: "step_decay:milestones=50,150:decay_rates=0.5,0.1"
gs_param_lr_scheduler: "step_decay:milestones=50,150:decay_rates=0.5,0.1"
```

### 方案 2：平衡方案（生产环境推荐）
```yaml
pose_lr_scheduler: "cosine_annealing:max_iterations=500:min_lr_ratio=0.05"
bias_lr_scheduler: "cosine_annealing:max_iterations=500:min_lr_ratio=0.1"
gs_pose_lr_scheduler: "cosine_annealing:max_iterations=500:min_lr_ratio=0.1"
gs_param_lr_scheduler: "cosine_annealing:max_iterations=500:min_lr_ratio=0.1"
```

### 方案 3：保守优化（强鲁棒性）
```yaml
pose_lr_scheduler: "multistage:stages=100,300:ratios=1.0,0.5"
gs_param_lr_scheduler: "exponential_decay:decay_rate=0.8:decay_steps=50"
```

## 工作原理

### 执行流程

1. **初始化**（`SIOBackend::Start()`）
   - 根据配置字符串创建四个 `LearningRateScheduler` 实例
   - 工厂方法 `LearningRateSchedulerFactory::Create()` 自动解析配置

2. **优化循环**（`SlidingWindowOptimize()`）
   - 每次迭代前，调用调度器计算当前学习率
   - 将动态学习率传入优化器的 `step()` 方法
   - 全局迭代计数器 `global_iteration_` 单调递增

3. **学习率计算**
   ```cpp
   double current_lr = scheduler->GetLearningRate(
       global_iteration_, base_lr);
   ```

### 全局迭代计数器

- 在每个优化步骤后自动递增 `++global_iteration_`
- 用于跟踪整个后端优化过程的迭代进度
- 与 SLAM 的关键帧队列独立，确保一致的学习率调度

## 调试与调优

### 监控学习率变化

在优化循环中可以添加日志：
```cpp
auto logger = rclcpp::get_logger("SIOBackend");
RCLCPP_INFO(logger, 
    "[Scheduler] Iteration %d: pose_lr=%.6f, param_lr=%.6f",
    global_iteration_, curr_pose_lr, curr_gs_param_lr);
```

### 调优策略

1. **学习率过高**（发散）
   - 减小初始学习率
   - 加快衰减速度（更早的 milestones 或更小的 decay_rate）

2. **学习率过低**（收敛慢）
   - 增大初始学习率
   - 减慢衰减速度

3. **优化不稳定**
   - 改用 `cosine_annealing` 策略
   - 增加 `min_lr_ratio` 确保最小学习率不要太小

4. **收敛质量差**
   - 使用 `multistage` 进行分阶段优化
   - 调整各阶段的学习率倍率

## API 参考

### `LearningRateScheduler` 基类

```cpp
class LearningRateScheduler {
  virtual double GetLearningRate(int iteration, double base_lr) const = 0;
  virtual std::string GetName() const = 0;
};
```

### 工厂方法

```cpp
std::unique_ptr<LearningRateScheduler> 
LearningRateSchedulerFactory::Create(
    const std::string& config_string, 
    int max_iterations = 1000);
```

### 在优化器中的使用

```cpp
// 位姿优化
bool BackwardAndStep(
    ImuStatesGroup& state, 
    const ImuStatesGroup& imu_ref,
    double pose_lr,       // 动态学习率
    double bias_gyr_lr,
    double bias_acc_lr);

// 高斯参数优化
void OptimizeGaussianParams(
    double photometric_loss,
    float min_cov,
    float max_cov,
    float opacity_lr);  // 传入动态计算的学习率
```

## 参考文献

- Loshchilov & Hutter, "SGDR: Stochastic Gradient Descent with Warm Restarts" (cosine annealing)
- 常见深度学习框架的学习率调度实现（PyTorch, TensorFlow）

# 创建独立 GitHub 实验仓库并上传

这套步骤创建新的 Git 仓库，不修改原 SIO_SLAM 工作区的 `origin`。导出程序不联网、不运行 git init/commit/push；上传由你执行。

新仓库首先服务于外部人员理解和审阅实验。首页与 [实验审阅说明](EXPERIMENT_REVIEW_BRIEF.md) 提供完整上下文，复跑说明是可选附录，不是参与讨论的前提。

## 已整理的内容

新首页围绕卡尔曼滤波实验介绍目标、数据、设计、结果与限制；保留完整算法源码、实验驱动、历史实现片段、原始结果、P/R/K 矩阵、93 项测试证据、54 帧模型视差、数据摘要及 SHA256。原始双目图像、ONNX、TensorRT 引擎和编译缓存通过准备说明接入。

导出目录：

```text
/home/ubuntu22/slam/SIO_SLAM/build/github-export/kalman-fusion-lab/
```

若以后需要重新导出，请指定尚不存在的目标，避免覆盖已经上传的快照：

```bash
cd /home/ubuntu22/slam/SIO_SLAM
python3 scripts/export_research_repo.py --output build/github-export/kalman-fusion-lab-next
```

## 1. 在 GitHub 新建空仓库

打开 <https://github.com/new>，仓库名称建议为 `kalman-fusion-lab`。按分享范围选择 Private 或 Public；**不要勾选自动添加 README、.gitignore 或 License**，保持空仓库。点击 Create repository 后，复制新仓库的 HTTPS 或 SSH URL。

## 2. 在导出目录提交

```bash
cd /home/ubuntu22/slam/SIO_SLAM/build/github-export/kalman-fusion-lab

# 确认这是独立实验快照，校验归档
test -f results/controls.json
python3 scripts/reproduce_published.py verify

# 只在这个目录初始化一个新仓库
git init -b main
git add .
git status --short
git commit -m "Publish Kalman fusion experiments, results and reproduction guide"
```

如果 commit 提示身份未设置，在当前导出目录设置你自己的姓名和 GitHub 邮箱，再执行 commit；无需修改全局设置：

```bash
git config user.name "你的 GitHub 用户名"
git config user.email "你的 GitHub 邮箱或 noreply 邮箱"
```

## 3. 连接新仓库并上传

将下方 URL 替换为刚创建的新仓库地址：

```bash
git remote add origin https://github.com/你的用户名/kalman-fusion-lab.git
git remote -v
git push -u origin main
```

如已配置 GitHub SSH，可改用 `git@github.com:你的用户名/kalman-fusion-lab.git`。HTTPS 按本机凭据管理器提示认证；若要求密码，应使用 GitHub personal access token，不是账户登录密码。不要把 token 写入远程 URL、文件或提交。

首次 push 完成后，新仓库首页应显示 **Kalman Fusion Lab**，并能看到 `results/`、`experiments/fusion/`、`scripts/`、`docs/` 以及源码目录。Private 仓库需为合作者授予访问权限。

## 后续更新

本目录是一次独立快照。原工作区里的后续修改不会自动同步到这里；后续开发可在新仓库继续，或者从原工作区导出新的快照后审查差异再提交。不要直接把原工作区的 origin 改成新仓库，也不要把旧项目历史或 build/data/models 缓存加入提交。

#!/usr/bin/env python3
"""Export a standalone research snapshot. Does not initialize Git or use a network."""

import argparse
import csv
from datetime import datetime
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

import yaml

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "artifacts/fusion_diagnostics/20261029_105246/manifest.json"
SOURCE_DIRS = ("apps", "config", "docs", "include", "src", "tests", "modules",
               "scripts", "experiments", "submodules", ".vscode")
SOURCE_FILES = ("CMakeLists.txt", "package.xml", "project_config.yaml", "env.sh",
                ".clangd", "requirements-model-tools.txt", "THIRD_PARTY_NOTICES.md")
SKIP_PARTS = {".git", "__pycache__", "build", "install", "local", ".idea"}
SKIP_SUFFIXES = {".pyc", ".so", ".a", ".o", ".onnx", ".plan", ".zip", ".pt"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy_file(source, destination):
    if source.is_symlink():
        raise ValueError(f"Export requires real files: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def copy_tree(source, destination, source_filter=False):
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if path.is_file() and not path.is_symlink():
            if source_filter and (SKIP_PARTS.intersection(relative.parts) or
                                  path.suffix in SKIP_SUFFIXES or
                                  path.name in {"settings.local.json", ".DS_Store"}):
                continue
            copy_file(path, destination / relative)


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True)


def inventory(output):
    rows = []
    for config in sorted((output / "results/fusion").glob("*/resolved.yaml")):
        cfg = yaml.safe_load(config.read_text())
        metrics = config.parent / "metrics.csv"
        records = list(csv.DictReader(metrics.open())) if metrics.is_file() else []
        rows.append({"run": config.parent.name, "sequence": cfg["sequence"],
                     "depth": cfg["depth"]["provider"],
                     "registration": cfg.get("registration_upstream", {}).get("commit", "pre-import"),
                     "completed_stages": ",".join(cfg.get("completed_stages", [])),
                     "records": len(records), "valid": sum(int(r["valid"]) for r in records),
                     "report": f"fusion/{config.parent.name}/report.html" if records else "",
                     "config_sha256": digest(config)})
    with (output / "results/RUNS.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def export(output):
    output = output.expanduser().resolve()
    if output.exists():
        raise ValueError(f"Destination already exists; select a new empty path: {output}")
    if not AUDIT.is_file():
        raise ValueError("This exporter requires the completed 20261029 covariance audit")
    output.mkdir(parents=True)
    for name in SOURCE_DIRS:
        copy_tree(ROOT / name, output / name, source_filter=True)
    for name in SOURCE_FILES:
        copy_file(ROOT / name, output / name)
    copy_file(ROOT / "README.md", output / "README_SIO_SLAM.md")
    copy_file(ROOT / "experiments/publication/README.md", output / "README.md")
    (output / ".gitignore").write_text(
        "# Published evidence is in results/ and must remain tracked.\n"
        "/build/\n/install/\n/log/\n/data/\n/models/\n/artifacts/\n"
        "/project_config.local.yaml\n/compile_commands.json\n"
        "/experiments/fusion/local/\n__pycache__/\n*.pyc\n"
        ".DS_Store\n.vscode/settings.local.json\n.idea/\n")

    copy_tree(ROOT / "artifacts/fusion", output / "results/fusion")
    copy_tree(ROOT / "artifacts/fusion_diagnostics", output / "results/diagnostics")
    copy_tree(ROOT / "data/cache/model_disparity", output / "results/disparity")
    for dataset in sorted((ROOT / "data/datasets/fr5_highspeed").iterdir()):
        if not dataset.is_dir():
            continue
        for name in ("frames.csv", "imu.csv", "groundtruth_tum.txt", "params.yaml", "metadata.json"):
            copy_file(dataset / name, output / "results/inputs" / dataset.name / name)
    recording = ROOT / "data/recordings/recordings_highspeed"
    copy_tree(recording / "calib", output / "results/recording/calib")
    copy_file(recording / "使用说明.md", output / "results/recording/使用说明.md")
    for report in sorted(recording.glob("*/save_data/*_report.json")):
        copy_file(report, output / "results/recording" / report.relative_to(recording))
    for name in ("manifest.json", "README.md", "onnx/disp_cost_volume.verified.json"):
        copy_file(ROOT / "models" / name, output / "results/models" / name)
    model_readme = output / "results/models/README.md"
    model_readme.write_text("此目录归档模型说明、接口与身份校验；ONNX 权重和 .plan 需按复现指南在 models/ 中另行准备。\n\n" +
        model_readme.read_text().replace("本目录已被 Git 忽略，所有模型和产物只保留本地，不执行 push。",
                                      "原工作区的 models/ 被 Git 忽略；本发布目录 results/models/ 随仓库跟踪。"))

    provenance = output / "results/provenance"
    for source, name in (("build/fusion-before-covariance-fixes", "before_covariance_fixes"),
                         ("build/registration-before-robotic-front", "before_registration_import")):
        for path in sorted((ROOT / source).rglob("*")):
            if path.is_file() and path.suffix in {".h", ".hpp", ".cpp"}:
                copy_file(path, provenance / name / path.relative_to(ROOT / source))
    for log in sorted((ROOT / "build").glob("*.log")):
        copy_file(log, output / "results/logs" / log.name)
    for log in sorted((ROOT / "models/tensorrt/logs").glob("*")):
        if log.is_file() and log.suffix in {".log", ".json", ".txt"}:
            target = output / "results/logs/tensorrt" / (log.name + ".gz")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(gzip.compress(log.read_bytes(), mtime=0))

    original = json.loads(AUDIT.read_text())
    manifest = json.loads(AUDIT.read_text())
    for entry in manifest["runs"]:
        for key in ("reference", "solver_only", "repropagated", "sensitivity"):
            entry[key] = "results/fusion/" + Path(entry[key]).name
    manifest["original_manifest"] = "results/diagnostics/20261029_105246/manifest.json"
    (output / "results/controls.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for entry in original["runs"]:
        for treatment in ("reference", "solver_only", "repropagated", "sensitivity"):
            path = Path(entry[treatment])
            cfg = yaml.safe_load((path / "resolved.yaml").read_text())
            dataset = output / "results/inputs" / cfg["sequence"]
            for name, key in (("groundtruth_tum.txt", "groundtruth_sha256"),):
                if digest(dataset / name) != cfg[key]:
                    raise ValueError(f"Archived report input mismatch: {path}")
            if treatment in {"reference", "repropagated"}:
                if digest(path / "observations.yml.gz") != cfg["observations_sha256"]:
                    raise ValueError(f"Archived observations changed: {path}")

    # Documentation links point at the published copy; original evidence stays byte-identical.
    for path in [output / "README_SIO_SLAM.md", *sorted((output / "docs").glob("*.md")),
                 output / "experiments/fusion/README.md"]:
        content = path.read_text()
        # Only links change: command examples continue to describe new runs in artifacts/.
        content = re.sub(r"(\]\([^)]*)artifacts/fusion_diagnostics/", r"\1results/diagnostics/", content)
        content = re.sub(r"(\]\([^)]*)artifacts/fusion/", r"\1results/fusion/", content)
        path.write_text(content)

    rows = inventory(output)
    (provenance / "workspace-diff.patch").write_text(git("diff", "HEAD", "--", ".",
        ":(exclude)build", ":(exclude).specstory"))
    (provenance / "recent-commits.txt").write_text(git("log", "-12", "--oneline"))
    validation = ROOT / "build/publication-validation.json"
    if validation.is_file():
        copy_file(validation, provenance / "publication-validation.json")
        copy_tree(ROOT / "build/publication-validation", output / "results/validation")
    details = {"exported_at": datetime.now().astimezone().isoformat(),
               "source_repository": "https://github.com/lihongliang0203/SIO_SLAM",
               "source_workspace": str(ROOT), "source_head": git("rev-parse", "HEAD").strip(),
               "source_status": git("status", "--short"),
               "history_note": "Independent snapshot; source HEAD alone does not describe uncommitted experiments.",
               "runs": len(rows), "metric_rows_including_replays": sum(r["records"] for r in rows),
               "excluded": ["Git metadata and old history", "raw stereo images", "ONNX and TensorRT engines",
                            "point-cloud/build/install caches", "private IDE/chat state and compiled binaries"]}
    (provenance / "workspace.json").write_text(json.dumps(details, indent=2, ensure_ascii=False) + "\n")
    (output / "results/README.md").write_text(
        "# 归档结果与输入\n\n"
        "只需理解实验时，先读 [实验记录](../实验记录.md)，再看下表中的汇总 CSV 与图；不需要配置环境或重跑。\n\n"
        f"共 {len(rows)} 个执行目录、{sum(r['records'] for r in rows)} 条指标记录，包含历史阶段、回放和调试；重复记录不代表独立样本。\n\n"
        "| 路径 | 内容 |\n|---|---|\n"
        "| diagnostics/20261029_105246/ | 当前三组协方差对照；从 report.html、comparison.png 或 summary.csv 开始 |\n"
        "| fusion/ | 全部原始逐次结果、配置、完整 P/R/K 和迭代矩阵 |\n"
        "| RUNS.csv | 实验阶段、视差、配准版本、完成状态和结果清单 |\n"
        "| controls.json | 可跨机器定位的六条对照索引 |\n"
        "| inputs/ | 六条序列的 GT、IMU、帧索引、标定和转换元数据 |\n"
        "| disparity/ | 54 帧实际模型输出及缓存校验；可在 CPU 重跑点云和配准 |\n"
        "| recording/ | 原始标定、时间/真值检查报告和使用说明 |\n"
        "| models/ | 原始模型身份、接口和图转换验证 |\n"
        "| logs/ | 环境、构建、测试及实验执行日志，含失败尝试 |\n"
        "| provenance/ | 原工程基点、补丁、两阶段旧实现源码 |\n"
        "| SHA256SUMS | 导出文件完整性校验，包含代码与结果 |\n\n"
        "原始观测与结果按字节保存，保留历史绝对路径；请用 [复现指南](../docs/REPRODUCING.md) 中的入口。"
        "新执行写入 artifacts/，不覆盖这里。\n")
    if validation.is_file():
        readme = output / "results/README.md"
        readme.write_text(readme.read_text() +
            "\n发布前独立编译、93 项测试、跨目录回放与单组全模块验证见 "
            "[publication-validation.json](provenance/publication-validation.json)；单组验证 CSV 与配置在 "
            "`validation/cached_disparity_pipeline/`，不计入上述研究执行目录。\n")
    files = sorted(p for p in output.rglob("*") if p.is_file())
    if any(p.stat().st_size >= 50 * 1024**2 for p in files):
        raise ValueError("Unexpected large file in research export")
    (output / "results/SHA256SUMS").write_text("".join(
        f"{digest(path)}  {path.relative_to(output).as_posix()}\n" for path in files))
    print(f"Prepared standalone directory: {output}")
    print(f"Files: {len(files) + 1}; size: {sum(p.stat().st_size for p in files) / 1024**2:.1f} MiB")
    print("No Git initialization, remote changes or network operations were performed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "build/github-export/kalman-fusion-lab")
    args = parser.parse_args()
    try:
        export(args.output)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()

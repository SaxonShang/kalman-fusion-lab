#!/usr/bin/env python3
"""Freeze completed E0–E7 runs and make a review bundle; never rerun/tune the filter.

Run after the six-sequence static, calibrated and prior-model experiments. Raw run
files retain their original bytes and local paths. MANIFEST.json maps those paths
to the portable archive and records the tested commit and input/cache identities.
"""

import argparse
import csv
from datetime import datetime
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import yaml

from esikf_eval import find_runs, summarize

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ["esikf_static", "esikf_calibrated", "esikf_prior_model"]
SEQUENCES = {"20261015_112553_up", "20261015_112659_forward", "20261015_112757_right",
             "20261015_112845_yaw_left", "20261015_112940_pitch_up", "20261015_113028_roll_right"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(target):
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "build/matplotlib-cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    rows = list(csv.DictReader((target / "summary.csv").open()))
    variants = [r["variant"] for r in rows if r["config"] == "esikf_calibrated"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5), layout="constrained")
    y = np.arange(len(variants))
    for config, offset, color, label in [
        ("esikf_calibrated", -0.17, "#3264a8", "Original IMU model"),
        ("esikf_prior_model", 0.17, "#cb7628", "SDK noise + observability P")]:
        group = {r["variant"]: r for r in rows if r["config"] == config}
        for ax, key in zip(axes, ["trans_rmse_mm", "rot_rmse_deg", "trans_rmse_unflagged_mm"]):
            ax.barh(y + offset, [float(group[v][key]) for v in variants],
                    height=0.32, color=color, label=label)
        for ax, key in zip(axes[:2], ["prior_trans_rmse_mm", "prior_rot_rmse_deg"]):
            ax.axvline(float(group[variants[0]][key]), color=color, linestyle="--", linewidth=1)
    for ax, title in zip(axes, ["Translation RMSE (mm), 30 pairs", "Rotation RMSE (deg), 30 pairs",
                              "Translation RMSE (mm), prior error <= 50 mm"]):
        ax.set_title(title, fontsize=10)
        ax.set_yticks(y, variants)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.suptitle("ESIKF E0–E7: frozen pairs, conditional pose error (not trajectory ATE)\n"
                 "Dashed lines: IMU prior. R scales fitted on separate static pairs without GT.", fontsize=12)
    fig.savefig(target / "comparison.png", dpi=160)
    fig.savefig(target / "comparison.svg")
    plt.close(fig)


def archive(args):
    target = args.destination.resolve()
    if target.exists():
        raise SystemExit(f"Refusing to overwrite existing archive: {target}")
    runs = find_runs([args.runs], CONFIGS)
    if len(runs) != 18:
        raise SystemExit(f"Expected 18 finished sequence runs, got {len(runs)}")
    for config in CONFIGS:
        if {cfg["sequence"] for _, cfg, c in runs if c == config} != SEQUENCES:
            raise SystemExit(f"Missing or unexpected sequences for {config}")
    manifest = {"created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "tested_commit": args.commit, "configs": CONFIGS,
                "full_matrices": args.full_matrices,
                "local_patch": "Two GTest macro arguments parenthesized for Eigen template commas; no algorithm edits.",
                "evaluation": "Independent fixed pairs; anchor-conditioned relative pose errors, not trajectory ATE.",
                "runs": []}
    all_rows = []
    for directory, cfg, config in runs:
        if cfg["git_commit"] != args.commit or cfg["completed_stages"] != ["esikf"]:
            raise SystemExit(f"Unexpected source/stage: {directory}")
        rows = list(csv.DictReader((directory / "metrics.csv").open()))
        expected = {(p["name"], v["name"]) for p in cfg["pairs"] for v in cfg["esikf"]["variants"]}
        actual = {(r["case"], r["variant"]) for r in rows if r["stage"] == "esikf"}
        if actual != expected or len(rows) != len(expected):
            raise SystemExit(f"Missing or repeated records: {directory}")
        for relative, expected_sha in cfg["source_hashes"].items():
            if sha(ROOT / relative) != expected_sha:
                raise SystemExit(f"Sources changed during experiment: {relative}")
        cache = Path(cfg["cache_dir"])
        disparity = Path(cfg["depth"]["disparity_dir"])
        for c in (cache, disparity):
            complete = json.loads((c / "complete.json").read_text())
            for name, expected_sha in complete["files"].items():
                if sha(c / name) != expected_sha:
                    raise SystemExit(f"Cache checksum mismatch: {c / name}")
        destination = target / "runs" / directory.name
        destination.mkdir(parents=True)
        omitted = {}
        for source in directory.iterdir():
            if source.is_file():
                if not args.full_matrices and source.name in {"esikf.yml.gz", "esikf_iterations.csv"}:
                    omitted[source.name] = {"bytes": source.stat().st_size, "sha256": sha(source)}
                    continue
                shutil.copy2(source, destination / source.name)
        if not args.full_matrices:
            with (directory / "esikf_iterations.csv").open() as stream:
                reader = csv.DictReader(stream)
                fields = reader.fieldnames[:10]
                iterations = list(reader)
                write_csv(destination / "iteration_trace.csv", [{k: row[k] for k in fields} for row in iterations])
                last = {(row["case"], row["variant"]): row for row in iterations}
                write_csv(destination / "last_iteration.csv", list(last.values()))
        provenance = destination / "cache_provenance"
        provenance.mkdir()
        shutil.copy2(cache / "complete.json", provenance / "frozen_inputs.json")
        shutil.copy2(disparity / "complete.json", provenance / "model_disparity.json")
        # Small pair-conditioned matrices allow direct inspection without copying point clouds.
        if args.full_matrices:
            for source in cache.glob("conditional_*.yml.gz"):
                shutil.copy2(source, provenance / source.name)
        for row in rows:
            all_rows.append({"config": config, "sequence": cfg["sequence"],
                             "run": str(destination.relative_to(target)), **row})
        manifest["runs"].append({"config": config, "sequence": cfg["sequence"],
                                  "source": str(directory), "archive": str(destination.relative_to(target)),
                                  "records": len(rows), "valid": sum(r["valid"] == "1" for r in rows),
                                  "local_full_matrix_files": omitted,
                                  "binary_sha256": cfg["binary_sha256"],
                                  "frozen_cache": cfg["cache_key"],
                                  "disparity_cache": disparity.name})
    write_csv(target / "metrics.csv", all_rows)
    (target / "metrics.csv.gz").write_bytes(gzip.compress((target / "metrics.csv").read_bytes(), mtime=0))
    (target / "metrics.csv").unlink()
    summarize(argparse.Namespace(directories=[target / "runs"], config=CONFIGS, csv=target / "summary.csv"))
    (target / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    plot_summary(target)
    logs = target / "logs"
    logs.mkdir()
    for name in ["fusion-lab-tests.log", "b9804c7-before-test-macro-fix.log", "esikf-gpu-configure.log",
                 "esikf-gpu-export-build.log", "esikf-static.log", "esikf-calibration.log",
                 "esikf-calibrated.log", "esikf-prior-model.log", "esikf-summary.log", "esikf-environment.log",
                 "esikf-legacy-validation.json"]:
        source = ROOT / "build" / name
        if source.is_file():
            shutil.copy2(source, logs / name)
    patch = subprocess.check_output(["git", "diff", args.commit, "--", "tests/unit/test_imu_integrator.cpp"], cwd=ROOT)
    (logs / "test-build-fix.patch").write_bytes(patch)
    print(f"Archived {len(runs)} runs / {len(all_rows)} records to {target}")
    print("Add the review README and finalize SHA256SUMS after writing the narrative.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=ROOT / "artifacts/fusion")
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--full-matrices", action="store_true",
                        help="Also copy full 21-state/iteration matrix dumps (default: final P/R/K CSV + scalar traces).")
    archive(parser.parse_args())

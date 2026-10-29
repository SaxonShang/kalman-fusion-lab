#!/usr/bin/env python3
"""Verify and reproduce a published snapshot without original absolute paths."""

import argparse
import copy
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def archived(relative):
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT / "results") or not path.is_file():
        raise ValueError(f"Archived file missing or outside results/: {relative}")
    sums = ROOT / "results/SHA256SUMS"
    expected = {name: checksum for checksum, name in
                (line.split("  ", 1) for line in sums.read_text().splitlines())}
    if digest(path) != expected[path.relative_to(ROOT).as_posix()]:
        raise ValueError(f"Archived file changed: {relative}")
    return path


def verify():
    sums = ROOT / "results/SHA256SUMS"
    if not sums.is_file():
        raise ValueError("Run this command in the exported research repository")
    lines = sums.read_text().splitlines()
    for line in lines:
        expected, relative = line.split("  ", 1)
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file() or digest(path) != expected:
            raise ValueError(f"Snapshot file missing or changed: {relative}")
    print(f"Verified {len(lines)} published files; archive is intact.")


def entries(sequence):
    manifest = json.loads(archived("results/controls.json").read_text())
    if sequence == "all":
        return manifest["runs"]
    selected = [entry for entry in manifest["runs"]
                if entry["sequence"] == sequence or entry["sequence"].split("_", 2)[-1] == sequence]
    if len(selected) != 1:
        raise ValueError(f"Unknown or ambiguous sequence: {sequence}")
    return selected


def audit():
    import fusion_audit
    manifest = json.loads(archived("results/controls.json").read_text())
    for entry in manifest["runs"]:
        for treatment in ("reference", "solver_only", "repropagated", "sensitivity"):
            relative = entry[treatment]
            for name in ("metrics.csv", "eskf.yml.gz", "observations.yml.gz"):
                if name != "observations.yml.gz" or treatment == "reference":
                    archived(f"{relative}/{name}")
            entry[treatment] = str((ROOT / relative).resolve())
    directory = ROOT / "artifacts/fusion_diagnostics" / datetime.now().strftime("published_%Y%m%d_%H%M%S_%f")
    directory.mkdir(parents=True, exist_ok=False)
    fusion_audit.save(manifest, directory)
    fusion_audit.render(manifest, directory)


def replay(args):
    import yaml
    import fusion_lab as lab
    if not math.isfinite(args.prior_scale) or args.prior_scale <= 0 or any(
            not math.isfinite(value) or value <= 0 for value in args.scales):
        raise ValueError("P and R scales must be finite and positive")
    binary = args.binary.expanduser().resolve() if args.binary else lab.BINARY
    if args.binary is None:
        lab.build()
    if not binary.is_file():
        raise ValueError(f"Experiment binary missing: {binary}")
    for entry in entries(args.sequence):
        source = entry[args.source]
        config_file = archived(source + "/resolved.yaml")
        observations = archived(source + "/observations.yml.gz")
        params = archived(source + "/params.yaml")
        cfg = yaml.safe_load(config_file.read_text())
        truth = archived(f"results/inputs/{entry['sequence']}/groundtruth_tum.txt")
        if digest(observations) != cfg["observations_sha256"] or digest(truth) != cfg["groundtruth_sha256"]:
            raise ValueError("Archived observation or GT provenance mismatch")
        output = lab.allocate(cfg["sequence"], "published_replay")
        shutil.copy2(params, output / "params.yaml")
        cfg["input_source_hashes"] = copy.deepcopy(cfg["source_hashes"])
        cfg["input_completed_stages"] = cfg.get("completed_stages", [])
        cfg["published_source"] = source
        cfg["published_config_sha256"] = digest(config_file)
        cfg["params_file"] = str(output / "params.yaml")
        cfg["params_sha256"] = digest(output / "params.yaml")
        cfg["groundtruth_file"] = str(truth)
        cfg["observations"] = str(observations)
        cfg["output_dir"] = str(output)
        cfg["source_hashes"] = lab.source_hashes()
        cfg["binary_sha256"] = digest(binary)
        cfg["replay_driver_sha256"] = digest(Path(__file__))
        cfg["git_commit"] = (subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            text=True).strip() if (ROOT / ".git").exists()
                            else "published-snapshot-not-yet-committed")
        cfg["eskf"]["covariance_scales"] = args.scales
        cfg["eskf"]["prior_covariance_scale"] = args.prior_scale
        cfg["completed_stages"] = []
        # ESKF is self-contained in the observation file: no images/cloud cache are read.
        for key in ("dataset_dir", "imu_file", "cache_dir"):
            cfg.pop(key, None)
        lab.execute([binary, "eskf", lab.save_config(cfg, output), output], output / "eskf.log")
        cfg["completed_stages"] = ["eskf"]
        lab.save_config(cfg, output)
        lab.report(output)
        print(f"Replayed {source}; output: {output}", flush=True)


def run_cache(args):
    import csv
    from types import SimpleNamespace
    import yaml
    import fusion_lab as lab
    for entry in entries(args.sequence):
        cfg = yaml.safe_load(archived(entry["repropagated"] + "/resolved.yaml").read_text())
        dataset = ROOT / "data/datasets/fr5_highspeed" / entry["sequence"]
        for name in ("frames.csv", "imu.csv"):
            if not (dataset / name).is_file() or digest(dataset / name) != cfg["input_hashes"][name]:
                raise ValueError(f"Prepare the matching original recording first: {dataset / name}")
        selected = {pair[key] for pair in cfg["pairs"] for key in ("anchor", "current")}
        with (dataset / "frames.csv").open() as stream:
            frames = [row for row in csv.DictReader(stream) if int(row["source_frame_id"]) in selected]
        for frame in frames:
            for side in ("left_path", "right_path"):
                path = dataset / frame[side]
                if not path.is_file() or digest(path) != cfg["input_hashes"][frame[side]]:
                    raise ValueError(f"Original stereo image changed or missing: {path}")
        cache_name = Path(cfg["depth"]["disparity_dir"]).name
        disparity = ROOT / "results/disparity" / entry["sequence"] / cache_name
        complete = json.loads(archived(str(disparity.relative_to(ROOT) / "complete.json")).read_text())
        lab.validate_cache(disparity, complete["cache_key"])
        cfg["depth"]["disparity_dir"] = str(disparity)
        directory = ROOT / "build/published-configs"
        directory.mkdir(parents=True, exist_ok=True)
        config = directory / f"{entry['sequence']}.yaml"
        config.write_text(yaml.safe_dump(cfg, sort_keys=False))
        lab.run(SimpleNamespace(config=config, sequence=entry["sequence"], case=args.case, mode="all"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("verify", help="Check the published files using only the Python standard library")
    commands.add_parser("audit", help="Regenerate the covariance comparison without models or raw images")
    replayer = commands.add_parser("replay", help="Run ESKF using archived observations; no images/models/GPU")
    replayer.add_argument("--sequence", default="up")
    replayer.add_argument("--source", choices=("reference", "repropagated"), default="repropagated")
    replayer.add_argument("--scales", type=float, nargs="+", default=[0.1, 1, 10])
    replayer.add_argument("--prior-scale", type=float, default=1)
    replayer.add_argument("--binary", type=Path, help="Use an existing CPU experiment binary instead of building")
    cached = commands.add_parser("run-cache", help="Rerun independent GICP/ESKF/ESIKF with archived disparity")
    cached.add_argument("--sequence", default="up")
    cached.add_argument("--case", action="append")
    args = parser.parse_args()
    try:
        if args.command == "verify":
            verify()
        elif args.command == "audit":
            audit()
        elif args.command == "replay":
            replay(args)
        else:
            run_cache(args)
    except (ValueError, OSError, KeyError, RuntimeError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()

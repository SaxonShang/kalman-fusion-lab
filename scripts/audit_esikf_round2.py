#!/usr/bin/env python3
"""Audit the completed second round using recorded matrices, without rerunning the filter."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from esikf_eval import find_runs
from fusion_report import mat

CONFIGS = ["esikf_round2_static", "esikf_round2_calibrated", "esikf_round2_prior_model"]


def skew(v):
    x, y, z = v
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def states(path):
    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    records = storage.getNode("records")
    result = {}
    for i in range(records.size()):
        node = records.at(i)
        result[(node.getNode("case").string(), node.getNode("variant").string())] = {
            "anchor": mat(node.getNode("anchor"), "pose"),
            "prior": mat(node.getNode("prior"), "pose"),
            "result": mat(node.getNode("result"), "pose"),
            "P0": mat(node.getNode("prior"), "covariance"),
            "P1": mat(node.getNode("result"), "covariance"),
            "anchor_id": int(node.getNode("anchor_id").real()),
            "current_id": int(node.getNode("current_id").real())}
    storage.release()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("artifacts/fusion"))
    parser.add_argument("--round1-manifest", type=Path, default=Path("results/esikf/20261029_b9804c7/MANIFEST.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    runs = find_runs([args.runs], CONFIGS)
    assert len(runs) == 18
    old_manifest = json.loads(args.round1_manifest.read_text())
    old_runs = {(r["config"], r["sequence"]): r for r in old_manifest["runs"]}
    details, same_means, calibration_comparison = [], {}, []
    maxima = Counter()
    for directory, cfg, config in runs:
        assert cfg["git_commit"] == "1b88c05e0496f8e8790f79a26f857c7c5e54bf35"
        options = {v["name"]: v for v in cfg["esikf"]["variants"]}
        data = states(directory / "esikf.yml.gz")
        if config == CONFIGS[0]:
            old_static = Path(old_runs[("esikf_static", cfg["sequence"])]["source"])
            old_rows = {(r["case"], r["variant"]): r for r in csv.DictReader((old_static / "metrics.csv").open())}
            for r in csv.DictReader((directory / "metrics.csv").open()):
                before = old_rows[(r["case"], r["variant"])]
                calibration_comparison.append({"sequence": cfg["sequence"], "case": r["case"], "variant": r["variant"],
                    **{f"{key}_iter{cap}": record[key] for cap, record in ((8, before), (20, r))
                       for key in ("solver_status", "iterations", "zero_motion_trans_m", "zero_motion_rot_deg",
                                   "zero_motion_whitened_by_R_rot", "zero_motion_whitened_by_R_trans")}})
        expected = {(p["name"], v) for p in cfg["pairs"] for v in options}
        assert set(data) == expected
        old_data = None
        if config != CONFIGS[0]:
            old_config = "esikf_calibrated" if config == CONFIGS[1] else "esikf_prior_model"
            old_data = states(Path(old_runs[(old_config, cfg["sequence"])]["source"]) / "esikf.yml.gz")
        for (case, variant), state in data.items():
            eig = np.linalg.eigvalsh((state["P1"] + state["P1"].T) / 2)
            row = {"config": config, "sequence": cfg["sequence"], "case": case, "variant": variant,
                   "posterior_min_eigenvalue_21": float(eig.min()),
                   "posterior_asymmetry": float(np.max(np.abs(state["P1"] - state["P1"].T)))}
            opt = options[variant]
            if opt.get("update_rotation", True) is False:
                row["f3s_rotation_max_abs_change"] = float(np.max(np.abs(state["result"][:3, :3] - state["prior"][:3, :3])))
                row["f3s_rotation_cov_max_abs_change"] = float(np.max(np.abs(state["P1"][:3, :3] - state["P0"][:3, :3])))
                assert row["f3s_rotation_max_abs_change"] < 1e-12
                assert row["f3s_rotation_cov_max_abs_change"] < 1e-12
                maxima["f3s_records"] += 1
            if opt.get("prior_covariance") == "anchor_relative":
                anchor, prior = state["anchor"], state["prior"]
                J = np.zeros((6, 6))
                J[:3, :3] = prior[:3, :3].T @ anchor[:3, :3]
                J[3:, :3] = -skew(prior[:3, 3] - anchor[:3, 3]) @ anchor[:3, :3]
                J[3:, 3:] = np.eye(3)
                cache = cv2.FileStorage(str(Path(cfg["cache_dir"]) /
                    f'conditional_{state["anchor_id"]}_{state["current_id"]}.yml.gz'), cv2.FILE_STORAGE_READ)
                P, M, Q = (cache.getNode(k).mat() for k in ("anchor_covariance", "transition", "process_noise"))
                conditional = cache.getNode("covariance").mat()
                cache.release()
                M[:6, :6] -= J
                relative = M @ ((P + P.T) / 2) @ M.T + Q
                relative = (relative + relative.T) / 2
                difference = float(np.max(np.abs(state["P0"] - relative)))
                row["relative_prior_reconstruction_max_abs_difference"] = difference
                assert difference < 1e-9
                if conditional is not None:
                    row["relative_minus_conditioned_pose_min_eigenvalue"] = float(np.linalg.eigvalsh(
                        ((relative - conditional)[:6, :6] + (relative - conditional)[:6, :6].T) / 2).min())
                maxima["relative_records"] += 1
            if old_data is not None:
                key = (cfg["sequence"], case)
                if key in same_means:
                    diff = float(np.max(np.abs(state["prior"] - same_means[key])))
                    maxima["prior_mean_max_abs_difference_between_models"] = max(maxima["prior_mean_max_abs_difference_between_models"], diff)
                    assert diff < 1e-12
                same_means[key] = state["prior"]
                original = old_data[(case, "e2_anchor_prior")]
                diff = float(np.max(np.abs(state["prior"] - original["prior"])))
                maxima["prior_mean_max_abs_difference_from_round1"] = max(maxima["prior_mean_max_abs_difference_from_round1"], diff)
                assert diff < 1e-12
                if variant == "e2_anchor_prior":
                    for key in ("result", "P0", "P1"):
                        diff = float(np.max(np.abs(state[key] - original[key])))
                        maxima[f"e2_{key}_max_abs_difference_from_round1"] = max(maxima[f"e2_{key}_max_abs_difference_from_round1"], diff)
                        assert diff < 1e-9
                    maxima["round1_e2_control_records"] += 1
            details.append(row)
    with (args.output / "matrix_checks.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, list(dict.fromkeys(k for row in details for k in row)))
        writer.writeheader()
        writer.writerows(details)
    with (args.output / "calibration_comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, list(calibration_comparison[0]))
        writer.writeheader()
        writer.writerows(calibration_comparison)
    groups = {}
    for row in details:
        key = (row["config"], row["variant"])
        groups.setdefault(key, []).append(row)
    summary = {"tested_commit": "1b88c05e0496f8e8790f79a26f857c7c5e54bf35",
               "runs": len(runs), "records": len(details), "checks": dict(maxima),
               "negative_eigenvalue_definition": "strictly below zero; magnitude reported to distinguish roundoff",
               "covariance_groups": [{"config": c, "variant": v, "records": len(rs),
                   "negative_full_posterior_eigenvalues": sum(r["posterior_min_eigenvalue_21"] < 0 for r in rs),
                   "min_full_posterior_eigenvalue": min(r["posterior_min_eigenvalue_21"] for r in rs)}
                   for (c, v), rs in sorted(groups.items())]}
    (args.output / "validation.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "covariance_groups"}, indent=2))


if __name__ == "__main__":
    main()

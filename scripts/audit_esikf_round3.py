#!/usr/bin/env python3
"""Audit round 3 against frozen round 2 inputs and independently recompute both NIS values."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from audit_esikf_round2 import skew, states
from esikf_eval import find_runs
from fusion_report import mat, write_csv

COMMIT = "8f2355077f1d583b4c2699a62d8dabd017e57899"


def cholesky_nis(residual, prior, noise):
    """Independent Cholesky whitening; production report uses symmetric eigen whitening."""
    lower = np.linalg.cholesky((prior + prior.T) / 2)
    left = np.linalg.solve(lower, (noise + noise.T) / 2)
    whitened = np.linalg.solve(lower, left.T).T
    ratio, basis = np.linalg.eigh((whitened + whitened.T) / 2)
    observed = ratio < 1e4
    z = basis.T @ np.linalg.solve(lower, residual)
    return float(np.sum(z[observed] ** 2 / (1 + ratio[observed]))), int(observed.sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("artifacts/fusion"))
    parser.add_argument("--derived", type=Path, default=Path("artifacts/derived_round2_final_nis_8f23550"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    old_manifest = json.loads(Path("results/esikf/20261030_1b88c05/MANIFEST.json").read_text())
    old_runs = {r["sequence"]: Path(r["source"]) for r in old_manifest["runs"]
                if r["config"] == "esikf_round2_prior_model"}
    runs = find_runs([args.runs], ["esikf_round3_gyro"])
    assert len(runs) == 6 and {cfg["sequence"] for _, cfg, _ in runs} == set(old_runs)
    details, comparisons, visual = [], [], []
    counts = Counter()
    max_nis_error = 0.
    for directory, cfg, _ in runs:
        assert cfg["git_commit"] == COMMIT and cfg["completed_stages"] == ["esikf"]
        old = old_runs[cfg["sequence"]]
        previous_cfg = yaml.safe_load((old / "resolved.yaml").read_text())
        assert cfg["pairs"] == previous_cfg["pairs"] and cfg["esikf"] == previous_cfg["esikf"]
        old_prior = dict(previous_cfg["prior"])
        new_prior = dict(cfg["prior"])
        assert new_prior.pop("gyro_correction") == yaml.safe_load(
            Path("experiments/fusion/configs/esikf_round3_gyro.yaml").read_text())["prior"]["gyro_correction"]
        assert old_prior == new_prior
        cache = Path(cfg["cache_dir"])
        assert yaml.safe_load((cache / "inputs.yaml").read_text())["gyro_correction"] == "applied"
        frame_ids = sorted({int(p[k]) for p in cfg["pairs"] for k in ("anchor", "current")})
        for frame in frame_ids:
            row = {"sequence": cfg["sequence"], "frame": frame}
            a = cv2.FileStorage(str(Path(previous_cfg["cache_dir"]) / f"frame_{frame}.yml.gz"), cv2.FILE_STORAGE_READ)
            b = cv2.FileStorage(str(cache / f"frame_{frame}.yml.gz"), cv2.FILE_STORAGE_READ)
            for key in ("points_xyzrgb", "depth_m", "disparity_pixels"):
                before, after = a.getNode(key).mat(), b.getNode(key).mat()
                assert before.shape == after.shape and np.array_equal(before, after, equal_nan=True), (frame, key)
                row[key + "_identical"] = True
            a.release()
            b.release()
            visual.append(row)
        variants = {v["name"]: v for v in cfg["esikf"]["variants"]}
        old_rows = {(r["case"], r["variant"]): r for r in csv.DictReader((old / "metrics.csv").open())}
        metrics = {(r["case"], r["variant"]): r for r in csv.DictReader((directory / "metrics.csv").open())}
        data = states(directory / "esikf.yml.gz")
        expected = {(p["name"], v) for p in cfg["pairs"] for v in variants}
        assert set(data) == set(metrics) == set(old_rows) == expected
        for key, state in data.items():
            case, variant = key
            r, before = metrics[key], old_rows[key]
            comparisons.append({"sequence": cfg["sequence"], "case": case, "variant": variant,
                **{f"{field}_{arm}": record[field] for arm, record in (("round2", before), ("round3", r))
                   for field in ("prior_trans_error_m", "prior_rot_error_deg", "result_trans_error_m",
                                 "result_rot_error_deg", "prior_flagged", "solver_status", "valid")}})
            eig = np.linalg.eigvalsh((state["P1"] + state["P1"].T) / 2)
            row = {"sequence": cfg["sequence"], "case": case, "variant": variant,
                   "posterior_min_eigenvalue_21": float(eig.min()),
                   "posterior_asymmetry": float(np.max(np.abs(state["P1"] - state["P1"].T)))}
            opt = variants[variant]
            if opt.get("update_rotation", True) is False:
                row["f3s_rotation_max_abs_change"] = float(np.max(np.abs(state["result"][:3, :3] - state["prior"][:3, :3])))
                row["f3s_rotation_cov_max_abs_change"] = float(np.max(np.abs(state["P1"][:3, :3] - state["P0"][:3, :3])))
                assert row["f3s_rotation_max_abs_change"] < 1e-12 and row["f3s_rotation_cov_max_abs_change"] < 1e-12
                counts["f3s_records"] += 1
            if opt.get("prior_covariance") == "anchor_relative":
                a, b = state["anchor"], state["prior"]
                J = np.zeros((6, 6))
                J[:3, :3] = b[:3, :3].T @ a[:3, :3]
                J[3:, :3] = -skew(b[:3, 3] - a[:3, 3]) @ a[:3, :3]
                J[3:, 3:] = np.eye(3)
                storage = cv2.FileStorage(str(cache / f'conditional_{state["anchor_id"]}_{state["current_id"]}.yml.gz'), cv2.FILE_STORAGE_READ)
                P, M, Q = (storage.getNode(k).mat() for k in ("anchor_covariance", "transition", "process_noise"))
                storage.release()
                M[:6, :6] -= J
                estimate = M @ ((P + P.T) / 2) @ M.T + Q
                estimate = (estimate + estimate.T) / 2
                row["relative_prior_reconstruction_max_abs_difference"] = float(np.max(np.abs(state["P0"] - estimate)))
                assert row["relative_prior_reconstruction_max_abs_difference"] < 1e-9
                counts["relative_records"] += 1
            details.append(row)
        storage = cv2.FileStorage(str(directory / "esikf.yml.gz"), cv2.FILE_STORAGE_READ)
        records = storage.getNode("records")
        for i in range(records.size()):
            node = records.at(i)
            key = (node.getNode("case").string(), node.getNode("variant").string())
            iterations = node.getNode("esikf_iterations")
            assert iterations.size() > 0
            P = mat(node.getNode("prior"), "covariance")[:6, :6]
            for prefix, it in (("nis", iterations.at(0)), ("nis_final", iterations.at(iterations.size() - 1))):
                residual, noise = mat(it, "virtual_residual").reshape(-1), mat(it, "R")
                if prefix == "nis_final":
                    residual = residual + mat(it, "prior_offset").reshape(-1)
                for part, block in (("", slice(None)), ("_rot", slice(0, 3)), ("_trans", slice(3, 6))):
                    value, dof = cholesky_nis(residual[block], P[block, block], noise[block, block])
                    label = prefix + part
                    reported = float(metrics[key][label])
                    assert int(metrics[key][label + "_dof"]) == dof
                    if dof:
                        assert np.isclose(value, reported, rtol=1e-6, atol=1e-8), (key, label, value, reported)
                        max_nis_error = max(max_nis_error, abs(value - reported) / max(1., abs(reported)))
                        counts["nis_cholesky_checks"] += 1
        storage.release()
    # Existing first-iteration diagnostics must not change when deriving the new last-iteration values.
    existing_max_change = 0.
    for run in old_manifest["runs"]:
        old = Path(run["source"])
        before = {(r["case"], r["variant"]): r for r in csv.DictReader((old / "metrics.csv").open())}
        after = list(csv.DictReader((args.derived / "runs" / old.name / "metrics.csv").open()))
        for r in after:
            for key in (k for k in before[(r["case"], r["variant"])] if k.startswith("nis")):
                a, b = float(before[(r["case"], r["variant"])][key]), float(r[key])
                assert (np.isnan(a) and np.isnan(b)) or np.isclose(a, b, rtol=1e-12, atol=1e-12)
                if np.isfinite(a):
                    existing_max_change = max(existing_max_change, abs(a - b))
    write_csv(args.output / "matrix_checks.csv", details)
    write_csv(args.output / "paired_comparison.csv", comparisons)
    write_csv(args.output / "visual_input_checks.csv", visual)
    result = {"tested_commit": COMMIT, "runs": len(runs), "records": len(details),
              "visual_frames_identical_to_round2": len(visual), "checks": dict(counts),
              "max_nis_scaled_absolute_error": max_nis_error,
              "round2_existing_first_nis_max_absolute_change": existing_max_change,
              "negative_full_posterior_eigenvalues": sum(r["posterior_min_eigenvalue_21"] < 0 for r in details),
              "min_full_posterior_eigenvalue": min(r["posterior_min_eigenvalue_21"] for r in details)}
    (args.output / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Aggregate ESIKF results across sequences and calibrate the virtual observation covariance.

summarize: read metrics.csv of finished runs, group by (config, variant) over all sequences.
calibrate: from static-pair runs (true relative motion is zero, so no groundtruth is used),
           estimate per-model scales s = mean(e^T R^-1 e / 3) for the rotation and translation
           blocks of the ESIKF virtual observation, then write esikf_calibrated.yaml.

Arguments may be run directories or a parent such as artifacts/fusion; with --config only runs
from those configuration files are used, and only the latest run per (config, sequence) is kept.

    python3 scripts/esikf_eval.py calibrate artifacts/fusion --config esikf_static
    python3 scripts/esikf_eval.py summarize artifacts/fusion --config esikf_calibrated esikf_prior_model

Round 2 calibrates from static runs with a higher iteration cap, without failed records, and
extends its own base:

    python3 scripts/esikf_eval.py calibrate artifacts/fusion --config esikf_round2_static --samples usable \
        --base experiments/fusion/configs/esikf_round2.yaml \
        --write experiments/fusion/configs/esikf_round2_calibrated.yaml
"""

import argparse
import csv
from datetime import datetime
import math
import os
from pathlib import Path
import statistics
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "experiments/fusion/configs"
CHI2_3_MEDIAN_PER_DOF = 2.3660 / 3
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fusion_lab import load_config  # noqa: E402


def find_runs(paths, configs):
    """Run directories under the given paths, latest per (config, sequence)."""
    candidates = []
    for path in paths:
        path = Path(path).resolve()
        if (path / "metrics.csv").is_file():
            candidates.append(path)
        elif path.is_dir():
            candidates += [d for d in path.iterdir()
                           if (d / "metrics.csv").is_file() and (d / "resolved.yaml").is_file()]
        else:
            raise SystemExit(f"Not a run directory: {path}")
    chosen = {}
    # Run directory names start with a timestamp, so later runs replace earlier ones.
    for run in sorted(candidates, key=lambda d: d.name):
        cfg = yaml.safe_load((run / "resolved.yaml").read_text(encoding="utf-8"))
        config = Path(cfg.get("config_file", "unknown")).stem
        if configs and config not in configs:
            continue
        chosen[(config, cfg["sequence"])] = (run, cfg, config)
    for (config, sequence), (run, _, _) in sorted(chosen.items()):
        print(f"using {config} / {sequence}: {run}")
    return list(chosen.values())


def read_runs(paths, configs=None):
    rows = []
    for directory, cfg, config in find_runs(paths, configs):
        metrics = directory / "metrics.csv"
        with metrics.open(newline="") as stream:
            for row in csv.DictReader(stream):
                if row["stage"] != "esikf":
                    continue
                row["config"] = config
                row["sequence"] = cfg["sequence"]
                row["directory"] = str(directory)
                rows.append(row)
    if not rows:
        raise SystemExit("No ESIKF records found in the given directories")
    return rows


def number(row, key):
    try:
        value = float(row.get(key, "nan"))
    except ValueError:
        return math.nan
    return value


def finite(values):
    return [v for v in values if math.isfinite(v)]


def rmse(values):
    values = finite(values)
    return math.sqrt(sum(v * v for v in values) / len(values)) if values else math.nan


def median(values):
    values = finite(values)
    return statistics.median(values) if values else math.nan


def mean(values):
    values = finite(values)
    return sum(values) / len(values) if values else math.nan


def summarize(args):
    rows = read_runs(args.directories, args.config)
    groups = {}
    for row in rows:
        groups.setdefault((row["config"], row["variant"]), []).append(row)
    summary = []
    for (config, variant), selected in sorted(groups.items()):
        valid = [r for r in selected if r["valid"] == "1"]
        unflagged = [r for r in valid if r.get("prior_flagged", "0") == "0"]
        statuses = {}
        for r in selected:
            statuses[r["solver_status"] or "-"] = statuses.get(r["solver_status"] or "-", 0) + 1
        get = lambda rs, key: [number(r, key) for r in rs]
        summary.append({
            "config": config, "variant": variant,
            "sequences": len({r["sequence"] for r in selected}),
            "valid": len(valid), "total": len(selected),
            "prior_trans_rmse_mm": 1000 * rmse(get(valid, "prior_trans_error_m")),
            "trans_rmse_mm": 1000 * rmse(get(valid, "result_trans_error_m")),
            "trans_median_mm": 1000 * median(get(valid, "result_trans_error_m")),
            "trans_rmse_unflagged_mm": 1000 * rmse(get(unflagged, "result_trans_error_m")),
            "prior_rot_rmse_deg": rmse(get(valid, "prior_rot_error_deg")),
            "rot_rmse_deg": rmse(get(valid, "result_rot_error_deg")),
            "rot_median_deg": median(get(valid, "result_rot_error_deg")),
            "rot_rmse_unflagged_deg": rmse(get(unflagged, "result_rot_error_deg")),
            "improved_trans": sum(number(r, "result_trans_error_m") < number(r, "prior_trans_error_m")
                                  for r in valid),
            "improved_rot": sum(number(r, "result_rot_error_deg") < number(r, "prior_rot_error_deg")
                                for r in valid),
            "posterior_whitened_rot": mean(get(valid, "posterior_whitened_by_P_rot")),
            "posterior_whitened_trans": mean(get(valid, "posterior_whitened_by_P_trans")),
            # Prior and first innovation do not depend on whether the update succeeded.
            "prior_whitened_rot": mean(get(selected, "prior_whitened_by_P_rot")),
            "prior_whitened_trans": mean(get(selected, "prior_whitened_by_P_trans")),
            "prior_whitened_rot_median": median(get(selected, "prior_whitened_by_P_rot")),
            "prior_whitened_trans_median": median(get(selected, "prior_whitened_by_P_trans")),
            "nis_per_dof": mean(get(selected, "nis_per_dof")),
            "nis_over_chi2_median": median(get(selected, "nis_over_chi2_median")),
            "nis_exceeds_95": mean(get(selected, "nis_exceeds_chi2_95")),
            "nis_rot_per_dof": mean(get(selected, "nis_rot_per_dof")),
            "nis_trans_per_dof": mean(get(selected, "nis_trans_per_dof")),
            "nis_dof_min": min(finite(get(selected, "nis_dof")), default=math.nan),
            # Innovation at the last linearization point; the first-iteration value above is biased low
            # when correspondences are re-associated every iteration.
            "nis_final_per_dof": mean(get(selected, "nis_final_per_dof")),
            "nis_final_over_chi2_median": median(get(selected, "nis_final_over_chi2_median")),
            "nis_final_exceeds_95": mean(get(selected, "nis_final_exceeds_chi2_95")),
            "nis_final_rot_per_dof": mean(get(selected, "nis_final_rot_per_dof")),
            "nis_final_trans_per_dof": mean(get(selected, "nis_final_trans_per_dof")),
            "max_prior_offset_rot_deg": max(finite(get(selected, "max_prior_offset_rot_deg")), default=math.nan),
            "iterations_mean": mean(get(selected, "iterations")),
            "status": " ".join(f"{k}:{v}" for k, v in sorted(statuses.items())),
        })
    fields = list(summary[0])
    if args.csv:
        with Path(args.csv).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fields)
            writer.writeheader()
            writer.writerows(summary)
        print(f"Wrote {args.csv}")
    shown = ["config", "variant", "valid", "total", "trans_rmse_mm", "trans_median_mm",
             "trans_rmse_unflagged_mm", "rot_rmse_deg", "rot_median_deg", "rot_rmse_unflagged_deg",
             "improved_trans", "improved_rot", "posterior_whitened_rot", "posterior_whitened_trans",
             "prior_whitened_rot", "prior_whitened_trans", "nis_per_dof", "nis_final_per_dof",
             "nis_final_over_chi2_median", "nis_final_exceeds_95", "max_prior_offset_rot_deg",
             "iterations_mean", "status"]
    def cell(value):
        return f"{value:.4g}" if isinstance(value, float) else str(value)
    table = [[cell(row[k]) for k in shown] for row in summary]
    widths = [max(len(k), *(len(t[i]) for t in table)) for i, k in enumerate(shown)]
    print("  ".join(k.ljust(w) for k, w in zip(shown, widths)))
    for t in table:
        print("  ".join(v.ljust(w) for v, w in zip(t, widths)))
    print("\nwhitened: mean e^T C^-1 e per degree of freedom; about 1 when the covariance matches the "
          "actual error. unflagged excludes pairs whose pure-IMU prior is off by more than 50 mm.\n"
          "prior_whitened checks the prior P alone against groundtruth. nis is the first-iteration "
          "r^T (P + R)^-1 r without groundtruth; it tests P + R together (mean per dof and median over the "
          "chi-square median both about 1 when consistent) and is also raised by bias, wrong correspondences "
          "and linearization. Unobserved directions are dropped from nis and its degrees of freedom.")


# Records that stopped this way are not an observation of the static pose (e.g. ray residuals
# lose their correspondences after a partial update) and must not enter the calibration.
FAILED_STATUSES = {"correspondences_lost", "numerical_failure", "too_few_correspondences", "skipped"}
SAMPLE_SETS = {
    "all": lambda r: r["valid"] == "1",
    "usable": lambda r: r["valid"] == "1" and r["solver_status"] not in FAILED_STATUSES,
    "converged": lambda r: r["valid"] == "1" and r["solver_status"] == "converged",
}


def scale_estimates(records):
    rot = finite(number(r, "zero_motion_whitened_by_R_rot") for r in records)
    trans = finite(number(r, "zero_motion_whitened_by_R_trans") for r in records)
    if not rot or not trans:
        return None
    # Unbiased: E[e^T R^-1 e / 3] = s. Robust: median of chi2(3)/3 is 0.7887, so s = median / 0.7887.
    return {"mean": (mean(rot), mean(trans)),
            "median": (median(rot) / CHI2_3_MEDIAN_PER_DOF, median(trans) / CHI2_3_MEDIAN_PER_DOF)}


def calibrate(args):
    rows = read_runs(args.directories, args.config)
    base_path = Path(args.base).resolve()
    base = load_config(base_path)
    variants = base["esikf"]["variants"]
    templates = base["esikf"]["calibrated_variants"]
    calibration = {}
    for source in sorted({t["from"] for t in templates}):
        records = [r for r in rows if r["variant"] == source]
        if not records:
            raise SystemExit(f"No static-pair records for {source}; run esikf_static.yaml first")
        by_set = {}
        for name, keep in SAMPLE_SETS.items():
            chosen = [r for r in records if keep(r)]
            by_set[name] = (len(chosen), scale_estimates(chosen))
            estimate = by_set[name][1]
            shown = (f"mean {[round(v, 2) for v in estimate['mean']]}, median {[round(v, 2) for v in estimate['median']]}"
                     if estimate else "no samples")
            print(f"{source} [{name}]: {len(chosen)}/{len(records)} pairs; rotation/translation scale {shown}")
        count, estimate = by_set[args.samples]
        statuses = {}
        for r in records:
            statuses[r["solver_status"]] = statuses.get(r["solver_status"], 0) + 1
        entry = {"samples": args.samples, "statistic": args.statistic, "pairs": count,
                 "records": len(records), "solver_status": statuses,
                 "sequences": sorted({r["sequence"] for r in records if SAMPLE_SETS[args.samples](r)})}
        for name, (n, e) in by_set.items():
            if e:
                entry[f"{name}_mean_estimate"] = [float(f"{v:.4g}") for v in e["mean"]]
                entry[f"{name}_median_estimate"] = [float(f"{v:.4g}") for v in e["median"]]
        if estimate is None:
            entry["excluded"] = f"no {args.samples} static-pair records; calibrated variants from it are not generated"
            print(f"{source}: excluded ({entry['excluded']})")
        else:
            entry["rotation"] = float(f"{estimate[args.statistic][0]:.4g}")
            entry["translation"] = float(f"{estimate[args.statistic][1]:.4g}")
        calibration[source] = entry
    by_name = {v["name"]: v for v in variants}
    calibrated = []
    for template in templates:
        scales = calibration[template["from"]]
        if "excluded" in scales:
            print(f"skipped {template['name']}: {template['from']} has no usable calibration")
            continue
        variant = dict(by_name[template["base"]])
        variant["name"] = template["name"]
        variant["observation_scale_rotation"] = scales["rotation"]
        variant["observation_scale_translation"] = scales["translation"]
        # Any other template key overrides the calibrated variant (e.g. a fixed rotation scale).
        variant.update({k: v for k, v in template.items() if k not in ("name", "base", "from")})
        calibrated.append(variant)
    target = Path(args.write).resolve()
    try:
        extends = os.path.relpath(base_path, target.parent).replace(os.sep, "/")
    except ValueError:  # different drives on Windows
        extends = str(base_path)
    output = {
        "extends": extends,
        "esikf": {
            "observation_calibration": {
                "generated_by": "scripts/esikf_eval.py calibrate",
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "statistic": (f"{args.statistic} of zero-motion e^T R^-1 e per degree of freedom on "
                              f"{args.samples} static pairs (failed statuses: {sorted(FAILED_STATUSES)})"),
                "runs": sorted({r["directory"] for r in rows}),
                "models": calibration,
            },
            "variants": variants + calibrated,
        },
    }
    header = ("# 由 scripts/esikf_eval.py calibrate 生成：静止帧对（真实运动为零，不用真值）上\n"
              f"# 每种观测模型的 {args.statistic}(eᵀR⁻¹e/3) 作为旋转 / 平移观测协方差的放大倍数，"
              f"样本为 {args.samples}，追加 calibrated_variants。\n")
    target.write_text(header + yaml.safe_dump(output, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"Wrote {target}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    s = commands.add_parser("summarize", help="Aggregate ESIKF metrics across runs")
    s.add_argument("directories", nargs="+", type=Path)
    s.add_argument("--config", nargs="+", help="Configuration file stems to include")
    s.add_argument("--csv", type=Path)
    c = commands.add_parser("calibrate", help="Calibrate observation scales from static-pair runs")
    c.add_argument("directories", nargs="+", type=Path)
    c.add_argument("--config", nargs="+", default=["esikf_static"],
                   help="Configuration file stems of the static-pair runs")
    c.add_argument("--base", type=Path, default=CONFIGS / "esikf_improvements.yaml",
                   help="Configuration whose variants and calibrated_variants the output extends")
    c.add_argument("--write", type=Path, default=CONFIGS / "esikf_calibrated.yaml")
    c.add_argument("--statistic", choices=("mean", "median"), default="mean",
                   help="mean is unbiased; median (scaled by the chi-square median) resists outliers")
    c.add_argument("--samples", choices=tuple(SAMPLE_SETS), default="all",
                   help="all: every valid record (default, reproduces round 1); usable: drop failed solver "
                        "statuses; converged: converged records only. All three estimates are printed and stored")
    args = parser.parse_args()
    summarize(args) if args.command == "summarize" else calibrate(args)


if __name__ == "__main__":
    main()

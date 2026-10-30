#!/usr/bin/env python3
"""F5: where the IMU relative-rotation error comes from, using only the published inputs.

For windows of groundtruth frames, compare the gyro-integrated body rotation with the groundtruth
body rotation (left-camera poses mapped through T_imu_cam, exactly as fusion_report.py builds the
reference) and fit, pooled over all sequences,

    θ_gt ≈ A θ_imu + c_seq · L,

where θ are rotation vectors over a window of length L and c_seq absorbs each recording's residual
gyro bias. The polar decomposition A = X·S separates a constant frame rotation X (camera–IMU
extrinsic or groundtruth hand-eye error; this data cannot tell the two apart) from a symmetric S
(gyro scale and axis non-orthogonality). Each window's raw error log(R_imuᵀ R_gt) is also split into
the part along the rotation axis (what a scale error produces) and the part perpendicular to it
(what a frame rotation or misalignment produces), and a residual IMU time offset is scanned.

    python3 scripts/imu_rotation_check.py --inputs results/inputs --output artifacts/imu_rotation_check

G1 (gyro scale correction): with --write-config, A is refitted only on windows at least
--margin-s away from every evaluation pair of --evaluation-config, and its symmetric part S is
written as prior.gyro_correction (ω' = S ω) of a new configuration. The frame rotation X is not
applied, because it may belong to the groundtruth hand-eye calibration. Windows near the evaluation
pairs are held out and reported before and after the correction.

    python3 scripts/imu_rotation_check.py --write-config experiments/fusion/configs/esikf_round3_gyro.yaml
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fusion_lab import load_config  # noqa: E402
# One frame (sync/main_motion pairs) and six frames (main_motion_long pairs).
WINDOW_FRAMES = (1, 6)
FIT_WINDOW_FRAMES = 6  # longer windows rotate more and condition A better
OFFSETS_MS = np.arange(-20.0, 20.01, 0.5)
MIN_AXIS_ANGLE_DEG = 1.0  # windows rotating less have no well-defined axis
INIT_SAMPLES = 200  # the integrator initializes on the first 200 IMU samples


def skew(v):
    z = np.zeros(len(v))
    return np.stack([np.stack([z, -v[:, 2], v[:, 1]], 1),
                     np.stack([v[:, 2], z, -v[:, 0]], 1),
                     np.stack([-v[:, 1], v[:, 0], z], 1)], 1)


def exp_so3(v):
    theta = np.linalg.norm(v, axis=1)
    safe = np.where(theta < 1e-12, 1.0, theta)
    a = np.where(theta < 1e-12, 1.0, np.sin(theta) / safe)
    b = np.where(theta < 1e-12, 0.5, (1 - np.cos(theta)) / safe ** 2)
    k = skew(v)
    return np.eye(3) + a[:, None, None] * k + b[:, None, None] * (k @ k)


def log_so3(r):
    theta = np.arccos(np.clip((np.trace(r, axis1=1, axis2=2) - 1) / 2, -1, 1))
    w = 0.5 * np.stack([r[:, 2, 1] - r[:, 1, 2], r[:, 0, 2] - r[:, 2, 0], r[:, 1, 0] - r[:, 0, 1]], 1)
    s = np.sin(theta)
    return w * np.where(theta < 1e-8, 1.0, theta / np.where(s == 0, 1.0, s))[:, None]


def quaternion_rotation(q):
    x, y, z, w = (q / np.linalg.norm(q, axis=1)[:, None]).T
    return np.stack([np.stack([1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], 1),
                     np.stack([2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)], 1),
                     np.stack([2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)], 1)], 1)


def extrinsic_rotation(params):
    """R of T_imu_cam, same convention as fusion_report.camera_extrinsic."""
    r, p, y = params["T_imu_cam"]["rotation"]
    rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
    ry = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]])
    rx = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]])
    return rz @ ry @ rx


class Sequence:
    def __init__(self, directory):
        self.name = directory.name
        imu = np.loadtxt(directory / "imu.csv", delimiter=",", skiprows=1, usecols=(1, 6, 7, 8))
        self.origin = imu[0, 0]
        self.tau = imu[:, 0] - self.origin
        gyro = np.radians(imu[:, 1:])  # CSV is deg/s in the SDK IMU image axes
        self.bias = gyro[:INIT_SAMPLES].mean(0)
        # Midpoint rate over each sample interval, as in the integrator; cumulative body rotation.
        self.rate = 0.5 * (gyro[:-1] + gyro[1:]) - self.bias
        steps = exp_so3(self.rate * np.diff(self.tau)[:, None])
        self.cumulative = np.empty((len(self.tau), 3, 3))
        self.cumulative[0] = np.eye(3)
        for k, step in enumerate(steps):
            self.cumulative[k + 1] = self.cumulative[k] @ step
        gt = np.array([[float(v) for v in line.split()] for line in (directory / "groundtruth_tum.txt").read_text().splitlines()
                       if line.strip() and not line.startswith("#")])
        self.gt_time = gt[:, 0] - self.origin
        r_bc = extrinsic_rotation(yaml.safe_load((directory / "params.yaml").read_text(encoding="utf-8")))
        self.gt_body = quaternion_rotation(gt[:, 4:8]) @ r_bc.T  # R_wb = R_wc R_bcᵀ
        self.frame_dt = float(np.median(np.diff(self.gt_time)))
        # Groundtruth rows follow frames.csv; pairs in the configurations use source frame ids.
        with (directory / "frames.csv").open(newline="") as stream:
            frames = list(csv.DictReader(stream))
        if len(frames) != len(gt) or max(abs(float(f["end_time"]) - self.origin - t)
                                         for f, t in zip(frames, self.gt_time)) > 1e-3:
            raise SystemExit(f"{self.name}: groundtruth rows do not match frames.csv")
        self.time_of_source = {int(f["source_frame_id"]): t for f, t in zip(frames, self.gt_time)}

    def rotation_at(self, t):
        k = np.clip(np.searchsorted(self.tau, t, side="right") - 1, 0, len(self.rate) - 1)
        return self.cumulative[k] @ exp_so3(self.rate[k] * (t - self.tau[k])[:, None])

    def windows(self, frames):
        """Index pairs (i, i+frames) without groundtruth gaps and inside IMU coverage."""
        i = np.arange(len(self.gt_time) - frames)
        j = i + frames
        keep = ((self.gt_time[j] - self.gt_time[i] < 1.5 * frames * self.frame_dt) &
                (self.gt_time[i] - 0.025 > self.tau[0]) & (self.gt_time[j] + 0.025 < self.tau[-1]))
        return i[keep], j[keep]

    def relative(self, i, j, offset_s=0.0):
        """Groundtruth and IMU body rotations over the windows; IMU clock shifted by offset_s."""
        gt = np.transpose(self.gt_body[i], (0, 2, 1)) @ self.gt_body[j]
        a, b = self.rotation_at(self.gt_time[i] - offset_s), self.rotation_at(self.gt_time[j] - offset_s)
        return gt, np.transpose(a, (0, 2, 1)) @ b, self.gt_time[j] - self.gt_time[i]


def collect(sequences, frames, offset_s=0.0, excluded=None):
    """Window data of all sequences; near[k] marks windows overlapping an excluded interval."""
    theta_gt, theta_imu, error, length, index, half, near = [], [], [], [], [], [], []
    for s, sequence in enumerate(sequences):
        i, j = sequence.windows(frames)
        gt, imu, L = sequence.relative(i, j, offset_s)
        theta_gt.append(log_so3(gt))
        theta_imu.append(log_so3(imu))
        error.append(log_so3(np.transpose(imu, (0, 2, 1)) @ gt))  # same definition as the prior error
        length.append(L)
        index.append(np.full(len(L), s))
        start, end = sequence.gt_time[i], sequence.gt_time[j]
        half.append((start >= np.median(start)).astype(int))
        overlap = np.zeros(len(L), bool)
        for low, high in (excluded or {}).get(sequence.name, []):
            overlap |= (start < high) & (end > low)
        near.append(overlap)
    return {k: np.concatenate(v) for k, v in
            (("gt", theta_gt), ("imu", theta_imu), ("error", error), ("length", length), ("seq", index),
             ("half", half), ("near", near))}


def subset(data, mask):
    return {key: value[mask] for key, value in data.items()}


def bias_rates(data, count, matrix):
    """Per-sequence bias rate c minimising |θ_gt − M θ_imu − c L| with M fixed (rows of design())."""
    target = data["gt"] - data["imu"] @ matrix.T
    rates = np.zeros((count, 3))
    for k in range(count):
        mask = data["seq"] == k
        length = data["length"][mask]
        if length.size:
            rates[k] = (length @ target[mask]) / (length @ length)
    return np.vstack([matrix.T, rates])


def design(data, count):
    onehot = np.zeros((len(data["length"]), count))
    onehot[np.arange(len(data["length"])), data["seq"]] = 1
    return np.hstack([data["imu"], data["length"][:, None] * onehot])


def fit(data, count):
    coefficients, *_ = np.linalg.lstsq(design(data, count), data["gt"], rcond=None)
    return coefficients


def residual_rms_deg(data, count, coefficients):
    residual = data["gt"] - design(data, count) @ coefficients
    return math.degrees(math.sqrt(np.mean(np.sum(residual ** 2, 1))))


def rms_deg(vectors):
    return math.degrees(math.sqrt(np.mean(np.sum(vectors ** 2, 1)))) if len(vectors) else math.nan


def axis_split(data):
    """Raw error along and perpendicular to the true rotation axis, for windows rotating enough."""
    angle = np.linalg.norm(data["gt"], axis=1)
    keep = angle > math.radians(MIN_AXIS_ANGLE_DEG)
    axis = data["gt"][keep] / angle[keep, None]
    error = data["error"][keep]
    along = np.sum(error * axis, 1)
    across = np.linalg.norm(error - along[:, None] * axis, axis=1)
    return {"windows": int(keep.sum()),
            "along_axis_rms_deg": math.degrees(math.sqrt(np.mean(along ** 2))) if keep.any() else math.nan,
            "across_axis_rms_deg": math.degrees(math.sqrt(np.mean(across ** 2))) if keep.any() else math.nan,
            # Least-squares slope of the error against the rotation angle, in percent of the rotation.
            "along_axis_percent_of_rotation": 100 * float(along @ angle[keep] / (angle[keep] @ angle[keep])) if keep.any() else math.nan,
            "across_axis_percent_of_rotation": 100 * float(across @ angle[keep] / (angle[keep] @ angle[keep])) if keep.any() else math.nan,
            "median_rotation_deg": math.degrees(float(np.median(angle[keep]))) if keep.any() else math.nan}


def polar(a):
    u, sigma, vt = np.linalg.svd(a)
    d = np.diag([1, 1, np.sign(np.linalg.det(u @ vt))])
    x = u @ d @ vt
    s = vt.T @ d @ np.diag(sigma) @ vt
    return x, s


def gyro_calibration(sequences, count, args):
    """G1: fit A away from the evaluation pairs; only its symmetric part S becomes the correction."""
    cfg = load_config(args.evaluation_config.resolve())
    excluded = {}
    for sequence in sequences:
        excluded[sequence.name] = [
            (sequence.time_of_source[int(pair["anchor"])] - args.margin_s,
             sequence.time_of_source[int(pair["current"])] + args.margin_s) for pair in cfg["pairs"]]
    fit_data = collect(sequences, FIT_WINDOW_FRAMES, excluded=excluded)
    calibration = subset(fit_data, ~fit_data["near"])
    coefficients = fit(calibration, count)
    x, s = polar(coefficients[:3].T)
    x_vector = log_so3(x[None])[0]
    evaluation_config = os.path.relpath(args.evaluation_config.resolve(), ROOT).replace(os.sep, "/")
    result = {
        "evaluation_config": evaluation_config,
        "margin_s": args.margin_s,
        "fit_window_frames": FIT_WINDOW_FRAMES,
        "fit_windows": int(len(calibration["length"])),
        "scale_matrix": s.tolist(),
        "scale_eigenvalues_percent": [float(100 * (v - 1)) for v in np.linalg.eigvalsh((s + s.T) / 2)],
        "frame_rotation_deg_not_applied": float(math.degrees(np.linalg.norm(x_vector))),
        "held_out_near_evaluation": {},
    }
    for frames in WINDOW_FRAMES:
        data = collect(sequences, frames, excluded=excluded)
        held = subset(data, data["near"])
        result["held_out_near_evaluation"][f"{frames}_frames"] = {
            "windows": int(len(held["length"])),
            # Bias rates refitted on the calibration windows with the matrix fixed.
            "bias_only_rms_deg": residual_rms_deg(held, count, bias_rates(calibration, count, np.eye(3))),
            "scale_corrected_rms_deg": residual_rms_deg(held, count, bias_rates(calibration, count, s)),
            "full_fit_rms_deg": residual_rms_deg(held, count, coefficients),
            "raw_axis_split": axis_split(held),
        }
    if args.write_config:
        target = args.write_config.resolve()
        try:
            extends = os.path.relpath(args.evaluation_config.resolve(), target.parent).replace(os.sep, "/")
        except ValueError:  # different drives on Windows
            extends = str(args.evaluation_config.resolve())
        output = {
            "extends": extends,
            "prior": {"gyro_correction": {
                "matrix": [float(f"{v:.8g}") for v in s.reshape(-1)],
                "generated_by": "scripts/imu_rotation_check.py --write-config",
                "fit": {
                    "model": "theta_gt = A theta_imu + c_seq L, A = X S; S applied, X not applied",
                    "window_frames": FIT_WINDOW_FRAMES,
                    "windows": result["fit_windows"],
                    "excluded": f"windows within {args.margin_s} s of every pair in {evaluation_config}",
                    "scale_eigenvalues_percent": [round(v, 4) for v in result["scale_eigenvalues_percent"]],
                    "frame_rotation_deg_not_applied": round(result["frame_rotation_deg_not_applied"], 4),
                },
            }},
        }
        header = ("# 由 scripts/imu_rotation_check.py --write-config 生成（G1，实验记录.md）：陀螺读数 ω' = S ω。\n"
                  "# S 为 θ_gt ≈ A θ_imu + c·L 拟合中 A = X·S 的对称部分，只用与评估帧对相隔 margin 以上的 6 帧窗口拟合；\n"
                  "# 坐标系旋转 X 不施加（可能属于真值手眼标定）。请勿手改，重新生成即可复现。\n")
        target.write_text(header + yaml.safe_dump(output, sort_keys=False, allow_unicode=True), encoding="utf-8")
        result["written_config"] = str(target)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inputs", type=Path, default=ROOT / "results/inputs")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/imu_rotation_check")
    parser.add_argument("--evaluation-config", type=Path,
                        default=ROOT / "experiments/fusion/configs/esikf_round2_prior_model.yaml",
                        help="G1: configuration whose pairs are excluded from the fit and which the output extends")
    parser.add_argument("--margin-s", type=float, default=1.0,
                        help="G1: windows closer than this to an evaluation pair are not used for the fit")
    parser.add_argument("--write-config", type=Path,
                        help="G1: write the scale correction as prior.gyro_correction of a new configuration")
    args = parser.parse_args()
    directories = sorted(d for d in args.inputs.iterdir() if (d / "imu.csv").is_file())
    if not directories:
        raise SystemExit(f"No sequences with imu.csv under {args.inputs}")
    sequences = [Sequence(d) for d in directories]
    count = len(sequences)
    args.output.mkdir(parents=True, exist_ok=True)

    fit_data = collect(sequences, FIT_WINDOW_FRAMES)
    coefficients = fit(fit_data, count)
    a = coefficients[:3].T
    x, s = polar(a)
    x_vector = log_so3(x[None])[0]
    scale_values, scale_axes = np.linalg.eigh((s + s.T) / 2)
    summary = {
        "sequences": [q.name for q in sequences],
        "fit_window_frames": FIT_WINDOW_FRAMES,
        "frame_rotation_deg": math.degrees(np.linalg.norm(x_vector)),
        "frame_rotation_vector_deg": [math.degrees(v) for v in x_vector],
        "scale_eigenvalues_percent": [100 * (v - 1) for v in scale_values],
        "scale_eigenvectors": scale_axes.T.tolist(),
        "residual_bias_deg_s": {q.name: [math.degrees(v) for v in coefficients[3 + k]] for k, q in enumerate(sequences)},
        "initial_bias_deg_s": {q.name: [math.degrees(v) for v in q.bias] for q in sequences},
        "windows": {},
    }

    rows = []
    for frames in WINDOW_FRAMES:
        data = collect(sequences, frames)
        bias_only = coefficients.copy()
        bias_only[:3] = np.eye(3)  # keep the fitted bias rates, drop the frame rotation and scale
        # Two-fold split in time within every sequence: fit on one half, test on the other. Leaving a
        # whole sequence out does not work: each rotation axis is excited mainly by one recording.
        held_out = []
        for fold in (0, 1):
            trained = fit({key: value[fit_data["half"] != fold] for key, value in fit_data.items()}, count)
            test = {key: value[data["half"] == fold] for key, value in data.items()}
            held_out.append(test["gt"] - design(test, count) @ trained)
        summary["windows"][f"{frames}_frames"] = {
            "count": int(len(data["length"])),
            "median_length_s": float(np.median(data["length"])),
            "raw_error_rms_deg": rms_deg(data["error"]),
            "bias_only_residual_rms_deg": residual_rms_deg(data, count, bias_only),
            "fitted_residual_rms_deg": residual_rms_deg(data, count, coefficients),
            "time_split_residual_rms_deg": rms_deg(np.concatenate(held_out)),
            "axis_split": axis_split(data),
            "per_sequence_raw_error_rms_deg": {q.name: rms_deg(data["error"][data["seq"] == k]) for k, q in enumerate(sequences)},
        }
        angle = np.degrees(np.linalg.norm(data["gt"], axis=1))
        along = np.degrees(np.sum(data["error"] * data["gt"], 1) / np.maximum(np.linalg.norm(data["gt"], axis=1), 1e-12))
        for k in range(len(angle)):
            rows.append({"sequence": sequences[data["seq"][k]].name, "frames": frames,
                         "length_s": data["length"][k], "rotation_deg": angle[k],
                         "error_deg": math.degrees(np.linalg.norm(data["error"][k])),
                         "error_along_axis_deg": along[k]})

    # Residual time offset: shift the IMU clock and refit on one-frame windows, where timing matters most.
    scan = []
    for offset in OFFSETS_MS:
        data = collect(sequences, 1, offset / 1000)
        scan.append((float(offset), residual_rms_deg(data, count, fit(data, count))))
    best = min(scan, key=lambda item: item[1])
    summary["time_offset_scan"] = {
        "residual_rms_deg_at_0_ms": dict(scan)[0.0],
        "best_offset_ms": best[0], "residual_rms_deg_at_best": best[1],
        "convention": "IMU timestamps + offset = groundtruth time",
        "curve": scan,
    }

    if args.write_config:
        summary["g1_gyro_correction"] = gyro_calibration(sequences, count, args)

    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with (args.output / "windows.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"frame rotation X: {summary['frame_rotation_deg']:.4f} deg, vector {np.round(summary['frame_rotation_vector_deg'], 4).tolist()} deg")
    print(f"scale S eigenvalues: {np.round(summary['scale_eigenvalues_percent'], 3).tolist()} %")
    for key, w in summary["windows"].items():
        split = w["axis_split"]
        print(f"{key} ({w['count']} windows, median {1000 * w['median_length_s']:.0f} ms): raw {w['raw_error_rms_deg']:.4f} deg, "
              f"bias only {w['bias_only_residual_rms_deg']:.4f}, fitted {w['fitted_residual_rms_deg']:.4f}, "
              f"time split {w['time_split_residual_rms_deg']:.4f}; "
              f"along axis {split['along_axis_rms_deg']:.4f} deg ({split['along_axis_percent_of_rotation']:.2f}% of rotation), "
              f"across {split['across_axis_rms_deg']:.4f} deg ({split['across_axis_percent_of_rotation']:.2f}%), "
              f"{split['windows']} windows over {MIN_AXIS_ANGLE_DEG} deg")
    t = summary["time_offset_scan"]
    print(f"time offset: best {t['best_offset_ms']:+.1f} ms, residual {t['residual_rms_deg_at_best']:.4f} deg "
          f"(at 0 ms {t['residual_rms_deg_at_0_ms']:.4f} deg)")
    if args.write_config:
        g = summary["g1_gyro_correction"]
        print(f"G1: fit on {g['fit_windows']} six-frame windows at least {g['margin_s']} s from the evaluation pairs; "
              f"S eigenvalues {np.round(g['scale_eigenvalues_percent'], 3).tolist()} %, "
              f"frame rotation {g['frame_rotation_deg_not_applied']:.4f} deg not applied")
        for key, h in g["held_out_near_evaluation"].items():
            print(f"  held out near evaluation, {key} ({h['windows']} windows): bias only {h['bias_only_rms_deg']:.4f} deg, "
                  f"with S {h['scale_corrected_rms_deg']:.4f} deg, with full A {h['full_fit_rms_deg']:.4f} deg")
        print(f"Wrote {g['written_config']}")
    print(f"Wrote {args.output / 'summary.json'} and windows.csv")


if __name__ == "__main__":
    main()

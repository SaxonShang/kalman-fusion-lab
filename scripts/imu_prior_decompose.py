#!/usr/bin/env python3
"""Where the translation error of the pure-IMU prior comes from, using only the published inputs.

Replays the prior of lab_prepare (ImuInit on the first image with ≥ 200 IMU samples, then
ImuPreintegrationCov frame by frame: midpoint steps, zero-order hold up to the image time) in
three arms — raw gyro, gyro corrected by the G1 matrix S, and groundtruth attitude with the raw
accelerometer — and evaluates every pair against the conditional reference of fusion_report.py,
X_a B G_a⁻¹ G_b B⁻¹. The translation error is split into

    velocity  (v̂_a − C v_a) Δt          C = R̂_a R_aᵀ maps the base frame into the estimator world
    tilt      ½ (ĝ − C g) Δt²            anchor attitude against gravity
    interval  the rest: what the IMU samples between a and b contribute

so "anchor velocity reset to truth" is the error minus the velocity term. --check compares the raw
and S arms with the published per-pair prior errors of a results package.

    python3 scripts/imu_prior_decompose.py --check results/esikf/20261030_8f23550/audit/paired_comparison.csv
"""

import argparse
import csv
import math
from pathlib import Path
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fusion_lab import load_config  # noqa: E402
from imu_rotation_check import exp_so3, extrinsic_rotation, log_so3, quaternion_rotation  # noqa: E402

INIT_SAMPLES = 200


def exp1(v):
    return exp_so3(v[None])[0]


def log1(r):
    return log_so3(r[None])[0]


class Sequence:
    def __init__(self, directory):
        self.name = directory.name
        imu = np.loadtxt(directory / "imu.csv", delimiter=",", skiprows=1, usecols=(1, 3, 4, 5, 6, 7, 8))
        self.t, self.acc, self.gyro = imu[:, 0], imu[:, 1:4], np.radians(imu[:, 4:7])  # gyro CSV is deg/s
        with (directory / "frames.csv").open(newline="") as stream:
            frames = list(csv.DictReader(stream))
        self.end = np.array([float(f["end_time"]) for f in frames])
        self.index_of = {int(f["source_frame_id"]): k for k, f in enumerate(frames)}
        gt = np.array([[float(v) for v in line.split()] for line in (directory / "groundtruth_tum.txt").read_text().splitlines()
                       if line.strip() and not line.startswith("#")])
        params = yaml.safe_load((directory / "params.yaml").read_text(encoding="utf-8"))
        self.tbc = np.eye(4)  # T_imu_cam, as fusion_report.camera_extrinsic
        self.tbc[:3, :3] = extrinsic_rotation(params)
        self.tbc[:3, 3] = params["T_imu_cam"]["translation"]
        self.cam = np.tile(np.eye(4), (len(gt), 1, 1))
        self.cam[:, :3, :3] = quaternion_rotation(gt[:, 4:8])
        self.cam[:, :3, 3] = gt[:, 1:4]
        body = self.cam @ np.linalg.inv(self.tbc)  # T_base_imu
        self.body_rot = body[:, :3, :3]
        self.body_vel = np.gradient(body[:, :3, 3], self.end, axis=0)

    def body_rotation_at(self, t):
        k = int(np.clip(np.searchsorted(self.end, t, side="right") - 1, 0, len(self.end) - 2))
        u = (t - self.end[k]) / (self.end[k + 1] - self.end[k])
        return self.body_rot[k] @ exp1(u * log1(self.body_rot[k].T @ self.body_rot[k + 1]))

    def propagate(self, gyro_matrix=None, true_attitude=False):
        """(R, p, v) at every image time after initialization, and the gravity vector."""
        t, acc = self.t, self.acc
        gyro = self.gyro if gyro_matrix is None else self.gyro @ gyro_matrix.T
        states, start = {}, None
        for f, T in enumerate(self.end):
            count = int(np.searchsorted(t, T, side="right"))
            if start is None:
                if count < INIT_SAMPLES:
                    continue
                mean = acc[:count].mean(0)
                bg = gyro[:count].mean(0)
                down = mean / np.linalg.norm(mean)
                axis = np.cross(down, [0.0, 0.0, 1.0])
                R = exp1(axis / np.linalg.norm(axis) * math.acos(np.clip(down[2], -1, 1)))
                g = np.array([0.0, 0.0, -np.linalg.norm(mean)])
                p, v = np.zeros(3), np.zeros(3)
                # Groundtruth attitude keeps the estimator's initial tilt, so only the propagation changes.
                align = R @ self.body_rotation_at(T).T
                start = T
            else:
                first = max(int(np.searchsorted(t, start, side="right")) - 1, 0)
                for a, b in zip(range(first, count - 1), range(first + 1, count)):
                    dt = t[b] - (start if (a == first and t[a] < start) else t[a])
                    if dt <= 0:
                        continue
                    R_prev = R
                    R = align @ self.body_rotation_at(t[b]) if true_attitude else \
                        R @ exp1(0.5 * (gyro[a] + gyro[b]) * dt - bg * dt)
                    a_world = 0.5 * (R_prev @ acc[a] + R @ acc[b]) + g
                    p, v = p + v * dt + 0.5 * a_world * dt * dt, v + a_world * dt
                last = count - 1
                tail = T - t[last]
                if tail > 0:  # zero-order hold of the last sample up to the image time
                    R_prev = R
                    R = align @ self.body_rotation_at(T) if true_attitude else R @ exp1((gyro[last] - bg) * tail)
                    a_world = 0.5 * (R_prev @ acc[last] + R @ acc[last]) + g
                    p, v = p + v * tail + 0.5 * a_world * tail * tail, v + a_world * tail
                start = T
            states[f] = (R, p, v)
        return states, g

    def errors(self, states, g, anchor, current):
        ia, ib = self.index_of[anchor], self.index_of[current]
        dt = self.end[ib] - self.end[ia]
        rel = self.tbc @ np.linalg.inv(self.cam[ia]) @ self.cam[ib] @ np.linalg.inv(self.tbc)
        (Ra, pa, va), (Rb, pb, _) = states[ia], states[ib]
        error = pb - (pa + Ra @ rel[:3, 3])
        C = Ra @ self.body_rot[ia].T
        velocity = (va - C @ self.body_vel[ia]) * dt
        tilt = 0.5 * (g - C @ np.array([0.0, 0.0, -np.linalg.norm(g)])) * dt * dt
        return {
            "trans_mm": 1e3 * np.linalg.norm(error),
            "rot_deg": math.degrees(np.linalg.norm(log1(Rb.T @ Ra @ rel[:3, :3]))),
            "velocity_mm": 1e3 * np.linalg.norm(velocity),
            "tilt_mm": 1e3 * np.linalg.norm(tilt),
            "interval_mm": 1e3 * np.linalg.norm(error - velocity - tilt),
            "velocity_reset_mm": 1e3 * np.linalg.norm(error - velocity),
            "anchor_velocity_error_mm_s": 1e3 * np.linalg.norm(va - C @ self.body_vel[ia]),
            "anchor_tilt_deg": math.degrees(math.acos(np.clip((C @ np.array([0.0, 0.0, 1.0]))[2], -1, 1))),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inputs", type=Path, default=ROOT / "results/inputs")
    parser.add_argument("--evaluation-config", type=Path,
                        default=ROOT / "experiments/fusion/configs/esikf_round2_prior_model.yaml")
    parser.add_argument("--gyro-config", type=Path, default=ROOT / "experiments/fusion/configs/esikf_round3_gyro.yaml",
                        help="configuration whose prior.gyro_correction.matrix is the S arm")
    parser.add_argument("--check", type=Path, help="paired_comparison.csv of a results package (round 2 vs 3 priors)")
    parser.add_argument("--csv", type=Path, help="write the per-pair split")
    args = parser.parse_args()
    pairs = load_config(args.evaluation_config.resolve())["pairs"]
    S = np.array(load_config(args.gyro_config.resolve())["prior"]["gyro_correction"]["matrix"], float).reshape(3, 3)
    rows = []
    for directory in sorted(d for d in args.inputs.iterdir() if (d / "imu.csv").is_file()):
        sequence = Sequence(directory)
        arms = {"raw": sequence.propagate(), "S": sequence.propagate(S),
                "true_attitude": sequence.propagate(true_attitude=True)}
        for pair in pairs:
            row = {"sequence": sequence.name, "case": pair["name"]}
            for arm, (states, g) in arms.items():
                for key, value in sequence.errors(states, g, int(pair["anchor"]), int(pair["current"])).items():
                    row[f"{arm}_{key}"] = value
            rows.append(row)
    if args.check:
        with args.check.open(encoding="utf-8") as stream:
            published = {(r["sequence"], r["case"]): r for r in csv.DictReader(stream)}
        worst = max(max(abs(r["raw_trans_mm"] - 1e3 * float(published[(r["sequence"], r["case"])]["prior_trans_error_m_round2"])),
                        abs(r["S_trans_mm"] - 1e3 * float(published[(r["sequence"], r["case"])]["prior_trans_error_m_round3"])))
                    for r in rows)
        print(f"largest difference to the published prior translation errors: {worst:.4f} mm over {len(rows)} pairs")
    columns = ("trans_mm", "velocity_reset_mm", "rot_deg")
    print(f"{'case':18s} {'n':>3s} " + " ".join(f"{arm + ' ' + c:>26s}" for arm in arms for c in columns))
    groups = [(c, lambda r, c=c: r["case"] == c) for c in dict.fromkeys(r["case"] for r in rows)]
    groups += [("all", lambda r: True)]
    for name, keep in groups:
        chosen = [r for r in rows if keep(r)]
        rms = [math.sqrt(sum(r[f"{arm}_{c}"] ** 2 for r in chosen) / len(chosen)) for arm in arms for c in columns]
        print(f"{name:18s} {len(chosen):3d} " + " ".join(f"{v:26.4f}" for v in rms))
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()

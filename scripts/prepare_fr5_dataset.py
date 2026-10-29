#!/usr/bin/env python3
"""Build SIO_SLAM manifests for FR5 high-speed recordings without copying images.

Requires Python 3.10+ and PyYAML (Ubuntu package python3-yaml).
See docs/FR5_DATASET.md for the timestamp and calibration conventions.
"""

import argparse
import bisect
import configparser
import csv
from decimal import Decimal
import hashlib
import io
import json
import math
import os
from pathlib import Path
import statistics
import tempfile

import yaml


ROOT = Path(__file__).resolve().parents[1]
GENERATOR = "SIO_SLAM/scripts/prepare_fr5_dataset.py"
FIELDS = ("ax", "ay", "az", "gx", "gy", "gz")
OUTPUTS = {"left", "right", "frames.csv", "imu.csv", "groundtruth_tum.txt",
           "params.yaml", "metadata.json"}


def seconds(ns):
    """Avoid losing sub-microsecond precision by converting epoch ns to float."""
    whole, fraction = divmod(ns, 1_000_000_000)
    return f"{whole}.{fraction:09d}"


def csv_text(header, rows):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return stream.getvalue()


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def strictly_increasing(values, name):
    require(len(values) > 1 and all(a < b for a, b in zip(values, values[1:])),
            f"{name}: timestamps must be strictly increasing")


def make_params(calib, template):
    intrinsics_file = calib / "camera_s003_refit.yaml"
    imu_file = calib / "camera_imu.yaml"
    intrinsics = yaml.safe_load(intrinsics_file.read_text())
    imu = yaml.safe_load(imu_file.read_text())
    require(intrinsics["rectified"] and not any(intrinsics["distortion"]),
            "Expected already-rectified stereo images with zero distortion")
    require(intrinsics["camera_serial"] == imu["camera_serial"],
            "Camera and IMU calibrations have different serial numbers")
    require(imu["coordinate_system"] == "IMAGE", "Expected SDK IMAGE coordinates")
    require(imu["gyroscope"]["unit"] == "SENSORS_UNIT.DEG_SEC" and
            imu["accelerometer"]["unit"] == "SENSORS_UNIT.M_SEC_2",
            "Unexpected IMU units")

    stereo_file = calib / f"SN{intrinsics['camera_serial']}.conf"
    stereo = configparser.ConfigParser()
    stereo.read(stereo_file)
    baseline = stereo.getfloat("STEREO", "Baseline") / 1000.0
    require(baseline > 0, "Invalid stereo baseline")

    # SDK supplies IMU -> camera; SIO_SLAM needs camera -> IMU.
    transform = imu["camera_imu_transform"]
    require(len(transform) == 4 and all(len(row) == 4 for row in transform),
            "Expected a 4x4 camera_imu_transform")
    rotation = [[transform[j][i] for j in range(3)] for i in range(3)]
    translation = [-sum(rotation[i][j] * transform[j][3] for j in range(3))
                   for i in range(3)]
    rpy = [math.atan2(rotation[2][1], rotation[2][2]),
           math.atan2(-rotation[2][0], math.hypot(rotation[0][0], rotation[1][0])),
           math.atan2(rotation[1][0], rotation[0][0])]

    params = yaml.safe_load(template.read_text())
    # Keep the project's working resolution to limit backend memory on the 8 GB GPU.
    sx = 1280 / intrinsics["image_width"]
    sy = 720 / intrinsics["image_height"]
    params["camera intrinsic"].update({
        "fx": intrinsics["fx"] * sx, "fy": intrinsics["fy"] * sy,
        "cx": intrinsics["cx"] * sx, "cy": intrinsics["cy"] * sy,
        "baseline": baseline, "distortion_coefficients": [0.0] * 5,
        "source_width": 1280, "source_height": 720,
        "inference_width": 896, "inference_height": 448,
    })
    params["T_imu_cam"] = {"rotation": rpy, "translation": translation}
    params["imu parameters"].update({
        "accel_noise_density": imu["accelerometer"]["noise_density"],
        "gyro_noise_density": math.radians(imu["gyroscope"]["noise_density"]),
    })
    params["imu_preprocessor"]["sample_rate_hz"] = imu["gyroscope"]["sampling_rate_hz"]
    files = [intrinsics_file, imu_file, stereo_file, template]
    provenance = {
        "input_sha256": {str(p.relative_to(ROOT) if p.is_relative_to(ROOT) else p):
                         hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        "camera_serial": intrinsics["camera_serial"],
        "raw_image_size": [intrinsics["image_width"], intrinsics["image_height"]],
        "working_image_size": [1280, 720],
        "crop_image_size": [1280, 640],
        "inference_image_size": [896, 448],
        "extrinsic": "T_imu_cam = inverse(SDK camera_imu_transform)",
        "noise_parameters": "SDK nominal values; YAML noise fields are not currently "
                            "consumed by the core IMU integrator",
    }
    return params, provenance


def prepare_sequence(source, skip, params, calibration):
    timeline = read_csv(source / "img_timeline.csv")
    image_ns = [int(row["zed_timestamp_ms"]) * 1_000_000 for row in timeline]
    strictly_increasing(image_ns, str(source / "img_timeline.csv"))
    require(0 <= skip < len(timeline) - 1, f"{source}: invalid number of skipped frames")
    for row in timeline:
        for column, side in (("filename_left", "left"), ("filename_right", "right")):
            path = Path(row[column])
            require(len(path.parts) == 2 and path.parts[0] == side and
                    path.stem == row["zed_timestamp_ms"] and (source / path).is_file(),
                    f"Missing or unexpected stereo image: {source / path}")

    report = json.loads((source / "frame_lag_report.json").read_text())
    lag_ms = float(report["lag_ms"])
    require(math.isfinite(lag_ms) and report.get("ok") is True and
            report.get("frames_late") == 0 and not report.get("at_scan_edge", True),
            f"{source}: frame lag check did not pass; inspect frame_lag_report.json")
    # Report compares image(t) to gyro(t - lag). Preserve image/GT timestamps and
    # instead shift every IMU sample: t_aligned = t_raw + lag_ms / 1000.
    shift_ns = round(lag_ms * 1_000_000)
    imu = read_csv(source / "zed_imu.csv")
    imu_ns = [int(row["zed_imu_timestamp_ns"]) + shift_ns for row in imu]
    strictly_increasing(imu_ns, str(source / "zed_imu.csv"))
    require(all(math.isfinite(float(row[key])) for row in imu for key in FIELDS),
            f"{source}: nonfinite IMU measurements")
    first_start = image_ns[skip - 1] if skip else image_ns[0] - (image_ns[1] - image_ns[0])
    # Capture ends before the last image's compensated exposure time. The current
    # integrator holds the last sample to t_end; permit at most its 10 ms gap limit.
    require(imu_ns[0] <= first_start and imu_ns[-1] >= image_ns[-1] - 10_000_000,
            f"{source}: IMU does not cover the selected interval within 10 ms")

    frame_rows, imu_rows = [], []
    cursor = bisect.bisect_right(imu_ns, first_start)
    first_imu = cursor
    start = first_start
    for frame_id, source_id in enumerate(range(skip, len(timeline))):
        end = image_ns[source_id]
        stop = bisect.bisect_right(imu_ns, end)
        require(stop > cursor, f"{source}: no IMU for frame {source_id}")
        row = timeline[source_id]
        frame_rows.append([frame_id, seconds(start), seconds(end), row["filename_left"],
                           row["filename_right"], stop - cursor, source_id])
        for idx in range(cursor, stop):
            # Loader converts deg/s to rad/s; keep raw angular velocity here.
            imu_rows.append([frame_id, seconds(imu_ns[idx]), row["zed_timestamp_ms"],
                             *(imu[idx][key] for key in FIELDS)])
        start, cursor = end, stop
    selected_ns = imu_ns[first_imu:cursor]
    gaps = [b - a for a, b in zip(selected_ns, selected_ns[1:])]
    require(max(gaps) <= 10_000_000, f"{source}: IMU gap exceeds frontend's 10 ms limit")
    require(len(selected_ns) >= 200, f"{source}: fewer than 200 initialization samples")
    # All six recordings are stationary until ~5.15 s from the first image.
    initialization_s = (selected_ns[199] - image_ns[0]) / 1e9
    require(initialization_s < 5.15,
            f"{source}: 200-sample IMU initialization extends beyond initial static segment")

    gt_by_ns = {}
    for line in (source / "groundtruth_tum.txt").read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        require(len(fields) == 8 and all(math.isfinite(float(x)) for x in fields),
                f"{source}: malformed TUM groundtruth")
        timestamp = int(Decimal(fields[0]) * 1_000_000_000)
        require(timestamp not in gt_by_ns, f"{source}: duplicate groundtruth timestamp")
        gt_by_ns[timestamp] = line
    require(all(t in gt_by_ns for t in image_ns[skip:]),
            f"{source}: missing frame-matched groundtruth")
    groundtruth = "# timestamp tx ty tz qx qy qz qw; corrected LEFT CAMERA pose in FR5 base\n"
    groundtruth += "\n".join(gt_by_ns[t] for t in image_ns[skip:]) + "\n"
    gt_report = json.loads((source / "groundtruth_report.json").read_text())
    metadata = {
        "generator": GENERATOR, "format_version": 1, "source": str(source),
        "raw_frames": len(timeline), "skipped_frames": skip, "frames": len(frame_rows),
        "imu_samples": len(imu_rows), "frame_lag_ms": lag_ms,
        "image_to_imu_td_ms": -lag_ms, "imu_time_shift_ns": shift_ns,
        "time_convention": "Images and GT unchanged; IMU aligned seconds = raw IMU "
                           "seconds + lag_ms/1000; intervals (previous_image, image]",
        "imu_units_in_csv": {"acceleration": "m/s^2", "angular_velocity": "deg/s"},
        "imu_measurement_frame": "SDK IMU IMAGE axes; not rotated to camera axes",
        "imu_gap_median_ms": statistics.median(gaps) / 1e6,
        "imu_gap_max_ms": max(gaps) / 1e6,
        "last_image_minus_last_imu_ms": (image_ns[-1] - selected_ns[-1]) / 1e6,
        "initialization_200th_sample_s_from_raw_first_image": initialization_s,
        "groundtruth": "Corrected source poses copied verbatim for selected images; "
                       "no additional exposure, lag, scale or endpoint correction",
        "groundtruth_warnings": gt_report.get("warnings", []),
        "calibration": calibration,
    }
    return {
        "frames.csv": csv_text(["frame_id", "st_time", "end_time", "left_path",
                                "right_path", "imu_count", "source_frame_id"], frame_rows),
        "imu.csv": csv_text(["frame_id", "timestamp_sec", "zed_timestamp_ms", *FIELDS],
                            imu_rows),
        "groundtruth_tum.txt": groundtruth,
        "params.yaml": "# Generated FR5 ZED X calibration; see metadata.json.\n" +
                       yaml.safe_dump(params, sort_keys=False, allow_unicode=True),
        "metadata.json": json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
    }, metadata


def check_destination(output):
    if not output.exists() and not output.is_symlink():
        return
    require(output.is_dir() and not output.is_symlink(), f"Not a generated directory: {output}")
    marker = output / "metadata.json"
    require(marker.is_file() and json.loads(marker.read_text()).get("generator") == GENERATOR,
            f"Refusing to overwrite a directory without this script's marker: {output}")
    require({p.name for p in output.iterdir()} <= OUTPUTS,
            f"{output}: contains additional files; choose a new destination")
    for side in ("left", "right"):
        require((output / side).is_symlink(), f"Refusing to replace original images: {output / side}")


def write_sequence(output, source, files):
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".fr5-", dir=output.parent) as tmp:
        stage = Path(tmp)
        for name, content in files.items():
            (stage / name).write_text(content)
        for side in ("left", "right"):
            (stage / side).symlink_to(os.path.relpath(source / side, stage), target_is_directory=True)
        if output.exists():
            # Replace only generated files/links; never recurse into the source images.
            for path in stage.iterdir():
                os.replace(path, output / path.name)
        else:
            os.replace(stage, output)


def activate(output):
    path = ROOT / "project_config.local.yaml"
    config = (yaml.safe_load(path.read_text()) or {}) if path.exists() else {}
    for key, value in (("dataset_dir", output), ("imu_file", output / "imu.csv"),
                       ("params_file", output / "params.yaml")):
        config[key] = str(value.relative_to(ROOT) if value.is_relative_to(ROOT) else value)
    content = "# This machine only; ignored by Git. Paths are relative to the project root.\n"
    content += yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(content)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=ROOT / "data/recordings/recordings_highspeed")
    parser.add_argument("--dest", type=Path, default=ROOT / "data/datasets/fr5_highspeed")
    parser.add_argument("--skip-frames", type=int, default=60)
    parser.add_argument("--activate", metavar="SEQUENCE",
                        help="Set local runtime paths; accepts full name or suffix, e.g. up")
    args = parser.parse_args()
    try:
        source, dest = args.source.resolve(), args.dest.resolve()
        require(not dest.is_relative_to(source) and not source.is_relative_to(dest),
                "Source and generated dataset roots must be separate")
        sequences = sorted(source.glob("*/save_data/img_timeline.csv"))
        require(bool(sequences), f"No FR5 recordings under {source}")
        chosen = None
        if args.activate:
            matches = [p.parent for p in sequences if p.parent.parent.name == args.activate or
                       p.parent.parent.name.split("_", 2)[-1] == args.activate]
            require(len(matches) == 1, f"Sequence must match exactly once: {args.activate}")
            chosen = dest / matches[0].parent.name
        params, calibration = make_params(source / "calib", ROOT / "config/params.yaml")
        prepared = []
        for timeline in sequences:
            raw = timeline.parent
            output = dest / raw.parent.name
            check_destination(output)
            files, metadata = prepare_sequence(raw, args.skip_frames, params, calibration)
            prepared.append((output, raw, files, metadata))
        for output, raw, files, metadata in prepared:
            write_sequence(output, raw, files)
            print(f"{output.name}: {metadata['frames']}/{metadata['raw_frames']} frames, "
                  f"{metadata['imu_samples']} IMU, td={metadata['image_to_imu_td_ms']:.3f} ms")
        if chosen:
            activate(chosen)
            print(f"Activated: {chosen}")
        print("Original recordings preserved; images referenced via relative symlinks.")
    except (ValueError, OSError, KeyError, csv.Error, yaml.YAMLError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()

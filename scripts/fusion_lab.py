#!/usr/bin/env python3
"""Build, run and inspect isolated frontend fusion experiments. No network access."""

import argparse
import copy
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import yaml

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build/local/fusion"
BINARY = BUILD / "sio_fusion_lab"
CONFIG = ROOT / "experiments/fusion/configs/model.yaml"
RESULTS = ROOT / "artifacts/fusion"
CACHES = ROOT / "data/cache/fusion"
SUITES = {
    "all": "*",
    "imu": "ImuIntegratorTest.*:ImuPreprocessorTest.*",
    "gicp": "StereoGicpTest.*:StereoObservationCompatTest.*:UtilsTest.*Gicp*:UtilsTest.*StereoCovariance*:ColorIcpConfigTest.*",
    "eskf": "StateEstimatorTest.*Eskf*",
    "esikf": ("StateEstimatorTest.*Esikf*:StateEstimatorTest.*Observation*:"
              "ImuIntegratorTest.*Anchor*:ImuIntegratorTest.*Init*:ImuIntegratorTest.*Noise*:"
              "ImuIntegratorTest.*GyroScale*:ImuIntegratorTest.*Transition*"),
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_config(path, seen=None):
    path = path.resolve()
    seen = set() if seen is None else seen
    if path in seen:
        raise ValueError(f"Recursive experiment configuration: {path}")
    seen.add(path)
    overlay = yaml.safe_load(path.read_text(encoding="utf-8"))
    parent = overlay.pop("extends", None)
    if parent is None:
        return overlay
    base = load_config(path.parent / parent, seen)
    def merge(destination, values):
        for key, value in values.items():
            if isinstance(value, dict) and isinstance(destination.get(key), dict):
                merge(destination[key], value)
            else:
                destination[key] = value
    merge(base, overlay)
    return base


def source_hashes():
    files = list((ROOT / "src").glob("*.cpp"))
    files += [p for p in (ROOT / "include").rglob("*") if p.suffix in (".h", ".hpp")]
    files += list((ROOT / "experiments/fusion/src").glob("*"))
    files += [ROOT / "apps/export_disparity.cpp", ROOT / "scripts/fusion_lab.py"]
    files += [p for p in (ROOT / "modules/registration").rglob("*")
              if p.suffix in (".h", ".cpp", ".json") and "tests" not in p.parts]
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(files) if p.is_file()}


def latest_observations():
    choices = sorted(RESULTS.glob("*/observations.yml.gz"), key=lambda p: p.stat().st_mtime)
    choices = [p for p in choices if "gicp" in yaml.safe_load(
        (p.parent / "resolved.yaml").read_text()).get("completed_stages", ["gicp"])]
    if not choices:
        raise ValueError("Run 'python3 scripts/fusion_lab.py run --mode gicp' first")
    return choices[-1]


def validate_cache(cache, key):
    metadata = json.loads((cache / "complete.json").read_text())
    if metadata["cache_key"] != key:
        raise ValueError(f"Input cache identity mismatch: {cache}")
    for name, expected in metadata["files"].items():
        if digest(cache / name) != expected:
            raise ValueError(f"Frozen input was modified: {cache / name}")


def execute(args, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as stream:
        process = subprocess.run(["bash", "-c", 'source "$1/env.sh"; shift; exec "$@"',
                                  "fusion-lab", str(ROOT), *map(str, args)],
                                 cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    if process.returncode:
        tail = "\n".join(log.read_text(errors="replace").splitlines()[-25:])
        raise RuntimeError(f"Command failed ({process.returncode}); log: {log}\n{tail}")


def build(with_tests=False):
    log = ROOT / "build/fusion-lab-build.log"
    if not (BUILD / "CMakeCache.txt").is_file():
        execute(["cmake", "-S", ROOT, "-B", BUILD, "-G", "Ninja",
                 "-DCMAKE_BUILD_TYPE=RelWithDebInfo", "-DSIO_BUILD_RUNTIME=OFF",
                 "-DSIO_BUILD_FUSION_LAB=ON", "-DSIO_ENABLE_ZED=OFF",
                 "-DSIO_ENABLE_LIBTORCH=OFF", "-DBUILD_TESTING=ON"], log)
    targets = ["sio_fusion_lab", "sio_unit_tests"] if with_tests else ["sio_fusion_lab"]
    print(f"Building fusion targets ({log.relative_to(ROOT)})", flush=True)
    execute(["cmake", "--build", BUILD, "--parallel", os.environ.get("SIO_BUILD_JOBS", "2"),
             "--target", *targets], log)
    commands = ROOT / "compile_commands.json"
    if commands.is_symlink() or not commands.exists():
        commands.unlink(missing_ok=True)
        commands.symlink_to("build/local/fusion/compile_commands.json")


def allocate(sequence, mode):
    name = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + f"_{sequence}_{mode}"
    output = RESULTS / name
    output.mkdir(parents=True, exist_ok=False)
    return output


def save_config(cfg, output):
    path = output / "resolved.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    return path


def model_disparity(cfg, inputs, output, hashes):
    plan = Path(cfg["depth"]["plan_file"]).expanduser()
    if not plan.is_absolute():
        plan = ROOT / plan
    plan = plan.resolve()
    if not plan.is_file():
        raise ValueError(f"Model engine missing: {plan}; run bash scripts/build_models.sh disp")
    origin = {}
    if cfg["depth"].get("onnx_file"):
        onnx_file = Path(cfg["depth"]["onnx_file"]).expanduser()
        if not onnx_file.is_absolute():
            onnx_file = ROOT / onnx_file
        original_sha = digest(onnx_file)
        built_source = Path(str(plan) + ".source-sha256").read_text().strip()
        origin = {"onnx_file": str(onnx_file.resolve()), "onnx_sha256": original_sha}
        if built_source != original_sha:
            proof = ROOT / "models/onnx/disp_cost_volume.verified.json"
            validation = json.loads(proof.read_text())
            if (validation["source_sha256"] != original_sha or
                    validation["output_sha256"] != built_source):
                raise ValueError("ONNX/engine version mismatch; rerun bash scripts/build_models.sh disp")
            origin["graph_validation"] = validation
    relevant = {k: v for k, v in hashes.items() if k in (
        "apps/export_disparity.cpp", "include/TrtEngine.hpp", "src/dataset_loader.cpp",
        "src/image_preprocessor.cpp", "src/sio_stereo_geometry.cpp", "src/sio_config.cpp")
        or k.startswith("include/")}
    identity = {"plan_sha256": digest(plan), "model_origin": origin, "inputs": inputs,
                "params_sha256": cfg["params_sha256"], "sources": relevant,
                "tensor_input": "RGB float32 NCHW 0..255", "output": "raw disparity pixels"}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    cache = ROOT / "data/cache/model_disparity" / cfg["sequence"] / key
    if (cache / "complete.json").is_file():
        validate_cache(cache, key)
        print(f"Reusing model disparity: {cache}", flush=True)
    else:
        if cache.exists():
            raise ValueError(f"Incomplete model cache exists: {cache}")
        gpu_build = ROOT / "build/local/gpu"
        if not (gpu_build / "CMakeCache.txt").is_file():
            raise ValueError("GPU build directory missing; run bash scripts/build.sh gpu first")
        execute(["cmake", "--build", gpu_build, "--target", "sio_export_disparity", "--parallel", "2"],
                ROOT / "build/model-tool-build.log")
        model_input = output / "model-input.yaml"
        model_input.write_text(yaml.safe_dump(cfg, sort_keys=False))
        cache.parent.mkdir(parents=True, exist_ok=True)
        print("Exporting model disparity for selected frames (no RAFT or backend).", flush=True)
        with tempfile.TemporaryDirectory(prefix=".export-", dir=cache.parent) as temporary:
            stage = Path(temporary)
            execute([gpu_build / "sio_export_disparity", "export", plan, model_input, stage],
                    output / "model-inference.log")
            files = {p.name: digest(p) for p in stage.iterdir() if p.is_file()}
            (stage / "complete.json").write_text(json.dumps(
                {"cache_key": key, "files": files, "identity": identity}, indent=2) + "\n")
            os.replace(stage, cache)
    cfg["depth"].update(provider="disparity_cache", disparity_dir=str(cache),
                        model_plan=str(plan), model_plan_sha256=identity["plan_sha256"],
                        model_origin=origin,
                        model_input=identity["tensor_input"], model_output=identity["output"])


def resolve_config(args, sequence, output):
    cfg = load_config(args.config)
    config_path = Path(args.config).resolve()
    cfg["config_file"] = str(config_path.relative_to(ROOT)) if config_path.is_relative_to(ROOT) else str(config_path)
    datasets = sorted((ROOT / "data/datasets/fr5_highspeed").glob("*/metadata.json"))
    matches = [p.parent for p in datasets if p.parent.name == sequence or
               p.parent.name.split("_", 2)[-1] == sequence]
    if len(matches) != 1:
        raise ValueError(f"Prepared sequence not found: {sequence}; run prepare_fr5_dataset.py first")
    dataset = matches[0]
    cfg["sequence"] = dataset.name
    cfg["dataset_dir"] = str(dataset)
    cfg["imu_file"] = str(dataset / "imu.csv")
    cfg["groundtruth_file"] = str(dataset / "groundtruth_tum.txt")
    if args.case:
        cfg["pairs"] = [p for p in cfg["pairs"] if p["name"] in args.case]
        if len(cfg["pairs"]) != len(set(args.case)):
            raise ValueError("Unknown or repeated --case name")
    if not cfg["pairs"] or len({p["name"] for p in cfg["pairs"]}) != len(cfg["pairs"]):
        raise ValueError("Case names must be unique and nonempty")
    params = yaml.safe_load((dataset / "params.yaml").read_text())
    params["color_icp"]["num_threads"] = cfg["threads"]
    params["color_icp"]["icp_max_iterations"] = cfg["gicp"]["max_iterations"]
    params["color_icp"].update(cfg["gicp"].get("stereo_options", {}))
    cfg["registration_upstream"] = json.loads((ROOT / "modules/registration/UPSTREAM.json").read_text())
    params_file = output / "params.yaml"
    params_file.write_text(yaml.safe_dump(params, sort_keys=False))
    cfg["params_file"] = str(params_file)
    cfg["params_sha256"] = digest(params_file)
    cfg["completed_stages"] = []

    hashes = source_hashes()
    selected = {int(p[k]) for p in cfg["pairs"] for k in ("anchor", "current")}
    with (dataset / "frames.csv").open(newline="") as stream:
        frames = {int(row["source_frame_id"]): row for row in csv.DictReader(stream)}
    if not selected <= frames.keys():
        raise ValueError("Selected source frame ids are not in this recording")
    inputs = {"frames.csv": digest(dataset / "frames.csv"), "imu.csv": digest(dataset / "imu.csv")}
    for frame_id in sorted(selected):
        for key in ("left_path", "right_path"):
            name = frames[frame_id][key]
            inputs[name] = digest(dataset / name)
    if cfg["depth"]["provider"] == "tensorrt":
        model_disparity(cfg, inputs, output, hashes)
    if cfg["depth"]["provider"] == "disparity_cache":
        disparity_root = Path(cfg["depth"]["disparity_dir"]).expanduser().resolve()
        cfg["depth"]["disparity_dir"] = str(disparity_root)
        for frame_id in sorted(selected):
            stamp = Path(frames[frame_id]["left_path"]).stem
            path = disparity_root / f"{stamp}.yml.gz"
            if not path.is_file():
                path = disparity_root / f"{stamp}.yml"
            inputs[str(path)] = digest(path)
    elif cfg["depth"]["provider"] != "sgbm":
        raise ValueError("depth.provider must be tensorrt, sgbm or disparity_cache")
    identity = {"dataset": str(dataset), "files": inputs, "params": params,
                "depth": cfg["depth"], "pairs": cfg["pairs"], "source_hashes": hashes}
    if cfg.get("prior"):
        # IMU noise, initial covariance and the static check change the frozen priors.
        identity["prior"] = cfg["prior"]
    cfg["cache_key"] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    cfg["cache_dir"] = str(CACHES / dataset.name / cfg["cache_key"])
    cfg["output_dir"] = str(output)
    cfg["source_hashes"] = hashes
    cfg["input_hashes"] = inputs
    cfg["groundtruth_sha256"] = digest(dataset / "groundtruth_tum.txt")
    cfg["git_commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    cfg["binary_sha256"] = digest(BINARY)
    return cfg


def prepare(cfg, config_file, output):
    cache = Path(cfg["cache_dir"])
    marker = cache / "complete.json"
    if marker.is_file():
        validate_cache(cache, cfg["cache_key"])
        print(f"Reusing frozen inputs: {cache.relative_to(ROOT)}", flush=True)
        return
    if cache.exists():
        raise ValueError(f"Incomplete input cache exists; choose another config or inspect: {cache}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    print("Freezing stereo inputs and pure-IMU priors; GT is reserved for reporting.", flush=True)
    with tempfile.TemporaryDirectory(prefix=".prepare-", dir=cache.parent) as directory:
        stage = Path(directory)
        execute([BINARY, "prepare", config_file, stage], output / "prepare.log")
        shutil.copy2(config_file, stage / "provenance.yaml")
        checksums = {p.name: digest(p) for p in stage.iterdir() if p.is_file()}
        (stage / "complete.json").write_text(json.dumps({"cache_key": cfg["cache_key"], "files": checksums}) + "\n")
        os.replace(stage, cache)


def report(output):
    subprocess.run([sys.executable, ROOT / "scripts/fusion_report.py", output], check=True, cwd=ROOT)


def run(args):
    build()
    outputs = []
    config_sequence = load_config(args.config).get("sequence", "up")
    sequence = args.sequence or config_sequence
    sequences = ("up", "forward", "right", "yaw_left", "pitch_up", "roll_right") if sequence == "all" else [sequence]
    for sequence in sequences:
        output = allocate(sequence, args.mode)
        cfg = resolve_config(args, sequence, output)
        config_file = save_config(cfg, output)
        prepare(cfg, config_file, output)
        if args.mode in ("all", "gicp"):
            print("Running standalone GICP; saving complete observations and covariances.", flush=True)
            execute([BINARY, "gicp", config_file, output], output / "gicp.log")
            cfg["completed_stages"].append("gicp")
            cfg["observations_sha256"] = digest(output / "observations.yml.gz")
            config_file = save_config(cfg, output)
        if args.mode == "all":
            cfg["observations"] = str(output / "observations.yml.gz")
            config_file = save_config(cfg, output)
            print("Replaying fixed GICP observations through ESKF covariance variants.", flush=True)
            execute([BINARY, "eskf", config_file, output], output / "eskf.log")
            cfg["completed_stages"].append("eskf")
            config_file = save_config(cfg, output)
        if args.mode in ("all", "esikf"):
            print("Running ESIKF from the same priors and point clouds.", flush=True)
            execute([BINARY, "esikf", config_file, output], output / "esikf.log")
            cfg["completed_stages"].append("esikf")
            save_config(cfg, output)
        report(output)
        print(f"Results: {output}\nReport: {output / 'report.html'}", flush=True)
        outputs.append(output)
    return outputs


def copy_for_replay(observations, mode):
    observations = observations.expanduser().resolve()
    if not observations.is_file():
        raise ValueError(f"Observations not found: {observations}")
    cfg = yaml.safe_load((observations.parent / "resolved.yaml").read_text())
    if cfg.get("observations_sha256") and digest(observations) != cfg["observations_sha256"]:
        raise ValueError(f"Frozen observations were modified: {observations}")
    validate_cache(Path(cfg["cache_dir"]), cfg["cache_key"])
    output = allocate(cfg["sequence"], mode)
    params = Path(cfg["params_file"])
    shutil.copy2(params, output / "params.yaml")
    cfg["params_file"] = str(output / "params.yaml")
    cfg["params_sha256"] = digest(output / "params.yaml")
    cfg["input_completed_stages"] = cfg.get("completed_stages", [])
    cfg["completed_stages"] = []
    cfg["observations"] = str(observations)
    cfg["observations_sha256"] = digest(observations)
    cfg["input_source_hashes"] = copy.deepcopy(cfg["source_hashes"])
    cfg["observation_registration_upstream"] = cfg.get("registration_upstream", {"commit": "pre-import"})
    cfg["registration_upstream"] = json.loads((ROOT / "modules/registration/UPSTREAM.json").read_text())
    cfg["source_hashes"] = source_hashes()
    cfg["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    cfg["binary_sha256"] = digest(BINARY)
    cfg["output_dir"] = str(output)
    return cfg, output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("build", help="Build a CPU experiment target with debug symbols")
    test = commands.add_parser("test", help="Run math and module regression tests")
    test.add_argument("--suite", choices=SUITES, default="all")
    run_parser = commands.add_parser("run", help="Run independent tests on frozen frame pairs")
    run_parser.add_argument("--config", type=Path, default=CONFIG)
    run_parser.add_argument("--sequence", help="up/forward/right/yaw_left/pitch_up/roll_right/all")
    run_parser.add_argument("--mode", choices=("all", "gicp", "esikf"), default="all")
    run_parser.add_argument("--case", action="append", help="Select a named case; may repeat")
    replay = commands.add_parser("replay", help="Change ESKF R while preserving frozen observations")
    replay.add_argument("observations", type=Path, nargs="?", help="Default: latest saved GICP run")
    replay.add_argument("--scales", type=float, nargs="+", default=[0.1, 1.0, 10.0])
    replay.add_argument("--prior-scale", type=float, default=1.0,
                        help="Scale frozen P for a controlled uncertainty experiment")
    debug = commands.add_parser("debug", help="Prepare a fresh VS Code debugger run")
    source = debug.add_mutually_exclusive_group()
    source.add_argument("--observations", type=Path,
                        help="Remember an observation source for subsequent IDE launches")
    source.add_argument("--latest", action="store_true", help="Resume following the latest GICP run")
    debug.add_argument("--mode", choices=("gicp", "eskf", "esikf"), default="eskf")
    report_parser = commands.add_parser("report", help="Regenerate CSV, plots and HTML")
    report_parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "build":
            build(with_tests=True)
        elif args.command == "test":
            build(with_tests=True)
            log = ROOT / "build/fusion-lab-tests.log"
            execute([BUILD / "sio_unit_tests", "--gtest_filter=" + SUITES[args.suite]], log)
            print("\n".join(log.read_text().splitlines()[-5:]))
        elif args.command == "run":
            run(args)
        elif args.command == "replay":
            build()
            cfg, output = copy_for_replay(args.observations or latest_observations(), "eskf_replay")
            cfg["eskf"]["covariance_scales"] = args.scales
            cfg["eskf"]["prior_covariance_scale"] = args.prior_scale
            execute([BINARY, "eskf", save_config(cfg, output), output], output / "eskf.log")
            cfg["completed_stages"].append("eskf")
            save_config(cfg, output)
            report(output)
            print(f"Results: {output}")
        elif args.command == "debug":
            build()
            local = ROOT / "experiments/fusion/local"
            local.mkdir(parents=True, exist_ok=True)
            selection = local / "selection.yaml"
            observations = args.observations
            if args.latest:
                selection.unlink(missing_ok=True)
            elif observations is not None:
                selection.write_text(yaml.safe_dump({"observations": str(observations.expanduser().resolve())}))
            elif selection.is_file():
                observations = Path(yaml.safe_load(selection.read_text())["observations"])
            if observations is None:
                observations = latest_observations()
            cfg, output = copy_for_replay(observations, "debug_" + args.mode)
            save_config(cfg, output)
            shutil.copy2(output / "resolved.yaml", local / "debug.yaml")
            print(f"Debugger ready: {args.mode}; new output directory: {output}")
        elif args.command == "report":
            report(args.directory.resolve())
    except (ValueError, OSError, RuntimeError, KeyError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()

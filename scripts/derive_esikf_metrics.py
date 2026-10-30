#!/usr/bin/env python3
"""Add current report diagnostics to archived runs without changing their original results.

Reads local full matrix dumps identified by the archived MANIFEST. Writes a separate
derived metrics directory; it never reruns C++ or modifies the source/archive files.
"""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import yaml

import fusion_report
from esikf_eval import summarize


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Refusing to overwrite {args.output}")
    manifest = json.loads(args.manifest.read_text())
    args.output.mkdir(parents=True)
    provenance = {"source_manifest_sha256": digest(args.manifest),
                  "tested_commit": manifest["tested_commit"],
                  "report_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  "report_sha256": digest(Path(fusion_report.__file__)),
                  "description": "Derived diagnostics only; original C++ results unchanged; no rerun.",
                  "runs": []}
    combined = []
    for run in manifest["runs"]:
        source = Path(run["source"])
        cfg = yaml.safe_load((source / "resolved.yaml").read_text())
        for key, sha_key in (("groundtruth_file", "groundtruth_sha256"), ("params_file", "params_sha256")):
            assert digest(Path(cfg[key])) == cfg[sha_key], f"Input changed: {cfg[key]}"
        raw = source / "esikf.yml.gz"
        assert digest(raw) == run["local_full_matrix_files"][raw.name]["sha256"]
        rows, _, _, _ = fusion_report.collect(source, cfg)
        old = {(r["case"], r["variant"]): r for r in csv.DictReader((source / "metrics.csv").open())}
        assert len(rows) == run["records"] == len(old)
        delta = max(abs(float(old[(r["case"], r["variant"])][key]) - r[key])
                    for r in rows for key in ("prior_trans_error_m", "prior_rot_error_deg",
                                              "result_trans_error_m", "result_rot_error_deg"))
        assert delta < 1e-12, f"Pose metric changed: {source} {delta}"
        target = args.output / "runs" / source.name
        target.mkdir(parents=True)
        shutil.copy2(source / "resolved.yaml", target / "resolved.yaml")
        fusion_report.write_csv(target / "metrics.csv", rows)
        combined.extend({"config": run["config"], "sequence": run["sequence"], **r} for r in rows)
        provenance["runs"].append({"source": str(source), "records": len(rows),
                                   "raw_sha256": digest(raw), "original_metrics_sha256": digest(source / "metrics.csv"),
                                   "max_pose_metric_change": delta})
    fusion_report.write_csv(args.output / "metrics.csv", combined)
    (args.output / "metrics.csv.gz").write_bytes(gzip.compress((args.output / "metrics.csv").read_bytes(), mtime=0))
    (args.output / "metrics.csv").unlink()
    (args.output / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2) + "\n")
    summarize(argparse.Namespace(directories=[args.output / "runs"], config=manifest["configs"],
                                 csv=args.output / "summary.csv"))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare frozen-observation ESKF solves, repropagated priors and P/R controls.

GT is read only when rendering reports; control values are fixed in advance.
"""
import argparse
import csv
from datetime import datetime
import hashlib
import html
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import yaml

import fusion_lab as lab
import fusion_report as reporting
import matplotlib.pyplot as plt


def save(manifest, directory):
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def replay(source, label, scales, prior_scale):
    cfg, output = lab.copy_for_replay(source / "observations.yml.gz", label)
    cfg["eskf"]["covariance_scales"] = scales
    cfg["eskf"]["prior_covariance_scale"] = prior_scale
    lab.execute([lab.BINARY, "eskf", lab.save_config(cfg, output), output], output / "eskf.log")
    cfg["completed_stages"].append("eskf")
    lab.save_config(cfg, output)
    lab.report(output)
    return output


def controls(references):
    lab.build()
    directory = lab.ROOT / "artifacts/fusion_diagnostics" / datetime.now().strftime("%Y%m%d_%H%M%S")
    directory.mkdir(parents=True, exist_ok=False)
    manifest = {"audit_script_sha256": lab.digest(Path(__file__)), "runs": [],
                "sensitivity": {"P_scale": 0.0001, "R_scales": [0.1, 1, 10, 1000, 100000]},
                "note": "Fixed controls, not GT-selected noise calibration; registration module unchanged"}
    for reference in references:
        cfg = yaml.safe_load((reference / "resolved.yaml").read_text())
        if cfg["depth"]["provider"] != "disparity_cache":
            raise ValueError("Use a completed model-cache run to avoid repeated model inference")
        entry = {"reference": str(reference), "sequence": cfg["sequence"],
                 "observations_sha256": lab.digest(reference / "observations.yml.gz"),
                 "reference_config_sha256": lab.digest(reference / "resolved.yaml")}
        manifest["runs"].append(entry)
        save(manifest, directory)
        print(f"[audit] {cfg['sequence']}: same frozen observation and P; new innovation solver", flush=True)
        entry["solver_only"] = str(replay(reference, "solver_audit", [0.1, 1, 10], 1.0))
        save(manifest, directory)
        print("[audit] Repropagate the corrected initial P; reuse immutable model disparity", flush=True)
        args = SimpleNamespace(config=reference / "resolved.yaml", sequence=cfg["sequence"],
                               case=None, mode="all")
        updated = lab.run(args)[0]
        entry["repropagated"] = str(updated)
        save(manifest, directory)
        print("[audit] Fixed P/R sensitivity sweep (diagnostic only)", flush=True)
        entry["sensitivity"] = str(replay(updated, "weight_audit", [0.1, 1, 10, 1000, 100000], 0.0001))
        save(manifest, directory)
    return manifest, directory


def covariance_rows(directory, sequence, treatment):
    result = []
    store = cv2.FileStorage(str(directory / "eskf.yml.gz"), cv2.FILE_STORAGE_READ)
    records = store.getNode("records")
    for i in range(records.size()):
        n = records.at(i)
        if n.getNode("variant").string() != "R_x1":
            continue
        p = n.getNode("prior").getNode("covariance").mat()
        r = n.getNode("observation_covariance").mat()
        gain = n.getNode("gain").mat()
        s = p[:6, :6] + r
        scale = np.sqrt(np.diag(s))
        normal = s / np.outer(scale, scale)
        expected = np.linalg.solve(normal, (p[:, :6] / scale).T).T / scale
        row = {"sequence": sequence, "treatment": treatment,
               "case": n.getNode("case").string(), "method": n.getNode("method").string(),
               "normalized_S_condition": float(np.linalg.cond(normal)),
               "gain_rotation_trace_mean": float(np.trace(gain[:3, :3]) / 3),
               "gain_position_trace_mean": float(np.trace(gain[3:6, 3:6]) / 3),
               "gain_rotation_error_vs_scaled_reference": float(np.linalg.norm(gain[:3]-expected[:3])),
               "gain_relative_error_vs_scaled_reference": float(np.linalg.norm(gain-expected)/np.linalg.norm(expected)),
               "NIS_actual": n.getNode("nis_regularized").real(),
               "solver": n.getNode("innovation_solver").string() or "legacy_trace_damping",
               "damping": n.getNode("damping").real()}
        for prefix, matrix in (("P", p[:6, :6]), ("R", r)):
            for axis, value in zip(("rx_deg", "ry_deg", "rz_deg", "px_m", "py_m", "pz_m"),
                                   np.sqrt(np.diag(matrix)) * np.array([180/math.pi]*3 + [1]*3)):
                row[prefix + "_sigma_" + axis] = float(value)
        result.append(row)
    store.release()
    return result


def render(manifest, directory):
    rows, covariance = [], []
    for entry in manifest["runs"]:
        reference = Path(entry["reference"])
        if lab.digest(reference / "observations.yml.gz") != entry["observations_sha256"]:
            raise ValueError("Reference observations changed")
        for treatment in ("reference", "solver_only", "repropagated", "sensitivity"):
            if treatment not in entry:
                continue
            path = Path(entry[treatment])
            with (path / "metrics.csv").open() as stream:
                for row in csv.DictReader(stream):
                    row.update(sequence=entry["sequence"], treatment=treatment, directory=str(path))
                    rows.append(row)
            covariance += covariance_rows(path, entry["sequence"], treatment)
    reporting.write_csv(directory / "covariance_audit.csv", covariance)
    reporting.write_csv(directory / "metrics.csv", rows)
    groups = {}
    for row in rows:
        key = tuple(row[k] for k in ("treatment", "stage", "method", "variant"))
        groups.setdefault(key, []).append(row)
    summary = []
    for key, group in sorted(groups.items()):
        valid = [r for r in group if int(r["valid"])]
        record = dict(zip(("treatment", "stage", "method", "variant"), key))
        record.update(valid=len(valid), total=len(group))
        for name in ("prior_trans_error_m", "result_trans_error_m", "prior_rot_error_deg", "result_rot_error_deg"):
            record[name + "_rmse"] = math.sqrt(np.mean([float(r[name])**2 for r in valid])) if valid else math.nan
        summary.append(record)
    reporting.write_csv(directory / "summary.csv", summary)
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    treatments = ["reference", "solver_only", "repropagated"]
    for column, method in enumerate(("small_gicp", "stereo")):
        selected = [next(r for r in summary if r["treatment"] == t and r["stage"] == "eskf"
                         and r["method"] == method and r["variant"] == "R_x1") for t in treatments]
        for line, key in enumerate(("trans_error_m_rmse", "rot_error_deg_rmse")):
            ax = axes[line, column]
            scale = 1000 if line == 0 else 1
            prior = selected[0]["prior_" + key] * scale
            values = [r["result_" + key]*scale for r in selected]
            ax.bar(range(3), values, color=["#a9b2bf", "#e1a240", "#2478aa"])
            ax.axhline(prior, color="black", linestyle="--", label="IMU prior")
            ax.set_xticks(range(3), ["Legacy", "Scaled solve\nsame frozen P", "Scaled solve\ncorrected initial P"])
            ax.set_title(method + (" translation" if line == 0 else " rotation"))
            ax.set_ylabel("conditional RMSE (mm)" if line == 0 else "conditional RMSE (deg)")
            ax.grid(axis="y", alpha=.2); ax.legend()
    fig.savefig(directory / "comparison.png", dpi=150)
    fig.savefig(directory / "comparison.svg"); plt.close(fig)

    fields = ["treatment", "stage", "method", "variant", "valid", "total",
              "prior_trans_error_m_rmse", "result_trans_error_m_rmse",
              "prior_rot_error_deg_rmse", "result_rot_error_deg_rmse"]
    table = "".join("<tr>" + "".join(f"<td>{html.escape(str(row[k]))}</td>" for k in fields) + "</tr>" for row in summary)
    links = "".join(f"<li>{html.escape(entry['sequence'])}: " + " · ".join(
        f"<a href='{html.escape(os.path.relpath(Path(entry[t])/'report.html', directory))}'>{t}</a>"
        for t in treatments + ["sensitivity"] if t in entry) + "</li>" for entry in manifest["runs"])
    page = """<!doctype html><meta charset='utf-8'><title>Fusion covariance audit</title>
<style>body{font:15px system-ui;margin:28px;color:#18304a}table{border-collapse:collapse;font-size:12px}td,th{padding:8px;border:1px solid #dce3eb;white-space:nowrap}th{background:#e7eff8}img{max-width:100%}input{padding:8px}.scroll{overflow:auto}</style>
<h1>融合协方差诊断</h1><p>先固定观测及 P 比较数值求解，再只修正初始 P 的轴索引并重新传播；模型视差沿用原缓存。GICP 核心和观测参数未调节。</p>
<p>敏感性实验使用事先固定的 P×0.0001 与多档 R；不依据真值选择默认值，也不视为噪声标定。GT 仅用于报告。此处为锚点条件帧对误差，不是全序列 ATE。</p>
<p>先验 P 是长时间纯惯性传播的绝对协方差，尚未建模与观测锚点的相关性；初始方差与 Hessian R 仍未标定。数值修正不保证精度改善，较小 NIS 不能单独证明滤波一致。</p>
<p><a href='covariance_audit.csv'>逐轴 P/R、K、NIS 与独立缩放求解对照</a> · <a href='metrics.csv'>全部记录</a> · <a href='summary.csv'>分组结果</a> · <a href='manifest.json'>对照来源与指纹</a></p>"""
    page += "<img src='comparison.png' alt='Controlled comparisons'><ul>" + links + "</ul><input id='q' placeholder='筛选 treatment / method / variant'><div class='scroll'><table><thead><tr>"
    page += "".join(f"<th>{k}</th>" for k in fields) + "</tr></thead><tbody>" + table
    page += "</tbody></table></div><script>document.querySelector('#q').addEventListener('input',e=>document.querySelectorAll('tbody tr').forEach(r=>r.hidden=!r.textContent.toLowerCase().includes(e.target.value.toLowerCase())))</script>"
    (directory / "report.html").write_text(page)
    index = directory.parent / "index.html"
    available = sorted(directory.parent.glob("*/report.html"), reverse=True)
    index.write_text("<!doctype html><meta charset='utf-8'><h1>融合协方差诊断</h1>" + "".join(
        f"<p><a href='{p.parent.name}/report.html'>{p.parent.name}</a></p>" for p in available))
    print(f"Audit report: {directory / 'report.html'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, nargs="+")
    parser.add_argument("--run-controls", action="store_true")
    parser.add_argument("--manifest", type=Path, help="Regenerate a completed audit report only")
    args = parser.parse_args()
    if args.manifest:
        directory = args.manifest.resolve().parent
        manifest = json.loads(args.manifest.read_text())
    elif args.run_controls and args.reference:
        manifest, directory = controls([p.resolve() for p in args.reference])
    else:
        parser.error("Use --reference DIR... --run-controls, or --manifest FILE")
    render(manifest, directory)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Export pairwise fusion diagnostics; groundtruth enters only in this report."""

import argparse
import csv
import html
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

import cv2
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "build/matplotlib-cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml


def mat(node, name):
    child = node.getNode(name)
    return None if child.empty() else child.mat()


def number(node, name):
    child = node.getNode(name)
    return math.nan if child.empty() else child.real()


def rotation_error(a, b):
    return cv2.Rodrigues(a.T @ b)[0].reshape(3)


def quaternion_pose(values):
    tx, ty, tz, x, y, z, w = values
    x, y, z, w = np.array([x, y, z, w]) / np.linalg.norm([x, y, z, w])
    t = np.eye(4)
    t[:3, :3] = [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                 [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                 [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]
    t[:3, 3] = [tx, ty, tz]
    return t


def camera_extrinsic(params):
    r, p, y = params["T_imu_cam"]["rotation"]
    sr, cr, sp, cp, sy, cy = math.sin(r), math.cos(r), math.sin(p), math.cos(p), math.sin(y), math.cos(y)
    t = np.eye(4)
    t[:3, :3] = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]) @ np.array(
        [[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]]) @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    t[:3, 3] = params["T_imu_cam"]["translation"]
    return t


def covariance_stats(covariance):
    if covariance is None or not np.isfinite(covariance).all():
        return [math.nan] * 4
    eigenvalues = np.linalg.eigvalsh((covariance + covariance.T) / 2)
    return [float(eigenvalues[0]), float(eigenvalues[-1]),
            float(np.trace(covariance[:3, :3])), float(np.trace(covariance[3:6, 3:6]))]


# Pairs whose pure-IMU prior is already this far off are reported separately: they
# measure recovery from long dead reckoning rather than the steady-state frontend.
FLAG_PRIOR_TRANSLATION_M = 0.05


def whitened_per_dof(error, covariance):
    """e^T C^-1 e / dim. Close to 1 on average when C describes the actual error."""
    if covariance is None or not np.isfinite(covariance).all() or not np.isfinite(error).all():
        return math.nan
    try:
        if np.linalg.eigvalsh((covariance + covariance.T) / 2).min() <= 0:
            return math.nan
        return float(error @ np.linalg.solve(covariance, error)) / len(error)
    except np.linalg.LinAlgError:
        return math.nan


def block_whitened(error, covariance):
    if covariance is None:
        return math.nan, math.nan
    return (whitened_per_dof(error[:3], covariance[:3, :3]),
            whitened_per_dof(error[3:], covariance[3:6, 3:6]))


def flatten(prefix, matrix):
    if matrix is None:
        return {}
    return {f"{prefix}_{r}{c}": float(matrix[r, c])
            for r in range(matrix.shape[0]) for c in range(matrix.shape[1])}


def write_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


def collect(directory, config):
    gt = {}
    for line in Path(config["groundtruth_file"]).read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        fields = [float(v) for v in line.split()]
        gt[round(fields[0] * 1000)] = quaternion_pose(fields[1:])
    params = yaml.safe_load(Path(config["params_file"]).read_text())
    tbc = camera_extrinsic(params)
    rows, matrices, iterations, gicp_iterations = [], [], [], []
    for stage, file in (("gicp", "observations.yml.gz"), ("eskf", "eskf.yml.gz"),
                        ("esikf", "esikf.yml.gz")):
        path = directory / file
        if not path.is_file():
            continue
        storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
        records = storage.getNode("records")
        for index in range(records.size()):
            n = records.at(index)
            anchor = mat(n.getNode("anchor"), "pose")
            prior = mat(n.getNode("prior"), "pose")
            result = mat(n.getNode("result"), "pose")
            # Conditional anchor evaluation: use estimated anchor only as the common
            # world origin; GT relative motion never enters prediction or updates.
            a = gt[round(number(n, "anchor_time") * 1000)]
            b = gt[round(number(n, "current_time") * 1000)]
            truth = anchor @ tbc @ np.linalg.inv(a) @ b @ np.linalg.inv(tbc)
            before = np.r_[rotation_error(prior[:3, :3], truth[:3, :3]), truth[:3, 3] - prior[:3, 3]]
            after = np.r_[rotation_error(result[:3, :3], truth[:3, :3]), truth[:3, 3] - result[:3, 3]]
            # Static pairs move by zero, so the estimated anchor is a reference that needs
            # no groundtruth; the ESIKF observation scale is calibrated on these columns.
            zero_motion = np.r_[rotation_error(result[:3, :3], anchor[:3, :3]),
                                anchor[:3, 3] - result[:3, 3]]
            p0 = mat(n.getNode("prior"), "covariance")
            p1 = mat(n.getNode("result"), "covariance")
            noise = mat(n, "observation_covariance")
            observed = stage in ("gicp", "esikf")
            result_by_r = block_whitened(after, noise) if observed else (math.nan, math.nan)
            zero_by_r = block_whitened(zero_motion, noise) if observed else (math.nan, math.nan)
            result_by_p = block_whitened(after, p1[:6, :6]) if stage != "gicp" else (math.nan, math.nan)
            eig_min, eig_max, rot_trace, pos_trace = covariance_stats(noise)
            covariance = noise if stage == "gicp" else p1[:6, :6]
            conditional_mahalanobis = math.nan
            if np.isfinite(covariance).all() and np.linalg.eigvalsh(covariance).min() > 0:
                conditional_mahalanobis = float(after @ np.linalg.solve(covariance, after))
            row = {"stage": stage, "case": n.getNode("case").string(),
                   "method": n.getNode("method").string(), "variant": n.getNode("variant").string(),
                   "status": n.getNode("status").string(), "valid": int(number(n, "valid")),
                   "solver_status": n.getNode("solver_status").string(),
                   "covariance_convention": n.getNode("covariance_convention").string(),
                   "fitness": number(n, "fitness"),
                   "registration_rmse_m": number(n, "registration_rmse"),
                   "anchor_id": int(number(n, "anchor_id")), "current_id": int(number(n, "current_id")),
                   "dt_s": number(n, "current_time") - number(n, "anchor_time"),
                   "converged": int(number(n, "converged")), "inliers": int(number(n, "inliers")),
                   "iterations": int(number(n, "iterations")), "elapsed_ms": number(n, "elapsed_ms"),
                   "covariance_from_hessian": int(number(n, "covariance_from_hessian")),
                   "prior_trans_error_m": float(np.linalg.norm(before[3:])),
                   "prior_rot_error_deg": math.degrees(np.linalg.norm(before[:3])),
                   "result_trans_error_m": float(np.linalg.norm(after[3:])),
                   "result_rot_error_deg": math.degrees(np.linalg.norm(after[:3])),
                   "correction_trans_m": float(np.linalg.norm(result[:3, 3] - prior[:3, 3])),
                   "correction_rot_deg": math.degrees(np.linalg.norm(rotation_error(prior[:3, :3], result[:3, :3]))),
                   "R_eig_min": eig_min, "R_eig_max": eig_max,
                   "R_condition_mixed_units": eig_max/eig_min if eig_min > 0 else math.nan,
                   "R_rot_trace_rad2": rot_trace, "R_pos_trace_m2": pos_trace,
                   "P_prior_pose_trace": float(np.trace(p0[:6, :6])),
                   "P_result_pose_trace": float(np.trace(p1[:6, :6])) if stage != "gicp" else math.nan,
                   "P_result_min_eig": float(np.linalg.eigvalsh((p1+p1.T)/2).min()) if stage != "gicp" else math.nan,
                   "P_result_asymmetry": float(np.max(np.abs(p1-p1.T))) if stage != "gicp" else math.nan,
                   "nis_regularized": number(n, "nis_regularized") if stage == "eskf" and number(n, "valid") else math.nan,
                   "conditional_pose_mahalanobis": conditional_mahalanobis,
                   "prior_flagged": int(float(np.linalg.norm(before[3:])) > FLAG_PRIOR_TRANSLATION_M),
                   "zero_motion_trans_m": float(np.linalg.norm(zero_motion[3:])),
                   "zero_motion_rot_deg": math.degrees(np.linalg.norm(zero_motion[:3])),
                   "result_whitened_by_R_rot": result_by_r[0],
                   "result_whitened_by_R_trans": result_by_r[1],
                   "zero_motion_whitened_by_R_rot": zero_by_r[0],
                   "zero_motion_whitened_by_R_trans": zero_by_r[1],
                   "posterior_whitened_by_P_rot": result_by_p[0],
                   "posterior_whitened_by_P_trans": result_by_p[1]}
            rows.append(row)
            matrices.append({k: row[k] for k in ("stage", "case", "method", "variant")} |
                flatten("R", noise) | flatten("P_prior_pose", p0[:6, :6]) |
                (flatten("P_result_pose", p1[:6, :6]) if stage != "gicp" else {}) |
                flatten("K_pose", mat(n, "gain")[:6, :] if mat(n, "gain") is not None else None))
            items = n.getNode("gicp_iterations")
            for j in range(items.size()):
                it = items.at(j)
                gicp_iterations.append({"stage": stage, "case": row["case"],
                    "method": row["method"], "variant": row["variant"]} |
                    {key: number(it, key) for key in (
                        "iteration", "inliers", "fitness", "mahalanobis_cost", "rmse",
                        "drot_rad", "dtrans_m", "accepted", "normalized_hessian_min",
                        "normalized_hessian_condition", "observable_dofs")})
            items = n.getNode("esikf_iterations")
            for j in range(items.size()):
                it = items.at(j)
                information, r_cov = mat(it, "information"), mat(it, "R")
                correction = mat(it, "pose_correction").reshape(-1)
                offset = mat(it, "prior_offset")
                offset = offset.reshape(-1) if offset is not None else np.full(6, math.nan)
                iterations.append({"case": row["case"], "variant": row["variant"],
                    "iteration": int(number(it, "iteration")),
                    "correspondences": int(number(it, "correspondences")),
                    "virtual_residual_norm": float(np.linalg.norm(mat(it, "virtual_residual"))),
                    "correction_rot_deg": math.degrees(np.linalg.norm(correction[:3])),
                    "correction_trans_m": float(np.linalg.norm(correction[3:])),
                    "prior_offset_rot_deg": math.degrees(np.linalg.norm(offset[:3])),
                    "prior_offset_trans_m": float(np.linalg.norm(offset[3:])),
                    "information_eig_min": float(np.linalg.eigvalsh(information).min())} |
                    flatten("H", information) | flatten("R", r_cov) | flatten("K_pose", mat(it, "pose_gain")))
        storage.release()
    return rows, matrices, iterations, gicp_iterations


def summarize(rows):
    groups = sorted({(r["stage"], r["method"], r["variant"]) for r in rows})
    summary = []
    for stage, method, variant in groups:
        selected = [r for r in rows if (r["stage"], r["method"], r["variant"]) == (stage, method, variant)]
        valid = [r for r in selected if r["valid"]]
        unflagged = [r for r in valid if not r["prior_flagged"]]
        def rmse(key, rows=valid):
            return math.sqrt(sum(r[key]**2 for r in rows)/len(rows)) if rows else None
        def median(key, rows=valid):
            values = [r[key] for r in rows if math.isfinite(r[key])]
            return statistics.median(values) if values else None
        def mean(key):
            values = [r[key] for r in valid if math.isfinite(r[key])]
            return sum(values) / len(values) if values else None
        summary.append({"stage": stage, "method": method, "variant": variant,
                        "valid": len(valid), "total": len(selected),
                        "prior_translation_rmse_m": rmse("prior_trans_error_m"),
                        "result_translation_rmse_m": rmse("result_trans_error_m"),
                        "prior_rotation_rmse_deg": rmse("prior_rot_error_deg"),
                        "result_rotation_rmse_deg": rmse("result_rot_error_deg"),
                        "prior_translation_median_m": median("prior_trans_error_m"),
                        "result_translation_median_m": median("result_trans_error_m"),
                        "prior_rotation_median_deg": median("prior_rot_error_deg"),
                        "result_rotation_median_deg": median("result_rot_error_deg"),
                        "unflagged": len(unflagged),
                        "result_translation_rmse_unflagged_m": rmse("result_trans_error_m", unflagged),
                        "result_rotation_rmse_unflagged_deg": rmse("result_rot_error_deg", unflagged),
                        "improved_translation": sum(r["result_trans_error_m"] < r["prior_trans_error_m"] for r in valid),
                        "improved_rotation": sum(r["result_rot_error_deg"] < r["prior_rot_error_deg"] for r in valid),
                        "result_whitened_by_R_rot_mean": mean("result_whitened_by_R_rot"),
                        "result_whitened_by_R_trans_mean": mean("result_whitened_by_R_trans"),
                        "posterior_whitened_by_P_rot_mean": mean("posterior_whitened_by_P_rot"),
                        "posterior_whitened_by_P_trans_mean": mean("posterior_whitened_by_P_trans")})
    return summary


def plots(directory, rows):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    groups = summarize(rows)
    for column, stage in enumerate(("gicp", "eskf", "esikf")):
        selected = [r for r in groups if r["stage"] == stage]
        for line, quantity in enumerate(("translation_rmse_m", "rotation_rmse_deg")):
            ax = axes[line, column]
            scale = 1000 if line == 0 else 1
            x = np.arange(len(selected))
            ax.bar(x - .2, [(r["prior_"+quantity] if r["prior_"+quantity] is not None else math.nan)*scale for r in selected], .4, label="IMU prior", color="#aab5c2")
            ax.bar(x + .2, [(r["result_"+quantity] if r["result_"+quantity] is not None else math.nan)*scale for r in selected],
                   .4, label="Result (valid)", color="#2375b8")
            for i, r in enumerate(selected):
                if not r["valid"]:
                    ax.axvspan(i-.45, i+.45, color="red", alpha=.12)
            ax.set_title(stage.upper() + (" translation" if line == 0 else " rotation"))
            ax.set_ylabel("mm" if line == 0 else "degrees")
            ax.set_xticks(x, [f"{r['method']} / {r['variant']}\nn={r['valid']}/{r['total']}" for r in selected],
                          rotation=45, ha="right", fontsize=8)
            ax.grid(axis="y", alpha=.2)
            if selected: ax.legend(fontsize=8)
    fig.suptitle("Frozen frame pairs: conditional error RMSE (not a trajectory ATE)")
    fig.savefig(directory / "errors.png", dpi=160)
    fig.savefig(directory / "errors.svg")
    plt.close(fig)


def html_report(directory, cfg, rows, summary):
    keys = ["stage", "case", "method", "variant", "status", "solver_status", "inliers", "converged",
            "covariance_convention", "covariance_from_hessian", "fitness", "registration_rmse_m",
            "prior_trans_error_m", "result_trans_error_m", "prior_rot_error_deg",
            "result_rot_error_deg", "prior_flagged", "R_rot_trace_rad2", "R_pos_trace_m2",
            "nis_regularized", "posterior_whitened_by_P_rot", "posterior_whitened_by_P_trans"]
    def value(v):
        return f"{v:.5g}" if isinstance(v, float) else html.escape(str(v))
    headers = "".join(f"<th>{html.escape(k)}</th>" for k in keys)
    body = "".join(f"<tr class='{'valid' if r['valid'] else 'failed'}' data-stage='{r['stage']}'>" +
                   "".join(f"<td>{value(r[k])}</td>" for k in keys) + "</tr>" for r in rows)
    summary_keys = ("stage", "method", "variant", "valid", "total", "result_translation_rmse_m",
                    "result_rotation_rmse_deg", "result_translation_median_m", "result_rotation_median_deg",
                    "result_translation_rmse_unflagged_m", "result_rotation_rmse_unflagged_deg",
                    "improved_translation", "improved_rotation",
                    "posterior_whitened_by_P_rot_mean", "posterior_whitened_by_P_trans_mean")
    summary_body = "".join("<tr>" + "".join(f"<td>{value(r[k])}</td>" for k in summary_keys) +
        "</tr>" for r in summary)
    page = """<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>Fusion Lab</title><style>
body{font:15px system-ui;margin:28px;color:#182f47;background:#f6f8fb}h1{margin-bottom:6px}
table{border-collapse:collapse;background:white;font-size:12px}th,td{padding:7px;border:1px solid #dce3eb;white-space:nowrap}
th{position:sticky;top:0;background:#e7eff8}tr.failed{background:#ffe6e6}.scroll{overflow:auto;max-height:600px}
img{width:100%;background:white}input,select{padding:8px;margin:10px 8px 10px 0}code{background:#e7eff8;padding:2px}
</style><h1>前端融合独立实验</h1>"""
    page += f"<p>{html.escape(cfg['sequence'])} · depth={html.escape(cfg['depth']['provider'])} · cache={cfg['cache_key']}</p>"
    if cfg["depth"].get("model_plan_sha256"):
        page += (f"<p>模型视差：<code>{html.escape(cfg['depth']['model_plan'])}</code>；"
                 f"engine SHA256=<code>{html.escape(cfg['depth']['model_plan_sha256'])}</code>。"
                 "输入为生产裁剪与光照处理后的 RGB 0..255；此实验使用原始浮点像素视差，不应用完整前端的邻域视差后处理。</p>")
        origin = cfg["depth"].get("model_origin", {})
        if origin.get("onnx_sha256"):
            page += f"<p>原始 ONNX SHA256：<code>{html.escape(origin['onnx_sha256'])}</code>。</p>"
        if origin.get("graph_validation"):
            comparison = origin["graph_validation"]["validation"]
            page += (f"<p>代价体图改写验证：ONNX Runtime 固定随机特征输入，最大绝对误差 "
                     f"{comparison['max_abs_error']:.6g}；该项检查图改写，完整模型精度仍需由录制数据评估。</p>")
    version = cfg.get("registration_upstream", {}).get("commit", "pre-import")
    observation_version = cfg.get("observation_registration_upstream", {}).get("commit", version)
    page += f"<p>配准代码版本：<code>{html.escape(version)}</code>；观测生成版本：<code>{html.escape(observation_version)}</code>。</p>"
    page += "<p>固定图像对与纯 IMU 先验；ESKF 回放固定 GICP 观测，ESIKF 直接读取点云。真值只用于此报告。</p>"
    page += "<p>同时查看 converged 与 solver_status：上游默认 Stereo 在 max_iter / stopped_at_minimum 时也可能置 converged=true。valid 表示输入检查通过，不表示精度提升。欧氏 GICP 未提供独立的点距 RMSE，该列显示 NaN。</p>"
    if cfg["depth"]["provider"] == "sgbm":
        page += "<p><b>SGBM 调试基线：</b>结果用于检查接口、坐标与融合行为，模型视差替换后需重新评估 GICP 精度和协方差。</p>"
    page += "<p>误差按每对图像的锚点计算，单位 m / degrees。此处 RMSE 不是全序列 ATE；末帧受同一锚点条件约束。NIS 与条件马氏距离是诊断量，少量相关图像对不能用于宣称协方差已校准。</p>"
    page += "<p><a href='metrics.csv'>全部指标 CSV</a> · <a href='covariances.csv'>R / P / K 矩阵 CSV</a> · <a href='gicp_iterations.csv'>Stereo 求解逐轮 CSV</a> · <a href='esikf_iterations.csv'>ESIKF 逐轮 CSV</a> · <a href='resolved.yaml'>配置快照</a> · <a href='errors.svg'>矢量图</a></p>"
    page += (f"<p>unflagged 排除纯惯性先验平移误差超过 {FLAG_PRIOR_TRANSLATION_M * 1000:.0f} mm 的图像对；"
             "whitened 为误差按协方差白化后的每自由度均值，协方差与实际误差相符时约为 1。</p>")
    page += ("<h2>分组结果</h2><div class='scroll'><table><thead><tr>" +
             "".join(f"<th>{html.escape(k)}</th>" for k in summary_keys) +
             "</tr></thead><tbody>" + summary_body + "</tbody></table></div>")
    page += "<img src='errors.png' alt='Pairwise errors'><h2>逐对检查</h2><input id='search' placeholder='筛选 case、method、variant'><select id='stage'><option value=''>全部模块</option><option>gicp</option><option>eskf</option><option>esikf</option></select><label><input id='failed' type='checkbox'>只看失败</label>"
    page += "<div class='scroll'><table id='metrics'><thead><tr>" + headers + "</tr></thead><tbody>" + body + "</tbody></table></div>"
    page += """<script>
function filter(){const q=document.querySelector('#search').value.toLowerCase(),s=document.querySelector('#stage').value,f=document.querySelector('#failed').checked;
document.querySelectorAll('#metrics tbody tr').forEach(r=>{r.hidden=!(r.textContent.toLowerCase().includes(q)&&(!s||r.dataset.stage===s)&&(!f||r.classList.contains('failed')))});}
document.querySelectorAll('input,select').forEach(e=>e.addEventListener('input',filter));</script></html>"""
    (directory / "report.html").write_text(page)


def report_index(directory):
    root = Path(__file__).resolve().parents[1] / "artifacts/fusion"
    if directory.parent != root:
        return
    entries = []
    for path in sorted(root.glob("*/summary.json"), reverse=True):
        config_file = path.parent / "resolved.yaml"
        if not config_file.is_file() or not (path.parent / "report.html").is_file():
            continue
        cfg = yaml.safe_load(config_file.read_text())
        summary = json.loads(path.read_text())
        total = sum(r["total"] for r in summary)
        valid = sum(r["valid"] for r in summary)
        label = html.escape(path.parent.name)
        sequence = html.escape(cfg["sequence"])
        provider = "TensorRT model cache" if cfg["depth"].get("model_plan_sha256") else html.escape(cfg["depth"]["provider"])
        stages = ", ".join(sorted({r["stage"] for r in summary}))
        version = html.escape(cfg.get("registration_upstream", {}).get("commit", "pre-import")[:12])
        observation_version = html.escape(cfg.get("observation_registration_upstream", {}).get(
            "commit", cfg.get("registration_upstream", {}).get("commit", "pre-import"))[:12])
        entries.append(f"<tr><td><a href='{label}/report.html'>{label}</a></td>"
                       f"<td>{sequence}</td><td>{stages}</td><td>{provider}</td>"
                       f"<td>{version}</td><td>{observation_version}</td><td>{valid}/{total}</td></tr>")
    page = """<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>Fusion Lab reports</title>
<style>body{font:15px system-ui;margin:28px;color:#182f47}table{border-collapse:collapse}td,th{border:1px solid #dce3eb;padding:9px}th{background:#e7eff8}input{padding:9px;width:380px}</style>
<h1>前端融合实验报告</h1><p>按实验目录保留配置与结果；数值更新有效不代表精度提升。SGBM 为调试基线。</p>
<p><input id='q' placeholder='筛选序列、模块、版本或实验时间'></p><table><thead><tr><th>实验 / 报告</th><th>序列</th><th>模块</th><th>视差来源</th><th>代码版本</th><th>观测版本</th><th>有效更新 / 总记录</th></tr></thead><tbody>"""
    page += "".join(entries)
    page += """</tbody></table><script>document.querySelector('#q').addEventListener('input',e=>{document.querySelectorAll('tbody tr').forEach(r=>{r.hidden=!r.textContent.toLowerCase().includes(e.target.value.toLowerCase())})})</script></html>"""
    (root / "index.html").write_text(page)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    directory = args.directory.resolve()
    cfg = yaml.safe_load((directory / "resolved.yaml").read_text())
    for key, checksum_key in (("groundtruth_file", "groundtruth_sha256"), ("params_file", "params_sha256")):
        if cfg.get(checksum_key) and hashlib.sha256(Path(cfg[key]).read_bytes()).hexdigest() != cfg[checksum_key]:
            parser.error(f"Recorded input changed: {cfg[key]}")
    rows, matrices, iterations, gicp_iterations = collect(directory, cfg)
    if not rows:
        parser.error("No experiment result files found")
    write_csv(directory / "metrics.csv", rows)
    write_csv(directory / "covariances.csv", matrices)
    write_csv(directory / "esikf_iterations.csv", iterations)
    write_csv(directory / "gicp_iterations.csv", gicp_iterations)
    summary = summarize(rows)
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plots(directory, rows)
    html_report(directory, cfg, rows, summary)
    report_index(directory)
    print(f"Report generated: {len(rows)} records, {len(gicp_iterations)} Stereo / {len(iterations)} ESIKF iterations; "
          f"{sum(r['valid'] for r in rows)}/{len(rows)} eligible/updated")


if __name__ == "__main__":
    main()

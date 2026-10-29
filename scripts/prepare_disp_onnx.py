#!/usr/bin/env python3
"""Replace this export's scattered disparity planes with equivalent Pad/Concat.

Original weights and feature/correlation computations are preserved. This script
requires ONNX and ONNX Runtime in build/venv/model-tools; no network access.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
BOUNDARY = {"/Slice_1_output_0": [1, 224, 112, 224],
            "/Slice_5_output_0": [1, 224, 112, 224],
            "/proj_cmb/Conv_output_0": [1, 12, 112, 224],
            "/proj_cmb_1/Conv_output_0": [1, 12, 112, 224]}
JOIN = "/Concat_835_output_0"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reachable(model, outputs, boundary=()):
    by_output = {o: n for n in model.graph.node for o in n.output}
    initializers = {t.name for t in model.graph.initializer}
    needed, leaves, pending = set(), set(), list(outputs)
    while pending:
        name = pending.pop()
        if not name:
            continue
        if name in boundary or name in initializers or name not in by_output:
            leaves.add(name)
            continue
        node = by_output[name]
        if node.output[0] in needed:
            continue
        needed.add(node.output[0])
        pending.extend(node.input)
    return needed, leaves


def cost_graph(model, name):
    needed, leaves = reachable(model, [JOIN], BOUNDARY)
    if leaves != set(BOUNDARY):
        raise ValueError(f"Unexpected cost-volume boundary: {leaves}")
    graph = helper.make_graph([n for n in model.graph.node if n.output[0] in needed], name,
        [helper.make_tensor_value_info(k, TensorProto.FLOAT16, v) for k, v in BOUNDARY.items()],
        [helper.make_tensor_value_info(JOIN, TensorProto.FLOAT16, [1, 32, 104, 112, 224])])
    result = helper.make_model(graph, opset_imports=list(model.opset_import))
    result.ir_version = model.ir_version
    onnx.checker.check_model(result)
    return result


def rewrite(model):
    nodes = list(model.graph.node)
    by_output = {o: n for n in nodes for o in n.output}
    scatter = [n for n in nodes if n.op_type == "ScatterND"]
    if len(scatter) != 312 or by_output[JOIN].op_type != "Concat":
        raise ValueError("This script supports only the supplied 104-plane stereo export")
    if list(by_output[JOIN].input) != [scatter[103].output[0], scatter[311].output[0]]:
        raise ValueError("Unexpected cost-volume join")

    def update_source(node):
        value = node.input[2]
        while value in by_output and by_output[value].op_type in ("Cast", "Reshape", "Expand"):
            value = by_output[value].input[0]
        return value

    replacement = []
    prefix = "sio_cost_volume/"
    def constant(name, array):
        name = prefix + name
        replacement.append(helper.make_node("Constant", [], [name], name=name,
                            value=numpy_helper.from_array(np.asarray(array, dtype=np.int64))))
        return name
    axis = constant("disparity_axis", [2])

    def plane(value, d, label, pad):
        if pad and d:
            pads = constant(label + "/pads", [0, 0, 0, d, 0, 0, 0, 0])
            padded = prefix + label + "/padded"
            replacement.append(helper.make_node("Pad", [value, pads], [padded], name=padded))
            value = padded
        output = prefix + label + "/plane"
        replacement.append(helper.make_node("Unsqueeze", [value, axis], [output], name=output))
        return output

    gwc, concat = [], []
    for d in range(104):
        expected = "/ReduceSum" + (f"_{d}" if d else "") + "_output_0"
        value = update_source(scatter[d])
        if value != expected:
            raise ValueError(f"Unexpected group correlation at disparity {d}: {value}")
        # The original export computes group correlation in FP32, then casts
        # each update into its FP16 volume. Keep that exact precision boundary.
        cast = prefix + f"gwc_{d}/half"
        replacement.append(helper.make_node("Cast", [value], [cast], name=cast,
                                             to=TensorProto.FLOAT16))
        value = cast
        gwc.append(plane(value, d, f"gwc_{d}", True))
        left = update_source(scatter[104 + 2*d])
        right = update_source(scatter[105 + 2*d])
        if left != "/proj_cmb/Conv_output_0":
            raise ValueError("This export must repeat the full left feature plane")
        # Preserve this model's full left plane, including columns < disparity.
        # The right plane alone is shifted and padded; this differs from some
        # other implementations that zero both sides in the invalid region.
        left_plane = plane(left, d, f"left_{d}", False)
        right_plane = plane(right, d, f"right_{d}", True)
        combined = prefix + f"concat_{d}"
        replacement.append(helper.make_node("Concat", [left_plane, right_plane],
                                             [combined], axis=1, name=combined))
        concat.append(combined)
    gwc_out, concat_out = prefix + "gwc", prefix + "concat"
    replacement.extend([helper.make_node("Concat", gwc, [gwc_out], axis=2, name=gwc_out),
                        helper.make_node("Concat", concat, [concat_out], axis=2, name=concat_out),
                        helper.make_node("Concat", [gwc_out, concat_out], [JOIN], axis=1,
                                         name=prefix + "join")])
    result = copy.deepcopy(model)
    expanded = []
    for n in nodes:
        expanded.extend(replacement if JOIN in n.output else [n])
    del result.graph.node[:]
    result.graph.node.extend(expanded)
    needed, leaves = reachable(result, [o.name for o in result.graph.output])
    kept = [n for n in result.graph.node if n.output[0] in needed]
    del result.graph.node[:]
    result.graph.node.extend(kept)
    initializers = [t for t in result.graph.initializer if t.name in leaves]
    del result.graph.initializer[:]
    result.graph.initializer.extend(initializers)
    # This machine's engine and the independent verification both use batch=1.
    for tensor in list(result.graph.input) + list(result.graph.output):
        tensor.type.tensor_type.shape.dim[0].dim_value = 1
    return result


def verify(original, candidate, directory):
    directory.mkdir(parents=True, exist_ok=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.enable_cpu_mem_arena = False
    rng = np.random.default_rng(20261029)
    feed = {k: rng.standard_normal(v).astype(np.float16) for k, v in BOUNDARY.items()}
    results = []
    for model, label in ((original, "original"), (candidate, "pad_concat")):
        file = directory / (label + ".onnx")
        onnx.save(cost_graph(model, label), str(file))
        print(f"Checking {label} cost volume against fixed random feature inputs", flush=True)
        session = ort.InferenceSession(str(file), sess_options=options,
                                       providers=["CPUExecutionProvider"])
        value = session.run([JOIN], feed)[0]
        if value.shape != (1, 32, 104, 112, 224) or not np.isfinite(value).all():
            raise ValueError("Invalid reference cost volume")
        results.append(value)
        del session
    error = float(np.max(np.abs(results[0] - results[1])))
    np.testing.assert_allclose(results[0], results[1], rtol=1e-6, atol=1e-6)
    return {"provider": "ONNX Runtime CPU", "seed": 20261029,
            "output_shape": list(results[0].shape), "max_abs_error": error,
            "rtol": 1e-6, "atol": 1e-6}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "models/disp.onnx")
    parser.add_argument("--output", type=Path, default=ROOT / "models/onnx/disp_cost_volume.onnx")
    args = parser.parse_args()
    checksum = digest(args.source)
    script_checksum = digest(Path(__file__))
    report_path = args.output.with_suffix(".verified.json")
    if args.output.is_file() and report_path.is_file():
        report = json.loads(report_path.read_text())
        if (report["source_sha256"] == checksum and report["output_sha256"] == digest(args.output)
                and report.get("script_sha256") == script_checksum):
            print(f"Verified cost-volume model already exists: {args.output}")
            return
    original = onnx.load(str(args.source))
    candidate = rewrite(original)
    validation = verify(original, candidate, ROOT / "build/model-cost-validation")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(candidate, str(args.output))
    onnx.checker.check_model(str(args.output))
    report = {"source": str(args.source), "source_sha256": checksum,
              "script_sha256": script_checksum,
              "output": str(args.output), "output_sha256": digest(args.output),
              "original_nodes": len(original.graph.node), "derived_nodes": len(candidate.graph.node),
              "transformation": "ScatterND disparity planes to Pad/Concat; original update tensors preserved",
              "validation": validation}
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

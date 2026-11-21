"""Phase 2.2: contextual mixture gates and learned peer-embedding feedback.

Only train/validation targets are accepted by fit(). Test labels are used by
evaluate() after the router is frozen. This is a deterministic, stdlib-only
research experiment, NOT online neural-weight training or live AR perception.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from neural_mesh import mesh, phase2

SCHEMA = "neural-mesh-phase-2.2"
FEATURE_DIM = 5
FEEDBACK_DIM = 17
NODE_COUNT = len(mesh.NODES)


def _finite_vector(value, dim):
    if (not isinstance(value, list) or len(value) != dim or
            any(not isinstance(x, (int, float)) or not math.isfinite(x) for x in value)):
        raise ValueError(f"expected {dim} finite numeric features")
    return value


def _check_digest(receipt, key, expected):
    if receipt.get(key) != expected:
        raise ValueError(f"incorrect {key} in worker receipt")


def observe(node, public, weights):
    """Specialist computes its own embeddings and forecasts without labels."""
    result = phase2.infer(node, public, weights)
    emb = {}
    for split in phase2.SPLITS:
        emb[split] = [{"id": item["id"],
                       "value": _finite_vector(mesh.features(item["tracks"][node], node), 9)}
                      for item in public["scenes"][split]]
    result["embeddings"] = emb
    return result


def collect(public, weights, receipts):
    """Verify six independent receipts, including numerical embeddings."""
    rows = phase2.collect(public, weights, receipts)
    files = list(Path(receipts).glob("node-*.json"))
    by_node = {mesh.load(f)["node"]: mesh.load(f) for f in files}
    for split in phase2.SPLITS:
        for i, record in enumerate(rows["batches"][split]):
            embedding = {}
            for node in mesh.NODES:
                packet = by_node[node]
                items = packet.get("embeddings", {}).get(split, [])
                if len(items) != len(rows["batches"][split]) or items[i].get("id") != record["id"]:
                    raise ValueError("missing, wrong-scene or reordered specialist embedding")
                actual = _finite_vector(items[i]["value"], 9)
                expected = mesh.features(public["scenes"][split][i]["tracks"][node], node)
                if any(abs(a-b) > 1e-12 for a, b in zip(actual, expected)):
                    raise ValueError("specialist embedding does not match observation")
                embedding[node] = actual
            record["embeddings"] = embedding
    return rows


def local_features(row):
    """Five real-valued confidence/context signals per receiving node."""
    pred = row["predictions"]
    c = [sum(pred[n][k] for n in mesh.NODES)/NODE_COUNT for k in (0, 1)]
    features = {}
    for node in mesh.NODES:
        z = _finite_vector(row["embeddings"][node], 9)
        a, b, scale = mesh.MASKS[node]
        velocity = math.hypot(z[0]/a, z[1]/a)
        accel = math.hypot(z[2]/b, z[3]/b)
        discrepancy = math.sqrt(mesh.squared_error(pred[node], c)*2)
        speed_inconsistency = math.hypot(z[0]/a-z[4]/scale, z[1]/a-z[5]/scale)
        features[node] = [1., min(3., velocity)/2., min(1., accel)*2.,
                          min(2., discrepancy), min(2., speed_inconsistency)]
    return features


def gate(row, model):
    """Input-dependent row-normalized attention; no test labels are used."""
    base = _finite_vector(model["base"], NODE_COUNT)
    theta = model["theta"]
    f = local_features(row)
    logits = [math.log(max(1e-9, base[i])) +
              sum(a*b for a, b in zip(_finite_vector(theta[i], FEATURE_DIM), f[node]))
              for i, node in enumerate(mesh.NODES)]
    high = max(logits)
    exp = [math.exp(max(-40., x-high)) for x in logits]
    total = sum(exp)
    return {node: exp[i]/total for i, node in enumerate(mesh.NODES)}


def blend(row, attention):
    return [sum(attention[node]*row["predictions"][node][k] for node in mesh.NODES)
            for k in (0, 1)]


def fit_gates(rows, labels, base, regularization, epochs=170):
    """Deterministic full-batch gradient descent through a softmax gate."""
    theta = [[0.]*FEATURE_DIM for _ in mesh.NODES]
    prepared = [(row, local_features(row), labels[row["id"]]) for row in rows]
    n = len(prepared)
    for epoch in range(epochs):
        grad = [[0.]*FEATURE_DIM for _ in mesh.NODES]
        for row, features, truth in prepared:
            model = {"base": base, "theta": theta}
            att = gate(row, model)
            c = blend(row, att)
            for i, node in enumerate(mesh.NODES):
                # derivative of 0.5*||c-truth||^2 through softmax
                diff = sum((c[k]-truth[k])*(row["predictions"][node][k]-c[k])
                           for k in (0, 1))
                dlogit = att[node]*diff/n
                for j in range(FEATURE_DIM):
                    grad[i][j] += dlogit*features[node][j]
        lr = 2.0/(1. + epoch/250)
        for i in range(NODE_COUNT):
            for j in range(FEATURE_DIM):
                theta[i][j] -= lr*(grad[i][j] + regularization*theta[i][j])
                theta[i][j] = max(-4., min(4., theta[i][j]))
    return {"base": list(base), "theta": theta, "regularization": regularization}


def _mean(values):
    return sum(values)/len(values)


def _error_rows(rows, labels, predict):
    return _mean([mesh.squared_error(predict(row), labels[row["id"]]) for row in rows])


def feedback_features(row, attention):
    """Compact aggregate of all six specialists' observation embeddings.

    The current receiver does not own these other five input embeddings.
    They must be transmitted from other runner jobs for recurrent correction.
    """
    all_z = {}
    for node in mesh.NODES:
        z = _finite_vector(row["embeddings"][node], 9)
        a, b, c = mesh.MASKS[node]
        all_z[node] = [z[0]/a, z[1]/a, z[2]/b, z[3]/b,
                       z[4]/c, z[5]/c, z[6], z[7]]
    weighted = [sum(attention[n]*all_z[n][j] for n in mesh.NODES) for j in range(8)]
    uniform = [sum(all_z[n][j] for n in mesh.NODES)/NODE_COUNT for j in range(8)]
    return [1.] + weighted + uniform


def _solve(a, b):
    """Small ridge linear-system solver with pivoting and no dependencies."""
    n = len(b)
    arr = [list(a[i]) + [b[i]] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(arr[r][col]))
        if abs(arr[pivot][col]) < 1e-12:
            raise ValueError("unstable ridge system")
        arr[pivot], arr[col] = arr[col], arr[pivot]
        norm = arr[col][col]
        for j in range(col, n+1):
            arr[col][j] /= norm
        for row in range(n):
            if row == col:
                continue
            factor = arr[row][col]
            for j in range(col, n+1):
                arr[row][j] -= factor*arr[col][j]
    return [arr[i][n] for i in range(n)]


def fit_feedback(rows, labels, gate_model, ridge):
    """Learn a linear residual head from peer embeddings on training only."""
    xtx = [[0.]*FEEDBACK_DIM for _ in range(FEEDBACK_DIM)]
    xty = [[0.]*FEEDBACK_DIM for _ in range(2)]
    for row in rows:
        att = gate(row, gate_model)
        x = _finite_vector(feedback_features(row, att), FEEDBACK_DIM)
        base = blend(row, att)
        target = [labels[row["id"]][k]-base[k] for k in (0, 1)]
        for i in range(FEEDBACK_DIM):
            for j in range(FEEDBACK_DIM):
                xtx[i][j] += x[i]*x[j]
            for k in (0, 1):
                xty[k][i] += x[i]*target[k]
    reg = [[xtx[i][j] + (ridge if i == j else 0.) for j in range(FEEDBACK_DIM)]
           for i in range(FEEDBACK_DIM)]
    return {"ridge": ridge, "readout": [_solve(reg, xty[k]) for k in (0, 1)]}


def correction(row, gate_model, feedback_model):
    x = feedback_features(row, gate(row, gate_model))
    return [sum(w[i]*x[i] for i in range(FEEDBACK_DIM))
            for w in feedback_model["readout"]]


def contextual_predict(row, router, use_feedback=True):
    att = gate(row, router["gate"])
    value = blend(row, att)
    if use_feedback:
        strength = router["feedback"]["beta"]
        if strength > 0:
            shift = correction(row, router["gate"], router["feedback"])
            value = [value[k]+strength*shift[k] for k in (0, 1)]
    return value


def fit(rows, weights, trainval):
    cfg = rows["config_sha256"]
    phase2._check_schema(rows, cfg)
    phase2._check_schema(weights, cfg)
    phase2._check_labels(trainval, cfg, ("train", "validation"))
    train, valid = rows["batches"]["train"], rows["batches"]["validation"]
    for split, section in (("train", train), ("validation", valid)):
        if set(trainval["labels"][split]) != {r["id"] for r in section}:
            raise ValueError("train/validation partitions do not match receipts")
    prior = phase2.train_router(rows, weights, trainval)
    base = prior["learned"]["weights"]
    tr = trainval["labels"]["train"]
    vl = trainval["labels"]["validation"]
    # Include theta=0 as a candidate: contextual routing is not presumed better.
    chosen = {"base": list(base), "theta": [[0.]*FEATURE_DIM for _ in mesh.NODES],
              "regularization": None}
    best_loss = _error_rows(valid, vl, lambda r: blend(r, gate(r, chosen)))
    searches = [{"regularization": None, "val_mse": best_loss}]
    for ridge in (0.001, 0.01, 0.05):
        model = fit_gates(train, tr, base, ridge)
        error = _error_rows(valid, vl, lambda r: blend(r, gate(r, model)))
        searches.append({"regularization": ridge, "val_mse": error})
        if error < best_loss-1e-12:
            chosen, best_loss = model, error
    # Train residual corrections only on training examples; choose beta on validation.
    residual = {"ridge": None, "readout": [[0.]*FEEDBACK_DIM for _ in range(2)]}
    selected_beta = 0.
    best_feedback = best_loss
    feedback_search = [{"ridge": None, "beta": 0., "val_mse": best_loss}]
    for ridge in (0.005, 0.05, 0.5):
        fit_model = fit_feedback(train, tr, chosen, ridge)
        for beta in (0.25, 0.5, 1.0):
            loss = _error_rows(valid, vl, lambda r: [
                blend(r, gate(r, chosen))[k] +
                beta*correction(r, chosen, fit_model)[k] for k in (0, 1)])
            feedback_search.append({"ridge": ridge, "beta": beta, "val_mse": loss})
            if loss < best_feedback-1e-12:
                residual, selected_beta, best_feedback = fit_model, beta, loss
    return {
        "schema": SCHEMA, "config_sha256": cfg, "round1_sha256": mesh.digest(rows),
        "models_sha256": mesh.digest(weights), "train_scenes": len(train),
        "validation_scenes": len(valid), "frozen_before_test": True,
        "static_router": prior, "gate": chosen,
        "gate_val_mse": best_loss, "gate_search": searches,
        "feedback": {**residual, "beta": selected_beta},
        "feedback_val_mse": best_feedback, "feedback_search": feedback_search,
    }


def feedback_worker(node, rowpacket, router):
    """Stage-two node applies signals learned from *other* specialists."""
    if node not in mesh.NODES:
        raise ValueError("unknown specialist")
    if router.get("schema") != SCHEMA or router.get("round1_sha256") != mesh.digest(rowpacket):
        raise ValueError("router/observations mismatch")
    result = {}
    for split in phase2.SPLITS:
        series = []
        for row in rowpacket["batches"][split]:
            value = row["predictions"][node][:]
            if router["feedback"]["beta"] > 0.:
                shift = correction(row, router["gate"], router["feedback"])
                value = [value[k] + router["feedback"]["beta"]*shift[k] for k in (0, 1)]
            series.append({"id": row["id"], "prediction": _finite_vector(value, 2)})
        result[split] = series
    return {"schema": SCHEMA, "config_sha256": router["config_sha256"],
            "node": node, "round1_sha256": mesh.digest(rowpacket),
            "router_sha256": mesh.digest(router), "predictions": result}


def _gather_feedback(rows, router, directory):
    paths = sorted(Path(directory).glob("node-*.json"))
    if len(paths) != NODE_COUNT:
        raise ValueError("expected six stage-two node receipts")
    by_node = {}
    for file in paths:
        result = mesh.load(file)
        if result.get("schema") != SCHEMA or result.get("config_sha256") != rows["config_sha256"]:
            raise ValueError("wrong feedback schema")
        node = result.get("node")
        if node not in mesh.NODES or node in by_node:
            raise ValueError("duplicate or unknown stage-two node")
        _check_digest(result, "round1_sha256", mesh.digest(rows))
        _check_digest(result, "router_sha256", mesh.digest(router))
        by_node[node] = result
    if set(by_node) != set(mesh.NODES):
        raise ValueError("missing stage-two node")
    combined = {}
    for split in phase2.SPLITS:
        combined[split] = []
        for idx, row in enumerate(rows["batches"][split]):
            predictions = {}
            for node in mesh.NODES:
                packet = by_node[node]["predictions"][split]
                if len(packet) != len(rows["batches"][split]) or packet[idx]["id"] != row["id"]:
                    raise ValueError("misaligned recurrent stage outputs")
                actual = _finite_vector(packet[idx]["prediction"], 2)
                expected = row["predictions"][node][:]
                if router["feedback"]["beta"] > 0.:
                    delta = correction(row, router["gate"], router["feedback"])
                    expected = [expected[k]+router["feedback"]["beta"]*delta[k] for k in (0, 1)]
                if mesh.squared_error(actual, expected) > 1e-16:
                    raise ValueError("unverified recurrent prediction")
                predictions[node] = actual
            combined[split].append({"id": row["id"], "predictions": predictions})
    return combined


def _paired_delta(a, b, rounds=600):
    if not a or len(a) != len(b):
        raise ValueError("unaligned error vectors")
    return phase2._paired_bootstrap(a, b, iterations=rounds)


def evaluate(rows, weights, router, round2_directory, test_labels):
    cfg = rows["config_sha256"]
    phase2._check_schema(rows, cfg)
    phase2._check_schema(weights, cfg)
    phase2._check_labels(test_labels, cfg, ("test",))
    if router.get("schema") != SCHEMA or router.get("config_sha256") != cfg:
        raise ValueError("incompatible frozen router")
    _check_digest(router, "round1_sha256", mesh.digest(rows))
    _check_digest(router, "models_sha256", mesh.digest(weights))
    if router.get("frozen_before_test") is not True:
        raise ValueError("missing frozen router attestation")
    round2 = _gather_feedback(rows, router, round2_directory)
    test = rows["batches"]["test"]
    labels = test_labels["labels"]["test"]
    if set(labels) != {x["id"] for x in test}:
        raise ValueError("test label IDs do not match frozen observations")
    errors = {x: [] for x in ("selected_single", "phase1_fixed", "phase21_static",
                               "phase22_context", "phase22_feedback")}
    regime_errors = {name: {key: [] for key in errors} for name in phase2.REGIMES}
    gate_sums = {n: 0. for n in mesh.NODES}
    gate_spans = {n: [1., 0.] for n in mesh.NODES}
    for idx, row in enumerate(test):
        target = labels[row["id"]]
        preds = row["predictions"]
        v0att = phase2._v0_attention(preds, weights)
        v0feedback = phase2._refine(preds, v0att, 0.28)
        fixed = phase2._consensus(v0feedback, phase2._v0_attention(v0feedback, weights))
        static = phase2._predict_final(
            preds, router["static_router"]["learned"]["weights"],
            router["static_router"]["learned"]["gamma"],
            router["static_router"]["learned"]["alpha"],
        )
        att = gate(row, router["gate"])
        context = blend(row, att)
        feedback = [sum(att[n]*round2["test"][idx]["predictions"][n][k]
                        for n in mesh.NODES) for k in (0, 1)]
        samples = {
            "selected_single": preds[router["static_router"]["selected_single_node"]],
            "phase1_fixed": fixed, "phase21_static": static,
            "phase22_context": context, "phase22_feedback": feedback,
        }
        for name, value in samples.items():
            loss = mesh.squared_error(value, target)
            errors[name].append(loss)
            regime_errors[row["regime"]][name].append(loss)
        for name, weight in att.items():
            gate_sums[name] += weight
            gate_spans[name] = [min(gate_spans[name][0], weight), max(gate_spans[name][1], weight)]
    results = {name: _mean(v) for name, v in errors.items()}
    return {
        "schema": SCHEMA, "config_sha256": cfg,
        "experiment": "contextual_gate_and_peer_embedding_feedback",
        "train_scenes": router["train_scenes"], "validation_scenes": router["validation_scenes"],
        "test_scenes": len(test), "test_mse": results,
        "by_regime": {reg: {key: _mean(value) for key, value in entry.items()}
                      for reg, entry in regime_errors.items()},
        "learned_connection_mean": {n: gate_sums[n]/len(test) for n in mesh.NODES},
        "learned_connection_range": gate_spans,
        "feedback_beta": router["feedback"]["beta"],
        "feedback_ridge": router["feedback"]["ridge"],
        "context_regularization": router["gate"]["regularization"],
        "feedback_effectively_active": router["feedback"]["beta"] > 0.,
        "contextual_beats_static": results["phase22_context"] < results["phase21_static"],
        "feedback_beats_context": results["phase22_feedback"] < results["phase22_context"],
        "feedback_beats_static": results["phase22_feedback"] < results["phase21_static"],
        "paired_feedback_minus_static": _paired_delta(errors["phase22_feedback"], errors["phase21_static"]),
        "paired_feedback_minus_context": _paired_delta(errors["phase22_feedback"], errors["phase22_context"]),
        "router_sha256": mesh.digest(router),
        "status": "SEALED_SYNTHETIC_BENCHMARK_NOT_LIVE_AR_OR_NEURAL_EMERGENCE",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sp = parser.add_subparsers(dest="command", required=True)
    p = sp.add_parser("observe")
    p.add_argument("--node", required=True, choices=mesh.NODES)
    p.add_argument("--public", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--out", required=True)
    p = sp.add_parser("fit")
    p.add_argument("--public", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--receipts", required=True)
    p.add_argument("--trainval", required=True)
    p.add_argument("--out", required=True)
    p = sp.add_parser("feedback")
    p.add_argument("--node", required=True, choices=mesh.NODES)
    p.add_argument("--rows", required=True)
    p.add_argument("--router", required=True)
    p.add_argument("--out", required=True)
    p = sp.add_parser("evaluate")
    p.add_argument("--rows", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--router", required=True)
    p.add_argument("--round2", required=True)
    p.add_argument("--test-labels", required=True)
    p.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.command == "observe":
        mesh.save(args.out, observe(args.node, mesh.load(args.public), mesh.load(args.weights)))
    elif args.command == "fit":
        public, weights = mesh.load(args.public), mesh.load(args.weights)
        rows = collect(public, weights, args.receipts)
        router = fit(rows, weights, mesh.load(args.trainval))
        out = Path(args.out)
        mesh.save(out/"rows.json", rows)
        mesh.save(out/"router.json", router)
        print(json.dumps({"context_val_mse": router["gate_val_mse"],
                          "feedback_val_mse": router["feedback_val_mse"],
                          "feedback_beta": router["feedback"]["beta"],
                          "context_regularization": router["gate"]["regularization"]}, indent=2))
    elif args.command == "feedback":
        mesh.save(args.out, feedback_worker(args.node, mesh.load(args.rows), mesh.load(args.router)))
    else:
        output = evaluate(mesh.load(args.rows), mesh.load(args.weights),
                          mesh.load(args.router), args.round2, mesh.load(args.test_labels))
        target = Path(args.out)
        mesh.save(target, output)
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

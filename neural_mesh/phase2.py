"""Phase 2.1: held-out multi-scenario synaptic-plasticity benchmark.

Only the evaluator receives test ground truth. Workers transmit numeric forecasts.
Python stdlib only; no large model claims or video generation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

from neural_mesh import mesh

SCHEMA = "neural-mesh-phase-2.1"
SPLITS = ("train", "validation", "test")
REGIMES = ("linear", "accelerating", "jerk", "turning")


def _seed(*parts):
    payload = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _pair(p):
    if not isinstance(p, list) or len(p) != 2 or any(
        not isinstance(x, (int, float)) or not math.isfinite(x) for x in p
    ):
        raise ValueError("invalid or non-finite vector")
    return p


def _check_schema(obj, config):
    if obj.get("schema") != SCHEMA or obj.get("config_sha256") != config:
        raise ValueError("incompatible phase2 schema or config digest")


def trajectory(rng, regime):
    x, y = rng.uniform(-2, 2), rng.uniform(-2, 2)
    vx, vy = rng.uniform(-0.9, 0.9), rng.uniform(-0.9, 0.9)
    ax, ay = rng.uniform(-0.16, 0.16), rng.uniform(-0.16, 0.16)
    if regime == "linear":
        ax = ay = 0.0
    if regime == "jerk":
        jx, jy = rng.uniform(-0.07, 0.07), rng.uniform(-0.07, 0.07)
    else:
        jx = jy = 0.0
    if regime == "turning":
        omega, phase = rng.uniform(0.13, 0.42), rng.uniform(-math.pi, math.pi)
        radius = rng.uniform(0.25, 0.9)
        return [[
            x + 0.35*vx*t + radius*(math.cos(phase+omega*t)-math.cos(phase)),
            y + 0.35*vy*t + radius*(math.sin(phase+omega*t)-math.sin(phase)),
        ] for t in range(5)]
    if regime not in REGIMES:
        raise ValueError("unknown motion regime")
    return [[x + vx*t + ax*t*t/2 + jx*t**3/6,
             y + vy*t + ay*t*t/2 + jy*t**3/6] for t in range(5)]


def prepare(seed, counts, out):
    if not isinstance(seed, int) or not 0 <= seed < 2**31:
        raise ValueError("invalid seed")
    if set(counts) != set(SPLITS) or any(
        not isinstance(counts[k], int) or not 4 <= counts[k] <= 4096 for k in SPLITS
    ):
        raise ValueError("split sizes must each be between 4 and 4096")
    config = {"seed": seed, "counts": counts, "regimes": list(REGIMES)}
    cfg_hash = mesh.digest(config)
    rng = random.Random(97211)
    nodes = {n: mesh.learn(n, rng) for n in mesh.NODES}
    model = {"schema": SCHEMA, "config_sha256": cfg_hash, "nodes": nodes}
    public = {"schema": SCHEMA, "config_sha256": cfg_hash, "scenes": {}}
    labels = {"train": {}, "validation": {}, "test": {}}
    for split in SPLITS:
        records = []
        for idx in range(counts[split]):
            regime = REGIMES[idx % len(REGIMES)]
            uid = f"{split}-{idx:05d}"
            motion = trajectory(random.Random(_seed(seed, split, idx, "motion")), regime)
            traces = {
                name: mesh.sensor(motion, name, random.Random(_seed(seed, split, idx, name)))
                for name in mesh.NODES
            }
            records.append({"id": uid, "regime": regime, "tracks": traces})
            labels[split][uid] = motion[4]
        public["scenes"][split] = records
    root = Path(out)
    mesh.save(root / "weights.json", model)
    mesh.save(root / "public.json", public)
    mesh.save(root / "trainval-labels.json", {
        "schema": SCHEMA, "config_sha256": cfg_hash,
        "labels": {s: labels[s] for s in ("train", "validation")},
    })
    mesh.save(root / "test-labels.json", {
        "schema": SCHEMA, "config_sha256": cfg_hash,
        "labels": {"test": labels["test"]},
    })
    mesh.save(root / "manifest.json", {"schema": SCHEMA, "config_sha256": cfg_hash,
              "seed": seed, "counts": counts, "train_label_visibility": "fit-only",
              "test_label_visibility": "evaluator-only"})


def infer(node, public, weights):
    if node not in mesh.NODES:
        raise ValueError("unknown specialist")
    cfg = public["config_sha256"]
    _check_schema(public, cfg)
    _check_schema(weights, cfg)
    result = {}
    for split in SPLITS:
        result[split] = []
        for sample in public["scenes"][split]:
            forecast = mesh.predict(sample["tracks"][node], node, weights["nodes"][node]["readout"])
            result[split].append({"id": sample["id"], "prediction": _pair(forecast)})
    return {"schema": SCHEMA, "config_sha256": cfg, "node": node,
            "model_sha256": mesh.digest(weights["nodes"][node]),
            "public_sha256": mesh.digest(public), "predictions": result}


def collect(public, weights, directory):
    cfg = public["config_sha256"]
    _check_schema(public, cfg)
    _check_schema(weights, cfg)
    receipts = sorted(Path(directory).glob("node-*.json"))
    if len(receipts) != len(mesh.NODES):
        raise ValueError("expected exactly six node receipts")
    results = [mesh.load(p) for p in receipts]
    by_node = {}
    for r in results:
        _check_schema(r, cfg)
        node = r.get("node")
        if node not in mesh.NODES or node in by_node:
            raise ValueError("missing or duplicate specialist")
        if r.get("model_sha256") != mesh.digest(weights["nodes"][node]):
            raise ValueError("specialist model mismatch")
        if r.get("public_sha256") != mesh.digest(public):
            raise ValueError("specialist observation mismatch")
        by_node[node] = r
    if set(by_node) != set(mesh.NODES):
        raise ValueError("incomplete specialist set")
    batches = {}
    for split in SPLITS:
        refs = public["scenes"][split]
        expected = [r["id"] for r in refs]
        node_maps = {}
        for name in mesh.NODES:
            entries = by_node[name]["predictions"][split]
            keys = [x["id"] for x in entries]
            if keys != expected or len(set(keys)) != len(keys):
                raise ValueError("missing, extra, reordered, or duplicate scene prediction")
            node_maps[name] = {p["id"]: _pair(p["prediction"]) for p in entries}
        batches[split] = [
            {"id": x["id"], "regime": x["regime"],
             "predictions": {name: node_maps[name][x["id"]] for name in mesh.NODES}}
            for x in refs
        ]
    return {"schema": SCHEMA, "config_sha256": cfg,
            "public_sha256": mesh.digest(public), "batches": batches,
            "models_sha256": mesh.digest(weights)}


def _check_labels(labels, cfg, expected):
    _check_schema(labels, cfg)
    if set(labels.get("labels", {})) != set(expected):
        raise ValueError("label partitions violate split isolation")
    for split in expected:
        if len(labels["labels"][split]) != len(set(labels["labels"][split])):
            raise ValueError("duplicate targets")
        for v in labels["labels"][split].values():
            _pair(v)


def _mean(errors):
    return sum(errors) / len(errors)


def _project_simplex(values):
    ordered = sorted(values, reverse=True)
    cumulative = 0.
    rho, theta = 0, 0.
    for j, v in enumerate(ordered, 1):
        cumulative += v
        candidate = (cumulative - 1.) / j
        if v > candidate:
            rho, theta = j, candidate
    if not rho:
        raise ValueError("simplex projection failed")
    result = [max(0., v-theta) for v in values]
    return [v/sum(result) for v in result]


def _prior(weights):
    q = [1./max(1e-7, weights["nodes"][n]["validation_mse"]) for n in mesh.NODES]
    return [v/sum(q) for v in q]


def _fit(rows, targets, prior, ridge):
    w = list(prior)
    n = len(rows)
    for _ in range(400):
        gradients = [ridge*(w[i]-prior[i]) for i in range(len(mesh.NODES))]
        for row in rows:
            pred = row["predictions"]
            center = [sum(w[i]*pred[node][k] for i, node in enumerate(mesh.NODES))
                      for k in (0, 1)]
            truth = targets[row["id"]]
            for i, node in enumerate(mesh.NODES):
                gradients[i] += sum((center[k]-truth[k])*pred[node][k]
                                    for k in (0, 1))/n
        w = _project_simplex([w[i]-0.6*gradients[i] for i in range(len(w))])
    return w


def _attention(pred, weights, gamma):
    center = [sum(pred[node][k] for node in mesh.NODES)/len(mesh.NODES)
              for k in (0, 1)]
    numer = {
        name: max(1e-5, weights[i]) *
        math.exp(-gamma*mesh.squared_error(pred[name], center))
        for i, name in enumerate(mesh.NODES)
    }
    denom = sum(numer.values())
    return {name: numer[name]/denom for name in mesh.NODES}


def _consensus(pred, att):
    return [sum(att[n]*pred[n][k] for n in mesh.NODES) for k in (0, 1)]


def _refine(pred, att, alpha):
    result = {}
    for receiver in mesh.NODES:
        neighbors = {n: max(1e-8, att[n]) for n in mesh.NODES if n != receiver}
        norm = sum(neighbors.values())
        result[receiver] = [
            (1-alpha)*pred[receiver][k] +
            alpha*sum(neighbors[n]*pred[n][k] for n in neighbors)/norm
            for k in (0, 1)
        ]
    return result


def _v0_attention(pred, weights):
    prior = [1./max(1e-6, weights["nodes"][name]["validation_mse"])
             for name in mesh.NODES]
    center = [sum(pred[n][k] for n in mesh.NODES)/len(mesh.NODES) for k in (0, 1)]
    scores = [prior[i]/(1+2*mesh.squared_error(pred[n], center))
              for i, n in enumerate(mesh.NODES)]
    return {n: scores[i]/sum(scores) for i, n in enumerate(mesh.NODES)}


def _predict_final(pred, w, gamma, alpha):
    p1 = _attention(pred, w, gamma)
    p2 = _refine(pred, p1, alpha)
    return _consensus(p2, _attention(p2, w, gamma))


def train_router(round1, weights, trainval):
    cfg = round1["config_sha256"]
    _check_schema(round1, cfg)
    _check_schema(weights, cfg)
    _check_labels(trainval, cfg, ("train", "validation"))
    train, valid = round1["batches"]["train"], round1["batches"]["validation"]
    for split, rows in (("train", train), ("validation", valid)):
        if {r["id"] for r in rows} != set(trainval["labels"][split]):
            raise ValueError("training/validation label mismatch")
    tr_labels = trainval["labels"]["train"]
    val_labels = trainval["labels"]["validation"]
    prior = _prior(weights)
    best = None
    search = []
    for ridge in (0.0, 0.01, 0.1):
        w = _fit(train, tr_labels, prior, ridge)
        for gamma in (0., 2., 8., 20.):
            for alpha in (0., 0.2, 0.5):
                loss = _mean([mesh.squared_error(
                    _predict_final(r["predictions"], w, gamma, alpha),
                    val_labels[r["id"]]) for r in valid])
                search.append({"ridge": ridge, "gamma": gamma, "alpha": alpha,
                               "validation_mse": loss})
                if best is None or loss < best["validation_mse"] - 1e-12:
                    best = {"weights": w, "ridge": ridge, "gamma": gamma,
                            "alpha": alpha, "validation_mse": loss}
    per_node_val = {n: _mean([
        mesh.squared_error(r["predictions"][n], val_labels[r["id"]])
        for r in valid]) for n in mesh.NODES}
    selected = min(mesh.NODES, key=lambda n: per_node_val[n])
    return {"schema": SCHEMA, "config_sha256": cfg,
            "round1_sha256": mesh.digest(round1), "model_sha256": mesh.digest(weights),
            "trained_scenes": len(train), "validation_scenes": len(valid),
            "frozen_before_test": True, "selected_single_node": selected,
            "selected_single_val_mse": per_node_val[selected],
            "learned": best, "search": search}


def refine(node, round1, router):
    cfg = round1["config_sha256"]
    _check_schema(round1, cfg)
    _check_schema(router, cfg)
    if router["round1_sha256"] != mesh.digest(round1):
        raise ValueError("router not tied to phase2 observations")
    if node not in mesh.NODES:
        raise ValueError("unknown specialist")
    best = router["learned"]
    results = {}
    for split in SPLITS:
        results[split] = []
        for row in round1["batches"][split]:
            p = row["predictions"]
            att = _attention(p, best["weights"], best["gamma"])
            updated = _refine(p, att, best["alpha"])[node]
            results[split].append({"id": row["id"], "prediction": _pair(updated)})
    return {"schema": SCHEMA, "config_sha256": cfg, "node": node,
            "round1_sha256": mesh.digest(round1), "router_sha256": mesh.digest(router),
            "predictions": results}


def _collect_refined(round1, router, directory):
    files = sorted(Path(directory).glob("node-*.json"))
    if len(files) != len(mesh.NODES):
        raise ValueError("expected exactly six refined receipts")
    by_node = {}
    for path in files:
        r = mesh.load(path)
        _check_schema(r, round1["config_sha256"])
        n = r.get("node")
        if n not in mesh.NODES or n in by_node:
            raise ValueError("missing or duplicate refined node")
        if r.get("round1_sha256") != mesh.digest(round1) or r.get("router_sha256") != mesh.digest(router):
            raise ValueError("mismatched recurrent packet")
        by_node[n] = r
    out = {}
    for split in SPLITS:
        sceneids = [r["id"] for r in round1["batches"][split]]
        each = {}
        for n in mesh.NODES:
            series = by_node[n]["predictions"][split]
            if [p["id"] for p in series] != sceneids:
                raise ValueError("reordered or missing recurrent predictions")
            each[n] = series
        out[split] = [
            {"id": uid, "predictions": {n: _pair(each[n][i]["prediction"]) for n in mesh.NODES}}
            for i, uid in enumerate(sceneids)
        ]
    return out


def _quantile(values, percentile):
    ordered = sorted(values)
    idx = (len(ordered)-1)*percentile
    a, b = int(math.floor(idx)), int(math.ceil(idx))
    return ordered[a]*(b-idx)+ordered[b]*(idx-a) if b != a else ordered[a]


def _paired_bootstrap(a, b, iterations=500):
    if len(a) != len(b) or not a:
        raise ValueError("paired bootstrap requires aligned vectors")
    rng = random.Random(60721)
    diffs = [a[i]-b[i] for i in range(len(a))]
    sampled = []
    for _ in range(iterations):
        sampled.append(sum(diffs[rng.randrange(len(diffs))]
                           for __ in range(len(diffs)))/len(diffs))
    return {"delta_mse": _mean(diffs), "ci95": [_quantile(sampled, 0.025),
                                                _quantile(sampled, 0.975)]}


def evaluate(round1, router, refined_dir, weights, test_labels):
    cfg = round1["config_sha256"]
    for item in (round1, router, weights):
        _check_schema(item, cfg)
    _check_labels(test_labels, cfg, ("test",))
    if router["model_sha256"] != mesh.digest(weights) or router["round1_sha256"] != mesh.digest(round1):
        raise ValueError("model/observation checkpoint mismatch")
    refined = _collect_refined(round1, router, refined_dir)
    test = round1["batches"]["test"]
    if {r["id"] for r in test} != set(test_labels["labels"]["test"]):
        raise ValueError("test split misaligned")
    best = router["learned"]
    metrics = {"selected_single": [], "uniform": [], "v0_round1": [],
               "v0_round2": [], "learned_round1": [], "learned_round2": []}
    individuals = {n: [] for n in mesh.NODES}
    regimes = {n: [] for n in REGIMES}
    for i, row in enumerate(test):
        uid, regime, preds = row["id"], row["regime"], row["predictions"]
        truth = test_labels["labels"]["test"][uid]
        stage2 = refined["test"][i]["predictions"]
        expected = _refine(preds, _attention(preds, best["weights"], best["gamma"]), best["alpha"])
        if any(mesh.squared_error(stage2[n], expected[n]) > 1e-16 for n in mesh.NODES):
            raise ValueError("refinement not derived from frozen learned routing")
        for n in mesh.NODES:
            individuals[n].append(mesh.squared_error(preds[n], truth))
        v0_att = _v0_attention(preds, weights)
        v0_refined = _refine(preds, v0_att, 0.28)
        values = {
            "selected_single": preds[router["selected_single_node"]],
            "uniform": [sum(preds[n][k] for n in mesh.NODES)/len(mesh.NODES) for k in (0, 1)],
            "v0_round1": _consensus(preds, v0_att),
            "v0_round2": _consensus(v0_refined, _v0_attention(v0_refined, weights)),
            "learned_round1": _consensus(preds, _attention(preds, best["weights"], best["gamma"])),
            "learned_round2": _consensus(stage2, _attention(stage2, best["weights"], best["gamma"])),
        }
        for name, p in values.items():
            metrics[name].append(mesh.squared_error(p, truth))
        regimes[regime].append(i)
    summary = {name: _mean(errors) for name, errors in metrics.items()}
    node_summary = {n: _mean(errors) for n, errors in individuals.items()}
    summary["retrospective_best_node"] = min(node_summary.values())
    per_regime = {regime: {name: _mean([errors[i] for i in ids])
                          for name, errors in metrics.items()}
                  for regime, ids in regimes.items() if ids}
    report = {"schema": SCHEMA, "config_sha256": cfg, "test_scenes": len(test),
              "train_scenes": router["trained_scenes"], "validation_scenes": router["validation_scenes"],
              "experiment": "synthetic_four_regime_motion_prediction",
              "models": summary, "individual_models": node_summary,
              "per_regime": per_regime, "selected_single_node": router["selected_single_node"],
              "selected_routing": best, "frozen_router_sha256": mesh.digest(router),
              "paired_difference_vs_v0_round2": _paired_bootstrap(
                  metrics["learned_round2"], metrics["v0_round2"]),
              "paired_difference_vs_selected_single": _paired_bootstrap(
                  metrics["learned_round2"], metrics["selected_single"]),
              "learned_beats_v0_round2": summary["learned_round2"] < summary["v0_round2"],
              "learned_beats_selected_single": summary["learned_round2"] < summary["selected_single"],
              "status": "HELD_OUT_SYNTHETIC_BENCHMARK_NOT_REAL_VIDEO_OR_GENERAL_INTELLIGENCE"}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sp = parser.add_subparsers(dest="command", required=True)
    p = sp.add_parser("prepare")
    p.add_argument("--seed", type=int, default=314159)
    p.add_argument("--train-count", type=int, default=512)
    p.add_argument("--validation-count", type=int, default=128)
    p.add_argument("--test-count", type=int, default=256)
    p.add_argument("--out", required=True)
    p = sp.add_parser("infer")
    p.add_argument("--node", choices=mesh.NODES, required=True)
    p.add_argument("--public", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--out", required=True)
    p = sp.add_parser("train")
    p.add_argument("--public", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--receipts", required=True)
    p.add_argument("--trainval", required=True)
    p.add_argument("--out", required=True)
    p = sp.add_parser("refine")
    p.add_argument("--node", choices=mesh.NODES, required=True)
    p.add_argument("--round1", required=True)
    p.add_argument("--router", required=True)
    p.add_argument("--out", required=True)
    p = sp.add_parser("evaluate")
    p.add_argument("--round1", required=True)
    p.add_argument("--router", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--refined", required=True)
    p.add_argument("--test-labels", required=True)
    p.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.seed, {"train": args.train_count, "validation": args.validation_count,
                            "test": args.test_count}, args.out)
    elif args.command == "infer":
        mesh.save(args.out, infer(args.node, mesh.load(args.public), mesh.load(args.weights)))
    elif args.command == "train":
        public, weights = mesh.load(args.public), mesh.load(args.weights)
        round1 = collect(public, weights, args.receipts)
        router = train_router(round1, weights, mesh.load(args.trainval))
        mesh.save(Path(args.out) / "round1.json", round1)
        mesh.save(Path(args.out) / "router.json", router)
        print(json.dumps({"selected": router["learned"], "selected_single": router["selected_single_node"]}, indent=2))
    elif args.command == "refine":
        mesh.save(args.out, refine(args.node, mesh.load(args.round1), mesh.load(args.router)))
    else:
        report = evaluate(mesh.load(args.round1), mesh.load(args.router), args.refined,
                          mesh.load(args.weights), mesh.load(args.test_labels))
        mesh.save(Path(args.out) / "metrics.json", report)
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

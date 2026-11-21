"""Phase 2.4 — fault-tolerant routing, novel motion tests, versioned neural memory.

The only persistent memory is an explicit, checksummed model checkpoint.
The learned state is fitted exclusively on Phase 2.3 train/validation examples.
Novel-motion target coordinates are never available to the probe/restore worker.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import statistics
import tempfile
from pathlib import Path

from neural_mesh import mesh, phase2, phase22, phase23

MEMORY_SCHEMA = "neuralmesh-memory-v1"
OOD_SCHEMA = "neuralmesh-phase24-ood-v1"
REGIMES = ("bounce", "stop_start", "swirl", "late_impulse")
FAULTS = ("none", "missing", "corrupt")
SEEDS = (48017, 58129, 68147)
NAMES = mesh.NODES


def _check_vector(x, dim=2):
    if not isinstance(x, list) or len(x) != dim or any(
        not isinstance(z, (float, int)) or not math.isfinite(z) for z in x
    ):
        raise ValueError("expected finite numeric vector")
    return x


def novel_trajectory(seed, regime):
    rng = random.Random(seed)
    x, y = rng.uniform(-2, 2), rng.uniform(-2, 2)
    vx, vy = rng.uniform(-0.85, 0.85), rng.uniform(-0.85, 0.85)
    if regime == "bounce":
        # Reflect at a wall between frames 2 and 3, a behavior absent in Phase 2.
        t0 = [x + vx*t for t in (0, 1, 2)]
        t1 = [y + vy*t for t in (0, 1, 2)]
        return [[t0[0], t1[0]], [t0[1], t1[1]],
                [t0[2], t1[2]], [t0[2]-vx, t1[2]+vy],
                [t0[2]-2*vx, t1[2]+2*vy]]
    if regime == "stop_start":
        return [[x, y], [x+vx, y+vy], [x+2*vx, y+2*vy],
                [x+2*vx, y+2*vy], [x+3.6*vx, y+3.6*vy]]
    if regime == "swirl":
        omega = rng.uniform(.5, 1.15)
        phase = rng.uniform(-math.pi, math.pi)
        radius = rng.uniform(.4, 1.2)
        return [[x + radius*math.cos(phase+omega*t)+.12*vx*t,
                 y + radius*math.sin(phase+omega*t)+.12*vy*t] for t in range(5)]
    if regime == "late_impulse":
        ax, ay = rng.uniform(-.12, .12), rng.uniform(-.12, .12)
        points = [[x+vx*t+ax*t*t/2, y+vy*t+ay*t*t/2] for t in range(5)]
        points[4][0] += rng.choice((-1, 1))*rng.uniform(.3, .85)
        points[4][1] += rng.choice((-1, 1))*rng.uniform(.3, .85)
        return points
    raise ValueError("unknown novelty regime")


def prepare_ood(seed, count, out):
    if count < 16 or count > 2048 or count % 4 != 0:
        raise ValueError("OOD count must be a multiple of four in [16, 2048]")
    if seed < 0:
        raise ValueError("invalid OOD seed")
    samples, targets = [], {}
    for i in range(count):
        regime = REGIMES[i % len(REGIMES)]
        uid = f"ood-{seed}-{i:05d}"
        pts = novel_trajectory(phase2._seed(seed, i, regime), regime)
        samples.append({"id": uid, "regime": regime,
                        "tracks": {n: mesh.sensor(pts, n, random.Random(
                            phase2._seed(seed, i, n, "sensor"))) for n in NAMES}})
        targets[uid] = pts[4]
    public = {"schema": OOD_SCHEMA, "seed": seed, "scenes": samples}
    target = {"schema": OOD_SCHEMA, "seed": seed,
              "public_sha256": mesh.digest(public), "targets": targets}
    root = Path(out)
    mesh.save(root/"public.json", public)
    mesh.save(root/"sealed-targets.json", target)
    return public


def make_row(scene, weights):
    return {"id": scene["id"], "regime": scene["regime"],
            "predictions": {n: mesh.predict(scene["tracks"][n], n,
                                           weights["nodes"][n]["readout"]) for n in NAMES},
            "embeddings": {n: mesh.features(scene["tracks"][n], n) for n in NAMES}}


def inject(row, kind, index):
    """Deterministic fault, not revealed to detection algorithm."""
    if kind not in FAULTS:
        raise ValueError("unknown fault type")
    damaged = copy.deepcopy(row)
    if kind == "none":
        return damaged
    node = NAMES[index % len(NAMES)]
    if kind == "missing":
        damaged["predictions"].pop(node)
        damaged["embeddings"].pop(node)
    else:
        damaged["predictions"][node][0] += 3.5
        damaged["predictions"][node][1] -= 2.9
        damaged["embeddings"][node][0] += 2.0
        damaged["embeddings"][node][1] -= 1.8
    return damaged


def _impute(row, active):
    if len(active) < 3:
        raise ValueError("insufficient healthy nodes")
    result = copy.deepcopy(row)
    for n in NAMES:
        if n not in active:
            result["predictions"][n] = [
                statistics.median(row["predictions"][k][j] for k in active) for j in (0, 1)]
            result["embeddings"][n] = [
                statistics.median(row["embeddings"][k][j] for k in active) for j in range(9)]
    return result


def _active(row, cutoff):
    good = [n for n in NAMES if n in row["predictions"] and n in row["embeddings"]]
    for n in good:
        _check_vector(row["predictions"][n])
        _check_vector(row["embeddings"][n], 9)
    if len(good) < 4:
        raise ValueError("too few live specialists")
    median = [statistics.median(row["predictions"][n][k] for n in good) for k in (0, 1)]
    distance = {n: math.dist(row["predictions"][n], median) for n in good}
    scale = statistics.median(distance.values())
    accepted = [n for n in good if distance[n] <= max(cutoff, 3.5*scale)]
    # Fail closed rather than reporting an invented confidence when most disagree.
    if len(accepted) < 4:
        raise ValueError("not enough consistent specialist predictions")
    return accepted


def _baseline(row, memory):
    base, selector = memory["phase22_router"], memory["phase23_selector"]
    return phase23._baseline(row, base), phase23._delta(row, base), phase23.activation(
        phase23._vector(row, base), selector["selected"])


def naive_prediction(row, memory):
    # Missing node is imputed by a naive median, but *not* masked in attention.
    available = [n for n in NAMES if n in row["predictions"] and n in row["embeddings"]]
    filled = _impute(row, available)
    base, delta, beta = _baseline(filled, memory)
    return [base[k]+beta*delta[k] for k in (0, 1)]


def robust_prediction(row, memory, cutoff, feedback_scale):
    active = _active(row, cutoff)
    filled = _impute(row, active)
    base = memory["phase22_router"]
    selector = memory["phase23_selector"]
    att = phase22.gate(filled, base["gate"])
    z = sum(att[n] for n in active)
    context = [sum(att[n]/z*row["predictions"][n][k] for n in active) for k in (0, 1)]
    beta = phase23.activation(phase23._vector(filled, base), selector["selected"])
    correction = phase23._delta(filled, base)
    strength = feedback_scale*beta*(len(active)/len(NAMES))
    return [context[k]+strength*correction[k] for k in (0, 1)], active


def fit_reliability(public, weights, router, selector, trainval):
    cfg = public["config_sha256"]
    phase2._check_schema(public, cfg)
    phase2._check_schema(weights, cfg)
    phase2._check_labels(trainval, cfg, ("train", "validation"))
    if router.get("round1_sha256") != selector.get("round1_sha256"):
        raise ValueError("source routers are inconsistent")
    memory = {"models": weights, "phase22_router": router, "phase23_selector": selector}
    rows = public["scenes"]["validation"]
    targets = trainval["labels"]["validation"]
    options = [(cutoff, s) for cutoff in (.25, .45, .75, 1.2)
               for s in (0., .4, .8, 1.)]
    ratings = []
    for cutoff, scale in options:
        losses = []
        for i, scene in enumerate(rows):
            clean = make_row(scene, weights)
            for fault in FAULTS:
                testrow = inject(clean, fault, i)
                try:
                    pred, _ = robust_prediction(testrow, memory, cutoff, scale)
                    err = mesh.squared_error(pred, targets[scene["id"]])
                except ValueError:
                    err = 1e3
                losses.append(err)
        ratings.append({"cutoff": cutoff, "feedback_scale": scale,
                        "validation_mse": statistics.fmean(losses)})
    selected = min(ratings, key=lambda x: x["validation_mse"])
    return {"selected": selected, "candidate_scores": ratings,
            "validation_scenes": len(rows), "faults": list(FAULTS),
            "calibration_is_test_blind": True}


def build_memory(public_file, weights_file, trainval_file, out):
    public = mesh.load(public_file)
    weights = mesh.load(weights_file)
    trainval = mesh.load(trainval_file)
    with tempfile.TemporaryDirectory() as temp:
        from neural_mesh.phase23 import fit_seed
        fit_seed(public_file, weights_file, trainval_file, temp)
        router = mesh.load(Path(temp)/"phase22-router.json")
        selector = mesh.load(Path(temp)/"phase23-selector.json")
    calibration = fit_reliability(public, weights, router, selector, trainval)
    payload = {"schema": MEMORY_SCHEMA, "version": 1,
               "config_sha256": public["config_sha256"],
               "models": weights, "phase22_router": router,
               "phase23_selector": selector, "reliability": calibration,
               "history": [{"phase": "2.3", "source": "trained-on-split"},
                           {"phase": "2.4", "source": "validated-fault-calibration"}],
               "source_data_sha256": mesh.digest(public),
               "adopted_automatically": False}
    checkpoint = {"payload": payload, "sha256": mesh.digest(payload)}
    mesh.save(out, checkpoint)
    return checkpoint


def restore_memory(path):
    wrapper = mesh.load(path)
    if set(wrapper) != {"payload", "sha256"}:
        raise ValueError("unknown checkpoint wrapper")
    payload = wrapper["payload"]
    if payload.get("schema") != MEMORY_SCHEMA or payload.get("version") != 1:
        raise ValueError("incompatible memory format")
    if wrapper["sha256"] != mesh.digest(payload):
        raise ValueError("memory digest mismatch")
    if payload["models"].get("config_sha256") != payload["config_sha256"]:
        raise ValueError("memory model/config mismatch")
    if payload["phase23_selector"].get("config_sha256") != payload["config_sha256"]:
        raise ValueError("memory selector/config mismatch")
    if payload["phase22_router"].get("config_sha256") != payload["config_sha256"]:
        raise ValueError("memory router/config mismatch")
    if payload.get("adopted_automatically") is not False:
        raise ValueError("auto-promoted memory is not authorized")
    chosen = payload["reliability"]["selected"]
    if chosen["cutoff"] <= 0 or not 0 <= chosen["feedback_scale"] <= 1:
        raise ValueError("invalid memory calibration")
    return wrapper


def probe(public, checkpoint):
    memory = checkpoint["payload"]
    if public.get("schema") != OOD_SCHEMA:
        raise ValueError("wrong OOD public data")
    scored = []
    for i, scene in enumerate(public["scenes"]):
        row = make_row(scene, memory["models"])
        faults = {}
        for kind in FAULTS:
            damaged = inject(row, kind, i)
            naive = naive_prediction(damaged, memory)
            chosen = memory["reliability"]["selected"]
            robust, accepted = robust_prediction(
                damaged, memory, chosen["cutoff"], chosen["feedback_scale"])
            faults[kind] = {"naive": _check_vector(naive),
                            "robust": _check_vector(robust),
                            "accepted_node_count": len(accepted)}
        scored.append({"id": scene["id"], "regime": scene["regime"], "faults": faults})
    return {"schema": OOD_SCHEMA, "seed": public["seed"],
            "public_sha256": mesh.digest(public),
            "memory_sha256": checkpoint["sha256"], "scenes": scored}


def evaluate_probe(predictions, truth):
    if predictions.get("schema") != OOD_SCHEMA or truth.get("schema") != OOD_SCHEMA:
        raise ValueError("OOD schema mismatch")
    if predictions.get("seed") != truth.get("seed") or predictions.get("public_sha256") != truth.get("public_sha256"):
        raise ValueError("OOD seed/public hash mismatch")
    entries = predictions["scenes"]
    if not entries or {e["id"] for e in entries} != set(truth["targets"]):
        raise ValueError("missing or duplicated OOD targets")
    regime = {r: {f:{"naive":[], "robust":[]} for f in FAULTS} for r in REGIMES}
    acc = {f:{"naive":[], "robust":[]} for f in FAULTS}
    accepted = {f:[] for f in FAULTS}
    for e in entries:
        if e["regime"] not in REGIMES or set(e["faults"]) != set(FAULTS):
            raise ValueError("unexpected OOD label or fault")
        target = _check_vector(truth["targets"][e["id"]])
        for fault, values in e["faults"].items():
            for approach in ("naive", "robust"):
                loss = mesh.squared_error(_check_vector(values[approach]), target)
                regime[e["regime"]][fault][approach].append(loss)
                acc[fault][approach].append(loss)
            accepted[fault].append(values["accepted_node_count"])
    return {"schema": OOD_SCHEMA, "seed": truth["seed"],
            "test_scenes":len(entries), "memory_sha256":predictions["memory_sha256"],
            "test_mse":{f:{a:statistics.fmean(vals) for a,vals in d.items()}
                        for f,d in acc.items()},
            "by_regime": {r:{f:{a:statistics.fmean(vals) for a,vals in data.items()}
                             for f,data in d.items()} for r,d in regime.items()},
            "mean_healthy_nodes":{f:statistics.fmean(vals) for f,vals in accepted.items()},
            "status":"HELD_OUT_OOD_SYNTHETIC_FAULT_BENCHMARK"}


def aggregate(results_dir, seeds, out):
    files = list(Path(results_dir).glob("seed-*.json"))
    if len(files) != len(seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("incorrect number of unique seed reports")
    reports = {int(r["seed"]):r for r in (mesh.load(f) for f in files)}
    if set(reports) != set(seeds):
        raise ValueError("wrong seed reports")
    digests = {r["memory_sha256"] for r in reports.values()}
    if len(digests) != 1:
        raise ValueError("mixed memory checkpoint versions")
    avg = {fault:{a:statistics.fmean(reports[s]["test_mse"][fault][a] for s in seeds)
                  for a in ("naive","robust")} for fault in FAULTS}
    report = {"schema":OOD_SCHEMA, "seeds":list(seeds), "memory_sha256":digests.pop(),
              "per_seed":[reports[s] for s in seeds], "mean_mse":avg,
              "test_scenes":sum(reports[s]["test_scenes"] for s in seeds),
              "fault_robustness_wins":{f:sum(reports[s]["test_mse"][f]["robust"] <
                  reports[s]["test_mse"][f]["naive"] for s in seeds) for f in FAULTS},
              "status":"CROSS_SEED_OOD_AND_FAULT_RESEARCH_ONLY"}
    mesh.save(out, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("make-ood")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--count", type=int, default=128)
    p.add_argument("--out", required=True)
    p = sub.add_parser("train-memory")
    p.add_argument("--public", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--trainval", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("verify-memory")
    p.add_argument("--memory", required=True)
    p = sub.add_parser("probe")
    p.add_argument("--memory", required=True)
    p.add_argument("--public", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("evaluate")
    p.add_argument("--predictions", required=True)
    p.add_argument("--targets", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("aggregate")
    p.add_argument("--results", required=True)
    p.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    p.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.command == "make-ood":
        prepare_ood(args.seed, args.count, args.out)
    elif args.command == "train-memory":
        state=build_memory(args.public,args.weights,args.trainval,args.out)
        print({"memory_sha256":state["sha256"],
               "selected_robustness":state["payload"]["reliability"]["selected"]})
    elif args.command == "verify-memory":
        print({"verified_memory_sha256":restore_memory(args.memory)["sha256"]})
    elif args.command == "probe":
        checkpoint=restore_memory(args.memory)
        result=probe(mesh.load(args.public), checkpoint)
        mesh.save(args.out,result)
        print({"seed":result["seed"],"scenes":len(result["scenes"])})
    elif args.command == "evaluate":
        result=evaluate_probe(mesh.load(args.predictions),mesh.load(args.targets))
        mesh.save(args.out,result)
        print({"seed":result["seed"],"test_mse":result["test_mse"]})
    else:
        report=aggregate(args.results,tuple(int(v) for v in args.seeds.split(",")),args.out)
        print({"mean_mse":report["mean_mse"],
               "fault_robustness_wins":report["fault_robustness_wins"]})


if __name__ == "__main__":
    main()

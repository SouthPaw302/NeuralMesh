"""Twenty distinct NeuralMesh specialists executed on twenty GitHub runner jobs.

Stdlib-only deterministic synthetic motion experiment. Two sequential waves of
20 parallel workers exchange verified all-to-all prediction packets. No claim
of biological neurons, real-time transport, or autonomous weight promotion.

The original six-node experiments remain unchanged and independently runnable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from pathlib import Path

from neural_mesh import mesh

SCHEMA = "neuralmesh-20-v1"
ROLES = ("position", "velocity", "acceleration", "curvature", "stabilizer")
NODES = tuple(f"{role}-{replica:02d}" for role in ROLES for replica in range(4))
assert len(NODES) == len(set(NODES)) == 20
ROLE_MASKS = {
    "position": (1.0, .18, .45, .30),
    "velocity": (1.30, .25, .20, .20),
    "acceleration": (.8, 1.4, .55, .4),
    "curvature": (.70, .45, .45, 1.5),
    "stabilizer": (.75, .32, 1.0, .60),
}
ROLE_NOISE = {
    "position": .07, "velocity": .12, "acceleration": .15,
    "curvature": .11, "stabilizer": .09,
}
SCENES = ("linear", "accelerating", "jerk", "turning", "bounce")
FEATURES = 11
SHARE = .28


def identity(node):
    if node not in NODES:
        raise ValueError("unknown specialist identity")
    role, replica = node.rsplit("-", 1)
    return role, int(replica)


def stable_seed(*parts):
    token = "|".join(map(str, parts)).encode()
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "big")


def scene(rng):
    mode = rng.choice(SCENES)
    x, y = rng.uniform(-2, 2), rng.uniform(-2, 2)
    vx, vy = rng.uniform(-.9, .9), rng.uniform(-.9, .9)
    ax, ay = rng.uniform(-.16, .16), rng.uniform(-.16, .16)
    jx, jy = rng.uniform(-.07, .07), rng.uniform(-.07, .07)
    if mode == "linear":
        return [[x + vx*t, y + vy*t] for t in range(5)]
    if mode == "accelerating":
        return [[x + vx*t + ax*t*t/2, y + vy*t + ay*t*t/2]
                for t in range(5)]
    if mode == "jerk":
        return [[x + vx*t + ax*t*t/2 + jx*t**3/6,
                 y + vy*t + ay*t*t/2 + jy*t**3/6] for t in range(5)]
    if mode == "turning":
        angle, radius = rng.uniform(-3.14, 3.14), rng.uniform(.3, .9)
        omega = rng.uniform(.13, .55)
        return [[x + .2*vx*t + radius*(math.cos(angle+omega*t)-math.cos(angle)),
                 y + .2*vy*t + radius*(math.sin(angle+omega*t)-math.sin(angle))]
                for t in range(5)]
    # A bounce after the second observation, including its consequences.
    return [[x + vx*(t if t <= 2 else 4-t), y + vy*t]
            for t in range(5)]


def sensor(points, node, rng):
    role, replica = identity(node)
    sigma = ROLE_NOISE[role] * (.91 + .06*replica)
    bias = [rng.gauss(0, sigma/3) for _ in (0, 1)]
    return [[pos[k] + bias[k] + rng.gauss(0, sigma) for k in (0, 1)]
            for pos in points[:4]]


def features(track, node):
    role, replica = identity(node)
    if (len(track) != 4 or any(len(p) != 2 for p in track) or
        any(not math.isfinite(z) for p in track for z in p)):
        raise ValueError("expected four finite 2D observations")
    d = [track[3][j] - track[2][j] for j in (0, 1)]
    prev = [track[2][j] - track[1][j] for j in (0, 1)]
    a = [d[j]-prev[j] for j in (0, 1)]
    avg = [(track[3][j] - track[0][j])/3 for j in (0, 1)]
    sx, sa, sm, sc = ROLE_MASKS[role]
    variety = 1. + .04*(replica-1.5)
    curvature = prev[0]*d[1] - prev[1]*d[0]
    z = [sx*d[0]*variety, sx*d[1]*variety,
         sa*a[0], sa*a[1], sm*avg[0], sm*avg[1],
         math.tanh(2*a[0]), math.tanh(2*a[1]),
         sc*math.tanh(curvature), sc*min(2., math.hypot(*a)), 1.]
    return z


def predict(track, node, readout):
    z = features(track, node)
    if len(readout) != 2 or any(len(w) != FEATURES for w in readout):
        raise ValueError("invalid readout shape")
    result = [track[-1][axis] + sum(wi*zi for wi, zi in zip(readout[axis], z))
              for axis in (0, 1)]
    if any(not math.isfinite(v) for v in result):
        raise ValueError("nonfinite prediction")
    return result


def learn(node):
    rng = random.Random(stable_seed("train", node))
    sx = ROLE_MASKS[identity(node)[0]][0] * (
        1. + .04*(identity(node)[1]-1.5))
    params = [[0.]*FEATURES for _ in range(2)]
    params[0][0] = 1./sx
    params[1][1] = 1./sx
    for epoch in range(3):
        for _ in range(440):
            points = scene(rng)
            observed = sensor(points, node, rng)
            z = features(observed, node)
            forecast = predict(observed, node, params)
            for j in (0, 1):
                error = max(-4., min(4., forecast[j]-points[4][j]))
                for k in range(FEATURES):
                    params[j][k] = max(-3., min(3.,
                        params[j][k] - .012 * error * z[k]))
    losses = []
    for _ in range(90):
        points = scene(rng)
        obs = sensor(points, node, rng)
        losses.append(mesh.squared_error(predict(obs, node, params), points[4]))
    return {"role": identity(node)[0], "replica": identity(node)[1],
            "readout": params, "validation_mse": statistics.fmean(losses)}


def prepare(seed, out):
    if not isinstance(seed, int) or not 0 <= seed < 2**31:
        raise ValueError("invalid deterministic scene seed")
    models = {node: learn(node) for node in NODES}
    points = scene(random.Random(seed))
    tracks = {node: sensor(points, node,
                random.Random(stable_seed("observation", seed, node)))
              for node in NODES}
    weights = {"schema": SCHEMA, "seed": seed, "nodes": models}
    observation = {"schema": SCHEMA, "seed": seed, "tracks": tracks}
    truth = {"schema": SCHEMA, "seed": seed, "future_position": points[4]}
    dest = Path(out)
    mesh.save(dest/"weights.json", weights)
    mesh.save(dest/"observation.json", observation)
    mesh.save(dest/"truth.json", truth)


def _validate_packet(message, observation, stage):
    if (message.get("schema") != SCHEMA or
        message.get("seed") != observation["seed"] or
        message.get("stage") != stage or
        message.get("worker_input_sha256") != mesh.digest(observation) or
        set(message.get("predictions", {})) != set(NODES) or
        set(message.get("routing", {})) != set(NODES)):
        raise ValueError("invalid all-to-all feedback packet")


def run_node(node, stage, observation, weights, message=None):
    if node not in NODES or stage not in (1, 2):
        raise ValueError("invalid worker or stage")
    if (observation.get("schema") != SCHEMA or weights.get("schema") != SCHEMA or
        observation.get("seed") != weights.get("seed") or
        set(observation.get("tracks", {})) != set(NODES) or
        set(weights.get("nodes", {})) != set(NODES)):
        raise ValueError("scene or specialist population mismatch")
    forecast = predict(observation["tracks"][node], node,
                       weights["nodes"][node]["readout"])
    prev_digest = None
    if stage == 2:
        if message is None:
            raise ValueError("stage 2 needs round 1 messages")
        _validate_packet(message, observation, 1)
        peers = message["routing"][node]
        if (set(peers) != set(NODES)-{node} or
            any(not isinstance(v, (int,float)) or not math.isfinite(v) or
                v < 0 for v in peers.values()) or
            abs(sum(peers.values())-1.) > 1e-6):
            raise ValueError("invalid 19-peer topology")
        neighbor = [sum(peers[n]*message["predictions"][n][j]
                        for n in peers) for j in (0, 1)]
        forecast = [(1-SHARE)*forecast[j]+SHARE*neighbor[j] for j in (0, 1)]
        prev_digest = mesh.digest(message)
    return {"schema": SCHEMA, "seed": observation["seed"], "stage": stage,
            "node": node,
            "model_sha256": mesh.digest(weights["nodes"][node]),
            "input_sha256": mesh.digest(observation),
            "message_sha256": prev_digest,
            "embedding": features(observation["tracks"][node], node),
            "prediction": forecast}


def integrate(stage, weights, folder, previous=None):
    if stage not in (1, 2) or set(weights.get("nodes", {})) != set(NODES):
        raise ValueError("invalid stage or model population")
    if stage == 2 and (previous is None or
                       previous.get("stage") != 1 or
                       previous.get("seed") != weights["seed"]):
        raise ValueError("round 2 needs verified round 1 packet")
    files = sorted(Path(folder).glob("node-*.json"))
    if len(files) != 20:
        raise ValueError(f"expected 20 worker receipts, found {len(files)}")
    receipt = [mesh.load(p) for p in files]
    by_node = {r.get("node"): r for r in receipt}
    if (len(by_node) != 20 or set(by_node) != set(NODES) or
        any(r.get("node") not in p.name for p,r in zip(files, receipt))):
        raise ValueError("duplicate/missing/spoofed worker receipts")
    for r in receipt:
        n = r["node"]
        if (r.get("schema") != SCHEMA or r.get("stage") != stage or
            r.get("seed") != weights["seed"] or
            r.get("model_sha256") != mesh.digest(weights["nodes"][n]) or
            (stage == 1 and r.get("message_sha256") is not None) or
            (stage == 2 and r.get("message_sha256") != mesh.digest(previous)) or
            len(r.get("prediction", [])) != 2 or
            len(r.get("embedding", [])) != FEATURES or
            any(not isinstance(v,(int,float)) or not math.isfinite(v)
                for v in r["prediction"]+r["embedding"])):
            raise ValueError("invalid node receipt or feedback ancestry")
    obs_hash = {r.get("input_sha256") for r in receipt}
    if len(obs_hash) != 1:
        raise ValueError("worker input digests disagree")
    if stage == 2 and obs_hash != {previous["worker_input_sha256"]}:
        raise ValueError("feedback and observation digests disagree")
    pred = {n: by_node[n]["prediction"] for n in NODES}
    center = [statistics.fmean(pred[n][j] for n in NODES) for j in (0, 1)]
    raw = {}
    for n in NODES:
        prior = 1./max(1e-6, weights["nodes"][n]["validation_mse"])
        raw[n] = prior/(1.+2.*mesh.squared_error(pred[n], center))
    total = sum(raw.values())
    attention = {n: raw[n]/total for n in NODES}
    routing = {}
    for receiver in NODES:
        others = {n: raw[n] for n in NODES if n != receiver}
        subtotal = sum(others.values())
        routing[receiver] = {n: others[n]/subtotal for n in others}
    consensus = [sum(attention[n]*pred[n][j] for n in NODES)
                 for j in (0, 1)]
    return {"schema": SCHEMA, "stage": stage, "seed": weights["seed"],
            "node_count": len(NODES), "neighbors_per_node": len(NODES)-1,
            "predictions": pred, "routing": routing,
            "attention": attention, "consensus": consensus,
            "worker_input_sha256": next(iter(obs_hash))}


def evaluate(round1, round2, truth, weights):
    if (round1.get("schema") != SCHEMA or round2.get("schema") != SCHEMA or
        truth.get("schema") != SCHEMA or
        any(item.get("seed") != weights["seed"]
            for item in (round1, round2, truth)) or
        (round1.get("stage"), round2.get("stage")) != (1, 2) or
        any(item.get("node_count") != 20 for item in (round1, round2))):
        raise ValueError("evaluation requires two consistent 20-node waves")
    target = truth["future_position"]
    losses = {n: mesh.squared_error(round1["predictions"][n], target)
              for n in NODES}
    first = mesh.squared_error(round1["consensus"], target)
    second = mesh.squared_error(round2["consensus"], target)
    checkpoint = json.loads(json.dumps(weights))
    for n in NODES:
        checkpoint["nodes"][n]["validation_mse"] = (
            .8*weights["nodes"][n]["validation_mse"] + .2*losses[n])
    checkpoint["learned_routing"] = round2["routing"]
    return ({
        "schema": SCHEMA, "seed": weights["seed"],
        "physical_workers_per_round": 20, "inference_waves": 2,
        "inference_job_executions": 40, "peers_per_worker": 19,
        "best_individual_mse": min(losses.values()),
        "round1_consensus_mse": first,
        "round2_consensus_mse": second,
        "recurrent_feedback_helped": second < first,
        "round2_better_than_best_node": second < min(losses.values()),
        "status": "MEASURED_SYNTHETIC_EXPERIMENT_NOT_REAL_TIME_BRAIN",
        "model_sha256": mesh.digest(weights),
    }, checkpoint)


def local_proof(seed, out):
    output = Path(out)
    prepare(seed, output/"state")
    state = output/"state"
    weights = mesh.load(state/"weights.json")
    obs = mesh.load(state/"observation.json")
    truth = mesh.load(state/"truth.json")
    r1 = output/"round1-receipts"
    for node in NODES:
        mesh.save(r1/f"node-{node}.json", run_node(node, 1, obs, weights))
    pkt1 = integrate(1, weights, r1)
    r2 = output/"round2-receipts"
    for node in NODES:
        mesh.save(r2/f"node-{node}.json", run_node(node, 2, obs, weights, pkt1))
    pkt2 = integrate(2, weights, r2, pkt1)
    result, checkpoint = evaluate(pkt1, pkt2, truth, weights)
    for path, obj in (("round1.json", pkt1), ("round2.json", pkt2),
                      ("metrics.json", result),
                      ("candidate-checkpoint.json", checkpoint)):
        mesh.save(output/path, obj)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--seed", type=int, default=314159)
    prep.add_argument("--out", required=True)
    node = sub.add_parser("node")
    node.add_argument("--name", choices=NODES, required=True)
    node.add_argument("--stage", type=int, choices=(1,2), required=True)
    node.add_argument("--observations", required=True)
    node.add_argument("--weights", required=True)
    node.add_argument("--messages")
    node.add_argument("--out", required=True)
    integrate_cmd = sub.add_parser("integrate")
    integrate_cmd.add_argument("--stage", type=int, choices=(1,2), required=True)
    integrate_cmd.add_argument("--weights", required=True)
    integrate_cmd.add_argument("--outputs", required=True)
    integrate_cmd.add_argument("--messages")
    integrate_cmd.add_argument("--out", required=True)
    ev = sub.add_parser("evaluate")
    ev.add_argument("--round1", required=True)
    ev.add_argument("--round2", required=True)
    ev.add_argument("--truth", required=True)
    ev.add_argument("--weights", required=True)
    ev.add_argument("--out", required=True)
    lc = sub.add_parser("local-proof")
    lc.add_argument("--seed", type=int, default=314159)
    lc.add_argument("--out", required=True)
    args = p.parse_args()
    if args.command == "prepare":
        prepare(args.seed, args.out)
    elif args.command == "node":
        mesh.save(args.out, run_node(
            args.name, args.stage, mesh.load(args.observations),
            mesh.load(args.weights),
            mesh.load(args.messages) if args.messages else None))
    elif args.command == "integrate":
        mesh.save(args.out, integrate(
            args.stage, mesh.load(args.weights), args.outputs,
            mesh.load(args.messages) if args.messages else None))
    elif args.command == "evaluate":
        report, child = evaluate(mesh.load(args.round1), mesh.load(args.round2),
                                 mesh.load(args.truth), mesh.load(args.weights))
        mesh.save(args.out, report)
        mesh.save(Path(args.out).with_name("candidate-checkpoint.json"), child)
    else:
        print(json.dumps(local_proof(args.seed, args.out), sort_keys=True))


if __name__ == "__main__":
    main()

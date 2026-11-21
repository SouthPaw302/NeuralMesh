"""Phase 2.3 — learn when *not* to trust peer feedback, test across seeds.

The Phase 2.2 specialists and peer-correction readout remain fixed. A small
input-dependent gate controls feedback strength. Train and validation labels
are used in fit; the sealed test targets enter only in evaluate. Stdlib only.
"""
from __future__ import annotations

import argparse
import math
import statistics
from pathlib import Path

from neural_mesh import mesh, phase2, phase22

SCHEMA = "neural-mesh-phase-2.3"
FEATURES = 8
DEFAULT_SEEDS = (64013, 74129, 85021, 91153, 104729)


def _check(router, rows):
    if router.get("schema") != SCHEMA or router.get("round1_sha256") != mesh.digest(rows):
        raise ValueError("selective router / observations mismatch")
    if router.get("frozen_before_test") is not True:
        raise ValueError("selective routing is not frozen")


def _vector(row, base_router):
    """Unlabelled context from peer neural embeddings, disagreement, correction."""
    speed, acceleration, jerk, curvature, inconsistency = [], [], [], [], []
    for name in mesh.NODES:
        z = phase22._finite_vector(row["embeddings"][name], 9)
        a, b, c = mesh.MASKS[name]
        v = [z[0]/a, z[1]/a]
        ac = [z[2]/b, z[3]/b]
        mean = [z[4]/c, z[5]/c]
        j = [3*(mean[k] - v[k] + ac[k]) for k in (0, 1)]
        magv = math.hypot(*v)
        speed.append(magv)
        acceleration.append(math.hypot(*ac))
        jerk.append(math.hypot(*j))
        curvature.append(abs(v[0]*ac[1]-v[1]*ac[0]) / (0.1 + magv))
        inconsistency.append(math.hypot(v[0]-mean[0],v[1]-mean[1]))
    preds = row["predictions"]
    centroid = [sum(preds[n][k] for n in mesh.NODES)/6 for k in (0, 1)]
    dispersion = sum(mesh.squared_error(preds[n], centroid) for n in mesh.NODES)/6
    delta = _delta(row, base_router)
    vector = [
        1.0,
        sum(speed)/6,
        sum(acceleration)/6,
        sum(jerk)/6,
        sum(curvature)/6,
        sum(inconsistency)/6,
        math.sqrt(dispersion),
        math.hypot(*delta),
    ]
    if any(not math.isfinite(v) for v in vector):
        raise ValueError("non-finite context feature")
    return vector


def _baseline(row, base_router):
    return phase22.contextual_predict(row, base_router, use_feedback=False)


def _delta(row, base_router):
    beta = base_router["feedback"]["beta"]
    if beta <= 0.:
        return [0., 0.]
    d = phase22.correction(row, base_router["gate"], base_router["feedback"])
    return [beta*d[k] for k in (0, 1)]


def _scale_features(v, mean, sd):
    return [1.] + [max(-4., min(4., (v[i]-mean[i])/sd[i]))
                   for i in range(1, FEATURES)]


def _sigmoid(x):
    if x < -25:
        return 0.
    if x > 25:
        return 1.
    return 1./(1.+math.exp(-x))


def activation(vector, selected):
    kind = selected["kind"]
    if kind == "off":
        return 0.
    if kind == "full":
        return 1.
    if kind != "gated":
        raise ValueError("invalid gate type")
    x = _scale_features(vector, selected["mean"], selected["std"])
    return _sigmoid(sum(a*b for a,b in zip(selected["theta"], x)))


def _examples(rows, target, base_router):
    return [(_vector(row, base_router), _baseline(row, base_router),
             _delta(row, base_router), target[row["id"]], row["id"])
            for row in rows]


def _normalization(samples):
    mean = [0.]*FEATURES
    std = [1.]*FEATURES
    for j in range(1, FEATURES):
        values = [v[0][j] for v in samples]
        mean[j] = statistics.fmean(values)
        sd = statistics.pstdev(values)
        std[j] = max(0.05, sd)
    return mean, std


def _fit_gate(samples, mean, std, ridge, init, epochs=260):
    features = [_scale_features(item[0], mean, std) for item in samples]
    theta = [float(init)] + [0.]*(FEATURES-1)
    count = len(samples)
    for epoch in range(epochs):
        grad = [0.]*FEATURES
        for i, (_, baseline, delta, target, _) in enumerate(samples):
            g = _sigmoid(sum(theta[j]*features[i][j] for j in range(FEATURES)))
            pred = [baseline[k] + g*delta[k] for k in (0, 1)]
            slope = sum((pred[k]-target[k])*delta[k] for k in (0, 1))*g*(1-g)/count
            for j in range(FEATURES):
                grad[j] += slope*features[i][j]
        lr = 6./(1.+epoch/250.)
        for j in range(FEATURES):
            theta[j] -= lr*(grad[j]+(ridge*theta[j] if j else 0.))
            theta[j] = max(-8.,min(8.,theta[j]))
    return theta


def _loss(samples, selected):
    total = 0.
    for v, base, delta, target, _ in samples:
        g = activation(v, selected)
        forecast = [base[k]+g*delta[k] for k in (0, 1)]
        total += mesh.squared_error(forecast, target)
    return total/len(samples)


def fit(rows, base_router, trainval):
    cfg = rows["config_sha256"]
    phase2._check_schema(rows, cfg)
    phase2._check_labels(trainval, cfg, ("train", "validation"))
    if base_router.get("schema") != phase22.SCHEMA or base_router.get("round1_sha256") != mesh.digest(rows):
        raise ValueError("Phase 2.2 frozen router mismatch")
    train, val = rows["batches"]["train"], rows["batches"]["validation"]
    if set(trainval["labels"]["train"]) != {row["id"] for row in train} or set(
        trainval["labels"]["validation"]) != {row["id"] for row in val}:
        raise ValueError("training/validation ids do not match scenes")
    tr = _examples(train, trainval["labels"]["train"], base_router)
    vl = _examples(val, trainval["labels"]["validation"], base_router)
    mean, std = _normalization(tr)
    candidates = [{"kind": "off"}, {"kind": "full"}]
    for ridge in (0.0001, 0.001, 0.01, 0.05):
        for init in (-1.,1.):
            theta = _fit_gate(tr, mean, std, ridge, init)
            candidates.append({"kind": "gated", "theta": theta,
                               "mean": mean, "std": std, "ridge": ridge,
                               "init": init})
    scored = [{"candidate": c, "validation_mse": _loss(vl, c)} for c in candidates]
    best = min(scored, key=lambda v:v["validation_mse"])
    return {"schema": SCHEMA, "config_sha256": cfg,
            "round1_sha256": mesh.digest(rows), "base_router_sha256": mesh.digest(base_router),
            "train_scenes": len(train), "validation_scenes": len(val),
            "selected": best["candidate"], "validation_mse": best["validation_mse"],
            "candidate_scores": [{"kind": entry["candidate"]["kind"],
                                  "ridge": entry["candidate"].get("ridge"),
                                  "init": entry["candidate"].get("init"),
                                  "mse": entry["validation_mse"]}
                                 for entry in scored],
            "frozen_before_test": True}


def _score_seed(rows, base_router, selector, testlabels):
    _check(selector, rows)
    phase2._check_labels(testlabels, rows["config_sha256"], ("test",))
    if selector["base_router_sha256"] != mesh.digest(base_router):
        raise ValueError("base router hash mismatch")
    test = rows["batches"]["test"]
    targets = testlabels["labels"]["test"]
    if {r["id"] for r in test} != set(targets):
        raise ValueError("test target ids do not match")
    per = {key: [] for key in ("phase21_static", "phase22_context", "phase22_full_feedback", "phase23_selective")}
    regimes = {reg: {key: [] for key in per} for reg in phase2.REGIMES}
    activity = {reg: [] for reg in phase2.REGIMES}
    for row in test:
        target = targets[row["id"]]
        context = _baseline(row, base_router)
        correction = _delta(row, base_router)
        strength = activation(_vector(row, base_router), selector["selected"])
        selected = [context[k]+strength*correction[k] for k in (0, 1)]
        candidates = {
            "phase21_static": phase2._predict_final(
                row["predictions"],
                base_router["static_router"]["learned"]["weights"],
                base_router["static_router"]["learned"]["gamma"],
                base_router["static_router"]["learned"]["alpha"]),
            "phase22_context": context,
            "phase22_full_feedback": [context[k]+correction[k] for k in (0,1)],
            "phase23_selective": selected,
        }
        for key, p in candidates.items():
            error = mesh.squared_error(p, target)
            per[key].append(error)
            regimes[row["regime"]][key].append(error)
        activity[row["regime"]].append(strength)
    metrics = {key: statistics.fmean(errors) for key, errors in per.items()}
    return {
        "test_scenes": len(test), "test_mse": metrics,
        "regime_mse": {reg:{key:statistics.fmean(values) for key,values in data.items()}
                      for reg,data in regimes.items()},
        "regime_feedback_strength": {reg:statistics.fmean(values) for reg,values in activity.items()},
        "mean_feedback_strength": statistics.fmean(v for values in activity.values() for v in values),
        "paired_selective_minus_full": phase2._paired_bootstrap(
            per["phase23_selective"], per["phase22_full_feedback"]),
        "paired_selective_minus_static": phase2._paired_bootstrap(
            per["phase23_selective"], per["phase21_static"]),
        "selective_beats_full": metrics["phase23_selective"] < metrics["phase22_full_feedback"],
        "selective_beats_static": metrics["phase23_selective"] < metrics["phase21_static"],
    }


def train(seed, counts, out):
    """For offline smoke only; CI uses prepare/fit/evaluate with isolated artifacts."""
    out = Path(out)
    phase2.prepare(seed, counts, out/"dataset")
    fit_seed(out/"dataset/public.json", out/"dataset/weights.json",
             out/"dataset/trainval-labels.json", out/"trained")


def fit_seed(public_file, weights_file, trainval_file, out):
    public, weights, trainval = mesh.load(public_file), mesh.load(weights_file), mesh.load(trainval_file)
    import tempfile
    with tempfile.TemporaryDirectory() as temp:
        receipts = Path(temp)
        for node in mesh.NODES:
            mesh.save(receipts/f"node-{node}.json", phase22.observe(node, public, weights))
        rows = phase22.collect(public, weights, receipts)
    base = phase22.fit(rows, weights, trainval)
    selector = fit(rows, base, trainval)
    out = Path(out)
    mesh.save(out/"rows.json", rows)
    mesh.save(out/"phase22-router.json", base)
    mesh.save(out/"phase23-selector.json", selector)
    return selector


def evaluate_seed(seed, rows_file, phase22_file, selector_file, testlabels_file, out):
    rows = mesh.load(rows_file)
    base = mesh.load(phase22_file)
    selector = mesh.load(selector_file)
    labels = mesh.load(testlabels_file)
    report = _score_seed(rows, base, selector, labels)
    result = {"schema": SCHEMA, "seed": seed, "config_sha256": rows["config_sha256"],
              "train_scenes": selector["train_scenes"],
              "validation_scenes": selector["validation_scenes"],
              **report,
              "selected_gate": selector["selected"]["kind"],
              "selected_gate_regularization": selector["selected"].get("ridge"),
              "router_sha256": mesh.digest(base),
              "selector_sha256": mesh.digest(selector),
              "status": "INDEPENDENT_SYNTHETIC_HOLDOUT_NOT_REAL_WORLD_NEURAL_INTELLIGENCE"}
    mesh.save(out, result)
    return result


def aggregate(directory, seeds, out):
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ValueError("require at least three unique seed ids")
    directory = Path(directory)
    paths = sorted(directory.glob("seed-*.json"))
    if len(paths) != len(seeds):
        raise ValueError(f"expected exactly {len(seeds)} seed reports")
    reports = [mesh.load(path) for path in paths]
    by_seed = {r.get("seed"):r for r in reports}
    if set(by_seed) != set(seeds) or len(by_seed) != len(reports):
        raise ValueError("missing, duplicate, or unexpected seed report")
    keys = ("phase21_static", "phase22_context", "phase22_full_feedback", "phase23_selective")
    cfgs = set()
    for seed in seeds:
        result=by_seed[seed]
        if result.get("schema")!=SCHEMA or result.get("status")!="INDEPENDENT_SYNTHETIC_HOLDOUT_NOT_REAL_WORLD_NEURAL_INTELLIGENCE":
            raise ValueError("incompatible or unverified seed report")
        if result["test_scenes"] < 4 or result["validation_scenes"] < 4 or result["train_scenes"] < 4:
            raise ValueError("unexpected dataset size")
        if set(result["test_mse"])!=set(keys) or any(
            not isinstance(v, (int,float)) or not math.isfinite(v) or v < 0
            for v in result["test_mse"].values()
        ):
            raise ValueError("missing or nonfinite test metric")
        cfgs.add(result["config_sha256"])
    if len(cfgs)!=len(seeds):
        raise ValueError("dataset configurations are not independent")
    averages = {k:statistics.fmean(by_seed[s]["test_mse"][k] for s in seeds) for k in keys}
    diff = [by_seed[s]["test_mse"]["phase23_selective"] -
            by_seed[s]["test_mse"]["phase22_full_feedback"] for s in seeds]
    result={"schema":SCHEMA,"seeds":list(seeds),"seed_count":len(seeds),
            "mean_test_mse":averages,"per_seed":[by_seed[s] for s in seeds],
            "wins_over_full":sum(delta < 0 for delta in diff),
            "wins_over_static":sum(by_seed[s]["selective_beats_static"] for s in seeds),
            "mean_selective_minus_full":statistics.fmean(diff),
            "min_selective_minus_full":min(diff),"max_selective_minus_full":max(diff),
            "selected_gate_types":{k:sum(by_seed[s]["selected_gate"]==k for s in seeds)
                                   for k in ("off","full","gated")},
            "status":"CROSS_SEED_SYNTHETIC_BENCHMARK_NO_CLAIM_OF_REAL_WORLD_GENERALIZATION"}
    mesh.save(out,result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sp=p.add_subparsers(dest="command",required=True)
    prep=sp.add_parser("prepare")
    prep.add_argument("--seed",type=int,required=True)
    prep.add_argument("--train-count",type=int,default=512)
    prep.add_argument("--validation-count",type=int,default=128)
    prep.add_argument("--test-count",type=int,default=256)
    prep.add_argument("--out",required=True)
    fitcmd=sp.add_parser("fit")
    fitcmd.add_argument("--public",required=True)
    fitcmd.add_argument("--weights",required=True)
    fitcmd.add_argument("--trainval",required=True)
    fitcmd.add_argument("--out",required=True)
    ev=sp.add_parser("evaluate")
    ev.add_argument("--seed",type=int,required=True)
    ev.add_argument("--rows",required=True)
    ev.add_argument("--phase22-router",required=True)
    ev.add_argument("--selector",required=True)
    ev.add_argument("--test-labels",required=True)
    ev.add_argument("--out",required=True)
    agg=sp.add_parser("aggregate")
    agg.add_argument("--results",required=True)
    agg.add_argument("--seeds",default=",".join(map(str,DEFAULT_SEEDS)))
    agg.add_argument("--out",required=True)
    args=p.parse_args()
    if args.command=="prepare":
        phase2.prepare(args.seed, {"train":args.train_count,
                                  "validation":args.validation_count,
                                  "test":args.test_count}, args.out)
    elif args.command=="fit":
        result=fit_seed(args.public,args.weights,args.trainval,args.out)
        print({"selected":result["selected"]["kind"],
               "validation_mse":result["validation_mse"]})
    elif args.command=="evaluate":
        result=evaluate_seed(args.seed,args.rows,args.phase22_router,
                             args.selector,args.test_labels,args.out)
        print(result["test_mse"])
    else:
        result=aggregate(args.results,tuple(int(x) for x in args.seeds.split(",")),args.out)
        print(result["mean_test_mse"])


if __name__=="__main__":
    main()

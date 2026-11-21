"""Six-frame temporal observation experiment, retaining the frozen four-frame model.

The old predictor still uses the final four observations to predict the next
frame. A new *observational* router may use six history frames to estimate
when that predictor's frozen physics adapter is trustworthy.

No user records, video, automatic memory promotion or test labels in fitting.
"""
from __future__ import annotations

import argparse
import math
import random
import statistics
from pathlib import Path

from neural_mesh import mesh, phase2, phase24, phase25, transfer26

SCHEMA="neuralmesh-temporal-six-v1"
ID=phase2.REGIMES
KNOWN=("bounce","swirl")
UNKNOWN=("stop_start","late_impulse")
TRAIN_CLASSES=ID+KNOWN
FAULTS=phase24.FAULTS
PARENT_SHA=phase25.PARENT_SHA
ADAPTER_SHA="f0223f0d6f3cdc2813d583ffb938c64ed11db46579f381d54c9207908f579a97"
ID_TEST=(31221,31222,31223)
KNOWN_TEST=(51221,51222,51223)
UNKNOWN_TEST=(51231,51232,51233)


def trajectory(rng,regime):
    x,y=rng.uniform(-2,2),rng.uniform(-2,2)
    vx,vy=rng.uniform(-.9,.9),rng.uniform(-.9,.9)
    ax,ay=rng.uniform(-.16,.16),rng.uniform(-.16,.16)
    if regime=="linear":ax=ay=0.
    if regime=="jerk":
        jx,jy=rng.uniform(-.07,.07),rng.uniform(-.07,.07)
    else:jx=jy=0.
    if regime=="turning":
        omega=rng.uniform(.13,.42)
        angle=rng.uniform(-math.pi,math.pi)
        radius=rng.uniform(.25,.9)
        return [[x+.35*vx*t+radius*(math.cos(angle+omega*t)-math.cos(angle)),
                 y+.35*vy*t+radius*(math.sin(angle+omega*t)-math.sin(angle))]
                for t in range(7)]
    if regime in ("linear","accelerating","jerk"):
        return [[x+vx*t+ax*t*t/2+jx*t**3/6,
                 y+vy*t+ay*t*t/2+jy*t**3/6] for t in range(7)]
    if regime=="bounce":
        return [[x+vx*(t if t<=2 else 4-t), y+vy*t] for t in range(7)]
    if regime=="stop_start":
        return [[x+vx*(t if t<=2 else 2 if t==3 else 2+1.6*(t-3)),
                 y+vy*(t if t<=2 else 2 if t==3 else 2+1.6*(t-3))]
                for t in range(7)]
    if regime=="swirl":
        omega=rng.uniform(.5,1.15)
        angle=rng.uniform(-math.pi,math.pi)
        radius=rng.uniform(.4,1.2)
        return [[x+radius*math.cos(angle+omega*t)+.12*vx*t,
                 y+radius*math.sin(angle+omega*t)+.12*vy*t] for t in range(7)]
    if regime=="late_impulse":
        points=[[x+vx*t+ax*t*t/2,y+vy*t+ay*t*t/2] for t in range(7)]
        points[6][0]+=rng.choice((-1,1))*rng.uniform(.3,.85)
        points[6][1]+=rng.choice((-1,1))*rng.uniform(.3,.85)
        return points
    raise ValueError("unknown motion family")


def _six_sensor(points,node,rng):
    sigma=mesh.NOISE[node]
    bias=[rng.gauss(0,sigma/3),rng.gauss(0,sigma/3)]
    return [[p[k]+bias[k]+rng.gauss(0,sigma) for k in range(2)]
            for p in points[:6]]


def make_data(seed,regimes,count,out):
    if (not isinstance(seed,int) or seed<0 or seed>=2**31 or
        not regimes or len(set(regimes))!=len(regimes) or
        not set(regimes).issubset(set(ID+KNOWN+UNKNOWN)) or
        count<16 or count%len(regimes)):
        raise ValueError("invalid experimental seed, regimes or count")
    scenes=[]
    labels={}
    for i in range(count):
        regime=regimes[i%len(regimes)]
        scene_seed=phase2._seed(seed,i,regime)
        points=trajectory(random.Random(scene_seed),regime)
        history={n:_six_sensor(points,n,random.Random(
                  phase2._seed(seed,i,regime,n,"six-sensor")))
                 for n in mesh.NODES}
        name=f"temporal-{seed}-{i:05d}"
        scenes.append({"id":name,"regime":regime,
                       "tracks":{n:history[n][-4:] for n in mesh.NODES},
                       "history":history})
        labels[name]=points[6]
    public={"schema":SCHEMA,"seed":seed,"regimes":list(regimes),"scenes":scenes}
    targets={"schema":SCHEMA,"seed":seed,"public_sha256":mesh.digest(public),
             "targets":labels}
    mesh.save(Path(out)/"public.json",public)
    mesh.save(Path(out)/"targets.json",targets)
    return public


def _load(public_path,labels_path,regimes):
    public=mesh.load(public_path)
    labels=mesh.load(labels_path)
    if (public.get("schema")!=SCHEMA or labels.get("schema")!=SCHEMA or
        public.get("regimes")!=list(regimes) or
        public["seed"]!=labels.get("seed") or
        labels.get("public_sha256")!=mesh.digest(public)):
        raise ValueError("invalid six-frame dataset provenance")
    samples=public["scenes"]
    if len(samples)!=len(set(s["id"] for s in samples)) or (
        {s["id"] for s in samples}!=set(labels.get("targets",{}))):
        raise ValueError("unmatched temporal target labels")
    return samples,labels["targets"],str(public["seed"])


def _sources(parent_path,adapter_path):
    parent=phase24.restore_memory(parent_path)
    adapter=mesh.load(adapter_path)
    if parent["sha256"]!=PARENT_SHA or adapter.get("sha256")!=ADAPTER_SHA or (
        adapter["sha256"]!=mesh.digest(adapter.get("payload")) or
        adapter["payload"].get("parent_sha256")!=parent["sha256"]):
        raise ValueError("original model fingerprint mismatch")
    return parent,adapter


def _med(values):
    return statistics.median(values)


def _features(scene,damaged,memory,four):
    # Aggregate only consistent live observations, not the injected corrupt
    # specialist's predictions. All six historical positions are label-free.
    active=phase24._active(damaged,memory["reliability"]["selected"]["cutoff"])
    pos=[[_med(scene["history"][n][i][k] for n in active) for k in (0,1)]
         for i in range(6)]
    steps=[[pos[i+1][k]-pos[i][k] for k in (0,1)] for i in range(5)]
    velocities=[math.hypot(*x) for x in steps]
    accelerations=[[steps[i+1][k]-steps[i][k] for k in (0,1)]
                   for i in range(4)]
    accel_mags=[math.hypot(*v) for v in accelerations]
    angles=[abs(math.atan2(
             steps[i][0]*steps[i+1][1]-steps[i][1]*steps[i+1][0],
             steps[i][0]*steps[i+1][0]+steps[i][1]*steps[i+1][1]))
            for i in range(4)]
    # Extra history reveals whether a turn/acceleration persists or is noisy.
    extra=[
        statistics.fmean(accel_mags),
        statistics.pstdev(accel_mags),
        statistics.fmean(angles),
        statistics.pstdev(angles),
        max(angles),
        max(velocities)-min(velocities),
        math.dist(steps[0],steps[-1]),
        math.dist(accelerations[0],accelerations[-1]),
        sum(a>b for a,b in zip(velocities[1:],velocities[:-1]))/4,
    ]
    if any(not math.isfinite(x) for x in extra):
        raise ValueError("invalid temporal motion features")
    return list(four)+extra


def _example(scene,target,memory,old_adapter,fault,index):
    base=transfer26._example(scene,target,memory,fault,index,"temporal")
    parent=transfer26.predict_example(base,None)
    upgraded=transfer26.predict_example(base,old_adapter)
    row=phase24.make_row(scene,memory["models"])
    damaged=phase24.inject(row,fault,index)
    four=list(base["features"])+[
        min(3.,math.dist(parent,upgraded)),
        abs(base["features"][2]*base["features"][3])]
    six=_features(scene,damaged,memory,four)
    return {"id":scene["id"],"regime":scene["regime"],"fault":fault,
            "four":four,"six":six,"parent":parent,
            "adapter":upgraded,"truth":target}


def _examples(scenes,labels,memory,old_adapter):
    rows=[]
    failures={f:0 for f in FAULTS}
    for i,scene in enumerate(scenes):
        for fault in FAULTS:
            try:
                rows.append(_example(scene,labels[scene["id"]],memory,
                                     old_adapter,fault,i))
            except ValueError as exc:
                if str(exc) not in (
                    "not enough consistent specialist predictions",
                    "insufficient healthy nodes"):
                    raise
                failures[fault]+=1
    return rows,failures


def _centroids(train,feature_key):
    length=len(train[0][feature_key])
    means=[statistics.fmean(x[feature_key][j] for x in train)
           for j in range(length)]
    scales=[max(.05,statistics.pstdev(x[feature_key][j] for x in train))
            for j in range(length)]
    centers={}
    for family in TRAIN_CLASSES:
        group=[x for x in train if x["regime"]==family]
        if not group:raise ValueError("missing supervised history class")
        centers[family]=[
            statistics.fmean((x[feature_key][j]-means[j])/scales[j]
                             for x in group) for j in range(length)]
    return {"feature_key":feature_key,"mean":means,"scale":scales,
            "centers":centers}


def classify(ex,model):
    v=ex[model["feature_key"]]
    z=[(v[j]-model["mean"][j])/model["scale"][j] for j in range(len(v))]
    dist={reg:sum((z[j]-center[j])**2 for j in range(1,len(z)))
          for reg,center in model["centers"].items()}
    pred=min(dist,key=dist.get)
    return pred,dist


def _accuracy(examples,model):
    confusion={r:{c:0 for c in TRAIN_CLASSES} for r in TRAIN_CLASSES}
    for e in examples:
        actual=e["regime"]
        if actual not in TRAIN_CLASSES:continue
        guessed,_=classify(e,model)
        confusion[actual][guessed]+=1
    valid=sum(sum(c.values()) for c in confusion.values())
    return {"overall":sum(confusion[c][c] for c in TRAIN_CLASSES)/valid,
            "by_regime":{r:confusion[r][r]/max(1,sum(confusion[r].values()))
                         for r in TRAIN_CLASSES},
            "confusion":confusion}


def _trust(ex,policy,classifier):
    if policy is None:return 0.
    name,dist=classify(ex,classifier)
    # A predicted fragile family is vetoed only if the nearest class has
    # sufficient margin over competing classes.
    ranks=sorted(dist.values())
    margin=(ranks[1]-ranks[0])/(.1+ranks[1])
    risky=name in ("linear","turning") and margin>=policy["margin"]
    return policy["veto"] if risky else policy["nominal"]


def _score(examples,policy,model):
    faults={k:[] for k in FAULTS}
    regimes={}
    for e in examples:
        t=_trust(e,policy,model)
        out=[e["parent"][k]+t*(e["adapter"][k]-e["parent"][k])
             for k in (0,1)]
        err=mesh.squared_error(out,e["truth"])
        faults[e["fault"]].append(err)
        regimes.setdefault(e["regime"],[]).append(err)
    return {"mse":statistics.fmean(v for arr in faults.values() for v in arr),
            "faults":{f:statistics.fmean(v) for f,v in faults.items()},
            "regimes":{r:statistics.fmean(v) for r,v in regimes.items()}}


def train(parent_file,adapter_file,train_public,train_labels,
          val_public,val_labels,out):
    parent,adapter=_sources(parent_file,adapter_file)
    memory=parent["payload"]
    base=adapter["payload"]["model"]
    train_s,train_t,train_seed=_load(train_public,train_labels,TRAIN_CLASSES)
    valid_s,valid_t,val_seed=_load(val_public,val_labels,TRAIN_CLASSES)
    if train_seed==val_seed:raise ValueError("same train and validation seed")
    cases,_=_examples(train_s,train_t,memory,base)
    validation,_=_examples(valid_s,valid_t,memory,base)
    models={"four":_centroids(cases,"four"),
            "six":_centroids(cases,"six")}
    comparison={kind:_accuracy(validation,model)
                for kind,model in models.items()}
    parent_val=_score(validation,None,models["six"])
    original_val=_score(validation,{"veto":1.,"nominal":1.,"margin":0.},
                        models["six"])
    best=None
    candidates=[]
    for feature in ("four","six"):
        for margin in (0.,.08,.18,.32,.5):
            for nominal in (.75,1.):
                for veto in (0.,.2,.4):
                    policy={"feature":feature,"margin":margin,
                            "nominal":nominal,"veto":veto}
                    result=_score(validation,policy,models[feature])
                    okay=(result["mse"]<parent_val["mse"]*.999 and
                          all(result["regimes"][r]<=parent_val["regimes"][r]*
                              (1.0 if r in ("linear","turning") else 1.02)
                              for r in ID) and
                          all(result["faults"][f]<=parent_val["faults"][f]*1.01
                              for f in FAULTS) and
                          all(result["regimes"][r]<=parent_val["regimes"][r]*.90
                              for r in KNOWN))
                    score=result["mse"]/parent_val["mse"]
                    candidates.append({"feature":feature,"margin":margin,
                                       "nominal":nominal,"veto":veto,
                                       "validation_mse":result["mse"],
                                       "linear_mse":result["regimes"]["linear"],
                                       "turning_mse":result["regimes"]["turning"],
                                       "eligible":okay})
                    if okay and (best is None or score<best[0]):
                        best=(score,policy,result)
    state={"schema":SCHEMA,"parent_sha256":parent["sha256"],
           "adapter_sha256":adapter["sha256"],
           "classifiers":models,
           "selected_policy":best[1] if best else None,
           "inference_uses_only_observed_history":True,
           "test_labels_not_seen":True,"auto_promote":False}
    wrapper={"payload":state,"sha256":mesh.digest(state)}
    report={"schema":SCHEMA,"parent_sha256":parent["sha256"],
            "adapter_sha256":adapter["sha256"],
            "temporal_sha256":wrapper["sha256"],
            "classification_validation":comparison,
            "parent_validation":parent_val,
            "adapter_validation":original_val,
            "selected_validation":best[2] if best else parent_val,
            "validation_pass":best is not None,
            "candidate_is_parent_fallback":best is None,
            "selection_trials":candidates,
            "no_test_label_access":True}
    root=Path(out)
    mesh.save(root/"parent.json",parent)
    mesh.save(root/"adapter.json",adapter)
    mesh.save(root/"temporal.json",wrapper)
    mesh.save(root/"training-decision.json",report)
    return report


def evaluate(parent_file,adapter_file,temporal_file,decision_file,
             id_sets,known_sets,unknown_sets,out):
    if any(len(x)!=3 for x in (id_sets,known_sets,unknown_sets)):
        raise ValueError("need three independent cohorts of each kind")
    parent,adapter=_sources(parent_file,adapter_file)
    state=mesh.load(temporal_file)
    decision=mesh.load(decision_file)
    if (state.get("sha256")!=mesh.digest(state.get("payload")) or
        state["payload"].get("schema")!=SCHEMA or
        state["payload"]["parent_sha256"]!=parent["sha256"] or
        state["payload"]["adapter_sha256"]!=adapter["sha256"] or
        decision.get("temporal_sha256")!=state["sha256"]):
        raise ValueError("temporal model integrity mismatch")
    selected=state["payload"]["selected_policy"]
    if bool(selected)!=bool(decision.get("validation_pass")):
        raise ValueError("invalid validation decision")
    memory=parent["payload"]
    base=adapter["payload"]["model"]
    outcomes=[]
    seen=set()
    for kind,sets,allowed in (
        ("id",id_sets,ID),("known",known_sets,KNOWN),
        ("unseen",unknown_sets,UNKNOWN)):
        for public_path,labels_path in sets:
            rows,labels,seed=_load(public_path,labels_path,allowed)
            if (kind,seed) in seen:raise ValueError("duplicate sealed data seed")
            seen.add((kind,seed))
            samples,failures=_examples(rows,labels,memory,base)
            model=state["payload"]["classifiers"][
                selected["feature"] if selected else "six"]
            outcomes.append({"kind":kind,"seed":seed,
                             "requested_scenes":len(rows),
                             "scored_fault_cases":len(samples),
                             "abstentions":failures,
                             "parent":_score(samples,None,model),
                             "adapter":_score(samples,{"veto":1.,"nominal":1.,"margin":0.},model),
                             "temporal":_score(samples,selected,model)})
    summary={}
    for group in ("id","known","unseen"):
        subset=[r for r in outcomes if r["kind"]==group]
        summary[group]={
            "parent":statistics.fmean(r["parent"]["mse"] for r in subset),
            "adapter":statistics.fmean(r["adapter"]["mse"] for r in subset),
            "temporal":statistics.fmean(r["temporal"]["mse"] for r in subset),
            "seed_wins":sum(r["temporal"]["mse"]<r["parent"]["mse"] for r in subset)}
        allowed=ID if group=="id" else KNOWN if group=="known" else UNKNOWN
        summary[group]["regimes"]={reg:{
            name:statistics.fmean(r[name]["regimes"][reg] for r in subset)
            for name in ("parent","adapter","temporal")} for reg in allowed}
    id_group=summary["id"];known=summary["known"];unseen=summary["unseen"]
    id_ok=id_group["temporal"]<id_group["parent"] and all(
        id_group["regimes"][reg]["temporal"]<=id_group["regimes"][reg]["parent"]*(
            1.0000001 if reg in ("linear","turning") else 1.02) for reg in ID)
    known_ok=known["temporal"]<=known["parent"]*.85 and (
        known["temporal"]<=known["adapter"]*1.10)
    unseen_ok=unseen["temporal"]<=unseen["parent"]*.995 and (
        unseen["temporal"]<=unseen["adapter"]*1.02)
    fault_ok=all(
        e["temporal"]["faults"][f] <= e["parent"]["faults"][f]*1.02
        for e in outcomes if e["kind"]=="id" for f in FAULTS)
    passed=bool(selected) and id_ok and known_ok and unseen_ok and fault_ok
    result={"schema":SCHEMA,
            "status":"SIX_FRAME_RETENTION_SUPPORTED" if passed else (
                "NO_VALIDATED_SIX_FRAME_GATE" if selected is None
                else "BLOCKED_TEMPORAL_GENERALIZATION"),
            "parent_sha256":parent["sha256"],
            "temporal_sha256":state["sha256"],
            "decision_sha256":mesh.digest(decision),
            "classification_validation":decision["classification_validation"],
            "id_retention_pass":id_ok,"known_transfer_pass":known_ok,
            "unseen_transfer_pass":unseen_ok,"fault_safety_pass":fault_ok,
            "summary":summary,"per_seed":outcomes,
            "total_requested_scenes":sum(x["requested_scenes"] for x in outcomes),
            "manual_review_required":True,"original_memory_unmodified":True}
    mesh.save(Path(out)/"evaluation.json",result)
    mesh.save(Path(out)/"quality-gate.json",{
        "schema":SCHEMA,"status":result["status"],
        "evaluation_sha256":mesh.digest(result),
        "never_auto_promote":True})
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest="command",required=True)
    p=commands.add_parser("prepare")
    p.add_argument("--seed",type=int,required=True)
    p.add_argument("--regimes",required=True)
    p.add_argument("--count",type=int,required=True)
    p.add_argument("--out",required=True)
    p=commands.add_parser("train")
    for name in ("parent","adapter","train-public","train-targets",
                 "val-public","val-targets","out"):
        p.add_argument("--"+name,required=True)
    p=commands.add_parser("evaluate")
    for name in ("parent","adapter","temporal","decision","out"):
        p.add_argument("--"+name,required=True)
    for name in ("id-set","known-set","unseen-set"):
        p.add_argument("--"+name,action="append",default=[])
    args=parser.parse_args()
    if args.command=="prepare":
        make_data(args.seed,tuple(args.regimes.split(",")),args.count,args.out)
    elif args.command=="train":
        print(train(args.parent,args.adapter,args.train_public,args.train_targets,
                    args.val_public,args.val_targets,args.out)["validation_pass"])
    else:
        def pairs(values):
            out=[]
            for v in values:
                chunks=v.split(",")
                if len(chunks)!=2:raise ValueError("expected public,target pair")
                out.append(tuple(chunks))
            return out
        report=evaluate(args.parent,args.adapter,args.temporal,args.decision,
                        pairs(args.id_set),pairs(args.known_set),
                        pairs(args.unseen_set),args.out)
        print({"status":report["status"],"summary":report["summary"]})


if __name__=="__main__":
    main()

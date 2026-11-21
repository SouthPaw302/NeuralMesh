"""Phase 2.6 cross-domain transfer: frozen parent + trainable physics-expert router.

Train a gate on prior in-distribution and *two* newly exposed motion families;
evaluate on fresh seeds of those families AND on two never-trained motion
families. The parent remains unchanged and is an explicit no-op fallback.

The router sees only four observed 2D frames and six specialist packets.
It NEVER receives target positions or the true regime name at inference.
"""
from __future__ import annotations

import argparse
import copy
import math
import random
import statistics
import tempfile
from pathlib import Path

from neural_mesh import mesh, phase2, phase24, phase25

SCHEMA = "neuralmesh-crossdomain-v1"
KNOWN = ("bounce", "swirl")
UNKNOWN = ("stop_start", "late_impulse")
EXPERTS = ("parent", "velocity", "acceleration", "rotation", "restart")
FEATURE_DIM = 7
FAULTS = phase24.FAULTS
IID_SEEDS = (26121, 26122, 26123)
KNOWN_SEEDS = (46121, 46122, 46123)
UNKNOWN_SEEDS = (46131, 46132, 46133)
TRAIN_SEEDS = {"new":26111, "replay":64013, "novel":46111, "novel_validation":46112}


def make_subset(seed, count, regimes, out):
    """Export only selected motion regimes (no hidden unseen-family examples)."""
    if not isinstance(seed,int) or seed < 0 or count < 16 or count%2:
        raise ValueError("invalid dataset seed or count")
    if not regimes or len(regimes)!=2 or len(set(regimes))!=2 or (
        not set(regimes).issubset(set(phase24.REGIMES))):
        raise ValueError("require exactly two distinct supported regimes")
    with tempfile.TemporaryDirectory() as tmp:
        phase24.prepare_ood(seed,2*count,tmp)
        original=mesh.load(Path(tmp)/"public.json")
        sealed=mesh.load(Path(tmp)/"sealed-targets.json")
    chosen=[s for s in original["scenes"] if s["regime"] in regimes]
    if len(chosen)!=count:
        raise ValueError("unexpected regime balance")
    public={"schema":phase24.OOD_SCHEMA,"seed":seed,
            "allowed_regimes":list(regimes),"scenes":chosen}
    labels={"schema":phase24.OOD_SCHEMA,"seed":seed,
            "public_sha256":mesh.digest(public),
            "targets":{s["id"]:sealed["targets"][s["id"]] for s in chosen}}
    mesh.save(Path(out)/"public.json",public)
    mesh.save(Path(out)/"targets.json",labels)
    return public


def _load_ood(public_file, labels_file, allowed):
    public=mesh.load(public_file)
    labels=mesh.load(labels_file)
    if (public.get("schema")!=phase24.OOD_SCHEMA or
        labels.get("schema")!=phase24.OOD_SCHEMA or
        public.get("seed")!=labels.get("seed") or
        labels.get("public_sha256")!=mesh.digest(public) or
        set(s["regime"] for s in public["scenes"])!=set(allowed) or
        not all(s["regime"] in allowed for s in public["scenes"])):
        raise ValueError("OOD dataset provenance or domain partition mismatch")
    ids=[s["id"] for s in public["scenes"]]
    if len(ids)!=len(set(ids)) or set(ids)!=set(labels.get("targets",{})):
        raise ValueError("mismatched OOD labels")
    return public,labels


def _id_rows(public_file, label_file, split):
    public,labels=mesh.load(public_file),mesh.load(label_file)
    phase2._check_schema(public,public.get("config_sha256"))
    phase2._check_labels(labels,public["config_sha256"],("train","validation") if split in (
        "train","validation") else ("test",))
    records=public["scenes"][split]
    targets=labels["labels"][split]
    if {s["id"] for s in records}!=set(targets):
        raise ValueError("ID scene label mismatch")
    return records,targets,public["config_sha256"]


def _median(x):
    return statistics.median(x)


def _rotate(vector,angle):
    cs,sn=math.cos(angle),math.sin(angle)
    return [cs*vector[0]-sn*vector[1],sn*vector[0]+cs*vector[1]]


def _physics(row, scene, memory):
    policy=memory["reliability"]["selected"]
    active=phase24._active(row,policy["cutoff"])
    tracks=scene["tracks"]
    if set(tracks)!=set(mesh.NODES):
        raise ValueError("all original specialist observations required")
    track=[[_median(tracks[n][i][k] for n in active) for k in (0,1)]
           for i in range(4)]
    d3=[track[3][k]-track[2][k] for k in (0,1)]
    d2=[track[2][k]-track[1][k] for k in (0,1)]
    speed3=math.hypot(*d3)
    speed2=math.hypot(*d2)
    accel=[d3[k]-d2[k] for k in (0,1)]
    angle=math.atan2(d2[0]*d3[1]-d2[1]*d3[0],
                     d2[0]*d3[0]+d2[1]*d3[1])
    angle=max(-1.25,min(1.25,angle))
    turned=_rotate(d3,angle)
    velocity=[track[3][k]+d3[k] for k in (0,1)]
    accelerating=[track[3][k]+d3[k]+accel[k] for k in (0,1)]
    circular=[track[3][k]+turned[k] for k in (0,1)]
    restart=[track[3][k]+(1.6*d2[k] if speed3 < .3*speed2 else d3[k])
             for k in (0,1)]
    return (velocity,accelerating,circular,restart),[
        1., min(4.,speed3), min(3.,math.hypot(*accel)),
        min(3.,abs(angle)), min(3.,speed3/(.1+speed2)),
        max(-1.,min(1.,sum(d2[k]*d3[k] for k in (0,1))/
                   (.01+speed2*speed3))),
        min(3.,statistics.pstdev([row["predictions"][n][0] for n in active]))
    ]


def _example(scene,target,memory,fault,index,domain):
    row=phase24.make_row(scene,memory["models"])
    damaged=phase24.inject(row,fault,index)
    policy=memory["reliability"]["selected"]
    baseline,_=phase24.robust_prediction(
        damaged,memory,policy["cutoff"],policy["feedback_scale"])
    alternatives,signals=_physics(damaged,scene,memory)
    forecasts=[baseline]+list(alternatives)
    return {"id":scene["id"],"regime":scene["regime"],"domain":domain,
            "fault":fault,"features":signals,"experts":forecasts,
            "truth":target}


def _make_examples(scenes,targets,memory,domain):
    if not scenes or {s["id"] for s in scenes}!=set(targets):
        raise ValueError("empty or misaligned dataset")
    out=[]
    for i,scene in enumerate(scenes):
        target=phase2._pair(targets[scene["id"]])
        for fault in FAULTS:
            out.append(_example(scene,target,memory,fault,i,domain))
    return out


def _standardizer(examples):
    means=[0.]*FEATURE_DIM
    scales=[1.]*FEATURE_DIM
    for j in range(1,FEATURE_DIM):
        nums=[e["features"][j] for e in examples]
        means[j]=statistics.fmean(nums)
        scales[j]=max(.05,statistics.pstdev(nums))
    return means,scales


def _normal_features(features,means,scales):
    return [1.]+[max(-4.,min(4.,(features[j]-means[j])/scales[j]))
                 for j in range(1,FEATURE_DIM)]


def _softmax(logits):
    m=max(logits)
    exp=[math.exp(max(-50.,z-m)) for z in logits]
    sm=sum(exp)
    return [z/sm for z in exp]


def _weights(theta,features,means,scales):
    x=_normal_features(features,means,scales)
    logits=[sum(theta[e][j]*x[j] for j in range(FEATURE_DIM))
            for e in range(len(EXPERTS))]
    return _softmax(logits)


def predict_example(example,model):
    baseline=example["experts"][0]
    if model is None:
        return baseline[:]
    att=_weights(model["theta"],example["features"],model["means"],model["scales"])
    blended=[sum(att[i]*example["experts"][i][k] for i in range(len(EXPERTS)))
             for k in (0,1)]
    strength=model["strength"]
    return [baseline[k]+strength*(blended[k]-baseline[k]) for k in (0,1)]


def _fit(id_examples,ood_examples,id_weight,l2,epochs=180):
    combined=id_examples+ood_examples
    means,scales=_standardizer(combined)
    theta=[[0.]*FEATURE_DIM for _ in EXPERTS]
    theta[0][0]=2.
    for i in range(1,len(EXPERTS)):
        theta[i][0]=-.75
    data=[(e,_normal_features(e["features"],means,scales),
           (id_weight/len(id_examples) if e["domain"]=="id"
            else (1-id_weight)/len(ood_examples))) for e in combined]
    for epoch in range(epochs):
        grads=[[0.]*FEATURE_DIM for _ in EXPERTS]
        for example,features,importance in data:
            logits=[sum(theta[t][j]*features[j] for j in range(FEATURE_DIM))
                    for t in range(len(EXPERTS))]
            attention=_softmax(logits)
            blended=[sum(attention[t]*example["experts"][t][k]
                         for t in range(len(EXPERTS))) for k in (0,1)]
            for t in range(len(EXPERTS)):
                derivative=attention[t]*sum(
                    (blended[k]-example["truth"][k])*
                    (example["experts"][t][k]-blended[k]) for k in (0,1))
                for j in range(FEATURE_DIM):
                    grads[t][j]+=importance*derivative*features[j]
        lr=1.0/(1.+epoch/160.)
        for t in range(len(EXPERTS)):
            for j in range(FEATURE_DIM):
                value=grads[t][j]+l2*theta[t][j] if j else grads[t][j]
                theta[t][j]=max(-6.,min(6.,theta[t][j]-lr*value))
    return {"theta":theta,"means":means,"scales":scales,
            "id_weight":id_weight,"regularization":l2,"strength":1.0}


def _metrics(examples,model):
    by_fault={n:[] for n in FAULTS}
    by_regime={}
    for ex in examples:
        pred=predict_example(ex,model)
        loss=mesh.squared_error(pred,ex["truth"])
        by_fault[ex["fault"]].append(loss)
        by_regime.setdefault(ex["regime"],[]).append(loss)
    return {"mse":statistics.fmean(v for arr in by_fault.values() for v in arr),
            "faults":{n:statistics.fmean(v) for n,v in by_fault.items()},
            "regimes":{n:statistics.fmean(v) for n,v in by_regime.items()}}


def train(parent_path, replay_public, replay_labels, fresh_public,fresh_labels,
          ood_train_public,ood_train_labels,ood_val_public,ood_val_labels,out):
    parent=phase24.restore_memory(parent_path)
    if parent["sha256"]!=phase25.PARENT_SHA:
        raise ValueError("unexpected parent state identity")
    past_train, past_labels,_=_id_rows(replay_public,replay_labels,"train")
    fresh_train,fresh_train_labels,_=_id_rows(fresh_public,fresh_labels,"train")
    past_val,past_val_labels,past_cfg=_id_rows(replay_public,replay_labels,"validation")
    fresh_val,fresh_val_labels,fresh_cfg=_id_rows(fresh_public,fresh_labels,"validation")
    if past_cfg==fresh_cfg:
        raise ValueError("replay data must differ from fresh ID data")
    ood_train,ood_train_targets=_load_ood(ood_train_public,ood_train_labels,KNOWN)
    ood_val,ood_val_targets=_load_ood(ood_val_public,ood_val_labels,KNOWN)
    if ood_train["seed"]==ood_val["seed"]:
        raise ValueError("training and validation OOD datasets not independent")
    memory=parent["payload"]
    id_train=_make_examples(past_train,past_labels,memory,"id")
    id_train+=_make_examples(fresh_train,fresh_train_labels,memory,"id")
    id_val=_make_examples(past_val,past_val_labels,memory,"id")
    id_val+=_make_examples(fresh_val,fresh_val_labels,memory,"id")
    novel_train=_make_examples(ood_train["scenes"],ood_train_targets["targets"],memory,"ood")
    novel_val=_make_examples(ood_val["scenes"],ood_val_targets["targets"],memory,"ood")
    baseline_id=_metrics(id_val,None)
    baseline_ood=_metrics(novel_val,None)
    best=None
    trials=[]
    for id_weight in (.55,.75,.9):
        for ridge in (.0005,.01):
            trained=_fit(id_train,novel_train,id_weight,ridge)
            for strength in (.25,.5,.75,1.):
                candidate={**trained,"strength":strength}
                id_metrics=_metrics(id_val,candidate)
                ood_metrics=_metrics(novel_val,candidate)
                good=(id_metrics["mse"]<=baseline_id["mse"]*1.005 and
                      ood_metrics["mse"]<baseline_ood["mse"]*.98 and
                      all(id_metrics["faults"][f]<=baseline_id["faults"][f]*1.01
                          for f in FAULTS) and
                      all(id_metrics["regimes"][reg]<=baseline_id["regimes"][reg]*1.03
                          for reg in baseline_id["regimes"]))
                score=(ood_metrics["mse"]/baseline_ood["mse"] +
                       id_metrics["mse"]/baseline_id["mse"]*.75)
                trials.append({"id_weight":id_weight,"l2":ridge,"strength":strength,
                               "id_mse":id_metrics["mse"],
                               "known_ood_mse":ood_metrics["mse"],
                               "eligible":good})
                if good and (best is None or score<best[0]):
                    best=(score,candidate,id_metrics,ood_metrics)
    payload={"schema":SCHEMA,"parent_sha256":parent["sha256"],
             "model":best[1] if best else None,
             "selection":"validation-only",
             "trained_domains":list(KNOWN),
             "unseen_domains_not_trainable":list(UNKNOWN),
             "parent_unchanged":True,
             "automatic_memory_promotion":False}
    wrapper={"payload":payload,"sha256":mesh.digest(payload)}
    decision={"schema":SCHEMA,"parent_sha256":parent["sha256"],
              "adapter_sha256":wrapper["sha256"],
              "validation_pass":best is not None,
              "baseline_id_validation":baseline_id,
              "baseline_known_validation":baseline_ood,
              "selected_id_validation":best[2] if best else baseline_id,
              "selected_known_validation":best[3] if best else baseline_ood,
              "trials":trials,
              "training_id_scenes":len(id_train)//len(FAULTS),
              "training_known_ood_scenes":len(novel_train)//len(FAULTS),
              "validation_id_scenes":len(id_val)//len(FAULTS),
              "validation_known_ood_scenes":len(novel_val)//len(FAULTS),
              "no_test_target_access":True,
              "regime_labels_not_in_inference":True}
    mesh.save(Path(out)/"parent.json",parent)
    mesh.save(Path(out)/"adapter.json",wrapper)
    mesh.save(Path(out)/"decision.json",decision)
    return decision


def _id_test(public_path,labels_path,memory):
    scenes,targets,cfg=_id_rows(public_path,labels_path,"test")
    return _make_examples(scenes,targets,memory,"id"),cfg


def _ood_test(public_path,labels_path,memory,allowed):
    public,labels=_load_ood(public_path,labels_path,allowed)
    return _make_examples(public["scenes"],labels["targets"],memory,"ood"),str(public["seed"])


def evaluate(parent_path,adapter_path,decision_path,id_sets,known_sets,unknown_sets,out):
    if any(len(x)!=3 for x in (id_sets,known_sets,unknown_sets)):
        raise ValueError("requires three independently seeded datasets in every category")
    parent=phase24.restore_memory(parent_path)
    adapter=mesh.load(adapter_path)
    decision=mesh.load(decision_path)
    if (adapter.get("sha256")!=mesh.digest(adapter.get("payload")) or
        adapter["payload"].get("schema")!=SCHEMA or
        decision.get("schema")!=SCHEMA or
        decision.get("adapter_sha256")!=adapter["sha256"] or
        decision.get("parent_sha256")!=parent["sha256"] or
        adapter["payload"].get("parent_sha256")!=parent["sha256"]):
        raise ValueError("adapter/parent/evaluation provenance invalid")
    if parent["sha256"]!=phase25.PARENT_SHA:
        raise ValueError("unexpected frozen memory parent")
    model=adapter["payload"]["model"]
    if bool(model)!=bool(decision["validation_pass"]):
        raise ValueError("adapter validation decision mismatch")
    all_results=[]
    seen=set()
    for category,items in (("id",id_sets),("known",known_sets),("unseen",unknown_sets)):
        for public,targets in items:
            if category=="id":
                examples,uid=_id_test(public,targets,parent["payload"])
            else:
                examples,uid=_ood_test(public,targets,parent["payload"],
                                        KNOWN if category=="known" else UNKNOWN)
            if (category,uid) in seen:
                raise ValueError("repeated held-out cohort")
            seen.add((category,uid))
            all_results.append({"category":category,"id":uid,"scenes":len(examples)//3,
                                "parent":_metrics(examples,None),
                                "adapter":_metrics(examples,model)})
    summary={}
    for category in ("id","known","unseen"):
        rows=[x for x in all_results if x["category"]==category]
        old=statistics.fmean(x["parent"]["mse"] for x in rows)
        new=statistics.fmean(x["adapter"]["mse"] for x in rows)
        summary[category]={
            "parent_mse":old,"candidate_mse":new,
            "relative_change":new/old-1,
            "seed_wins":sum(x["adapter"]["mse"]<x["parent"]["mse"] for x in rows),
            "seeds":len(rows)}
    # Predeclare a high bar to avoid previous Phase 2.5's false promotion.
    # Improvement on known OOD alone is insufficient: unknown physics must
    # transfer without damage, and familiar scenes must be preserved.
    id_ok=(summary["id"]["relative_change"] <= 0.005 and
           all(x["adapter"]["faults"][f]<=x["parent"]["faults"][f]*1.015
               for x in all_results if x["category"]=="id" for f in FAULTS))
    known_ok=(summary["known"]["relative_change"]<=-.025 and
              summary["known"]["seed_wins"]>=2)
    unknown_ok=(summary["unseen"]["relative_change"]<=0 and
                all(x["adapter"]["faults"][f] <= x["parent"]["faults"][f]*1.025
                    for x in all_results if x["category"]=="unseen" for f in FAULTS))
    passed=bool(model) and id_ok and known_ok and unknown_ok
    status="TRANSFER_SUPPORTED" if passed else (
        "NO_VALIDATED_ADAPTER" if model is None else "BLOCKED_CROSS_DOMAIN_TRANSFER")
    report={"schema":SCHEMA,"status":status,
            "parent_sha256":parent["sha256"],"adapter_sha256":adapter["sha256"],
            "decision_sha256":mesh.digest(decision),
            "validation_pass":bool(model),
            "summary":summary,"id_preservation_pass":id_ok,
            "known_ood_transfer_pass":known_ok,
            "unseen_physics_transfer_pass":unknown_ok,
            "per_dataset":all_results,
            "manual_review_required":True,
            "checkpoint_was_not_modified":True,
            "test_scenarios":sum(x["scenes"] for x in all_results),
            "information_leakage_claim":"train jobs received no held-out test labels"}
    mesh.save(Path(out)/"evaluation.json",report)
    mesh.save(Path(out)/"gate.json",{
        "schema":SCHEMA,"status":status,"evaluation_sha256":mesh.digest(report),
        "adapter_sha256":adapter["sha256"],
        "parent_sha256":parent["sha256"],"auto_promote":False})
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="command",required=True)
    subset=sub.add_parser("subset")
    subset.add_argument("--seed",type=int,required=True)
    subset.add_argument("--count",type=int,required=True)
    subset.add_argument("--regimes",required=True)
    subset.add_argument("--out",required=True)
    traincmd=sub.add_parser("train")
    for n in ("parent","replay-public","replay-labels","fresh-public","fresh-labels",
              "ood-train-public","ood-train-labels","ood-val-public","ood-val-labels","out"):
        traincmd.add_argument("--"+n,required=True)
    ev=sub.add_parser("evaluate")
    for n in ("parent","adapter","decision","out"):
        ev.add_argument("--"+n,required=True)
    for n in ("id-set","known-set","unseen-set"):
        ev.add_argument("--"+n,action="append",default=[])
    a=p.parse_args()
    if a.command=="subset":
        result=make_subset(a.seed,a.count,tuple(a.regimes.split(",")),a.out)
        print({"seed":result["seed"],"scenes":len(result["scenes"])})
    elif a.command=="train":
        d=train(a.parent,a.replay_public,a.replay_labels,a.fresh_public,a.fresh_labels,
                a.ood_train_public,a.ood_train_labels,a.ood_val_public,
                a.ood_val_labels,a.out)
        print({"validation_pass":d["validation_pass"],"adapter":d["adapter_sha256"],
               "known_val":d["selected_known_validation"]["mse"],
               "id_val":d["selected_id_validation"]["mse"]})
    else:
        def pairs(items):
            result=[]
            for value in items:
                split=value.split(",")
                if len(split)!=2: raise ValueError("expected PUBLIC,TARGET path pair")
                result.append(tuple(split))
            return result
        report=evaluate(a.parent,a.adapter,a.decision,pairs(a.id_set),
                        pairs(a.known_set),pairs(a.unseen_set),a.out)
        print({"status":report["status"],"summary":report["summary"]})


if __name__=="__main__":
    main()

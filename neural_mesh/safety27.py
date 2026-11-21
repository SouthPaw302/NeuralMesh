"""Phase 2.6.1: learn a supervised safety gate around the frozen physics adapter.

This is a targeted repair for familiar linear and turning regressions. The
Phase 2.6 adapter and Phase 2.4 model remain byte-for-byte frozen. A bounded
secondary gate estimates how much to trust the existing adapter, using only
observable motion features. It is trained on fresh/known train partitions,
selected on fresh/known validation, and tested on NEW held-out seeds.

No test labels enter fitting, no regime labels enter inference, and no memory
or Git registry checkpoint is automatically promoted.
"""
from __future__ import annotations

import argparse
import copy
import math
import statistics
from pathlib import Path

from neural_mesh import mesh, phase2, phase24, phase25, transfer26

SCHEMA="neuralmesh-retention-safety-v1"
PARENT_SHA=phase25.PARENT_SHA
BASE_ADAPTER_SHA="f0223f0d6f3cdc2813d583ffb938c64ed11db46579f381d54c9207908f579a97"
FEATURE_COUNT=11
ID_TEST_SEEDS=(27221,27222,27223)
KNOWN_TEST_SEEDS=(47221,47222,47223)
UNSEEN_TEST_SEEDS=(47231,47232,47233)
FAULTS=phase24.FAULTS
ID_REGIMES=phase2.REGIMES


def _load_sources(parent_path,adapter_path):
    parent=phase24.restore_memory(parent_path)
    adapter=mesh.load(adapter_path)
    if parent["sha256"]!=PARENT_SHA:
        raise ValueError("unexpected protected parent fingerprint")
    if (adapter.get("sha256")!=BASE_ADAPTER_SHA or
        adapter["sha256"]!=mesh.digest(adapter.get("payload")) or
        adapter["payload"].get("schema")!=transfer26.SCHEMA or
        adapter["payload"].get("parent_sha256")!=parent["sha256"] or
        not adapter["payload"].get("model")):
        raise ValueError("frozen Phase 2.6 adapter fingerprint invalid")
    return parent,adapter


def _collect_id(public_file,labels_file,split,memory):
    scenes,labels,cfg=transfer26._id_rows(public_file,labels_file,split)
    return transfer26._make_examples(scenes,labels,memory,"id"),cfg


def _collect_ood(public_file,labels_file,allowed,memory):
    public,labels=transfer26._load_ood(public_file,labels_file,allowed)
    return transfer26._make_examples(
        public["scenes"],labels["targets"],memory,"ood"),str(public["seed"])


def _extended_features(ex,parent_prediction,adapted_prediction):
    base=ex["features"]
    if len(base)!=transfer26.FEATURE_DIM:
        raise ValueError("unexpected observed motion feature dimension")
    delta=[adapted_prediction[k]-parent_prediction[k] for k in range(2)]
    acceleration=base[2]
    angle=base[3]
    values=base+[acceleration**2,angle**2,acceleration*angle,
                 min(3.,math.hypot(*delta))]
    if len(values)!=FEATURE_COUNT or not all(math.isfinite(v) for v in values):
        raise ValueError("unsafe trust features")
    return values


def _candidate_example(ex,base_model):
    previous=transfer26.predict_example(ex,None)
    adapted=transfer26.predict_example(ex,base_model)
    return {"id":ex["id"],"regime":ex["regime"],"fault":ex["fault"],
            "features":_extended_features(ex,previous,adapted),
            "parent":previous,"adapter":adapted,"truth":ex["truth"]}


def _augment(examples,base_model):
    return [_candidate_example(x,base_model) for x in examples]


def _normalize(training):
    means=[0.]*FEATURE_COUNT
    scales=[1.]*FEATURE_COUNT
    for j in range(1,FEATURE_COUNT):
        z=[x["features"][j] for x in training]
        means[j]=statistics.fmean(z)
        scales[j]=max(.05,statistics.pstdev(z))
    return means,scales


def _feature(feat,mean,scale):
    return [1.]+[max(-4.,min(4.,(feat[i]-mean[i])/scale[i]))
                 for i in range(1,FEATURE_COUNT)]


def _sigmoid(z):
    if z>24.:return 1.
    if z<-24.:return 0.
    return 1./(1.+math.exp(-z))


def _trust(ex,policy):
    if policy is None:return 0.
    theta=policy["theta"]
    z=_feature(ex["features"],policy["mean"],policy["scale"])
    fraction=_sigmoid(sum(theta[j]*z[j] for j in range(FEATURE_COUNT)))
    return policy["floor"]+(1.-policy["floor"])*fraction


def prediction(ex,policy):
    trust=_trust(ex,policy)
    return [ex["parent"][k]+trust*(ex["adapter"][k]-ex["parent"][k])
            for k in (0,1)]


def _score(examples,policy):
    if not examples:
        raise ValueError("no motion observations")
    errors=[]
    by_fault={f:[] for f in FAULTS}
    by_regime={}
    activations={}
    for ex in examples:
        p=prediction(ex,policy)
        loss=mesh.squared_error(p,ex["truth"])
        errors.append(loss)
        by_fault[ex["fault"]].append(loss)
        by_regime.setdefault(ex["regime"],[]).append(loss)
        activations.setdefault(ex["regime"],[]).append(_trust(ex,policy))
    return {"mse":statistics.fmean(errors),
            "faults":{k:statistics.fmean(v) for k,v in by_fault.items()},
            "regimes":{k:statistics.fmean(v) for k,v in by_regime.items()},
            "trust_by_regime":{k:statistics.fmean(v) for k,v in activations.items()}}


def _fit(id_examples,known_examples,id_importance,ridge,init,floor,
         epochs=150):
    combined=id_examples+known_examples
    mean,scale=_normalize(combined)
    theta=[float(init)]+[0.]*(FEATURE_COUNT-1)
    examples=[]
    for ex in id_examples:
        # Prefer safety on historically fragile linear and turning regimes.
        factor={"linear":3.0,"turning":2.5,"accelerating":1.0,"jerk":1.0}[ex["regime"]]
        examples.append((ex,id_importance*factor/len(id_examples)))
    for ex in known_examples:
        examples.append((ex,(1.-id_importance)/len(known_examples)))
    norm=sum(w for _,w in examples)
    prepared=[(ex,w/norm,_feature(ex["features"],mean,scale)) for ex,w in examples]
    for epoch in range(epochs):
        grad=[0.]*FEATURE_COUNT
        for ex,importance,z in prepared:
            sig=_sigmoid(sum(theta[j]*z[j] for j in range(FEATURE_COUNT)))
            trust=floor+(1.-floor)*sig
            delta=[ex["adapter"][k]-ex["parent"][k] for k in (0,1)]
            forecast=[ex["parent"][k]+trust*delta[k] for k in (0,1)]
            slope=(sum((forecast[k]-ex["truth"][k])*delta[k] for k in (0,1))
                   * (1.-floor)*sig*(1.-sig)*importance)
            for j in range(FEATURE_COUNT):
                grad[j]+=slope*z[j]
        lr=2.5/(1.+epoch/180.)
        for j in range(FEATURE_COUNT):
            penalty=ridge*theta[j] if j else 0.
            theta[j]=max(-9.,min(9.,theta[j]-lr*(grad[j]+penalty)))
    return {"theta":theta,"mean":mean,"scale":scale,"floor":floor,
            "id_importance":id_importance,"ridge":ridge,"init":init}


def _validation_safe(parent_id,selected_id,parent_known,selected_known):
    if selected_id["mse"]>parent_id["mse"]*.999:
        return False
    for regime in ID_REGIMES:
        allowable=1.0 if regime in ("linear","turning") else 1.02
        if selected_id["regimes"][regime]>parent_id["regimes"][regime]*allowable:
            return False
    for f in FAULTS:
        if selected_id["faults"][f]>parent_id["faults"][f]*1.01:
            return False
    if selected_known["mse"]>parent_known["mse"]*.90:
        return False
    return True


def train(parent_path,adapter_path,id_public,id_labels,known_train_public,
          known_train_labels,known_val_public,known_val_labels,out):
    parent,adapter=_load_sources(parent_path,adapter_path)
    memory=parent["payload"]
    base=adapter["payload"]["model"]
    id_train,cfg=_collect_id(id_public,id_labels,"train",memory)
    id_validation,valcfg=_collect_id(id_public,id_labels,"validation",memory)
    if cfg!=valcfg:raise ValueError("mismatched train/validation cohort")
    known_train,seen=_collect_ood(known_train_public,known_train_labels,
                                   transfer26.KNOWN,memory)
    known_validation,valid_seed=_collect_ood(known_val_public,known_val_labels,
                                            transfer26.KNOWN,memory)
    if seen==valid_seed:
        raise ValueError("known motion training and validation seed overlap")
    fitted_id=_augment(id_train,base)
    fitted_known=_augment(known_train,base)
    validation_id=_augment(id_validation,base)
    validation_known=_augment(known_validation,base)
    baseline_id=_score(validation_id,None)
    baseline_known=_score(validation_known,None)
    base_policy={"theta":[100.]+[0.]*(FEATURE_COUNT-1),
                 "mean":[0.]*FEATURE_COUNT,
                 "scale":[1.]*FEATURE_COUNT,"floor":1.}
    adapter_id=_score(validation_id,base_policy)
    adapter_known=_score(validation_known,base_policy)
    best=None
    candidates=[]
    for importance in (.55,.75,.90):
        for ridge in (.001,.02):
            for init in (-1.,0.):
                for floor in (0.,.15):
                    model=_fit(fitted_id,fitted_known,importance,ridge,init,floor)
                    test_id=_score(validation_id,model)
                    test_known=_score(validation_known,model)
                    accepted=_validation_safe(
                        baseline_id,test_id,baseline_known,test_known)
                    candidates.append({"id_weight":importance,"ridge":ridge,
                                       "init":init,"floor":floor,
                                       "id_validation_mse":test_id["mse"],
                                       "known_validation_mse":test_known["mse"],
                                       "safe":accepted})
                    score=test_id["mse"]/baseline_id["mse"] + (
                        test_known["mse"]/baseline_known["mse"])
                    if accepted and (best is None or score<best[0]):
                        best=(score,model,test_id,test_known)
    state={"schema":SCHEMA,"parent_sha256":parent["sha256"],
           "prior_adapter_sha256":adapter["sha256"],
           "policy":best[1] if best else None,
           "policy_fitted_only_on_train":True,
           "selection_uses_validation_only":True,
           "model_memory_unchanged":True,
           "automatic_promotion":False}
    wrapper={"payload":state,"sha256":mesh.digest(state)}
    decision={"schema":SCHEMA,"parent_sha256":parent["sha256"],
              "prior_adapter_sha256":adapter["sha256"],
              "safety_sha256":wrapper["sha256"],
              "validation_pass":best is not None,
              "fallback_is_original_parent":best is None,
              "baseline_id_validation":baseline_id,
              "original_adapter_id_validation":adapter_id,
              "baseline_known_validation":baseline_known,
              "original_adapter_known_validation":adapter_known,
              "selected_id_validation":best[2] if best else baseline_id,
              "selected_known_validation":best[3] if best else baseline_known,
              "trials":candidates,
              "training_examples":{"id":len(id_train)//len(FAULTS),
                                   "known":len(known_train)//len(FAULTS)},
              "validation_examples":{"id":len(id_validation)//len(FAULTS),
                                     "known":len(known_validation)//len(FAULTS)},
              "no_heldout_labels_seen":True}
    target=Path(out)
    mesh.save(target/"parent.json",parent)
    mesh.save(target/"prior-adapter.json",adapter)
    mesh.save(target/"safety.json",wrapper)
    mesh.save(target/"decision.json",decision)
    return decision


def _all_three(examples,model,policy):
    parent=_score(examples,None)
    prev=_score(examples,{"theta":[100.]+[0.]*(FEATURE_COUNT-1),
                          "mean":[0.]*FEATURE_COUNT,
                          "scale":[1.]*FEATURE_COUNT,"floor":1.})
    safe=_score(examples,policy)
    return {"parent":parent,"adapter":prev,"protected":safe}


def _group(scenes):
    mean=lambda xs:statistics.fmean(xs)
    return {"parent_mse":mean(x["scores"]["parent"]["mse"] for x in scenes),
            "adapter_mse":mean(x["scores"]["adapter"]["mse"] for x in scenes),
            "protected_mse":mean(x["scores"]["protected"]["mse"] for x in scenes),
            "seed_wins_vs_parent":sum(
                x["scores"]["protected"]["mse"]<x["scores"]["parent"]["mse"]
                for x in scenes),
            "seed_wins_vs_adapter":sum(
                x["scores"]["protected"]["mse"]<x["scores"]["adapter"]["mse"]
                for x in scenes),
            "regimes":{reg:{
                name:mean(x["scores"][name]["regimes"][reg] for x in scenes)
                for name in ("parent","adapter","protected")}
                for reg in scenes[0]["scores"]["parent"]["regimes"]},
            "trust_by_regime":{reg:mean(
                x["scores"]["protected"]["trust_by_regime"][reg] for x in scenes)
                for reg in scenes[0]["scores"]["parent"]["regimes"]}}


def evaluate(parent_path,adapter_path,safety_path,decision_path,
             id_sets,known_sets,unknown_sets,out):
    if any(len(s)!=3 for s in (id_sets,known_sets,unknown_sets)):
        raise ValueError("require exactly 3 seeds for each tested domain")
    parent,prior=_load_sources(parent_path,adapter_path)
    safety=mesh.load(safety_path)
    decision=mesh.load(decision_path)
    if (safety.get("sha256")!=mesh.digest(safety.get("payload")) or
        safety["payload"].get("schema")!=SCHEMA or
        safety["payload"].get("parent_sha256")!=parent["sha256"] or
        safety["payload"].get("prior_adapter_sha256")!=prior["sha256"] or
        decision.get("schema")!=SCHEMA or
        decision.get("safety_sha256")!=safety["sha256"] or
        decision.get("parent_sha256")!=parent["sha256"] or
        decision.get("prior_adapter_sha256")!=prior["sha256"]):
        raise ValueError("safety policy lineage or digest mismatch")
    model=prior["payload"]["model"]
    policy=safety["payload"]["policy"]
    if bool(policy)!=bool(decision.get("validation_pass")):
        raise ValueError("invalid safety validation decision")
    results=[]
    seen=set()
    for domain,sets in (("id",id_sets),("known",known_sets),("unseen",unknown_sets)):
        for public_path,label_path in sets:
            if domain=="id":
                raw,uid=_collect_id(public_path,label_path,"test",parent["payload"])
            else:
                raw,uid=_collect_ood(public_path,label_path,
                                   transfer26.KNOWN if domain=="known" else transfer26.UNKNOWN,
                                   parent["payload"])
            if (domain,uid) in seen:
                raise ValueError("duplicate held-out dataset")
            seen.add((domain,uid))
            examples=_augment(raw,model)
            results.append({"domain":domain,"id":uid,
                            "scenes":len(examples)//len(FAULTS),
                            "scores":_all_three(examples,model,policy)})
    summary={domain:_group([r for r in results if r["domain"]==domain])
             for domain in ("id","known","unseen")}
    old=summary["id"]["regimes"]
    familiar_ok=(all(old[r]["protected"]<=old[r]["parent"]*(1.0000001 if r in
                    ("linear","turning") else 1.02) for r in ID_REGIMES) and
                 summary["id"]["protected_mse"] < summary["id"]["parent_mse"])
    retained_transfer=(summary["known"]["protected_mse"]<=
                        summary["known"]["parent_mse"]*.85 and
                       summary["known"]["protected_mse"]<=
                        summary["known"]["adapter_mse"]*1.10 and
                       summary["known"]["seed_wins_vs_parent"]>=2)
    novel_ok=(summary["unseen"]["protected_mse"]<=
              summary["unseen"]["parent_mse"]*.995 and
              summary["unseen"]["protected_mse"]<=
              summary["unseen"]["adapter_mse"]*1.02)
    fault_ok=all(
        r["scores"]["protected"]["faults"][f] <=
        r["scores"]["parent"]["faults"][f]*1.02
        for r in results if r["domain"]=="id" for f in FAULTS)
    passed=bool(policy) and familiar_ok and retained_transfer and novel_ok and fault_ok
    status="RETENTION_REPAIR_SUPPORTED" if passed else (
        "NO_VALIDATED_SAFETY_GATE" if policy is None else
        "BLOCKED_OLD_SKILL_OR_TRANSFER_REGRESSION")
    report={"schema":SCHEMA,"status":status,
            "parent_sha256":parent["sha256"],
            "prior_adapter_sha256":prior["sha256"],
            "safety_sha256":safety["sha256"],
            "decision_sha256":mesh.digest(decision),
            "retains_linear_and_turning":familiar_ok,
            "retains_known_transfer":retained_transfer,
            "retains_unseen_transfer":novel_ok,
            "fault_regression_guard":fault_ok,
            "summary":summary,"per_seed":results,
            "test_scenarios":sum(x["scenes"] for x in results),
            "separate_test_cohorts":len(results),
            "manual_review_required":True,
            "parent_checkpoint_unchanged":True}
    dest=Path(out)
    mesh.save(dest/"evaluation.json",report)
    mesh.save(dest/"promotion-gate.json",{
        "schema":SCHEMA,"status":status,
        "parent_sha256":parent["sha256"],
        "safety_sha256":safety["sha256"],
        "evaluation_sha256":mesh.digest(report),
        "never_autopromote":True})
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    train=sub.add_parser("train")
    for n in ("parent","adapter","id-public","id-labels",
              "known-train-public","known-train-labels",
              "known-val-public","known-val-labels","out"):
        train.add_argument("--"+n,required=True)
    ev=sub.add_parser("evaluate")
    for n in ("parent","adapter","safety","decision","out"):
        ev.add_argument("--"+n,required=True)
    for n in ("id-set","known-set","unseen-set"):
        ev.add_argument("--"+n,action="append",default=[])
    args=parser.parse_args()
    if args.command=="train":
        d=train(args.parent,args.adapter,args.id_public,args.id_labels,
                args.known_train_public,args.known_train_labels,
                args.known_val_public,args.known_val_labels,args.out)
        print({"validation_pass":d["validation_pass"],
               "id_validation":d["selected_id_validation"]["mse"],
               "known_validation":d["selected_known_validation"]["mse"],
               "safety_sha256":d["safety_sha256"]})
    else:
        def pairs(s):
            r=[]
            for item in s:
                v=item.split(",")
                if len(v)!=2:raise ValueError("expected public,target pair")
                r.append(tuple(v))
            return r
        report=evaluate(args.parent,args.adapter,args.safety,args.decision,
                        pairs(args.id_set),pairs(args.known_set),
                        pairs(args.unseen_set),args.out)
        print({"status":report["status"],"guards":{
            "linear_and_turning":report["retains_linear_and_turning"],
            "known":report["retains_known_transfer"],
            "unseen":report["retains_unseen_transfer"],
            "faults":report["fault_regression_guard"]},
            "mean":{k:{name:v[name] for name in
                  ("parent_mse","adapter_mse","protected_mse")}
                   for k,v in report["summary"].items()}})


if __name__=="__main__":
    main()

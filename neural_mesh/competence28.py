"""Phase 2.6.2 — a regime-blind competence tree for regression-safe transfer.

A supervised decision tree partitions *observable* kinematics and learns how
much of the frozen 2.6 physics correction to apply in each region. No true
regime name or future position is passed to inference. Train/validation
decide policy; fresh sealed targets are only scored in a later runner.

Memory, original physics adapter, and Git main are all read-only.
"""
from __future__ import annotations

import argparse
import copy
import math
import statistics
from pathlib import Path

from neural_mesh import mesh, phase2, phase24, safety27, transfer26

SCHEMA="neuralmesh-competence-tree-v1"
FAULTS=phase24.FAULTS
REGIMES=phase2.REGIMES
DEPTHS=(2,3,4)
MIN_LEAVES=(90,150)
RIDGES=(0.0001,0.002)
SCALES=(0.50,0.75,1.0)
TEST_ID=(28221,28222,28223)
TEST_KNOWN=(48221,48222,48223)
TEST_UNSEEN=(48231,48232,48233)


def _example_weight(example, domain, count):
    if domain=="id":
        multiplier={"linear":4.,"turning":3.,"accelerating":1.,
                    "jerk":1.}[example["regime"]]
        return .70*multiplier/count
    return .30/count


def _training_cases(id_cases,known_cases):
    rows=[]
    for domain,examples in (("id",id_cases),("known",known_cases)):
        for ex in examples:
            delta=[ex["adapter"][k]-ex["parent"][k] for k in (0,1)]
            rest=[ex["truth"][k]-ex["parent"][k] for k in (0,1)]
            a=sum(d*d for d in delta)
            b=sum(delta[k]*rest[k] for k in (0,1))
            rows.append({"features":ex["features"],
                         "weight":_example_weight(ex,domain,len(examples)),
                         "a":a,"b":b})
    total=sum(r["weight"] for r in rows)
    for r in rows:r["weight"]/=total
    return rows


def _optimal(rows,ridge):
    """Optimal scalar correction for a leaf's local weighted squared error."""
    a=sum(r["weight"]*r["a"] for r in rows)
    b=sum(r["weight"]*r["b"] for r in rows)
    t=min(1.,max(0.,b/(a+ridge))) if a+ridge>0 else 0.
    # Baseline-constant-free loss. Lower is better.
    return t, (a+ridge)*t*t-2*b*t


def _thresholds(values):
    samples=sorted(values)
    unique=sorted(set(samples[max(0,min(len(samples)-1,int(q*(len(samples)-1))))]
                 for q in (.1,.25,.4,.55,.7,.85)))
    return [(a+b)/2 for a,b in zip(unique[:-1],unique[1:])]


def _grow(rows,max_depth,min_leaf,ridge,depth=0):
    t,unsplit_loss=_optimal(rows,ridge)
    if depth>=max_depth or len(rows)<2*min_leaf:
        return {"trust":t}
    best=None
    for feat in range(1,safety27.FEATURE_COUNT):
        for thresh in _thresholds([r["features"][feat] for r in rows]):
            left=[r for r in rows if r["features"][feat]<=thresh]
            if len(left)<min_leaf or len(rows)-len(left)<min_leaf:
                continue
            right=[r for r in rows if r["features"][feat]>thresh]
            _,l_loss=_optimal(left,ridge)
            _,r_loss=_optimal(right,ridge)
            # Split is worthwhile only if it reduces actual fitted error.
            reduction=unsplit_loss-(l_loss+r_loss)
            if best is None or reduction>best[0]:
                best=(reduction,feat,thresh,left,right)
    if best is None or best[0]<=1e-6:
        return {"trust":t}
    _,feat,thresh,left,right=best
    return {"feature":feat,"threshold":thresh,
            "left":_grow(left,max_depth,min_leaf,ridge,depth+1),
            "right":_grow(right,max_depth,min_leaf,ridge,depth+1)}


def _leaf_trust(features,tree):
    current=tree
    for _ in range(32):
        if "trust" in current:
            t=current["trust"]
            if not isinstance(t,(int,float)) or not math.isfinite(t) or not 0<=t<=1:
                raise ValueError("invalid leaf trust")
            return t
        idx=current.get("feature")
        if not isinstance(idx,int) or not 1<=idx<safety27.FEATURE_COUNT:
            raise ValueError("invalid decision node")
        current=current["left"] if features[idx]<=current["threshold"] else current["right"]
    raise ValueError("tree depth limit exceeded")


def trust(ex,policy):
    if policy is None:
        return 0.
    if policy.get("kind")=="curvature_risk":
        # Infer possible turning only from observed kinematics. NEVER read
        # the supplied true regime name or any target coordinate here.
        angle=abs(ex["features"][3])
        acceleration=max(0.,ex["features"][2])
        risk=angle/(0.05+acceleration)
        guarded=(policy["low"]<=angle<=policy["high"] and
                 risk>=policy["risk_threshold"])
        value=policy["veto_trust"] if guarded else policy["scale"]
    else:
        value=policy["scale"]*_leaf_trust(ex["features"],policy["tree"])
    return max(0.,min(1.,value))


def prediction(ex,policy):
    t=trust(ex,policy)
    return [ex["parent"][k]+t*(ex["adapter"][k]-ex["parent"][k])
            for k in (0,1)]


def _score(examples,policy):
    if not examples:
        raise ValueError("empty group")
    all_errors=[]
    faults={x:[] for x in FAULTS}
    regimes={}
    activations={}
    for ex in examples:
        pred=prediction(ex,policy)
        loss=mesh.squared_error(pred,ex["truth"])
        all_errors.append(loss)
        faults[ex["fault"]].append(loss)
        regimes.setdefault(ex["regime"],[]).append(loss)
        activations.setdefault(ex["regime"],[]).append(trust(ex,policy))
    mean=statistics.fmean
    return {"mse":mean(all_errors),
            "faults":{k:mean(v) for k,v in faults.items()},
            "regimes":{k:mean(v) for k,v in regimes.items()},
            "trust_by_regime":{k:mean(v) for k,v in activations.items()}}


def _original_adapter_score(examples):
    return _score(examples,{"tree":{"trust":1.},"scale":1.})


def train(parent_path,adapter_path,id_public,id_trainval,
          known_train_public,known_train_targets,
          known_val_public,known_val_targets,out):
    parent,adapter=safety27._load_sources(parent_path,adapter_path)
    memory=parent["payload"]
    trained=adapter["payload"]["model"]
    id_train,_=safety27._collect_id(id_public,id_trainval,"train",memory)
    id_valid,_=safety27._collect_id(id_public,id_trainval,"validation",memory)
    known_train,seen=safety27._collect_ood(
        known_train_public,known_train_targets,transfer26.KNOWN,memory)
    known_valid,validated=safety27._collect_ood(
        known_val_public,known_val_targets,transfer26.KNOWN,memory)
    if seen==validated:
        raise ValueError("train/validation share a known physics seed")
    tr_id=safety27._augment(id_train,trained)
    tr_known=safety27._augment(known_train,trained)
    val_id=safety27._augment(id_valid,trained)
    val_known=safety27._augment(known_valid,trained)
    parent_id=_score(val_id,None)
    parent_known=_score(val_known,None)
    prior_id=_original_adapter_score(val_id)
    prior_known=_original_adapter_score(val_known)
    fit_rows=_training_cases(tr_id,tr_known)
    results=[]
    winner=None
    for depth in DEPTHS:
        for leaf in MIN_LEAVES:
            for ridge in RIDGES:
                fitted=_grow(fit_rows,depth,leaf,ridge)
                for scale in SCALES:
                    policy={"tree":fitted,"scale":scale,
                            "depth":depth,"min_leaf":leaf,"ridge":ridge}
                    id_score=_score(val_id,policy)
                    known_score=_score(val_known,policy)
                    safe=safety27._validation_safe(
                        parent_id,id_score,parent_known,known_score)
                    value=id_score["mse"]/parent_id["mse"]+(
                        known_score["mse"]/parent_known["mse"])
                    results.append({"depth":depth,"min_leaf":leaf,
                                    "ridge":ridge,"scale":scale,
                                    "id_mse":id_score["mse"],
                                    "known_mse":known_score["mse"],
                                    "linear_mse":id_score["regimes"]["linear"],
                                    "turning_mse":id_score["regimes"]["turning"],
                                    "eligible":safe})
                    if safe and (winner is None or value<winner[0]):
                        winner=(value,policy,id_score,known_score)
    # The first fitted tree search showed a constant trust policy on
    # validation. Explicitly evaluate predeclared observed-curvature risk
    # guards; these are never conditional on test set or a true regime label.
    for low in (0.,.06,.13):
        for high in (.30,.45,.65,.85):
            for ratio in (0.,1.,2.,3.,4.):
                for veto in (0.,.20):
                    for scale in (.75,1.):
                        policy={"kind":"curvature_risk","low":low,
                                "high":high,"risk_threshold":ratio,
                                "veto_trust":veto,"scale":scale}
                        id_score=_score(val_id,policy)
                        known_score=_score(val_known,policy)
                        safe=safety27._validation_safe(
                            parent_id,id_score,parent_known,known_score)
                        value=(id_score["mse"]/parent_id["mse"]+
                               known_score["mse"]/parent_known["mse"])
                        results.append({"kind":"curvature_risk","low":low,
                                        "high":high,"risk_threshold":ratio,
                                        "veto_trust":veto,"scale":scale,
                                        "id_mse":id_score["mse"],
                                        "known_mse":known_score["mse"],
                                        "linear_mse":id_score["regimes"]["linear"],
                                        "turning_mse":id_score["regimes"]["turning"],
                                        "eligible":safe})
                        if safe and (winner is None or value<winner[0]):
                            winner=(value,policy,id_score,known_score)
    payload={"schema":SCHEMA,"parent_sha256":parent["sha256"],
             "adapter_sha256":adapter["sha256"],
             "policy":winner[1] if winner else None,
             "regime_label_never_used_in_inference":True,
             "parent_unchanged":True,"no_automatic_promotion":True}
    checkpoint={"payload":payload,"sha256":mesh.digest(payload)}
    decision={"schema":SCHEMA,"parent_sha256":parent["sha256"],
              "adapter_sha256":adapter["sha256"],
              "candidate_sha256":checkpoint["sha256"],
              "validation_pass":winner is not None,
              "parent_fallback":winner is None,
              "validation_parent_id":parent_id,
              "validation_parent_known":parent_known,
              "validation_prior_adapter_id":prior_id,
              "validation_prior_adapter_known":prior_known,
              "validation_chosen_id":winner[2] if winner else parent_id,
              "validation_chosen_known":winner[3] if winner else parent_known,
              "trial_metrics":results,"no_test_labels_seen":True}
    output=Path(out)
    mesh.save(output/"parent.json",parent)
    mesh.save(output/"prior-adapter.json",adapter)
    mesh.save(output/"competence.json",checkpoint)
    mesh.save(output/"decision.json",decision)
    return decision


def _cohort(examples,policy):
    return {"parent":_score(examples,None),
            "adapter":_original_adapter_score(examples),
            "protected":_score(examples,policy)}


def evaluate(parent_path,adapter_path,candidate_path,decision_path,
             id_sets,known_sets,unknown_sets,out):
    if any(len(x)!=3 for x in (id_sets,known_sets,unknown_sets)):
        raise ValueError("expected 3 independent cohorts per domain")
    parent,adapter=safety27._load_sources(parent_path,adapter_path)
    candidate=mesh.load(candidate_path)
    decision=mesh.load(decision_path)
    if (candidate.get("sha256")!=mesh.digest(candidate.get("payload")) or
        candidate["payload"].get("schema")!=SCHEMA or
        candidate["payload"].get("parent_sha256")!=parent["sha256"] or
        candidate["payload"].get("adapter_sha256")!=adapter["sha256"] or
        decision.get("schema")!=SCHEMA or
        decision.get("candidate_sha256")!=candidate["sha256"] or
        decision.get("parent_sha256")!=parent["sha256"] or
        decision.get("adapter_sha256")!=adapter["sha256"]):
        raise ValueError("invalid competence lineage or evidence")
    policy=candidate["payload"]["policy"]
    if bool(policy)!=bool(decision.get("validation_pass")):
        raise ValueError("validator and candidate disagree")
    results=[]
    used=set()
    for group,paths in (("id",id_sets),("known",known_sets),("unseen",unknown_sets)):
        for pub,labels in paths:
            if group=="id":
                raw,identity=safety27._collect_id(pub,labels,"test",parent["payload"])
            else:
                raw,identity=safety27._collect_ood(
                    pub,labels,transfer26.KNOWN if group=="known" else
                    transfer26.UNKNOWN,parent["payload"])
            if (group,identity) in used:
                raise ValueError("duplicate sealed test cohort")
            used.add((group,identity))
            samples=safety27._augment(raw,adapter["payload"]["model"])
            results.append({"group":group,"id":identity,"scenes":len(samples)//3,
                            "scores":_cohort(samples,policy)})
    summary={}
    for group in ("id","known","unseen"):
        subset=[e for e in results if e["group"]==group]
        # safety27._group operates on generic scores dicts with our same schema.
        summary[group]=safety27._group(subset)
    old=summary["id"]["regimes"]
    old_skills_ok=(summary["id"]["protected_mse"]<
                   summary["id"]["parent_mse"] and
                   all(old[reg]["protected"] <= old[reg]["parent"]*(
                       1.0000001 if reg in ("linear","turning") else 1.02)
                       for reg in REGIMES))
    known_ok=(summary["known"]["protected_mse"]<=
              summary["known"]["parent_mse"]*.85 and
              summary["known"]["protected_mse"]<=
              summary["known"]["adapter_mse"]*1.10 and
              summary["known"]["seed_wins_vs_parent"]>=2)
    unseen_ok=(summary["unseen"]["protected_mse"]<=
               summary["unseen"]["parent_mse"]*.995 and
               summary["unseen"]["protected_mse"]<=
               summary["unseen"]["adapter_mse"]*1.02)
    faults_ok=all(
        x["scores"]["protected"]["faults"][f] <=
        x["scores"]["parent"]["faults"][f]*1.02
        for x in results if x["group"]=="id" for f in FAULTS)
    passed=bool(policy) and old_skills_ok and known_ok and unseen_ok and faults_ok
    status="RETENTION_REPAIR_SUPPORTED" if passed else (
        "NO_VALIDATED_COMPETENCE_MAP" if policy is None
        else "BLOCKED_SKILL_RETENTION_OR_TRANSFER")
    report={"schema":SCHEMA,"status":status,
            "parent_sha256":parent["sha256"],
            "adapter_sha256":adapter["sha256"],
            "candidate_sha256":candidate["sha256"],
            "decision_sha256":mesh.digest(decision),
            "familiar_regime_retention_pass":old_skills_ok,
            "known_motion_gain_pass":known_ok,
            "unseen_transfer_pass":unseen_ok,
            "fault_safety_pass":faults_ok,
            "summary":summary,"per_seed":results,
            "test_scenarios":sum(x["scenes"] for x in results),
            "manual_promotion_required":True,
            "parent_model_frozen":True}
    mesh.save(Path(out)/"evaluation.json",report)
    mesh.save(Path(out)/"promotion-gate.json",{
        "schema":SCHEMA,"status":status,
        "candidate_sha256":candidate["sha256"],
        "evaluation_sha256":mesh.digest(report),
        "no_autopromotion":True})
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sp=parser.add_subparsers(dest="command",required=True)
    tr=sp.add_parser("train")
    for name in ("parent","adapter","id-public","id-trainval",
                 "known-train-public","known-train-targets",
                 "known-val-public","known-val-targets","out"):
        tr.add_argument("--"+name,required=True)
    ev=sp.add_parser("evaluate")
    for name in ("parent","adapter","candidate","decision","out"):
        ev.add_argument("--"+name,required=True)
    for name in ("id-set","known-set","unseen-set"):
        ev.add_argument("--"+name,action="append",default=[])
    args=parser.parse_args()
    if args.command=="train":
        decision=train(args.parent,args.adapter,args.id_public,args.id_trainval,
                       args.known_train_public,args.known_train_targets,
                       args.known_val_public,args.known_val_targets,args.out)
        print({"validation_pass":decision["validation_pass"],
               "validation_id_mse":decision["validation_chosen_id"]["mse"],
               "validation_known_mse":decision["validation_chosen_known"]["mse"],
               "candidate_sha256":decision["candidate_sha256"]})
    else:
        def pair(vals):
            answer=[]
            for val in vals:
                paths=val.split(",")
                if len(paths)!=2:raise ValueError("expected public,target paths")
                answer.append(tuple(paths))
            return answer
        result=evaluate(args.parent,args.adapter,args.candidate,args.decision,
                        pair(args.id_set),pair(args.known_set),
                        pair(args.unseen_set),args.out)
        print({"status":result["status"],
               "gates":{name:result[name] for name in (
                   "familiar_regime_retention_pass","known_motion_gain_pass",
                   "unseen_transfer_pass","fault_safety_pass")},
               "summary":{name:{key:result["summary"][name][key]
                                 for key in ("parent_mse","adapter_mse","protected_mse")}
                          for name in ("id","known","unseen")}})


if __name__=="__main__":
    main()

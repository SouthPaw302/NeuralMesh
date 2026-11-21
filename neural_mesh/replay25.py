"""Repair Phase 2.5: rehearse prior examples and train the *deployed* predictor.

The legacy updater fitted unscaled Phase 2.2 residuals although production
robust_prediction scales them by reliability, availability and selective gate.
This version learns the exact linear residual in robust_prediction, replays
earlier scenes, validates on disjoint new/anchor partitions, and blocks
promotion without across-seed held-out improvement. No state auto-promotion.
"""
from __future__ import annotations

import argparse
import copy
import math
import statistics
from pathlib import Path

from neural_mesh import mesh, phase2, phase22, phase23, phase24, phase25

SCHEMA = "neuralmesh-rehearsal-repair-v1"
FAULTS = phase24.FAULTS
FEATURE_COUNT = 17
RIDGES = (0.0001, 0.001, 0.01, 0.1, 1.0)
REPLAY_SHARES = (0.25, 0.5, 0.75)
STEPS = (0.25, 0.5, 1.0)


def _assert_split(public, labels, expected):
    cfg = public.get("config_sha256")
    phase2._check_schema(public, cfg)
    phase2._check_labels(labels, cfg, expected)
    for split in expected:
        ids = [x["id"] for x in public["scenes"][split]]
        if len(ids) != len(set(ids)) or set(ids) != set(labels["labels"][split]):
            raise ValueError("labels do not match partition")


def _deployed_design(row, parent):
    """Return exact context and design matrix for active robust inference.

    For readout W, deployed prediction == context + W @ design.
    Confidence/outlier gating is independent of W, so this is exact.
    """
    policy = parent["reliability"]["selected"]
    active = phase24._active(row, policy["cutoff"])
    filled = phase24._impute(row, active)
    base = parent["phase22_router"]
    attention = phase22.gate(filled, base["gate"])
    denom = sum(attention[n] for n in active)
    context = [sum(attention[n] / denom * row["predictions"][n][k]
                   for n in active) for k in (0, 1)]
    selective = phase23.activation(
        phase23._vector(filled, base), parent["phase23_selector"]["selected"])
    scale = (policy["feedback_scale"] * selective * len(active) / len(phase24.NAMES)
             * base["feedback"]["beta"])
    x = phase22.feedback_features(filled, attention)
    return context, [scale*z for z in x]


def _precompute(public, labels, split, parent):
    _assert_split(public, labels, ("train", "validation"))
    examples = []
    for idx, scene in enumerate(public["scenes"][split]):
        row = phase24.make_row(scene, parent["models"])
        target = phase22._finite_vector(labels["labels"][split][scene["id"]], 2)
        for fault in FAULTS:
            damaged = phase24.inject(row, fault, idx)
            context, x = _deployed_design(damaged, parent)
            examples.append({"id": scene["id"], "regime": scene["regime"],
                             "fault": fault, "context": context, "x": x,
                             "target": target})
    return examples


def _normal_equations(examples):
    gram = [[0.]*FEATURE_COUNT for _ in range(FEATURE_COUNT)]
    targets = [[0.]*FEATURE_COUNT for _ in range(2)]
    count = len(examples)
    if not count:
        raise ValueError("no train scenes")
    for example in examples:
        x = example["x"]
        y = [example["target"][k]-example["context"][k] for k in (0, 1)]
        for i in range(FEATURE_COUNT):
            for j in range(FEATURE_COUNT):
                gram[i][j] += x[i]*x[j]/count
            for k in (0, 1):
                targets[k][i] += x[i]*y[k]/count
    return gram, targets


def _solve_blended(parent, new_stats, replay_stats, share, ridge):
    old = parent["phase22_router"]["feedback"]["readout"]
    a,b = new_stats, replay_stats
    matrix = [[(1-share)*a[0][i][j]+share*b[0][i][j]
               + (ridge if i==j else 0.) for j in range(FEATURE_COUNT)]
              for i in range(FEATURE_COUNT)]
    fitted=[]
    for k in (0,1):
        rhs=[(1-share)*a[1][k][i]+share*b[1][k][i]+ridge*old[k][i]
             for i in range(FEATURE_COUNT)]
        fitted.append(phase22._solve(matrix,rhs))
    if any(not math.isfinite(z) for w in fitted for z in w):
        raise ValueError("non-finite adapted weights")
    return fitted


def _interpolate(parent, fitted, step):
    baseline=parent["phase22_router"]["feedback"]["readout"]
    return [[baseline[k][i]+step*(fitted[k][i]-baseline[k][i])
             for i in range(FEATURE_COUNT)] for k in (0,1)]


def _score(examples, readout):
    by_fault={fault:[] for fault in FAULTS}
    by_regime={}
    for row in examples:
        pred=[row["context"][k]+sum(readout[k][i]*row["x"][i]
                                    for i in range(FEATURE_COUNT)) for k in (0,1)]
        err=mesh.squared_error(pred,row["target"])
        by_fault[row["fault"]].append(err)
        by_regime.setdefault(row["regime"],[]).append(err)
    return {"mse":statistics.fmean(v for arr in by_fault.values() for v in arr),
            "by_fault":{k:statistics.fmean(v) for k,v in by_fault.items()},
            "by_regime":{k:statistics.fmean(v) for k,v in by_regime.items()}}


def _validate_candidate(parent_score, new_score, past_base, past_score):
    # Must actually improve fresh validation without meaningful forgetting.
    if not new_score["mse"] < parent_score["mse"]*0.997:
        return False
    for fault in FAULTS:
        if (new_score["by_fault"][fault] > parent_score["by_fault"][fault]*1.005
                or past_score["by_fault"][fault] > past_base["by_fault"][fault]*1.005):
            return False
    for regime in past_base["by_regime"]:
        if past_score["by_regime"][regime] > past_base["by_regime"][regime]*1.02:
            return False
    return True


def train(parent_path, new_public_path, new_labels_path,
          replay_public_path, replay_labels_path, out):
    parent_checkpoint=phase24.restore_memory(parent_path)
    if parent_checkpoint["sha256"]!=phase25.PARENT_SHA:
        raise ValueError("unverified memory parent")
    parent=parent_checkpoint["payload"]
    new_public, new_labels=mesh.load(new_public_path), mesh.load(new_labels_path)
    replay_public,replay_labels=mesh.load(replay_public_path),mesh.load(replay_labels_path)
    _assert_split(new_public,new_labels,("train","validation"))
    _assert_split(replay_public,replay_labels,("train","validation"))
    if new_public["config_sha256"] == replay_public["config_sha256"]:
        raise ValueError("rehearsal must use independent scenes")
    new_tr=_precompute(new_public,new_labels,"train",parent)
    replay_tr=_precompute(replay_public,replay_labels,"train",parent)
    new_val=_precompute(new_public,new_labels,"validation",parent)
    replay_val=_precompute(replay_public,replay_labels,"validation",parent)
    original=parent["phase22_router"]["feedback"]["readout"]
    baseline_new=_score(new_val,original)
    baseline_past=_score(replay_val,original)
    equations_new=_normal_equations(new_tr)
    equations_past=_normal_equations(replay_tr)
    trials=[]
    choices=[]
    for share in REPLAY_SHARES:
        for ridge in RIDGES:
            fitted=_solve_blended(parent,equations_new,equations_past,share,ridge)
            for step in STEPS:
                readout=_interpolate(parent,fitted,step)
                fresh=_score(new_val,readout)
                past=_score(replay_val,readout)
                eligible=_validate_candidate(baseline_new,fresh,baseline_past,past)
                trials.append({"replay_share":share,"ridge":ridge,"step":step,
                               "new_val":fresh,"replay_val":past,
                               "validation_pass":eligible})
                if eligible:
                    choices.append((fresh["mse"],past["mse"],ridge,share,step,readout))
    best=min(choices,key=lambda x:x[:5]) if choices else None
    chosen=best[-1] if best else original
    if best:
        payload=phase25._updated_payload(parent,chosen,parent_checkpoint["sha256"])
        # Changing memory must be independently verifiable by Phase 2.4 restore.
        candidate=phase25._memory(payload)
    else:
        candidate=parent_checkpoint
    decision={"schema":SCHEMA,"parent_sha256":parent_checkpoint["sha256"],
              "candidate_sha256":candidate["sha256"],"validation_pass":best is not None,
              "candidate_is_parent":best is None,
              "new_config_sha256":new_public["config_sha256"],
              "replay_config_sha256":replay_public["config_sha256"],
              "training_scenes":{"new":len(new_public["scenes"]["train"]),
                                 "replay":len(replay_public["scenes"]["train"])},
              "validation_scenes":{"new":len(new_public["scenes"]["validation"]),
                                   "replay":len(replay_public["scenes"]["validation"])},
              "parent_new_val":baseline_new,"parent_replay_val":baseline_past,
              "selected":({"ridge":best[2],"replay_share":best[3],"step":best[4],
                           "new_val_mse":best[0],"replay_val_mse":best[1]} if best else None),
              "trial_metrics":trials,"heldout_labels_used_in_training":False,
              "promotion_requires_human_review":True}
    root=Path(out)
    mesh.save(root/"parent.json",parent_checkpoint)
    mesh.save(root/"candidate.json",candidate)
    mesh.save(root/"decision.json",decision)
    return decision


def _score_test_synthetic(public,labels,checkpoint):
    if "test" not in public["scenes"]:
        raise ValueError("ID test partition missing")
    _assert_split(public,labels,("test",))
    examples=[]
    parent=checkpoint["payload"]
    for idx,scene in enumerate(public["scenes"]["test"]):
        row=phase24.make_row(scene,parent["models"])
        target=phase22._finite_vector(labels["labels"]["test"][scene["id"]],2)
        for fault in FAULTS:
            context,x=_deployed_design(phase24.inject(row,fault,idx),parent)
            examples.append({"regime":scene["regime"],"fault":fault,
                             "context":context,"x":x,"target":target})
    return _score(examples,parent["phase22_router"]["feedback"]["readout"])


def _score_test_ood(public, labels,checkpoint):
    if (public.get("schema")!=phase24.OOD_SCHEMA or
        labels.get("schema")!=phase24.OOD_SCHEMA or
        public["seed"]!=labels["seed"] or
        labels["public_sha256"]!=mesh.digest(public)):
        raise ValueError("OOD public/target mismatch")
    if len(public["scenes"])!=len(labels["targets"]) or {
        r["id"] for r in public["scenes"]}!=set(labels["targets"]):
        raise ValueError("OOD test labels mismatch")
    examples=[]
    parent=checkpoint["payload"]
    for idx,scene in enumerate(public["scenes"]):
        row=phase24.make_row(scene,parent["models"])
        target=phase22._finite_vector(labels["targets"][scene["id"]],2)
        for fault in FAULTS:
            context,x=_deployed_design(phase24.inject(row,fault,idx),parent)
            examples.append({"regime":scene["regime"],"fault":fault,
                             "context":context,"x":x,"target":target})
    return _score(examples,parent["phase22_router"]["feedback"]["readout"])


def _regressions(parent,child,limit):
    return [x for x in parent["by_regime"] if
            child["by_regime"][x] > parent["by_regime"][x]*(1+limit)]


def _judgement(entries,decision):
    # Predeclared strong acceptance gate: baseline is never silently replaced.
    if not decision["validation_pass"]:
        return "BLOCKED_VALIDATION", {"reason":"no validation-eligible update"}
    id_entries=[x for x in entries if x["kind"]=="id"]
    ood_entries=[x for x in entries if x["kind"]=="ood"]
    id_changes=[(r["child"]["mse"]/r["parent"]["mse"]-1) for r in id_entries]
    ood_changes=[(r["child"]["mse"]/r["parent"]["mse"]-1) for r in ood_entries]
    wins=sum(x < 0 for x in id_changes)
    healthy_id=sum(r["child"]["by_fault"]["none"] for r in id_entries)/len(id_entries)
    healthy_old=sum(r["parent"]["by_fault"]["none"] for r in id_entries)/len(id_entries)
    global_id=statistics.fmean(id_changes)
    global_ood=statistics.fmean(ood_changes)
    # Must win on at least 2/3 independently prepared fresh ID test seeds
    # AND mean ID gain >= 0.5%; no average OOD degradation.
    passed=(wins>=2 and global_id <= -0.005 and global_ood <= 0 and
            healthy_id < healthy_old and
            all(r["child"]["by_fault"][f] <= r["parent"]["by_fault"][f]*1.005
                for r in entries for f in FAULTS) and
            all(not _regressions(r["parent"],r["child"],.05 if r["kind"]=="id" else .08)
                for r in entries))
    return ("PROMOTION_ELIGIBLE" if passed else "BLOCKED_GENERALIZATION",
            {"id_wins":wins,"id_fractional_change_mean":global_id,
             "ood_fractional_change_mean":global_ood,
             "id_healthy_mse_parent_mean":healthy_old,
             "id_healthy_mse_candidate_mean":healthy_id,
             "passes_predeclared_gate":passed})


def evaluate(parent_path,candidate_path,decision_path,id_sets,ood_sets,out):
    if len(id_sets)<3 or len(ood_sets)<3:
        raise ValueError("need >=3 unseen ID and >=3 OOD datasets")
    parent=phase24.restore_memory(parent_path)
    child=phase24.restore_memory(candidate_path)
    decision=mesh.load(decision_path)
    if parent["sha256"]!=phase25.PARENT_SHA or decision.get("schema")!=SCHEMA or (
        decision.get("parent_sha256")!=parent["sha256"] or
        decision.get("candidate_sha256")!=child["sha256"]):
        raise ValueError("invalid proposed memory lineage")
    if decision["validation_pass"]:
        if child["sha256"]==parent["sha256"] or (
            child["payload"].get("parent_sha256")!=parent["sha256"]):
            raise ValueError("invalid evolved checkpoint")
    elif child["sha256"]!=parent["sha256"]:
        raise ValueError("rejected checkpoint must preserve parent")
    results=[]
    seen=set()
    for kind,pairs in (("id",id_sets),("ood",ood_sets)):
        for pub_path,lab_path in pairs:
            public,labels=mesh.load(pub_path),mesh.load(lab_path)
            if kind=="id":
                _assert_split(public,labels,("test",))
                identity=public["config_sha256"]
                fn=_score_test_synthetic
            else:
                identity=str(public["seed"])
                fn=_score_test_ood
            if (kind,identity) in seen:
                raise ValueError("repeated benchmark seed")
            seen.add((kind,identity))
            results.append({"kind":kind,"dataset_id":identity,
                            "parent":fn(public,labels,parent),
                            "child":fn(public,labels,child)})
    status, summary=_judgement(results,decision)
    result={"schema":SCHEMA,"status":status,
            "parent_sha256":parent["sha256"],"candidate_sha256":child["sha256"],
            "decision_sha256":mesh.digest(decision),
            "test_sets":len(results),"summary":summary,
            "evaluations":results,"manual_promotion_required":True,
            "sealed_targets_training_exposure":False}
    root=Path(out)
    mesh.save(root/"evaluation.json",result)
    mesh.save(root/"promotion-gate.json",{
        "schema":SCHEMA,"status":status,
        "parent_sha256":parent["sha256"],"candidate_sha256":child["sha256"],
        "evaluation_sha256":mesh.digest(result),
        "manual_review_required":True})
    return result



def check_promotion(parent_file, candidate_file, decision_file, report_file, gate_file):
    """Fail-closed independent-six-set gate for a human-reviewed registry PR.

    Unlike legacy Phase 2.5, mere bounded regression is NOT promotable.
    """
    parent=phase24.restore_memory(parent_file)
    candidate=phase24.restore_memory(candidate_file)
    decision=mesh.load(decision_file)
    report=mesh.load(report_file)
    gate=mesh.load(gate_file)
    if (parent["sha256"]!=phase25.PARENT_SHA or
        decision.get("schema")!=SCHEMA or report.get("schema")!=SCHEMA or
        gate.get("schema")!=SCHEMA):
        raise ValueError("unsupported memory promotion evidence")
    if (not decision.get("validation_pass") or
        decision.get("candidate_is_parent") is not False or
        candidate["sha256"]==parent["sha256"] or
        candidate["payload"].get("parent_sha256")!=parent["sha256"]):
        raise ValueError("no verified adapted memory candidate")
    if not (report.get("status")=="PROMOTION_ELIGIBLE" and
            gate.get("status")=="PROMOTION_ELIGIBLE" and
            report.get("summary",{}).get("passes_predeclared_gate") is True and
            report.get("manual_promotion_required") is True and
            gate.get("manual_review_required") is True):
        raise ValueError("cross-seed promotion gate is closed")
    if (decision.get("candidate_sha256")!=candidate["sha256"] or
        report.get("candidate_sha256")!=candidate["sha256"] or
        gate.get("candidate_sha256")!=candidate["sha256"] or
        decision.get("parent_sha256")!=parent["sha256"] or
        report.get("parent_sha256")!=parent["sha256"] or
        gate.get("parent_sha256")!=parent["sha256"] or
        report.get("decision_sha256")!=mesh.digest(decision) or
        gate.get("evaluation_sha256")!=mesh.digest(report)):
        raise ValueError("candidate lineage or test evidence digest mismatch")
    entries=report.get("evaluations",[])
    if (len(entries)<6 or
        sum(x.get("kind")=="id" for x in entries)<3 or
        sum(x.get("kind")=="ood" for x in entries)<3 or
        len(set((x.get("kind"),x.get("dataset_id")) for x in entries))!=len(entries)):
        raise ValueError("insufficient independent unseen datasets")
    status,summary=_judgement(entries,decision)
    if status!="PROMOTION_ELIGIBLE" or summary!=report["summary"]:
        raise ValueError("independent generalization check failed")
    return {"schema":SCHEMA,"parent":parent["sha256"],
            "candidate":candidate["sha256"],"evaluation":mesh.digest(report),
            "status":"CHECKED_FOR_MANUAL_PR_ONLY"}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="command",required=True)
    tr=sub.add_parser("train")
    for n in ("parent","new-public","new-trainval","replay-public","replay-trainval","out"):
        tr.add_argument("--"+n,required=True)
    ev=sub.add_parser("evaluate")
    for n in ("parent","candidate","decision","out"):
        ev.add_argument("--"+n,required=True)
    ev.add_argument("--id-set",action="append",default=[],
                    help="public.json,target.json")
    ev.add_argument("--ood-set",action="append",default=[],
                    help="public.json,target.json")
    args=p.parse_args()
    if args.command=="train":
        result=train(args.parent,args.new_public,args.new_trainval,
                     args.replay_public,args.replay_trainval,args.out)
        print({"validation_pass":result["validation_pass"],
               "candidate":result["candidate_sha256"],
               "selected":result["selected"]})
    else:
        def pair(s):
            parts=s.split(",")
            if len(parts)!=2: raise ValueError("expected exactly public,targets")
            return (parts[0],parts[1])
        result=evaluate(args.parent,args.candidate,args.decision,
                        [pair(x) for x in args.id_set],
                        [pair(x) for x in args.ood_set],args.out)
        print({"status":result["status"],"summary":result["summary"]})


if __name__=="__main__":
    main()

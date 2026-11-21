"""Phase 2.5: supervised neural-memory evolution with fail-closed promotion checks.

Explicit lineage, frozen parent checkpoint, separate train/validation/test splits,
and a durable Git registry through a *manually approved* pull request.

No state encryption, no auto-merge, no background autonomous model modification.
"""
from __future__ import annotations

import argparse
import copy
import math
import statistics
from pathlib import Path

from neural_mesh import mesh, phase2, phase22, phase23, phase24

SCHEMA = "neuralmesh-evolution-v1"
PARENT_SHA = "8c147e3d6bad62cd9b6553738cba40dddc4f5120755273f01a366f3e235d22bc"
FAULTS = phase24.FAULTS
NODES = mesh.NODES
RIDGES = (0.02, 0.2, 2., 20., 200.)


def _samples(scenes, targets, model):
    if {r["id"] for r in scenes} != set(targets):
        raise ValueError("scene and label set mismatch")
    examples = []
    for scene in scenes:
        row = phase24.make_row(scene, model["models"])
        examples.append((row, phase22._finite_vector(targets[scene["id"]], 2)))
    return examples


def _fit_residual(parent, examples, ridge):
    """Ridge-to-parent update of 17-feature peer residual weights."""
    source = parent["phase22_router"]["feedback"]
    coef = source["readout"]
    if source["beta"] <= 0 or len(coef) != 2 or any(len(v) != 17 for v in coef):
        raise ValueError("phase 2.2 residual head cannot evolve")
    gram = [[0.]*17 for _ in range(17)]
    rhs = [[0.]*17 for _ in range(2)]
    for row, truth in examples:
        att = phase22.gate(row, parent["phase22_router"]["gate"])
        x = phase22.feedback_features(row, att)
        context = phase22.blend(row, att)
        for i in range(17):
            for j in range(17):
                gram[i][j] += x[i]*x[j]
            for k in (0, 1):
                rhs[k][i] += x[i]*(truth[k]-context[k])/source["beta"]
    matrix = [[gram[i][j] + (ridge if i==j else 0.) for j in range(17)]
              for i in range(17)]
    weights = [phase22._solve(matrix, [rhs[k][i]+ridge*coef[k][i] for i in range(17)])
               for k in (0, 1)]
    if any(not math.isfinite(v) or abs(v)>1e3 for line in weights for v in line):
        raise ValueError("unstable learned memory")
    return weights


def _updated_payload(parent, readout, source, generation=2):
    payload = copy.deepcopy(parent)
    before = payload["phase22_router"]["feedback"]["readout"]
    payload["phase22_router"]["feedback"]["readout"] = copy.deepcopy(readout)
    payload["version"] = 2
    payload["generation"] = generation
    payload["parent_sha256"] = source
    payload["history"].append({
        "phase":"2.5", "update":"ridge-to-parent peer residual readout",
        "parent_sha256": source, "generation": generation,
        "policy":"manual-reviewed-candidate-only"})
    payload["adopted_automatically"] = False
    payload["registry_status"] = "candidate"
    if before == readout:
        raise ValueError("no synaptic memory difference")
    return payload


def _fault_mean(examples, payload):
    policy = payload["reliability"]["selected"]
    errs = []
    for i, (row, label) in enumerate(examples):
        for fault in FAULTS:
            pred, _ = phase24.robust_prediction(
                phase24.inject(row, fault, i), payload,
                policy["cutoff"], policy["feedback_scale"])
            errs.append(mesh.squared_error(pred, label))
    return statistics.fmean(errs)


def _clean_mean(examples, payload):
    policy = payload["reliability"]["selected"]
    return statistics.fmean(mesh.squared_error(
        phase24.robust_prediction(row, payload,
                                  policy["cutoff"],policy["feedback_scale"])[0], target)
        for row,target in examples)


def _memory(payload):
    return {"payload":payload, "sha256":mesh.digest(payload)}


def train_candidate(parent_file, new_public_file, new_trainval_file,
                    anchor_public_file, anchor_trainval_file, out):
    checkpoint=phase24.restore_memory(parent_file)
    if checkpoint["sha256"] != PARENT_SHA:
        raise ValueError("prototype expects exact verified Phase 2.4 parent checkpoint")
    parent=checkpoint["payload"]
    new_public=mesh.load(new_public_file)
    new_labels=mesh.load(new_trainval_file)
    anchor_public=mesh.load(anchor_public_file)
    anchor_labels=mesh.load(anchor_trainval_file)
    for public, labels in ((new_public,new_labels),(anchor_public,anchor_labels)):
        cfg=public.get("config_sha256")
        phase2._check_schema(public,cfg)
        phase2._check_labels(labels,cfg,("train","validation"))
        if set(labels["labels"]["train"]) != {x["id"] for x in public["scenes"]["train"]}:
            raise ValueError("training labels mismatch")
        if set(labels["labels"]["validation"]) != {x["id"] for x in public["scenes"]["validation"]}:
            raise ValueError("validation labels mismatch")
    if new_public["config_sha256"]==anchor_public["config_sha256"]:
        raise ValueError("new and anchor data must be distinct")
    train=_samples(new_public["scenes"]["train"],new_labels["labels"]["train"],parent)
    valid=_samples(new_public["scenes"]["validation"],new_labels["labels"]["validation"],parent)
    anchor=_samples(anchor_public["scenes"]["validation"],anchor_labels["labels"]["validation"],parent)
    base_new=_clean_mean(valid,parent)
    base_anchor=_clean_mean(anchor,parent)
    base_fault=_fault_mean(valid,parent)
    trials=[]
    for ridge in RIDGES:
        fitted=_fit_residual(parent,train,ridge)
        candidate=_updated_payload(parent,fitted,checkpoint["sha256"])
        new_score=_clean_mean(valid,candidate)
        anchor_score=_clean_mean(anchor,candidate)
        fault_score=_fault_mean(valid,candidate)
        accepted=(new_score <= base_new*0.995 and
                  anchor_score <= base_anchor*1.03 and
                  fault_score <= base_fault*1.05)
        trials.append({"ridge":ridge, "new_validation_mse":new_score,
                       "anchor_validation_mse":anchor_score,
                       "fault_validation_mse":fault_score,
                       "eligible":accepted, "payload":candidate})
    approved=[t for t in trials if t["eligible"]]
    best=min(approved,key=lambda r:r["new_validation_mse"]) if approved else None
    proposal_payload=best["payload"] if best is not None else copy.deepcopy(parent)
    candidate=_memory(proposal_payload) if best is not None else checkpoint
    # Assert newly issued snapshots remain properly tied to the immutable parent.
    if best is not None:
        if proposal_payload["parent_sha256"]!=checkpoint["sha256"]:
            raise ValueError("broken parent chain")
    decision={"schema":SCHEMA, "parent_sha256":checkpoint["sha256"],
              "candidate_sha256":candidate["sha256"],
              "new_dataset_sha256":mesh.digest(new_public),
              "anchor_dataset_sha256":mesh.digest(anchor_public),
              "new_validation_baseline_mse":base_new,
              "anchor_validation_baseline_mse":base_anchor,
              "fault_validation_baseline_mse":base_fault,
              "selected_ridge":best["ridge"] if best else None,
              "validation_pass":best is not None,
              "candidate_is_identical_to_parent":best is None,
              "trial_metrics":[{k:v for k,v in t.items() if k!="payload"} for t in trials],
              "criteria":{"min_new_validation_reduction":0.005,
                          "max_anchor_relative_regression":0.03,
                          "max_fault_relative_regression":0.05},
              "promotion_requires_manual_pr":True}
    out=Path(out)
    mesh.save(out/"parent.json",checkpoint)
    mesh.save(out/"candidate.json",candidate)
    mesh.save(out/"training-decision.json",decision)
    return decision


def _score_rows(scenes, targets, checkpoint):
    memory=checkpoint["payload"]
    policy=memory["reliability"]["selected"]
    result={f:[] for f in FAULTS}
    by_regime={}
    for i,scene in enumerate(scenes):
        row=phase24.make_row(scene,memory["models"])
        label=phase22._finite_vector(targets[scene["id"]],2)
        by_regime.setdefault(scene["regime"],{f:[] for f in FAULTS})
        for fault in FAULTS:
            pred,_=phase24.robust_prediction(phase24.inject(row,fault,i),
                     memory,policy["cutoff"],policy["feedback_scale"])
            error=mesh.squared_error(pred,label)
            result[fault].append(error)
            by_regime[scene["regime"]][fault].append(error)
    return {"all":{f:statistics.fmean(v) for f,v in result.items()},
            "regimes":{reg:{f:statistics.fmean(v) for f,v in d.items()}
                       for reg,d in by_regime.items()},
            "errors":result}


def _run_suite(public, targets, parent, candidate):
    if len(public["scenes"]) != len(targets):
        raise ValueError("test examples and labels count mismatch")
    if {s["id"] for s in public["scenes"]} != set(targets):
        raise ValueError("test examples and labels mismatch")
    before=_score_rows(public["scenes"],targets,parent)
    after=_score_rows(public["scenes"],targets,candidate)
    return {"parent_mse":before["all"],"candidate_mse":after["all"],
            "parent_regimes":before["regimes"],
            "candidate_regimes":after["regimes"],
            "paired_difference":{f:phase2._paired_bootstrap(after["errors"][f],before["errors"][f])
                                 for f in FAULTS}}


def _suite_guard(suite, max_relative, max_regime_relative):
    for fault in FAULTS:
        baseline=suite["parent_mse"][fault]
        current=suite["candidate_mse"][fault]
        if current > baseline*(1+max_relative):
            return False
    for regime, previous in suite["parent_regimes"].items():
        for fault in FAULTS:
            if (suite["candidate_regimes"][regime][fault] >
                    previous[fault]*(1+max_regime_relative)):
                return False
    return True


def evaluate_candidate(parent_file, candidate_file, decision_file, new_public_file,
                       new_test_file, ood_public_file, ood_test_file, out):
    parent=phase24.restore_memory(parent_file)
    candidate=phase24.restore_memory(candidate_file)
    decision=mesh.load(decision_file)
    if decision.get("schema")!=SCHEMA or decision.get("parent_sha256")!=parent["sha256"] or (
        decision.get("candidate_sha256")!=candidate["sha256"]):
        raise ValueError("untrusted training decision or memory lineage")
    if parent["sha256"] != PARENT_SHA:
        raise ValueError("unexpected parent memory")
    accepted=(decision.get("validation_pass") is True and
              candidate["sha256"]!=parent["sha256"])
    if accepted:
        payload=candidate["payload"]
        if payload.get("parent_sha256")!=parent["sha256"] or payload.get("generation")!=2:
            raise ValueError("candidate lineage invalid")
    elif not (decision.get("validation_pass") is False and
              candidate["sha256"]==parent["sha256"]):
        raise ValueError("candidate decision and memory disagree")
    public=mesh.load(new_public_file)
    test=mesh.load(new_test_file)
    phase2._check_schema(public,public.get("config_sha256"))
    phase2._check_labels(test,public["config_sha256"],("test",))
    if decision["new_dataset_sha256"]!=mesh.digest(public):
        raise ValueError("new data split does not match trained proposal")
    ood=mesh.load(ood_public_file)
    ood_labels=mesh.load(ood_test_file)
    if ood.get("schema")!=phase24.OOD_SCHEMA or ood_labels.get("schema")!=phase24.OOD_SCHEMA or (
        ood.get("seed")!=ood_labels.get("seed") or mesh.digest(ood)!=ood_labels.get("public_sha256")):
        raise ValueError("sealed OOD provenance invalid")
    in_dist=_run_suite({"scenes":public["scenes"]["test"]},test["labels"]["test"],parent,candidate)
    out_dist=_run_suite(ood,ood_labels["targets"],parent,candidate)
    # Promotion guard is deliberately conservative and must be satisfied by
    # frozen held-out metrics. No evaluation result flows into training.
    in_ok=_suite_guard(in_dist,0.04,0.15)
    ood_ok=_suite_guard(out_dist,0.07,0.20)
    status=("BLOCKED_VALIDATION" if not accepted else
            "PROMOTION_ELIGIBLE" if in_ok and ood_ok else "BLOCKED_REGRESSION")
    report={"schema":SCHEMA,"status":status,"parent_sha256":parent["sha256"],
            "candidate_sha256":candidate["sha256"],"dataset_config":public["config_sha256"],
            "ood_seed":ood["seed"],"train_validation_pass":accepted,
            "indistribution_guard_pass":in_ok,"ood_guard_pass":ood_ok,
            "indistribution":in_dist,"ood":out_dist,
            "requires_human_review":True,
            "no_automatic_state_promotion":True}
    mesh.save(Path(out)/"evaluation.json",report)
    mesh.save(Path(out)/"promotion-gate.json",{
        "schema":SCHEMA,"status":status,
        "parent_sha256":parent["sha256"],"candidate_sha256":candidate["sha256"],
        "evaluation_sha256":mesh.digest(report),
        "manual_review_required":True, "test_metrics_frozen":True})
    return report


def check_promotion(parent_file,candidate_file,decision_file,report_file,gate_file):
    parent=phase24.restore_memory(parent_file)
    candidate=phase24.restore_memory(candidate_file)
    decision=mesh.load(decision_file)
    report=mesh.load(report_file)
    gate=mesh.load(gate_file)
    if (gate.get("schema")!=SCHEMA or gate.get("status")!="PROMOTION_ELIGIBLE" or
        report.get("schema")!=SCHEMA or report.get("status")!="PROMOTION_ELIGIBLE" or
        decision.get("validation_pass") is not True or
        gate.get("manual_review_required") is not True or
        gate.get("test_metrics_frozen") is not True):
        raise ValueError("promotion gate is closed")
    if parent["sha256"]!=PARENT_SHA or candidate["sha256"]==parent["sha256"]:
        raise ValueError("invalid promoted lineage")
    if candidate["payload"].get("parent_sha256")!=parent["sha256"]:
        raise ValueError("broken chain")
    if gate.get("evaluation_sha256")!=mesh.digest(report):
        raise ValueError("evaluation digest mismatch")
    for field in ("parent_sha256","candidate_sha256"):
        if any(d.get(field)!= (parent["sha256"] if field=="parent_sha256" else candidate["sha256"])
               for d in (gate,report,decision)):
            raise ValueError("candidate/evidence hash mismatch")
    if not report["indistribution_guard_pass"] or not report["ood_guard_pass"]:
        raise ValueError("regression guard failed")
    return {"schema":SCHEMA,"parent":parent["sha256"],"candidate":candidate["sha256"],
            "evaluation":mesh.digest(report),"status":"CHECKED_FOR_MANUAL_PR_ONLY"}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    subs=ap.add_subparsers(dest="command",required=True)
    p=subs.add_parser("train")
    for name in ("parent","new-public","new-trainval","anchor-public","anchor-trainval","out"):
        p.add_argument("--"+name,required=True)
    p=subs.add_parser("evaluate")
    for name in ("parent","candidate","decision","new-public","new-test",
                 "ood-public","ood-test","out"):
        p.add_argument("--"+name,required=True)
    p=subs.add_parser("check-promotion")
    for name in ("parent","candidate","decision","report","gate"):
        p.add_argument("--"+name,required=True)
    args=ap.parse_args()
    if args.command=="train":
        d=train_candidate(args.parent,args.new_public,args.new_trainval,
                          args.anchor_public,args.anchor_trainval,args.out)
        print({"validation_pass":d["validation_pass"],
               "candidate_sha256":d["candidate_sha256"],
               "selected_ridge":d["selected_ridge"]})
    elif args.command=="evaluate":
        r=evaluate_candidate(args.parent,args.candidate,args.decision,
                             args.new_public,args.new_test,args.ood_public,
                             args.ood_test,args.out)
        print({"status":r["status"],"indistribution_guard_pass":r["indistribution_guard_pass"],
               "ood_guard_pass":r["ood_guard_pass"]})
    else:
        r=check_promotion(args.parent,args.candidate,args.decision,args.report,args.gate)
        print(r)


if __name__=="__main__":
    main()

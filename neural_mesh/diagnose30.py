"""Diagnose whether observed motion can support a no-forgetting transfer gate.

Uses *only* training/validation data. Regime-aware results are explicitly
diagnostic/oracle controls; no true regime is ever provided to the deployed
predictor. No sealed test labels, automatic model update or promotion.
"""
from __future__ import annotations

import argparse
import math
import statistics
from pathlib import Path

from neural_mesh import mesh, phase24, safety27, transfer26

SCHEMA="neuralmesh-observable-separability-v1"
REGIMES=("linear","accelerating","jerk","turning","bounce","swirl")


def _optimal_alpha(rows):
    numerator=0.
    denominator=0.
    for e in rows:
        d=[e["adapter"][k]-e["parent"][k] for k in (0,1)]
        rest=[e["truth"][k]-e["parent"][k] for k in (0,1)]
        numerator+=sum(d[k]*rest[k] for k in (0,1))
        denominator+=sum(z*z for z in d)
    return min(1.,max(0.,numerator/denominator)) if denominator else 0.


def _score(rows,selector):
    by_regime={r:[] for r in REGIMES}
    for e in rows:
        alpha=selector(e)
        p=[e["parent"][k]+alpha*(e["adapter"][k]-e["parent"][k])
           for k in (0,1)]
        by_regime[e["regime"]].append(mesh.squared_error(p,e["truth"]))
    if any(not v for v in by_regime.values()):
        raise ValueError("incomplete diagnostic regime coverage")
    return {"overall_mse":statistics.fmean(x for arr in by_regime.values() for x in arr),
            "by_regime":{r:statistics.fmean(v) for r,v in by_regime.items()}}


def _classifier(train,validation):
    n=len(train[0]["features"])
    mean=[statistics.fmean(e["features"][j] for e in train) for j in range(n)]
    scale=[max(.04,statistics.pstdev(e["features"][j] for e in train))
           for j in range(n)]
    # Class centroids from training observations; regime label is used only to
    # fit this diagnostic separability control.
    centroids={}
    for reg in REGIMES:
        rows=[e for e in train if e["regime"]==reg]
        if not rows:
            raise ValueError("missing class in classifier training")
        centroids[reg]=[
            statistics.fmean((e["features"][j]-mean[j])/scale[j] for e in rows)
            for j in range(n)]
    counts={r:{k:0 for k in REGIMES} for r in REGIMES}
    for e in validation:
        z=[(e["features"][j]-mean[j])/scale[j] for j in range(n)]
        distances={reg:sum((z[j]-v[j])**2 for j in range(1,n))
                   for reg,v in centroids.items()}
        pred=min(distances,key=distances.get)
        counts[e["regime"]][pred]+=1
    return {
        "validation_accuracy":sum(counts[r][r] for r in REGIMES)/len(validation),
        "accuracy_by_regime":{
            reg:counts[reg][reg]/sum(counts[reg].values()) for reg in REGIMES},
        "confusion_matrix":counts,
        "method":"training-class-centroids-on-observed-kinematics_only",
        "labels_used_only_for_diagnostic_not_for_live_prediction":True}


def run(parent_file,adapter_file,id_public,id_labels,
        novel_train_public,novel_train_targets,novel_val_public,
        novel_val_targets,out):
    parent,adapter=safety27._load_sources(parent_file,adapter_file)
    memory=parent["payload"]
    original=adapter["payload"]["model"]
    old_train,_=safety27._collect_id(id_public,id_labels,"train",memory)
    old_valid,_=safety27._collect_id(id_public,id_labels,"validation",memory)
    new_train,_=safety27._collect_ood(novel_train_public,novel_train_targets,
                                      transfer26.KNOWN,memory)
    new_valid,_=safety27._collect_ood(novel_val_public,novel_val_targets,
                                      transfer26.KNOWN,memory)
    train=safety27._augment(old_train+new_train,original)
    valid=safety27._augment(old_valid+new_valid,original)
    by_regime={r:[e for e in train if e["regime"]==r] for r in REGIMES}
    opt_from_train={reg:_optimal_alpha(rows) for reg,rows in by_regime.items()}
    opt_from_valid={reg:_optimal_alpha([e for e in valid if e["regime"]==reg])
                    for reg in REGIMES}
    result={
        "schema":SCHEMA,
        "parent_sha256":parent["sha256"],"adapter_sha256":adapter["sha256"],
        "training_scenes":len(train)//len(phase24.FAULTS),
        "validation_scenes":len(valid)//len(phase24.FAULTS),
        "regime_optimal_training_trust":opt_from_train,
        "regime_optimal_validation_trust_oracle":opt_from_valid,
        "validation_parent":_score(valid,lambda e:0.),
        "validation_original_adapter":_score(valid,lambda e:1.),
        "validation_training_oracle_regime_trust":_score(
            valid,lambda e:opt_from_train[e["regime"]]),
        "validation_infeasible_label_oracle_ceiling":_score(
            valid,lambda e:opt_from_valid[e["regime"]]),
        "observation_separability":_classifier(train,valid),
        "no_test_labels_seen":True,
        "diagnostic_oracles_not_deployable":True,
        "memory_unchanged":True}
    mesh.save(out,result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("parent","adapter","id-public","id-labels",
                 "novel-train-public","novel-train-targets",
                 "novel-val-public","novel-val-targets","out"):
        p.add_argument("--"+name,required=True)
    a=p.parse_args()
    res=run(a.parent,a.adapter,a.id_public,a.id_labels,
            a.novel_train_public,a.novel_train_targets,
            a.novel_val_public,a.novel_val_targets,a.out)
    print({"mse_parent":res["validation_parent"]["overall_mse"],
           "mse_adapter":res["validation_original_adapter"]["overall_mse"],
           "mse_class_oracle":res["validation_training_oracle_regime_trust"]["overall_mse"],
           "class_separability":res["observation_separability"]["validation_accuracy"],
           "training_trust":res["regime_optimal_training_trust"]})


if __name__=="__main__":
    main()

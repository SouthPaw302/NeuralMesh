"""Write an immutable, reviewable Git-backed model-memory registry proposal.

This CLI only stages files in the current working tree. It cannot push,
merge, or promote memory. The GitHub workflow invoking it requires manual
dispatch and opens a pull request rather than modifying main.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from neural_mesh import mesh, phase24, phase25

SCHEMA = "neuralmesh-registry-v1"


def _hex(value):
    if not isinstance(value,str) or not re.fullmatch("[a-f0-9]{64}",value):
        raise ValueError("invalid sha256 identifier")
    return value


def stage(parent_file, candidate_file, decision_file, evaluation_file,
          gate_file, registry_root, source_run):
    if not isinstance(source_run,int) or source_run < 1:
        raise ValueError("source run ID must be positive")
    result=phase25.check_promotion(
        parent_file,candidate_file,decision_file,evaluation_file,gate_file)
    parent=phase24.restore_memory(parent_file)
    candidate=phase24.restore_memory(candidate_file)
    root=Path(registry_root)
    idxpath=root/"index.json"
    if idxpath.exists():
        current=mesh.load(idxpath)
        if current.get("schema")!=SCHEMA or current.get("active_sha256")!=parent["sha256"]:
            raise ValueError("stale memory-registry head; reconcile before promotion")
        entries=list(current.get("versions",[]))
        if not all(isinstance(x,dict) for x in entries):
            raise ValueError("registry index malformed")
    else:
        # Initial root only: bootstrap must be the verifiable Phase 2.4 checkpoint.
        if parent["sha256"]!=phase25.PARENT_SHA:
            raise ValueError("registry does not exist and parent is not genesis")
        entries=[{"sha256":_hex(parent["sha256"]),
                  "parent_sha256":None,"generation":1,
                  "phase":"2.4-genesis","source_run":44871190473}]
    known={x["sha256"] for x in entries}
    if candidate["sha256"] in known:
        raise ValueError("candidate already appears in registry")
    if parent["sha256"] not in known:
        raise ValueError("parent missing from registry")
    for item in (parent,candidate):
        sha=_hex(item["sha256"])
        version=root.parent/"versions"/(sha+".json")
        if version.exists():
            if mesh.load(version)!=item:
                raise ValueError("existing immutable memory version differs")
        else:
            mesh.save(version,item)
    entries.append({"sha256":_hex(candidate["sha256"]),
                    "parent_sha256":parent["sha256"],
                    "generation":candidate["payload"]["generation"],
                    "phase":"2.5",
                    "source_run":source_run,
                    "evaluation_sha256":result["evaluation"]})
    idx={"schema":SCHEMA,"active_sha256":candidate["sha256"],
         "versions":entries,
         "activation_policy":"this index becomes active only after PR merge",
         "review_required":True}
    audit={"schema":SCHEMA,
           "operation":"PROPOSE_MODEL_MEMORY_PROMOTION",
           "source_run":source_run,
           "parent_sha256":parent["sha256"],
           "candidate_sha256":candidate["sha256"],
           "evaluation_sha256":result["evaluation"],
           "human_merge_required":True}
    mesh.save(idxpath,idx)
    mesh.save(root/"reviews"/(candidate["sha256"][:16]+".json"),audit)
    return audit


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("parent","candidate","decision","evaluation","gate","registry-root"):
        p.add_argument("--"+key,required=True)
    p.add_argument("--source-run",required=True,type=int)
    a=p.parse_args()
    result=stage(a.parent,a.candidate,a.decision,a.evaluation,a.gate,
                 a.registry_root,a.source_run)
    print(result)


if __name__=="__main__":
    main()

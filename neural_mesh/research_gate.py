"""Scientific quality gate: code can pass while a learned candidate fails.

Exits nonzero for a blocked or unverifiable research model. A green GitHub
test suite alone is never treated as a successful learning generation.
"""
from __future__ import annotations

import argparse
from neural_mesh import mesh

ACCEPTED={"RETENTION_REPAIR_SUPPORTED"}


def require_accepted(report_path,gate_path):
    report=mesh.load(report_path)
    gate=mesh.load(gate_path)
    if (report.get("status") not in ACCEPTED or
        gate.get("status")!=report.get("status") or
        gate.get("evaluation_sha256")!=mesh.digest(report) or
        not report.get("parent_model_frozen") or
        report.get("familiar_regime_retention_pass") is not True or
        report.get("known_motion_gain_pass") is not True or
        report.get("unseen_transfer_pass") is not True or
        report.get("fault_safety_pass") is not True):
        raise ValueError(
            "LEARNING_NOT_ACCEPTED: implementation may pass, but the model "
            "did not demonstrate the required old-skill and transfer gains")
    return {"status":"ACCEPTED_FOR_HUMAN_REVIEW_ONLY",
            "candidate_sha256":report["candidate_sha256"],
            "report_sha256":mesh.digest(report)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report",required=True)
    parser.add_argument("--gate",required=True)
    args=parser.parse_args()
    print(require_accepted(args.report,args.gate))


if __name__=="__main__":
    main()

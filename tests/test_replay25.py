"""Regression tests for repaired learning objective and promotion criteria."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from neural_mesh import mesh, phase2, phase24, phase25, replay25


class ReplayRepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory()
        cls.root=Path(cls.tmp.name)
        cls.anchor=cls.root/"anchor"
        cls.fresh=cls.root/"fresh"
        phase2.prepare(64013,{"train":48,"validation":16,"test":16},cls.anchor)
        phase2.prepare(26051,{"train":48,"validation":16,"test":16},cls.fresh)
        cls.parent_path=cls.root/"parent.json"
        cls.memory=phase24.build_memory(cls.anchor/"public.json",
            cls.anchor/"weights.json",cls.anchor/"trainval-labels.json",
            cls.parent_path)
        with patch.object(phase25,"PARENT_SHA",cls.memory["sha256"]):
            cls.decision=replay25.train(
                cls.parent_path,cls.fresh/"public.json",cls.fresh/"trainval-labels.json",
                cls.anchor/"public.json",cls.anchor/"trainval-labels.json",
                cls.root/"result")
        cls.candidate=phase24.restore_memory(cls.root/"result/candidate.json")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_exact_deployed_output_for_all_fault_states(self):
        model=self.memory["payload"]
        scene=mesh.load(self.fresh/"public.json")["scenes"]["validation"][0]
        row=phase24.make_row(scene,model["models"])
        w=model["phase22_router"]["feedback"]["readout"]
        policy=model["reliability"]["selected"]
        for kind in phase24.FAULTS:
            damaged=phase24.inject(row,kind,0)
            context,x=replay25._deployed_design(damaged,model)
            analytically=[context[k]+sum(w[k][i]*x[i] for i in range(17))
                          for k in (0,1)]
            actual,_=phase24.robust_prediction(damaged,model,
                        policy["cutoff"],policy["feedback_scale"])
            self.assertLess(mesh.squared_error(analytically,actual),1e-24)

    def test_parent_memory_immutable_and_candidate_lineage(self):
        original=phase24.restore_memory(self.parent_path)
        self.assertEqual(original,self.memory)
        self.assertEqual(self.decision["parent_sha256"],original["sha256"])
        self.assertEqual(self.decision["candidate_sha256"],self.candidate["sha256"])
        self.assertEqual(len(self.decision["trial_metrics"]),
                         len(replay25.REPLAY_SHARES)*len(replay25.RIDGES)*len(replay25.STEPS))
        self.assertTrue(self.decision["promotion_requires_human_review"])
        if self.decision["validation_pass"]:
            self.assertEqual(self.candidate["payload"]["parent_sha256"],original["sha256"])
            self.assertEqual(self.candidate["payload"]["version"],2)
            self.assertNotEqual(self.candidate["sha256"],original["sha256"])
        else:
            self.assertEqual(self.candidate,original)

    def test_sealed_target_partition_never_enters_training(self):
        leaked=copy.deepcopy(mesh.load(self.fresh/"trainval-labels.json"))
        leaked["labels"]["test"]=mesh.load(self.fresh/"test-labels.json")["labels"]["test"]
        location=self.root/"leaked.json"
        mesh.save(location,leaked)
        with patch.object(phase25,"PARENT_SHA",self.memory["sha256"]):
            with self.assertRaisesRegex(ValueError,"label partitions"):
                replay25.train(self.parent_path,self.fresh/"public.json",location,
                    self.anchor/"public.json",self.anchor/"trainval-labels.json",
                    self.root/"should-not-exist")

    def test_strict_generalization_gate_rejects_even_tolerated_losses(self):
        def make(kind,par,child):
            orig={"mse":par,"by_fault":{f:par for f in phase24.FAULTS},
                  "by_regime":{"linear":par}}
            new={"mse":child,"by_fault":{f:child for f in phase24.FAULTS},
                 "by_regime":{"linear":child}}
            return {"kind":kind,"parent":orig,"child":new}
        entries=[make("id",1,1.002) for _ in range(3)]
        entries += [make("ood",10,9.999) for _ in range(3)]
        state,why=replay25._judgement(entries,{"validation_pass":True})
        self.assertEqual(state,"BLOCKED_GENERALIZATION")
        self.assertFalse(why["passes_predeclared_gate"])
        entries=[make("id",1,.98) for _ in range(3)]
        entries += [make("ood",10,9.9) for _ in range(3)]
        state,why=replay25._judgement(entries,{"validation_pass":True})
        self.assertEqual(state,"PROMOTION_ELIGIBLE")
        self.assertTrue(why["passes_predeclared_gate"])

    def test_3_seed_sealed_replay_reports_and_hashes(self):
        ids,oods=[],[]
        for seed in (26051,26052,26053):
            folder=self.root/f"id-{seed}"
            if seed != 26051:
                phase2.prepare(seed,{"train":16,"validation":16,"test":16},folder)
            else:
                folder=self.fresh
            ids.append((folder/"public.json",folder/"test-labels.json"))
        for seed in (97313,97417,97523):
            folder=self.root/f"ood-{seed}"
            phase24.prepare_ood(seed,16,folder)
            oods.append((folder/"public.json",folder/"sealed-targets.json"))
        with patch.object(phase25,"PARENT_SHA",self.memory["sha256"]):
            report=replay25.evaluate(self.root/"result/parent.json",
                self.root/"result/candidate.json",self.root/"result/decision.json",
                ids,oods,self.root/"evaluation")
        self.assertEqual(len(report["evaluations"]),6)
        self.assertIn(report["status"],("BLOCKED_VALIDATION",
                      "BLOCKED_GENERALIZATION","PROMOTION_ELIGIBLE"))
        self.assertEqual(report["parent_sha256"],self.memory["sha256"])
        self.assertTrue(report["manual_promotion_required"])
        with patch.object(phase25,"PARENT_SHA",self.memory["sha256"]):
            with self.assertRaisesRegex(ValueError,"repeated benchmark seed"):
                replay25.evaluate(self.root/"result/parent.json",
                    self.root/"result/candidate.json",self.root/"result/decision.json",
                    [ids[0],ids[0],ids[1]],oods,self.root/"repeated")

    def test_candidate_remains_parent_when_training_rejected(self):
        state,_=replay25._judgement([],{"validation_pass":False})
        self.assertEqual(state,"BLOCKED_VALIDATION")

    def test_human_promotion_rejects_forged_or_incomplete_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            parent=self.memory
            adjusted=copy.deepcopy(parent["payload"]["phase22_router"]["feedback"]["readout"])
            adjusted[0][0]+=0.001
            with patch.object(phase25,"PARENT_SHA",parent["sha256"]):
                new=phase25._memory(phase25._updated_payload(
                    parent["payload"],adjusted,parent["sha256"]))
                decision={"schema":replay25.SCHEMA,"parent_sha256":parent["sha256"],
                          "candidate_sha256":new["sha256"],
                          "validation_pass":True,"candidate_is_parent":False}
                report={"schema":replay25.SCHEMA,"status":"PROMOTION_ELIGIBLE",
                        "parent_sha256":parent["sha256"],
                        "candidate_sha256":new["sha256"],
                        "decision_sha256":mesh.digest(decision),
                        "summary":{"passes_predeclared_gate":True},
                        "manual_promotion_required":True,"evaluations":[]}
                gate={"schema":replay25.SCHEMA,"status":"PROMOTION_ELIGIBLE",
                      "parent_sha256":parent["sha256"],
                      "candidate_sha256":new["sha256"],
                      "evaluation_sha256":mesh.digest(report),
                      "manual_review_required":True}
                for name,obj in (("parent.json",parent),("candidate.json",new),
                                 ("decision.json",decision),("evaluation.json",report),
                                 ("promotion-gate.json",gate)):
                    mesh.save(folder/name,obj)
                with self.assertRaisesRegex(ValueError,"insufficient independent unseen"):
                    phase25.check_promotion(
                        folder/"parent.json",folder/"candidate.json",
                        folder/"decision.json",folder/"evaluation.json",
                        folder/"promotion-gate.json")
                # Old Phase 2.5 schema is not a valid substitute for six-set evidence.
                decision["schema"]=phase25.SCHEMA
                mesh.save(folder/"decision.json",decision)
                with self.assertRaisesRegex(ValueError,"legacy single-seed"):
                    phase25.check_promotion(
                        folder/"parent.json",folder/"candidate.json",
                        folder/"decision.json",folder/"evaluation.json",
                        folder/"promotion-gate.json")



if __name__=="__main__":
    unittest.main()

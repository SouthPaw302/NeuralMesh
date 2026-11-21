"""Phase 2.5: model memory lineage, incremental training and promotion gating."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from neural_mesh import mesh, phase2, phase24, phase25


class NeuralMemoryEvolutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory()
        cls.root=Path(cls.temp.name)
        cls.parent_data=cls.root/"parent-data"
        phase2.prepare(64013,{"train":48,"validation":16,"test":32},cls.parent_data)
        cls.parentfile=cls.root/"parent.json"
        cls.parent=phase24.build_memory(
            cls.parent_data/"public.json",cls.parent_data/"weights.json",
            cls.parent_data/"trainval-labels.json",cls.parentfile)
        cls.new_data=cls.root/"new"
        phase2.prepare(2505,{"train":48,"validation":16,"test":32},cls.new_data)
        cls.candidate_dir=cls.root/"candidate"
        with patch.object(phase25,"PARENT_SHA",cls.parent["sha256"]):
            cls.decision=phase25.train_candidate(
                cls.parentfile,cls.new_data/"public.json",
                cls.new_data/"trainval-labels.json",
                cls.parent_data/"public.json",cls.parent_data/"trainval-labels.json",
                cls.candidate_dir)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_candidate_is_sealed_and_traceable(self):
        parent=phase24.restore_memory(self.candidate_dir/"parent.json")
        proposal=phase24.restore_memory(self.candidate_dir/"candidate.json")
        self.assertEqual(parent["sha256"],self.parent["sha256"])
        self.assertEqual(proposal["sha256"],self.decision["candidate_sha256"])
        self.assertEqual(len(self.decision["trial_metrics"]),len(phase25.RIDGES))
        self.assertTrue(self.decision["promotion_requires_manual_pr"])
        if self.decision["validation_pass"]:
            self.assertNotEqual(proposal["sha256"],parent["sha256"])
            self.assertEqual(proposal["payload"]["parent_sha256"],parent["sha256"])
            self.assertEqual(proposal["payload"]["version"],2)
            self.assertFalse(proposal["payload"]["adopted_automatically"])
        else:
            self.assertEqual(proposal["sha256"],parent["sha256"])

    def test_test_labels_cannot_be_supplied_to_training(self):
        fake=copy.deepcopy(mesh.load(self.new_data/"trainval-labels.json"))
        fake["labels"]["test"]=mesh.load(self.new_data/"test-labels.json")["labels"]["test"]
        testpath=self.root/"leaked-trainval.json"
        mesh.save(testpath,fake)
        with patch.object(phase25,"PARENT_SHA",self.parent["sha256"]):
            with self.assertRaisesRegex(ValueError,"label partitions"):
                phase25.train_candidate(self.parentfile,self.new_data/"public.json",testpath,
                    self.parent_data/"public.json",self.parent_data/"trainval-labels.json",
                    self.root/"rejected")

    def test_parent_readout_is_unchanged_by_candidate_fit(self):
        parent=copy.deepcopy(self.parent["payload"])
        proposed=phase24.restore_memory(self.candidate_dir/"candidate.json")["payload"]
        self.assertEqual(parent["models"],proposed["models"])
        self.assertEqual(parent["reliability"],proposed["reliability"])
        self.assertEqual(parent["phase23_selector"],proposed["phase23_selector"])

    def test_tampering_with_checkpoint_fails(self):
        if not self.decision["validation_pass"]:
            readout=copy.deepcopy(self.parent["payload"]["phase22_router"]["feedback"]["readout"])
            readout[0][0] += 0.0001
            tamper=phase25._memory(phase25._updated_payload(
                self.parent["payload"],readout,self.parent["sha256"]))
        else:
            tamper=mesh.load(self.candidate_dir/"candidate.json")
        tamper["payload"]["models"]["nodes"]["position"]["validation_mse"]+=1
        invalid=self.root/"tampered.json"
        mesh.save(invalid,tamper)
        with self.assertRaisesRegex(ValueError,"memory digest mismatch"):
            phase24.restore_memory(invalid)

    def test_validation_safe_guard_rejects_large_new_motion_degradation(self):
        report={"parent_mse":{n:1 for n in phase24.FAULTS},
                "candidate_mse":{n:1 for n in phase24.FAULTS},
                "parent_regimes":{"linear":{n:1 for n in phase24.FAULTS}},
                "candidate_regimes":{"linear":{n:1 for n in phase24.FAULTS}}}
        self.assertTrue(phase25._suite_guard(report,.04,.15))
        report["candidate_regimes"]["linear"]["corrupt"]=1.21
        self.assertFalse(phase25._suite_guard(report,.04,.15))
        report["candidate_regimes"]["linear"]["corrupt"]=1.
        report["candidate_mse"]["none"]=1.05
        self.assertFalse(phase25._suite_guard(report,.04,.15))

    def test_legacy_single_split_promotion_is_revoked(self):
        with tempfile.TemporaryDirectory() as tmp:
            loc=Path(tmp)
            if self.decision["validation_pass"]:
                candidate=mesh.load(self.candidate_dir/"candidate.json")
            else:
                updated=copy.deepcopy(self.parent["payload"]["phase22_router"]["feedback"]["readout"])
                updated[0][0]+=1e-5
                candidate=phase25._memory(phase25._updated_payload(
                    self.parent["payload"],updated,self.parent["sha256"]))
            mesh.save(loc/"candidate.json",candidate)
            mesh.save(loc/"parent.json",self.parent)
            decision={"schema":phase25.SCHEMA,"validation_pass":True,
                      "parent_sha256":self.parent["sha256"],
                      "candidate_sha256":candidate["sha256"]}
            report={"schema":phase25.SCHEMA,"status":"PROMOTION_ELIGIBLE",
                    "parent_sha256":self.parent["sha256"],
                    "candidate_sha256":candidate["sha256"],
                    "indistribution_guard_pass":True,"ood_guard_pass":True}
            gate={"schema":phase25.SCHEMA,"status":"PROMOTION_ELIGIBLE",
                  "parent_sha256":self.parent["sha256"],"candidate_sha256":candidate["sha256"],
                  "evaluation_sha256":mesh.digest(report),"manual_review_required":True,
                  "test_metrics_frozen":True}
            for filename,object_ in (("decision.json",decision),
                                     ("report.json",report),("gate.json",gate)):
                mesh.save(loc/filename,object_)
            with patch.object(phase25,"PARENT_SHA",self.parent["sha256"]):
                with self.assertRaisesRegex(ValueError,"legacy single-seed"):
                    phase25.check_promotion(
                        loc/"parent.json",loc/"candidate.json",
                        loc/"decision.json",loc/"report.json",loc/"gate.json")

    def test_parent_identity_is_pinned(self):
        with self.assertRaisesRegex(ValueError,"exact verified"):
            phase25.train_candidate(
                self.parentfile,self.new_data/"public.json",
                self.new_data/"trainval-labels.json",
                self.parent_data/"public.json",
                self.parent_data/"trainval-labels.json",
                self.root/"unsafe")


if __name__=="__main__":
    unittest.main()

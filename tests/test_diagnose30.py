"""Diagnostic and scientific-gate tests using training/validation only."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from neural_mesh import diagnose30, mesh, phase2, phase24, research_gate, safety27, transfer26


class ObservabilityDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory()
        cls.root=Path(cls.tmp.name)
        cls.data=cls.root/"id"
        cls.known_train=cls.root/"novel_train"
        cls.known_val=cls.root/"novel_val"
        phase2.prepare(30211,{"train":32,"validation":8,"test":8},cls.data)
        transfer26.make_subset(50211,16,transfer26.KNOWN,cls.known_train)
        transfer26.make_subset(50212,16,transfer26.KNOWN,cls.known_val)
        cls.parent_file=cls.root/"parent.json"
        cls.parent=phase24.build_memory(
            cls.data/"public.json",cls.data/"weights.json",
            cls.data/"trainval-labels.json",cls.parent_file)
        payload={"schema":transfer26.SCHEMA,
                 "parent_sha256":cls.parent["sha256"],
                 "model":{"theta":[[0.]*transfer26.FEATURE_DIM
                                   for _ in transfer26.EXPERTS],
                          "means":[0.]*transfer26.FEATURE_DIM,
                          "scales":[1.]*transfer26.FEATURE_DIM,
                          "strength":.25}}
        cls.adapter={"payload":payload,"sha256":mesh.digest(payload)}
        cls.adapter_file=cls.root/"adapter.json"
        mesh.save(cls.adapter_file,cls.adapter)
        with patch.object(safety27,"PARENT_SHA",cls.parent["sha256"]),patch.object(
            safety27,"BASE_ADAPTER_SHA",cls.adapter["sha256"]):
            cls.report=diagnose30.run(
                cls.parent_file,cls.adapter_file,
                cls.data/"public.json",cls.data/"trainval-labels.json",
                cls.known_train/"public.json",cls.known_train/"targets.json",
                cls.known_val/"public.json",cls.known_val/"targets.json",
                cls.root/"diagnostic.json")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_oracles_are_diagnostic_and_information_sources_separate(self):
        r=self.report
        self.assertTrue(r["no_test_labels_seen"])
        self.assertTrue(r["diagnostic_oracles_not_deployable"])
        self.assertTrue(r["memory_unchanged"])
        self.assertEqual(r["training_scenes"],32+16)
        self.assertEqual(r["validation_scenes"],8+16)
        self.assertEqual(set(r["regime_optimal_training_trust"]),set(diagnose30.REGIMES))
        self.assertTrue(all(0<=v<=1 for v in r["regime_optimal_training_trust"].values()))
        self.assertTrue(0<=r["observation_separability"]["validation_accuracy"]<=1)
        self.assertEqual(phase24.restore_memory(self.parent_file),self.parent)

    def test_per_regime_validation_oracle_is_mathematical_ceiling(self):
        trainopt=self.report["validation_training_oracle_regime_trust"]
        valopt=self.report["validation_infeasible_label_oracle_ceiling"]
        for regime in diagnose30.REGIMES:
            self.assertLessEqual(valopt["by_regime"][regime],
                                 trainopt["by_regime"][regime]+1e-10)

    def test_scientific_gate_blocks_green_ci_with_rejected_model(self):
        rejected={"schema":"neuralmesh-competence-tree-v1",
                  "status":"NO_VALIDATED_COMPETENCE_MAP",
                  "parent_model_frozen":True,
                  "familiar_regime_retention_pass":False,
                  "known_motion_gain_pass":False,
                  "unseen_transfer_pass":False,
                  "fault_safety_pass":True,
                  "candidate_sha256":"a"*64}
        gate={"status":rejected["status"],
              "evaluation_sha256":mesh.digest(rejected)}
        path=self.root/"rejected.json"
        gpath=self.root/"blocked.json"
        mesh.save(path,rejected)
        mesh.save(gpath,gate)
        with self.assertRaisesRegex(ValueError,"LEARNING_NOT_ACCEPTED"):
            research_gate.require_accepted(path,gpath)

    def test_valid_evidence_may_open_human_review_but_never_automerge(self):
        accepted={"status":"RETENTION_REPAIR_SUPPORTED",
                  "parent_model_frozen":True,
                  "familiar_regime_retention_pass":True,
                  "known_motion_gain_pass":True,
                  "unseen_transfer_pass":True,
                  "fault_safety_pass":True,
                  "candidate_sha256":"b"*64}
        gate={"status":accepted["status"],
              "evaluation_sha256":mesh.digest(accepted)}
        path=self.root/"accepted.json"
        gpath=self.root/"proof.json"
        mesh.save(path,accepted)
        mesh.save(gpath,gate)
        out=research_gate.require_accepted(path,gpath)
        self.assertEqual(out["status"],"ACCEPTED_FOR_HUMAN_REVIEW_ONLY")


if __name__=="__main__":
    unittest.main()

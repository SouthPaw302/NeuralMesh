"""Phase 2.3 regression, split isolation, artifact integrity, and seed aggregation."""
import copy
import math
import tempfile
import unittest
from pathlib import Path

from neural_mesh import mesh, phase2, phase23


class Phase23Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.path = Path(cls.temp.name)
        cls.dataset = cls.path / "dataset"
        phase2.prepare(2303, {"train": 48, "validation": 16, "test": 32}, cls.dataset)
        cls.selector = phase23.fit_seed(
            cls.dataset/"public.json",cls.dataset/"weights.json",
            cls.dataset/"trainval-labels.json",cls.path/"trained")
        cls.rows=mesh.load(cls.path/"trained/rows.json")
        cls.base=mesh.load(cls.path/"trained/phase22-router.json")
        cls.test_labels=mesh.load(cls.dataset/"test-labels.json")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_frozen_gate_is_finite_and_normalized(self):
        self.assertEqual(self.selector["schema"],phase23.SCHEMA)
        self.assertTrue(self.selector["frozen_before_test"])
        self.assertIn(self.selector["selected"]["kind"],("off","full","gated"))
        self.assertEqual(len(self.selector["candidate_scores"]),10)
        row=self.rows["batches"]["test"][0]
        x=phase23._vector(row,self.base)
        self.assertEqual(len(x),phase23.FEATURES)
        self.assertTrue(all(math.isfinite(v) for v in x))
        g=phase23.activation(x,self.selector["selected"])
        self.assertTrue(0 <= g <= 1)

    def test_held_out_scores_report_regime_harm(self):
        result=phase23.evaluate_seed(
            2303, self.path/"trained/rows.json",
            self.path/"trained/phase22-router.json",
            self.path/"trained/phase23-selector.json",
            self.dataset/"test-labels.json",self.path/"seed-2303.json")
        self.assertEqual(result["test_scenes"],32)
        self.assertEqual(set(result["regime_mse"]),set(phase2.REGIMES))
        self.assertEqual(len(result["test_mse"]),4)
        self.assertEqual(len(result["paired_selective_minus_full"]["ci95"]),2)
        for val in result["regime_feedback_strength"].values():
            self.assertTrue(0 <= val <= 1)
        same=phase23.evaluate_seed(
            2303, self.path/"trained/rows.json",
            self.path/"trained/phase22-router.json",
            self.path/"trained/phase23-selector.json",
            self.dataset/"test-labels.json",self.path/"seed-2303.json")
        self.assertEqual(result,same)

    def test_cannot_fit_with_test_targets(self):
        with self.assertRaisesRegex(ValueError,"label partitions"):
            phase23.fit(self.rows,self.base,self.test_labels)
        leaked=mesh.load(self.dataset/"trainval-labels.json")
        leaked["labels"]["test"]=self.test_labels["labels"]["test"]
        with self.assertRaisesRegex(ValueError,"label partitions"):
            phase23.fit(self.rows,self.base,leaked)

    def test_bad_router_and_test_partition_fail(self):
        fake=copy.deepcopy(self.selector)
        fake["round1_sha256"]="0"*64
        with self.assertRaisesRegex(ValueError,"observations mismatch"):
            phase23._score_seed(self.rows,self.base,fake,self.test_labels)
        fake=copy.deepcopy(self.selector)
        fake["base_router_sha256"]="bad"
        with self.assertRaisesRegex(ValueError,"hash mismatch"):
            phase23._score_seed(self.rows,self.base,fake,self.test_labels)
        targets=copy.deepcopy(self.test_labels)
        targets["labels"]["train"]={}
        with self.assertRaisesRegex(ValueError,"label partitions"):
            phase23._score_seed(self.rows,self.base,self.selector,targets)

    def test_hard_off_and_full_ablation(self):
        v=phase23._vector(self.rows["batches"]["test"][0],self.base)
        self.assertEqual(phase23.activation(v,{"kind":"off"}),0.)
        self.assertEqual(phase23.activation(v,{"kind":"full"}),1.)
        with self.assertRaisesRegex(ValueError,"invalid gate"):
            phase23.activation(v,{"kind":"unsafe"})

    def test_aggregation_rejects_missing_and_duplicate_seeds(self):
        sample=mesh.load(self.path/"seed-2303.json")
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)
            for seed in (17,23,29):
                item=copy.deepcopy(sample)
                item["seed"]=seed
                item["config_sha256"]=f"hash-{seed}"
                mesh.save(directory/f"seed-{seed}.json",item)
            report=phase23.aggregate(directory,(17,23,29),directory/"summary.json")
            self.assertEqual(report["seed_count"],3)
            self.assertEqual(len(report["per_seed"]),3)
            with self.assertRaisesRegex(ValueError,"unique seed ids"):
                phase23.aggregate(directory,(17,17,29),directory/"other.json")
            (directory/"seed-23.json").unlink()
            with self.assertRaisesRegex(ValueError,"expected exactly"):
                phase23.aggregate(directory,(17,23,29),directory/"other.json")


if __name__=="__main__":
    unittest.main()

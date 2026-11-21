"""Phase 2.4: OOD, node failure, and versioned memory regression tests."""
import copy
import tempfile
import unittest
from pathlib import Path

from neural_mesh import mesh, phase2, phase24


class Phase24Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.train = cls.root/"train"
        phase2.prepare(2404, {"train":48,"validation":16,"test":32},cls.train)
        cls.memory_file=cls.root/"memory.json"
        cls.state=phase24.build_memory(cls.train/"public.json",cls.train/"weights.json",
                                       cls.train/"trainval-labels.json",cls.memory_file)
        cls.ood=cls.root/"ood"
        phase24.prepare_ood(18931,32,cls.ood)
        cls.public=mesh.load(cls.ood/"public.json")
        cls.targets=mesh.load(cls.ood/"sealed-targets.json")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_checkpoint_immutable_and_restorable(self):
        self.assertEqual(phase24.restore_memory(self.memory_file),self.state)
        self.assertEqual(self.state["payload"]["schema"], phase24.MEMORY_SCHEMA)
        self.assertEqual(self.state["payload"]["adopted_automatically"],False)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"restored.json"
            mesh.save(p,self.state)
            self.assertEqual(phase24.restore_memory(p)["sha256"],self.state["sha256"])

    def test_memory_digest_fails_when_model_modified(self):
        wrong=copy.deepcopy(self.state)
        wrong["payload"]["models"]["nodes"]["position"]["validation_mse"]+=1
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"tampered.json"
            mesh.save(p,wrong)
            with self.assertRaisesRegex(ValueError,"memory digest mismatch"):
                phase24.restore_memory(p)

    def test_ood_is_new_physics_and_no_target_leak(self):
        self.assertEqual(set(s["regime"] for s in self.public["scenes"]),set(phase24.REGIMES))
        self.assertNotIn("targets",str(self.public))
        self.assertNotIn("future_position",str(self.public))
        self.assertEqual(self.targets["public_sha256"],mesh.digest(self.public))
        self.assertEqual(len(self.public["scenes"]),32)

    def test_missing_node_or_corruption_is_detected(self):
        row=phase24.make_row(self.public["scenes"][0],self.state["payload"]["models"])
        missing=phase24.inject(row,"missing",0)
        pred,nodes=phase24.robust_prediction(missing,self.state["payload"],.45,.5)
        self.assertEqual(len(nodes),5)
        self.assertEqual(len(pred),2)
        damaged=phase24.inject(row,"corrupt",0)
        pred,nodes=phase24.robust_prediction(damaged,self.state["payload"],.45,.5)
        self.assertLess(len(nodes),6)
        self.assertNotIn(mesh.NODES[0],nodes)
        self.assertTrue(all(abs(p)<100 for p in pred))

    def test_probe_and_sealed_evaluation(self):
        packet=phase24.probe(self.public,self.state)
        self.assertEqual(packet["memory_sha256"],self.state["sha256"])
        report=phase24.evaluate_probe(packet,self.targets)
        self.assertEqual(report["test_scenes"],32)
        self.assertEqual(set(report["by_regime"]),set(phase24.REGIMES))
        self.assertEqual(set(report["test_mse"]),set(phase24.FAULTS))
        for fault in phase24.FAULTS:
            self.assertGreaterEqual(report["test_mse"][fault]["naive"],0)
            self.assertGreaterEqual(report["test_mse"][fault]["robust"],0)
        self.assertEqual(report,phase24.evaluate_probe(packet,self.targets))

    def test_faulty_ood_label_and_bad_seed_fail(self):
        packet=phase24.probe(self.public,self.state)
        wrong=copy.deepcopy(self.targets)
        wrong["seed"]+=1
        with self.assertRaisesRegex(ValueError,"seed/public hash mismatch"):
            phase24.evaluate_probe(packet,wrong)
        wrong=copy.deepcopy(self.targets)
        wrong["targets"].pop(next(iter(wrong["targets"])))
        with self.assertRaisesRegex(ValueError,"targets"):
            phase24.evaluate_probe(packet,wrong)

    def test_aggregation_requires_three_seeds_and_same_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            packet=phase24.probe(self.public,self.state)
            base=phase24.evaluate_probe(packet,self.targets)
            for seed in (31,37,41):
                value=copy.deepcopy(base)
                value["seed"]=seed
                mesh.save(folder/f"seed-{seed}.json",value)
            merged=phase24.aggregate(folder,(31,37,41),folder/"summary.json")
            self.assertEqual(merged["test_scenes"],96)
            self.assertEqual(set(merged["fault_robustness_wins"]),set(phase24.FAULTS))
            bad=copy.deepcopy(base)
            bad["seed"]=37
            bad["memory_sha256"]="0"*64
            mesh.save(folder/"seed-37.json",bad)
            with self.assertRaisesRegex(ValueError,"mixed memory checkpoint"):
                phase24.aggregate(folder,(31,37,41),folder/"fail.json")


if __name__=="__main__":
    unittest.main()

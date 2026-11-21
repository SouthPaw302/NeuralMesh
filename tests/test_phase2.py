"""Phase 2.1 split isolation, plasticity, runner receipt, and benchmark tests."""
import copy
import tempfile
import unittest
from pathlib import Path

from neural_mesh import mesh, phase2


class Phase21Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        phase2.prepare(1741, {"train": 48, "validation": 16, "test": 32}, cls.root)
        cls.public = mesh.load(cls.root / "public.json")
        cls.weights = mesh.load(cls.root / "weights.json")
        cls.trv = mesh.load(cls.root / "trainval-labels.json")
        cls.testlabels = mesh.load(cls.root / "test-labels.json")
        cls.receipts = cls.root / "receipts"
        for node in mesh.NODES:
            mesh.save(cls.receipts / f"node-{node}.json",
                      phase2.infer(node, cls.public, cls.weights))
        cls.first = phase2.collect(cls.public, cls.weights, cls.receipts)
        cls.router = phase2.train_router(cls.first, cls.weights, cls.trv)
        cls.refined = cls.root / "refined"
        for node in mesh.NODES:
            mesh.save(cls.refined / f"node-{node}.json",
                      phase2.refine(node, cls.first, cls.router))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_three_nonoverlapping_splits_and_sealed_test_targets(self):
        ids = [{x["id"] for x in self.public["scenes"][split]}
               for split in phase2.SPLITS]
        self.assertEqual(len(ids[0] | ids[1] | ids[2]), 96)
        self.assertFalse(ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
        self.assertEqual(set(self.trv["labels"]), {"train", "validation"})
        self.assertEqual(set(self.testlabels["labels"]), {"test"})
        self.assertNotIn("future_position", str(self.public))
        self.assertNotIn("test", self.router.get("search", {}))
        self.assertEqual({r["regime"] for r in self.public["scenes"]["test"]},
                         set(phase2.REGIMES))

    def test_full_synthetic_benchmark_and_adaptive_checkpoint(self):
        report = phase2.evaluate(self.first, self.router, self.refined,
                                 self.weights, self.testlabels)
        self.assertEqual(report["test_scenes"], 32)
        self.assertEqual(report["train_scenes"], 48)
        self.assertEqual(report["validation_scenes"], 16)
        self.assertEqual(len(report["per_regime"]), 4)
        self.assertTrue(all(v >= 0 for v in report["models"].values()))
        self.assertEqual(len(report["paired_difference_vs_v0_round2"]["ci95"]), 2)
        self.assertAlmostEqual(sum(self.router["learned"]["weights"]), 1.0)
        self.assertEqual(self.router["frozen_before_test"], True)
        self.assertEqual(set(self.router["learned"]), {"weights", "ridge", "gamma", "alpha",
                                                       "validation_mse"})
        self.assertEqual(report, phase2.evaluate(self.first, self.router,
                                                self.refined, self.weights, self.testlabels))

    def test_training_must_never_receive_test_labels(self):
        leaked = copy.deepcopy(self.trv)
        leaked["labels"]["test"] = self.testlabels["labels"]["test"]
        with self.assertRaisesRegex(ValueError, "label partitions"):
            phase2.train_router(self.first, self.weights, leaked)
        leaked = copy.deepcopy(self.testlabels)
        leaked["labels"]["train"] = self.trv["labels"]["train"]
        with self.assertRaisesRegex(ValueError, "label partitions"):
            phase2.evaluate(self.first, self.router, self.refined, self.weights, leaked)

    def test_wrong_model_or_missing_worker_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            for node in mesh.NODES[:-1]:
                mesh.save(folder / f"node-{node}.json",
                          phase2.infer(node, self.public, self.weights))
            with self.assertRaisesRegex(ValueError, "exactly six"):
                phase2.collect(self.public, self.weights, folder)
            forged = phase2.infer(mesh.NODES[-1], self.public, self.weights)
            forged["model_sha256"] = "0" * 64
            mesh.save(folder / f"node-{mesh.NODES[-1]}.json", forged)
            with self.assertRaisesRegex(ValueError, "model mismatch"):
                phase2.collect(self.public, self.weights, folder)

    def test_refined_output_not_accepted_if_forged(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            for node in mesh.NODES:
                output = phase2.refine(node, self.first, self.router)
                if node == "position":
                    output["predictions"]["test"][0]["prediction"][0] += 42
                mesh.save(folder / f"node-{node}.json", output)
            with self.assertRaisesRegex(ValueError, "not derived from frozen"):
                phase2.evaluate(self.first, self.router, folder,
                                self.weights, self.testlabels)

    def test_input_hash_and_seed_are_reproducible(self):
        with tempfile.TemporaryDirectory() as tmp:
            phase2.prepare(1741, {"train": 48, "validation": 16, "test": 32}, tmp)
            self.assertEqual((Path(tmp) / "public.json").read_bytes(),
                             (self.root / "public.json").read_bytes())
            self.assertEqual((Path(tmp) / "weights.json").read_bytes(),
                             (self.root / "weights.json").read_bytes())
        changed = copy.deepcopy(self.public)
        changed["scenes"]["test"][0]["tracks"]["position"][0][0] += 0.1
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            for n in mesh.NODES:
                mesh.save(folder / f"node-{n}.json",
                          phase2.infer(n, self.public, self.weights))
            with self.assertRaisesRegex(ValueError, "observation mismatch"):
                phase2.collect(changed, self.weights, folder)

    def test_routing_requires_unchanged_prediction_packet(self):
        altered = copy.deepcopy(self.first)
        altered["batches"]["test"][0]["predictions"]["velocity"][0] += 0.01
        with self.assertRaisesRegex(ValueError, "router not tied"):
            phase2.refine("position", altered, self.router)

    def test_simplex_optimization_is_nonnegative(self):
        w = phase2._project_simplex([0.9, -3., 2., 1., -0.5, 0.8])
        self.assertAlmostEqual(sum(w), 1.0)
        self.assertTrue(all(x >= 0 for x in w))


if __name__ == "__main__":
    unittest.main()

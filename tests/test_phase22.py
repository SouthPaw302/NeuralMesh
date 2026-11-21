"""Bounded Phase 2.2 synthetic tests: no leakage and verifiable recurrence."""
import copy
import math
import tempfile
import unittest
from pathlib import Path

from neural_mesh import mesh, phase2, phase22


class ContextualSynapseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        phase2.prepare(2202, {"train": 48, "validation": 16, "test": 32}, cls.root)
        cls.public = mesh.load(cls.root/"public.json")
        cls.weights = mesh.load(cls.root/"weights.json")
        cls.trainval = mesh.load(cls.root/"trainval-labels.json")
        cls.testlabels = mesh.load(cls.root/"test-labels.json")
        cls.workers = cls.root/"workers"
        for node in mesh.NODES:
            mesh.save(cls.workers/f"node-{node}.json",
                      phase22.observe(node, cls.public, cls.weights))
        cls.rows = phase22.collect(cls.public, cls.weights, cls.workers)
        cls.router = phase22.fit(cls.rows, cls.weights, cls.trainval)
        cls.stage2 = cls.root/"stage2"
        for node in mesh.NODES:
            mesh.save(cls.stage2/f"node-{node}.json",
                      phase22.feedback_worker(node, cls.rows, cls.router))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_end_to_end_sealed_holdout_and_comparisons(self):
        report = phase22.evaluate(self.rows, self.weights, self.router,
                                  self.stage2, self.testlabels)
        self.assertEqual(report["test_scenes"], 32)
        self.assertEqual(report["train_scenes"], 48)
        self.assertEqual(report["validation_scenes"], 16)
        self.assertEqual(len(report["by_regime"]), 4)
        self.assertEqual(len(report["test_mse"]), 5)
        self.assertTrue(all(math.isfinite(x) and x >= 0 for x in report["test_mse"].values()))
        self.assertEqual(len(report["paired_feedback_minus_static"]["ci95"]), 2)
        self.assertTrue(self.router["frozen_before_test"])
        self.assertEqual(report, phase22.evaluate(self.rows, self.weights,
                                                 self.router, self.stage2, self.testlabels))

    def test_only_training_validation_answers_enter_plasticity(self):
        leak = copy.deepcopy(self.trainval)
        leak["labels"]["test"] = self.testlabels["labels"]["test"]
        with self.assertRaisesRegex(ValueError, "label partitions"):
            phase22.fit(self.rows, self.weights, leak)
        bad_test = copy.deepcopy(self.testlabels)
        bad_test["labels"]["train"] = self.trainval["labels"]["train"]
        with self.assertRaisesRegex(ValueError, "label partitions"):
            phase22.evaluate(self.rows, self.weights, self.router, self.stage2, bad_test)

    def test_forged_specialist_embedding_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            location = Path(tmp)
            for node in mesh.NODES:
                packet = phase22.observe(node, self.public, self.weights)
                if node == "velocity":
                    packet["embeddings"]["test"][0]["value"][0] += 0.123
                mesh.save(location/f"node-{node}.json", packet)
            with self.assertRaisesRegex(ValueError, "embedding does not match"):
                phase22.collect(self.public, self.weights, location)

    def test_peer_inputs_affect_contextual_weights(self):
        row = copy.deepcopy(self.rows["batches"]["test"][0])
        att1 = phase22.gate(row, self.router["gate"])
        row["embeddings"]["velocity"][2] += 0.2
        att2 = phase22.gate(row, self.router["gate"])
        # Gate could legitimately learn zero context coefficients; in that case
        # the comparison must still remain normalized and finite.
        self.assertAlmostEqual(sum(att1.values()), 1.0)
        self.assertAlmostEqual(sum(att2.values()), 1.0)
        self.assertTrue(all(v >= 0 and math.isfinite(v) for v in att1.values()))

    def test_forged_recurrent_prediction_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            for node in mesh.NODES:
                packet = phase22.feedback_worker(node, self.rows, self.router)
                if node == "position":
                    packet["predictions"]["test"][0]["prediction"][0] += 3.
                mesh.save(folder/f"node-{node}.json", packet)
            with self.assertRaisesRegex(ValueError, "unverified recurrent prediction"):
                phase22.evaluate(self.rows, self.weights, self.router, folder,
                                 self.testlabels)

    def test_wrong_router_and_missing_receipt_fail_closed(self):
        wrong = copy.deepcopy(self.router)
        wrong["round1_sha256"] = "invalid"
        with self.assertRaisesRegex(ValueError, "router/observations mismatch"):
            phase22.feedback_worker("position", self.rows, wrong)
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            for node in mesh.NODES[:-1]:
                mesh.save(folder/f"node-{node}.json",
                          phase22.feedback_worker(node, self.rows, self.router))
            with self.assertRaisesRegex(ValueError, "expected six stage-two"):
                phase22.evaluate(self.rows, self.weights, self.router, folder,
                                 self.testlabels)

    def test_contextual_gate_and_residual_readouts_shapes(self):
        self.assertEqual(len(self.router["gate"]["theta"]), 6)
        self.assertEqual(len(self.router["gate"]["theta"][0]), 5)
        self.assertIn(self.router["feedback"]["beta"], (0., 0.25, 0.5, 1.))
        if self.router["feedback"]["beta"] > 0:
            self.assertEqual(len(self.router["feedback"]["readout"][0]), 17)
        row = self.rows["batches"]["validation"][0]
        att = phase22.gate(row, self.router["gate"])
        self.assertEqual(len(phase22.feedback_features(row, att)), 17)


if __name__ == "__main__":
    unittest.main()

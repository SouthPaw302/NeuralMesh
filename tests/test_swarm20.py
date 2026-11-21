"""Real 20-node protocol tests; never substitute six logical nodes."""
import copy
import tempfile
import unittest
from pathlib import Path

from neural_mesh import mesh, swarm20


class ParallelTwentyWorkersTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = Path(cls.tmp.name)
        cls.state = cls.path/"state"
        swarm20.prepare(271828, cls.state)
        cls.obs = mesh.load(cls.state/"observation.json")
        cls.weights = mesh.load(cls.state/"weights.json")
        cls.truth = mesh.load(cls.state/"truth.json")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _receipts(self, stage, packet=None):
        folder = self.path/f"receipts-{stage}"
        folder.mkdir(exist_ok=True)
        for node in swarm20.NODES:
            receipt = swarm20.run_node(node, stage, self.obs, self.weights, packet)
            mesh.save(folder/f"node-{node}.json", receipt)
        return folder

    def test_twenty_distinct_trained_specialists(self):
        self.assertEqual(len(swarm20.NODES), 20)
        self.assertEqual(len(set(swarm20.NODES)), 20)
        self.assertEqual(len(self.obs["tracks"]), 20)
        self.assertEqual(len(self.weights["nodes"]), 20)
        self.assertNotIn("future_position", str(self.obs))
        for role in swarm20.ROLES:
            self.assertEqual(sum(swarm20.identity(n)[0] == role
                                 for n in swarm20.NODES), 4)
        hashes = {mesh.digest(model) for model in self.weights["nodes"].values()}
        self.assertEqual(len(hashes), 20, "all 20 models must be distinct")
        self.assertTrue(all(len(model["readout"]) == 2 and
                            len(model["readout"][0]) == swarm20.FEATURES
                            for model in self.weights["nodes"].values()))

    def test_all_to_all_nineteen_peers_in_each_wave(self):
        r1 = self._receipts(1)
        packet1 = swarm20.integrate(1, self.weights, r1)
        self.assertEqual(packet1["node_count"], 20)
        self.assertEqual(packet1["neighbors_per_node"], 19)
        self.assertEqual(len(packet1["routing"]), 20)
        for node in swarm20.NODES:
            peer = packet1["routing"][node]
            self.assertEqual(set(peer), set(swarm20.NODES)-{node})
            self.assertAlmostEqual(sum(peer.values()), 1.)
        r2 = self._receipts(2, packet1)
        packet2 = swarm20.integrate(2, self.weights, r2, packet1)
        self.assertNotEqual(packet1["predictions"], packet2["predictions"])
        scores, child = swarm20.evaluate(packet1, packet2, self.truth, self.weights)
        self.assertEqual(scores["physical_workers_per_round"], 20)
        self.assertEqual(scores["inference_job_executions"], 40)
        self.assertEqual(scores["peers_per_worker"], 19)
        self.assertIn("learned_routing", child)
        self.assertEqual(len(child["learned_routing"]), 20)

    def test_one_missing_receipt_fails_closed(self):
        r1 = self._receipts(1)
        (r1/f"node-{swarm20.NODES[-1]}.json").unlink()
        with self.assertRaisesRegex(ValueError,"expected 20 worker receipts"):
            swarm20.integrate(1, self.weights, r1)

    def test_rejects_duplicate_worker_which_masks_a_missing_one(self):
        r1 = self._receipts(1)
        forged=mesh.load(r1/f"node-{swarm20.NODES[0]}.json")
        forged["node"]=swarm20.NODES[1]
        mesh.save(r1/f"node-{swarm20.NODES[0]}.json",forged)
        with self.assertRaisesRegex(ValueError,"duplicate/missing/spoofed"):
            swarm20.integrate(1,self.weights,r1)

    def test_rejects_cross_input_or_fake_model_digest(self):
        r1=self._receipts(1)
        first=r1/f"node-{swarm20.NODES[0]}.json"
        forged=mesh.load(first)
        forged["model_sha256"]="0"*64
        mesh.save(first,forged)
        with self.assertRaisesRegex(ValueError,"receipt"):
            swarm20.integrate(1,self.weights,r1)
        forged["model_sha256"]=mesh.digest(self.weights["nodes"][swarm20.NODES[0]])
        forged["input_sha256"]="1"*64
        mesh.save(first,forged)
        with self.assertRaisesRegex(ValueError,"digests disagree"):
            swarm20.integrate(1,self.weights,r1)

    def test_feedback_requires_19_verified_neighbors(self):
        pkt1=swarm20.integrate(1,self.weights,self._receipts(1))
        invalid=copy.deepcopy(pkt1)
        invalid["routing"][swarm20.NODES[0]].pop(swarm20.NODES[1])
        with self.assertRaisesRegex(ValueError,"19-peer"):
            swarm20.run_node(swarm20.NODES[0],2,self.obs,self.weights,invalid)
        with self.assertRaisesRegex(ValueError,"stage 2 needs"):
            swarm20.run_node(swarm20.NODES[0],2,self.obs,self.weights)

    def test_reproducibility_and_separate_sealed_target(self):
        with tempfile.TemporaryDirectory() as other:
            swarm20.prepare(271828, other)
            for filename in ("weights.json","observation.json","truth.json"):
                self.assertEqual(
                    (self.state/filename).read_bytes(),
                    (Path(other)/filename).read_bytes())

    def test_local_end_to_end_check(self):
        with tempfile.TemporaryDirectory() as temp:
            r=swarm20.local_proof(333111,temp)
            self.assertEqual(r["physical_workers_per_round"],20)
            self.assertIsInstance(r["recurrent_feedback_helped"],bool)
            self.assertTrue((Path(temp)/"candidate-checkpoint.json").exists())


if __name__=="__main__":
    unittest.main()

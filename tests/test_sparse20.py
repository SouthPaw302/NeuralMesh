"""Contract tests for real 20-way local learning and sparse cooperation."""
import copy
import tempfile
import unittest
from pathlib import Path

from neural_mesh import mesh, sparse20, swarm20


class SparseTwentyLearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory()
        cls.dir=Path(cls.tmp.name)
        cls.config=sparse20.prepare(801001,cls.dir/"prepare")
        cls.config["train_labeled_scenes_per_node"]=24
        cls.config["update_labeled_scenes_per_node"]=12
        cls.folders={}
        first=cls.dir/"first"
        first.mkdir()
        cls.receipts={}
        for node in sparse20.NODES:
            r=sparse20.first(node,cls.config)
            cls.receipts[node]=r
            mesh.save(first/f"node-{node}.json",r)
        cls.folders[1]=first
        cls.route=sparse20.aggregate_first(cls.config,first)
        second=cls.dir/"second"
        second.mkdir()
        cls.children={}
        for node in sparse20.NODES:
            r=sparse20.second(node,cls.config,cls.receipts[node],cls.route)
            cls.children[node]=r
            mesh.save(second/f"node-{node}.json",r)
        cls.folders[2]=second

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_20_independent_models_training_on_local_data(self):
        self.assertEqual(len(sparse20.NODES),20)
        self.assertEqual(len(self.receipts),20)
        self.assertEqual({sparse20._verified(r)["node"] for r in self.receipts.values()},
                         set(sparse20.NODES))
        self.assertEqual(len({
            mesh.digest(sparse20._verified(r)["trained_model"])
            for r in self.receipts.values()}),20)
        for node,r in self.receipts.items():
            record=sparse20._verified(r)
            self.assertEqual(record["local_labeled_training"],24)
            self.assertEqual(record["stage"],1)
            self.assertNotEqual(record["initial_model_sha256"],
                                mesh.digest(record["trained_model"]))
            self.assertEqual(len(record["probes"]),sparse20.PROBES)

    def test_five_distinct_reliability_weighted_peers_per_node(self):
        package=sparse20._verified(self.route)
        self.assertEqual(package["node_count"],20)
        self.assertEqual(package["peers_per_node"],5)
        for node,peers in package["routing"].items():
            self.assertEqual(len(peers),5)
            self.assertNotIn(node,peers)
            self.assertEqual(set(swarm20.identity(p)[0] for p in peers),
                             set(swarm20.ROLES))
            self.assertAlmostEqual(sum(peers.values()),1.0)

    def test_peer_distillation_changes_local_weights_with_identical_labeled_stream(self):
        changed=0
        for node,receipt in self.children.items():
            record=sparse20._verified(receipt)
            self.assertEqual(record["local_labeled_update"],12)
            self.assertEqual(record["unlabelled_peer_experiences"],32)
            self.assertTrue(record["communicated"])
            self.assertEqual(len(record["selected_peers"]),5)
            if record["social_model"]!=record["control_model"]:
                changed+=1
        self.assertEqual(changed,20)

    def test_no_peer_ablation_matches_independent_control(self):
        node=sparse20.NODES[0]
        ablation=sparse20._verified(sparse20.second(
            node,self.config,self.receipts[node],self.route,distill=False))
        self.assertEqual(ablation["social_model"],ablation["control_model"])
        self.assertFalse(ablation["communicated"])

    def test_untrusted_or_missing_recipients_fail_closed(self):
        missing=self.dir/"missing"
        missing.mkdir()
        for node,r in list(self.receipts.items())[1:]:
            mesh.save(missing/f"node-{node}.json",r)
        with self.assertRaisesRegex(ValueError,"expected 20"):
            sparse20.aggregate_first(self.config,missing)
        corrupt=copy.deepcopy(self.route)
        corrupt["payload"]["routing"][sparse20.NODES[0]].popitem()
        corrupt["sha256"]=mesh.digest(corrupt["payload"])
        with self.assertRaisesRegex(ValueError,"invalid sparse"):
            sparse20.second(sparse20.NODES[0],self.config,
                            self.receipts[sparse20.NODES[0]],corrupt)

    def test_no_stale_cross_generation_peer_packets(self):
        changed=copy.deepcopy(self.config)
        changed["generation"]+=1
        with self.assertRaisesRegex(ValueError,"routing provenance"):
            sparse20.second(sparse20.NODES[0],changed,
                            self.receipts[sparse20.NODES[0]],self.route)

    def test_sealed_three_seed_eval_returns_symmetric_ablation(self):
        report,checkpoint=sparse20.finish(
            self.config,self.folders[1],self.folders[2],self.route)
        self.assertEqual(report["node_count"],20)
        self.assertEqual(report["peers_per_node"],5)
        self.assertEqual(report["independent_learning_control"]["scene_count"],480)
        self.assertEqual(set(report["sparse_peer_learning"]["per_seed_mse"]),
                         set(map(str,sparse20.TEST_SEEDS)))
        self.assertTrue(report["trained_models_differ"])
        self.assertTrue(report["candidate_only_not_promoted"])
        self.assertIsInstance(report["beat_independent_control"],bool)
        self.assertIsInstance(report["beat_best_individual"],bool)
        self.assertEqual(len(sparse20._verified(checkpoint)["nodes"]),20)
        self.assertEqual(sparse20._verified(checkpoint)["generation"],1)

    def test_checkpoint_persists_between_executions_and_resumes_generation(self):
        _,checkpoint=sparse20.finish(
            self.config,self.folders[1],self.folders[2],self.route)
        file=self.dir/"checkpoint.json"
        mesh.save(file,checkpoint)
        next_config=sparse20.prepare(801002,self.dir/"resume",str(file))
        self.assertEqual(next_config["generation"],2)
        self.assertEqual(next_config["parent_sha256"],checkpoint["sha256"])
        next_first=sparse20._verified(sparse20.first(
            sparse20.NODES[0],next_config,str(file)))
        self.assertEqual(next_first["initial_model_sha256"],
            mesh.digest(sparse20._verified(checkpoint)["nodes"][sparse20.NODES[0]]))
        bad=copy.deepcopy(checkpoint)
        bad["payload"]["nodes"][sparse20.NODES[0]][0][0]=99
        mesh.save(self.dir/"bad.json",bad)
        with self.assertRaisesRegex(ValueError,"invalid node/model digest"):
            sparse20.prepare(801003,self.dir/"bad-resume",str(self.dir/"bad.json"))

    def test_features_are_shared_predictions_not_unaligned_hidden_state(self):
        node=sparse20.NODES[0]
        first=sparse20._verified(self.receipts[node])
        self.assertEqual(len(first["probes"]),sparse20.PROBES)
        self.assertTrue(all(len(p)==2 for p in first["probes"]))
        self.assertNotIn("truth",first)
        self.assertNotIn("target",first)
        self.assertNotIn("test_labels",first)


if __name__=="__main__":
    unittest.main()

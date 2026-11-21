"""Temporal six-frame observability, source integrity and retention tests."""
import copy
import math
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from neural_mesh import mesh, phase2, phase24, temporal31, transfer26


class TemporalSixFrameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory()
        cls.root=Path(cls.tmp.name)
        cls.parent_data=cls.root/"original"
        phase2.prepare(64013,{"train":48,"validation":16,"test":16},cls.parent_data)
        cls.parent_file=cls.root/"parent.json"
        cls.parent=phase24.build_memory(
            cls.parent_data/"public.json",cls.parent_data/"weights.json",
            cls.parent_data/"trainval-labels.json",cls.parent_file)
        cls.adapter_payload={"schema":transfer26.SCHEMA,
                            "parent_sha256":cls.parent["sha256"],
                            "model":{"theta":[[0.]*transfer26.FEATURE_DIM
                                              for _ in transfer26.EXPERTS],
                                     "means":[0.]*transfer26.FEATURE_DIM,
                                     "scales":[1.]*transfer26.FEATURE_DIM,
                                     "strength":.25}}
        cls.adapter={"payload":cls.adapter_payload,
                     "sha256":mesh.digest(cls.adapter_payload)}
        cls.adapter_file=cls.root/"adapter.json"
        mesh.save(cls.adapter_file,cls.adapter)
        cls.training=cls.root/"training"
        cls.validation=cls.root/"validation"
        temporal31.make_data(31211,temporal31.TRAIN_CLASSES,48,cls.training)
        temporal31.make_data(31212,temporal31.TRAIN_CLASSES,24,cls.validation)
        cls.output=cls.root/"frozen"
        with patch.object(temporal31,"PARENT_SHA",cls.parent["sha256"]),patch.object(
            temporal31,"ADAPTER_SHA",cls.adapter["sha256"]):
            cls.decision=temporal31.train(
                cls.parent_file,cls.adapter_file,
                cls.training/"public.json",cls.training/"targets.json",
                cls.validation/"public.json",cls.validation/"targets.json",
                cls.output)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_all_eight_regimes_provide_seven_true_positions(self):
        for regime in temporal31.ID+temporal31.KNOWN+temporal31.UNKNOWN:
            points=temporal31.trajectory(random.Random(6),regime)
            self.assertEqual(len(points),7)
            self.assertTrue(all(len(p)==2 and all(math.isfinite(z) for z in p)
                                for p in points))

    def test_old_four_frame_predictor_and_new_six_frame_history_align(self):
        public=mesh.load(self.training/"public.json")
        targets=mesh.load(self.training/"targets.json")
        self.assertNotIn("targets",public)
        self.assertEqual(targets["public_sha256"],mesh.digest(public))
        scene=public["scenes"][0]
        for n in mesh.NODES:
            self.assertEqual(len(scene["history"][n]),6)
            self.assertEqual(len(scene["tracks"][n]),4)
            self.assertEqual(scene["tracks"][n],scene["history"][n][-4:])
        example=temporal31._example(scene,targets["targets"][scene["id"]],
                                   self.parent["payload"],
                                   self.adapter_payload["model"],"missing",0)
        self.assertEqual(len(example["four"]),9)
        self.assertEqual(len(example["six"]),18)
        self.assertEqual(len(example["parent"]),2)
        self.assertEqual(len(example["adapter"]),2)

    def test_classification_is_observation_only(self):
        public=mesh.load(self.training/"public.json")
        targets=mesh.load(self.training/"targets.json")
        scene=public["scenes"][3]
        x=temporal31._example(scene,targets["targets"][scene["id"]],
                              self.parent["payload"],
                              self.adapter_payload["model"],"none",0)
        altered=copy.deepcopy(x)
        altered["truth"]=[99999.,-99999.]
        altered["regime"]="fake-secret-label"
        state=mesh.load(self.output/"temporal.json")
        for feature in ("four","six"):
            classifier=state["payload"]["classifiers"][feature]
            self.assertEqual(temporal31.classify(x,classifier),
                             temporal31.classify(altered,classifier))
            policy={"feature":feature,"margin":.05,
                    "veto":0.,"nominal":1.}
            self.assertEqual(temporal31._trust(x,policy,classifier),
                             temporal31._trust(altered,policy,classifier))

    def test_parent_fallback_is_exact_and_untouched(self):
        self.assertEqual(phase24.restore_memory(self.parent_file),self.parent)
        self.assertEqual(mesh.load(self.adapter_file),self.adapter)
        wrapper=mesh.load(self.output/"temporal.json")
        self.assertEqual(wrapper["sha256"],mesh.digest(wrapper["payload"]))
        self.assertFalse(wrapper["payload"]["auto_promote"])
        self.assertEqual(wrapper["payload"]["parent_sha256"],self.parent["sha256"])
        self.assertTrue(self.decision["no_test_label_access"])

    def test_invalid_histories_or_tampered_frozen_sources_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError,"invalid experimental"):
                temporal31.make_data(0,temporal31.TRAIN_CLASSES,17,tmp)
        wrong=copy.deepcopy(self.adapter)
        wrong["sha256"]="0"*64
        p=self.root/"tampered.json"
        mesh.save(p,wrong)
        with patch.object(temporal31,"PARENT_SHA",self.parent["sha256"]),patch.object(
            temporal31,"ADAPTER_SHA",self.adapter["sha256"]):
            with self.assertRaisesRegex(ValueError,"fingerprint mismatch"):
                temporal31._sources(self.parent_file,p)

    def test_nine_sealed_cohorts_and_regime_specific_reporting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            groups=[]
            for kind,seeds,regimes in (
                ("id",temporal31.ID_TEST,temporal31.ID),
                ("known",temporal31.KNOWN_TEST,temporal31.KNOWN),
                ("unseen",temporal31.UNKNOWN_TEST,temporal31.UNKNOWN)):
                items=[]
                for seed in seeds:
                    folder=root/f"{kind}-{seed}"
                    temporal31.make_data(seed,regimes,16,folder)
                    items.append((folder/"public.json",folder/"targets.json"))
                groups.append(items)
            with patch.object(temporal31,"PARENT_SHA",self.parent["sha256"]),patch.object(
                temporal31,"ADAPTER_SHA",self.adapter["sha256"]):
                report=temporal31.evaluate(
                    self.output/"parent.json",self.output/"adapter.json",
                    self.output/"temporal.json",
                    self.output/"training-decision.json",
                    *groups,root/"result")
                self.assertEqual(report["total_requested_scenes"],144)
                self.assertEqual(len(report["per_seed"]),9)
                self.assertEqual(set(report["summary"]["id"]["regimes"]),set(temporal31.ID))
                self.assertIn(report["status"],(
                    "SIX_FRAME_RETENTION_SUPPORTED","NO_VALIDATED_SIX_FRAME_GATE",
                    "BLOCKED_TEMPORAL_GENERALIZATION"))
                with self.assertRaisesRegex(ValueError,"duplicate sealed data"):
                    temporal31.evaluate(
                        self.output/"parent.json",self.output/"adapter.json",
                        self.output/"temporal.json",
                        self.output/"training-decision.json",
                        [groups[0][0],groups[0][0],groups[0][1]],
                        groups[1],groups[2],root/"repeat")

    def test_train_cli_uses_train_function_not_argument_parser(self):
        import sys
        args=["temporal31","train","--parent","p","--adapter","a",
              "--train-public","tp","--train-targets","tt",
              "--val-public","vp","--val-targets","vt","--out","o"]
        with patch.object(sys,"argv",args),patch.object(
                temporal31,"train",return_value={"validation_pass":False}) as fn:
            temporal31.main()
            fn.assert_called_once()


if __name__=="__main__":
    unittest.main()

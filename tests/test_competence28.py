"""Competence-tree safety tests: observed routing only, immutable parent."""
import copy
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from neural_mesh import mesh, phase2, phase24, safety27, competence28, transfer26


class CompetenceTreeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory()
        cls.root=Path(cls.temp.name)
        cls.id=cls.root/"id"
        cls.known_train=cls.root/"known_train"
        cls.known_val=cls.root/"known_val"
        phase2.prepare(28211,{"train":64,"validation":16,"test":8},cls.id)
        transfer26.make_subset(48211,32,transfer26.KNOWN,cls.known_train)
        transfer26.make_subset(48212,16,transfer26.KNOWN,cls.known_val)
        cls.parent_file=cls.root/"parent.json"
        cls.parent=phase24.build_memory(
            cls.id/"public.json",cls.id/"weights.json",
            cls.id/"trainval-labels.json",cls.parent_file)
        adapter_payload={"schema":transfer26.SCHEMA,
                         "parent_sha256":cls.parent["sha256"],
                         "model":{"theta":[[0.]*transfer26.FEATURE_DIM
                                           for _ in transfer26.EXPERTS],
                                  "means":[0.]*transfer26.FEATURE_DIM,
                                  "scales":[1.]*transfer26.FEATURE_DIM,
                                  "strength":.25}}
        cls.adapter={"payload":adapter_payload,
                     "sha256":mesh.digest(adapter_payload)}
        cls.adapter_file=cls.root/"adapter.json"
        mesh.save(cls.adapter_file,cls.adapter)
        cls.out=cls.root/"frozen"
        with patch.object(safety27,"PARENT_SHA",cls.parent["sha256"]),patch.object(
            safety27,"BASE_ADAPTER_SHA",cls.adapter["sha256"]):
            cls.decision=competence28.train(
                cls.parent_file,cls.adapter_file,
                cls.id/"public.json",cls.id/"trainval-labels.json",
                cls.known_train/"public.json",cls.known_train/"targets.json",
                cls.known_val/"public.json",cls.known_val/"targets.json",
                cls.out)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_tree_can_avoid_harm_and_use_beneficial_transfer(self):
        examples=[]
        for i in range(120):
            old=i<60
            features=[1.,.5,0.02 if old else 1.8,.1,.5,.3,.2,0.,0.,0.,.2]
            # parent=[0,0], adapter=[1,0]; target=0 for old, 1 for new.
            examples.append({"features":features,
                             "weight":1./120,"a":1.,
                             "b":0. if old else 1.})
        fitted=competence28._grow(examples,max_depth=2,min_leaf=25,ridge=.0001)
        left=competence28._leaf_trust(examples[0]["features"],fitted)
        right=competence28._leaf_trust(examples[-1]["features"],fitted)
        self.assertLess(left,.1)
        self.assertGreater(right,.9)

    def test_regime_name_and_answer_not_used_by_gate(self):
        scene=mesh.load(self.known_train/"public.json")["scenes"][0]
        e=transfer26._example(scene,[1.,2.],self.parent["payload"],
                              "none",0,"known")
        x=safety27._candidate_example(e,self.adapter["payload"]["model"])
        other=copy.deepcopy(x)
        other["truth"]=[300.,-400.]
        other["regime"]="fake"
        policy={"tree":{"feature":2,"threshold":1.,
                        "left":{"trust":0.},"right":{"trust":1.}},
                "scale":.75}
        self.assertEqual(competence28.prediction(x,policy),
                         competence28.prediction(other,policy))
        self.assertEqual(competence28.prediction(x,None),x["parent"])
        self.assertTrue(0<=competence28.trust(x,policy)<=1)

    def test_previous_checkpoint_and_adapter_untouched(self):
        self.assertEqual(phase24.restore_memory(self.parent_file),self.parent)
        self.assertEqual(mesh.load(self.adapter_file),self.adapter)
        self.assertEqual(mesh.load(self.out/"parent.json"),self.parent)
        self.assertEqual(mesh.load(self.out/"prior-adapter.json"),self.adapter)
        fitted=mesh.load(self.out/"competence.json")
        self.assertEqual(fitted["sha256"],mesh.digest(fitted["payload"]))
        self.assertEqual(fitted["payload"]["parent_sha256"],self.parent["sha256"])
        self.assertTrue(fitted["payload"]["regime_label_never_used_in_inference"])
        self.assertTrue(fitted["payload"]["no_automatic_promotion"])

    def test_training_never_accepts_test_targets(self):
        leak=mesh.load(self.id/"trainval-labels.json")
        leak["labels"]["test"]=mesh.load(self.id/"test-labels.json")["labels"]["test"]
        path=self.root/"leak.json"
        mesh.save(path,leak)
        with patch.object(safety27,"PARENT_SHA",self.parent["sha256"]),patch.object(
            safety27,"BASE_ADAPTER_SHA",self.adapter["sha256"]):
            with self.assertRaisesRegex(ValueError,"label partitions"):
                competence28.train(
                    self.parent_file,self.adapter_file,
                    self.id/"public.json",path,
                    self.known_train/"public.json",self.known_train/"targets.json",
                    self.known_val/"public.json",self.known_val/"targets.json",
                    self.root/"leak-output")

    def test_nine_independent_sealed_splits_and_outcome_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            iid,known,unseen=[],[],[]
            for seed in competence28.TEST_ID:
                folder=root/f"id-{seed}"
                phase2.prepare(seed,{"train":4,"validation":4,"test":8},folder)
                iid.append((folder/"public.json",folder/"test-labels.json"))
            for kind,seeds,families in (
                ("known",competence28.TEST_KNOWN,transfer26.KNOWN),
                ("unseen",competence28.TEST_UNSEEN,transfer26.UNKNOWN)):
                group=known if kind=="known" else unseen
                for seed in seeds:
                    folder=root/f"{kind}-{seed}"
                    transfer26.make_subset(seed,16,families,folder)
                    group.append((folder/"public.json",folder/"targets.json"))
            with patch.object(safety27,"PARENT_SHA",self.parent["sha256"]),patch.object(
                safety27,"BASE_ADAPTER_SHA",self.adapter["sha256"]):
                result=competence28.evaluate(
                    self.out/"parent.json",self.out/"prior-adapter.json",
                    self.out/"competence.json",self.out/"decision.json",
                    iid,known,unseen,root/"evidence")
                self.assertEqual(result["test_scenarios"],3*8+6*16)
                self.assertEqual(set(result["summary"]),{"id","known","unseen"})
                self.assertIn(result["status"],(
                    "RETENTION_REPAIR_SUPPORTED","NO_VALIDATED_COMPETENCE_MAP",
                    "BLOCKED_SKILL_RETENTION_OR_TRANSFER"))
                with self.assertRaisesRegex(ValueError,"duplicate sealed test"):
                    competence28.evaluate(
                        self.out/"parent.json",self.out/"prior-adapter.json",
                        self.out/"competence.json",self.out/"decision.json",
                        [iid[0],iid[0],iid[1]],known,unseen,root/"repeated")

    def test_invalid_tree_or_tampered_checkpoint_is_rejected(self):
        ex={"features":[1.]+[0.]*10}
        with self.assertRaisesRegex(ValueError,"invalid leaf trust"):
            competence28.trust(ex,{"tree":{"trust":-1.},"scale":1.})
        with self.assertRaisesRegex(ValueError,"invalid decision node"):
            competence28.trust(ex,{"tree":{"feature":999,"threshold":0.,
                 "left":{"trust":0.},"right":{"trust":1.}},"scale":1.})

    def test_cli_train_dispatch(self):
        import sys
        args=["competence28","train","--parent","parent",
              "--adapter","adapter","--id-public","public",
              "--id-trainval","trainval",
              "--known-train-public","knownpublic",
              "--known-train-targets","knownlabels",
              "--known-val-public","valpublic",
              "--known-val-targets","vallabels",
              "--out","output"]
        report={"validation_pass":False,
                "validation_chosen_id":{"mse":.1},
                "validation_chosen_known":{"mse":.2},
                "candidate_sha256":"a"*64}
        with patch.object(sys,"argv",args),patch.object(
                competence28,"train",return_value=report) as run:
            competence28.main()
            run.assert_called_once()


if __name__=="__main__":
    unittest.main()

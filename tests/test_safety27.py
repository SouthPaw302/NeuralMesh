"""Phase 2.6.1 safety-routing integrity and forgetting regression tests."""
import copy
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from neural_mesh import mesh, phase2, phase24, phase25, safety27, transfer26


class SafetyRouterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory()
        cls.root=Path(cls.tmp.name)
        cls.train=cls.root/"iid"
        cls.cal_known=cls.root/"known"
        cls.val_known=cls.root/"validation"
        phase2.prepare(27211,{"train":32,"validation":8,"test":8},cls.train)
        transfer26.make_subset(47211,16,transfer26.KNOWN,cls.cal_known)
        transfer26.make_subset(47212,16,transfer26.KNOWN,cls.val_known)
        cls.parent_path=cls.root/"parent.json"
        cls.parent=phase24.build_memory(
            cls.train/"public.json",cls.train/"weights.json",
            cls.train/"trainval-labels.json",cls.parent_path)
        # Controlled read-only fixture stands in for an earlier adapter.
        cls.adapter_payload={"schema":transfer26.SCHEMA,
                             "parent_sha256":cls.parent["sha256"],
                             "model":{"theta":[[0.]*transfer26.FEATURE_DIM
                                               for _ in transfer26.EXPERTS],
                                      "means":[0.]*transfer26.FEATURE_DIM,
                                      "scales":[1.]*transfer26.FEATURE_DIM,
                                      "strength":.25}}
        cls.adapter={"payload":cls.adapter_payload,
                     "sha256":mesh.digest(cls.adapter_payload)}
        cls.adapter_path=cls.root/"adapter.json"
        mesh.save(cls.adapter_path,cls.adapter)
        cls.output=cls.root/"fitted"
        with patch.object(safety27,"PARENT_SHA",cls.parent["sha256"]),patch.object(
                safety27,"BASE_ADAPTER_SHA",cls.adapter["sha256"]):
            cls.decision=safety27.train(
                cls.parent_path,cls.adapter_path,
                cls.train/"public.json",cls.train/"trainval-labels.json",
                cls.cal_known/"public.json",cls.cal_known/"targets.json",
                cls.val_known/"public.json",cls.val_known/"targets.json",
                cls.output)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_frozen_sources_are_not_changed(self):
        self.assertEqual(phase24.restore_memory(self.parent_path),self.parent)
        self.assertEqual(mesh.load(self.adapter_path),self.adapter)
        self.assertTrue(self.decision["no_heldout_labels_seen"])
        self.assertEqual(mesh.load(self.output/"parent.json"),self.parent)
        self.assertEqual(mesh.load(self.output/"prior-adapter.json"),self.adapter)
        wrapper=mesh.load(self.output/"safety.json")
        self.assertEqual(wrapper["sha256"],mesh.digest(wrapper["payload"]))
        self.assertFalse(wrapper["payload"]["automatic_promotion"])

    def test_inference_is_blind_to_regime_label_and_targets(self):
        public=mesh.load(self.cal_known/"public.json")
        scene=public["scenes"][0]
        altered=copy.deepcopy(scene)
        altered["regime"]="secret-no-regime-at-inference"
        memory=self.parent["payload"]
        a=transfer26._example(scene,[1.,2.],memory,"none",0,"ood")
        b=transfer26._example(altered,[-100.,200.],memory,"none",0,"ood")
        model=self.adapter["payload"]["model"]
        x=safety27._candidate_example(a,model)
        y=safety27._candidate_example(b,model)
        self.assertEqual(x["features"],y["features"])
        self.assertEqual(x["parent"],y["parent"])
        self.assertEqual(x["adapter"],y["adapter"])
        self.assertEqual(safety27.prediction(x,None),x["parent"])
        policy={"theta":[0.]*safety27.FEATURE_COUNT,
                "mean":[0.]*safety27.FEATURE_COUNT,
                "scale":[1.]*safety27.FEATURE_COUNT,"floor":.1}
        self.assertEqual(safety27.prediction(x,policy),
                         safety27.prediction(y,policy))

    def test_gate_is_bounded_and_parent_fallback_exact(self):
        scene=mesh.load(self.train/"public.json")["scenes"]["train"][0]
        ex=transfer26._example(scene,[0.,0.],self.parent["payload"],
                               "missing",0,"id")
        sample=safety27._candidate_example(ex,self.adapter["payload"]["model"])
        self.assertEqual(safety27._trust(sample,None),0.)
        for floor in (0.,.15):
            policy={"theta":[0.]*safety27.FEATURE_COUNT,
                    "mean":[0.]*safety27.FEATURE_COUNT,
                    "scale":[1.]*safety27.FEATURE_COUNT,"floor":floor}
            t=safety27._trust(sample,policy)
            self.assertTrue(floor<=t<=1.)
            self.assertTrue(all(math.isfinite(v)
                                for v in safety27.prediction(sample,policy)))

    def test_validation_guard_rejects_linear_and_turning_forgetting(self):
        parent={"mse":1.0,"faults":{k:1.0 for k in safety27.FAULTS},
                "regimes":{k:1.0 for k in safety27.ID_REGIMES}}
        candidate=copy.deepcopy(parent)
        known={"mse":1.0}
        transferred={"mse":.8}
        candidate["mse"]=.95
        self.assertTrue(safety27._validation_safe(
            parent,candidate,known,transferred))
        candidate["regimes"]["linear"]=1.005
        self.assertFalse(safety27._validation_safe(
            parent,candidate,known,transferred))
        candidate["regimes"]["linear"]=1.
        candidate["regimes"]["turning"]=1.001
        self.assertFalse(safety27._validation_safe(
            parent,candidate,known,transferred))

    def test_training_rejects_test_labels_and_disallowed_ood(self):
        label=mesh.load(self.train/"trainval-labels.json")
        label["labels"]["test"]=mesh.load(self.train/"test-labels.json")["labels"]["test"]
        leak=self.root/"leak.json"
        mesh.save(leak,label)
        with patch.object(safety27,"PARENT_SHA",self.parent["sha256"]),patch.object(
                safety27,"BASE_ADAPTER_SHA",self.adapter["sha256"]):
            with self.assertRaisesRegex(ValueError,"label partitions"):
                safety27.train(
                    self.parent_path,self.adapter_path,
                    self.train/"public.json",leak,
                    self.cal_known/"public.json",self.cal_known/"targets.json",
                    self.val_known/"public.json",self.val_known/"targets.json",
                    self.root/"unsafe")
            with tempfile.TemporaryDirectory() as tmp:
                transfer26.make_subset(47241,16,transfer26.UNKNOWN,tmp)
                with self.assertRaisesRegex(ValueError,"domain partition mismatch"):
                    safety27.train(
                        self.parent_path,self.adapter_path,
                        self.train/"public.json",self.train/"trainval-labels.json",
                        Path(tmp)/"public.json",Path(tmp)/"targets.json",
                        self.val_known/"public.json",self.val_known/"targets.json",
                        self.root/"unsafe")

    def test_train_cli_dispatch_does_not_shadow_trainer(self):
        import sys
        argv=["python -m neural_mesh.safety27","train",
              "--parent",str(self.parent_path),
              "--adapter",str(self.adapter_path),
              "--id-public",str(self.train/"public.json"),
              "--id-labels",str(self.train/"trainval-labels.json"),
              "--known-train-public",str(self.cal_known/"public.json"),
              "--known-train-labels",str(self.cal_known/"targets.json"),
              "--known-val-public",str(self.val_known/"public.json"),
              "--known-val-labels",str(self.val_known/"targets.json"),
              "--out",str(self.root/"cli")]
        result={"validation_pass":False,"selected_id_validation":{"mse":.2},
                "selected_known_validation":{"mse":.3},
                "safety_sha256":"a"*64}
        with patch.object(sys,"argv",argv),patch.object(
                safety27,"train",return_value=result) as trainer:
            safety27.main()
            trainer.assert_called_once()

    def test_tampered_adapter_identity_is_rejected(self):
        forged=copy.deepcopy(self.adapter)
        forged["payload"]["parent_sha256"]="0"*64
        mesh.save(self.root/"forged.json",forged)
        with patch.object(safety27,"PARENT_SHA",self.parent["sha256"]),patch.object(
                safety27,"BASE_ADAPTER_SHA",self.adapter["sha256"]):
            with self.assertRaisesRegex(ValueError,"fingerprint invalid"):
                safety27._load_sources(self.parent_path,self.root/"forged.json")

    def test_heldout_status_separate_from_test_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            ids,known,unknown=[],[],[]
            for seed in (27221,27222,27223):
                folder=root/f"id-{seed}"
                phase2.prepare(seed,{"train":4,"validation":4,"test":8},folder)
                ids.append((folder/"public.json",folder/"test-labels.json"))
            for tag,seeds,families in (
                ("known",(47221,47222,47223),transfer26.KNOWN),
                ("unknown",(47231,47232,47233),transfer26.UNKNOWN)):
                group=known if tag=="known" else unknown
                for seed in seeds:
                    folder=root/f"{tag}-{seed}"
                    transfer26.make_subset(seed,16,families,folder)
                    group.append((folder/"public.json",folder/"targets.json"))
            with patch.object(safety27,"PARENT_SHA",self.parent["sha256"]),patch.object(
                    safety27,"BASE_ADAPTER_SHA",self.adapter["sha256"]):
                report=safety27.evaluate(
                    self.output/"parent.json",self.output/"prior-adapter.json",
                    self.output/"safety.json",self.output/"decision.json",
                    ids,known,unknown,root/"scored")
                self.assertEqual(report["separate_test_cohorts"],9)
                self.assertEqual(report["test_scenarios"],3*8+6*16)
                self.assertEqual(set(report["summary"]),{"id","known","unseen"})
                self.assertIn(report["status"],(
                    "RETENTION_REPAIR_SUPPORTED","NO_VALIDATED_SAFETY_GATE",
                    "BLOCKED_OLD_SKILL_OR_TRANSFER_REGRESSION"))
                with self.assertRaisesRegex(ValueError,"duplicate held-out"):
                    safety27.evaluate(
                        self.output/"parent.json",self.output/"prior-adapter.json",
                        self.output/"safety.json",self.output/"decision.json",
                        [ids[0],ids[0],ids[1]],known,unknown,root/"repeat")


if __name__=="__main__":
    unittest.main()

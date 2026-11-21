"""Phase 2.6: regime-blind routing, provenance and cross-domain safety checks."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from neural_mesh import mesh, phase2, phase24, phase25, transfer26


class CrossDomainTransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory()
        cls.root=Path(cls.temp.name)
        cls.replay=cls.root/"replay"
        cls.fresh=cls.root/"fresh"
        cls.domain_train=cls.root/"novel-train"
        cls.domain_val=cls.root/"novel-val"
        phase2.prepare(64013,{"train":24,"validation":8,"test":8},cls.replay)
        phase2.prepare(26111,{"train":24,"validation":8,"test":8},cls.fresh)
        transfer26.make_subset(46111,16,transfer26.KNOWN,cls.domain_train)
        transfer26.make_subset(46112,16,transfer26.KNOWN,cls.domain_val)
        cls.parent_path=cls.root/"parent.json"
        cls.parent=phase24.build_memory(
            cls.replay/"public.json",cls.replay/"weights.json",
            cls.replay/"trainval-labels.json",cls.parent_path)
        cls.proposal=cls.root/"proposal"
        with patch.object(phase25,"PARENT_SHA",cls.parent["sha256"]):
            cls.decision=transfer26.train(
                cls.parent_path,
                cls.replay/"public.json",cls.replay/"trainval-labels.json",
                cls.fresh/"public.json",cls.fresh/"trainval-labels.json",
                cls.domain_train/"public.json",cls.domain_train/"targets.json",
                cls.domain_val/"public.json",cls.domain_val/"targets.json",
                cls.proposal)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_domain_holdout_is_explicit(self):
        train=mesh.load(self.domain_train/"public.json")
        val=mesh.load(self.domain_val/"public.json")
        self.assertEqual({x["regime"] for x in train["scenes"]},set(transfer26.KNOWN))
        self.assertEqual({x["regime"] for x in val["scenes"]},set(transfer26.KNOWN))
        self.assertFalse(set(transfer26.UNKNOWN)&
                         {x["regime"] for x in train["scenes"]})
        with tempfile.TemporaryDirectory() as tmp:
            transfer26.make_subset(46131,16,transfer26.UNKNOWN,tmp)
            public=mesh.load(Path(tmp)/"public.json")
            self.assertEqual({x["regime"] for x in public["scenes"]},set(transfer26.UNKNOWN))

    def test_physics_experts_do_not_read_target_or_regime_label(self):
        scene=mesh.load(self.domain_train/"public.json")["scenes"][0]
        one=copy.deepcopy(scene)
        two=copy.deepcopy(scene)
        two["regime"]="totally-unrelated-label"
        target=[123.0,-999.0]
        memory=self.parent["payload"]
        a=transfer26._example(one,target,memory,"none",0,"ood")
        b=transfer26._example(two,target,memory,"none",0,"ood")
        self.assertEqual(a["experts"],b["experts"])
        self.assertEqual(a["features"],b["features"])
        self.assertEqual(len(a["experts"]),5)
        self.assertEqual(len(a["features"]),7)
        self.assertEqual(transfer26.predict_example(a,None),a["experts"][0])

    def test_all_three_fault_conditions_have_finite_predictions(self):
        scene=mesh.load(self.domain_val/"public.json")["scenes"][0]
        memory=self.parent["payload"]
        for fault in transfer26.FAULTS:
            example=transfer26._example(scene,[0.,0.],memory,fault,0,"ood")
            for prediction in example["experts"]:
                self.assertEqual(len(prediction),2)
                self.assertTrue(all(abs(x)<1e6 for x in prediction))

    def test_parent_is_unmodified_and_adapter_is_digest_bound(self):
        current=phase24.restore_memory(self.parent_path)
        self.assertEqual(self.parent,current)
        frozen=mesh.load(self.proposal/"parent.json")
        adapter=mesh.load(self.proposal/"adapter.json")
        self.assertEqual(frozen,self.parent)
        self.assertEqual(adapter["sha256"],mesh.digest(adapter["payload"]))
        self.assertEqual(adapter["payload"]["parent_sha256"],self.parent["sha256"])
        self.assertTrue(adapter["payload"]["parent_unchanged"])
        self.assertFalse(adapter["payload"]["automatic_memory_promotion"])
        self.assertEqual(adapter["sha256"],self.decision["adapter_sha256"])

    def test_training_rejects_held_out_test_labels(self):
        labels=mesh.load(self.fresh/"trainval-labels.json")
        leaked=copy.deepcopy(labels)
        leaked["labels"]["test"]=mesh.load(self.fresh/"test-labels.json")["labels"]["test"]
        path=self.root/"leaked.json"
        mesh.save(path,leaked)
        with patch.object(phase25,"PARENT_SHA",self.parent["sha256"]):
            with self.assertRaisesRegex(ValueError,"label partitions"):
                transfer26.train(
                    self.parent_path,
                    self.replay/"public.json",self.replay/"trainval-labels.json",
                    self.fresh/"public.json",path,
                    self.domain_train/"public.json",self.domain_train/"targets.json",
                    self.domain_val/"public.json",self.domain_val/"targets.json",
                    self.root/"rejected")

    def test_wrong_training_domain_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            transfer26.make_subset(46141,16,transfer26.UNKNOWN,tmp)
            with patch.object(phase25,"PARENT_SHA",self.parent["sha256"]):
                with self.assertRaisesRegex(ValueError,"domain partition mismatch"):
                    transfer26.train(
                        self.parent_path,
                        self.replay/"public.json",self.replay/"trainval-labels.json",
                        self.fresh/"public.json",self.fresh/"trainval-labels.json",
                        Path(tmp)/"public.json",Path(tmp)/"targets.json",
                        self.domain_val/"public.json",self.domain_val/"targets.json",
                        self.root/"rejected")

    def test_three_by_three_disjoint_heldout_sets_and_blocking_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            idsets,known,unknown=[],[],[]
            for seed in (26121,26122,26123):
                folder=root/f"id-{seed}"
                phase2.prepare(seed,{"train":4,"validation":4,"test":8},folder)
                idsets.append((folder/"public.json",folder/"test-labels.json"))
            for label,seeds,regimes in (("known",(46121,46122,46123),transfer26.KNOWN),
                                       ("unknown",(46131,46132,46133),transfer26.UNKNOWN)):
                target=known if label=="known" else unknown
                for seed in seeds:
                    folder=root/f"{label}-{seed}"
                    transfer26.make_subset(seed,16,regimes,folder)
                    target.append((folder/"public.json",folder/"targets.json"))
            with patch.object(phase25,"PARENT_SHA",self.parent["sha256"]):
                report=transfer26.evaluate(self.proposal/"parent.json",
                    self.proposal/"adapter.json",self.proposal/"decision.json",
                    idsets,known,unknown,root/"results")
                self.assertEqual(report["test_scenarios"],3*8+6*16)
                self.assertEqual(len(report["per_dataset"]),9)
                self.assertIn(report["status"],(
                    "TRANSFER_SUPPORTED","BLOCKED_CROSS_DOMAIN_TRANSFER",
                    "NO_VALIDATED_ADAPTER"))
                with self.assertRaisesRegex(ValueError,"repeated held-out cohort"):
                    transfer26.evaluate(self.proposal/"parent.json",
                        self.proposal/"adapter.json",self.proposal/"decision.json",
                        [idsets[0],idsets[0],idsets[1]],known,unknown,root/"bad")
                bad=copy.deepcopy(mesh.load(self.proposal/"adapter.json"))
                bad["payload"]["parent_sha256"]="0"*64
                mesh.save(root/"tampered.json",bad)
                with self.assertRaisesRegex(ValueError,"adapter/parent/evaluation"):
                    transfer26.evaluate(self.proposal/"parent.json",
                        root/"tampered.json",self.proposal/"decision.json",
                        idsets,known,unknown,root/"bad")

    def test_parent_fallback_and_normalized_expert_gating(self):
        scene=mesh.load(self.domain_train/"public.json")["scenes"][0]
        example=transfer26._example(scene,[0.,0.],self.parent["payload"],
                                    "missing",0,"ood")
        parent=transfer26.predict_example(example,None)
        neutral={"theta":[[0.]*transfer26.FEATURE_DIM for _ in transfer26.EXPERTS],
                 "means":[0.]*transfer26.FEATURE_DIM,
                 "scales":[1.]*transfer26.FEATURE_DIM,
                 "strength":0.}
        self.assertEqual(parent,transfer26.predict_example(example,neutral))
        att=transfer26._weights(neutral["theta"],example["features"],
                                neutral["means"],neutral["scales"])
        self.assertAlmostEqual(sum(att),1.)
        self.assertTrue(all(x>=0 for x in att))


if __name__=="__main__":
    unittest.main()

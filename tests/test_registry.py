"""Registry staging never bypasses manual approval or an immutable history."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from neural_mesh import mesh, phase25, registry


class MemoryRegistryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.folder=Path(self.temp.name)
        self.parent={"sha256":phase25.PARENT_SHA,"payload":{"version":1}}
        self.child={"sha256":"a"*64,"payload":{"generation":2}}
        self.root=self.folder/"memory/registry"
        for name,data in (("parent.json",self.parent),("child.json",self.child)):
            mesh.save(self.folder/name,data)

    def tearDown(self):
        self.temp.cleanup()

    def _stage(self,run=99227):
        def restore(path):
            name=Path(path).name
            if name=="parent.json":
                return self.parent
            if name=="child.json":
                return self.child
            raise ValueError("unexpected test path")
        with patch.object(registry.phase25,"check_promotion",
                          return_value={"evaluation":"b"*64}),patch.object(
                          registry.phase24,"restore_memory",side_effect=restore):
            return registry.stage(
                self.folder/"parent.json",self.folder/"child.json",
                self.folder/"decision.json",self.folder/"report.json",
                self.folder/"gate.json",self.root,run)

    def test_review_staging_is_a_git_file_operation_not_merge(self):
        result=self._stage()
        self.assertEqual(result["operation"],"PROPOSE_MODEL_MEMORY_PROMOTION")
        self.assertTrue(result["human_merge_required"])
        index=mesh.load(self.root/"index.json")
        self.assertEqual(index["active_sha256"],"a"*64)
        self.assertTrue(index["review_required"])
        self.assertEqual([v["generation"] for v in index["versions"]],[1,2])
        self.assertEqual(mesh.load(self.root.parent/"versions"/(phase25.PARENT_SHA+".json")),
                         self.parent)
        self.assertEqual(mesh.load(self.root.parent/"versions"/("a"*64+".json")),
                         self.child)

    def test_duplicate_candidate_not_accepted(self):
        self._stage()
        with self.assertRaisesRegex(ValueError,"stale memory-registry head"):
            self._stage()

    def test_requires_explicit_run_and_pinned_parent(self):
        with self.assertRaisesRegex(ValueError,"source run ID"):
            self._stage(0)
        with patch.object(registry.phase25,"PARENT_SHA","f"*64):
            with self.assertRaisesRegex(ValueError,"parent is not genesis"):
                self._stage()

    def test_existing_immutable_version_must_match(self):
        version=self.root.parent/"versions"/(phase25.PARENT_SHA+".json")
        mesh.save(version,{"sha256":phase25.PARENT_SHA,"payload":{"forged":True}})
        with self.assertRaisesRegex(ValueError,"immutable memory version differs"):
            self._stage()


if __name__=="__main__":
    unittest.main()

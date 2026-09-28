"""A resumed release build must contain exactly the pinned source modification."""
import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("build_patched_llama", ROOT / "scripts/build-patched-llama.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class PinnedPatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.source = Path(self.temp.name) / "source"
        self.source.mkdir()
        self.git("init", "-q")
        self.git("config", "core.autocrlf", "false")
        self.git("config", "user.name", "Runtime test")
        self.git("config", "user.email", "runtime-test@example.invalid")
        self.file = self.source / "parser.cpp"
        self.file.write_text("fenced-only\nunchanged\n", newline="\n")
        self.git("add", "parser.cpp")
        self.git("commit", "-qm", "base")
        commit = self.git("rev-parse", "HEAD").strip()
        self.file.write_text("plain-or-fenced\nunchanged\n", newline="\n")
        patch = Path(self.temp.name) / "parser.diff"
        patch.write_text(self.git("diff"), newline="\n")
        self.file.write_text("fenced-only\nunchanged\n", newline="\n")
        self.manifest = {"patch": str(patch), "patchSha256": builder.sha(patch), "upstreamCommit": commit}

    def git(self, *args):
        return subprocess.check_output(["git", *args], cwd=self.source, text=True)

    def test_applies_and_resumes_exact_patch(self):
        builder.prepare(self.source, self.manifest)
        builder.prepare(self.source, self.manifest)
        self.assertEqual(self.file.read_text(), "plain-or-fenced\nunchanged\n")

    def test_rejects_unrelated_edit_in_the_same_file(self):
        builder.prepare(self.source, self.manifest)
        self.file.write_text(self.file.read_text() + "unrecorded-change\n", newline="\n")
        with self.assertRaisesRegex(ValueError, "exact pinned patch"):
            builder.prepare(self.source, self.manifest)
        self.assertIn("unrecorded-change", self.file.read_text())

    def test_rejects_changed_patch_before_modifying_source(self):
        Path(self.manifest["patch"]).write_text("different patch")
        with self.assertRaisesRegex(ValueError, "pinned manifest"):
            builder.prepare(self.source, self.manifest)
        self.assertEqual(self.file.read_text(), "fenced-only\nunchanged\n")

    def test_rejects_wrong_upstream_commit(self):
        with self.assertRaisesRegex(ValueError, "pinned upstream commit"):
            builder.prepare(self.source, {**self.manifest, "upstreamCommit": "0" * 40})

    def test_release_patch_is_complete_and_matches_manifest(self):
        manifest = json.loads((ROOT / "scripts/llama-patch.json").read_text())
        patch = ROOT / manifest["patch"]
        self.assertEqual(builder.sha(patch), manifest["patchSha256"])
        output = subprocess.check_output(["git", "apply", "--numstat", str(patch)], cwd=ROOT, text=True)
        self.assertIn("common/parsers/gemma4.cpp", output)
        self.assertIn("tests/test-chat.cpp", output)

    def test_product_and_ministral_builds_preserve_both_parser_fixes(self):
        manifest = json.loads((ROOT / "scripts/llama-patch-ministral.json").read_text())
        patch = ROOT / manifest["patch"]
        self.assertEqual(builder.sha(patch), manifest["patchSha256"])
        output = subprocess.check_output(["git", "apply", "--numstat", str(patch)], cwd=ROOT, text=True)
        for path in ("common/parsers/gemma4.cpp", "common/parsers/ministral3.cpp", "tests/test-chat.cpp"):
            self.assertIn(path, output)
        args = builder.configure_args(Path('/source'), Path('/build'), 'macos-arm64', manifest)
        self.assertIn('-DLLAMA_BUILD_COMMIT=6f767fe960-jurilix1', args)
        original = json.loads((ROOT / "scripts/llama-patch.json").read_text())
        self.assertEqual(original, manifest)
        self.assertEqual(builder.patched_commit(original), '6f767fe960-jurilix1')

    def test_rejects_ambiguous_patch_identity(self):
        with self.assertRaisesRegex(ValueError, 'Patch ID'):
            builder.patched_commit({'upstreamTag':'b10956','upstreamCommit':'a'*40,'id':'b10956'})


if __name__ == "__main__":
    unittest.main()

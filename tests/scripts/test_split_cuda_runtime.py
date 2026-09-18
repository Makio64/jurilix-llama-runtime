import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("split_cuda", ROOT / "scripts/split-cuda-runtime.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class SplitRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.files = {name: name.encode() for name in ["llama-server.exe", "ggml-cuda.dll", "cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll"]}
        self.receipt = {"variant": "win-cuda-x64", "id": "b10956-jurilix1", "tests": [{"name": "parser", "exitCode": 0}],
                        "files": {name: {"bytes": len(data), "sha256": module.digest(data)} for name, data in self.files.items()}}
        self.files["RUNTIME.json"] = json.dumps(self.receipt).encode()
        self.archive = self.root / "original.zip"
        with zipfile.ZipFile(self.archive, "w") as bundle:
            for name, data in self.files.items():
                bundle.writestr(name, data)
        data = self.archive.read_bytes()
        self.receipt["archive"] = {"filename": self.archive.name, "bytes": len(data), "sha256": module.digest(data)}
        self.path = self.root / "receipt.json"
        self.path.write_text(json.dumps(self.receipt))

    def test_split_downloads_reconstruct_every_original_file_without_changes(self):
        out = self.root / "split"
        result = module.split(self.path, out)
        installed = {}
        for archive in [result["archive"], *result["extras"]]:
            with zipfile.ZipFile(out / archive["filename"]) as bundle:
                installed.update({name: bundle.read(name) for name in bundle.namelist()})
        self.assertEqual(installed, self.files)
        self.assertEqual(result["repackagedFrom"], self.receipt["archive"])
        self.assertEqual(result["nativeReceiptSha256"], module.digest(self.path.read_bytes()))

    def test_changed_archive_is_rejected_before_creating_outputs(self):
        self.archive.write_bytes(self.archive.read_bytes() + b"tampered")
        with self.assertRaisesRegex(ValueError, "differs from the native build receipt"):
            module.split(self.path, self.root / "split")
        self.assertFalse((self.root / "split").exists())

    def test_changed_native_test_receipt_is_rejected(self):
        self.receipt["tests"][0]["exitCode"] = 1
        self.path.write_text(json.dumps(self.receipt))
        with self.assertRaisesRegex(ValueError, "passing native"):
            module.split(self.path, self.root / "split")


if __name__ == "__main__":
    unittest.main()

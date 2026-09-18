import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("compose_cuda", ROOT / "scripts/compose-cuda-runtime.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CudaCompositionTests(unittest.TestCase):
    def fixture(self, root, *, commit=None, filename="ggml.dll"):
        pin = json.loads((ROOT / "scripts/llama-patch.json").read_text())
        data = b"synthetic native runtime"
        receipt = {"variant": "win-cpu-x64", "id": pin["id"],
                   "upstreamCommit": commit or pin["upstreamCommit"], "patchSha256": pin["patchSha256"],
                   "tests": [{"exitCode": 0} for _ in range(5)],
                   "files": {filename: {"sha256": MODULE.sha(data), "bytes": len(data)}}}
        archive = root / "cpu.zip"
        with zipfile.ZipFile(archive, "w") as bundle:
            bundle.writestr(filename, data)
            bundle.writestr("RUNTIME.json", json.dumps(receipt))
        receipt["archive"] = {"filename": archive.name, "sha256": MODULE.sha(archive.read_bytes()), "bytes": archive.stat().st_size}
        path = root / "receipt.json"
        path.write_text(json.dumps(receipt))
        return path, archive

    def test_accepts_exact_source_receipt_and_file_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt, _ = self.fixture(Path(directory))
            _, files = MODULE.verified_files(receipt)
            self.assertEqual(files, {"ggml.dll": b"synthetic native runtime"})

    def test_rejects_changed_archive_and_wrong_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt, archive = self.fixture(root)
            archive.write_bytes(archive.read_bytes() + b"changed")
            with self.assertRaisesRegex(ValueError, "archive differs"):
                MODULE.verified_files(receipt)
            receipt, _ = self.fixture(root, commit="0" * 40)
            with self.assertRaisesRegex(ValueError, "exact pinned source"):
                MODULE.verified_files(receipt)

    def test_rejects_unsafe_paths_even_with_consistent_receipts(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt, _ = self.fixture(Path(directory), filename="../ggml.dll")
            with self.assertRaisesRegex(ValueError, "flat runtime"):
                MODULE.verified_files(receipt)

    def test_only_an_absent_driver_can_skip_cuda_registration(self):
        with tempfile.TemporaryDirectory() as directory:
            driver = Path(directory) / "nvcuda.dll"
            error = FileNotFoundError("ctypes reports missing module without winerror")
            self.assertTrue(MODULE.missing_nvidia_driver("nvcuda.dll", error, driver))
            self.assertFalse(MODULE.missing_nvidia_driver("ggml-base.dll", error, driver))
            self.assertFalse(MODULE.missing_nvidia_driver("nvcuda.dll", PermissionError("denied"), driver))
            driver.write_bytes(b"present but dependency missing")
            self.assertFalse(MODULE.missing_nvidia_driver("nvcuda.dll", error, driver))


if __name__ == "__main__":
    unittest.main()

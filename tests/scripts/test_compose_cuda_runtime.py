import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("compose_cuda", ROOT / "scripts/compose-cuda-runtime.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
SPLIT_SPEC = importlib.util.spec_from_file_location("split_cuda", ROOT / "scripts/split-cuda-runtime.py")
SPLIT = importlib.util.module_from_spec(SPLIT_SPEC)
SPLIT_SPEC.loader.exec_module(SPLIT)


class CudaCompositionTests(unittest.TestCase):
    def test_version_identity_ignores_startup_timing_but_preserves_build_and_compiler(self):
        identity = "version: 0.4.1-dev (build 11056, commit e613ef2c81-jurilix2)\nbuilt with MSVC for Windows x64\n"
        self.assertEqual(MODULE.version_identity("0.00.000.142 I srv initializing\n" + identity),
                         MODULE.version_identity("0.00.000.311 I srv initializing\n" + identity))
        self.assertNotEqual(MODULE.version_identity(identity),
                            MODULE.version_identity(identity.replace("jurilix2", "jurilix1")))
        self.assertNotEqual(MODULE.version_identity(identity),
                            MODULE.version_identity(identity.replace("x64", "arm64")))
        for invalid in ("", "initializing", identity + identity):
            with self.assertRaisesRegex(ValueError, "version identity"):
                MODULE.version_identity(invalid)

    def fixture(self, root, *, commit=None, filename="ggml.dll"):
        pin = json.loads((ROOT / "scripts/llama-patch.json").read_text())
        data = b"synthetic native runtime"
        tests = []
        for name in [*pin["tests"], "test-chat-peg-parser-shared"]:
            suffix = "shared-peg-parser" if name == "test-chat-peg-parser-shared" else name
            log = root / f"win-cpu-x64-{suffix}.log"
            log.write_bytes((name + " passed\n").encode())
            tests.append({"name": name, "exitCode": 0, "logSha256": MODULE.file_sha(log)})
        receipt = {"variant": "win-cpu-x64", "id": pin["id"],
                   "upstreamCommit": commit or pin["upstreamCommit"], "patchSha256": pin["patchSha256"],
                   "tests": tests,
                   "versionOutput": "version: synthetic pinned build\nbuilt with fixture for Windows x64\n",
                   "files": {filename: {"sha256": MODULE.sha(data), "bytes": len(data)}}}
        archive = root / "cpu.zip"
        with zipfile.ZipFile(archive, "w") as bundle:
            bundle.writestr(filename, data)
            bundle.writestr("RUNTIME.json", json.dumps(receipt))
        receipt["archive"] = {"filename": archive.name, "sha256": MODULE.sha(archive.read_bytes()), "bytes": archive.stat().st_size}
        path = root / "receipt.json"
        path.write_text(json.dumps(receipt))
        return path, archive

    def compose_fixture(self, root):
        receipt, _ = self.fixture(root)
        upstream = {}
        for name in MODULE.UPSTREAM:
            archive = root / name
            members = ["ggml-cuda.dll"] if name.startswith("llama-") else [
                "cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll"]
            with zipfile.ZipFile(archive, "w") as bundle:
                for member in members:
                    bundle.writestr(member, (member + " native bytes\n").encode() * 10)
            upstream[name] = archive

        def native_command(command, *, cwd, stdout, stderr, timeout):
            output = (json.dumps({"driverAvailable": False, "gpuInferenceQualified": False})
                      if "--probe" in command else "No NVIDIA driver on the fixture runner")
            stdout.write((output + "\n").encode())
            return MODULE.subprocess.CompletedProcess(command, 0)

        version = json.loads(receipt.read_text())["versionOutput"]
        with patch.object(MODULE.platform, "system", return_value="Windows"), \
             patch.object(MODULE.platform, "machine", return_value="AMD64"), \
             patch.object(MODULE, "download", side_effect=lambda name, _: upstream[name]) as downloads, \
             patch.object(MODULE.subprocess, "run", side_effect=native_command) as probes, \
             patch.object(MODULE.subprocess, "check_output", return_value=version):
            output = root / "composed"
            result = MODULE.compose(receipt, output, root / "cache")
        self.assertEqual(downloads.call_count, 2)
        self.assertEqual(probes.call_count, 2)
        self.assertEqual(probes.call_args_list[0].args[0][-2], "--probe")
        self.assertEqual(probes.call_args_list[1].args[0][-1], "--list-devices")
        return output / (Path(result["archive"]["filename"]).stem + ".json"), result

    def test_stored_intermediate_round_trips_through_verified_bounded_downloads(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt, composed = self.compose_fixture(root)
            intermediate = receipt.parent / composed["archive"]["filename"]
            self.assertEqual(composed["archive"]["sha256"], MODULE.file_sha(intermediate))
            self.assertEqual(composed["archive"]["bytes"], intermediate.stat().st_size)
            self.assertEqual(len(composed["tests"]), 7)
            self.assertTrue(all(test["exitCode"] == 0 for test in composed["tests"]))
            for test in composed["tests"]:
                suffix = "shared-peg-parser" if test["name"] == "test-chat-peg-parser-shared" else test["name"]
                self.assertEqual(test["logSha256"], MODULE.file_sha(receipt.parent / f"win-cuda-x64-{suffix}.log"))
            with zipfile.ZipFile(intermediate) as bundle:
                self.assertTrue(all(member.compress_type == zipfile.ZIP_STORED for member in bundle.infolist()))
                originals = {name: bundle.read(name) for name in bundle.namelist()}
            output = root / "downloads"
            result = SPLIT.split(receipt, output)
            restored = {}
            for archive in [result["archive"], *result["extras"]]:
                path = output / archive["filename"]
                self.assertLessEqual(path.stat().st_size, SPLIT.LIMIT)
                self.assertEqual(archive["sha256"], MODULE.file_sha(path))
                with zipfile.ZipFile(path) as bundle:
                    self.assertTrue(all(member.compress_type == zipfile.ZIP_DEFLATED for member in bundle.infolist()))
                    restored.update({name: bundle.read(name) for name in bundle.namelist()})
            self.assertEqual(restored, originals)
            self.assertEqual(result["repackagedFrom"], composed["archive"])
            self.assertEqual(result["nativeReceiptSha256"], MODULE.file_sha(receipt))
            self.assertEqual((output / "native-build-receipt.json").read_bytes(), receipt.read_bytes())

    def test_split_still_enforces_size_limit_for_a_stored_intermediate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt, _ = self.compose_fixture(root)
            with patch.object(SPLIT, "LIMIT", 1), self.assertRaisesRegex(ValueError, "app download limit"):
                SPLIT.split(receipt, root / "downloads")

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

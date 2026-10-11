#!/usr/bin/env python3
"""Verify native Windows CI runtime artifacts, then stage them for a release.

Each receipt must match the pinned source, patch and builder; every parser log,
archive digest and packaged file is checked, and CUDA must match its
composition and split receipts. Only verified archives, receipts and log
bundles are copied to --out, with ci-validation.json describing them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PIN = json.loads((ROOT / "scripts/llama-patch.json").read_text())


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def script_sha(name: str) -> str:
    return digest((ROOT / "scripts" / name).read_bytes())


def verify(ci: Path, variant: str) -> dict:
    root = ci / ("gemma-runtime-" + variant)
    receipts = list(root.rglob(f"llama-{PIN['id']}-bin-{variant}.json"))
    assert len(receipts) == 1, receipts
    path = receipts[0]
    receipt = json.loads(path.read_text())
    assert receipt["variant"] == variant
    for key in ("id", "upstreamRepository", "upstreamCommit", "upstreamTag", "patchSha256"):
        assert receipt[key] == PIN[key], key
    assert receipt["builderSha256"] == script_sha("build-patched-llama.py")
    assert receipt["sourceDiffSha256"] == PIN["patchSha256"]
    build = PIN["upstreamTag"][1:]
    commit = PIN["upstreamCommit"][:10] + PIN["id"][len(PIN["upstreamTag"]):]
    assert f"build {build}" in receipt["versionOutput"] and f"commit {commit}" in receipt["versionOutput"]
    expected_tests = set(PIN["tests"])
    if variant.startswith("win-"):
        expected_tests.add("test-chat-peg-parser-shared")
    if variant == "win-cuda-x64":
        expected_tests.update(("test-cuda-dependency-imports", "test-packaged-device-discovery"))
    assert {test["name"] for test in receipt["tests"]} == expected_tests
    for test in receipt["tests"]:
        assert test["exitCode"] == 0
        suffix = "shared-peg-parser" if test["name"] == "test-chat-peg-parser-shared" else test["name"]
        log = path.parent / f"{variant}-{suffix}.log"
        assert digest(log.read_bytes()) == test["logSha256"], log
    metadata = {path.name: path}
    archives = [receipt["archive"]]
    archive_root = path.parent
    if variant == "win-cuda-x64":
        split_path, = root.rglob("split-runtime-receipt.json")
        native_path = split_path.parent / "native-build-receipt.json"
        split = json.loads(split_path.read_text())
        assert native_path.read_bytes() == path.read_bytes()
        assert split["nativeReceiptSha256"] == digest(path.read_bytes())
        assert split["repackagedFrom"] == receipt["archive"]
        assert split["packagerSha256"] == script_sha("split-cuda-runtime.py")
        assert split["composition"] == receipt["composition"]
        assert receipt["composition"]["scriptSha256"] == script_sha("compose-cuda-runtime.py")
        assert all(source["tag"] == PIN["upstreamTag"] for source in receipt["composition"]["cudaSources"])
        archives = [split["archive"], *split["extras"]]
        archive_root = split_path.parent
        metadata[f"llama-{PIN['id']}-split-{variant}.json"] = split_path
    names = set()
    for archive in archives:
        file = archive_root / archive["filename"]
        data = file.read_bytes()
        assert len(data) == archive["bytes"] and digest(data) == archive["sha256"], file
        with zipfile.ZipFile(file) as bundle:
            for name in bundle.namelist():
                value = bundle.read(name)
                assert Path(name).name == name and "/" not in name and "\\" not in name
                assert name not in names, name
                names.add(name)
                if name == "RUNTIME.json":
                    assert json.loads(value) == {k: v for k, v in receipt.items() if k != "archive"}
                else:
                    assert {"sha256": digest(value), "bytes": len(value)} == receipt["files"][name], name
    assert names == set(receipt["files"]) | {"RUNTIME.json"}
    result = {"variant": variant, "receiptSha256": digest(path.read_bytes()),
              "filesVerified": len(receipt["files"]), "testsVerified": len(receipt["tests"]),
              "archives": archives}
    if "composition" in receipt:
        result["hardwareChecks"] = receipt["composition"]["hardwareChecks"]
        result["gpuInferenceQualified"] = receipt["composition"]["gpuInferenceQualified"]
    return result, [archive_root / a["filename"] for a in archives], metadata, sorted(path.parent.glob(f"{variant}-*.log"))


def stage(ci: Path, out: Path, variants: list[str]) -> list[dict]:
    out.mkdir(parents=True)
    results = []
    for variant in variants:
        result, archives, metadata, logs = verify(ci, variant)
        for file in archives:
            shutil.copy2(file, out / file.name)
        for name, file in metadata.items():
            shutil.copy2(file, out / name)
        with tarfile.open(out / f"native-test-evidence-{variant}.tar.gz", "w:gz") as bundle:
            for log in logs:
                bundle.add(log, arcname=log.name, recursive=False)
        results.append(result)
    (out / "ci-validation.json").write_text(json.dumps(results, indent=2) + "\n")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ci", type=Path, required=True, help="directory holding gemma-runtime-<variant> artifacts")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tag", required=True, help="release tag; must equal the pinned runtime ID")
    parser.add_argument("variants", nargs="+")
    options = parser.parse_args()
    if options.tag != PIN["id"]:
        raise SystemExit(f"Release {options.tag} does not match pinned runtime {PIN['id']}")
    print(json.dumps(stage(options.ci, options.out, options.variants), indent=2))

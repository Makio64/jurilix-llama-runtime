#!/usr/bin/env python3
"""Split a verified native CUDA build into the app's two bounded downloads.

No binary is modified. The native build receipt remains the authority for
source, tests and file hashes; a separate receipt records the repackaging.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

LIMIT = 512 * 1024 * 1024


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def split(receipt_path: Path, output: Path) -> dict:
    original = json.loads(receipt_path.read_text())
    if original["variant"] != "win-cuda-x64" or not original.get("tests") or any(t["exitCode"] != 0 for t in original["tests"]):
        raise ValueError("A passing native Windows CUDA build is required")
    source = receipt_path.parent / original["archive"]["filename"]
    data = source.read_bytes()
    if len(data) != original["archive"]["bytes"] or digest(data) != original["archive"]["sha256"]:
        raise ValueError("Source archive differs from the native build receipt")
    with zipfile.ZipFile(source) as bundle:
        if len(bundle.namelist()) != len(set(bundle.namelist())):
            raise ValueError("Duplicate archive entries")
        files = {name: bundle.read(name) for name in bundle.namelist()}
    if set(files) != set(original["files"]) | {"RUNTIME.json"}:
        raise ValueError("Archive file set differs from the native receipt")
    runtime = json.loads(files["RUNTIME.json"])
    if runtime != {k: v for k, v in original.items() if k != "archive"}:
        raise ValueError("Embedded and external build receipts differ")
    for name, expected in original["files"].items():
        if Path(name).name != name or "/" in name or "\\" in name:
            raise ValueError("Only a flat native runtime is supported")
        if digest(files[name]) != expected["sha256"] or len(files[name]) != expected["bytes"]:
            raise ValueError("Runtime file hash mismatch: " + name)
    extras = {name for name in files if name.lower().startswith(("cudart64", "cublas64", "cublaslt64")) and name.endswith(".dll")}
    if len(extras) != 3:
        raise ValueError("Expected exactly CUDA runtime, cuBLAS and cuBLASLt DLLs")
    names = [f"llama-{original['id']}-bin-win-cuda-12.4-x64.zip", "cudart-llama-bin-win-cuda-12.4-x64.zip"]
    if output.exists():
        raise FileExistsError("Output exists; reconcile receipts before retrying")
    output.mkdir(parents=True)
    archives = []
    for name, entries in zip(names, [set(files) - extras, extras], strict=True):
        archive = output / name
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            for entry in sorted(entries):
                bundle.writestr(entry, files[entry])
        data = archive.read_bytes()
        if len(data) > LIMIT:
            raise ValueError("Archive exceeds the app download limit: " + name)
        archives.append({"filename": name, "bytes": len(data), "sha256": digest(data)})
    receipt = {**original, "archive": archives[0], "extras": archives[1:],
               "repackagedFrom": original["archive"],
               "nativeReceiptSha256": digest(receipt_path.read_bytes()),
               "packagerSha256": digest(Path(__file__).read_bytes())}
    (output / "native-build-receipt.json").write_bytes(receipt_path.read_bytes())
    (output / "split-runtime-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = split(args.receipt, args.output)
    print(json.dumps({"archive": result["archive"], "extras": result["extras"]}, indent=2))

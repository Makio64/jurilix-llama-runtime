#!/usr/bin/env python3
"""Combine the tested patched Windows runtime with its unchanged upstream CUDA plugin.

The patch changes only Gemma/Ministral chat parsers and parser tests. GGML's
dynamic backend API checks its ABI version when loading the CUDA plugin. This
recipe retains the exact upstream b11056 CUDA 12.4 DLL and redistributables,
then requires native Windows import resolution and server device discovery.
Full plugin registration also runs when an NVIDIA driver is available.
It does not claim GPU inference on a runner without a GPU.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = {
    "llama-b11056-bin-win-cuda-12.4-x64.zip": (254212232, "cb6e838cad17e9920b99ab8496ca9aa7cdc3d3c218128957179bb1fbe0772c4c"),
    "cudart-llama-bin-win-cuda-12.4-x64.zip": (391443627, "8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6"),
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def version_identity(output: str) -> tuple[str, str]:
    versions = [line for line in output.splitlines() if line.startswith("version:")]
    compilers = [line for line in output.splitlines() if line.startswith("built with ")]
    if len(versions) != 1 or len(compilers) != 1:
        raise ValueError("Missing or ambiguous runtime version identity")
    return versions[0], compilers[0]


def verified_files(receipt_path: Path) -> tuple[dict, dict[str, bytes]]:
    receipt = json.loads(receipt_path.read_text())
    pin = json.loads((ROOT / "scripts/llama-patch.json").read_text())
    if (receipt["variant"] != "win-cpu-x64" or receipt["id"] != pin["id"] or
        receipt["upstreamCommit"] != pin["upstreamCommit"] or receipt["patchSha256"] != pin["patchSha256"] or
        len(receipt["tests"]) != 5 or any(t["exitCode"] != 0 for t in receipt["tests"])):
        raise ValueError("Expected the passing native CPU runtime from the exact pinned source/patch")
    archive = receipt_path.parent / receipt["archive"]["filename"]
    data = archive.read_bytes()
    if len(data) != receipt["archive"]["bytes"] or sha(data) != receipt["archive"]["sha256"]:
        raise ValueError("CPU archive differs from its native receipt")
    with zipfile.ZipFile(archive) as bundle:
        if len(bundle.namelist()) != len(set(bundle.namelist())):
            raise ValueError("Duplicate CPU archive entries")
        files = {name: bundle.read(name) for name in bundle.namelist()}
    if set(files) != set(receipt["files"]) | {"RUNTIME.json"}:
        raise ValueError("CPU runtime file set differs from its receipt")
    if json.loads(files.pop("RUNTIME.json")) != {k: v for k, v in receipt.items() if k != "archive"}:
        raise ValueError("Embedded CPU receipt differs")
    for name, data in files.items():
        if Path(name).name != name or "/" in name or "\\" in name:
            raise ValueError("Only flat runtime files are accepted")
        if receipt["files"][name] != {"sha256": sha(data), "bytes": len(data)}:
            raise ValueError("CPU runtime file differs: " + name)
    return receipt, files


def download(name: str, cache: Path) -> Path:
    size, digest = UPSTREAM[name]
    path = cache / name
    cache.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        request = urllib.request.Request(
            f"https://github.com/ggml-org/llama.cpp/releases/download/b11056/{name}",
            headers={"User-Agent": "jurilix-runtime-build/1"})
        with urllib.request.urlopen(request, timeout=120) as response, path.open("xb") as stream:
            copied = 0
            while data := response.read(1024 * 1024):
                copied += len(data)
                if copied > size:
                    raise ValueError("Upstream archive exceeds its pin")
                stream.write(data)
    data = path.read_bytes()
    if len(data) != size or sha(data) != digest:
        raise ValueError("Upstream archive differs from its pin: " + name)
    return path


def missing_nvidia_driver(name: str, error: OSError, driver_path: Path) -> bool:
    # ctypes 3.12 replaces error 126 with FileNotFoundError lacking winerror.
    # A present driver with a broken dependency is not an absent-driver skip.
    return (name.casefold() == "nvcuda.dll" and not driver_path.is_file() and
            (isinstance(error, FileNotFoundError) or getattr(error, "winerror", None) == 126))


def native_backend_probe(stage: Path) -> None:
    import pefile

    # nvcuda.dll belongs to the installed NVIDIA driver, not the app package.
    # Resolve every other imported symbol against the actual shipped/OS DLLs.
    # A driver-less runner cannot execute the CUDA DLL's initialization path.
    backend_pe = pefile.PE(str(stage / "ggml-cuda.dll"), fast_load=True)
    backend_pe.parse_data_directories(directories=[
        pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"],
        pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_EXPORT"],
    ])
    if backend_pe.FILE_HEADER.Machine != 0x8664 or not any(
        symbol.name == b"ggml_backend_init" for symbol in backend_pe.DIRECTORY_ENTRY_EXPORT.symbols
    ):
        raise ValueError("Expected an x64 GGML backend plugin")
    driver_available, resolved, retained = True, 0, []
    with os.add_dll_directory(str(stage)):
        for entry in backend_pe.DIRECTORY_ENTRY_IMPORT:
            name = entry.dll.decode("ascii")
            local = stage / name
            try:
                library = ctypes.CDLL(str(local) if local.is_file() else name)
            except OSError as error:
                driver_path = Path(os.environ["SystemRoot"]) / "System32/nvcuda.dll"
                if missing_nvidia_driver(name, error, driver_path):
                    driver_available = False
                    continue
                raise
            retained.append(library)
            for symbol in entry.imports:
                library[symbol.name.decode("ascii") if symbol.name else symbol.ordinal]
                resolved += 1
        if not driver_available:
            print(json.dumps({"driverAvailable": False, "resolvedImports": resolved,
                              "backendRegistration": "not-run-no-nvidia-driver", "gpuInferenceQualified": False}))
            return
        core = ctypes.CDLL(str(stage / "ggml.dll"))
        base = ctypes.CDLL(str(stage / "ggml-base.dll"))
        core.ggml_backend_load.argtypes = [ctypes.c_char_p]
        core.ggml_backend_load.restype = ctypes.c_void_p
        base.ggml_backend_reg_name.argtypes = [ctypes.c_void_p]
        base.ggml_backend_reg_name.restype = ctypes.c_char_p
        base.ggml_backend_reg_dev_count.argtypes = [ctypes.c_void_p]
        base.ggml_backend_reg_dev_count.restype = ctypes.c_size_t
        backend = core.ggml_backend_load(os.fsencode(stage / "ggml-cuda.dll"))
        if not backend or base.ggml_backend_reg_name(backend) != b"CUDA":
            raise RuntimeError("The shipped core did not load/register the upstream CUDA plugin")
        print(json.dumps({"driverAvailable": True, "resolvedImports": resolved,
                          "backendLoaded": "CUDA", "devices": base.ggml_backend_reg_dev_count(backend),
                          "gpuInferenceQualified": False}))


def compose(receipt_path: Path, output: Path, cache: Path) -> dict:
    if platform.system() != "Windows" or platform.machine().lower() not in ("amd64", "x86_64"):
        raise ValueError("Native Windows x64 validation is required")
    if output.exists():
        raise FileExistsError("Output exists; reconcile before retrying")
    base, files = verified_files(receipt_path)
    # Validate all archived parser-test evidence before creating a package.
    evidence = []
    for test in base["tests"]:
        suffix = "shared-peg-parser" if test["name"] == "test-chat-peg-parser-shared" else test["name"]
        data = (receipt_path.parent / f"win-cpu-x64-{suffix}.log").read_bytes()
        if sha(data) != test["logSha256"]:
            raise ValueError("Native CPU parser-test evidence differs")
        evidence.append((f"win-cuda-x64-{suffix}.log", data))
    sources = []
    for name in UPSTREAM:
        archive = download(name, cache)
        with zipfile.ZipFile(archive) as bundle:
            wanted = [n for n in bundle.namelist() if Path(n).name == "ggml-cuda.dll"] if name.startswith("llama-") else [
                n for n in bundle.namelist() if Path(n).name.lower().startswith(("cudart64", "cublas64", "cublaslt64")) and n.lower().endswith(".dll")]
            if len(wanted) != (1 if name.startswith("llama-") else 3):
                raise ValueError("Unexpected upstream CUDA file set")
            for member in wanted:
                basename = Path(member).name
                if basename in files:
                    raise ValueError("CUDA dependency would overwrite a tested runtime file")
                files[basename] = bundle.read(member)
        sources.append({"repository": "ggml-org/llama.cpp", "tag": "b11056", "filename": name,
                        "bytes": UPSTREAM[name][0], "sha256": UPSTREAM[name][1]})
    output.mkdir(parents=True)
    stage = output / "runtime"
    stage.mkdir()
    for name, data in files.items():
        (stage / name).write_bytes(data)
    for name, data in evidence:
        (output / name).write_bytes(data)
    tests = list(base["tests"])
    commands = [
        ("test-cuda-dependency-imports", [sys.executable, str(Path(__file__).resolve()), "--probe", str(stage)]),
        ("test-packaged-device-discovery", [str(stage / "llama-server.exe"), "--list-devices"]),
    ]
    hardware = None
    for name, command in commands:
        log = output / f"win-cuda-x64-{name}.log"
        with log.open("wb") as stream:
            result = subprocess.run(command, cwd=stage, stdout=stream, stderr=subprocess.STDOUT, timeout=120)
        if result.returncode != 0:
            raise RuntimeError(name + " failed; native logs retained")
        if name == "test-cuda-dependency-imports":
            hardware = json.loads(log.read_text().splitlines()[-1])
        if name == "test-packaged-device-discovery" and hardware["driverAvailable"] and b"loaded CUDA backend" not in log.read_bytes():
            raise RuntimeError("The actual server did not discover the CUDA plugin")
        tests.append({"name": name, "exitCode": 0, "linkage": "shared", "logSha256": sha(log.read_bytes())})
    version = subprocess.check_output([str(stage / "llama-server.exe"), "--version"], cwd=stage, stderr=subprocess.STDOUT, text=True)
    if version_identity(version) != version_identity(base["versionOutput"]):
        raise ValueError("Composed server version differs from the tested CPU executable")
    receipt = {**{k: v for k, v in base.items() if k not in ("archive", "files", "tests")},
               "variant": "win-cuda-x64", "tests": tests,
               "composition": {"scriptSha256": sha(Path(__file__).read_bytes()),
                   "nativeBaseReceiptSha256": sha(receipt_path.read_bytes()), "nativeBaseArchive": base["archive"],
                   "cudaSources": sources, "gpuInferenceQualified": False, "hardwareChecks": hardware,
                   "scope": "Unmodified upstream CUDA plugin at the exact same source commit; all non-driver imports resolved natively and packaged server discovery. CUDA registration runs only with an NVIDIA driver; GPU inference is not qualified."},
               "files": {name: {"sha256": sha(data), "bytes": len(data)} for name, data in sorted(files.items())}}
    (stage / "RUNTIME.json").write_text(json.dumps(receipt, indent=2) + "\n")
    stem = f"llama-{base['id']}-bin-win-cuda-x64"
    archive = output / (stem + ".zip")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for p in sorted(stage.iterdir()):
            bundle.write(p, p.name)
    data = archive.read_bytes()
    receipt["archive"] = {"filename": archive.name, "bytes": len(data), "sha256": sha(data)}
    (output / (stem + ".json")).write_text(json.dumps(receipt, indent=2) + "\n")
    shutil.rmtree(stage)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-receipt", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cache", type=Path, default=ROOT / "tmp/upstream-cuda-archives")
    parser.add_argument("--probe", type=Path)
    args = parser.parse_args()
    if args.probe:
        native_backend_probe(args.probe.resolve())
    else:
        if not args.cpu_receipt or not args.output:
            parser.error("--cpu-receipt and --output are required")
        print(json.dumps(compose(args.cpu_receipt.resolve(), args.output.resolve(), args.cache.resolve())["archive"], indent=2))

#!/usr/bin/env python3
"""Build, test and package pinned JSON parser fixes on native runners.

This does not change the product pin or publish anything. The resulting archive
and receipt are inputs to release qualification. Source, patch and every packaged
file are identified separately so a patched build cannot pass as an unpatched upstream build.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tarfile
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BUILDER_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
VARIANTS = ("macos-arm64", "macos-x64", "linux-x64", "win-cpu-x64", "win-vulkan-x64", "win-cuda-x64")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def patched_commit(spec: dict) -> str:
    prefix = spec["upstreamTag"] + "-"
    if not spec["id"].startswith(prefix) or not spec["id"][len(prefix):]:
        raise ValueError("Patch ID must extend the pinned upstream tag")
    return spec["upstreamCommit"][:10] + "-" + spec["id"][len(prefix):]


def run(args: list[str], cwd: Path, log: Path | None = None) -> str:
    print("RUN " + json.dumps(args), flush=True)
    if log:
        with log.open("w", encoding="utf-8") as stream:
            subprocess.run(args, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT, check=True)
        return log.read_text(encoding="utf-8", errors="replace")
    return subprocess.check_output(args, cwd=cwd, stderr=subprocess.STDOUT, encoding="utf-8")


def prepare(source: Path, spec: dict) -> None:
    patch = ROOT / spec["patch"]
    if sha(patch) != spec["patchSha256"]:
        raise ValueError("Patch differs from the pinned manifest")
    if not source.exists():
        source.mkdir(parents=True)
        run(["git", "init"], source)
        run(["git", "config", "core.autocrlf", "false"], source)
        run(["git", "remote", "add", "origin", f"https://github.com/{spec['upstreamRepository']}.git"], source)
        run(["git", "fetch", "--depth=1", "origin", spec["upstreamCommit"]], source)
        run(["git", "checkout", "--detach", "FETCH_HEAD"], source)
    if run(["git", "rev-parse", "HEAD"], source).strip() != spec["upstreamCommit"]:
        raise ValueError("Source checkout is not the pinned upstream commit")
    # A separate index computes the exact expected diff without modifying an
    # existing checkout. Reverse applicability alone misses unrelated edits in
    # the same files, which would contaminate the qualification provenance.
    clean = run(["git", "diff", "--binary", "--no-ext-diff", "HEAD", "--"], source)
    with tempfile.TemporaryDirectory(prefix="jurilix-patch-index-") as directory:
        env = {**os.environ, "GIT_INDEX_FILE": str(Path(directory) / "index")}
        subprocess.run(["git", "read-tree", "HEAD"], cwd=source, env=env, check=True)
        subprocess.run(["git", "apply", "--cached", str(patch)], cwd=source, env=env, check=True)
        expected = subprocess.check_output(["git", "diff", "--cached", "--binary", "--no-ext-diff", "HEAD"], cwd=source, env=env, encoding="utf-8")
    if clean and clean != expected:
        raise ValueError("Source contains changes other than the exact pinned patch")
    if not clean:
        run(["git", "apply", "--check", str(patch)], source)
        run(["git", "apply", str(patch)], source)


def configure_args(source: Path, build: Path, variant: str, spec: dict) -> list[str]:
    args = ["cmake", "-S", str(source), "-B", str(build), "-G", "Ninja",
            "-DCMAKE_BUILD_TYPE=Release", "-DBUILD_SHARED_LIBS=ON", "-DGGML_NATIVE=OFF",
            "-DLLAMA_BUILD_TESTS=ON", "-DLLAMA_BUILD_EXAMPLES=OFF", "-DLLAMA_BUILD_SERVER=ON",
            "-DLLAMA_OPENSSL=OFF", "-DLLAMA_BUILD_BORINGSSL=OFF",
            "-DGGML_METAL=OFF", "-DGGML_CUDA=OFF", "-DGGML_VULKAN=OFF",
            f"-DLLAMA_BUILD_NUMBER={spec['upstreamTag'][1:]}",
            f"-DLLAMA_BUILD_COMMIT={patched_commit(spec)}"]
    if variant.startswith("macos"):
        args += ["-DCMAKE_OSX_DEPLOYMENT_TARGET=13.3", "-DCMAKE_BUILD_RPATH=@loader_path",
                 "-DGGML_BLAS=ON", "-DGGML_BLAS_VENDOR=Apple"]
        if variant == "macos-arm64":
            # Precompiled shaders: no ~15 s Metal compile at launch when the system cache misses.
            args += ["-DGGML_METAL=ON", "-DGGML_METAL_EMBED_LIBRARY=OFF", "-DGGML_METAL_MACOSX_VERSION_MIN=13.3"]
    elif variant == "linux-x64":
        args += ["-DCMAKE_BUILD_RPATH=$ORIGIN"]
    if variant.startswith("win-"):
        args += ["-DGGML_BACKEND_DL=ON", "-DGGML_CPU_ALL_VARIANTS=ON"]
    if variant == "win-vulkan-x64":
        args += ["-DGGML_VULKAN=ON"]
    if variant == "win-cuda-x64":
        args += ["-DGGML_CUDA=ON", "-DGGML_CUDA_CUB_3DOT2=ON"]
    return args


def build_package(args: argparse.Namespace, spec: dict) -> None:
    source, work, out = args.source.resolve(), args.work.resolve(), args.output.resolve()
    prepare(source, spec)
    if args.prepare_only:
        return
    expected_os = "Windows" if args.variant.startswith("win-") else "Darwin" if args.variant.startswith("macos-") else "Linux"
    expected_arch = "arm64" if args.variant == "macos-arm64" else "x64"
    actual_arch = "arm64" if platform.machine().lower() in ("arm64", "aarch64") else "x64"
    if platform.system() != expected_os or actual_arch != expected_arch:
        raise ValueError("Native build/test runner does not match the requested variant")
    work.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    build = work / args.variant
    configure = configure_args(source, build, args.variant, spec)
    run(configure, source, out / f"{args.variant}-configure.log")
    # Upstream disables test-chat and grammar tests with Windows shared libs:
    # those tests intentionally call non-exported internals. Build the same
    # parser sources statically for their native tests, then build the shipped
    # shared runtime and test its exported PEG parser plus packaged executable.
    test_build = build
    if expected_os == "Windows":
        test_build = work / (args.variant + "-parser-tests")
        test_configure = configure_args(source, test_build, "win-cpu-x64", spec) + [
            "-DBUILD_SHARED_LIBS=OFF", "-DGGML_BACKEND_DL=OFF", "-DGGML_CPU_ALL_VARIANTS=OFF"]
        run(test_configure, source, out / f"{args.variant}-test-configure.log")
        run(["cmake", "--build", str(test_build), "--parallel", str(args.jobs), "--target", *spec["tests"]],
            source, out / f"{args.variant}-test-build.log")
    else:
        metal = ["ggml-metal-lib"] if args.variant == "macos-arm64" else []
        run(["cmake", "--build", str(build), "--parallel", str(args.jobs), "--target", "llama-server", *metal, *spec["tests"]],
            source, out / f"{args.variant}-build.log")
    suffix = ".exe" if expected_os == "Windows" else ""
    tests = []
    for name in spec["tests"]:
        log = out / f"{args.variant}-{name}.log"
        run([str(test_build / "bin" / (name + suffix))], source, log)
        tests.append({"name": name, "exitCode": 0, "linkage": "static" if expected_os == "Windows" else "shared", "logSha256": sha(log)})
    if expected_os == "Windows":
        run(["cmake", "--build", str(build), "--parallel", str(args.jobs), "--target", "llama-server", "test-chat-peg-parser"],
            source, out / f"{args.variant}-build.log")
        log = out / f"{args.variant}-shared-peg-parser.log"
        run([str(build / "bin/test-chat-peg-parser.exe")], source, log)
        tests.append({"name": "test-chat-peg-parser-shared", "exitCode": 0, "linkage": "shared", "logSha256": sha(log)})
    stage = work / (args.variant + "-package")
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir()
    for file in (build / "bin").iterdir():
        if file.is_file() and (file.name == "llama-server" + suffix or file.suffix in (".dll", ".dylib", ".metal", ".metallib") or ".so" in file.name):
            shutil.copy2(file, stage / file.name, follow_symlinks=True)
    if args.variant == "win-cuda-x64":
        cuda = Path(os.environ["CUDA_PATH"])
        for pattern in ("cudart64*.dll", "cublas64*.dll", "cublasLt64*.dll"):
            files = list((cuda / "bin").glob(pattern))
            if not files:
                raise ValueError(f"Missing CUDA redistributable: {pattern}")
            for file in files:
                shutil.copy2(file, stage / file.name)
    shutil.copy2(source / "LICENSE", stage / "LICENSE")
    (stage / "VERSION").write_text(spec["id"] + "\n")
    version = run([str(stage / ("llama-server" + suffix)), "--version"], stage)
    if f"build {spec['upstreamTag'][1:]}" not in version or patched_commit(spec) not in version:
        raise ValueError("Packaged executable does not identify the source and patch")
    receipt = {"schema": "jurilix.patched-runtime.v1", "id": spec["id"], "variant": args.variant,
               "upstreamRepository": spec["upstreamRepository"], "upstreamCommit": spec["upstreamCommit"],
               "upstreamTag": spec["upstreamTag"], "patchSha256": spec["patchSha256"],
               "sourceDiffSha256": hashlib.sha256(run(["git", "diff", "HEAD"], source).encode()).hexdigest(),
               "builderSha256": BUILDER_SHA256, "configure": configure,
               "tests": tests, "versionOutput": version,
               "files": {p.name: {"sha256": sha(p), "bytes": p.stat().st_size} for p in sorted(stage.iterdir())}}
    (stage / "RUNTIME.json").write_text(json.dumps(receipt, indent=2) + "\n")
    stem = f"llama-{spec['id']}-bin-{args.variant}"
    if expected_os == "Windows":
        archive = out / (stem + ".zip")
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            for p in sorted(stage.iterdir()):
                bundle.write(p, p.name)
    else:
        archive = out / (stem + ".tar.gz")
        with tarfile.open(archive, "w:gz") as bundle:
            for p in sorted(stage.iterdir()):
                bundle.add(p, arcname=p.name, recursive=False)
    receipt["archive"] = {"filename": archive.name, "sha256": sha(archive), "bytes": archive.stat().st_size}
    (out / (stem + ".json")).write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"archive": str(archive), **receipt["archive"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--source", type=Path, default=ROOT / "tmp/gemma-runtime/source")
    parser.add_argument("--work", type=Path, default=ROOT / "tmp/gemma-runtime/build")
    parser.add_argument("--output", type=Path, default=ROOT / "tmp/gemma-runtime/artifacts")
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--manifest", type=Path, default=ROOT / "scripts/llama-patch.json")
    options = parser.parse_args()
    build_package(options, json.loads(options.manifest.read_text()))

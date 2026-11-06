#!/usr/bin/env python3
"""Check the exact native CUDA package's imports and server discovery on Windows.

This does not require, or claim, GPU inference on driver-less CI runners.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def file_sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_report(receipt_path: Path, report_path: Path, *, probe_sha: str | None = None) -> dict:
    report = json.loads(report_path.read_text())
    if report['nativeReceiptSha256'] != file_sha(receipt_path):
        raise ValueError('Native CUDA probe belongs to a different receipt')
    if report['scriptSha256'] != (probe_sha or file_sha(Path(__file__))):
        raise ValueError('Native CUDA probe script differs')
    if report['gpuInferenceQualified'] is not False:
        raise ValueError('Native CI checks do not qualify GPU inference')
    if {test['name'] for test in report['tests']} != {'test-cuda-dependency-imports', 'test-packaged-device-discovery'}:
        raise ValueError('Both native CUDA package probes are required')
    for test in report['tests']:
        if test['exitCode'] != 0:
            raise ValueError('Native CUDA package probe failed')
        log = receipt_path.parent / f"win-cuda-x64-{test['name']}.log"
        if file_sha(log) != test['logSha256']:
            raise ValueError('Native CUDA probe log differs')
    if report['hardwareChecks'].get('gpuInferenceQualified') is not False:
        raise ValueError('Native CI checks do not qualify GPU inference')
    return report


def probe(receipt_path: Path, stage: Path) -> Path:
    if platform.system() != 'Windows':
        raise ValueError('Native Windows validation is required')
    receipt = json.loads(receipt_path.read_text())
    if receipt['variant'] != 'win-cuda-x64' or 'composition' in receipt:
        raise ValueError('Expected a fully native CUDA build')
    if json.loads((stage / 'RUNTIME.json').read_text()) != {k:v for k,v in receipt.items() if k != 'archive'}:
        raise ValueError('Staged runtime and native receipt differ')
    if {p.name for p in stage.iterdir()} != set(receipt['files']) | {'RUNTIME.json'}:
        raise ValueError('Native CUDA package file set differs')
    for name, expected in receipt['files'].items():
        file = stage / name
        if file.stat().st_size != expected['bytes'] or file_sha(file) != expected['sha256']:
            raise ValueError('Native CUDA package differs: ' + name)
    commands = [
        ('test-cuda-dependency-imports', [sys.executable, str(ROOT / 'scripts/compose-cuda-runtime.py'), '--probe', str(stage)]),
        ('test-packaged-device-discovery', [str(stage / 'llama-server.exe'), '--list-devices']),
    ]
    tests = []
    hardware = None
    for name, command in commands:
        log = receipt_path.parent / f'win-cuda-x64-{name}.log'
        with log.open('wb') as stream:
            result = subprocess.run(command, cwd=stage, stdout=stream, stderr=subprocess.STDOUT, timeout=120)
        if result.returncode:
            raise RuntimeError(name + ' failed; native evidence retained')
        if name == 'test-cuda-dependency-imports':
            hardware = json.loads(log.read_text().splitlines()[-1])
        elif hardware['driverAvailable'] and b'loaded CUDA backend' not in log.read_bytes():
            raise RuntimeError('The packaged server did not discover the CUDA plugin')
        tests.append({'name':name, 'exitCode':0, 'logSha256':file_sha(log)})
    path = receipt_path.parent / 'native-cuda-probe.json'
    path.write_text(json.dumps({'schema':'jurilix.native-cuda-probe.v1',
        'nativeReceiptSha256':file_sha(receipt_path), 'scriptSha256':file_sha(Path(__file__)),
        'probeDependencySha256':file_sha(ROOT / 'scripts/compose-cuda-runtime.py'),
        'tests':tests, 'hardwareChecks':hardware, 'gpuInferenceQualified':False},indent=2)+'\n')
    verify_report(receipt_path,path)
    return path


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt',type=Path,required=True)
    parser.add_argument('--stage',type=Path,required=True)
    options=parser.parse_args()
    print(probe(options.receipt.resolve(),options.stage.resolve()))

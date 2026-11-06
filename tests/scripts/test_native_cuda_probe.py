import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('native_cuda_probe', ROOT / 'scripts/native-cuda-probe.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class NativeCudaProbeEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.receipt = self.root / 'native-receipt.json'
        self.receipt.write_text('{"variant":"win-cuda-x64"}')
        tests = []
        for name in ('test-cuda-dependency-imports', 'test-packaged-device-discovery'):
            log = self.root / f'win-cuda-x64-{name}.log'
            log.write_text(name + ' passed\n')
            tests.append({'name':name,'exitCode':0,'logSha256':probe.file_sha(log)})
        self.report = {'nativeReceiptSha256':probe.file_sha(self.receipt),
            'scriptSha256':probe.file_sha(ROOT / 'scripts/native-cuda-probe.py'),
            'tests':tests,'hardwareChecks':{'driverAvailable':False,'gpuInferenceQualified':False},
            'gpuInferenceQualified':False}
        self.path = self.root / 'native-cuda-probe.json'

    def write_report(self):
        self.path.write_text(json.dumps(self.report))

    def test_accepts_exact_successful_evidence_without_claiming_gpu_inference(self):
        self.write_report()
        self.assertEqual(probe.verify_report(self.receipt,self.path), self.report)

    def test_rejects_probe_from_another_native_runtime(self):
        self.write_report()
        self.receipt.write_text('{"variant":"different-runtime"}')
        with self.assertRaisesRegex(ValueError,'different receipt'):
            probe.verify_report(self.receipt,self.path)

    def test_rejects_changed_probe_log(self):
        self.write_report()
        (self.root/'win-cuda-x64-test-cuda-dependency-imports.log').write_text('changed')
        with self.assertRaisesRegex(ValueError,'log differs'):
            probe.verify_report(self.receipt,self.path)

    def test_requires_both_probes_and_rejects_gpu_qualification_claims(self):
        self.report['tests'].pop()
        self.write_report()
        with self.assertRaisesRegex(ValueError,'Both native'):
            probe.verify_report(self.receipt,self.path)
        self.report['gpuInferenceQualified']=True
        self.write_report()
        with self.assertRaisesRegex(ValueError,'do not qualify'):
            probe.verify_report(self.receipt,self.path)


if __name__ == '__main__':
    unittest.main()

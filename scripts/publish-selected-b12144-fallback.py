from pathlib import Path
import hashlib
import json
import subprocess
import tempfile
import time
import zipfile
import tarfile

ROOT = Path(__file__).resolve().parents[1]
REPO = 'Makio64/jurilix-llama-runtime'
TAG = 'b12144-jurilix1'
RUN = '42174018431'
SOURCE_HEAD = '15ddacd9d5aad44d593ce252a8fa396b6098fa9e'
EVIDENCE = ROOT / 'tmp/llama-runtime-b12144/fallback-publication'
CODE = ROOT


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def get_json(*command):
    return json.loads(subprocess.check_output(command,text=True))


def get_release():
    releases = get_json('gh','api',f'repos/{REPO}/releases?per_page=100')
    release, = [item for item in releases if item['tag_name']==TAG]
    return get_json('gh','api',f"repos/{REPO}/releases/{release['id']}")


run = get_json('gh' ,'run','view',RUN,'--repo',REPO,'--json','conclusion,headSha,workflowName')
assert run == {'conclusion':'success','headSha':SOURCE_HEAD,'workflowName':'Build patched llama.cpp runtime'},run
pin = json.loads((CODE/'scripts/llama-patch.json').read_text())
EVIDENCE.mkdir(parents=True,exist_ok=True)
subprocess.run(['gh','run','download',RUN,'--repo',REPO,'--name','jurilix-cuda-fallback-metadata','--dir',str(EVIDENCE)],check=True)
compact = json.loads((EVIDENCE/'ci-validation.json').read_text())
assert {row['variant'] for row in compact} == {'win-cpu-x64','win-vulkan-x64','win-cuda-x64'}

with tempfile.TemporaryDirectory(prefix='jurilix-b12144-selected-publication-') as directory:
    payload = Path(directory)/'payload'
    for attempt in range(3):
        result = subprocess.run(['gh','run','download',RUN,'--repo',REPO,'--name','jurilix-cuda-fallback-verified','--dir',str(payload)],capture_output=True,text=True)
        if result.returncode == 0:
            break
        print('Payload download attempt',attempt+1,'failed:',result.stderr.rsplit(': ',1)[-1].strip(),flush=True)
        if attempt == 2:
            raise SystemExit(1)
        time.sleep(2)
    metadata_path, = payload.rglob('ci-validation.json')
    publish = metadata_path.parent
    metadata = json.loads(metadata_path.read_text())
    assert metadata == compact
    receipts = {}
    for row in metadata:
        variant = row['variant']
        receipt_path = publish/f'llama-{TAG}-bin-{variant}.json'
        receipt = json.loads(receipt_path.read_text())
        assert sha(receipt_path) == row['receiptSha256']
        for key in ('id','upstreamRepository','upstreamCommit','upstreamTag','patchSha256'):
            assert receipt[key] == pin[key],(variant,key)
        assert receipt['sourceDiffSha256'] == pin['patchSha256']
        assert receipt['builderSha256'] == sha(CODE/'scripts/build-patched-llama.py')
        assert all(test['exitCode']==0 for test in receipt['tests'])
        assert len(receipt['tests']) == row['testsVerified']
        assert len(receipt['files']) == row['filesVerified']
        with tarfile.open(publish/f'native-test-evidence-{variant}.tar.gz') as logs:
            for test in receipt['tests']:
                suffix = 'shared-peg-parser' if test['name']=='test-chat-peg-parser-shared' else test['name']
                data = logs.extractfile(f'{variant}-{suffix}.log').read()
                assert hashlib.sha256(data).hexdigest() == test['logSha256']
        names = set()
        for archive in row['archives']:
            path = publish/archive['filename']
            assert path.stat().st_size == archive['bytes'] and sha(path) == archive['sha256']
            with zipfile.ZipFile(path) as bundle:
                for name in bundle.namelist():
                    assert Path(name).name == name and '/' not in name and '\\' not in name and name not in names
                    names.add(name)
                    data = bundle.read(name)
                    if name=='RUNTIME.json':
                        assert json.loads(data) == {k:v for k,v in receipt.items() if k!='archive'}
                    else:
                        assert {'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()} == receipt['files'][name]
        assert names == set(receipt['files']) | {'RUNTIME.json'}
        receipts[variant] = receipt
    cuda = receipts['win-cuda-x64']
    composition = cuda['composition']
    assert composition['scriptSha256'] == sha(CODE/'scripts/compose-cuda-runtime-b12144.py')
    assert composition['nativeBaseReceiptSha256'] == sha(publish/f'llama-{TAG}-bin-win-cpu-x64.json')
    assert composition['nativeBaseArchive'] == receipts['win-cpu-x64']['archive']
    assert composition['cudaSources'] == [
        dict(repository='ggml-org/llama.cpp',tag='b12144',filename='llama-b12144-bin-win-cuda-12.4-x64.zip',bytes=288363637,sha256='fab315401a4784fa9c98c3105bf6e2ede6d3d2dafceb873e9bccd169f0f32cb7'),
        dict(repository='ggml-org/llama.cpp',tag='b12144',filename='cudart-llama-bin-win-cuda-12.4-x64.zip',bytes=391443627,sha256='8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6')]
    assert composition['gpuInferenceQualified'] is False
    assert composition['hardwareChecks'] == metadata[2]['hardwareChecks']
    split = json.loads((publish/f'llama-{TAG}-split-win-cuda-x64.json').read_text())
    assert split['nativeReceiptSha256'] == sha(publish/f'llama-{TAG}-bin-win-cuda-x64.json')
    assert split['repackagedFrom'] == cuda['archive']
    assert split['composition'] == composition
    assert [split['archive'],*split['extras']] == metadata[2]['archives']
    assert split['packagerSha256'] == sha(CODE/'scripts/split-cuda-runtime.py')
    release = get_release()
    assert release['draft'] is True
    print('Selected payload verified: 59 runtime files, 17 native tests, four archive digests.',flush=True)
    files = sorted(path for path in publish.iterdir() if path.is_file())
    subprocess.run(['gh','release','upload',TAG,*map(str,files),'--repo',REPO,'--clobber'],check=True)
    release = get_release()
    uploaded = {asset['name']:asset for asset in release['assets']}
    for row in metadata:
        for archive in row['archives']:
            asset = uploaded[archive['filename']]
            assert asset['size'] == archive['bytes']
            assert asset['digest'] == 'sha256:'+archive['sha256']
    evidence = {'route':'patched-jurilix-core-with-official-b12144-cuda','runId':RUN,'sourceHead':SOURCE_HEAD,
        'releaseId':release['id'],'releaseDraft':release['draft'],'targetCommitish':release['target_commitish'],
        'filesVerified':sum(row['filesVerified'] for row in metadata),'testsVerified':sum(row['testsVerified'] for row in metadata),
        'assets':[{key:asset[key] for key in ('name','size','digest','browser_download_url')} for asset in release['assets']]}
    (EVIDENCE/'release-assets-verification.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps(evidence,indent=2),flush=True)
    print('Temporary payload bytes removed on exit:',sum(path.stat().st_size for path in payload.rglob('*') if path.is_file()),flush=True)

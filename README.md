# Jurilix llama.cpp runtime

Reproducible native builds of llama.cpp with the Gemma structured-output parser
fix used by Jurilix. The release identity is **b10956-jurilix1**.

## Source and change

- Upstream: [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp), commit
  `21f6b0d22c6c87efd42d22485cd8b13f61ad5736` (b10956).
- Patch: `scripts/finetune/patches/llama-cpp-gemma4-plain-json-b10956-2026-09-15.diff`.
- Patch SHA-256: `a45ebb7cd44e8da57af1b237c34dd4487684c617963d861568cfdcbf4835c73c`.

The Gemma parser accepts schema-constrained plain JSON as well as its existing
fenced JSON format. Tests cover structured output, reasoning and tool calls.
The engine identifies itself with the upstream build and `-jurilix1` suffix.
This repository contains runtime build sources; model weights and application
sources are not included.

## Build

Use a native runner matching the requested variant, with Python 3.10+, CMake,
Ninja and a C++ compiler. Windows Vulkan uses the Vulkan SDK. The workflow
records the native runner setup.

```sh
python scripts/build-patched-llama.py --variant macos-arm64 --jobs 3
```

Other variants: `macos-x64`, `linux-x64`, `win-cpu-x64`, `win-vulkan-x64`,
`win-cuda-x64`. Each build checks the exact source diff, executes parser and
grammar tests, checks the packaged executable version, and writes a receipt
with archive and individual file SHA-256 values. Windows uses a static build
for parser internals and a shared build for the delivered runtime, with an
additional shared PEG parser test.

### Windows CUDA

The release uses the tested patched Windows CPU/core runtime and the unchanged
**CUDA 12.4 plugin from upstream b10956**, at the same pinned source commit.
Only the Gemma JSON parser and tests are patched; the CUDA kernel code is not.
Both upstream archives are pinned by size and SHA-256 in the composition recipe.
This avoids rebuilding unchanged CUDA kernels for a parser-only change.

On native Windows x64, first build/test the CPU runtime, then compose CUDA:

```sh
python scripts/build-patched-llama.py --variant win-cpu-x64 --jobs 3
python -m pip install pefile==2024.8.26
python scripts/compose-cuda-runtime.py \
  --cpu-receipt tmp/gemma-runtime/artifacts/llama-b10956-jurilix1-bin-win-cpu-x64.json \
  --output tmp/gemma-runtime/cuda-original
```

Composition verifies the original native receipt, every runtime file and all
five parser-test logs. Native checks resolve every non-driver import from the
CUDA plugin against the actual shipped or Windows DLL, and run the packaged
server device-discovery path. Missing dependency exports or executable-version
mismatch fail the package. `nvcuda.dll` belongs to the NVIDIA driver; the app
only downloads this variant when that driver is installed.

On a runner with that driver, the shipped GGML core must also load/register
the plugin through its version-checked backend API and the actual server must
discover it. Without a driver, the receipt explicitly records that CUDA
registration was **not run**; it is not reported as a passing GPU test. The
composition receipt identifies every dependency and retained native test.
GPU inference remains unqualified by these packaging checks.

The validated package is split without changing file bytes:

```sh
python scripts/split-cuda-runtime.py \
  --receipt tmp/gemma-runtime/cuda-original/llama-b10956-jurilix1-bin-win-cuda-x64.json \
  --output tmp/gemma-runtime/cuda-downloads
```

Extract both the main CUDA archive and its `cudart-*.zip` companion into the
same directory. The split receipt links both downloads to the native build.

## Per-request LoRA use

With this upstream version, always pass `--cache-ram 0` when changing LoRA
strengths between requests. The RAM prompt cache does not record LoRA identity
and can restore KV states computed at different strengths. Slot-local prompt
caching still performs its ordinary invalidation when LoRA strengths change.
Load adapters with `--lora-init-without-apply`, then supply the explicit scale
for each request, including zero outside the adapter's intended tasks.

Native parser/build checks do not establish model quality, GPU performance or
application-level qualification. Those depend on the exact weights, request
policy and target machine. The release receipts describe the checks performed.

## License

llama.cpp and the parser modification are distributed under the MIT license.
See `LICENSE`. CUDA redistributable libraries retain NVIDIA's applicable terms.

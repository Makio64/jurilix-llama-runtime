# Jurilix llama.cpp runtime

Reproducible native builds of llama.cpp with the Gemma and Ministral structured-output parser
fixes used by Jurilix. The release identity is **b11056-jurilix2**.

## Source and change

- Upstream: [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp), commit
  `e613ef2c81bae98d59850d061ac29e6e3e88cb00` (b11056).
- Patch: `scripts/finetune/patches/llama-cpp-gemma4-ministral3-plain-json-b11056-2026-09-20.diff`.
- Patch SHA-256: `5db33e588e8bca8051b8cdafd6d339f79a0ba2b5f461c09c49784d64dc65b155`.

The Gemma and Ministral parsers accept schema-constrained plain JSON as well as their existing
fenced JSON formats. Tests cover structured output, reasoning and tool calls.
The engine identifies itself with the upstream build and `-jurilix2` suffix.
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
**CUDA 12.4 plugin from upstream b11056**, at the same pinned source commit.
Only the Gemma/Ministral JSON parsers and tests are patched; the CUDA kernel code is not.
Both upstream archives are pinned by size and SHA-256 in the composition recipe.
This avoids rebuilding unchanged CUDA kernels for a parser-only change.

On native Windows x64, first build/test the CPU runtime, then compose CUDA:

```sh
python scripts/build-patched-llama.py --variant win-cpu-x64 --jobs 3
python -m pip install pefile==2024.8.26
python scripts/compose-cuda-runtime.py \
  --cpu-receipt tmp/gemma-runtime/artifacts/llama-b11056-jurilix2-bin-win-cpu-x64.json \
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
  --receipt tmp/gemma-runtime/cuda-original/llama-b11056-jurilix2-bin-win-cuda-x64.json \
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

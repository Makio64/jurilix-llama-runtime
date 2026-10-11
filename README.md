# Jurilix llama.cpp runtime

Reproducible native builds of `b11552-jurilix1`, based on upstream llama.cpp
commit `23b0202a189c44a54625aadcb37a946dd1d6278d`.

The pinned patch lets Gemma 4 and Ministral 3 emit either direct JSON or fenced
JSON when a response schema is used. Both paths, empty extractions and
reasoning variants are covered by native parser tests. The patch also covers
Ministral 3 reasoning-disabled extraction.

`python scripts/build-patched-llama.py --variant macos-arm64` builds and tests
on a native Apple Silicon runner. Other native variants are listed by `--help`.
`python -m unittest discover -s tests/scripts -p 'test_*.py'` checks provenance
and packaging behavior without downloading model weights.

The manually dispatched build workflow produces macOS, Linux and Windows
CPU/Vulkan artifacts. Windows CUDA combines the exact same upstream b11552
CUDA 12.4 backend with the native-tested patched core, checks native imports
and device discovery, then splits the result into bounded download archives.
A runner without an NVIDIA driver cannot qualify CUDA GPU inference.

Dispatching the same workflow with `publish_run_id` and `release_tag` verifies
that run's Windows receipts, parser logs and archives, then attaches them to
the release without downloading them locally.

Every runtime archive includes a version identity and a receipt with source,
patch, file hashes and parser-test evidence. Published archives are independent
of the source build recipe: updating this repository does not activate an
application pin or publish a runtime release. Model quality requires separate
qualification on the actual model and adapter bytes.

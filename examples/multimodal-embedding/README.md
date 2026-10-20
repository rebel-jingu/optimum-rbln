# EmbeddingGemma 2: text, images and audio

This fork provides `RBLNEmbeddingGemma2Model`, its native configuration, automatic
model discovery, and the ordinary optimum-rbln compile/save/reload lifecycle.
The implementation migrates the previously validated standalone multimodal
encoder graphs, metadata cache and CPU media overlap.

This is an **embedding-only API**. `forward(...).embeddings` is a normalized CPU
float32 tensor of shape `[1, dimensions]`; `return_dict=False` returns
`(embeddings,)`. Supported dimensions are 768, 512, 256 and 128. It does not
return the HF model's per-token hidden states. Video, custom positions/embeddings,
hidden-state/attention output requests, invalid masks and capacity overflow are
rejected. Call requests sequentially; the model and metadata cache are not a
concurrent serving interface.

## Isolated environment

EmbeddingGemma 2 requires Transformers 5.20.1. The upstream project currently
pins an older stack. This example keeps those shared pins intact and uses a
separate, model-specific Linux x86-64 environment. Editable installation below
uses `--no-deps` deliberately: this environment's pyproject/lock supplies the
newer stack rather than replacing it with upstream's default pins. Other
upstream models are not certified against this environment.

From the repository root:

```bash
export UV_PROJECT_ENVIRONMENT="$PWD/.venv-embeddinggemma2"
uv sync --project examples/multimodal-embedding --python 3.12
uv pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" /absolute/path/to/rebel_compiler.whl
uv pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" --no-deps -e .
export OMP_NUM_THREADS=4 RBLN_NUM_THREADS=4
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 RBLN_PROFILER=0 RBLN_RUNTIME_TIMER=0
```

Use SDK >=0.12 for CR13. The existing CA25 measurements use SDK 0.11.0.post1.
The private SDK wheel is installed explicitly and is not stored in Git.
After setup, use `uv run --project examples/multimodal-embedding --no-sync ...`
so the explicitly installed SDK/editable package remain installed.

## Compile or migrate a validated bundle

```bash
# CR13: one device exposes four chiplets.
uv run --project examples/multimodal-embedding --no-sync python \
  examples/multimodal-embedding/run_embedding_gemma2.py compile --devices 0 \
  --npu RBLN-CR13 --output .cache/compiled-cr13

# CA25: use four logical NPUs. The compiler chooses their distribution.
uv run --project examples/multimodal-embedding --no-sync python \
  examples/multimodal-embedding/run_embedding_gemma2.py compile --devices 0 1 2 3 \
  --npu RBLN-CA25 --audio-layout selector --output .cache/compiled-ca25

# Import existing verified binaries, preserving hashes and layout provenance.
# Loads the pinned source checkpoint ONCE to extract token embeddings and
# positional buffers; subsequent native loads do not need that checkpoint.
uv run --project examples/multimodal-embedding --no-sync python \
  examples/multimodal-embedding/run_embedding_gemma2.py import --devices 0 1 2 3 \
  --bundle /absolute/path/to/ca25-sdk011-bf16-70-quad-selector1056 \
  --output .cache/native-ca25
```

The selected older bundle has the selector layout only in `audio:1056`; other
graphs retain their original layout. Import copies these exact binaries and
stores the complete original manifest in `legacy_provenance.json`. It is not a
new compile or a new performance measurement. The `audio_layout` configuration
controls **future compilation**, not the contents of imported graphs. Fresh
`--audio-layout selector` compilation applies the selector to every audio bucket.

## Native Python API

```python
from transformers import AutoProcessor
from optimum.rbln import RBLNAutoModel

directory = ".cache/native-ca25"
processor = AutoProcessor.from_pretrained(directory)
model = RBLNAutoModel.from_pretrained(directory, export=False, rbln_device=[0, 1, 2, 3])

inputs = processor(text="오로라가 생기는 원인은?", return_tensors="pt")
embedding = model(**inputs).embeddings

# Raw file inputs also include decode, audio resampling and CPU processing.
embedding = model.encode(
    {
        "text": "Picture <|image|> and sound <|audio|>",
        "images": ["/absolute/path/photo.png"],
        "audio": ["/absolute/path/speech.flac"],
    },
    processor=processor,
    dimensions=256,
)
model.save_pretrained(".cache/native-ca25-saved")
model.close()
```

When processing images directly through `AutoProcessor`, set
`images_kwargs={"max_soft_tokens": model.rbln_config.image_soft_tokens}` to match
the compiled shape. `model.prepare` and `model.encode` set this automatically.
In-memory audio arrays must be finite mono float waveforms sampled at 16 kHz.
Compiled reload loads only native configs, the `.rbln` graphs, and tensor-only
`host_state.pt`; it does not construct/download the full HF model. `from_model`
and `from_pretrained(export=True)` provide the native fresh-compilation path.
When calling `from_legacy_bundle` directly, pass `source_revision` equal to the
revision used to load `source_model`; newer Transformers may omit the config's
commit metadata. A conflicting stored revision is still rejected.

## E2E and Perfetto

Create a JSON object mapping case names to raw requests, for example
`{"text": {"text": "Hello"}, "speech": {"audio": ["/absolute/path/speech.flac"]}}`.

```bash
uv run --project examples/multimodal-embedding --no-sync python \
  examples/multimodal-embedding/run_embedding_gemma2.py benchmark --devices 0 1 2 3 \
  --model .cache/native-ca25 --requests /absolute/path/requests.json \
  --output .cache/native-benchmark --warmup 3 --repeats 100 --trace
```

`benchmark.json` records raw samples and median/p95 for CPU prepare, prepared
native forward, and E2E. E2E includes decode/processor work, CPU masks/token
assembly, transfers, synchronous encoder execution and normalized CPU output.
Model load, compilation and warmup are excluded. `--trace` makes an additional
pass after timing; open its `host-trace.json` at https://ui.perfetto.dev.
Host spans cannot establish device-internal idle/bubble intervals. The older
CA25 SDK's native device profiler previously failed; this example does not
claim an internal device trace is available.

```bash
uv run --project examples/multimodal-embedding --no-sync python -m pytest -q \
  tests/unit/cpu/test_embedding_gemma2.py
```

These tests skip under upstream's default Transformers 5.15.1 environment.
With the model-specific 5.20.1 stack, upstream's Qwen2.5-VL CPU test currently
fails collection because `Qwen2_5_VisionRotaryEmbedding` was removed upstream;
that unrelated adapter is not changed by this migration.

## Verified on CA25, 2026-10-20

The measured implementation is commit `92aa1c96cd0083718092c9fde026ff859cb8ad2c`.
The model-specific environment uses PyTorch 2.14.1+cpu, Transformers 5.20.1,
rebel-compiler 0.11.0.post1, BF16 weights, and four logical CA25 NPUs.

- **99 new CPU tests pass**, plus 35 existing compatible CPU tests (134 total).
  Ruff passes. The Qwen2.5-VL collection incompatibility described above is
  excluded from this count, rather than reported as a passing full suite.
- **22 full-weight native checks pass** on the imported bundle: text, images,
  audio, mixed/multiple media, modified speech, dimensions and capacity errors.
  All 18 prepared-input hashes match the previous selector-bundle validation;
  minimum final cosine against the official reference is **0.9988199262**.
- **Fresh native compilation succeeds** for `text_128`, `vision_630` and
  `audio_64` with the selector layout and four devices. Its separate eight
  checks pass, including all four modality combinations; minimum numerical
  cosine is **0.9998226806**. This is a small compile smoke bundle, not a claim
  that all default buckets were freshly recompiled in this run.
- All 12 imported binaries retain their original hashes. Actual binary target
  and device counts were inspected for the imported and fresh graphs.
- Compiled reload retains 134,217,728 token-embedding parameters on CPU,
  versus the reference model's 744,371,488 total parameters. It constructs no
  encoder layers; the CPU test makes full HF constructors/loaders fail during
  compiled reload to verify this boundary.

Four public requests, three warmups and 100 uninstrumented samples each, with
Torch and SDK both at four CPU threads:

| Request | CPU prepare median (ms) | Native forward median (ms) | E2E median (ms) |
| --- | ---: | ---: | ---: |
| Text | 0.157 | 2.750 | 2.907 |
| Photo | 7.045 | 12.451 | 19.495 |
| Speech | 4.486 | 16.888 | 21.379 |
| Text + photo + speech | 8.083 | 26.900 | 34.980 |

The medians are computed independently and need not add exactly. These are new
native-adapter measurements; older standalone/H200/CR13 results retain their
original attribution. The benchmark stores its Git commit and package versions.
This migration does not establish a new CR13 measurement: the CR13 SSH master
connection expired, so that target's fork-based physical revalidation remains
outstanding.

The additional host trace contains four requests and a `host.validate` span
for each; the official Perfetto processor parsed all 40 slices successfully.
Grid validation uses integer coordinate counts rather than sorting
2D rows; an isolated 400-sample CPU check measured 0.195 ms for the previous
unique-row check and 0.00875 ms for the count check. The final separate trace
observed 0.420 ms of validation for photo and 0.553 ms for mixed input. Trace
durations are diagnostic and are excluded from latency samples.

The generated files live under `.cache/native-benchmark-final/` and
`.cache/native-validation-bincount.json` on the validation checkout. They are
not included in Git. All work processes exited and devices 0–3 were checked
at zero allocated bytes and zero utilization afterward. `model.close()` closes
CPU preprocessing workers; native runtimes follow the model object's lifetime.

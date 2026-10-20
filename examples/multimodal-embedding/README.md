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
uv pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" --no-deps \
  /absolute/path/to/rebel_compiler.whl -e .
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
  examples/multimodal-embedding/run_embedding_gemma2.py --devices 0 compile \
  --npu RBLN-CR13 --output .cache/compiled-cr13

# CA25: use four logical NPUs. The compiler chooses their distribution.
uv run --project examples/multimodal-embedding --no-sync python \
  examples/multimodal-embedding/run_embedding_gemma2.py --devices 0 1 2 3 compile \
  --npu RBLN-CA25 --audio-layout selector --output .cache/compiled-ca25

# Import existing verified binaries, preserving hashes and layout provenance.
# Loads the pinned source checkpoint ONCE to extract token embeddings and
# positional buffers; subsequent native loads do not need that checkpoint.
uv run --project examples/multimodal-embedding --no-sync python \
  examples/multimodal-embedding/run_embedding_gemma2.py --devices 0 1 2 3 import \
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
    {"text": "Picture <|image|> and sound <|audio|>",
     "images": ["/absolute/path/photo.png"], "audio": ["/absolute/path/speech.flac"]},
    processor=processor, dimensions=256,
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

## E2E and Perfetto

Create a JSON object mapping case names to raw requests, for example
`{"text": {"text": "Hello"}, "speech": {"audio": ["/absolute/path/speech.flac"]}}`.

```bash
uv run --project examples/multimodal-embedding --no-sync python \
  examples/multimodal-embedding/run_embedding_gemma2.py --devices 0 1 2 3 benchmark \
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

# EmbeddingGemma 2 integration

## Intent and authority

The user requested moving the existing, physically validated text/image/audio
EmbeddingGemma 2 implementation into a fork of optimum-rbln. This changes the
packaging and model lifecycle of the already approved implementation. Preserve
its bidirectional semantics, precision, numerical checks, performance work, and
reproducible profiling. The fork is rebel-jingu/optimum-rbln; upstream dev starts
at f5a14555263843dffc127dc36673908153ac519f.

## Public interface

Add RBLNEmbeddingGemma2Model and RBLNEmbeddingGemma2ModelConfig, exported through
the existing lazy import system and discoverable through RBLNAutoModel. Use
from_pretrained(export=True), from_model, save_pretrained, and compiled
from_pretrained through the ordinary RBLNModel lifecycle. forward accepts the
official processor tensors and returns RBLNEmbeddingGemma2Output.embeddings,
a CPU float32 tensor of shape [1, dimensions]. return_dict=False returns the
single-element tuple (embeddings,). Normalize after dimension truncation. This
is an embedding-only API, not HF per-token output compatibility. Batch size
remains one; supported dimensions are
768, 512, 256, and 128. Explicitly reject unsupported options and capacity
overflow before any device execution.

## Architecture

Adapt the verified text, vision, and audio graph implementations into the model
directory. Use named RBLNCompileConfig entries for every static bucket, carrying
the common npu and num_devices settings. Instantiate native runtimes in the same
order and route each stage through its compiled input signature. Preserve
constant-folded input removal. Keep metadata caches and optional request-local
CPU media overlap; never cache input media or embeddings.

Store only the token embedding and small host positional state alongside the
compiled model. Reload must not fetch the full original checkpoint or allocate
CPU copies of NPU encoder weights. Use tensor-only state loading. The compiled
model config, native RBLN config, processor files, and host tensor artifacts
must survive a save/reload cycle.

Provide an explicit conversion entry point for existing verified standalone
bundles. Validate their file hashes and signatures; copy the unchanged binaries
into the native naming/configuration layout, preserving original provenance.
Conversion accepts the pinned source model once to extract host state.
Compiled reload constructs neither the full HF model nor a checkpoint loader.
Conversion is not recompilation, and historical numerical/performance results
must not be relabeled as new optimum-rbln measurements.

Validate paired image/audio tensors, shapes and binary masks before any runtime
call. Default a missing text attention mask to all-valid. Reject non-prefix
audio validity masks, videos, custom positions/embeddings, hidden-state and
attention output requests, and unknown processor keys explicitly.

## Numerical and performance requirements

Keep original model revision 914f7f89142e33e77833254d9c9b90c3cef7303b,
FP32/BF16 only, local/global bidirectional masks, scaled token embeddings,
per-layer inputs, modality projection and validity masks, and masked pooling.
Retain the CA25 selector formulation as an explicit audio layout option; do
not silently apply a CA25 experiment to every target. The native forward path
always uses RBLN runtimes for all encoder stages.

The adapter requires Transformers 5.20.1 or newer. Keep unrelated upstream
dependency pins intact; provide an isolated uv environment recipe with explicit
overrides for this model. CA25 SDK 0.11 is available; CR13 needs SDK >=0.12.
SDK/version differences and unsupported native device profiling remain visible.

## Verification and deliverables

1. CPU tiny-model numerical tests for all graphs and modalities, validation,
   named runtime routing, and tensor-only host-state save/reload.
2. Native config serialization and auto-class discovery tests.
3. On CA25, convert the existing validated bundle, run native public API
   inference and save/reload, and compare full-weight outputs to the official
   reference with cosine >=0.99 for every fixture.
4. Provide a benchmark/example that separates CPU prepare, prepared encode,
   and E2E, with an optional Perfetto-compatible host trace in a separate pass.
5. Commit and push the feature branch to the requested fork. A PR to upstream
   is a separate action; the user has requested a fork and implementation.

Builder choices: one native class owns named encoder graphs rather than causal
LM submodules; existing binaries can be imported without expensive recompilation;
physical CR13 revalidation waits for working SSH rather than claiming a CA25
result establishes CR13 support.

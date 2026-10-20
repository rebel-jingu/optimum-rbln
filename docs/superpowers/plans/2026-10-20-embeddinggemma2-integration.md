# EmbeddingGemma 2 Integration Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Provide native optimum-rbln text/image/audio embedding support and migrate the verified bundles into its lifecycle.

**Architecture:** Named encoder graph buckets share native configuration and runtime handling. A lightweight host model holds only token embeddings, positional helpers and configuration. Existing graph mathematics and preparation optimizations move into the adapter.

**Tech Stack:** Python, PyTorch, Transformers >=5.20.1, rebel-compiler, uv, pytest, Ruff.

**Spec:** docs/superpowers/specs/2026-10-20-embeddinggemma2-integration-design.md

## Global Constraints

- FP32/BF16 only; batch size one; dimensions 768/512/256/128.
- Preserve bidirectional text, full multimodal validity and explicit capacity errors.
- No full encoder CPU weights/checkpoint download on compiled reload.
- Native runtimes must use the requested NPU group and inspected input signature.
- Historical benchmarks retain their original implementation/version attribution.

## Review Focus

- Processor keys or flags that would otherwise be silently ignored.
- Constant-folded compiler inputs and stable graph-to-runtime ordering.
- CPU positional buffers after BF16 host-state reload.
- Imported artifact provenance and inability to mutate original bundles.
- Invalid device groups and over-capacity mixed requests before NPU work.

### Task 1: Configuration, graph and host primitives

**Files:** model package under src/optimum/rbln/transformers/models/embedding_gemma2/; tests/unit/cpu/test_embedding_gemma2.py.

**Interfaces:** Produces RBLNEmbeddingGemma2ModelConfig, graph classes/input builders, lightweight host-state builder, and bounded metadata cache.

- [ ] Write tests for config validation/roundtrip, numerical graphs and host-state equivalence.
- [ ] Run them and observe missing implementation failures.
- [ ] Move verified mathematics/cache and implement native config and tensor-only host state.
- [ ] Run tests and Ruff; commit.

### Task 2: Native lifecycle and existing bundle conversion

**Files:** modeling_embedding_gemma2.py, embedding_gemma2_runtime.py, tests/unit/cpu/test_embedding_gemma2.py.

**Interfaces:** Consumes Task 1. Produces native compile/from_model/forward, named runtime routing, legacy conversion and compiled save/reload.

- [ ] Write tests for native discovery, runtime dispatch, multimodal outputs, capacity errors and reload without full model weights.
- [ ] Observe failures, then integrate through RBLNModel hooks and preserve inspected input order.
- [ ] Add explicit legacy conversion with hash/version/provenance validation.
- [ ] Verify tiny-model numerical tests and applicable existing CPU tests; commit.

### Task 3: uv recipe, physical validation and profiling

**Files:** examples/multimodal-embedding/, adapter documentation and measured validation summaries.

**Interfaces:** Consumes native public API from Task 2; produces reproducible prepare/encode/E2E benchmark and optional host trace.

- [ ] Provide isolated uv setup and text/image/audio/mixed example.
- [ ] Convert existing CA25 selector bundle and run actual native API inference, official-reference numerical checks and save/reload.
- [ ] Run warmed timing separately from trace capture; record versions, inputs and measurement boundaries.
- [ ] Review final diff, run relevant tests and Ruff, commit and push the feature branch.

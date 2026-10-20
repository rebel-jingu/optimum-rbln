"""Embedding-only API and native artifact lifecycle, without an attached NPU."""

from tempfile import TemporaryDirectory
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F


pytest.importorskip("transformers", minversion="5.20.1")
from transformers import EmbeddingGemma2Config, EmbeddingGemma2Model, EmbeddingGemma2TextConfig  # noqa: E402
from transformers.models.gemma4.configuration_gemma4 import Gemma4AudioConfig, Gemma4VisionConfig  # noqa: E402


@pytest.fixture
def tiny_model():
    torch.manual_seed(42)
    text = EmbeddingGemma2TextConfig(
        vocab_size=100,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        global_head_dim=16,
        num_global_key_value_heads=1,
        hidden_size_per_layer_input=16,
        layer_types=["sliding_attention", "full_attention"],
        sliding_window=4,
    )
    vision = Gemma4VisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=8,
        patch_size=2,
        position_embedding_size=64,
    )
    audio = Gemma4AudioConfig(
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        subsampling_conv_channels=[16, 4],
        output_proj_dims=48,
    )
    config = EmbeddingGemma2Config(
        text_config=text,
        vision_config=vision,
        audio_config=audio,
        image_token_id=90,
        audio_token_id=91,
        video_token_id=92,
    )
    model = EmbeddingGemma2Model(config).eval()
    model.set_attn_implementation("eager")
    return model


@pytest.mark.parametrize(
    "kwargs",
    [
        {"batch_size": 2},
        {"dtype": "float16"},
        {"text_buckets": []},
        {"text_buckets": [0]},
        {"text_buckets": [8193]},
        {"audio_buckets": [3]},
        {"image_soft_tokens": 71},
        {"audio_layout": "unknown"},
        {"num_devices": 0},
    ],
)
def test_config_rejects_unsupported_settings(kwargs):
    from optimum.rbln import RBLNEmbeddingGemma2ModelConfig

    with pytest.raises(ValueError):
        RBLNEmbeddingGemma2ModelConfig(**kwargs)


def test_config_roundtrip_and_auto_discovery(tmp_path):
    from optimum.rbln import (
        RBLNAutoConfig,
        RBLNAutoModel,
        RBLNCompileConfig,
        RBLNEmbeddingGemma2Model,
        RBLNEmbeddingGemma2ModelConfig,
    )

    cfg = RBLNEmbeddingGemma2ModelConfig(npu="RBLN-CA25", num_devices=4, text_buckets=[16, 32])
    cfg.set_compile_cfgs([RBLNCompileConfig(compiled_model_name="text_16", input_info=[("x", [1], "float32")])])
    cfg.save(tmp_path)
    loaded = RBLNAutoConfig.from_pretrained(tmp_path)
    assert loaded.text_buckets == [16, 32]
    assert loaded.num_devices == 4
    assert loaded.npu == "RBLN-CA25"
    (tmp_path / "text_16.rbln").write_bytes(b"discovery only")
    assert RBLNAutoModel.get_rbln_cls(tmp_path, export=False) is RBLNEmbeddingGemma2Model


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_host_state_reconstructs_metadata_without_encoder_weights(tiny_model, tmp_path, dtype):
    from optimum.rbln.transformers.models.embedding_gemma2.embedding_gemma2_architecture import (
        audio_inputs,
        text_inputs,
        vision_inputs,
    )
    from optimum.rbln.transformers.models.embedding_gemma2.embedding_gemma2_host import (
        load_host_model,
        save_host_model,
    )

    model = tiny_model.to(dtype)
    save_host_model(model, tmp_path / "host_state.pt")
    host = load_host_model(model.config, tmp_path / "host_state.pt", dtype)
    assert not any("layers" in name for name, _ in host.named_parameters())
    ids = torch.tensor([[2, 3, 4, 0]])
    embeds = model.get_input_embeddings()(ids)
    torch.testing.assert_close(host.get_input_embeddings()(ids), embeds, atol=0, rtol=0)
    positions = torch.tensor([[(x, y) for y in range(6) for x in range(6)]])
    args = [
        (text_inputs, (embeds, ids.ne(0))),
        (vision_inputs, (torch.randn(1, 36, 12), positions)),
        (audio_inputs, (torch.randn(1, 60, 16), torch.ones(1, 60, dtype=torch.bool))),
    ]
    for builder, inputs in args:
        expected, actual = builder(model, *inputs), builder(host, *inputs)
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)


def make_native_model(tiny_model):
    from optimum.rbln import RBLNEmbeddingGemma2Model, RBLNEmbeddingGemma2ModelConfig
    from optimum.rbln.transformers.models.embedding_gemma2.embedding_gemma2_architecture import (
        AudioGraph,
        TextGraph,
        VisionGraph,
    )

    cfg = RBLNEmbeddingGemma2ModelConfig(text_buckets=[16], audio_buckets=[60], image_soft_tokens=70)
    RBLNEmbeddingGemma2Model.update_rbln_config([], tiny_model, tiny_model.config, cfg)
    graphs = {
        "text": TextGraph(tiny_model, normalize_output=False),
        "vision": VisionGraph(tiny_model),
        "audio": AudioGraph(tiny_model),
    }
    runtimes = [Mock(side_effect=graphs[c.compiled_model_name.split("_")[0]]) for c in cfg.compile_cfgs]
    directory = TemporaryDirectory()
    from pathlib import Path

    RBLNEmbeddingGemma2Model.save_torch_artifacts(tiny_model, Path(directory.name), "", cfg)
    cfg.freeze()
    return RBLNEmbeddingGemma2Model(runtimes, tiny_model.config, cfg, model_save_dir=directory), runtimes


@pytest.mark.parametrize("dimensions", [128, 256, 512, 768])
def test_native_forward_returns_normalized_embeddings(tiny_model, dimensions):
    model, _ = make_native_model(tiny_model)
    ids = torch.tensor([[2, 3, 4]])
    with torch.no_grad():
        expected = tiny_model(input_ids=ids, attention_mask=torch.ones_like(ids)).last_hidden_state.float().mean(1)
        expected = torch.nn.functional.normalize(expected[:, :dimensions], dim=-1)
    actual = model(input_ids=ids, dimensions=dimensions)
    torch.testing.assert_close(actual.embeddings, expected, atol=1e-5, rtol=1e-5)
    assert actual.embeddings.dtype == torch.float32
    torch.testing.assert_close(model(input_ids=ids, dimensions=dimensions, return_dict=False)[0], expected)


@pytest.mark.parametrize(
    "inputs",
    [
        {"pixel_values": torch.zeros(1, 36, 12)},
        {"image_position_ids": torch.zeros(1, 36, 2, dtype=torch.long)},
        {"input_features": torch.zeros(1, 60, 16)},
        {"input_features_mask": torch.ones(1, 60, dtype=torch.bool)},
        {"input_features": torch.zeros(1, 4, 16), "input_features_mask": torch.tensor([[1, 0, 1, 0]])},
        {"attention_mask": torch.tensor([[1, 2]])},
        {"attention_mask": torch.ones(1, 3)},
        {"attention_mask": [[1, 1]]},
        {"input_ids": torch.tensor([[-1, 2]])},
        {"output_hidden_states": True},
        {"position_ids": torch.tensor([[0, 1]])},
        {"pixel_values_videos": torch.zeros(1, 1)},
        {"input_ids": torch.ones(2, 2, dtype=torch.long)},
        {"input_ids": torch.ones(1, 17, dtype=torch.long)},
    ],
)
def test_invalid_inputs_fail_before_runtime(tiny_model, inputs):
    model, runtimes = make_native_model(tiny_model)
    with pytest.raises((ValueError, TypeError)):
        model(**{"input_ids": torch.tensor([[2, 3]]), **inputs})
    for runtime in runtimes:
        runtime.assert_not_called()


@pytest.mark.parametrize("device,expected", [(0, [0, 1, 2, 3]), ([4, 5, 6, 7], [4, 5, 6, 7])])
def test_native_runtime_routes_every_graph_to_group(monkeypatch, device, expected):
    import optimum.rbln.transformers.models.embedding_gemma2.modeling_embedding_gemma2 as module
    from optimum.rbln import RBLNCompileConfig, RBLNEmbeddingGemma2ModelConfig

    cfg = RBLNEmbeddingGemma2ModelConfig(num_devices=4, device=device)
    cfg.set_compile_cfgs([RBLNCompileConfig(compiled_model_name="text_16", input_info=[("x", [1], "float32")])])
    runtime = Mock()
    monkeypatch.setattr(module.rebel, "Runtime", runtime)
    module.RBLNEmbeddingGemma2Model._create_runtimes([object()], cfg)
    assert runtime.call_args.kwargs["device"] == expected


@pytest.mark.parametrize("group", [[0, 0, 1, 2], [0, 1], [-1, 0, 1, 2], [True, 1, 2, 3]])
def test_native_runtime_rejects_invalid_group_before_allocation(monkeypatch, group):
    import optimum.rbln.transformers.models.embedding_gemma2.modeling_embedding_gemma2 as module
    from optimum.rbln import RBLNCompileConfig, RBLNEmbeddingGemma2ModelConfig

    cfg = RBLNEmbeddingGemma2ModelConfig(num_devices=4, device=group)
    cfg.set_compile_cfgs([RBLNCompileConfig(compiled_model_name="text_16", input_info=[("x", [1], "float32")])])
    runtime = Mock()
    monkeypatch.setattr(module.rebel, "Runtime", runtime)
    with pytest.raises(ValueError):
        module.RBLNEmbeddingGemma2Model._create_runtimes([object()], cfg)
    runtime.assert_not_called()


def test_compiled_reload_never_constructs_full_reference_model(tiny_model, tmp_path, monkeypatch):
    from optimum.rbln import RBLNAutoModel, RBLNEmbeddingGemma2Model

    model, runtimes = make_native_model(tiny_model)
    source = model.model_save_dir
    model.rbln_config.save(source)
    for cfg in model.rbln_config.compile_cfgs:
        (source / (cfg.compiled_model_name + ".rbln")).write_bytes(b"test compiler object")
    model.save_pretrained(tmp_path / "saved")
    monkeypatch.setattr(EmbeddingGemma2Model, "__init__", Mock(side_effect=AssertionError("full model constructed")))
    monkeypatch.setattr(EmbeddingGemma2Model, "from_pretrained", Mock(side_effect=AssertionError("checkpoint loaded")))
    monkeypatch.setattr(
        RBLNEmbeddingGemma2Model,
        "_load_compiled_models",
        classmethod(lambda cls, path, names: dict.fromkeys(names, object())),
    )
    monkeypatch.setattr(RBLNEmbeddingGemma2Model, "_create_runtimes", classmethod(lambda cls, models, cfg: runtimes))
    loaded = RBLNAutoModel.from_pretrained(tmp_path / "saved", export=False)
    inputs = {"input_ids": torch.tensor([[2, 3, 4]])}
    torch.testing.assert_close(loaded(**inputs).embeddings, model(**inputs).embeddings)


def test_selector_audio_graph_matches_row_graph(tiny_model):
    from optimum.rbln.transformers.models.embedding_gemma2.embedding_gemma2_architecture import (
        AudioGraph,
        audio_inputs,
        configure_audio_layout,
    )

    row = AudioGraph(tiny_model)
    selector = AudioGraph(tiny_model)
    configure_audio_layout(selector, "selector")
    inputs = audio_inputs(tiny_model, torch.randn(1, 60, 16), torch.arange(60)[None] < 41)
    with torch.no_grad():
        expected, actual = row(**inputs), selector(**inputs)
    for a, b in zip(actual, expected, strict=False):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_invalid_audio_width_precedes_vision_execution(tiny_model):
    model, runtimes = make_native_model(tiny_model)
    positions = torch.full((1, 630, 2), -1, dtype=torch.long)
    positions[0, :36] = torch.tensor([(x, y) for y in range(6) for x in range(6)])
    ids = torch.tensor([[2] + [90] * 4 + [91] * 3])
    with pytest.raises(ValueError, match="feature"):
        model(
            input_ids=ids,
            pixel_values=torch.zeros(1, 630, 12),
            image_position_ids=positions,
            input_features=torch.zeros(1, 12, 17),
            input_features_mask=torch.ones(1, 12, dtype=torch.bool),
        )
    for runtime in runtimes:
        runtime.assert_not_called()


def test_legacy_import_rejects_tampered_binary_before_writing(tiny_model, tmp_path):
    import json

    from optimum.rbln import RBLNEmbeddingGemma2Model
    from optimum.rbln.transformers.models.embedding_gemma2.embedding_gemma2_utils import MODEL_REVISION

    source = tmp_path / "legacy"
    source.mkdir()
    (source / "text.rbln").write_bytes(b"bad binary")
    (source / "manifest.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "model_revision": MODEL_REVISION,
                "dtype": "float32",
                "image_soft_tokens": 70,
                "graphs": {
                    "text:16": {
                        "file": "text.rbln",
                        "sha256": "0" * 64,
                        "inputs": [{"name": "x", "shape": [1], "dtype": "float32"}],
                    }
                },
            }
        )
    )
    with pytest.raises(ValueError, match="hash"):
        RBLNEmbeddingGemma2Model.from_legacy_bundle(source, tiny_model, tmp_path / "native")
    assert not (tmp_path / "native").exists()


def test_native_mixed_embeddings_match_reference(tiny_model):
    model, _ = make_native_model(tiny_model)
    # The tiny audio convolution uses 16 features; production uses 128.
    audio_metadata = model.engine.manifest["graphs"]["audio:60"]["inputs"]
    next(item for item in audio_metadata if item["name"] == "input_features")["shape"][-1] = 16
    for cfg in model.rbln_config.compile_cfgs:
        if cfg.compiled_model_name.startswith("audio_"):
            cfg.input_info = [
                (name, [1, 60, 16] if name == "input_features" else shape, dtype)
                for name, shape, dtype in cfg.input_info
            ]
    # One real 3x3 cell, with the rest of the processor's 70-cell budget padded.
    positions = torch.full((1, 630, 2), -1, dtype=torch.long)
    positions[0, :9] = torch.tensor([(x, y) for y in range(3) for x in range(3)])
    inputs = {
        "input_ids": torch.tensor([[2, 90, 91, 91, 91]]),
        "pixel_values": torch.randn(1, 630, 12),
        "image_position_ids": positions,
        "input_features": torch.randn(1, 60, 16),
        "input_features_mask": torch.arange(60)[None] < 12,
    }
    with torch.no_grad():
        expected = tiny_model(**inputs).last_hidden_state.float().mean(1)
        expected = torch.nn.functional.normalize(expected, dim=-1)
    actual = model(**inputs).embeddings
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_prepare_and_encode_preserve_native_output(tiny_model):
    model, _ = make_native_model(tiny_model)
    inputs = {"input_ids": torch.tensor([[2, 3]])}
    processor = Mock(return_value=inputs)
    assert model.prepare({"text": "hello"}, processor=processor) is inputs
    processor.assert_called_once_with(text="hello", return_tensors="pt", truncation=False)
    torch.testing.assert_close(model.encode({"text": "hello"}, processor=processor), model(**inputs).embeddings)


@pytest.mark.parametrize(
    "mode,extra",
    [
        ("compile", ["--npu", "RBLN-CA25", "--output", "/tmp/compiled"]),
        ("import", ["--bundle", "/tmp/legacy", "--output", "/tmp/native"]),
        ("run", ["--model", "/tmp/native", "--requests", "/tmp/requests.json"]),
        ("benchmark", ["--model", "/tmp/native", "--requests", "/tmp/requests.json", "--output", "/tmp/results"]),
    ],
)
def test_example_parser_accepts_multidevice_subcommands(mode, extra):
    import runpy
    from pathlib import Path

    script = Path(__file__).parents[3] / "examples/multimodal-embedding/run_embedding_gemma2.py"
    namespace = runpy.run_path(str(script))
    args = namespace["build_parser"]().parse_args([mode, "--devices", "0", "1", "2", "3", *extra])
    assert args.devices == [0, 1, 2, 3]


@pytest.mark.parametrize("mask_dtype", [torch.bool, torch.float32, torch.int32])
def test_binary_text_masks_are_canonicalized_before_runtime(tiny_model, mask_dtype):
    model, _ = make_native_model(tiny_model)
    ids = torch.tensor([[2, 3, 4]], dtype=torch.int32)
    expected = model(input_ids=ids.long(), attention_mask=torch.ones(1, 3, dtype=torch.long)).embeddings
    actual = model(input_ids=ids, attention_mask=torch.ones(1, 3, dtype=mask_dtype)).embeddings
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(model(input_ids=ids).embeddings, expected)


@pytest.mark.parametrize("kind", ["empty", "half_padding", "duplicate", "incomplete", "outside"])
def test_invalid_image_coordinates_fail_before_runtime(tiny_model, kind):
    model, runtimes = make_native_model(tiny_model)
    positions = torch.full((1, 630, 2), -1, dtype=torch.long)
    if kind != "empty":
        positions[0, :9] = torch.tensor([(x, y) for y in range(3) for x in range(3)])
    if kind == "half_padding":
        positions[0, 0] = torch.tensor([-1, 0])
    elif kind == "duplicate":
        positions[0, 0] = positions[0, 1]
    elif kind == "incomplete":
        positions[0, 0] = torch.tensor([3, 0])
    elif kind == "outside":
        positions[0, :9] += tiny_model.config.vision_config.position_embedding_size
    image_tokens = 0 if kind in ("empty", "half_padding") else 1
    ids = torch.tensor([[2] + [90] * image_tokens])
    with pytest.raises(ValueError):
        model(input_ids=ids, pixel_values=torch.zeros(1, 630, 12), image_position_ids=positions)
    for runtime in runtimes:
        runtime.assert_not_called()


def test_legacy_import_with_explicit_revision_preserves_originals(tiny_model, tmp_path, monkeypatch):
    import json

    from optimum.rbln import RBLNEmbeddingGemma2ModelConfig
    from optimum.rbln.transformers.models.embedding_gemma2 import embedding_gemma2_conversion as module
    from optimum.rbln.transformers.models.embedding_gemma2.embedding_gemma2_artifacts import file_sha256
    from optimum.rbln.transformers.models.embedding_gemma2.embedding_gemma2_utils import (
        MODEL_REVISION,
        package_versions,
    )

    model, _ = make_native_model(tiny_model)
    source = tmp_path / "legacy"
    source.mkdir()
    graphs, inspected = {}, {}
    for cfg in model.rbln_config.compile_cfgs:
        stage, capacity = cfg.compiled_model_name.split("_")
        filename = cfg.compiled_model_name + ".rbln"
        path = source / filename
        path.write_bytes(filename.encode())
        inputs = [
            {"name": name, "shape": list(shape), "dtype": str(dtype).removeprefix("torch.")}
            for name, shape, dtype in cfg.input_info
        ]
        graphs[f"{stage}:{capacity}"] = {"file": filename, "sha256": file_sha256(path), "inputs": inputs}
        inspected[filename] = {"npu": "RBLN-CA25", "num_devices": 4, "inputs": inputs}
    (source / "manifest.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "model_revision": MODEL_REVISION,
                "dtype": "float32",
                "image_soft_tokens": 70,
                "npu": "RBLN-CA25",
                "num_devices": 4,
                "versions": package_versions(),
                "graphs": graphs,
            }
        )
    )
    before = {p.name: file_sha256(p) for p in source.iterdir()}
    monkeypatch.setattr(module.rebel.RBLNCompiledModel, "inspect", lambda p: inspected[p.name])
    destination = module.import_legacy_bundle(source, tiny_model, tmp_path / "native", source_revision=MODEL_REVISION)
    loaded = RBLNEmbeddingGemma2ModelConfig.from_pretrained(destination)
    assert loaded.npu == "RBLN-CA25" and loaded.num_devices == 4
    assert before == {p.name: file_sha256(p) for p in source.iterdir()}
    for graph in graphs.values():
        assert file_sha256(destination / graph["file"]) == graph["sha256"]


@pytest.mark.parametrize("invalid", [False, "shape", "hardware"])
def test_compile_preflights_every_mapped_device_group(tiny_model, monkeypatch, invalid):
    import optimum.rbln.transformers.models.embedding_gemma2.modeling_embedding_gemma2 as module
    from optimum.rbln import RBLNEmbeddingGemma2Model, RBLNEmbeddingGemma2ModelConfig

    cfg = RBLNEmbeddingGemma2ModelConfig(num_devices=4, npu="RBLN-CA25", text_buckets=[16], audio_buckets=[60])
    RBLNEmbeddingGemma2Model.update_rbln_config([], tiny_model, tiny_model.config, cfg)
    cfg.device_map = {c.compiled_model_name: [4, 5, 6, 7] for c in cfg.compile_cfgs}
    if invalid == "shape":
        cfg.device_map[cfg.compile_cfgs[-1].compiled_model_name] = [4, 4, 6, 7]
    if invalid == "hardware":
        cfg.device_map[cfg.compile_cfgs[-1].compiled_model_name] = [8, 9, 10, 11]
    monkeypatch.setattr(
        module,
        "tp_and_devices_are_ok",
        lambda **kwargs: "Wrong target NPU" if kwargs["device"][0] == 8 else None,
        raising=False,
    )
    names = [c.compiled_model_name for c in cfg.compile_cfgs]
    monkeypatch.setattr(
        module,
        "graph_examples",
        lambda model, config: iter((name, torch.nn.Identity(), {"x": torch.ones(1)}) for name in names),
    )
    compiled = Mock()
    compile_call = Mock(return_value=compiled)
    monkeypatch.setattr(RBLNEmbeddingGemma2Model, "compile", compile_call)
    monkeypatch.setattr(
        module.rebel.RBLNCompiledModel,
        "inspect",
        lambda path: {"inputs": [{"name": "x", "shape": [1], "dtype": "float32"}]},
    )
    if invalid:
        with pytest.raises(ValueError):
            RBLNEmbeddingGemma2Model.get_compiled_model(tiny_model, cfg)
        compile_call.assert_not_called()
    else:
        RBLNEmbeddingGemma2Model.get_compiled_model(tiny_model, cfg)
        assert all(call.args[3] == [4, 5, 6, 7] for call in compile_call.call_args_list)
        assert all(c.input_info == [("x", [1], "float32")] for c in cfg.compile_cfgs)


def test_static_l2_normalization_matches_reference_at_small_norms():
    from optimum.rbln.transformers.models.embedding_gemma2.embedding_gemma2_architecture import static_normalize

    vectors = torch.tensor([[3.0, 4.0, 0.0], [1e-20, 0.0, 0.0], [0.0, 0.0, 0.0]])
    torch.testing.assert_close(static_normalize(vectors), F.normalize(vectors, dim=-1))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_clipped_linear_preserves_tensor_bound_values(dtype):
    from transformers.models.gemma4.configuration_gemma4 import Gemma4AudioConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4ClippableLinear

    from optimum.rbln.transformers.models.embedding_gemma2.embedding_gemma2_architecture import StaticClippableLinear

    original = Gemma4ClippableLinear(Gemma4AudioConfig(use_clipped_linears=True), 8, 8).to(dtype)
    original.input_min.fill_(-26.625)
    original.input_max.fill_(26.5)
    original.output_min.fill_(-101.5)
    original.output_max.fill_(100.5)
    states = torch.linspace(-200, 200, 80).reshape(1, 10, 8).to(dtype)
    actual = StaticClippableLinear(original)(states)
    torch.testing.assert_close(actual, original(states), atol=0, rtol=0)


@pytest.mark.parametrize("length", [1, 17, 32, 65])
def test_static_audio_windows_match_official_without_mutating_reference(tiny_model, length):
    from optimum.rbln.transformers.models.embedding_gemma2.embedding_gemma2_architecture import AudioGraph

    original = tiny_model.audio_tower.layers[0].self_attn
    graph = AudioGraph(tiny_model)
    attention = graph.layers[0].self_attn
    assert attention is not original
    assert next(attention.q_proj.parameters()) is next(original.q_proj.parameters())
    assert tiny_model.audio_tower.layers[0].self_attn is original
    states = torch.randn(1, length, original.num_heads, original.head_dim)
    torch.testing.assert_close(attention._extract_block_context(states), original._extract_block_context(states))


def test_text_graph_preserves_bidirectional_projection_pooling(tiny_model):
    from optimum.rbln.transformers.models.embedding_gemma2.embedding_gemma2_architecture import TextGraph, text_inputs

    ids = torch.tensor([[2, 5, 7, 8, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 1, 0, 0]])
    with torch.no_grad():
        states = tiny_model.language_model(input_ids=ids, attention_mask=mask).last_hidden_state
        expected = F.normalize(states[:, :4].float().mean(1), dim=-1)
        embeds = tiny_model.get_input_embeddings()(ids)
        actual = TextGraph(tiny_model)(**text_inputs(tiny_model, embeds, mask))
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("width,height", [(6, 6), (3, 6)])
def test_vision_pooling_matches_official_for_aspect_ratios(tiny_model, width, height):
    from optimum.rbln.transformers.models.embedding_gemma2.embedding_gemma2_architecture import (
        VisionGraph,
        vision_inputs,
    )

    positions = torch.full((1, 36, 2), -1, dtype=torch.long)
    positions[0, : width * height] = torch.tensor([(x, y) for y in range(height) for x in range(width)])
    pixels = torch.randn(1, 36, 12)
    with torch.no_grad():
        expected = tiny_model.get_image_features(pixels, positions).pooler_output
        expected = torch.cat(expected, dim=0)
        tokens, valid = VisionGraph(tiny_model)(**vision_inputs(tiny_model, pixels, positions))
    torch.testing.assert_close(tokens[valid], expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("capacity", [60, 96])
def test_audio_projection_and_validity_match_official(tiny_model, capacity):
    from optimum.rbln.transformers.models.embedding_gemma2.embedding_gemma2_architecture import (
        AudioGraph,
        audio_inputs,
    )

    features = torch.randn(1, capacity, 16)
    mask = torch.arange(capacity)[None, :] < 41
    with torch.no_grad():
        expected = tiny_model.get_audio_features(features, mask)
        tokens, valid = AudioGraph(tiny_model)(**audio_inputs(tiny_model, features, mask))
    assert torch.equal(valid, expected.attention_mask)
    torch.testing.assert_close(tokens[valid], expected.pooler_output[valid], atol=1e-5, rtol=1e-5)


def test_static_text_graph_exports_without_python_data_dependencies(tiny_model):
    from optimum.rbln.transformers.models.embedding_gemma2.embedding_gemma2_architecture import TextGraph, text_inputs

    ids = torch.tensor([[2, 5, 7, 0]])
    with torch.no_grad():
        inputs = text_inputs(tiny_model, tiny_model.get_input_embeddings()(ids), ids.ne(0))
        graph = TextGraph(tiny_model)
        exported = torch.export.export(graph, tuple(inputs.values())).module()
        torch.testing.assert_close(exported(*inputs.values()), graph(**inputs))


@pytest.mark.parametrize("stage", ["vision", "audio"])
def test_static_media_graph_exports(tiny_model, stage):
    from optimum.rbln.transformers.models.embedding_gemma2.embedding_gemma2_architecture import (
        AudioGraph,
        VisionGraph,
        audio_inputs,
        vision_inputs,
    )

    with torch.no_grad():
        if stage == "vision":
            positions = torch.tensor([[(x, y) for y in range(6) for x in range(6)]])
            inputs = vision_inputs(tiny_model, torch.randn(1, 36, 12), positions)
            graph = VisionGraph(tiny_model)
        else:
            inputs = audio_inputs(tiny_model, torch.randn(1, 60, 16), torch.ones(1, 60, dtype=torch.bool))
            graph = AudioGraph(tiny_model)
        exported = torch.export.export(graph, tuple(inputs.values())).module()
        torch.testing.assert_close(exported(*inputs.values()), graph(**inputs))
        if stage == "audio":
            assert not any("aten.unfold" in str(node.target) for node in exported.graph.nodes)


@pytest.mark.parametrize("stage", ["text", "vision", "audio"])
def test_all_valid_graph_signatures_contain_only_tensors(tiny_model, stage):
    from optimum.rbln.transformers.models.embedding_gemma2.embedding_gemma2_architecture import (
        audio_inputs,
        text_inputs,
        vision_inputs,
    )

    if stage == "text":
        inputs = text_inputs(tiny_model, torch.ones(1, 6, 32), torch.ones(1, 6, dtype=torch.long))
    elif stage == "vision":
        positions = torch.tensor([[(x, y) for y in range(6) for x in range(6)]])
        inputs = vision_inputs(tiny_model, torch.ones(1, 36, 12), positions)
    else:
        inputs = audio_inputs(tiny_model, torch.ones(1, 60, 16), torch.ones(1, 60, dtype=torch.bool))
    assert all(isinstance(value, torch.Tensor) for value in inputs.values())


def test_text_graph_can_pool_without_redundant_device_normalization(tiny_model):
    from optimum.rbln.transformers.models.embedding_gemma2.embedding_gemma2_architecture import TextGraph, text_inputs

    ids = torch.tensor([[2, 5, 7, 8, 0, 0]])
    mask = ids.ne(0)
    with torch.no_grad():
        states = tiny_model.language_model(input_ids=ids, attention_mask=mask).last_hidden_state
        expected = states[:, :4].float().mean(1)
        embeds = tiny_model.get_input_embeddings()(ids)
        actual = TextGraph(tiny_model, normalize_output=False)(**text_inputs(tiny_model, embeds, mask))
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("valid_length", [18, 36])
def test_vision_sdpa_padding_preserves_masked_attention(tiny_model, valid_length):
    from optimum.rbln.transformers.models.embedding_gemma2.embedding_gemma2_architecture import (
        AlignedVisionAttention,
        static_modules,
        vision_inputs,
    )

    original = tiny_model.vision_tower.encoder.layers[0].self_attn
    positions = torch.full((1, 36, 2), -1, dtype=torch.long)
    positions[0, :valid_length] = torch.tensor([(x, y) for y in range(6) for x in range(6)])[:valid_length]
    inputs = vision_inputs(tiny_model, torch.randn(1, 36, 12), positions)
    states = torch.randn(1, 36, 32)
    kwargs = {
        "position_embeddings": (inputs["cos"], inputs["sin"]),
        "attention_mask": inputs["attention_mask"],
        "position_ids": positions,
    }
    with torch.no_grad():
        expected = original(states, **kwargs)[0]
        attention = static_modules(original)
        assert isinstance(attention, AlignedVisionAttention)
        actual = attention(states, **kwargs)[0]
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    assert tiny_model.vision_tower.encoder.layers[0].self_attn is original
    torch.testing.assert_close(F.normalize(actual, dim=-1), F.normalize(expected, dim=-1))


def test_hit_reuses_metadata_and_does_not_rebuild():
    from optimum.rbln.transformers.models.embedding_gemma2.embedding_gemma2_cache import BoundedTensorCache

    cache = BoundedTensorCache()
    metadata = {"mask": torch.ones(2, 4, dtype=torch.bool)}
    assert cache.get_or_create("text", lambda: metadata) is metadata

    def should_not_build():
        pytest.fail("a hit must not rebuild metadata")

    assert cache.get_or_create("text", should_not_build) is metadata
    assert cache.info() == {"hits": 1, "misses": 1, "entries": 1, "bytes": 8}


def test_entry_limit_evicts_the_least_recently_used_key():
    from optimum.rbln.transformers.models.embedding_gemma2.embedding_gemma2_cache import BoundedTensorCache

    cache = BoundedTensorCache(max_entries=2)
    first = cache.get_or_create("first", lambda: {"mask": torch.ones(1)})
    cache.get_or_create("second", lambda: {"mask": torch.ones(1)})
    assert cache.get_or_create("first", lambda: {}) is first
    cache.get_or_create("third", lambda: {"mask": torch.ones(1)})
    rebuilt = {"mask": torch.zeros(1)}
    assert cache.get_or_create("second", lambda: rebuilt) is rebuilt
    assert cache.info() == {"hits": 1, "misses": 4, "entries": 2, "bytes": 8}


def test_byte_limit_evicts_enough_entries_and_counts_tensor_bytes():
    from optimum.rbln.transformers.models.embedding_gemma2.embedding_gemma2_cache import BoundedTensorCache

    cache = BoundedTensorCache(max_bytes=10)
    cache.get_or_create("first", lambda: {"mask": torch.ones(3, dtype=torch.bool)})
    cache.get_or_create("second", lambda: {"mask": torch.ones(4, dtype=torch.bool)})
    cache.get_or_create("third", lambda: {"positions": torch.ones(1, dtype=torch.int64)})
    assert cache.info() == {"hits": 0, "misses": 3, "entries": 1, "bytes": 8}


def test_oversized_entry_is_returned_without_evicting_existing_metadata():
    from optimum.rbln.transformers.models.embedding_gemma2.embedding_gemma2_cache import BoundedTensorCache

    cache = BoundedTensorCache(max_bytes=8)
    small = cache.get_or_create("small", lambda: {"mask": torch.ones(4, dtype=torch.bool)})
    oversized = {"mask": torch.ones(9, dtype=torch.bool)}
    for _ in range(2):
        assert cache.get_or_create("large", lambda: oversized) is oversized
    assert cache.get_or_create("small", lambda: {}) is small
    assert cache.info() == {"hits": 1, "misses": 3, "entries": 1, "bytes": 4}


def test_small_view_cannot_retain_storage_larger_than_the_byte_budget():
    from optimum.rbln.transformers.models.embedding_gemma2.embedding_gemma2_cache import BoundedTensorCache

    cache = BoundedTensorCache(max_bytes=8)
    metadata = {"mask": torch.ones(100, dtype=torch.bool)[:1]}
    assert cache.get_or_create("view", lambda: metadata) is metadata
    assert cache.info() == {"hits": 0, "misses": 1, "entries": 0, "bytes": 0}


@pytest.mark.parametrize("limits", [{"max_bytes": 0}, {"max_entries": 0}])
def test_zero_limit_disables_storage_but_still_builds_and_counts_misses(limits):
    from optimum.rbln.transformers.models.embedding_gemma2.embedding_gemma2_cache import BoundedTensorCache

    cache = BoundedTensorCache(**limits)
    for _ in range(2):
        assert cache.get_or_create("same", lambda: {}) == {}
    assert cache.info() == {"hits": 0, "misses": 2, "entries": 0, "bytes": 0}


def test_clear_releases_metadata_and_resets_statistics():
    from optimum.rbln.transformers.models.embedding_gemma2.embedding_gemma2_cache import BoundedTensorCache

    cache = BoundedTensorCache()
    cache.get_or_create("mask", lambda: {"mask": torch.ones(4)})
    cache.get_or_create("mask", lambda: {})
    cache.clear()
    assert cache.info() == {"hits": 0, "misses": 0, "entries": 0, "bytes": 0}
    cache.get_or_create("mask", lambda: {"mask": torch.ones(1)})
    assert cache.info()["misses"] == 1


@pytest.mark.parametrize("result", [[], {"mask": [True]}, {1: torch.ones(1)}])
def test_builder_must_return_a_dictionary_with_string_keys_and_tensor_values(result):
    from optimum.rbln.transformers.models.embedding_gemma2.embedding_gemma2_cache import BoundedTensorCache

    cache = BoundedTensorCache()
    with pytest.raises(TypeError, match="dict|str|Tensor"):
        cache.get_or_create("bad", lambda: result)
    assert cache.info() == {"hits": 0, "misses": 1, "entries": 0, "bytes": 0}


def test_builder_rejects_non_cpu_tensors_even_when_cache_is_disabled():
    from optimum.rbln.transformers.models.embedding_gemma2.embedding_gemma2_cache import BoundedTensorCache

    cache = BoundedTensorCache(max_bytes=0)
    with pytest.raises(ValueError, match="CPU"):
        cache.get_or_create("bad", lambda: {"mask": torch.empty(1, device="meta")})
    assert cache.info()["entries"] == 0


@pytest.mark.parametrize("limit", ["max_bytes", "max_entries"])
def test_limits_must_be_nonnegative_integers(limit):
    from optimum.rbln.transformers.models.embedding_gemma2.embedding_gemma2_cache import BoundedTensorCache

    with pytest.raises(ValueError, match=limit):
        BoundedTensorCache(**{limit: -1})
    with pytest.raises(TypeError, match=limit):
        BoundedTensorCache(**{limit: 1.5})


@pytest.mark.parametrize("dtype", [torch.bool, torch.int64, torch.bfloat16])
def test_tensor_key_uses_logical_values_and_accepts_noncontiguous_views(dtype):
    from optimum.rbln.transformers.models.embedding_gemma2.embedding_gemma2_cache import tensor_key

    tensor = torch.tensor([[0, 1, 1], [1, 0, 1]], dtype=dtype).t()
    assert not tensor.is_contiguous()
    key = tensor_key(tensor)
    assert key == tensor_key(tensor.contiguous())
    hash(key)
    changed = tensor.clone()
    changed[0, 0] = 1
    assert key != tensor_key(changed)
    assert key != tensor_key(tensor.reshape(-1))


def test_tensor_key_distinguishes_dtype_and_handles_scalar_empty_and_grad_tensors():
    from optimum.rbln.transformers.models.embedding_gemma2.embedding_gemma2_cache import tensor_key

    assert tensor_key(torch.tensor([1], dtype=torch.int64)) != tensor_key(torch.tensor([1.0]))
    assert tensor_key(torch.tensor(1.0, requires_grad=True)) == tensor_key(torch.tensor(1.0))
    assert tensor_key(torch.empty(0, dtype=torch.bfloat16)) == tensor_key(torch.empty(0, dtype=torch.bfloat16))


def test_tensor_key_rejects_non_tensor_and_non_cpu_inputs():
    from optimum.rbln.transformers.models.embedding_gemma2.embedding_gemma2_cache import tensor_key

    with pytest.raises(TypeError, match="Tensor"):
        tensor_key([True])
    with pytest.raises(ValueError, match="CPU"):
        tensor_key(torch.empty(1, device="meta"))

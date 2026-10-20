# Copyright 2026 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Native RBLN lifecycle for pooled text, image and audio embeddings."""

from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import rebel
import torch
from packaging.version import Version
from transformers.utils import ModelOutput

from ....configuration_utils import RBLNCompileConfig
from ....modeling import RBLNModel
from ....utils.runtime_utils import tp_and_devices_are_ok
from .embedding_gemma2_architecture import (
    AudioGraph,
    TextGraph,
    VisionGraph,
    audio_inputs,
    configure_audio_layout,
    text_inputs,
    vision_inputs,
)
from .embedding_gemma2_host import load_host_model, save_host_model
from .embedding_gemma2_preprocessing import ParallelMediaProcessor
from .embedding_gemma2_profiling import trace_span
from .embedding_gemma2_runtime import EmbeddingEngine


if Version(version("transformers")) < Version("5.20.1"):
    raise ImportError("EmbeddingGemma 2 requires transformers>=5.20.1; see the model-specific uv recipe")


@dataclass
class RBLNEmbeddingGemma2Output(ModelOutput):
    """CPU float32 normalized ``embeddings`` of shape ``[1, dimensions]``."""

    embeddings: torch.Tensor | None = None


def resolve_device_group(device, num_devices):
    if device is None:
        device = 0
    group = list(range(device, device + num_devices)) if type(device) is int else list(device)
    if len(group) != num_devices or len(set(group)) != len(group) or any(type(n) is not int or n < 0 for n in group):
        raise ValueError("device must select num_devices distinct nonnegative NPU IDs")
    return group[0] if num_devices == 1 else group


def graph_examples(model, cfg):
    for length in cfg.text_buckets:
        ids = torch.full((1, length), 2, dtype=torch.long)
        inputs = text_inputs(model, model.get_input_embeddings()(ids), torch.ones_like(ids))
        yield f"text_{length}", TextGraph(model, normalize_output=False), inputs
    if model.vision_tower is not None:
        tower = model.vision_tower
        k = tower.config.pooling_kernel_size
        width = 7 if cfg.image_soft_tokens == 70 else 14
        height = cfg.image_soft_tokens // width
        coords = torch.tensor([[(x, y) for y in range(height * k) for x in range(width * k)]])
        patches = cfg.image_soft_tokens * k * k
        pixels = torch.zeros(1, patches, tower.config.patch_size**2 * 3)
        yield f"vision_{patches}", VisionGraph(model), vision_inputs(model, pixels, coords)
    if model.audio_tower is not None:
        for frames in cfg.audio_buckets:
            features = torch.zeros(1, frames, 128)
            mask = torch.ones(1, frames, dtype=torch.bool)
            graph = configure_audio_layout(AudioGraph(model), cfg.audio_layout)
            yield f"audio_{frames}", graph, audio_inputs(model, features, mask)


class RBLNEmbeddingGemma2Model(RBLNModel):
    """Embedding-only EmbeddingGemma 2 adapter, including vision and audio.

    The text graph returns masked pooled projections. ``forward`` returns
    ``RBLNEmbeddingGemma2Output.embeddings`` rather than HF per-token states.
    Use ``AutoProcessor`` on CPU, then pass its tensors to this model.
    """

    _tp_support = True

    @classmethod
    def update_kwargs(cls, kwargs):
        kwargs.setdefault("attn_implementation", "eager")
        return kwargs

    @classmethod
    def from_legacy_bundle(
        cls,
        bundle_dir,
        source_model,
        model_save_dir,
        *,
        processor=None,
        device=0,
        create_runtimes=True,
        source_revision=None,
    ):
        """Import existing graphs once; source_model supplies the missing host state.

        The saved native model reloads without source_model or checkpoint access.
        Original graph hashes, compiler metadata and layout overrides are retained
        in legacy_provenance.json. No compiler is invoked during this operation.
        """
        from .embedding_gemma2_conversion import import_legacy_bundle

        directory = import_legacy_bundle(
            bundle_dir, source_model, model_save_dir, processor, source_revision=source_revision
        )
        return cls.from_pretrained(directory, export=False, rbln_device=device, rbln_create_runtimes=create_runtimes)

    @classmethod
    def _update_rbln_config(cls, preprocessors=None, model=None, model_config=None, rbln_config=None):
        if model.dtype not in (torch.float32, torch.bfloat16):
            raise ValueError("EmbeddingGemma 2 requires float32 or bfloat16")
        with torch.no_grad():
            configs = [
                RBLNCompileConfig(
                    compiled_model_name=name,
                    input_info=[(key, list(value.shape), value.dtype) for key, value in inputs.items()],
                )
                for name, _, inputs in graph_examples(model, rbln_config)
            ]
        rbln_config.set_compile_cfgs(configs)
        return rbln_config

    @classmethod
    @torch.no_grad()
    def get_compiled_model(cls, model, rbln_config):
        results = {}
        configs = {cfg.compiled_model_name: cfg for cfg in rbln_config.compile_cfgs}
        groups = {
            name: resolve_device_group(rbln_config.device_map[name], cfg.num_devices) for name, cfg in configs.items()
        }
        if rbln_config.create_runtimes:
            for name, cfg in configs.items():
                error = tp_and_devices_are_ok(num_devices=cfg.num_devices, device=groups[name], npu=cfg.npu)
                if error:
                    raise ValueError(error)
        for name, graph, inputs in graph_examples(model, rbln_config):
            compiled = cls.compile(
                graph.eval(),
                configs[name],
                rbln_config.create_runtimes,
                groups[name],
                example_inputs=list(inputs.values()),
            )
            # Constant positions can be folded away. Save only the published
            # signature, which the native runtime must subsequently receive.
            with TemporaryDirectory() as directory:
                path = Path(directory) / f"{name}.rbln"
                compiled.save(path)
                inspected = rebel.RBLNCompiledModel.inspect(path)
            configs[name].input_info = [(item["name"], item["shape"], item["dtype"]) for item in inspected["inputs"]]
            results[name] = compiled
        return results

    @classmethod
    def _create_runtimes(cls, compiled_models, rbln_config):
        if len(compiled_models) != len(rbln_config.compile_cfgs):
            raise ValueError("Compiled graph count differs from native configuration")
        groups = [
            resolve_device_group(rbln_config.device_map[cfg.compiled_model_name], cfg.num_devices)
            for cfg in rbln_config.compile_cfgs
        ]
        return [
            rebel.Runtime(
                compiled,
                tensor_type="pt",
                device=group,
                activate_profiler=rbln_config.activate_profiler,
                timeout=rbln_config.timeout,
            )
            for compiled, group in zip(compiled_models, groups, strict=False)
        ]

    @classmethod
    def save_torch_artifacts(cls, model, save_dir_path, subfolder, rbln_config):
        save_host_model(model, Path(save_dir_path) / subfolder / "host_state.pt")

    def __post_init__(self, **kwargs):
        self.host_model = load_host_model(
            self.config, self.model_save_dir / self.subfolder / "host_state.pt", self.dtype
        )
        self.engine = None
        if self.rbln_config.create_runtimes:
            self.engine = EmbeddingEngine(
                self.host_model,
                compile_cfgs=self.rbln_config.compile_cfgs,
                runtimes=self.model,
                image_soft_tokens=self.rbln_config.image_soft_tokens,
                cache_metadata=self.rbln_config.cache_metadata,
            )

    def _validate_inputs(self, inputs):
        if any(not isinstance(value, torch.Tensor) for value in inputs.values()):
            raise TypeError("All processor inputs must be tensors")
        ids, mask = inputs["input_ids"], inputs["attention_mask"]
        if (
            not isinstance(ids, torch.Tensor)
            or ids.ndim != 2
            or ids.shape[0] != 1
            or ids.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("input_ids must be one integer token sequence")
        text_ids = ids.masked_fill(
            ids.eq(self.config.image_token_id)
            | ids.eq(self.config.audio_token_id)
            | ids.eq(self.config.video_token_id),
            0,
        )
        if (text_ids < 0).any() or (text_ids >= self.config.text_config.vocab_size).any():
            raise ValueError("Text token IDs must be within the embedding vocabulary")
        if mask.shape != ids.shape or not ((mask == 0) | (mask == 1)).all():
            raise ValueError("attention_mask must be binary and match input_ids")
        for name, value in inputs.items():
            if not isinstance(value, torch.Tensor) or value.device.type != "cpu":
                raise ValueError(f"{name} must be a CPU tensor")
            if value.is_floating_point() and not torch.isfinite(value).all():
                raise ValueError(f"{name} contains nonfinite values")
        for first, second in (("pixel_values", "image_position_ids"), ("input_features", "input_features_mask")):
            if (first in inputs) != (second in inputs):
                raise ValueError(f"Supply {first} and {second} together")
        if "pixel_values" in inputs:
            pixels, positions = inputs["pixel_values"], inputs["image_position_ids"]
            if (
                pixels.ndim != 3
                or not pixels.shape[0]
                or positions.shape != (*pixels.shape[:2], 2)
                or positions.dtype not in (torch.int32, torch.int64)
            ):
                raise ValueError("Invalid pixel_values/image_position_ids shapes or position dtype")
            k = self.config.vision_config.pooling_kernel_size
            if pixels.shape[1:] != (
                self.rbln_config.image_soft_tokens * k * k,
                self.config.vision_config.patch_size**2 * 3,
            ):
                raise ValueError("Image shape differs from compiled capacity; use the configured max_soft_tokens")
            padding = positions.eq(-1)
            if (padding.any(-1) != padding.all(-1)).any() or (positions < -1).any():
                raise ValueError("Image padding coordinates must be complete (-1, -1) pairs")
            for coords, padded in zip(positions, padding.all(-1), strict=True):
                valid = coords[~padded]
                if not len(valid) or (valid >= self.config.vision_config.position_embedding_size).any():
                    raise ValueError("Each image requires a nonempty bounded coordinate grid")
                width, height = (valid.max(0).values + 1).tolist()
                if (
                    width % k
                    or height % k
                    or len(valid) != width * height
                    or len(torch.unique(valid, dim=0)) != len(valid)
                ):
                    raise ValueError("Image coordinates must contain complete, distinct pooling cells")
        if "input_features" in inputs:
            features, valid = inputs["input_features"], inputs["input_features_mask"]
            if (
                features.ndim != 3
                or not features.shape[0]
                or valid.shape != features.shape[:2]
                or not ((valid == 0) | (valid == 1)).all()
                or (valid[:, 1:] > valid[:, :-1]).any()
                or not valid.bool().any(-1).all()
            ):
                raise ValueError("Audio features require nonempty binary prefix validity masks")
            signatures = [
                item
                for cfg in self.rbln_config.compile_cfgs
                if cfg.compiled_model_name.startswith("audio_")
                for item in cfg.input_info
                if item[0] == "input_features"
            ]
            if not signatures or any(features.shape[-1] != item[1][-1] for item in signatures):
                raise ValueError("Audio feature width differs from the compiled input features")
        for name in ("pixel_values", "image_position_ids", "input_features"):
            if name in inputs:
                runtime_name = "pixel_position_ids" if name == "image_position_ids" else name
                signatures = [
                    item for cfg in self.rbln_config.compile_cfgs for item in cfg.input_info if item[0] == runtime_name
                ]
                if not signatures or any(
                    str(inputs[name].dtype).removeprefix("torch.") != str(item[2]).removeprefix("torch.")
                    for item in signatures
                ):
                    raise ValueError(f"{name} dtype differs from the compiled input")

    @torch.no_grad()
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        pixel_values=None,
        image_position_ids=None,
        input_features=None,
        input_features_mask=None,
        dimensions=768,
        return_dict=True,
        **kwargs,
    ):
        unsupported = [key for key, value in kwargs.items() if value is not None and value is not False]
        if unsupported:
            raise ValueError(f"Unsupported EmbeddingGemma 2 inputs/options: {unsupported}")
        if input_ids is None:
            raise ValueError("input_ids from AutoProcessor are required")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        for name, value in (
            ("pixel_values", pixel_values),
            ("image_position_ids", image_position_ids),
            ("input_features", input_features),
            ("input_features_mask", input_features_mask),
        ):
            if value is not None:
                inputs[name] = value
        self._validate_inputs(inputs)
        inputs["attention_mask"] = inputs["attention_mask"].to(torch.int64)
        if "input_features_mask" in inputs:
            inputs["input_features_mask"] = inputs["input_features_mask"].bool()
        if self.engine is None:
            raise RuntimeError("Load with rbln_create_runtimes=True to execute the NPU model")
        embeddings = torch.from_numpy(self.engine.encode_inputs(inputs, dimensions)).unsqueeze(0)
        return RBLNEmbeddingGemma2Output(embeddings=embeddings) if return_dict else (embeddings,)

    def close(self):
        if self.engine is not None:
            self.engine.close()

    def prepare(self, request, *, processor=None):
        """CPU decode/processing for one raw request, including optional media overlap."""
        if self.engine is None:
            raise RuntimeError("Load with rbln_create_runtimes=True before preparing requests")
        if processor is not None and self.engine.processor is not processor:
            if self.engine._parallel_preprocessor is not None:
                self.engine._parallel_preprocessor.close()
            self.engine.processor = processor
            self.engine._parallel_preprocessor = ParallelMediaProcessor(processor)
        with trace_span("prepare"):
            return self.engine.prepare(request)

    def encode(self, request, *, processor=None, dimensions=768):
        """Return one pooled embedding from raw text/image/audio input."""
        inputs = self.prepare(request, processor=processor)
        return self.forward(**inputs, dimensions=dimensions).embeddings


__all__ = ["RBLNEmbeddingGemma2Model", "RBLNEmbeddingGemma2Output"]

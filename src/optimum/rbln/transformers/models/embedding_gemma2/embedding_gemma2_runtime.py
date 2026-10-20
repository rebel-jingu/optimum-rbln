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

"""Host preprocessing/token assembly and explicitly selected encoder runners."""

from math import gcd

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .embedding_gemma2_architecture import audio_inputs, text_inputs, vision_inputs
from .embedding_gemma2_artifacts import select_bucket
from .embedding_gemma2_cache import BoundedTensorCache, tensor_key
from .embedding_gemma2_preprocessing import ParallelMediaProcessor, should_parallelize
from .embedding_gemma2_profiling import trace_span
from .embedding_gemma2_utils import DIMENSIONS, normalize_embedding


@torch.no_grad()
def assemble_embeddings(model, input_ids, image_tokens=None, audio_tokens=None):
    image_mask = input_ids.eq(model.config.image_token_id)
    audio_mask = input_ids.eq(model.config.audio_token_id)
    if input_ids.eq(model.config.video_token_id).any():
        raise ValueError("Video input is outside the text/image/audio interface")
    for name, mask, tokens in [("image", image_mask, image_tokens), ("audio", audio_mask, audio_tokens)]:
        count = int(mask.sum())
        if count != (0 if tokens is None else tokens.shape[0]):
            raise ValueError(f"{name} placeholder count ({count}) does not match encoded soft tokens")
    safe_ids = input_ids.masked_fill(image_mask | audio_mask, model.config.text_config.pad_token_id)
    embeddings = model.get_input_embeddings()(safe_ids)
    if image_tokens is not None:
        embeddings[image_mask] = image_tokens.to(embeddings.dtype)
    if audio_tokens is not None:
        embeddings[audio_mask] = audio_tokens.to(embeddings.dtype)
    return embeddings


class EmbeddingEngine:
    def __init__(
        self,
        model,
        processor=None,
        *,
        compile_cfgs,
        runtimes,
        image_soft_tokens=70,
        cache_metadata=True,
        parallel_preprocessing="auto",
    ):
        self.model, self.processor = model.eval(), processor
        self.cache_metadata = cache_metadata
        self.parallel_preprocessing = parallel_preprocessing
        self._parallel_preprocessor = (
            ParallelMediaProcessor(processor)
            if processor is not None and parallel_preprocessing is not False
            else None
        )
        self.metadata_cache = BoundedTensorCache()
        self.image_soft_tokens = image_soft_tokens
        self.last_stages, self.last_graphs = {}, []
        self.manifest = {"graphs": {}}
        self.runtimes = {}
        for cfg, runtime in zip(compile_cfgs, runtimes, strict=False):
            stage, capacity = cfg.compiled_model_name.split("_")
            key = f"{stage}:{capacity}"
            self.manifest["graphs"][key] = {
                "inputs": [
                    {"name": name, "shape": list(shape), "dtype": str(dtype).removeprefix("torch.")}
                    for name, shape, dtype in cfg.input_info
                ]
            }
            self.runtimes[key] = runtime
        self.text_buckets = self._capacities("text")
        self.audio_buckets = self._capacities("audio")

    def _capacities(self, stage):
        return tuple(int(key.split(":")[1]) for key in self.manifest["graphs"] if key.startswith(stage + ":"))

    def _graph_inputs(self, stage, dynamic, key_tensors, builder):
        if not self.cache_metadata:
            return builder()
        key = (
            stage,
            tuple((name, tuple(value.shape), str(value.dtype)) for name, value in dynamic.items()),
            tuple(tensor_key(value) for value in key_tensors),
        )

        def build_metadata():
            return {name: value for name, value in builder().items() if name not in dynamic}

        metadata = self.metadata_cache.get_or_create(key, build_metadata)
        # Raw embeddings, pixels and features remain request-local. Only derived
        # masks, RoPE and pooling metadata enter the bounded cache.
        return {**dynamic, **metadata}

    @torch.no_grad()
    def _run(self, stage, capacity, inputs):
        key = f"{stage}:{capacity}"
        metadata = self.manifest["graphs"][key]
        ordered = []
        for item in metadata["inputs"]:
            value = inputs[item["name"]].contiguous()
            if list(value.shape) != item["shape"] or str(value.dtype).removeprefix("torch.") != item["dtype"]:
                raise ValueError(f"Runtime input signature mismatch: {key}/{item['name']}")
            ordered.append(value)
        with trace_span(f"{stage}.npu", capacity=capacity):
            output = self.runtimes[key](*ordered)
        self.last_stages[stage] = "rbln"
        self.last_graphs.append(key)
        return tuple(output) if isinstance(output, list) else output

    def prepare(self, request):
        if self.processor is None:
            raise ValueError("A processor is required for raw requests")
        unknown = set(request) - {"text", "images", "audio", "prompt", "title"}
        if unknown:
            raise ValueError(f"Unknown request fields: {sorted(unknown)}")
        text = request.get("text")
        images = request.get("images", [])
        audio = request.get("audio", [])
        if not isinstance(images, (list, tuple)):
            images = [images]
        if not isinstance(audio, (list, tuple)):
            audio = [audio]
        if not text and not images and not audio:
            raise ValueError("Supply text, images, or audio")
        if text is not None and not isinstance(text, str):
            raise ValueError("text must be one string")
        prompt = request.get("prompt")
        if prompt:
            prefixes = {
                "SearchQuery": "task: search result | query: ",
                "QuestionAnswering": "task: question answering | query: ",
                "Document": f"title: {request.get('title', 'none')} | text: ",
            }
            if prompt not in prefixes or not text:
                raise ValueError("Supply text and a supported prompt: SearchQuery, QuestionAnswering, Document")
            text = prefixes[prompt] + text
        if text is None:
            # One API call is one sample, including several media items. A flat
            # audio list without text otherwise means several samples to HF.
            text = " ".join(["<|image|>"] * len(images) + ["<|audio|>"] * len(audio))
        if text is not None:
            # For simple text+media requests, append placeholders. Explicit mixed ordering is preserved.
            if images and "<|image|>" not in text:
                text += " " + " ".join(["<|image|>"] * len(images))
            if audio and "<|audio|>" not in text:
                text += " " + " ".join(["<|audio|>"] * len(audio))
        if (
            self._parallel_preprocessor is not None
            and images
            and audio
            and should_parallelize(self.parallel_preprocessing, audio)
        ):
            return self._parallel_preprocessor.prepare(
                text=text, images=images, audio=audio, image_soft_tokens=self.image_soft_tokens
            )
        loaded_images = []
        for item in images:
            if isinstance(item, Image.Image):
                loaded_images.append(item.convert("RGB"))
            else:
                with Image.open(item) as image:
                    loaded_images.append(image.convert("RGB"))
        waveforms = []
        for item in audio:
            if isinstance(item, np.ndarray):
                waveform = item.astype(np.float32)
            else:
                import soundfile as sf
                from scipy.signal import resample_poly

                waveform, rate = sf.read(item, dtype="float32", always_2d=True)
                waveform = waveform.mean(-1)
                if rate != 16000:
                    divisor = gcd(rate, 16000)
                    waveform = resample_poly(waveform, 16000 // divisor, rate // divisor).astype(np.float32)
            if waveform.ndim != 1 or not len(waveform) or not np.isfinite(waveform).all():
                raise ValueError("Audio must be a nonempty, finite mono waveform")
            waveforms.append(waveform)
        kwargs = {"return_tensors": "pt", "truncation": False}
        if text is not None:
            kwargs["text"] = text
        if loaded_images:
            kwargs.update(images=loaded_images, images_kwargs={"max_soft_tokens": self.image_soft_tokens})
        if waveforms:
            kwargs.update(audio=waveforms, sampling_rate=16000)
        return self.processor(**kwargs)

    def close(self):
        """Release the optional persistent CPU media workers."""
        if self._parallel_preprocessor is not None:
            self._parallel_preprocessor.close()

    def encode(self, request: dict, dimensions=768) -> np.ndarray:
        with trace_span("host.prepare"):
            inputs = self.prepare(request)
        return self.encode_inputs(inputs, dimensions)

    @torch.no_grad()
    def encode_inputs(self, inputs, dimensions=768) -> np.ndarray:
        self.last_stages = {}
        self.last_graphs = []
        if dimensions not in DIMENSIONS:
            raise ValueError(f"Unsupported embedding dimension: {dimensions}")
        ids, mask = inputs["input_ids"], inputs["attention_mask"]
        if ids.shape[0] != 1:
            raise ValueError("Only batch size one is supported; call encode sequentially")
        if not mask.any() or ids.shape[1] > 8192:
            raise ValueError("Input must contain 1–8192 tokens within compiled capacity")
        text_capacity = select_bucket(ids.shape[1], self.text_buckets)
        # Preflight every capacity before launching any encoder.
        expected_images = 0
        if "pixel_values" in inputs:
            valid_patches = inputs["image_position_ids"].ne(-1).all(-1).sum(-1)
            k = self.model.vision_tower.config.pooling_kernel_size
            expected_images = int((valid_patches // (k * k)).sum())
        expected_audio = int(inputs["input_features_mask"][:, ::4].sum()) if "input_features" in inputs else 0
        for token, count, label in [
            (self.model.config.image_token_id, expected_images, "image"),
            (self.model.config.audio_token_id, expected_audio, "audio"),
        ]:
            if int(ids.eq(token).sum()) != count:
                raise ValueError(f"{label} placeholder count does not match media inputs")
        if ids.eq(self.model.config.video_token_id).any():
            raise ValueError("Video input is outside the text/image/audio interface")
        if "pixel_values" in inputs and self.manifest:
            select_bucket(inputs["pixel_values"].shape[1], self._capacities("vision"))
        audio_capacities = []
        if "input_features" in inputs:
            for valid in inputs["input_features_mask"]:
                audio_capacities.append(select_bucket(int(valid.sum()), self.audio_buckets))
        image_tokens, audio_tokens = None, None
        if "pixel_values" in inputs:
            outputs = []
            for pixels, positions in zip(inputs["pixel_values"], inputs["image_position_ids"], strict=False):
                with trace_span("host.vision_inputs"):
                    args = self._graph_inputs(
                        "vision",
                        {"pixel_values": pixels[None], "pixel_position_ids": positions[None]},
                        [positions],
                        lambda pixels=pixels, positions=positions: vision_inputs(
                            self.model, pixels[None], positions[None]
                        ),
                    )
                tokens, valid = self._run("vision", pixels.shape[0], args)
                outputs.append(tokens[valid.bool()])
            image_tokens = torch.cat(outputs)
        if "input_features" in inputs:
            outputs = []
            for features, feature_mask, capacity in zip(
                inputs["input_features"], inputs["input_features_mask"], audio_capacities, strict=False
            ):
                length = int(feature_mask.sum())
                features = F.pad(features[:length], (0, 0, 0, capacity - length))[None]
                valid = F.pad(feature_mask[:length], (0, capacity - length))[None]
                with trace_span("host.audio_inputs"):
                    args = self._graph_inputs(
                        "audio",
                        {"input_features": features, "input_features_mask": valid},
                        [valid],
                        lambda features=features, valid=valid: audio_inputs(self.model, features, valid),
                    )
                tokens, valid = self._run("audio", capacity, args)
                outputs.append(tokens[valid.bool()])
            audio_tokens = torch.cat(outputs)
        with trace_span("host.token_assembly"):
            embeddings = assemble_embeddings(self.model, ids, image_tokens, audio_tokens)
            embeddings = F.pad(embeddings, (0, 0, 0, text_capacity - embeddings.shape[1]))
            pooling_mask = F.pad(mask, (0, text_capacity - mask.shape[1]))
        with trace_span("host.text_inputs"):
            args = self._graph_inputs(
                "text",
                {"inputs_embeds": embeddings, "pooling_mask": pooling_mask},
                [pooling_mask],
                lambda: text_inputs(self.model, embeddings, pooling_mask),
            )
        result = self._run("text", text_capacity, args)
        if isinstance(result, tuple):
            result = result[0]
        with trace_span("host.output_normalize"):
            return normalize_embedding(result, dimensions)[0].cpu().numpy()

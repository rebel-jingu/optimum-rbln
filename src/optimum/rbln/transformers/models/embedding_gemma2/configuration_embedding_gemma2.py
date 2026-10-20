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

"""Static encoder capacities for embedding-only multimodal inference."""

import torch

from ....configuration_utils import RBLNModelConfig


class RBLNEmbeddingGemma2ModelConfig(RBLNModelConfig):
    """EmbeddingGemma 2 configuration. All encoder buckets share ``num_devices``.

    ``audio_layout='selector'`` opts into the constant selection-matrix relative
    shift measured on CA25. ``row`` retains the validated CR13 formulation.
    These graphs return pooled embeddings, not per-token hidden states.
    """

    def __init__(
        self,
        batch_size=1,
        text_buckets=None,
        audio_buckets=None,
        image_soft_tokens=70,
        audio_layout="row",
        cache_metadata=True,
        **kwargs,
    ):
        saved = kwargs.get("_compile_cfgs") or []
        if saved:
            first = saved[0] if isinstance(saved[0], dict) else saved[0].asdict()
            kwargs.setdefault("npu", first.get("npu"))
            kwargs.setdefault("num_devices", first.get("num_devices") or 1)
        else:
            kwargs.setdefault("num_devices", 1)
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.text_buckets = list((32, 64, 96, 128, 288, 352, 512) if text_buckets is None else text_buckets)
        self.audio_buckets = list((64, 128, 1056, 1536) if audio_buckets is None else audio_buckets)
        self.image_soft_tokens = image_soft_tokens
        self.audio_layout = audio_layout
        self.cache_metadata = cache_metadata
        if batch_size != 1 or isinstance(batch_size, bool):
            raise ValueError("EmbeddingGemma 2 supports batch_size=1")
        if self.dtype not in (torch.float32, torch.bfloat16):
            raise ValueError("EmbeddingGemma 2 supports float32 or bfloat16")
        for name, buckets, low, high in (
            ("text", self.text_buckets, 1, 8192),
            ("audio", self.audio_buckets, 4, 3072),
        ):
            if not buckets or any(type(n) is not int or not low <= n <= high for n in buckets):
                raise ValueError(f"{name} buckets must be integers within {low}–{high}")
            setattr(self, name + "_buckets", sorted(set(buckets)))
        if image_soft_tokens not in (70, 280):
            raise ValueError("image_soft_tokens must be 70 or 280")
        if audio_layout not in ("row", "selector"):
            raise ValueError("audio_layout must be row or selector")
        if type(self.num_devices) is not int or self.num_devices < 1:
            raise ValueError("num_devices must be a positive integer")


__all__ = ["RBLNEmbeddingGemma2ModelConfig"]

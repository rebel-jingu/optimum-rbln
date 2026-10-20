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

"""Pinned reference loading and embedding postprocessing."""

import importlib.metadata

import torch


MODEL_ID = "google/embeddinggemma-2"
MODEL_REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"
DIMENSIONS = (128, 256, 512, 768)


def validate_dtype(dtype: str) -> torch.dtype:
    if dtype not in ("float32", "bfloat16"):
        raise ValueError("Use float32 or bfloat16; float16 is unsafe for EmbeddingGemma 2")
    return getattr(torch, dtype)


def masked_mean(states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.unsqueeze(-1).to(torch.float32)
    return (states.float() * weights).sum(1) / weights.sum(1).clamp_min(1)


def normalize_embedding(tensor: torch.Tensor, dimensions: int = 768) -> torch.Tensor:
    if dimensions not in DIMENSIONS or tensor.shape[-1] < dimensions:
        raise ValueError(f"Unsupported embedding dimension: {dimensions}")
    if not torch.isfinite(tensor).all():
        raise ValueError("Embedding is nonfinite or has zero norm")
    vector = tensor[..., :dimensions].float()
    if (vector.norm(dim=-1) < 1e-12).any():
        raise ValueError("Embedding is nonfinite or has zero norm")
    return torch.nn.functional.normalize(vector, p=2, dim=-1)


def load_reference(model_id=MODEL_ID, revision=MODEL_REVISION, dtype="bfloat16"):
    from transformers import AutoModel, AutoProcessor

    precision = validate_dtype(dtype)
    model = AutoModel.from_pretrained(
        model_id,
        revision=revision,
        dtype=precision,
        # Transformers 5.20 fixes Gemma4 audio's additive eager mask and
        # explicitly disallows SDPA for its custom chunked attention.
        attn_implementation="eager",
    ).eval()
    processor = AutoProcessor.from_pretrained(model_id, revision=revision)
    return model, processor


def package_versions() -> dict:
    versions = {}
    for name in ("torch", "transformers", "torchvision", "rebel-compiler", "numpy"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions

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

"""Small host state: token embeddings and positional helpers, no encoder layers."""

from types import MethodType

import torch
from torch import nn
from transformers.models.embedding_gemma2.modeling_embedding_gemma2 import (
    EmbeddingGemma2RotaryEmbedding,
    EmbeddingGemma2TextScaledWordEmbedding,
)
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4AudioModel,
    Gemma4AudioRelPositionalEncoding,
    Gemma4VisionRotaryEmbedding,
)


class HostModel(nn.Module):
    def __init__(self, config, dtype):
        super().__init__()
        self.config = config
        # Mask builders must produce explicit eager additive masks.
        for cfg in (config.text_config, config.vision_config, config.audio_config):
            if cfg is not None:
                cfg._attn_implementation = "eager"
        self.language_model = nn.Module()
        self.language_model.config = config.text_config
        text = config.text_config
        with torch.device("meta"):
            self.language_model.embed_tokens = EmbeddingGemma2TextScaledWordEmbedding(
                text.vocab_size,
                text.hidden_size,
                text.pad_token_id,
                embed_scale=text.hidden_size**0.5,
            )
        self.language_model.rotary_emb = EmbeddingGemma2RotaryEmbedding(text)
        self.vision_tower = None
        if config.vision_config is not None:
            self.vision_tower = nn.Module()
            self.vision_tower.config = config.vision_config
            self.vision_tower.register_parameter("_dtype_anchor", nn.Parameter(torch.empty(0, dtype=dtype)))
            self.vision_tower.encoder = nn.Module()
            self.vision_tower.encoder.rotary_emb = Gemma4VisionRotaryEmbedding(config.vision_config)
        self.audio_tower = None
        if config.audio_config is not None:
            self.audio_tower = nn.Module()
            self.audio_tower.config = config.audio_config
            self.audio_tower.register_parameter("_dtype_anchor", nn.Parameter(torch.empty(0, dtype=dtype)))
            self.audio_tower.rel_pos_enc = Gemma4AudioRelPositionalEncoding(config.audio_config)
            self.audio_tower._convert_4d_mask_to_blocked_5d = MethodType(
                Gemma4AudioModel._convert_4d_mask_to_blocked_5d,
                self.audio_tower,
            )

    def get_input_embeddings(self):
        return self.language_model.embed_tokens


def save_host_model(model, path):
    host = HostModel(model.config, next(model.parameters()).dtype)
    buffers = {
        name: value.detach().cpu().clone()
        for name, value in model.named_buffers()
        if name in dict(host.named_buffers())
    }
    torch.save(
        {"schema": 1, "embeddings": model.get_input_embeddings().state_dict(), "position_buffers": buffers}, path
    )


def load_host_model(config, path, dtype):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("schema") != 1:
        raise ValueError("Unsupported EmbeddingGemma 2 host-state schema")
    host = HostModel(config, dtype)
    host.get_input_embeddings().load_state_dict(state["embeddings"], assign=True)
    expected = {name for name, _ in host.named_buffers()}
    if set(state["position_buffers"]) != expected:
        raise ValueError("Host positional state does not match the model configuration")
    for name, value in state["position_buffers"].items():
        parent, key = name.rsplit(".", 1)
        module = host.get_submodule(parent)
        if value.shape != getattr(module, key).shape:
            raise ValueError(f"Invalid host buffer shape: {name}")
        module.register_buffer(key, value, persistent=False)
    if host.get_input_embeddings().weight.dtype != dtype:
        raise ValueError("Host embeddings and compiled dtype differ")
    return host.eval()

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

"""Stateless graph boundaries; dynamic masks and compaction stay on the host.

The vision layer-loop/pool boundary follows optimum-rbln's Gemma4 wrapper.
The original HF layer modules/weights remain intact for reference comparisons.
"""

import copy
from types import MethodType

import torch
import torch.nn.functional as F
from torch import nn
from transformers.masking_utils import (
    create_bidirectional_mask,
    create_bidirectional_sliding_window_mask,
)
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4AudioAttention,
    Gemma4ClippableLinear,
    Gemma4VisionAttention,
    apply_multidimensional_rope,
    repeat_kv,
    sliding_window_mask_function,
)

from .embedding_gemma2_utils import masked_mean


def static_normalize(vector):
    # Equivalent to F.normalize(p=2, eps=1e-12), without linalg_vector_norm,
    # which the tested CR13 compiler cannot lower.
    vector = vector.float()
    squared_norm = (vector * vector).sum(-1, keepdim=True).clamp_min(1e-24)
    return vector * torch.pow(squared_norm, -0.5)


@torch.no_grad()
def text_inputs(model, inputs_embeds, pooling_mask):
    language = model.language_model
    kwargs = {
        "config": language.config,
        "inputs_embeds": inputs_embeds,
        "attention_mask": pooling_mask,
        "allow_is_bidirectional_skip": False,
    }
    full = create_bidirectional_mask(**kwargs)
    sliding = create_bidirectional_sliding_window_mask(**kwargs)
    positions = torch.arange(inputs_embeds.shape[1]).unsqueeze(0)
    full_cos, full_sin = language.rotary_emb(inputs_embeds, positions, "full_attention")
    sliding_cos, sliding_sin = language.rotary_emb(inputs_embeds, positions, "sliding_attention")
    return {
        "inputs_embeds": inputs_embeds,
        "full_mask": full,
        "sliding_mask": sliding,
        "full_cos": full_cos,
        "full_sin": full_sin,
        "sliding_cos": sliding_cos,
        "sliding_sin": sliding_sin,
        "pooling_mask": pooling_mask,
    }


class TextGraph(nn.Module):
    def __init__(self, model, *, normalize_output=True):
        super().__init__()
        self.normalize_output = normalize_output
        text = model.language_model
        self.layers, self.ple = text.layers, text.ple
        self.norm, self.projection = text.norm, text.embedding_projection
        self.layer_types = tuple(text.config.layer_types)

    def forward(
        self, inputs_embeds, full_mask, sliding_mask, full_cos, full_sin, sliding_cos, sliding_sin, pooling_mask
    ):
        per_layer = self.ple(inputs_embeds)
        states = inputs_embeds
        for i, layer in enumerate(self.layers):
            local = self.layer_types[i] == "sliding_attention"
            states = layer(
                states,
                per_layer[:, :, i, :],
                attention_mask=sliding_mask if local else full_mask,
                position_embeddings=(sliding_cos, sliding_sin) if local else (full_cos, full_sin),
            )
        projected = self.projection(self.norm(states))
        pooled = masked_mean(projected, pooling_mask)
        return static_normalize(pooled) if self.normalize_output else pooled


@torch.no_grad()
def vision_inputs(model, pixel_values, pixel_position_ids):
    tower = model.vision_tower
    padding = pixel_position_ids.eq(-1).all(-1)
    dtype = next(tower.parameters()).dtype
    dummy = torch.empty((*pixel_values.shape[:2], tower.config.hidden_size), dtype=dtype)
    mask = create_bidirectional_mask(
        config=tower.config, inputs_embeds=dummy, attention_mask=~padding, allow_is_bidirectional_skip=False
    )
    cos, sin = tower.encoder.rotary_emb(dummy, pixel_position_ids)
    k = tower.config.pooling_kernel_size
    length = pixel_values.shape[1] // (k * k)
    coords = pixel_position_ids.clamp_min(0)
    width = coords[..., 0].max(-1, keepdim=True).values + 1
    cells = coords // k
    indices = cells[..., 0] + (width // k) * cells[..., 1]
    weights = F.one_hot(indices.long(), length).float() / (k * k)
    return {
        "pixel_values": pixel_values,
        "pixel_position_ids": pixel_position_ids,
        "padding_positions": padding,
        "attention_mask": mask,
        "cos": cos,
        "sin": sin,
        "pool_weights": weights,
        "pool_mask": weights.ne(0).any(1),
    }


class VisionGraph(nn.Module):
    def __init__(self, model):
        super().__init__()
        tower = model.vision_tower
        self.patch_embedder = static_modules(tower.patch_embedder)
        self.layers = static_modules(tower.encoder.layers)
        self.projector = model.embed_vision
        self.scale = tower.config.hidden_size**0.5
        self.standardize = tower.config.standardize
        if self.standardize:
            self.register_buffer("std_bias", tower.std_bias)
            self.register_buffer("std_scale", tower.std_scale)

    def forward(
        self, pixel_values, pixel_position_ids, padding_positions, attention_mask, cos, sin, pool_weights, pool_mask
    ):
        states = self.patch_embedder(pixel_values, pixel_position_ids, padding_positions)
        for layer in self.layers:
            states = layer(
                states, attention_mask=attention_mask, position_embeddings=(cos, sin), position_ids=pixel_position_ids
            )
        states = states.masked_fill(padding_positions.unsqueeze(-1), 0)
        pooled = (pool_weights.transpose(1, 2) @ states.float()).to(states.dtype)
        pooled = pooled.float() * self.scale
        if self.standardize:
            pooled = (pooled - self.std_bias.float()) * self.std_scale.float()
        return self.projector(pooled.to(states.dtype)), pool_mask


@torch.no_grad()
def audio_inputs(model, input_features, input_features_mask):
    tower = model.audio_tower
    # Both stride-2 convolutions sample the validity mask at every other frame.
    output_mask = input_features_mask[:, ::4].bool()
    dtype = next(tower.parameters()).dtype
    dummy = torch.empty((*output_mask.shape, tower.config.hidden_size), dtype=dtype)
    mask = create_bidirectional_mask(
        config=tower.config,
        inputs_embeds=dummy,
        attention_mask=output_mask,
        and_mask_function=sliding_window_mask_function(
            (tower.config.attention_context_left - 1, tower.config.attention_context_right)
        ),
    )
    blocked = tower._convert_4d_mask_to_blocked_5d(mask)
    return {
        "input_features": input_features,
        "input_features_mask": input_features_mask,
        "attention_mask": blocked,
        "position_embeddings": tower.rel_pos_enc(dummy),
    }


class StaticClippableLinear(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.linear = original.linear
        self.use_clipped_linears = original.use_clipped_linears
        if self.use_clipped_linears:
            # SDK dev316 decodes BF16 scalar Tensor clamp bounds as uint16
            # bit patterns. Python floats preserve their actual stored values.
            for name in ("input_min", "input_max", "output_min", "output_max"):
                setattr(self, name, float(getattr(original, name)))

    def forward(self, states):
        if self.use_clipped_linears:
            states = torch.clamp(states, self.input_min, self.input_max)
        states = self.linear(states)
        if self.use_clipped_linears:
            states = torch.clamp(states, self.output_min, self.output_max)
        return states


class ConstantPositionProjection(nn.Module):
    """Fold a request-independent sinusoid projection for the pinned model."""

    def __init__(self, projection, positions):
        super().__init__()
        with torch.no_grad():
            self.register_buffer("projected", projection(positions).detach(), persistent=False)

    def forward(self, positions):
        return self.projected


class AlignedVisionAttention(nn.Module):
    """Keep real queries/patches while aligning masked K/V for SDK fused SDPA."""

    def __init__(self, original):
        super().__init__()
        for name in (
            "head_dim",
            "scaling",
            "num_key_value_groups",
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "q_norm",
            "k_norm",
            "v_norm",
        ):
            setattr(self, name, getattr(original, name))

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None, position_ids=None, **kwargs):
        shape = hidden_states.shape[:-1]
        heads = (*shape, -1, self.head_dim)
        cos, sin = position_embeddings
        query = self.q_norm(self.q_proj(hidden_states).view(heads))
        query = apply_multidimensional_rope(query, cos, sin, position_ids).transpose(1, 2)
        key = self.k_norm(self.k_proj(hidden_states).view(heads))
        key = apply_multidimensional_rope(key, cos, sin, position_ids).transpose(1, 2)
        value = self.v_norm(self.v_proj(hidden_states).view(heads)).transpose(1, 2)
        key = repeat_kv(key, self.num_key_value_groups)
        value = repeat_kv(value, self.num_key_value_groups)
        # SDK dev316 fuses noncausal SDPA when K length/head dimensions align to
        # 64. Masked extra keys do not change patch count, RoPE or pool order.
        padding = (-key.shape[-2]) % 64
        if attention_mask is None:
            attention_mask = query.new_zeros(1, 1, 1, key.shape[-2])
        fill = False if attention_mask.dtype == torch.bool else float("-inf")
        attention_mask = F.pad(attention_mask, (0, padding), value=fill)
        key = F.pad(key, (0, 0, 0, padding))
        value = F.pad(value, (0, 0, 0, padding))
        states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False, scale=self.scaling
        )
        states = states.transpose(1, 2).reshape(*shape, -1).contiguous()
        return self.o_proj(states), None


class StaticAudioAttention(Gemma4AudioAttention):
    def __init__(self, original):
        nn.Module.__init__(self)
        for name in (
            "config",
            "layer_idx",
            "attention_logits_soft_cap",
            "head_dim",
            "num_heads",
            "q_scale",
            "k_scale",
            "chunk_size",
            "max_past_horizon",
            "max_future_horizon",
            "context_size",
        ):
            setattr(self, name, getattr(original, name))
        for name in ("q_proj", "k_proj", "v_proj", "post", "relative_k_proj"):
            setattr(self, name, getattr(original, name))
        self.per_dim_scale = original.per_dim_scale
        self.register_buffer("softcap", original.softcap, persistent=False)

    def _extract_block_context(self, hidden_states):
        # aten::unfold is unsupported even by SDK dev316. The graph's fixed
        # sequence length lets us express the same overlapping windows as slices.
        count = (hidden_states.shape[1] + self.chunk_size - 1) // self.chunk_size
        padded = F.pad(
            hidden_states, (0, 0, 0, 0, self.max_past_horizon, self.max_future_horizon + self.chunk_size - 1)
        )
        return torch.stack(
            [padded[:, i * self.chunk_size : i * self.chunk_size + self.context_size] for i in range(count)], dim=1
        ).contiguous()

    def _rel_shift(self, states):
        # The original flatten/slice/reshape skew becomes a host graph break.
        # Express the same contiguous flattened interval one output row at a
        # time, without reshaping a shortened physical buffer on the device.
        block_size, position_length = states.shape[-2:]
        context = self.context_size
        padded = F.pad(states, (0, context + 1 - position_length))
        rows = []
        for query in range(block_size):
            row, col = divmod(query * context, context + 1)
            first = min(context, context + 1 - col)
            segment = padded[..., row, col : col + first]
            if first < context:
                segment = torch.cat((segment, padded[..., row + 1, : context - first]), dim=-1)
            rows.append(segment)
        return torch.stack(rows, dim=-2)


def static_modules(original):
    """Copy module maps, share weights, and adapt only unsupported operations."""
    if isinstance(original, Gemma4ClippableLinear):
        return StaticClippableLinear(original)
    if isinstance(original, Gemma4AudioAttention):
        result = StaticAudioAttention(original)
    elif isinstance(original, Gemma4VisionAttention):
        result = AlignedVisionAttention(original)
    else:
        result = copy.copy(original)
    result._modules = {
        name: static_modules(child) if child is not None else None for name, child in result._modules.items()
    }
    return result


class AudioGraph(nn.Module):
    def __init__(self, model):
        super().__init__()
        tower = model.audio_tower
        self.subsample = tower.subsample_conv_projection
        self.layers = static_modules(tower.layers)
        # Relative positions depend only on the fixed attention configuration,
        # never on request features or their valid length. Folding each learned
        # projection removes its repeated host/device layout conversion.
        weight = next(tower.parameters())
        dummy = torch.empty(1, 1, tower.config.hidden_size, dtype=weight.dtype, device=weight.device)
        positions = tower.rel_pos_enc(dummy)
        for layer in self.layers:
            attention = layer.self_attn
            attention.relative_k_proj = ConstantPositionProjection(attention.relative_k_proj, positions)
        self.output_proj = tower.output_proj
        self.projector = model.embed_audio

    def forward(self, input_features, input_features_mask, attention_mask, position_embeddings):
        states, valid = self.subsample(input_features, input_features_mask)
        for layer in self.layers:
            states = layer(states, attention_mask=attention_mask, position_embeddings=position_embeddings)
        return self.projector(self.output_proj(states)), valid


def _selector_relative_shift(self, states):
    shape = states.shape
    values = states.reshape(*shape[:-2], self.chunk_size * self._selector_position_length)
    shifted = values @ self._shift_selector
    return shifted.view(*shape[:-2], self.chunk_size, self.context_size)


def configure_audio_layout(graph, layout):
    """Opt into the CA25-tested fixed selection matrix; preserve the source model."""
    if layout == "row":
        return graph
    if layout != "selector":
        raise ValueError("Unknown audio layout")
    for layer in graph.layers:
        attention = layer.self_attn
        chunk, context = attention.chunk_size, attention.context_size
        positions = context // 2 + 1
        selector = torch.zeros(chunk * positions, chunk * context, dtype=torch.float32)
        for output in range(chunk * context):
            row, col = divmod(output, context + 1)
            if col < positions:
                selector[row * positions + col, output] = 1
        attention.register_buffer("_shift_selector", selector, persistent=False)
        attention._selector_position_length = positions
        attention._rel_shift = MethodType(_selector_relative_shift, attention)
    return graph

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

"""Bounded caches for reusable CPU masks, positions, and rotary metadata."""

import hashlib
from collections import OrderedDict
from collections.abc import Callable, Hashable

import torch


def tensor_key(tensor: torch.Tensor) -> tuple:
    """Hash CPU tensor shape, dtype, and logical contents, independent of strides.

    The byte view supports BF16 without asking NumPy to represent its dtype.
    This is intended for small metadata tensors, not embeddings or media features.
    Callers must not mutate a tensor while its key is being computed.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("tensor_key requires a torch.Tensor")
    if tensor.device.type != "cpu":
        raise ValueError("tensor_key requires a CPU tensor")
    if tensor.layout != torch.strided:
        raise ValueError("tensor_key requires a dense strided tensor")
    values = tensor.detach().resolve_conj().resolve_neg().contiguous().reshape(-1)
    digest = hashlib.sha256(values.view(torch.uint8).numpy().tobytes()).digest()
    return tuple(tensor.shape), str(tensor.dtype), digest


class BoundedTensorCache:
    """LRU cache of CPU tensor dictionaries for single-threaded inference.

    Dictionaries and tensors are returned by reference, without cloning. Builders
    and callers must treat cached dictionaries and tensor storage as read-only;
    mutating them invalidates both cached results and byte accounting. No locks
    are provided. Values must be dense strided CPU tensors. Byte accounting sums
    retained storage sizes, conservatively counting aliases more than once, so
    a small view cannot retain an unaccounted large buffer. Either zero limit
    disables storage.
    """

    def __init__(self, max_bytes: int = 64 * 1024**2, max_entries: int = 32):
        for name, value in (("max_bytes", max_bytes), ("max_entries", max_entries)):
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be a nonnegative integer")
            if value < 0:
                raise ValueError(f"{name} must be nonnegative")
        self._max_bytes = max_bytes
        self._max_entries = max_entries
        self._entries: OrderedDict[Hashable, tuple[dict[str, torch.Tensor], int]] = OrderedDict()
        self._bytes = 0
        self._hits = 0
        self._misses = 0

    def get_or_create(
        self,
        key: Hashable,
        builder: Callable[[], dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        """Return cached metadata, or validate/build it and apply both LRU limits."""
        if key in self._entries:
            self._hits += 1
            self._entries.move_to_end(key)
            return self._entries[key][0]

        self._misses += 1
        metadata = builder()
        if not isinstance(metadata, dict):
            raise TypeError("metadata builder must return a dict[str, torch.Tensor]")
        entry_bytes = 0
        for name, tensor in metadata.items():
            if not isinstance(name, str):
                raise TypeError("metadata dictionary keys must be str")
            if not isinstance(tensor, torch.Tensor):
                raise TypeError("metadata dictionary values must be torch.Tensor")
            if tensor.device.type != "cpu":
                raise ValueError("metadata dictionary values must be CPU tensors")
            if tensor.layout != torch.strided:
                raise ValueError("metadata dictionary values must be dense strided tensors")
            entry_bytes += tensor.untyped_storage().nbytes()

        if not self._max_entries or not self._max_bytes or entry_bytes > self._max_bytes:
            return metadata
        while len(self._entries) >= self._max_entries or self._bytes + entry_bytes > self._max_bytes:
            _, (_, evicted_bytes) = self._entries.popitem(last=False)
            self._bytes -= evicted_bytes
        self._entries[key] = metadata, entry_bytes
        self._bytes += entry_bytes
        return metadata

    def clear(self) -> None:
        """Release all cached references and reset hit/miss statistics."""
        self._entries.clear()
        self._bytes = 0
        self._hits = 0
        self._misses = 0

    def info(self) -> dict[str, int]:
        """Return a snapshot of hits, misses, entry count, and tensor bytes."""
        return {"hits": self._hits, "misses": self._misses, "entries": len(self._entries), "bytes": self._bytes}

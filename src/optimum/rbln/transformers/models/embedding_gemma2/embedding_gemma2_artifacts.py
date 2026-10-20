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

"""Validated artifact manifests and bounded shape selection."""

import hashlib
import json
from pathlib import Path

from .embedding_gemma2_utils import MODEL_REVISION, validate_dtype


def select_bucket(required: int, capacities) -> int:
    if required <= 0:
        raise ValueError("Input length must be positive")
    for capacity in sorted(capacities):
        if required <= capacity:
            return capacity
    raise ValueError(f"Input length {required} exceeds compiled capacity {sorted(capacities)}")


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(directory) -> dict:
    root = Path(directory).resolve()
    doc = json.loads((root / "manifest.json").read_text())
    if doc.get("schema") != 1:
        raise ValueError("Unsupported artifact manifest schema")
    if doc.get("model_revision") != MODEL_REVISION:
        raise ValueError("Artifact model revision does not match this implementation")
    validate_dtype(doc.get("dtype"))
    if not doc.get("graphs"):
        raise ValueError("Artifact manifest contains no graphs")
    for name, graph in doc["graphs"].items():
        path = (root / graph["file"]).resolve()
        if Path(graph["file"]).is_absolute() or root not in path.parents:
            raise ValueError(f"Invalid graph path: {graph['file']}")
        if not path.is_file():
            raise ValueError(f"Missing compiled graph {name}: {path}")
        if file_sha256(path) != graph["sha256"]:
            raise ValueError(f"Artifact hash mismatch: {name}")
        if not graph.get("inputs"):
            raise ValueError(f"Missing input signature: {name}")
        for item in graph["inputs"]:
            if item["dtype"] == "float16":
                raise ValueError("Unsafe float16 artifact input signature")
            if any(not isinstance(n, int) or n < 1 for n in item["shape"]):
                raise ValueError(f"Invalid static graph shape: {name}")
    return doc


def save_manifest(directory, doc):
    root = Path(directory)
    staging = root / "manifest.json.partial"
    staging.write_text(json.dumps(doc, indent=2) + "\n")
    staging.replace(root / "manifest.json")

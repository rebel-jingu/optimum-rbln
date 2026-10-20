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

"""One-time import of verified standalone graphs into the native model layout."""

import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import rebel

from ....configuration_utils import RBLNCompileConfig
from .configuration_embedding_gemma2 import RBLNEmbeddingGemma2ModelConfig
from .embedding_gemma2_artifacts import file_sha256, load_manifest
from .embedding_gemma2_host import save_host_model
from .embedding_gemma2_utils import MODEL_REVISION, package_versions


def import_legacy_bundle(source, source_model, destination, processor=None, *, source_revision=None):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    manifest = load_manifest(source)
    if destination.exists():
        raise ValueError("Use a new native artifact directory")
    config_revision = getattr(source_model.config, "_commit_hash", None)
    if (source_revision or config_revision) != MODEL_REVISION or config_revision not in (None, MODEL_REVISION):
        raise ValueError("The source model must be loaded from the pinned model revision")
    if str(source_model.dtype).removeprefix("torch.") != manifest["dtype"]:
        raise ValueError("Source model and legacy artifact dtype differ")
    current = package_versions()
    for name in ("torch", "transformers", "rebel-compiler"):
        if current[name] != manifest["versions"].get(name):
            raise ValueError(f"Legacy artifact package version mismatch: {name}")
    configs = []
    for key, graph in manifest["graphs"].items():
        stage, capacity = key.split(":")
        if stage not in ("text", "vision", "audio") or int(capacity) < 1:
            raise ValueError(f"Unknown legacy graph: {key}")
        inspected = rebel.RBLNCompiledModel.inspect(source / graph["file"])
        if (
            inspected.get("npu") != manifest["npu"]
            or inspected.get("num_devices") != manifest["num_devices"]
            or inspected["inputs"] != graph["inputs"]
        ):
            raise ValueError(f"Legacy manifest differs from the binary: {key}")
        configs.append(
            RBLNCompileConfig(
                compiled_model_name=f"{stage}_{capacity}",
                input_info=[(item["name"], item["shape"], item["dtype"]) for item in graph["inputs"]],
            )
        )
    # Exact per-graph provenance survives import, including the CA25 audio1056
    # selector override. audio_layout governs future compilation, not conversion.
    cfg = RBLNEmbeddingGemma2ModelConfig(
        dtype=manifest["dtype"],
        npu=manifest["npu"],
        num_devices=manifest["num_devices"],
        text_buckets=[int(key.split(":")[1]) for key in manifest["graphs"] if key.startswith("text:")],
        audio_buckets=[int(key.split(":")[1]) for key in manifest["graphs"] if key.startswith("audio:")],
        image_soft_tokens=manifest["image_soft_tokens"],
    )
    cfg.set_compile_cfgs(configs)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".embeddinggemma2-import-", dir=destination.parent) as directory:
        staging = Path(directory)
        for key, graph in manifest["graphs"].items():
            target = staging / (key.replace(":", "_") + ".rbln")
            shutil.copyfile(source / graph["file"], target)
            if file_sha256(target) != graph["sha256"]:
                raise ValueError(f"Binary changed while importing {key}")
        cfg.save(staging)
        source_model.config.save_pretrained(staging)
        save_host_model(source_model, staging / "host_state.pt")
        if processor is not None:
            processor.save_pretrained(staging)
        provenance = {
            "operation": "import unchanged compiled binaries; no recompilation",
            "source_manifest_sha256": file_sha256(source / "manifest.json"),
            "source_manifest": manifest,
        }
        (staging / "legacy_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        staging.rename(destination)
    return destination

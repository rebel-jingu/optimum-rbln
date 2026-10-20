"""Full-weight numerical validation on deterministic synthetic media fixtures."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from optimum.rbln.transformers.models.embedding_gemma2.embedding_gemma2_utils import MODEL_REVISION, package_versions


def embedding_metrics(actual, reference):
    actual, reference = np.asarray(actual, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    finite = bool(np.isfinite(actual).all() and np.isfinite(reference).all())
    same_shape = actual.shape == reference.shape
    if not finite or not same_shape or not np.linalg.norm(actual) or not np.linalg.norm(reference):
        return {"passed": False, "finite": finite, "same_shape": same_shape}
    cosine = float(np.dot(actual, reference) / (np.linalg.norm(actual) * np.linalg.norm(reference)))
    norm = float(np.linalg.norm(actual))
    return {
        "passed": bool(cosine >= 0.99 and abs(norm - 1) < 1e-4),
        "finite": True,
        "cosine_similarity": cosine,
        "max_absolute_error": float(np.abs(actual - reference).max()),
        "norm": norm,
    }


def fixture_requests():
    y, x = np.mgrid[:120, :240]
    array = np.stack([(x % 256), (y * 2 % 256), ((x + y) % 256)], axis=-1).astype(np.uint8)
    image = Image.fromarray(array)
    wave = np.sin(np.arange(8000, dtype=np.float64) * (2 * np.pi * 220 / 16000)).astype(np.float32)
    long_wave = np.sin(np.arange(17600, dtype=np.float64) * (2 * np.pi * 440 / 16000)).astype(np.float32)
    return [
        ("english", {"text": "What causes the northern lights?", "prompt": "SearchQuery"}),
        ("korean", {"text": "오로라가 생기는 원인은 무엇인가요?", "prompt": "SearchQuery"}),
        ("document", {"text": "Charged particles from the sun cause the northern lights.", "prompt": "Document"}),
        ("image_wide", {"images": [image]}),
        ("image_tall", {"images": [image.transpose(Image.Transpose.ROTATE_90)]}),
        ("audio_short", {"audio": [wave]}),
        ("audio_long", {"audio": [long_wave]}),
        ("mixed", {"text": "Picture <|image|> and sound <|audio|>", "images": [image], "audio": [wave]}),
    ]


def input_hash(inputs):
    digest = hashlib.sha256()
    for name, tensor in sorted(inputs.items()):
        digest.update(name.encode())
        digest.update(str((tensor.dtype, list(tensor.shape))).encode())
        digest.update(tensor.contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def main():
    import argparse

    import rebel
    import soundfile as sf
    from transformers import AutoModel, AutoProcessor

    from optimum.rbln import RBLNAutoModel

    parser = argparse.ArgumentParser(description="Official-reference parity for native multimodal embeddings")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--requests", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-validation", type=Path)
    parser.add_argument("--devices", nargs="+", type=int, default=[0])
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a new validation output path")
    torch.set_num_threads(4)
    controls = rebel if hasattr(rebel, "set_num_threads") else rebel._C
    controls.set_num_threads(4)
    model = RBLNAutoModel.from_pretrained(args.model, export=False, rbln_device=args.devices)
    processor = AutoProcessor.from_pretrained(args.model)
    reference = AutoModel.from_pretrained(
        "google/embeddinggemma-2", revision=MODEL_REVISION, dtype=model.dtype, attn_implementation="eager"
    ).eval()
    requests = dict(fixture_requests())
    image, wave = requests["image_wide"]["images"][0], requests["audio_short"]["audio"][0]
    if args.requests:
        public = json.loads(args.requests.read_text())
        requests.update(public)
        if "speech" in public:
            speech, rate = sf.read(public["speech"]["audio"][0], dtype="float32", always_2d=True)
            if rate != 16000 or len(speech) <= 8 * rate:
                raise ValueError("Dynamic fixtures require >8 seconds of 16kHz speech")
            speech = speech.mean(-1)
            requests.update(
                speech_truncated_8s={"audio": [speech[: 8 * rate].copy()]},
                speech_reversed={"audio": [speech[::-1].copy()]},
            )
            if "mixed_real" in public:
                requests["mixed_photo_truncated_8s"] = {**public["mixed_real"], "audio": [speech[: 8 * rate].copy()]}
    requests.update(
        two_images={"images": [image, image]},
        two_audio={"audio": [wave, wave]},
        mixed_media_only={"images": [image], "audio": [wave]},
    )
    results = []
    try:
        for name, request in requests.items():
            inputs = model.prepare(request, processor=processor)
            states = reference(**inputs).last_hidden_state.float()
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            expected = torch.nn.functional.normalize((states * mask).sum(1) / mask.sum(1), dim=-1)[0].numpy()
            actual = model(**inputs).embeddings[0].numpy()
            metrics = embedding_metrics(actual, expected)
            stages = {"text"}
            if "pixel_values" in inputs:
                stages.add("vision")
            if "input_features" in inputs:
                stages.add("audio")
            metrics["passed"] &= model.engine.last_stages == dict.fromkeys(stages, "rbln")
            results.append(
                {
                    "name": name,
                    "input_sha256": input_hash(inputs),
                    "graphs": model.engine.last_graphs,
                    "stages": model.engine.last_stages.copy(),
                    **metrics,
                }
            )
            print(json.dumps({"name": name, **metrics}), flush=True)
        for dimensions in (128, 256, 512):
            actual = model.encode({"text": "Embedding dimension check"}, processor=processor, dimensions=dimensions)
            results.append(
                {
                    "name": f"dimensions_{dimensions}",
                    "passed": actual.shape == (1, dimensions)
                    and bool(torch.isfinite(actual).all())
                    and abs(float(actual.norm()) - 1) < 1e-4,
                }
            )
        try:
            model(input_ids=torch.full((1, max(model.rbln_config.text_buckets) + 1), 2, dtype=torch.long))
            passed = False
        except ValueError:
            passed = not model.engine.last_stages
        results.append({"name": "over_capacity_rejected_before_execution", "passed": passed})
        report = {
            "passed": all(item["passed"] for item in results),
            "implementation": "native optimum-rbln",
            "model_revision": MODEL_REVISION,
            "versions": package_versions(),
            "cases": results,
            "npu": model.rbln_config.npu,
            "num_devices": model.rbln_config.num_devices,
            "host_parameters": sum(p.numel() for p in model.host_model.parameters()),
            "reference_parameters": sum(p.numel() for p in reference.parameters()),
        }
        if args.baseline_validation:
            old = json.loads(args.baseline_validation.read_text())
            expected_hashes = {case["name"]: case["input_sha256"] for case in old["cases"] if "input_sha256" in case}
            actual_hashes = {case["name"]: case["input_sha256"] for case in results if "input_sha256" in case}
            report["baseline_input_hashes_match"] = all(
                actual_hashes.get(name) == value for name, value in expected_hashes.items()
            )
            report["passed"] &= report["baseline_input_hashes_match"]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        if not report["passed"]:
            raise RuntimeError("Native validation failed; see JSON report")
    finally:
        model.close()


if __name__ == "__main__":
    main()

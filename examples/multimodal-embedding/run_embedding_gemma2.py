"""Native compile/import/run and separately timed host profiling."""

import argparse
import json
import math
import os
import statistics
import subprocess
from pathlib import Path
from time import perf_counter_ns


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=4)
    sub = parser.add_subparsers(dest="command", required=True)
    for mode in ("compile", "import"):
        task = sub.add_parser(mode)
        task.add_argument("--devices", type=int, nargs="+", default=[0])
        task.add_argument("--output", type=Path, required=True)
        task.add_argument("--revision", default="914f7f89142e33e77833254d9c9b90c3cef7303b")
        task.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
        if mode == "import":
            task.add_argument("--bundle", type=Path, required=True)
        else:
            task.add_argument("--npu", required=True)
            task.add_argument("--audio-layout", choices=["row", "selector"], default="row")
            task.add_argument("--text-buckets", nargs="+", type=int, default=[32, 64, 96, 128, 288, 352, 512])
            task.add_argument("--audio-buckets", nargs="+", type=int, default=[64, 128, 1056, 1536])
    for mode in ("run", "benchmark"):
        task = sub.add_parser(mode)
        task.add_argument("--devices", type=int, nargs="+", default=[0])
        task.add_argument("--model", type=Path, required=True)
        task.add_argument("--requests", type=Path, required=True, help="JSON object mapping names to raw requests")
        if mode == "benchmark":
            task.add_argument("--output", type=Path, required=True)
            task.add_argument("--warmup", type=int, default=3)
            task.add_argument("--repeats", type=int, default=100)
            task.add_argument("--trace", action="store_true", help="Additional pass, excluded from latency samples")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.threads < 1 or (args.command == "benchmark" and (args.repeats < 1 or args.warmup < 0)):
        parser.error("threads/repeats must be positive and warmup nonnegative")
    os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    import rebel
    import torch
    from transformers import AutoModel, AutoProcessor

    from optimum.rbln import RBLNAutoModel, RBLNEmbeddingGemma2Model
    from optimum.rbln.transformers.models.embedding_gemma2.embedding_gemma2_profiling import TraceSession, trace_span
    from optimum.rbln.transformers.models.embedding_gemma2.embedding_gemma2_utils import package_versions

    torch.set_num_threads(args.threads)
    controls = rebel if hasattr(rebel, "set_num_threads") else rebel._C
    controls.set_num_threads(args.threads)
    if args.command in ("compile", "import"):
        if args.output.exists():
            raise ValueError("Use a new model output directory")
        source = AutoModel.from_pretrained(
            "google/embeddinggemma-2",
            revision=args.revision,
            dtype=getattr(torch, args.dtype),
            attn_implementation="eager",
        ).eval()
        processor = AutoProcessor.from_pretrained("google/embeddinggemma-2", revision=args.revision)
        if args.command == "import":
            model = RBLNEmbeddingGemma2Model.from_legacy_bundle(
                args.bundle,
                source,
                args.output,
                processor=processor,
                device=args.devices,
                source_revision=args.revision,
            )
        else:
            model = RBLNEmbeddingGemma2Model.from_model(
                source,
                preprocessors=[processor],
                model_save_dir=args.output,
                rbln_config={
                    "npu": args.npu,
                    "num_devices": len(args.devices),
                    "device": args.devices,
                    "audio_layout": args.audio_layout,
                    "text_buckets": args.text_buckets,
                    "audio_buckets": args.audio_buckets,
                    "image_soft_tokens": 70,
                },
            )
        print(
            json.dumps(
                {"saved": str(args.output), "npu": model.rbln_config.npu, "num_devices": model.rbln_config.num_devices}
            )
        )
        return
    if args.command == "benchmark":
        if args.output.exists():
            raise ValueError("Use a new benchmark output directory")
        for flag in ("RBLN_PROFILER", "RBLN_RUNTIME_TIMER"):
            if os.environ.get(flag, "0").lower() in ("1", "true", "on", "yes") or bool(getattr(rebel.flags, flag)):
                raise ValueError(f"Disable {flag} for uninstrumented benchmark samples")
    model = RBLNAutoModel.from_pretrained(args.model, export=False, rbln_device=args.devices)
    processor = AutoProcessor.from_pretrained(args.model)
    requests = json.loads(args.requests.read_text())
    try:
        if args.command == "run":
            print(
                json.dumps(
                    {name: model.encode(request, processor=processor).tolist() for name, request in requests.items()}
                )
            )
            return
        args.output.mkdir(parents=True)
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            text=True,
        ).strip()
        report = {
            "git_commit": git_commit,
            "implementation": "native optimum-rbln",
            "versions": package_versions(),
            "npu": model.rbln_config.npu,
            "num_devices": model.rbln_config.num_devices,
            "devices": args.devices,
            "torch_threads": torch.get_num_threads(),
            "sdk_threads": controls.get_num_threads(),
            "warmup": args.warmup,
            "repeats": args.repeats,
            "measurement": "CPU decode/processor -> synchronous native forward -> normalized CPU embedding; excludes model load, compile, warmup and tracing",
            "cases": {},
        }
        for name, request in requests.items():
            for _ in range(args.warmup):
                model.encode(request, processor=processor)
            samples = []
            for _ in range(args.repeats):
                start = perf_counter_ns()
                inputs = model.prepare(request, processor=processor)
                ready = perf_counter_ns()
                model(**inputs)
                done = perf_counter_ns()
                samples.append(
                    {
                        "prepare_ms": (ready - start) / 1e6,
                        "encode_ms": (done - ready) / 1e6,
                        "total_ms": (done - start) / 1e6,
                    }
                )
            summary = {}
            for field in samples[0]:
                values = sorted(sample[field] for sample in samples)
                summary[field] = {
                    "median": statistics.median(values),
                    "p95": values[math.ceil(0.95 * len(values)) - 1],
                }
            report["cases"][name] = {
                "summary_ms": summary,
                "samples_ms": samples,
                "graphs": model.engine.last_graphs,
                "stages": model.engine.last_stages,
            }
        (args.output / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
        if args.trace:
            with TraceSession(args.output / "host-trace.json"):
                for name, request in requests.items():
                    with trace_span("request", case=name):
                        model.encode(request, processor=processor)
        print(json.dumps({name: case["summary_ms"] for name, case in report["cases"].items()}, indent=2))
    finally:
        model.close()


if __name__ == "__main__":
    main()

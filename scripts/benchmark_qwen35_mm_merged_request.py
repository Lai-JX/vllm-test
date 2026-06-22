#!/usr/bin/env python3
"""Benchmark Qwen3.5 multimodal inference by merging a logical batch into one request.

This script is complementary to benchmark_qwen35_mm.py. Instead of sending N
requests in one offline batch, it builds one prompt containing N image/text
segments and sends exactly one vLLM request. This reduces scheduler effects when
comparing ViT/LLM/model forward time for a logical batch.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

# Must be set before importing vLLM.
os.environ.setdefault("VLLM_PLUGINS", "qwen35_custom_model")
os.environ.setdefault("QWEN35_PLUGIN_PROFILE", "1")
os.environ.setdefault("QWEN35_PLUGIN_PROFILE_SYNC", "1")

import torch  # noqa: E402
from PIL import Image  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

ALPAMAYO_MIN_PIXELS = 163840
ALPAMAYO_MAX_PIXELS = 196608

VISION_TOKEN = "<|vision_start|><|image_pad|><|vision_end|>"
ASSISTANT_MARKER = "<|im_end|>\n<|im_start|>assistant\n"
VLLM_PYTHON_PROFILE_PHASES = (
    ("vllm_generate", "vllm:LLM.generate"),
    ("vllm_run_completion", "vllm:LLM._run_completion"),
    ("vllm_add_completion_requests", "vllm:LLM._add_completion_requests"),
    ("vllm_render_and_add_requests", "vllm:LLM._render_and_add_requests"),
    ("vllm_preprocess_cmpl_one", "vllm:LLM._preprocess_cmpl_one"),
    ("vllm_preprocess_cmpl", "vllm:LLM._preprocess_cmpl"),
    ("vllm_renderer_render_cmpl", "vllm:BaseRenderer.render_cmpl"),
    ("vllm_renderer_render_prompts", "vllm:BaseRenderer.render_prompts"),
    ("vllm_renderer_tokenize_prompts", "vllm:BaseRenderer.tokenize_prompts"),
    ("vllm_renderer_process_for_engine", "vllm:BaseRenderer.process_for_engine"),
    ("vllm_renderer_process_tokens", "vllm:BaseRenderer._process_tokens"),
    ("vllm_renderer_process_multimodal", "vllm:BaseRenderer._process_multimodal"),
    ("vllm_add_request", "vllm:LLM._add_request"),
    ("vllm_run_engine", "vllm:LLM._run_engine"),
    ("vllm_engine_add_request", "vllm:LLMEngine.add_request"),
    ("vllm_engine_step", "vllm:LLMEngine.step"),
    ("vllm_output_add_request", "vllm:OutputProcessor.add_request"),
    ("vllm_process_outputs", "vllm:OutputProcessor.process_outputs"),
    ("vllm_worker_execute_model", "vllm:Worker.execute_model"),
    ("vllm_gpu_execute_model", "vllm:GPUModelRunner.execute_model"),
    ("vllm_execute_mm_encoder", "vllm:GPUModelRunner._execute_mm_encoder"),
    ("vllm_gather_mm_embeddings", "vllm:GPUModelRunner._gather_mm_embeddings"),
)


def parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/share/models/Qwen3.5-2B")
    parser.add_argument(
        "--dataset-jsonl",
        default=None,
        help=(
            "Optional JSONL dataset. Each line should contain image path(s) "
            "(`image`, `image_path`, `image_file`, `images`, `image_paths`, "
            "or `image_files`) and either `text`, `question`, `query`, or "
            "`prompt`."
        ),
    )
    parser.add_argument("--dataset-limit", type=int, default=0, help="0 means use all dataset rows.")
    parser.add_argument(
        "--input-lens",
        default="128",
        help="Synthetic mode: comma-separated approximate text token counts per logical item.",
    )
    parser.add_argument(
        "--input-mode",
        choices=("tokenized", "text"),
        default="text",
        help=(
            "Send the merged request as prompt_token_ids or prompt text. "
            "Dataset rows keep their original text/images; tokenized mode only "
            "changes the final vLLM request input representation."
        ),
    )
    parser.add_argument(
        "--batch-sizes",
        default="1,2,4,8",
        help="Comma-separated logical batch sizes to merge into one request.",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-num-batched-tokens", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument(
        "--disable-enforce-eager",
        action="store_true",
        help="Do not pass enforce_eager=True to vLLM.",
    )
    parser.add_argument(
        "--enable-prefix-caching",
        action="store_true",
        help="Enable vLLM prefix caching. Benchmark mode disables it by default.",
    )
    parser.add_argument(
        "--mm-min-pixels",
        type=int,
        default=ALPAMAYO_MIN_PIXELS,
        help="Minimum image pixels for the HF multimodal processor.",
    )
    parser.add_argument(
        "--mm-max-pixels",
        type=int,
        default=ALPAMAYO_MAX_PIXELS,
        help="Maximum image pixels for the HF multimodal processor.",
    )
    parser.add_argument(
        "--disable-mm-do-rescale",
        action="store_true",
        help="Do not pass do_rescale=True to the multimodal processor.",
    )
    parser.add_argument(
        "--enable-mm-embeds",
        action="store_true",
        help=(
            "Use precomputed image embeddings from dataset JSONL rows instead "
            "of raw images. Each row must contain `image_embeds` and "
            "`image_grid_thw` paths to .pt files."
        ),
    )
    parser.add_argument("--image", default=None, help="Synthetic fallback image path.")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--output-dir",
        default="/workspace/project/RL-learning/vllm-test/outputs/benchmark_qwen35_mm_merged_request",
    )
    parser.add_argument("--profile-jsonl", default=None)
    parser.add_argument("--csv", default=None)
    parser.add_argument("--aggregate-csv", default=None)
    parser.add_argument(
        "--enable-vllm-python-profile",
        action="store_true",
        help=(
            "Enable lightweight vLLM Python wall-time profiling through the "
            "Qwen3.5 plugin. Records are appended to profile_records.jsonl and "
            "selected phase sums are written to summary.csv."
        ),
    )
    parser.add_argument(
        "--enable-torch-profile",
        action="store_true",
        help=(
            "Enable vLLM's built-in torch profiler. Traces are saved under "
            "<output-dir>/torch_profile."
        ),
    )
    parser.add_argument(
        "--torch-profile-record-shapes",
        action="store_true",
        help="Record tensor shapes in torch profiler traces.",
    )
    parser.add_argument(
        "--torch-profile-with-stack",
        action="store_true",
        help="Record Python stack traces in torch profiler traces.",
    )
    parser.add_argument(
        "--torch-profile-with-memory",
        action="store_true",
        help="Record memory usage in torch profiler traces.",
    )
    parser.add_argument(
        "--torch-profile-with-flops",
        action="store_true",
        help="Estimate FLOPs in torch profiler traces.",
    )
    parser.add_argument(
        "--otlp-traces-endpoint",
        default=None,
        help=(
            "Enable vLLM OpenTelemetry tracing and export traces to this OTLP "
            "endpoint, for example http://localhost:4317."
        ),
    )
    parser.add_argument(
        "--collect-detailed-traces",
        choices=("model", "worker", "all"),
        default=None,
        help=(
            "Collect vLLM detailed traces for model, worker, or all modules. "
            "Requires --otlp-traces-endpoint. Defaults to all when an endpoint "
            "is provided."
        ),
    )
    parser.add_argument(
        "--otlp-traces-protocol",
        choices=("grpc", "http/protobuf"),
        default=None,
        help=(
            "Optional OTLP trace exporter protocol. If omitted, vLLM/OpenTelemetry "
            "uses its default, usually grpc."
        ),
    )
    args = parser.parse_args()
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.collect_detailed_traces and not args.otlp_traces_endpoint:
        parser.error("--collect-detailed-traces requires --otlp-traces-endpoint")
    if args.enable_mm_embeds and not args.dataset_jsonl:
        parser.error("--enable-mm-embeds requires --dataset-jsonl")
    return args


def write_run_config(path: Path, args: argparse.Namespace, input_lens: list[int], batch_sizes: list[int]) -> None:
    config = dict(vars(args))
    config["input_lens_parsed"] = input_lens
    config["batch_sizes_parsed"] = batch_sizes
    config["request_mode"] = "merged_logical_batch"
    config["input_mode_effective"] = args.input_mode
    config["enable_mm_embeds"] = args.enable_mm_embeds
    config["vllm_python_profile_enabled"] = args.enable_vllm_python_profile
    config["torch_profile_dir"] = (
        str(Path(args.output_dir) / "torch_profile")
        if args.enable_torch_profile
        else None
    )
    config["torch_profile_scope"] = (
        "measured_only" if args.enable_torch_profile else None
    )
    config["otel_enabled"] = args.otlp_traces_endpoint is not None
    config["otel_collect_detailed_traces_effective"] = (
        args.collect_detailed_traces or "all"
        if args.otlp_traces_endpoint
        else None
    )
    config["disable_log_stats_effective"] = (
        False if args.otlp_traces_endpoint else None
    )
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_image(path: str | None, size: int) -> Image.Image:
    if path:
        return Image.open(path).convert("RGB")
    image = Image.new("RGB", (size, size), color=(128, 128, 128))
    for i in range(min(size, 32)):
        image.putpixel((i, i), (255, 64, 64))
    return image


def clone_image(image: Image.Image) -> Image.Image:
    return image.copy()


def clone_images(images: list[Image.Image]) -> list[Image.Image]:
    return [clone_image(image) for image in images]


def _encode_no_specials(tokenizer, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def build_synthetic_text(tokenizer, target_tokens: int, label: str, nonce_text: str) -> str:
    prefix = f"Item {label}. "
    suffix = f" Answer based on this image. {nonce_text}"
    fixed_tokens = len(_encode_no_specials(tokenizer, prefix + suffix))
    filler_count = max(0, target_tokens - fixed_tokens)
    filler = " ".join(["context"] * filler_count)
    return f"{prefix}{filler}{suffix}".strip()


def _resolve_image_paths(dataset_path: Path, record: dict[str, Any]) -> list[str]:
    raw = (
        record.get("images")
        or record.get("image_paths")
        or record.get("image_files")
        or record.get("image")
        or record.get("image_path")
        or record.get("image_file")
    )
    if not raw:
        raise ValueError("dataset row is missing image(s)/image_path(s)/image_file(s)")
    raw_paths = raw if isinstance(raw, list) else [raw]
    paths = []
    for item in raw_paths:
        path = Path(str(item))
        if not path.is_absolute():
            path = dataset_path.parent / path
        paths.append(str(path))
    return paths


def _resolve_pt_path(dataset_path: Path, raw_path: Any, field_name: str) -> str:
    if not raw_path:
        raise ValueError(f"dataset row is missing {field_name}")
    path = Path(str(raw_path))
    if not path.is_absolute():
        path = dataset_path.parent / path
    return str(path)


def _dataset_text(record: dict[str, Any]) -> str:
    text = (
        record.get("text")
        or record.get("question")
        or record.get("query")
        or record.get("prompt")
    )
    if text is None:
        raise ValueError("dataset row needs text/question/query/prompt")
    return str(text)


def infer_dataset_max_images(
    dataset_jsonl: str,
    limit: int,
    *,
    enable_mm_embeds: bool = False,
) -> int:
    dataset_path = Path(dataset_jsonl)
    max_images = 1
    seen = 0
    with dataset_path.open("r", encoding="utf-8") as f:
        for line in f:
            if limit > 0 and seen >= limit:
                break
            if not line.strip():
                continue
            record = json.loads(line)
            if enable_mm_embeds:
                image_grid_thw = torch.load(
                    _resolve_pt_path(dataset_path, record.get("image_grid_thw"), "image_grid_thw"),
                    map_location="cpu",
                )
                max_images = max(max_images, int(image_grid_thw.shape[0]))
            else:
                max_images = max(max_images, len(_resolve_image_paths(dataset_path, record)))
            seen += 1
    return max_images


def load_dataset_samples(
    dataset_jsonl: str,
    limit: int,
    *,
    enable_mm_embeds: bool = False,
) -> list[dict[str, Any]]:
    dataset_path = Path(dataset_jsonl)
    samples: list[dict[str, Any]] = []
    with dataset_path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if limit > 0 and len(samples) >= limit:
                break
            if not line.strip():
                continue
            record = json.loads(line)
            sample = {
                "cid": str(record.get("id") or record.get("cid") or line_idx),
                "text": _dataset_text(record),
            }
            if enable_mm_embeds:
                image_embeds = torch.load(
                    _resolve_pt_path(dataset_path, record.get("image_embeds"), "image_embeds"),
                    map_location="cpu",
                )
                image_grid_thw = torch.load(
                    _resolve_pt_path(dataset_path, record.get("image_grid_thw"), "image_grid_thw"),
                    map_location="cpu",
                )
                if image_grid_thw.ndim != 2:
                    raise ValueError(
                        "image_grid_thw must have shape (num_images, 3), "
                        f"got {tuple(image_grid_thw.shape)}"
                    )
                sample.update({
                    "image_embeds": image_embeds,
                    "image_grid_thw": image_grid_thw,
                    "image_count": int(image_grid_thw.shape[0]),
                })
            else:
                image_paths = _resolve_image_paths(dataset_path, record)
                sample.update({
                    "images": [load_image(path, size=0) for path in image_paths],
                    "image_count": len(image_paths),
                })
            samples.append(sample)
    if not samples:
        raise ValueError(f"No samples loaded from {dataset_jsonl}")
    return samples


def build_synthetic_samples(
    tokenizer,
    image: Image.Image,
    target_len: int,
    count: int,
) -> list[dict[str, Any]]:
    samples = []
    for idx in range(count):
        item_id = f"synthetic-len{target_len}-item{idx}"
        samples.append({
            "cid": item_id,
            "text": build_synthetic_text(tokenizer, target_len, str(idx), ""),
            "images": [clone_image(image)],
            "image_count": 1,
        })
    return samples


def iter_batches(samples: list[dict[str, Any]], batch_size: int) -> Iterable[tuple[int, list[dict[str, Any]]]]:
    for group_idx, start in enumerate(range(0, len(samples), batch_size)):
        yield group_idx, samples[start:start + batch_size]


def _extract_user_content(text: str) -> str:
    user_marker = "<|im_start|>user\n"
    if user_marker in text:
        text = text.split(user_marker, 1)[1]
    if ASSISTANT_MARKER in text:
        text = text.split(ASSISTANT_MARKER, 1)[0]
    return text.strip()


def build_merged_prompt(samples: list[dict[str, Any]]) -> str:
    parts = [
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n",
        "<|im_start|>user\n",
        "The following items were originally separate requests. Answer each item briefly.\n",
    ]
    for idx, sample in enumerate(samples):
        item_text = _extract_user_content(str(sample["text"]))
        if VISION_TOKEN not in item_text:
            item_text = f"{VISION_TOKEN * int(sample['image_count'])}{item_text}"
        parts.append(f"\nItem {idx}:\n{item_text}\n")
    parts.append(ASSISTANT_MARKER)
    return "".join(parts)


def read_profile_records(path: Path, start_offset: int) -> tuple[list[dict[str, Any]], int]:
    if not path.exists():
        return [], start_offset
    with path.open("r", encoding="utf-8") as f:
        f.seek(start_offset)
        lines = f.readlines()
        end_offset = f.tell()
    records = [json.loads(line) for line in lines if line.strip()]
    return records, end_offset


def phase_sum(records: list[dict[str, Any]], phase: str, key: str = "cuda_ms") -> float:
    values = [r.get(key) for r in records if r.get("phase") == phase]
    return sum(float(v) for v in values if v is not None)


def phase_first(records: list[dict[str, Any]], phase: str, key: str = "cuda_ms") -> float:
    for record in records:
        if record.get("phase") != phase:
            continue
        value = record.get(key)
        if value is not None:
            return float(value)
    return 0.0


def phase_count(records: list[dict[str, Any]], phase: str) -> int:
    return sum(1 for r in records if r.get("phase") == phase)


def maybe_prompt_token_count(output) -> int | None:
    token_ids = getattr(output, "prompt_token_ids", None)
    if token_ids is None:
        return None
    return len(token_ids)


def completion_token_count(output) -> int:
    total = 0
    for completion in getattr(output, "outputs", []) or []:
        total += len(getattr(completion, "token_ids", []) or [])
    return total


def summarize_output(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, (list, tuple)):
        return f"{type(value).__name__}(len={len(value)})"
    if isinstance(value, dict):
        return f"dict(len={len(value)})"
    shape = getattr(value, "shape", None)
    if shape is not None:
        return f"{type(value).__name__}(shape={tuple(shape)})"
    return type(value).__name__


def write_benchmark_profile_record(
    profile_path: Path,
    *,
    phase: str,
    request_id: str,
    start_perf_counter: float,
    end_perf_counter: float,
    start_unix: float,
    end_unix: float,
    output: Any = None,
) -> None:
    record = {
        "phase": phase,
        "request_id": request_id,
        "pid": os.getpid(),
        "wall_ms": (end_perf_counter - start_perf_counter) * 1000.0,
        "cuda_ms": None,
        "output": summarize_output(output),
        "start_perf_counter": start_perf_counter,
        "end_perf_counter": end_perf_counter,
        "start_unix": start_unix,
        "end_unix": end_unix,
        "time_unix": time.time(),
    }
    with profile_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True) + "\n")


def benchmark_phase_sum(
    records: list[dict[str, Any]],
    phase: str,
) -> float:
    return phase_sum(records, phase, key="wall_ms")


def torch_cuda_synchronize() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def enable_vllm_python_profile() -> None:
    os.environ["QWEN35_PLUGIN_PROFILE_VLLM"] = "1"
    try:
        from qwen35_vllm_plugin import _install_vllm_profile_patches

        _install_vllm_profile_patches()
    except Exception as exc:
        raise RuntimeError("Failed to enable Qwen3.5 vLLM Python profiling") from exc


def run_one_merged_batch(
    *,
    llm: LLM,
    tokenizer,
    samples: list[dict[str, Any]],
    sampling_params: SamplingParams,
    profile_path: Path,
    profile_offset: int,
    logical_batch_size: int,
    target_prompt_tokens: int | str,
    repeat_idx: int,
    group_idx: int,
    is_warmup: bool,
    source: str,
    input_mode: str,
    enable_mm_embeds: bool,
) -> tuple[dict[str, Any], int]:
    group_id = f"merged_{source}_bs{logical_batch_size}_g{group_idx}_r{repeat_idx}_{uuid.uuid4().hex[:8]}"
    os.environ["QWEN35_PLUGIN_REQUEST_ID"] = group_id
    run_start_perf = time.perf_counter()
    run_start_unix = time.time()

    phase_start = time.perf_counter()
    phase_start_unix = time.time()
    prompt = build_merged_prompt(samples)
    phase_end = time.perf_counter()
    phase_end_unix = time.time()
    write_benchmark_profile_record(
        profile_path,
        phase="benchmark:build_prompt",
        request_id=group_id,
        start_perf_counter=phase_start,
        end_perf_counter=phase_end,
        start_unix=phase_start_unix,
        end_unix=phase_end_unix,
        output=prompt,
    )

    mm_image_data: Any
    image_uuids: list[str] = []
    phase_start = time.perf_counter()
    phase_start_unix = time.time()
    if enable_mm_embeds:
        image_embeds = torch.cat(
            [sample["image_embeds"] for sample in samples],
            dim=0,
        )
        image_grid_thw = torch.cat(
            [sample["image_grid_thw"] for sample in samples],
            dim=0,
        )
        mm_image_data = {
            "image_embeds": image_embeds,
            "image_grid_thw": image_grid_thw,
        }
        for sample_idx, sample in enumerate(samples):
            image_uuids.extend(
                f"{group_id}_sample{sample_idx}_image{image_idx}_{sample['cid']}"
                for image_idx in range(int(sample["image_count"]))
            )
    else:
        images: list[Image.Image] = []
        for sample_idx, sample in enumerate(samples):
            sample_images = sample["images"]
            images.extend(clone_images(sample_images))
            image_uuids.extend(
                f"{group_id}_sample{sample_idx}_image{image_idx}_{sample['cid']}"
                for image_idx in range(len(sample_images))
            )
        mm_image_data = images if len(images) > 1 else images[0]
    phase_end = time.perf_counter()
    phase_end_unix = time.time()
    write_benchmark_profile_record(
        profile_path,
        phase="benchmark:prepare_mm_data",
        request_id=group_id,
        start_perf_counter=phase_start,
        end_perf_counter=phase_end,
        start_unix=phase_start_unix,
        end_unix=phase_end_unix,
        output=mm_image_data,
    )

    request: dict[str, Any]
    if input_mode == "tokenized":
        phase_start = time.perf_counter()
        phase_start_unix = time.time()
        prompt_token_ids = _encode_no_specials(tokenizer, prompt)
        phase_end = time.perf_counter()
        phase_end_unix = time.time()
        write_benchmark_profile_record(
            profile_path,
            phase="benchmark:encode_prompt",
            request_id=group_id,
            start_perf_counter=phase_start,
            end_perf_counter=phase_end,
            start_unix=phase_start_unix,
            end_unix=phase_end_unix,
            output=prompt_token_ids,
        )
        request = {"prompt_token_ids": prompt_token_ids}
    else:
        request = {"prompt": prompt}

    phase_start = time.perf_counter()
    phase_start_unix = time.time()
    request.update({
        "multi_modal_data": {"image": mm_image_data},
        "multi_modal_uuids": {
            "image": image_uuids if len(image_uuids) > 1 else image_uuids[0]
        },
    })
    phase_end = time.perf_counter()
    phase_end_unix = time.time()
    write_benchmark_profile_record(
        profile_path,
        phase="benchmark:build_request",
        request_id=group_id,
        start_perf_counter=phase_start,
        end_perf_counter=phase_end,
        start_unix=phase_start_unix,
        end_unix=phase_end_unix,
        output=request,
    )

    t0 = time.perf_counter()
    generate_start_unix = time.time()
    outputs = llm.generate([request], sampling_params=sampling_params, use_tqdm=False)
    generate_end = time.perf_counter()
    generate_end_unix = time.time()
    write_benchmark_profile_record(
        profile_path,
        phase="benchmark:llm_generate",
        request_id=group_id,
        start_perf_counter=t0,
        end_perf_counter=generate_end,
        start_unix=generate_start_unix,
        end_unix=generate_end_unix,
        output=outputs,
    )
    sync_start = time.perf_counter()
    sync_start_unix = time.time()
    torch_cuda_synchronize()
    sync_end = time.perf_counter()
    sync_end_unix = time.time()
    write_benchmark_profile_record(
        profile_path,
        phase="benchmark:cuda_sync",
        request_id=group_id,
        start_perf_counter=sync_start,
        end_perf_counter=sync_end,
        start_unix=sync_start_unix,
        end_unix=sync_end_unix,
        output=None,
    )
    write_benchmark_profile_record(
        profile_path,
        phase="benchmark:run_total",
        request_id=group_id,
        start_perf_counter=run_start_perf,
        end_perf_counter=sync_end,
        start_unix=run_start_unix,
        end_unix=sync_end_unix,
        output=outputs,
    )
    e2e_ms = (sync_end - t0) * 1000.0

    records, profile_offset = read_profile_records(profile_path, profile_offset)
    records = [r for r in records if r.get("request_id") in {"", group_id}]

    prompt_token_counts = [maybe_prompt_token_count(output) for output in outputs]
    total_prompt_tokens = sum(v for v in prompt_token_counts if v is not None)
    vit_cuda_sum_ms = phase_sum(records, "vit_forward")
    llm_cuda_sum_ms = phase_sum(records, "llm_forward")
    qwen_forward_cuda_sum_ms = phase_sum(records, "qwen_forward")
    vit_cuda_first_ms = phase_first(records, "vit_forward")
    llm_cuda_first_ms = phase_first(records, "llm_forward")
    qwen_forward_cuda_first_ms = phase_first(records, "qwen_forward")
    vit_wall_sum_ms = phase_sum(records, "vit_forward", key="wall_ms")
    llm_wall_sum_ms = phase_sum(records, "llm_forward", key="wall_ms")
    qwen_forward_wall_sum_ms = phase_sum(records, "qwen_forward", key="wall_ms")
    vit_wall_first_ms = phase_first(records, "vit_forward", key="wall_ms")
    llm_wall_first_ms = phase_first(records, "llm_forward", key="wall_ms")
    qwen_forward_wall_first_ms = phase_first(records, "qwen_forward", key="wall_ms")
    benchmark_build_prompt_wall_ms = benchmark_phase_sum(
        records, "benchmark:build_prompt"
    )
    benchmark_prepare_mm_data_wall_ms = benchmark_phase_sum(
        records, "benchmark:prepare_mm_data"
    )
    benchmark_encode_prompt_wall_ms = benchmark_phase_sum(
        records, "benchmark:encode_prompt"
    )
    benchmark_build_request_wall_ms = benchmark_phase_sum(
        records, "benchmark:build_request"
    )
    benchmark_llm_generate_wall_ms = benchmark_phase_sum(
        records, "benchmark:llm_generate"
    )
    benchmark_cuda_sync_wall_ms = benchmark_phase_sum(records, "benchmark:cuda_sync")
    benchmark_run_total_wall_ms = benchmark_phase_sum(records, "benchmark:run_total")
    benchmark_pre_generate_wall_ms = (
        benchmark_build_prompt_wall_ms
        + benchmark_prepare_mm_data_wall_ms
        + benchmark_encode_prompt_wall_ms
        + benchmark_build_request_wall_ms
    )
    benchmark_e2e_accounted_wall_ms = (
        benchmark_llm_generate_wall_ms + benchmark_cuda_sync_wall_ms
    )
    benchmark_total_accounted_wall_ms = (
        benchmark_pre_generate_wall_ms + benchmark_e2e_accounted_wall_ms
    )
    vllm_python_metrics: dict[str, float | int] = {}
    for metric_prefix, phase in VLLM_PYTHON_PROFILE_PHASES:
        vllm_python_metrics[f"{metric_prefix}_wall_sum_ms"] = phase_sum(
            records, phase, key="wall_ms"
        )
        vllm_python_metrics[f"{metric_prefix}_wall_first_ms"] = phase_first(
            records, phase, key="wall_ms"
        )
        vllm_python_metrics[f"{metric_prefix}_calls"] = phase_count(records, phase)

    mm_uuids_unique = len(set(image_uuids)) == len(image_uuids)
    unique_mm_uuid_count = len(set(image_uuids))
    mm_uuids = ";".join(image_uuids)

    row = {
        "source": source,
        "request_mode": "merged_logical_batch",
        "input_mode": input_mode,
        "enable_mm_embeds": enable_mm_embeds,
        "benchmark_request_id": group_id,
        "logical_batch_size": logical_batch_size,
        "batch_size": logical_batch_size,
        "target_prompt_tokens": target_prompt_tokens,
        "actual_prompt_tokens": prompt_token_counts[0] if prompt_token_counts else "",
        "repeat_idx": repeat_idx,
        "group_idx": group_idx,
        "is_warmup": is_warmup,
        "request_count": len(outputs),
        "merged_item_count": len(samples),
        "image_count": len(image_uuids),
        "e2e_ms": e2e_ms,
        "per_engine_request_e2e_ms": e2e_ms / max(1, len(outputs)),
        "per_logical_item_e2e_ms": e2e_ms / max(1, len(samples)),
        "engine_requests_per_s": len(outputs) / (e2e_ms / 1000.0) if e2e_ms > 0 else 0.0,
        "logical_items_per_s": len(samples) / (e2e_ms / 1000.0) if e2e_ms > 0 else 0.0,
        "total_prompt_tokens": total_prompt_tokens,
        "prompt_tokens_per_s": total_prompt_tokens / (e2e_ms / 1000.0) if e2e_ms > 0 else 0.0,
        "vit_cuda_sum_ms": vit_cuda_sum_ms,
        "llm_cuda_sum_ms": llm_cuda_sum_ms,
        "qwen_forward_cuda_sum_ms": qwen_forward_cuda_sum_ms,
        "model_total_cuda_sum_ms": vit_cuda_sum_ms + qwen_forward_cuda_sum_ms,
        "vit_cuda_first_ms": vit_cuda_first_ms,
        "llm_cuda_first_ms": llm_cuda_first_ms,
        "qwen_forward_cuda_first_ms": qwen_forward_cuda_first_ms,
        "model_total_cuda_first_ms": vit_cuda_first_ms + qwen_forward_cuda_first_ms,
        "vit_wall_sum_ms": vit_wall_sum_ms,
        "llm_wall_sum_ms": llm_wall_sum_ms,
        "qwen_forward_wall_sum_ms": qwen_forward_wall_sum_ms,
        "model_total_wall_sum_ms": vit_wall_sum_ms + qwen_forward_wall_sum_ms,
        "e2e_minus_model_total_wall_ms": (
            e2e_ms - (vit_wall_sum_ms + qwen_forward_wall_sum_ms)
        ),
        "benchmark_build_prompt_wall_ms": benchmark_build_prompt_wall_ms,
        "benchmark_prepare_mm_data_wall_ms": benchmark_prepare_mm_data_wall_ms,
        "benchmark_clone_images_wall_ms": benchmark_prepare_mm_data_wall_ms,
        "benchmark_encode_prompt_wall_ms": benchmark_encode_prompt_wall_ms,
        "benchmark_build_request_wall_ms": benchmark_build_request_wall_ms,
        "benchmark_pre_generate_wall_ms": benchmark_pre_generate_wall_ms,
        "benchmark_llm_generate_wall_ms": benchmark_llm_generate_wall_ms,
        "benchmark_cuda_sync_wall_ms": benchmark_cuda_sync_wall_ms,
        "benchmark_e2e_accounted_wall_ms": benchmark_e2e_accounted_wall_ms,
        "benchmark_e2e_residual_wall_ms": (
            e2e_ms - benchmark_e2e_accounted_wall_ms
        ),
        "benchmark_run_total_wall_ms": benchmark_run_total_wall_ms,
        "benchmark_total_accounted_wall_ms": benchmark_total_accounted_wall_ms,
        "benchmark_total_residual_wall_ms": (
            benchmark_run_total_wall_ms - benchmark_total_accounted_wall_ms
        ),
        "vit_wall_first_ms": vit_wall_first_ms,
        "llm_wall_first_ms": llm_wall_first_ms,
        "qwen_forward_wall_first_ms": qwen_forward_wall_first_ms,
        "model_total_wall_first_ms": vit_wall_first_ms + qwen_forward_wall_first_ms,
        "vit_calls": phase_count(records, "vit_forward"),
        "llm_calls": phase_count(records, "llm_forward"),
        "qwen_forward_calls": phase_count(records, "qwen_forward"),
        **vllm_python_metrics,
        "completion_tokens": sum(completion_token_count(output) for output in outputs),
        "engine_request_count": len(outputs),
        "mm_uuid_count": len(image_uuids),
        "mm_uuids_unique": mm_uuids_unique,
        "mm_uuids_unique_global": True,
        "duplicate_mm_uuids_global": "",
        "unique_mm_uuid_count": unique_mm_uuid_count,
        "mm_uuids": mm_uuids,
    }
    return row, profile_offset


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_row(row: dict[str, Any]) -> None:
    compact_row = {
        key: value
        for key, value in row.items()
        if key != "mm_uuids"
    }
    print(json.dumps(compact_row, ensure_ascii=False), flush=True)


def start_torch_profile_if_enabled(
    llm: LLM,
    enabled: bool,
    torch_profile_dir: Path,
    profile_prefix: str,
) -> None:
    if not enabled:
        return
    print(f"Torch profiler traces: {torch_profile_dir}", flush=True)
    llm.start_profile(profile_prefix)


def stop_torch_profile_if_enabled(llm: LLM, enabled: bool) -> None:
    if enabled:
        llm.stop_profile()


def main() -> None:
    args = parse_args()
    input_lens = parse_int_list(args.input_lens)
    batch_sizes = parse_int_list(args.batch_sizes)
    if not batch_sizes:
        raise ValueError("--batch-sizes cannot be empty")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_path = Path(args.profile_jsonl or output_dir / "profile_records.jsonl")
    csv_path = Path(args.csv or output_dir / "summary.csv")
    aggregate_csv_path = Path(args.aggregate_csv or output_dir / "aggregate_summary.csv")
    write_run_config(output_dir / "run_config.json", args, input_lens, batch_sizes)
    profile_path.write_text("")
    os.environ["QWEN35_PLUGIN_PROFILE_PATH"] = str(profile_path)
    if args.enable_vllm_python_profile:
        enable_vllm_python_profile()

    max_images_per_item = (
        infer_dataset_max_images(
            args.dataset_jsonl,
            args.dataset_limit,
            enable_mm_embeds=args.enable_mm_embeds,
        )
        if args.dataset_jsonl
        else 1
    )
    max_images = max(batch_sizes) * max_images_per_item
    llm_kwargs: dict[str, Any] = {
        "model": args.model,
        "dtype": "bfloat16",
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "limit_mm_per_prompt": {"image": max_images},
        "mm_processor_kwargs": {
            "do_rescale": not args.disable_mm_do_rescale,
            "min_pixels": args.mm_min_pixels,
            "max_pixels": args.mm_max_pixels,
        },
        "enforce_eager": not args.disable_enforce_eager,
        "enable_prefix_caching": args.enable_prefix_caching,
        "enable_mm_embeds": args.enable_mm_embeds,
    }
    if args.max_num_batched_tokens > 0:
        llm_kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens
    torch_profile_dir = output_dir / "torch_profile"
    if args.enable_torch_profile:
        torch_profile_dir.mkdir(parents=True, exist_ok=True)
        llm_kwargs["profiler_config"] = {
            "profiler": "torch",
            "torch_profiler_dir": str(torch_profile_dir),
            "torch_profiler_record_shapes": args.torch_profile_record_shapes,
            "torch_profiler_with_stack": args.torch_profile_with_stack,
            "torch_profiler_with_memory": args.torch_profile_with_memory,
            "torch_profiler_with_flops": args.torch_profile_with_flops,
        }
    if args.otlp_traces_endpoint:
        if args.otlp_traces_protocol:
            os.environ["OTEL_EXPORTER_OTLP_TRACES_PROTOCOL"] = (
                args.otlp_traces_protocol
            )
        # vLLM's request-level tracing reads RequestState.stats in
        # OutputProcessor.do_tracing(). The offline LLM entrypoint disables
        # stats by default, so tracing needs to re-enable them explicitly.
        llm_kwargs["disable_log_stats"] = False
        llm_kwargs["otlp_traces_endpoint"] = args.otlp_traces_endpoint
        llm_kwargs["collect_detailed_traces"] = [
            args.collect_detailed_traces or "all"
        ]
    llm = LLM(**llm_kwargs)
    profile_offset = profile_path.stat().st_size if profile_path.exists() else 0
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)

    rows: list[dict[str, Any]] = []
    seen_mm_uuids: set[str] = set()

    if args.dataset_jsonl:
        source = "dataset"
        dataset_samples = load_dataset_samples(
            args.dataset_jsonl,
            args.dataset_limit,
            enable_mm_embeds=args.enable_mm_embeds,
        )
        for batch_size in batch_sizes:
            warmup_batch = dataset_samples[:batch_size]
            for warmup_idx in range(args.warmup):
                row, profile_offset = run_one_merged_batch(
                    llm=llm,
                    tokenizer=tokenizer,
                    samples=warmup_batch,
                    sampling_params=sampling_params,
                    profile_path=profile_path,
                    profile_offset=profile_offset,
                    logical_batch_size=batch_size,
                    target_prompt_tokens="dataset",
                    repeat_idx=-1,
                    group_idx=warmup_idx,
                    is_warmup=True,
                    source=source,
                    input_mode=args.input_mode,
                    enable_mm_embeds=args.enable_mm_embeds,
                )
                mm_uuids = [uuid for uuid in row["mm_uuids"].split(";") if uuid]
                duplicates = sorted(set(mm_uuids) & seen_mm_uuids)
                row["mm_uuids_unique_global"] = not duplicates
                row["duplicate_mm_uuids_global"] = ";".join(duplicates)
                if duplicates:
                    raise RuntimeError(f"Duplicate image UUIDs: {duplicates[:5]}")
                seen_mm_uuids.update(mm_uuids)
                rows.append(row)
                print_row(row)

            start_torch_profile_if_enabled(
                llm,
                args.enable_torch_profile,
                torch_profile_dir,
                f"merged_request_dataset_bs{batch_size}",
            )
            try:
                for repeat_idx in range(args.repeats):
                    for group_idx, batch_samples in iter_batches(dataset_samples, batch_size):
                        row, profile_offset = run_one_merged_batch(
                            llm=llm,
                            tokenizer=tokenizer,
                            samples=batch_samples,
                            sampling_params=sampling_params,
                            profile_path=profile_path,
                            profile_offset=profile_offset,
                            logical_batch_size=batch_size,
                            target_prompt_tokens="dataset",
                            repeat_idx=repeat_idx,
                            group_idx=group_idx,
                            is_warmup=False,
                            source=source,
                            input_mode=args.input_mode,
                            enable_mm_embeds=args.enable_mm_embeds,
                        )
                        mm_uuids = [uuid for uuid in row["mm_uuids"].split(";") if uuid]
                        duplicates = sorted(set(mm_uuids) & seen_mm_uuids)
                        row["mm_uuids_unique_global"] = not duplicates
                        row["duplicate_mm_uuids_global"] = ";".join(duplicates)
                        if duplicates:
                            raise RuntimeError(f"Duplicate image UUIDs: {duplicates[:5]}")
                        seen_mm_uuids.update(mm_uuids)
                        rows.append(row)
                        print_row(row)
            finally:
                stop_torch_profile_if_enabled(llm, args.enable_torch_profile)
    else:
        source = "synthetic"
        base_image = load_image(args.image, args.image_size)
        for batch_size in batch_sizes:
            for target_len in input_lens:
                for warmup_idx in range(args.warmup):
                    samples = build_synthetic_samples(
                        tokenizer,
                        base_image,
                        target_len,
                        batch_size,
                    )
                    row, profile_offset = run_one_merged_batch(
                        llm=llm,
                        tokenizer=tokenizer,
                        samples=samples,
                        sampling_params=sampling_params,
                        profile_path=profile_path,
                        profile_offset=profile_offset,
                        logical_batch_size=batch_size,
                        target_prompt_tokens=target_len,
                        repeat_idx=-1,
                        group_idx=warmup_idx,
                        is_warmup=True,
                        source=source,
                        input_mode=args.input_mode,
                        enable_mm_embeds=False,
                    )
                    mm_uuids = [uuid for uuid in row["mm_uuids"].split(";") if uuid]
                    duplicates = sorted(set(mm_uuids) & seen_mm_uuids)
                    row["mm_uuids_unique_global"] = not duplicates
                    row["duplicate_mm_uuids_global"] = ";".join(duplicates)
                    if duplicates:
                        raise RuntimeError(f"Duplicate image UUIDs: {duplicates[:5]}")
                    seen_mm_uuids.update(mm_uuids)
                    rows.append(row)
                    print_row(row)

                start_torch_profile_if_enabled(
                    llm,
                    args.enable_torch_profile,
                    torch_profile_dir,
                    f"merged_request_synthetic_bs{batch_size}_len{target_len}",
                )
                try:
                    for repeat_idx in range(args.repeats):
                        samples = build_synthetic_samples(
                            tokenizer,
                            base_image,
                            target_len,
                            batch_size,
                        )
                        row, profile_offset = run_one_merged_batch(
                            llm=llm,
                            tokenizer=tokenizer,
                            samples=samples,
                            sampling_params=sampling_params,
                            profile_path=profile_path,
                            profile_offset=profile_offset,
                            logical_batch_size=batch_size,
                            target_prompt_tokens=target_len,
                            repeat_idx=repeat_idx,
                            group_idx=0,
                            is_warmup=False,
                            source=source,
                            input_mode=args.input_mode,
                            enable_mm_embeds=False,
                        )
                        mm_uuids = [uuid for uuid in row["mm_uuids"].split(";") if uuid]
                        duplicates = sorted(set(mm_uuids) & seen_mm_uuids)
                        row["mm_uuids_unique_global"] = not duplicates
                        row["duplicate_mm_uuids_global"] = ";".join(duplicates)
                        if duplicates:
                            raise RuntimeError(f"Duplicate image UUIDs: {duplicates[:5]}")
                        seen_mm_uuids.update(mm_uuids)
                        rows.append(row)
                        print_row(row)
                finally:
                    stop_torch_profile_if_enabled(llm, args.enable_torch_profile)

    write_csv(csv_path, rows)

    measured = [row for row in rows if not row["is_warmup"]]
    aggregate_rows: list[dict[str, Any]] = []
    print("\nSummary by logical_batch_size:")
    for batch_size in batch_sizes:
        group = [row for row in measured if row["logical_batch_size"] == batch_size]
        if not group:
            continue

        def avg(key: str) -> float:
            return statistics.mean(float(row[key]) for row in group)

        total_items = sum(int(row["merged_item_count"]) for row in group)
        total_engine_requests = sum(int(row["request_count"]) for row in group)
        total_e2e_ms = sum(float(row["e2e_ms"]) for row in group)
        total_prompt_tokens = sum(int(row["total_prompt_tokens"]) for row in group)
        aggregate_row = {
            "source": group[0]["source"],
            "request_mode": "merged_logical_batch",
            "input_mode": group[0]["input_mode"],
            "logical_batch_size": batch_size,
            "measured_groups": len(group),
            "total_logical_items": total_items,
            "total_engine_requests": total_engine_requests,
            "total_prompt_tokens": total_prompt_tokens,
            "total_e2e_ms": total_e2e_ms,
            "logical_items_per_s": total_items / (total_e2e_ms / 1000.0) if total_e2e_ms > 0 else 0.0,
            "engine_requests_per_s": total_engine_requests / (total_e2e_ms / 1000.0)
            if total_e2e_ms > 0
            else 0.0,
            "prompt_tokens_per_s": total_prompt_tokens / (total_e2e_ms / 1000.0) if total_e2e_ms > 0 else 0.0,
            "avg_e2e_ms": avg("e2e_ms"),
            "avg_per_logical_item_e2e_ms": avg("per_logical_item_e2e_ms"),
            "avg_vit_cuda_sum_ms": avg("vit_cuda_sum_ms"),
            "avg_llm_cuda_sum_ms": avg("llm_cuda_sum_ms"),
            "avg_qwen_forward_cuda_sum_ms": avg("qwen_forward_cuda_sum_ms"),
            "avg_model_total_cuda_sum_ms": avg("model_total_cuda_sum_ms"),
            "avg_vit_cuda_first_ms": avg("vit_cuda_first_ms"),
            "avg_llm_cuda_first_ms": avg("llm_cuda_first_ms"),
            "avg_qwen_forward_cuda_first_ms": avg("qwen_forward_cuda_first_ms"),
            "avg_model_total_cuda_first_ms": avg("model_total_cuda_first_ms"),
            "avg_e2e_minus_model_total_wall_ms": avg(
                "e2e_minus_model_total_wall_ms"
            ),
            "avg_benchmark_build_prompt_wall_ms": avg(
                "benchmark_build_prompt_wall_ms"
            ),
            "avg_benchmark_clone_images_wall_ms": avg(
                "benchmark_clone_images_wall_ms"
            ),
            "avg_benchmark_prepare_mm_data_wall_ms": avg(
                "benchmark_prepare_mm_data_wall_ms"
            ),
            "avg_benchmark_encode_prompt_wall_ms": avg(
                "benchmark_encode_prompt_wall_ms"
            ),
            "avg_benchmark_build_request_wall_ms": avg(
                "benchmark_build_request_wall_ms"
            ),
            "avg_benchmark_pre_generate_wall_ms": avg(
                "benchmark_pre_generate_wall_ms"
            ),
            "avg_benchmark_llm_generate_wall_ms": avg(
                "benchmark_llm_generate_wall_ms"
            ),
            "avg_benchmark_cuda_sync_wall_ms": avg("benchmark_cuda_sync_wall_ms"),
            "avg_benchmark_e2e_accounted_wall_ms": avg(
                "benchmark_e2e_accounted_wall_ms"
            ),
            "avg_benchmark_e2e_residual_wall_ms": avg(
                "benchmark_e2e_residual_wall_ms"
            ),
            "avg_benchmark_run_total_wall_ms": avg("benchmark_run_total_wall_ms"),
            "avg_benchmark_total_accounted_wall_ms": avg(
                "benchmark_total_accounted_wall_ms"
            ),
            "avg_benchmark_total_residual_wall_ms": avg(
                "benchmark_total_residual_wall_ms"
            ),
            "avg_vit_calls": avg("vit_calls"),
            "avg_llm_calls": avg("llm_calls"),
            "avg_qwen_forward_calls": avg("qwen_forward_calls"),
            "mm_uuids_unique_all": all(row["mm_uuids_unique"] is True for row in group),
            "mm_uuids_unique_global_all": all(
                row["mm_uuids_unique_global"] is True for row in group
            ),
        }
        for metric_prefix, _ in VLLM_PYTHON_PROFILE_PHASES:
            aggregate_row[f"avg_{metric_prefix}_wall_sum_ms"] = avg(
                f"{metric_prefix}_wall_sum_ms"
            )
            aggregate_row[f"avg_{metric_prefix}_wall_first_ms"] = avg(
                f"{metric_prefix}_wall_first_ms"
            )
            aggregate_row[f"avg_{metric_prefix}_calls"] = avg(
                f"{metric_prefix}_calls"
            )
        aggregate_rows.append(aggregate_row)
        print(
            f"logical_bs={batch_size}: "
            f"e2e={aggregate_row['avg_e2e_ms']:.2f} ms, "
            f"per_item={aggregate_row['avg_per_logical_item_e2e_ms']:.2f} ms, "
            f"items/s={aggregate_row['logical_items_per_s']:.2f}, "
            f"vit_sum={aggregate_row['avg_vit_cuda_sum_ms']:.2f} ms, "
            f"llm_sum={aggregate_row['avg_llm_cuda_sum_ms']:.2f} ms, "
            f"model_total_sum={aggregate_row['avg_model_total_cuda_sum_ms']:.2f} ms, "
            f"e2e_residual={aggregate_row['avg_benchmark_e2e_residual_wall_ms']:.3f} ms"
        )

    write_csv(aggregate_csv_path, aggregate_rows)
    print(f"\nCSV: {csv_path}")
    print(f"Aggregate CSV: {aggregate_csv_path}")
    print(f"Raw profile JSONL: {profile_path}")


if __name__ == "__main__":
    main()

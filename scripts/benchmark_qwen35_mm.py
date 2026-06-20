#!/usr/bin/env python3
"""Benchmark Qwen3.5 multimodal offline inference with the vLLM plugin.

Primary mode sweeps offline batch size over real dataset samples loaded from
JSONL. The synthetic token-length mode is kept as a fallback for controlled
input-token experiments.

The benchmark records:
- ViT forward time from `self.visual(...)`
- LLM forward time from `self.language_model.model(...)`
- Qwen top-level forward time from `Qwen3_5ForConditionalGeneration.forward(...)`
- End-to-end `llm.generate(...)` latency

Internal timings are emitted by the plugin into a JSONL file and summarized by
this script into CSV output.
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

from PIL import Image  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

ALPAMAYO_MIN_PIXELS = 163840
ALPAMAYO_MAX_PIXELS = 196608


def parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def write_run_config(path: Path, args: argparse.Namespace, input_lens: list[int], batch_sizes: list[int]) -> None:
    config = dict(vars(args))
    config["input_lens_parsed"] = input_lens
    config["batch_sizes_parsed"] = batch_sizes
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/share/models/Qwen3.5-2B")
    parser.add_argument(
        "--dataset-jsonl",
        default=None,
        help=(
            "Optional JSONL dataset. Each line should contain image path(s) "
            "(`image`, `image_path`, `image_file`, `images`, `image_paths`, "
            "or `image_files`) and either `text`, `question`, `prompt`, or "
            "`prompt_token_ids`."
        ),
    )
    parser.add_argument("--dataset-limit", type=int, default=0, help="0 means use all dataset rows.")
    parser.add_argument(
        "--input-lens",
        default="128,256,512,1024,2048",
        help="Synthetic fallback: comma-separated target prompt token counts.",
    )
    parser.add_argument(
        "--batch-sizes",
        default="1",
        help="Comma-separated offline batch sizes to sweep.",
    )
    parser.add_argument(
        "--input-mode",
        choices=("tokenized", "text"),
        default="tokenized",
        help="For synthetic or dataset text rows, send prompt_token_ids or prompt text.",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1, help="Number of warmup batches per sweep point.")
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-num-batched-tokens", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument(
        "--disable-enforce-eager",
        action="store_true",
        help=(
            "Do not pass enforce_eager=True to vLLM. Profiling hooks that "
            "synchronize CUDA are incompatible with CUDA graph capture, so "
            "the default benchmark mode enforces eager execution."
        ),
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
    parser.add_argument("--image", default=None, help="Synthetic fallback image path.")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--output-dir",
        default="/workspace/project/RL-learning/vllm-test/outputs/benchmark_qwen35_mm",
    )
    parser.add_argument("--profile-jsonl", default=None)
    parser.add_argument("--csv", default=None)
    parser.add_argument("--aggregate-csv", default=None)
    args = parser.parse_args()
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    return args


def load_image(path: str | None, size: int) -> Image.Image:
    if path:
        return Image.open(path).convert("RGB")
    image = Image.new("RGB", (size, size), color=(128, 128, 128))
    for i in range(min(size, 32)):
        image.putpixel((i, i), (255, 64, 64))
    return image


def make_qwen3_vl_prompt(question: str) -> str:
    return (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        "<|vision_start|><|image_pad|><|vision_end|>"
        f"{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _encode_no_specials(tokenizer, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def _pick_filler_token_id(tokenizer) -> int:
    special_ids = set(tokenizer.all_special_ids or [])
    for text in (" hello", " test", " data", " a", "0"):
        for token_id in _encode_no_specials(tokenizer, text):
            if token_id not in special_ids:
                return int(token_id)

    vocab = tokenizer.get_vocab()
    for token_id in vocab.values():
        if token_id not in special_ids:
            return int(token_id)
    raise RuntimeError("Tokenizer has no non-special token for prompt padding.")


def build_tokenized_qwen3_vl_prompt_ids(tokenizer, target_tokens: int) -> list[int]:
    prefix = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        "<|vision_start|><|image_pad|><|vision_end|>"
    )
    suffix = "<|im_end|>\n<|im_start|>assistant\n"
    prefix_ids = _encode_no_specials(tokenizer, prefix)
    suffix_ids = _encode_no_specials(tokenizer, suffix)
    base_tokens = len(prefix_ids) + len(suffix_ids)
    if target_tokens < base_tokens:
        raise ValueError(
            f"target prompt length {target_tokens} is shorter than the fixed "
            f"Qwen3-VL chat/image template length {base_tokens}. Use --input-lens >= {base_tokens}."
        )
    filler_token_id = _pick_filler_token_id(tokenizer)
    filler_ids = [filler_token_id] * (target_tokens - base_tokens)
    return prefix_ids + filler_ids + suffix_ids


def build_text_qwen3_vl_prompt(tokenizer, target_tokens: int) -> str:
    prompt_ids = build_tokenized_qwen3_vl_prompt_ids(tokenizer, target_tokens)
    return tokenizer.decode(prompt_ids, skip_special_tokens=False)


def clone_value(value: Any) -> Any:
    if isinstance(value, Image.Image):
        return value.copy()
    if isinstance(value, list):
        return [clone_value(item) for item in value]
    if isinstance(value, dict):
        return {key: clone_value(item) for key, item in value.items()}
    return value


def materialize_request(
    sample: dict[str, Any],
    request_id: str,
) -> dict[str, Any]:
    if "prompt_token_ids" in sample:
        request: dict[str, Any] = {"prompt_token_ids": list(sample["prompt_token_ids"])}
    else:
        request = {"prompt": sample["prompt"]}

    request["multi_modal_data"] = clone_value(sample["multi_modal_data"])
    images = request["multi_modal_data"].get("image")
    if isinstance(images, list):
        request["multi_modal_uuids"] = {
            "image": [f"{request_id}_image{idx}" for idx in range(len(images))]
        }
    else:
        request["multi_modal_uuids"] = {"image": request_id}
    return request


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


def _prompt_from_dataset_record(record: dict[str, Any], image_count: int) -> str:
    if record.get("prompt"):
        return str(record["prompt"])
    text = record.get("text") or record.get("question") or record.get("query")
    if text is None:
        raise ValueError("dataset row needs text/question/query/prompt/prompt_token_ids")
    vision_tokens = "<|vision_start|><|image_pad|><|vision_end|>" * max(1, image_count)
    return (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        f"{vision_tokens}{text}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def infer_dataset_max_images(dataset_jsonl: str, limit: int) -> int:
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
            max_images = max(max_images, len(_resolve_image_paths(dataset_path, record)))
            seen += 1
    return max_images


def load_dataset_samples(
    dataset_jsonl: str,
    tokenizer,
    *,
    input_mode: str,
    limit: int,
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
            image_paths = _resolve_image_paths(dataset_path, record)
            images = [load_image(path, size=0) for path in image_paths]
            sample: dict[str, Any] = {
                "cid": str(record.get("id") or record.get("cid") or line_idx),
                "source": "dataset",
                "multi_modal_data": {"image": images if len(images) > 1 else images[0]},
            }
            if record.get("prompt_token_ids") is not None:
                sample["prompt_token_ids"] = [int(token_id) for token_id in record["prompt_token_ids"]]
            else:
                prompt = _prompt_from_dataset_record(record, len(images))
                sample["prompt_base"] = prompt
                if input_mode == "tokenized":
                    sample["prompt_token_ids"] = _encode_no_specials(tokenizer, prompt)
                else:
                    sample["prompt"] = prompt
            samples.append(sample)
    if not samples:
        raise ValueError(f"No samples loaded from {dataset_jsonl}")
    return samples


def build_synthetic_samples(
    tokenizer,
    image: Image.Image,
    input_lens: Iterable[int],
    input_mode: str,
) -> dict[int, dict[str, Any]]:
    samples_by_len: dict[int, dict[str, Any]] = {}
    for target_len in input_lens:
        if input_mode == "tokenized":
            samples_by_len[target_len] = {
                "cid": f"synthetic-len{target_len}",
                "source": "synthetic",
                "prompt_token_ids": build_tokenized_qwen3_vl_prompt_ids(tokenizer, target_len),
                "multi_modal_data": {"image": image},
            }
        else:
            samples_by_len[target_len] = {
                "cid": f"synthetic-len{target_len}",
                "source": "synthetic",
                "prompt": build_text_qwen3_vl_prompt(tokenizer, target_len),
                "multi_modal_data": {"image": image},
            }
    return samples_by_len


def iter_batches(samples: list[dict[str, Any]], batch_size: int) -> Iterable[tuple[int, list[dict[str, Any]]]]:
    for group_idx, start in enumerate(range(0, len(samples), batch_size)):
        yield group_idx, samples[start:start + batch_size]


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
    numeric = [float(v) for v in values if v is not None]
    return sum(numeric)


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


def torch_cuda_synchronize() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def _token_stat(values: list[int | None], fn, default: Any = "") -> Any:
    numeric = [v for v in values if v is not None]
    if not numeric:
        return default
    return fn(numeric)


def enforce_global_unique_request_ids(
    row: dict[str, Any],
    seen_request_ids: set[str],
) -> None:
    request_ids = [
        request_id
        for request_id in str(row.get("unique_request_ids", "")).split(";")
        if request_id
    ]
    duplicates = sorted(set(request_ids) & seen_request_ids)
    row["request_ids_unique_global"] = not duplicates
    row["duplicate_request_ids_global"] = ";".join(duplicates)
    if duplicates:
        raise RuntimeError(
            "Duplicate benchmark request ids would make cache-hit avoidance "
            f"invalid: {duplicates[:5]}"
        )
    seen_request_ids.update(request_ids)


def run_one_batch(
    *,
    llm: LLM,
    samples: list[dict[str, Any]],
    sampling_params: SamplingParams,
    profile_path: Path,
    profile_offset: int,
    batch_size: int,
    target_prompt_tokens: int | str,
    repeat_idx: int,
    group_idx: int,
    is_warmup: bool,
    source: str,
    input_mode: str,
) -> tuple[dict[str, Any], int]:
    group_id = f"{source}_bs{batch_size}_g{group_idx}_r{repeat_idx}_{uuid.uuid4().hex[:8]}"
    os.environ["QWEN35_PLUGIN_REQUEST_ID"] = group_id
    request_ids = [
        f"{group_id}_req{req_idx}_{sample.get('cid', req_idx)}"
        for req_idx, sample in enumerate(samples)
    ]
    requests = [
        materialize_request(
            sample,
            request_id,
        )
        for sample, request_id in zip(samples, request_ids)
    ]

    t0 = time.perf_counter()
    outputs = llm.generate(requests, sampling_params=sampling_params, use_tqdm=False)
    torch_cuda_synchronize()
    e2e_ms = (time.perf_counter() - t0) * 1000.0

    records, profile_offset = read_profile_records(profile_path, profile_offset)
    # In a separate engine process, env updates after model construction may not
    # propagate. Empty request_id records are therefore accepted and grouped by
    # file offset per generate() call.
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
    row = {
        "source": source,
        "batch_size": batch_size,
        "target_prompt_tokens": target_prompt_tokens,
        "actual_prompt_tokens_min": _token_stat(prompt_token_counts, min),
        "actual_prompt_tokens_max": _token_stat(prompt_token_counts, max),
        "actual_prompt_tokens_mean": _token_stat(prompt_token_counts, statistics.mean),
        "repeat_idx": repeat_idx,
        "group_idx": group_idx,
        "is_warmup": is_warmup,
        "request_count": len(outputs),
        "e2e_ms": e2e_ms,
        "per_request_e2e_ms": e2e_ms / max(1, len(outputs)),
        "requests_per_s": len(outputs) / (e2e_ms / 1000.0) if e2e_ms > 0 else 0.0,
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
        "vit_wall_first_ms": vit_wall_first_ms,
        "llm_wall_first_ms": llm_wall_first_ms,
        "qwen_forward_wall_first_ms": qwen_forward_wall_first_ms,
        "model_total_wall_first_ms": vit_wall_first_ms + qwen_forward_wall_first_ms,
        "vit_calls": phase_count(records, "vit_forward"),
        "llm_calls": phase_count(records, "llm_forward"),
        "qwen_forward_calls": phase_count(records, "qwen_forward"),
        "completion_tokens": sum(completion_token_count(output) for output in outputs),
        "input_mode": input_mode,
        "request_ids_unique": len(set(request_ids)) == len(request_ids),
        "unique_request_count": len(set(request_ids)),
        "unique_request_ids": ";".join(request_ids),
    }
    return row, profile_offset


def main() -> None:
    args = parse_args()
    input_lens = parse_int_list(args.input_lens)
    batch_sizes = parse_int_list(args.batch_sizes)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_path = Path(args.profile_jsonl or output_dir / "profile_records.jsonl")
    csv_path = Path(args.csv or output_dir / "summary.csv")
    aggregate_csv_path = Path(args.aggregate_csv or output_dir / "aggregate_summary.csv")
    write_run_config(output_dir / "run_config.json", args, input_lens, batch_sizes)
    profile_path.write_text("")

    os.environ["QWEN35_PLUGIN_PROFILE_PATH"] = str(profile_path)
    max_images_per_prompt = (
        infer_dataset_max_images(args.dataset_jsonl, args.dataset_limit)
        if args.dataset_jsonl
        else 1
    )

    llm_kwargs: dict[str, Any] = {
        "model": args.model,
        "dtype": "bfloat16",
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "limit_mm_per_prompt": {"image": max_images_per_prompt},
        "mm_processor_kwargs": {
            "do_rescale": not args.disable_mm_do_rescale,
            "min_pixels": args.mm_min_pixels,
            "max_pixels": args.mm_max_pixels,
        },
        "enforce_eager": not args.disable_enforce_eager,
        "enable_prefix_caching": args.enable_prefix_caching,
    }
    if args.max_num_batched_tokens > 0:
        llm_kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens
    llm = LLM(**llm_kwargs)
    # Ignore model initialization, profiling, and warmup records. The benchmark
    # rows below should only include records generated by each llm.generate call.
    profile_offset = profile_path.stat().st_size if profile_path.exists() else 0
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    rows: list[dict[str, Any]] = []
    seen_request_ids: set[str] = set()

    if args.dataset_jsonl:
        source = "dataset"
        dataset_samples = load_dataset_samples(
            args.dataset_jsonl,
            tokenizer,
            input_mode=args.input_mode,
            limit=args.dataset_limit,
        )
        for batch_size in batch_sizes:
            if batch_size <= 0:
                raise ValueError("--batch-sizes values must be positive")
            warmup_batch = dataset_samples[:batch_size]
            for warmup_idx in range(args.warmup):
                row, profile_offset = run_one_batch(
                    llm=llm,
                    samples=warmup_batch,
                    sampling_params=sampling_params,
                    profile_path=profile_path,
                    profile_offset=profile_offset,
                    batch_size=batch_size,
                    target_prompt_tokens="dataset",
                    repeat_idx=-1,
                    group_idx=warmup_idx,
                    is_warmup=True,
                    source=source,
                    input_mode=args.input_mode,
                )
                enforce_global_unique_request_ids(row, seen_request_ids)
                rows.append(row)
                print(json.dumps(row, ensure_ascii=False), flush=True)

            for repeat_idx in range(args.repeats):
                for group_idx, batch_samples in iter_batches(dataset_samples, batch_size):
                    row, profile_offset = run_one_batch(
                        llm=llm,
                        samples=batch_samples,
                        sampling_params=sampling_params,
                        profile_path=profile_path,
                        profile_offset=profile_offset,
                        batch_size=batch_size,
                        target_prompt_tokens="dataset",
                        repeat_idx=repeat_idx,
                        group_idx=group_idx,
                        is_warmup=False,
                        source=source,
                        input_mode=args.input_mode,
                    )
                    enforce_global_unique_request_ids(row, seen_request_ids)
                    rows.append(row)
                    print(json.dumps(row, ensure_ascii=False), flush=True)
    else:
        source = "synthetic"
        image = load_image(args.image, args.image_size)
        samples_by_len = build_synthetic_samples(tokenizer, image, input_lens, args.input_mode)
        for batch_size in batch_sizes:
            if batch_size <= 0:
                raise ValueError("--batch-sizes values must be positive")
            for target_len in input_lens:
                total_runs = args.warmup + args.repeats
                for run_idx in range(total_runs):
                    is_warmup = run_idx < args.warmup
                    sample = samples_by_len[target_len]
                    row, profile_offset = run_one_batch(
                        llm=llm,
                        samples=[sample] * batch_size,
                        sampling_params=sampling_params,
                        profile_path=profile_path,
                        profile_offset=profile_offset,
                        batch_size=batch_size,
                        target_prompt_tokens=target_len,
                        repeat_idx=max(0, run_idx - args.warmup),
                        group_idx=0,
                        is_warmup=is_warmup,
                        source=source,
                        input_mode=args.input_mode,
                    )
                    enforce_global_unique_request_ids(row, seen_request_ids)
                    rows.append(row)
                    print(json.dumps(row, ensure_ascii=False), flush=True)

    if rows:
        fieldnames = list(rows[0].keys())
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    measured = [r for r in rows if not r["is_warmup"]]
    aggregate_rows: list[dict[str, Any]] = []
    base_requests_per_s: float | None = None
    print("\nSummary by batch_size:")
    for batch_size in batch_sizes:
        group = [r for r in measured if r["batch_size"] == batch_size]
        if not group:
            continue

        def avg(key: str) -> float:
            return statistics.mean(float(r[key]) for r in group)

        total_requests = sum(int(r["request_count"]) for r in group)
        total_prompt_tokens = sum(int(r["total_prompt_tokens"]) for r in group)
        total_e2e_ms = sum(float(r["e2e_ms"]) for r in group)
        total_requests_per_s = total_requests / (total_e2e_ms / 1000.0) if total_e2e_ms > 0 else 0.0
        total_prompt_tokens_per_s = (
            total_prompt_tokens / (total_e2e_ms / 1000.0) if total_e2e_ms > 0 else 0.0
        )
        if base_requests_per_s is None:
            base_requests_per_s = total_requests_per_s
        speedup_vs_first_bs = (
            total_requests_per_s / base_requests_per_s
            if base_requests_per_s and base_requests_per_s > 0
            else 0.0
        )
        aggregate_row = {
            "source": group[0]["source"],
            "batch_size": batch_size,
            "measured_groups": len(group),
            "total_requests": total_requests,
            "total_prompt_tokens": total_prompt_tokens,
            "total_e2e_ms": total_e2e_ms,
            "request_ids_unique_all": all(r["request_ids_unique"] is True for r in group),
            "request_ids_unique_global_all": all(r["request_ids_unique_global"] is True for r in group),
            "avg_e2e_ms": avg("e2e_ms"),
            "avg_per_request_e2e_ms": avg("per_request_e2e_ms"),
            "avg_requests_per_s": avg("requests_per_s"),
            "total_requests_per_s": total_requests_per_s,
            "speedup_vs_first_bs": speedup_vs_first_bs,
            "avg_prompt_tokens_per_s": avg("prompt_tokens_per_s"),
            "total_prompt_tokens_per_s": total_prompt_tokens_per_s,
            "avg_vit_cuda_sum_ms": avg("vit_cuda_sum_ms"),
            "avg_llm_cuda_sum_ms": avg("llm_cuda_sum_ms"),
            "avg_qwen_forward_cuda_sum_ms": avg("qwen_forward_cuda_sum_ms"),
            "avg_model_total_cuda_sum_ms": avg("model_total_cuda_sum_ms"),
            "avg_vit_cuda_first_ms": avg("vit_cuda_first_ms"),
            "avg_llm_cuda_first_ms": avg("llm_cuda_first_ms"),
            "avg_qwen_forward_cuda_first_ms": avg("qwen_forward_cuda_first_ms"),
            "avg_model_total_cuda_first_ms": avg("model_total_cuda_first_ms"),
            "avg_vit_calls": avg("vit_calls"),
            "avg_llm_calls": avg("llm_calls"),
            "avg_qwen_forward_calls": avg("qwen_forward_calls"),
        }
        aggregate_rows.append(aggregate_row)
        print(
            f"bs={batch_size}: "
            f"e2e={aggregate_row['avg_e2e_ms']:.2f} ms, "
            f"per_req={aggregate_row['avg_per_request_e2e_ms']:.2f} ms, "
            f"req/s={aggregate_row['total_requests_per_s']:.2f}, "
            f"speedup={aggregate_row['speedup_vs_first_bs']:.2f}x, "
            f"prompt_tok/s={aggregate_row['avg_prompt_tokens_per_s']:.0f}, "
            f"vit_sum={aggregate_row['avg_vit_cuda_sum_ms']:.2f} ms, "
            f"llm_sum={aggregate_row['avg_llm_cuda_sum_ms']:.2f} ms, "
            f"qwen_forward_sum={aggregate_row['avg_qwen_forward_cuda_sum_ms']:.2f} ms, "
            f"model_total_sum={aggregate_row['avg_model_total_cuda_sum_ms']:.2f} ms"
        )

    if aggregate_rows:
        with aggregate_csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(aggregate_rows[0].keys()))
            writer.writeheader()
            writer.writerows(aggregate_rows)

    if not args.dataset_jsonl:
        print("\nSynthetic detail by batch_size and target_prompt_tokens:")
        for batch_size in batch_sizes:
            for target_len in input_lens:
                group = [
                    r for r in measured
                    if r["batch_size"] == batch_size and r["target_prompt_tokens"] == target_len
                ]
                if not group:
                    continue
                print(
                    f"bs={batch_size} len={target_len}: "
                    f"per_req={statistics.mean(float(r['per_request_e2e_ms']) for r in group):.2f} ms, "
                    f"model_total_sum={statistics.mean(float(r['model_total_cuda_sum_ms']) for r in group):.2f} ms"
                )

    print(f"\nCSV: {csv_path}")
    print(f"Aggregate CSV: {aggregate_csv_path}")
    print(f"Raw profile JSONL: {profile_path}")


if __name__ == "__main__":
    main()

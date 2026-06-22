#!/usr/bin/env python3
"""Build an accounting/timeline view for merged Qwen3.5 benchmark runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output_dir",
        help="Benchmark output directory containing summary.csv and profile_records.jsonl.",
    )
    parser.add_argument("--summary-csv", default=None)
    parser.add_argument("--profile-jsonl", default=None)
    parser.add_argument("--accounting-csv", default=None)
    parser.add_argument("--timeline-csv", default=None)
    parser.add_argument(
        "--include-warmup",
        action="store_true",
        help="Include warmup rows. By default only measured rows are analyzed.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            records.append(json.loads(line))
    return records


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def as_float(row: dict[str, str], key: str) -> float:
    raw = row.get(key, "")
    if raw == "":
        return 0.0
    return float(raw)


def is_true(raw: str) -> bool:
    return raw.lower() in {"1", "true", "yes"}


def phase_component(phase: str) -> str:
    if phase.startswith("benchmark:"):
        return "benchmark"
    if phase.startswith("vllm:"):
        return "vllm"
    if phase in {"vit_forward", "llm_forward", "qwen_forward"}:
        return "model"
    return "other"


def overlaps(record: dict[str, Any], start: float, end: float) -> bool:
    rec_start = record.get("start_perf_counter")
    rec_end = record.get("end_perf_counter")
    if rec_start is None or rec_end is None:
        return False
    rec_start = float(rec_start)
    rec_end = float(rec_end)
    return rec_start <= end and rec_end >= start


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    summary_path = Path(args.summary_csv or output_dir / "summary.csv")
    profile_path = Path(args.profile_jsonl or output_dir / "profile_records.jsonl")
    accounting_path = Path(args.accounting_csv or output_dir / "accounting_summary.csv")
    timeline_path = Path(args.timeline_csv or output_dir / "timeline.csv")

    summary_rows = read_csv(summary_path)
    profile_records = read_jsonl(profile_path)

    run_total_by_request: dict[str, dict[str, Any]] = {}
    for record in profile_records:
        if record.get("phase") != "benchmark:run_total":
            continue
        request_id = str(record.get("request_id") or "")
        if request_id:
            run_total_by_request[request_id] = record

    accounting_rows: list[dict[str, Any]] = []
    timeline_rows: list[dict[str, Any]] = []

    for row_idx, row in enumerate(summary_rows):
        if not args.include_warmup and is_true(row.get("is_warmup", "")):
            continue

        request_id = row.get("benchmark_request_id", "")
        run_total = run_total_by_request.get(request_id)
        run_start = run_total.get("start_perf_counter") if run_total else None
        run_end = run_total.get("end_perf_counter") if run_total else None
        run_start_f = float(run_start) if run_start is not None else None
        run_end_f = float(run_end) if run_end is not None else None

        e2e_ms = as_float(row, "e2e_ms")
        e2e_accounted = as_float(row, "benchmark_e2e_accounted_wall_ms")
        total_wall = as_float(row, "benchmark_run_total_wall_ms")
        total_accounted = as_float(row, "benchmark_total_accounted_wall_ms")
        model_total = as_float(row, "model_total_wall_sum_ms")

        accounting_rows.append({
            "row_idx": row_idx,
            "benchmark_request_id": request_id,
            "is_warmup": row.get("is_warmup", ""),
            "logical_batch_size": row.get("logical_batch_size", ""),
            "group_idx": row.get("group_idx", ""),
            "repeat_idx": row.get("repeat_idx", ""),
            "e2e_ms": e2e_ms,
            "e2e_accounted_wall_ms": e2e_accounted,
            "e2e_residual_wall_ms": e2e_ms - e2e_accounted,
            "run_total_wall_ms": total_wall,
            "run_total_accounted_wall_ms": total_accounted,
            "run_total_residual_wall_ms": total_wall - total_accounted,
            "model_total_wall_sum_ms": model_total,
            "e2e_minus_model_total_wall_ms": e2e_ms - model_total,
            "benchmark_pre_generate_wall_ms": as_float(
                row, "benchmark_pre_generate_wall_ms"
            ),
            "benchmark_llm_generate_wall_ms": as_float(
                row, "benchmark_llm_generate_wall_ms"
            ),
            "benchmark_cuda_sync_wall_ms": as_float(
                row, "benchmark_cuda_sync_wall_ms"
            ),
            "vllm_generate_wall_sum_ms": as_float(
                row, "vllm_generate_wall_sum_ms"
            ),
            "vllm_run_completion_wall_sum_ms": as_float(
                row, "vllm_run_completion_wall_sum_ms"
            ),
            "vllm_add_completion_requests_wall_sum_ms": as_float(
                row, "vllm_add_completion_requests_wall_sum_ms"
            ),
            "vllm_render_and_add_requests_wall_sum_ms": as_float(
                row, "vllm_render_and_add_requests_wall_sum_ms"
            ),
            "vllm_preprocess_cmpl_one_wall_sum_ms": as_float(
                row, "vllm_preprocess_cmpl_one_wall_sum_ms"
            ),
            "vllm_preprocess_cmpl_wall_sum_ms": as_float(
                row, "vllm_preprocess_cmpl_wall_sum_ms"
            ),
            "vllm_renderer_render_cmpl_wall_sum_ms": as_float(
                row, "vllm_renderer_render_cmpl_wall_sum_ms"
            ),
            "vllm_renderer_render_prompts_wall_sum_ms": as_float(
                row, "vllm_renderer_render_prompts_wall_sum_ms"
            ),
            "vllm_renderer_tokenize_prompts_wall_sum_ms": as_float(
                row, "vllm_renderer_tokenize_prompts_wall_sum_ms"
            ),
            "vllm_renderer_process_for_engine_wall_sum_ms": as_float(
                row, "vllm_renderer_process_for_engine_wall_sum_ms"
            ),
            "vllm_renderer_process_tokens_wall_sum_ms": as_float(
                row, "vllm_renderer_process_tokens_wall_sum_ms"
            ),
            "vllm_renderer_process_multimodal_wall_sum_ms": as_float(
                row, "vllm_renderer_process_multimodal_wall_sum_ms"
            ),
            "vllm_run_engine_wall_sum_ms": as_float(
                row, "vllm_run_engine_wall_sum_ms"
            ),
            "vllm_engine_step_wall_sum_ms": as_float(
                row, "vllm_engine_step_wall_sum_ms"
            ),
            "vllm_gpu_execute_model_wall_sum_ms": as_float(
                row, "vllm_gpu_execute_model_wall_sum_ms"
            ),
            "vllm_execute_mm_encoder_wall_sum_ms": as_float(
                row, "vllm_execute_mm_encoder_wall_sum_ms"
            ),
            "vllm_gather_mm_embeddings_wall_sum_ms": as_float(
                row, "vllm_gather_mm_embeddings_wall_sum_ms"
            ),
        })

        if run_start_f is None or run_end_f is None:
            continue

        for record in profile_records:
            record_request_id = str(record.get("request_id") or "")
            if record_request_id not in {"", request_id} and not overlaps(
                record, run_start_f, run_end_f
            ):
                continue
            if not overlaps(record, run_start_f, run_end_f):
                continue
            rec_start = float(record["start_perf_counter"])
            rec_end = float(record["end_perf_counter"])
            phase = str(record.get("phase") or "")
            timeline_rows.append({
                "row_idx": row_idx,
                "benchmark_request_id": request_id,
                "record_request_id": record_request_id,
                "is_warmup": row.get("is_warmup", ""),
                "logical_batch_size": row.get("logical_batch_size", ""),
                "group_idx": row.get("group_idx", ""),
                "repeat_idx": row.get("repeat_idx", ""),
                "component": phase_component(phase),
                "phase": phase,
                "pid": record.get("pid", ""),
                "start_rel_ms": (rec_start - run_start_f) * 1000.0,
                "end_rel_ms": (rec_end - run_start_f) * 1000.0,
                "wall_ms": record.get("wall_ms", ""),
                "cuda_ms": record.get("cuda_ms", ""),
                "output": record.get("output", ""),
            })

    timeline_rows.sort(
        key=lambda item: (
            int(item["row_idx"]),
            float(item["start_rel_ms"]),
            float(item["end_rel_ms"]),
            str(item["phase"]),
        )
    )

    write_csv(accounting_path, accounting_rows)
    write_csv(timeline_path, timeline_rows)

    max_abs_e2e_residual = max(
        (abs(float(row["e2e_residual_wall_ms"])) for row in accounting_rows),
        default=0.0,
    )
    max_abs_total_residual = max(
        (abs(float(row["run_total_residual_wall_ms"])) for row in accounting_rows),
        default=0.0,
    )
    print(f"Accounting CSV: {accounting_path}")
    print(f"Timeline CSV: {timeline_path}")
    print(f"Rows analyzed: {len(accounting_rows)}")
    print(f"Timeline records: {len(timeline_rows)}")
    print(f"Max |e2e residual|: {max_abs_e2e_residual:.6f} ms")
    print(f"Max |run_total residual|: {max_abs_total_residual:.6f} ms")


if __name__ == "__main__":
    main()

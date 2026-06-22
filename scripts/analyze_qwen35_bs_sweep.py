#!/usr/bin/env python3
"""Analyze Qwen3.5 vLLM batch-sweep benchmark CSV files.

Inputs can be either summary.csv files or directories containing summary.csv.
The script only uses non-warmup rows, verifies request-id uniqueness fields, and
writes combined CSV/Markdown reports. Rows are grouped by scenario first, then
batch size, so synthetic runs with different prompt lengths are not mixed.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="+",
        help="summary.csv files or benchmark output directories containing summary.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="/workspace/project/RL-learning/vllm-test/outputs/benchmark_qwen35_mm_bs_sweep_combined",
    )
    parser.add_argument(
        "--plateau-ratio",
        type=float,
        default=0.97,
        help="Smallest bs whose throughput is at least this fraction of best is recommended; lower values prefer smaller near-plateau bs.",
    )
    return parser.parse_args()


def resolve_summary_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_dir():
        path = path / "summary.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def infer_config(path: Path) -> str:
    run_config_path = path.parent / "run_config.json"
    if run_config_path.exists():
        run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
        max_num_batched_tokens = int(run_config.get("max_num_batched_tokens") or 0)
        if max_num_batched_tokens > 0:
            return f"mnbt{max_num_batched_tokens}"
        return "default"

    text = str(path.parent).lower()
    if "mnbt16384" in text or "max_num_batched_tokens=16384" in text:
        return "mnbt16384"
    return "default"


def read_measured_rows(path: Path) -> list[dict[str, Any]]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    measured = [row for row in rows if row.get("is_warmup") == "False"]
    for row in measured:
        row["_source_path"] = str(path)
        row["_config"] = infer_config(path)
    return measured


def as_float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    raw = row.get(key)
    if raw in {None, ""}:
        return default
    return float(raw)


def as_float_any(row: dict[str, Any], keys: tuple[str, ...], default: float = 0.0) -> float:
    for key in keys:
        raw = row.get(key)
        if raw not in {None, ""}:
            return float(raw)
    return default


def as_int(row: dict[str, Any], key: str, default: int = 0) -> int:
    raw = row.get(key)
    if raw in {None, ""}:
        return default
    return int(float(raw))


def bool_field(row: dict[str, Any], key: str) -> bool:
    value = str(row.get(key, "")).strip().lower()
    return value in {"true", "1", "yes"}


def bool_field_any(row: dict[str, Any], keys: tuple[str, ...]) -> bool:
    for key in keys:
        if row.get(key) not in {None, ""}:
            return bool_field(row, key)
    return False


def request_mode(row: dict[str, Any]) -> str:
    mode = str(row.get("request_mode") or "").strip()
    if mode:
        return mode
    if row.get("merged_item_count") not in {None, ""}:
        return "merged_logical_batch"
    return "batch_requests"


def scenario_key(row: dict[str, Any]) -> tuple[str, str, str, str, str, int]:
    return (
        str(row["_config"]),
        request_mode(row),
        str(row.get("source") or "unknown"),
        str(row.get("target_prompt_tokens") or "unknown"),
        str(row.get("input_mode") or "unknown"),
        as_int(row, "batch_size"),
    )


def recommendation_key(row: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(row["config"]),
        str(row["request_mode"]),
        str(row["source"]),
        str(row["target_prompt_tokens"]),
        str(row["input_mode"]),
    )


def summarize_group(
    config: str,
    mode: str,
    source: str,
    target_prompt_tokens: str,
    input_mode: str,
    batch_size: int,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    total_requests = sum(as_int(row, "request_count") for row in rows)
    total_engine_requests = sum(
        as_int(row, "engine_request_count", as_int(row, "request_count"))
        for row in rows
    )
    total_logical_items = sum(
        as_int(row, "merged_item_count", as_int(row, "request_count"))
        for row in rows
    )
    total_prompt_tokens = sum(as_int(row, "total_prompt_tokens") for row in rows)
    total_e2e_ms = sum(as_float(row, "e2e_ms") for row in rows)

    def avg(key: str) -> float:
        return statistics.mean(as_float(row, key) for row in rows)

    def avg_any(*keys: str) -> float:
        return statistics.mean(as_float_any(row, keys) for row in rows)

    def stdev(key: str) -> float:
        if len(rows) <= 1:
            return 0.0
        return statistics.stdev(as_float(row, key) for row in rows)

    def stdev_any(*keys: str) -> float:
        if len(rows) <= 1:
            return 0.0
        return statistics.stdev(as_float_any(row, keys) for row in rows)

    request_ids_unique_all = all(
        bool_field_any(row, ("request_ids_unique", "mm_uuids_unique"))
        for row in rows
    )
    if "request_ids_unique_global" in rows[0] or "mm_uuids_unique_global" in rows[0]:
        request_ids_unique_global_all = all(
            bool_field_any(
                row,
                ("request_ids_unique_global", "mm_uuids_unique_global"),
            )
            for row in rows
        )
    else:
        request_ids_unique_global_all = "missing"

    return {
        "config": config,
        "request_mode": mode,
        "source": source,
        "target_prompt_tokens": target_prompt_tokens,
        "input_mode": input_mode,
        "batch_size": batch_size,
        "measured_groups": len(rows),
        "total_requests": total_requests,
        "total_engine_requests": total_engine_requests,
        "total_logical_items": total_logical_items,
        "total_prompt_tokens": total_prompt_tokens,
        "total_e2e_ms": total_e2e_ms,
        "total_requests_per_s": (
            total_requests / (total_e2e_ms / 1000.0) if total_e2e_ms > 0 else 0.0
        ),
        "engine_requests_per_s": (
            total_engine_requests / (total_e2e_ms / 1000.0)
            if total_e2e_ms > 0
            else 0.0
        ),
        "logical_items_per_s": (
            total_logical_items / (total_e2e_ms / 1000.0)
            if total_e2e_ms > 0
            else 0.0
        ),
        "avg_per_request_e2e_ms": avg_any(
            "per_request_e2e_ms",
            "per_engine_request_e2e_ms",
        ),
        "stdev_per_request_e2e_ms": stdev_any(
            "per_request_e2e_ms",
            "per_engine_request_e2e_ms",
        ),
        "avg_per_engine_request_e2e_ms": avg_any(
            "per_engine_request_e2e_ms",
            "per_request_e2e_ms",
        ),
        "avg_per_logical_item_e2e_ms": avg_any(
            "per_logical_item_e2e_ms",
            "per_request_e2e_ms",
        ),
        "total_prompt_tokens_per_s": (
            total_prompt_tokens / (total_e2e_ms / 1000.0)
            if total_e2e_ms > 0
            else 0.0
        ),
        "avg_vit_cuda_sum_ms": avg_any("vit_cuda_sum_ms", "vit_cuda_ms"),
        "avg_llm_cuda_sum_ms": avg_any("llm_cuda_sum_ms", "llm_cuda_ms"),
        "avg_qwen_forward_cuda_sum_ms": avg_any(
            "qwen_forward_cuda_sum_ms", "qwen_forward_cuda_ms"
        ),
        "avg_model_total_cuda_sum_ms": avg_any(
            "model_total_cuda_sum_ms", "model_total_cuda_ms"
        ),
        "avg_vit_cuda_first_ms": avg_any("vit_cuda_first_ms", "vit_cuda_sum_ms", "vit_cuda_ms"),
        "avg_llm_cuda_first_ms": avg_any("llm_cuda_first_ms", "llm_cuda_sum_ms", "llm_cuda_ms"),
        "avg_qwen_forward_cuda_first_ms": avg_any(
            "qwen_forward_cuda_first_ms",
            "qwen_forward_cuda_sum_ms",
            "qwen_forward_cuda_ms",
        ),
        "avg_model_total_cuda_first_ms": avg_any(
            "model_total_cuda_first_ms",
            "model_total_cuda_sum_ms",
            "model_total_cuda_ms",
        ),
        "avg_vit_calls": avg("vit_calls"),
        "avg_llm_calls": avg("llm_calls"),
        "avg_qwen_forward_calls": avg("qwen_forward_calls"),
        "request_ids_unique_all": request_ids_unique_all,
        "request_ids_unique_global_all": request_ids_unique_global_all,
        "source_paths": ";".join(sorted({row["_source_path"] for row in rows})),
    }


def build_recommendations(rows: list[dict[str, Any]], plateau_ratio: float) -> list[dict[str, Any]]:
    recommendations = []
    scenarios = sorted({recommendation_key(row) for row in rows})
    for config, mode, source, target_prompt_tokens, input_mode in scenarios:
        scenario_rows = [
            row for row in rows
            if recommendation_key(row) == (
                config,
                mode,
                source,
                target_prompt_tokens,
                input_mode,
            )
        ]
        throughput_key = (
            "logical_items_per_s"
            if mode == "merged_logical_batch"
            else "total_requests_per_s"
        )
        best = max(scenario_rows, key=lambda row: float(row[throughput_key]))
        threshold = float(best[throughput_key]) * plateau_ratio
        plateau_rows = [
            row for row in scenario_rows if float(row[throughput_key]) >= threshold
        ]
        recommended = min(plateau_rows, key=lambda row: int(row["batch_size"]))
        recommendations.append({
            "config": config,
            "request_mode": mode,
            "source": source,
            "target_prompt_tokens": target_prompt_tokens,
            "input_mode": input_mode,
            "throughput_metric": throughput_key,
            "best_batch_size": int(best["batch_size"]),
            "best_requests_per_s": float(best[throughput_key]),
            "recommended_batch_size": int(recommended["batch_size"]),
            "recommended_requests_per_s": float(recommended[throughput_key]),
            "plateau_ratio": plateau_ratio,
        })
    return recommendations


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(
    path: Path,
    rows: list[dict[str, Any]],
    recs: list[dict[str, Any]],
) -> None:
    summary_header = (
        "| config | mode | source | target | input | bs | groups | "
        "logical_items/s | engine_req/s | per_engine_req_ms | "
        "per_logical_item_ms | prompt_tok/s | vit_sum_ms | llm_sum_ms | "
        "model_total_sum_ms | unique | global_unique |"
    )
    recommendation_header = (
        "| config | mode | source | target | input | metric | best_bs | "
        "best_value | recommended_bs | recommended_value | plateau_ratio |"
    )
    lines = [
        "# Qwen3.5 vLLM Batch Sweep Analysis",
        "",
        "## Summary",
        "",
        summary_header,
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['config']} | {row['request_mode']} | {row['source']} "
            f"| {row['target_prompt_tokens']} | {row['input_mode']} "
            f"| {row['batch_size']} | {row['measured_groups']} "
            f"| {float(row['logical_items_per_s']):.2f} "
            f"| {float(row['engine_requests_per_s']):.2f} "
            f"| {float(row['avg_per_engine_request_e2e_ms']):.2f} "
            f"| {float(row['avg_per_logical_item_e2e_ms']):.2f} "
            f"| {float(row['total_prompt_tokens_per_s']):.0f} "
            f"| {float(row['avg_vit_cuda_sum_ms']):.2f} "
            f"| {float(row['avg_llm_cuda_sum_ms']):.2f} "
            f"| {float(row['avg_model_total_cuda_sum_ms']):.2f} "
            f"| {row['request_ids_unique_all']} | {row['request_ids_unique_global_all']} |"
        )

    lines.extend([
        "",
        "## Recommendations",
        "",
        recommendation_header,
        "|---|---|---|---|---|---|---:|---:|---:|---:|---:|",
    ])
    for rec in recs:
        lines.append(
            f"| {rec['config']} | {rec['request_mode']} | {rec['source']} "
            f"| {rec['target_prompt_tokens']} | {rec['input_mode']} "
            f"| {rec['throughput_metric']} | {rec['best_batch_size']} "
            f"| {rec['best_requests_per_s']:.2f} "
            f"| {rec['recommended_batch_size']} "
            f"| {rec['recommended_requests_per_s']:.2f} "
            f"| {rec['plateau_ratio']:.2f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    summary_paths = [resolve_summary_path(item) for item in args.inputs]
    measured_rows: list[dict[str, Any]] = []
    for path in summary_paths:
        measured_rows.extend(read_measured_rows(path))
    if not measured_rows:
        raise RuntimeError("No measured rows found in inputs.")

    grouped: dict[
        tuple[str, str, str, str, str, int],
        list[dict[str, Any]],
    ] = defaultdict(list)
    for row in measured_rows:
        grouped[scenario_key(row)].append(row)

    summary_rows = [
        summarize_group(
            config,
            mode,
            source,
            target_prompt_tokens,
            input_mode,
            batch_size,
            rows,
        )
        for (
            config,
            mode,
            source,
            target_prompt_tokens,
            input_mode,
            batch_size,
        ), rows in grouped.items()
    ]
    summary_rows.sort(key=lambda row: (
        row["config"] != "default",
        str(row["request_mode"]),
        str(row["source"]),
        str(row["target_prompt_tokens"]),
        str(row["input_mode"]),
        int(row["batch_size"]),
    ))
    recommendations = build_recommendations(summary_rows, args.plateau_ratio)

    output_dir = Path(args.output_dir)
    write_csv(output_dir / "combined_by_bs_summary.csv", summary_rows)
    write_csv(output_dir / "recommendations.csv", recommendations)
    write_markdown(output_dir / "combined_by_bs_summary.md", summary_rows, recommendations)

    print((output_dir / "combined_by_bs_summary.md").read_text(encoding="utf-8"))
    print(f"CSV: {output_dir / 'combined_by_bs_summary.csv'}")
    print(f"Recommendations: {output_dir / 'recommendations.csv'}")


if __name__ == "__main__":
    main()

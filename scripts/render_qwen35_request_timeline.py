#!/usr/bin/env python3
"""Render a single-request HTML timeline from Qwen3.5 benchmark timeline.csv."""

from __future__ import annotations

import argparse
import csv
import html
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


PHASE_ORDER = [
    "benchmark:run_total",
    "benchmark:build_prompt",
    "benchmark:prepare_mm_data",
    "benchmark:encode_prompt",
    "benchmark:build_request",
    "benchmark:llm_generate",
    "vllm:LLM.generate",
    "vllm:LLM._run_completion",
    "vllm:LLM._add_completion_requests",
    "vllm:LLM._render_and_add_requests",
    "vllm:LLM._preprocess_cmpl_one",
    "vllm:LLM._preprocess_cmpl",
    "vllm:BaseRenderer.render_cmpl",
    "vllm:BaseRenderer.render_prompts",
    "vllm:BaseRenderer.tokenize_prompts",
    "vllm:BaseRenderer.process_for_engine",
    "vllm:BaseRenderer._process_tokens",
    "vllm:BaseRenderer._process_multimodal",
    "vllm:LLM._add_request",
    "vllm:LLMEngine.add_request",
    "vllm:OutputProcessor.add_request",
    "vllm:LLM._run_engine",
    "vllm:LLMEngine.step",
    "vllm:Worker.execute_model",
    "vllm:GPUModelRunner.execute_model",
    "vllm:GPUModelRunner._execute_mm_encoder",
    "vllm:GPUModelRunner._gather_mm_embeddings",
    "vit_forward",
    "qwen_forward",
    "llm_forward",
    "vllm:OutputProcessor.process_outputs",
    "benchmark:cuda_sync",
]

COMPONENT_COLORS = {
    "benchmark": "#4E79A7",
    "vllm": "#F28E2B",
    "model": "#59A14F",
    "other": "#9C755F",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("timeline_csv", help="Path to timeline.csv.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output HTML path. Defaults to request_timeline.html next to the CSV.",
    )
    parser.add_argument(
        "--svg-output",
        default=None,
        help="Optional static SVG output path for inline Markdown previews.",
    )
    parser.add_argument("--request-id", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--group-idx", type=int, default=None)
    parser.add_argument("--repeat-idx", type=int, default=None)
    parser.add_argument(
        "--include-warmup",
        action="store_true",
        help="Allow selecting warmup requests. Default only selects measured rows.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Timeline title. Defaults to selected request id.",
    )
    parser.add_argument(
        "--min-wall-ms",
        type=float,
        default=0.0,
        help="Hide records shorter than this wall time, except benchmark:run_total.",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def as_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value else 0.0


def as_int_or_none(value: str) -> int | None:
    if value == "":
        return None
    return int(float(value))


def phase_sort_key(phase: str) -> tuple[int, str]:
    try:
        return PHASE_ORDER.index(phase), phase
    except ValueError:
        return len(PHASE_ORDER), phase


def select_request(rows: list[dict[str, str]], args: argparse.Namespace) -> str:
    request_ids: list[str] = []
    for row in rows:
        if not args.include_warmup and row.get("is_warmup", "").lower() == "true":
            continue
        if args.request_id and row.get("benchmark_request_id") != args.request_id:
            continue
        if args.batch_size is not None and as_int_or_none(
            row.get("logical_batch_size", "")
        ) != args.batch_size:
            continue
        if args.group_idx is not None and as_int_or_none(
            row.get("group_idx", "")
        ) != args.group_idx:
            continue
        if args.repeat_idx is not None and as_int_or_none(
            row.get("repeat_idx", "")
        ) != args.repeat_idx:
            continue
        request_id = row.get("benchmark_request_id", "")
        if request_id and request_id not in request_ids:
            request_ids.append(request_id)

    if not request_ids:
        raise SystemExit("No matching request found in timeline.csv")
    return request_ids[0]


def compact_phase_label(phase: str) -> str:
    if phase.startswith("vllm:"):
        return phase.removeprefix("vllm:")
    if phase.startswith("benchmark:"):
        return phase.removeprefix("benchmark:")
    return phase


def render_html(
    rows: list[dict[str, str]],
    request_id: str,
    title: str,
    source_path: Path,
) -> str:
    selected = [row for row in rows if row.get("benchmark_request_id") == request_id]
    if not selected:
        raise SystemExit(f"Request id not found: {request_id}")

    max_end = max(as_float(row, "end_rel_ms") for row in selected)
    min_start = min(as_float(row, "start_rel_ms") for row in selected)
    total_ms = max_end - min_start
    if total_ms <= 0:
        total_ms = max_end or 1.0

    meta_source = selected[0]
    phases: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in selected:
        phases[row.get("phase", "")].append(row)

    ordered_phases = sorted(phases, key=phase_sort_key)
    row_height = 34
    axis_height = 48
    label_width = 315
    track_width = 1180
    timeline_height = axis_height + len(ordered_phases) * row_height
    px_per_ms = track_width / total_ms

    def x_pct(ms: float) -> float:
        return (ms - min_start) / total_ms * 100.0

    tick_count = 8
    ticks = [total_ms * idx / tick_count for idx in range(tick_count + 1)]

    rows_html: list[str] = []
    for phase in ordered_phases:
        lane_records = phases[phase]
        component = lane_records[0].get("component", "other")
        color = COMPONENT_COLORS.get(component, COMPONENT_COLORS["other"])
        bars: list[str] = []
        for record in lane_records:
            start = as_float(record, "start_rel_ms")
            end = as_float(record, "end_rel_ms")
            wall = as_float(record, "wall_ms")
            width = max((end - start) / total_ms * 100.0, 0.18)
            left = x_pct(start)
            cuda = record.get("cuda_ms") or ""
            tooltip = {
                "phase": phase,
                "component": component,
                "start_ms": round(start, 3),
                "end_ms": round(end, 3),
                "wall_ms": round(wall, 3),
                "cuda_ms": round(float(cuda), 3) if cuda else None,
                "pid": record.get("pid", ""),
                "output": record.get("output", ""),
            }
            bars.append(
                '<div class="bar" '
                f'style="left:{left:.4f}%;width:{width:.4f}%;background:{color}" '
                f'data-tip="{html.escape(json.dumps(tooltip, ensure_ascii=False))}">'
                "</div>"
            )

        rows_html.append(
            '<div class="lane">'
            f'<div class="lane-label"><span>{html.escape(compact_phase_label(phase))}</span>'
            f'<small>{html.escape(component)}</small></div>'
            f'<div class="lane-track">{"".join(bars)}</div>'
            "</div>"
        )

    axis_ticks = []
    for tick in ticks:
        left = tick / total_ms * 100.0
        label = f"{tick:.0f} ms" if total_ms >= 100 else f"{tick:.1f} ms"
        axis_ticks.append(
            f'<div class="tick" style="left:{left:.4f}%"><span>{label}</span></div>'
        )

    legend = "".join(
        f'<span class="legend-item"><i style="background:{color}"></i>{name}</span>'
        for name, color in COMPONENT_COLORS.items()
    )

    meta = (
        f"request={request_id} | bs={meta_source.get('logical_batch_size', '')} | "
        f"group={meta_source.get('group_idx', '')} | repeat={meta_source.get('repeat_idx', '')} | "
        f"total={max_end:.2f} ms | source={source_path}"
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --bg: #f7f4ee;
      --panel: #ffffff;
      --ink: #1f2933;
      --muted: #667085;
      --grid: rgba(31, 41, 51, 0.12);
      --line: rgba(31, 41, 51, 0.10);
      --label-width: {label_width}px;
      --track-width: {track_width}px;
      --row-height: {row_height}px;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Arial, Helvetica, sans-serif;
    }}
    .shell {{
      max-width: 1580px;
      margin: 0 auto;
      padding: 24px;
    }}
    .hero {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 18px 20px;
      box-shadow: 0 14px 34px rgba(31, 41, 51, 0.08);
      position: sticky;
      top: 0;
      z-index: 5;
    }}
    h1 {{
      margin: 0;
      font-size: 24px;
      line-height: 1.2;
    }}
    .meta {{
      margin-top: 8px;
      color: var(--muted);
      font-size: 13px;
      word-break: break-all;
    }}
    .legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin-top: 14px;
      color: var(--muted);
      font-size: 13px;
    }}
    .legend-item {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }}
    .legend-item i {{
      width: 12px;
      height: 12px;
      border-radius: 3px;
      display: inline-block;
    }}
    .timeline-card {{
      margin-top: 18px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 14px;
      overflow: auto;
      box-shadow: 0 14px 34px rgba(31, 41, 51, 0.08);
    }}
    .timeline {{
      min-width: calc(var(--label-width) + var(--track-width));
      width: calc(var(--label-width) + var(--track-width));
      height: {timeline_height}px;
    }}
    .axis {{
      display: grid;
      grid-template-columns: var(--label-width) var(--track-width);
      height: {axis_height}px;
      border-bottom: 1px solid var(--line);
      position: sticky;
      top: 0;
      background: rgba(255, 255, 255, 0.96);
      z-index: 3;
    }}
    .axis-title {{
      padding: 17px 14px;
      color: var(--muted);
      font-weight: 700;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      border-right: 1px solid var(--line);
    }}
    .axis-track {{
      position: relative;
    }}
    .tick {{
      position: absolute;
      top: 0;
      bottom: 0;
      width: 1px;
      background: var(--grid);
    }}
    .tick span {{
      position: absolute;
      top: 12px;
      transform: translateX(-50%);
      white-space: nowrap;
      color: var(--muted);
      font-size: 12px;
    }}
    .lane {{
      display: grid;
      grid-template-columns: var(--label-width) var(--track-width);
      height: var(--row-height);
      border-bottom: 1px solid var(--line);
    }}
    .lane-label {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      padding: 0 12px;
      border-right: 1px solid var(--line);
      background: #fbfbfa;
      overflow: hidden;
    }}
    .lane-label span {{
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      font-size: 13px;
    }}
    .lane-label small {{
      color: var(--muted);
      font-size: 11px;
      flex-shrink: 0;
    }}
    .lane-track {{
      position: relative;
      background-image: linear-gradient(90deg, var(--grid) 1px, transparent 1px);
      background-size: {max(px_per_ms * max(total_ms / 8.0, 1.0), 1.0):.1f}px 100%;
    }}
    .bar {{
      position: absolute;
      top: 8px;
      height: 18px;
      border-radius: 999px;
      box-shadow: 0 3px 9px rgba(31, 41, 51, 0.18);
      border: 1px solid rgba(255, 255, 255, 0.7);
      cursor: pointer;
      opacity: 0.9;
    }}
    .bar:hover {{
      opacity: 1;
      transform: scaleY(1.18);
      z-index: 4;
    }}
    .tooltip {{
      position: fixed;
      pointer-events: none;
      max-width: 360px;
      background: rgba(17, 24, 39, 0.94);
      color: #fff;
      border-radius: 10px;
      padding: 10px 12px;
      font-size: 12px;
      line-height: 1.45;
      opacity: 0;
      transform: translateY(6px);
      transition: opacity 100ms ease, transform 100ms ease;
      z-index: 10;
      box-shadow: 0 12px 30px rgba(17, 24, 39, 0.24);
    }}
    .tooltip.visible {{
      opacity: 1;
      transform: translateY(0);
    }}
    .tooltip b {{
      display: block;
      margin-bottom: 5px;
      font-size: 13px;
      word-break: break-all;
    }}
  </style>
</head>
<body>
  <main class="shell">
    <section class="hero">
      <h1>{html.escape(title)}</h1>
      <div class="meta">{html.escape(meta)}</div>
      <div class="legend">{legend}</div>
    </section>
    <section class="timeline-card">
      <div class="timeline">
        <div class="axis">
          <div class="axis-title">Phase</div>
          <div class="axis-track">{''.join(axis_ticks)}</div>
        </div>
        {''.join(rows_html)}
      </div>
    </section>
  </main>
  <div class="tooltip" id="tooltip"></div>
  <script>
    const tooltip = document.getElementById('tooltip');
    document.querySelectorAll('.bar').forEach((bar) => {{
      bar.addEventListener('mousemove', (event) => {{
        const data = JSON.parse(bar.dataset.tip);
        tooltip.innerHTML = `<b>${{data.phase}}</b>`
          + `<div>component: ${{data.component}}</div>`
          + `<div>start: ${{data.start_ms}} ms</div>`
          + `<div>end: ${{data.end_ms}} ms</div>`
          + `<div>wall: ${{data.wall_ms}} ms</div>`
          + (data.cuda_ms === null ? '' : `<div>cuda: ${{data.cuda_ms}} ms</div>`)
          + `<div>pid: ${{data.pid}}</div>`
          + `<div>output: ${{data.output}}</div>`;
        const pad = 14;
        tooltip.style.left = Math.min(event.clientX + pad, window.innerWidth - 380) + 'px';
        tooltip.style.top = Math.min(event.clientY + pad, window.innerHeight - 180) + 'px';
        tooltip.classList.add('visible');
      }});
      bar.addEventListener('mouseleave', () => {{
        tooltip.classList.remove('visible');
      }});
    }});
  </script>
</body>
</html>
"""


def render_svg(
    rows: list[dict[str, str]],
    request_id: str,
    title: str,
    output_path: Path,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "matplotlib is required for --svg-output. Install matplotlib or omit "
            "--svg-output."
        ) from exc

    selected = [row for row in rows if row.get("benchmark_request_id") == request_id]
    if not selected:
        raise SystemExit(f"Request id not found: {request_id}")

    max_end = max(as_float(row, "end_rel_ms") for row in selected)
    min_start = min(as_float(row, "start_rel_ms") for row in selected)
    total_ms = max(max_end - min_start, 1.0)

    phases: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in selected:
        phases[row.get("phase", "")].append(row)

    ordered_phases = sorted(phases, key=phase_sort_key)
    fig_height = max(7.0, 0.34 * len(ordered_phases) + 1.8)
    fig, ax = plt.subplots(figsize=(15.2, fig_height), dpi=140)

    for y_idx, phase in enumerate(ordered_phases):
        lane_records = phases[phase]
        component = lane_records[0].get("component", "other")
        color = COMPONENT_COLORS.get(component, COMPONENT_COLORS["other"])
        for record in lane_records:
            start = as_float(record, "start_rel_ms") - min_start
            end = as_float(record, "end_rel_ms") - min_start
            width = max(end - start, total_ms * 0.0015)
            ax.broken_barh(
                [(start, width)],
                (y_idx - 0.34, 0.68),
                facecolors=color,
                edgecolors="white",
                linewidth=0.8,
                alpha=0.92,
            )
            if as_float(record, "wall_ms") >= total_ms * 0.12:
                ax.text(
                    start + width / 2,
                    y_idx,
                    f"{as_float(record, 'wall_ms'):.0f} ms",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white",
                    clip_on=True,
                )

    ax.set_yticks(range(len(ordered_phases)))
    ax.set_yticklabels([compact_phase_label(phase) for phase in ordered_phases])
    ax.invert_yaxis()
    ax.set_xlim(0, total_ms * 1.01)
    ax.set_xlabel("Time since benchmark:run_total start (ms)")
    ax.set_title(title)
    ax.grid(True, axis="x", linestyle="--", linewidth=0.7, alpha=0.35)
    ax.grid(False, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", labelsize=8)
    ax.tick_params(axis="x", labelsize=9)
    ax.legend(
        handles=[
            Patch(facecolor=color, label=name)
            for name, color in COMPONENT_COLORS.items()
        ],
        loc="upper right",
        frameon=True,
    )
    meta = selected[0]
    fig.text(
        0.01,
        0.01,
        (
            f"request={request_id} | bs={meta.get('logical_batch_size', '')} | "
            f"group={meta.get('group_idx', '')} | repeat={meta.get('repeat_idx', '')} | "
            f"total={max_end:.2f} ms"
        ),
        fontsize=8,
        color="#667085",
    )
    fig.tight_layout(rect=(0, 0.025, 1, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    timeline_path = Path(args.timeline_csv)
    rows = read_rows(timeline_path)

    if args.min_wall_ms > 0:
        rows = [
            row
            for row in rows
            if row.get("phase") == "benchmark:run_total"
            or as_float(row, "wall_ms") >= args.min_wall_ms
        ]

    request_id = select_request(rows, args)
    title = args.title or f"Qwen3.5 request timeline: {request_id}"
    output_path = Path(args.output or timeline_path.with_name("request_timeline.html"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        render_html(rows, request_id, title, timeline_path),
        encoding="utf-8",
    )
    print(f"Timeline HTML: {output_path}")
    if args.svg_output:
        svg_output_path = Path(args.svg_output)
        render_svg(rows, request_id, title, svg_output_path)
        print(f"Timeline SVG: {svg_output_path}")


if __name__ == "__main__":
    main()

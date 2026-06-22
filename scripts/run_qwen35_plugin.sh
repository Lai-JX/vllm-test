#!/usr/bin/env bash
set -euo pipefail

# Start Qwen3.5 with the non-invasive vLLM plugin.
#
# Usage:
#   ./scripts/run_qwen35_plugin.sh online
#   ./scripts/run_qwen35_plugin.sh offline
#   ./scripts/run_qwen35_plugin.sh benchmark-smoke
#   ./scripts/run_qwen35_plugin.sh benchmark-merged-smoke
#   ./scripts/run_qwen35_plugin.sh analyze
#   ./scripts/run_qwen35_plugin.sh analyze-timeline
#
# Optional env overrides:
#   MODEL_PATH=/share/models/Qwen3.5-2B
#   SERVED_MODEL_NAME=qwen3.5-2b
#   HOST=0.0.0.0
#   PORT=8000
#   MAX_MODEL_LEN=32768
#   MAX_NUM_BATCHED_TOKENS=0
#   GPU_MEMORY_UTILIZATION=0.9
#   PROMPT="你好，用一句话介绍你自己。"
#   DATASET_JSONL=/path/to/dataset.jsonl
#   DATASET_LIMIT=32
#   INPUT_LENS=128
#   BATCH_SIZES=1,2,4
#   REPEATS=1
#   WARMUP=1
#   MAX_TOKENS=1
#   IMAGE_SIZE=224
#   MM_MIN_PIXELS=163840
#   MM_MAX_PIXELS=196608
#   DISABLE_MM_DO_RESCALE=0
#   ENABLE_MM_EMBEDS=0
#   INPUT_MODE=tokenized
#   ENABLE_PREFIX_CACHING=0
#   ENABLE_VLLM_PYTHON_PROFILE=0
#   OTLP_TRACES_ENDPOINT=http://localhost:4317
#   OTLP_TRACES_PROTOCOL=grpc
#   COLLECT_DETAILED_TRACES=all
#   OUTPUT_DIR=/workspace/project/RL-learning/vllm-test/outputs/benchmark_qwen35_mm_smoke
#   QWEN35_PLUGIN_LOG_FORWARD=1
#   QWEN35_PLUGIN_ENABLE_HOOK=1
#   ANALYZE_INPUTS="/path/to/run1 /path/to/run2"
#   ANALYZE_OUTPUT_DIR=/workspace/project/RL-learning/vllm-test/outputs/benchmark_qwen35_mm_bs_sweep_combined
#   PLATEAU_RATIO=0.97
#   TIMELINE_OUTPUT_DIR=/workspace/project/RL-learning/vllm-test/outputs/accounting_smoke

MODE="${1:-online}"

MODEL_PATH="${MODEL_PATH:-/share/models/Qwen3.5-2B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.5-2b}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
PROMPT="${PROMPT:-你好，用一句话介绍你自己。}"

export VLLM_PLUGINS="${VLLM_PLUGINS:-qwen35_custom_model}"

case "${MODE}" in
  online)
    vllm serve "${MODEL_PATH}" \
      --served-model-name "${SERVED_MODEL_NAME}" \
      --host "${HOST}" \
      --port "${PORT}" \
      --dtype bfloat16 \
      --max-model-len "${MAX_MODEL_LEN}" \
      --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    ;;

  offline)
    python /workspace/project/RL-learning/vllm-test/scripts/offline_infer.py \
      --model "${MODEL_PATH}" \
      --prompt "${PROMPT}" \
      --max-model-len "${MAX_MODEL_LEN}" \
      --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    ;;

  benchmark-smoke)
    cmd=(
      python /workspace/project/RL-learning/vllm-test/scripts/benchmark_qwen35_mm.py
      --model "${MODEL_PATH}"
      --input-lens "${INPUT_LENS:-128}"
      --batch-sizes "${BATCH_SIZES:-1}"
      --repeats "${REPEATS:-1}"
      --warmup "${WARMUP:-1}"
      --max-model-len "${MAX_MODEL_LEN}"
      --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-0}"
      --max-tokens "${MAX_TOKENS:-1}"
      --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
      --image-size "${IMAGE_SIZE:-224}"
      --mm-min-pixels "${MM_MIN_PIXELS:-163840}"
      --mm-max-pixels "${MM_MAX_PIXELS:-196608}"
      --input-mode "${INPUT_MODE:-tokenized}"
      --output-dir "${OUTPUT_DIR:-/workspace/project/RL-learning/vllm-test/outputs/benchmark_qwen35_mm_smoke}"
    )
    if [[ "${DISABLE_MM_DO_RESCALE:-0}" == "1" ]]; then
      cmd+=(--disable-mm-do-rescale)
    fi
    if [[ "${ENABLE_MM_EMBEDS:-0}" == "1" ]]; then
      cmd+=(--enable-mm-embeds)
    fi
    if [[ "${ENABLE_PREFIX_CACHING:-0}" == "1" ]]; then
      cmd+=(--enable-prefix-caching)
    fi
    if [[ -n "${DATASET_JSONL:-}" ]]; then
      cmd+=(--dataset-jsonl "${DATASET_JSONL}")
    fi
    if [[ -n "${DATASET_LIMIT:-}" ]]; then
      cmd+=(--dataset-limit "${DATASET_LIMIT}")
    fi
    "${cmd[@]}"
    ;;

  benchmark-merged-smoke)
    cmd=(
      python /workspace/project/RL-learning/vllm-test/scripts/benchmark_qwen35_mm_merged_request.py
      --model "${MODEL_PATH}"
      --input-lens "${INPUT_LENS:-128}"
      --batch-sizes "${BATCH_SIZES:-1}"
      --repeats "${REPEATS:-1}"
      --warmup "${WARMUP:-1}"
      --max-model-len "${MAX_MODEL_LEN}"
      --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-0}"
      --max-tokens "${MAX_TOKENS:-1}"
      --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
      --image-size "${IMAGE_SIZE:-224}"
      --mm-min-pixels "${MM_MIN_PIXELS:-163840}"
      --mm-max-pixels "${MM_MAX_PIXELS:-196608}"
      --input-mode "${INPUT_MODE:-text}"
      --output-dir "${OUTPUT_DIR:-/workspace/project/RL-learning/vllm-test/outputs/benchmark_qwen35_mm_merged_request_smoke}"
    )
    if [[ "${DISABLE_MM_DO_RESCALE:-0}" == "1" ]]; then
      cmd+=(--disable-mm-do-rescale)
    fi
    if [[ "${ENABLE_MM_EMBEDS:-0}" == "1" ]]; then
      cmd+=(--enable-mm-embeds)
    fi
    if [[ "${ENABLE_PREFIX_CACHING:-0}" == "1" ]]; then
      cmd+=(--enable-prefix-caching)
    fi
    if [[ -n "${DATASET_JSONL:-}" ]]; then
      cmd+=(--dataset-jsonl "${DATASET_JSONL}")
    fi
    if [[ -n "${DATASET_LIMIT:-}" ]]; then
      cmd+=(--dataset-limit "${DATASET_LIMIT}")
    fi
    if [[ "${ENABLE_VLLM_PYTHON_PROFILE:-0}" == "1" ]]; then
      cmd+=(--enable-vllm-python-profile)
    fi
    if [[ -n "${OTLP_TRACES_ENDPOINT:-}" ]]; then
      cmd+=(--otlp-traces-endpoint "${OTLP_TRACES_ENDPOINT}")
    fi
    if [[ -n "${OTLP_TRACES_PROTOCOL:-}" ]]; then
      cmd+=(--otlp-traces-protocol "${OTLP_TRACES_PROTOCOL}")
    fi
    if [[ -n "${COLLECT_DETAILED_TRACES:-}" ]]; then
      cmd+=(--collect-detailed-traces "${COLLECT_DETAILED_TRACES}")
    fi
    "${cmd[@]}"
    ;;

  analyze)
    if [[ -z "${ANALYZE_INPUTS:-}" ]]; then
      echo "ANALYZE_INPUTS is required for analyze mode" >&2
      exit 2
    fi
    # shellcheck disable=SC2206
    analyze_inputs=(${ANALYZE_INPUTS})
    python /workspace/project/RL-learning/vllm-test/scripts/analyze_qwen35_bs_sweep.py       "${analyze_inputs[@]}"       --output-dir "${ANALYZE_OUTPUT_DIR:-/workspace/project/RL-learning/vllm-test/outputs/benchmark_qwen35_mm_bs_sweep_combined}"       --plateau-ratio "${PLATEAU_RATIO:-0.97}"
    ;;

  analyze-timeline)
    python /workspace/project/RL-learning/vllm-test/scripts/analyze_qwen35_timeline.py \
      "${TIMELINE_OUTPUT_DIR:-${OUTPUT_DIR:-/workspace/project/RL-learning/vllm-test/outputs/benchmark_qwen35_mm_merged_request_smoke}}"
    ;;

  *)
    echo "Unknown mode: ${MODE}" >&2
    echo "Usage: $0 {online|offline|benchmark-smoke|benchmark-merged-smoke|analyze|analyze-timeline}" >&2
    exit 2
    ;;
esac

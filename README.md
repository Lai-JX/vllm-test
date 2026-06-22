# vLLM Qwen3.5 Testbed

这个目录是 Qwen3.5 在 vLLM 下的实验工作区，主要用于：

- 通过插件机制非侵入式替换 Qwen3.5 model class。
- 跑 Qwen3.5 多模态 offline benchmark。
- 拆分 `renderer_mm`、ViT、LLM forward、vLLM 调度等阶段耗时。
- 对比原图输入、不同 `OMP_NUM_THREADS`、`enable_mm_embeds` 的性能影响。

当前主报告：

```text
docs/benchmark/qwen35_enable_mm_embeds_perf_report.md
```

历史 batch size sweep 和脚本字段说明：

```text
docs/benchmark/qwen35_mm_bs_benchmark.md
```

## 目录结构

- `qwen35_vllm_plugin/`：vLLM 插件工程。负责注册自定义 Qwen3.5 model class，并记录模型侧 profiling。
- `scripts/`：离线推理、benchmark、数据转换、timeline 分析和 OTel collector 启动脚本。
- `docs/model/`：Qwen3.5 多模态 forward 链路说明。
- `docs/benchmark/`：benchmark 使用说明、性能报告和 OTel trace 说明。
- `data/`：本地转换后的数据集，默认被 `.gitignore` 忽略。
- `outputs/`：benchmark 输出目录，默认只跟踪 `outputs/README.md`。

## 安装插件

先安装插件包，否则 vLLM 找不到 `qwen35_custom_model` entry point：

```bash
cd /workspace/project/RL-learning/vllm-test/qwen35_vllm_plugin
pip install -e .
```

验证插件是否能被 vLLM 加载：

```bash
VLLM_PLUGINS=qwen35_custom_model python -c "
from vllm.plugins import load_general_plugins
from vllm import ModelRegistry
load_general_plugins()
cls = ModelRegistry._try_load_model_cls('Qwen3_5ForConditionalGeneration')
print(cls.__module__ + ':' + cls.__name__)
"
```

期望输出：

```text
qwen35_vllm_plugin.model:CustomQwen35ForConditionalGeneration
```

插件细节见：

```text
qwen35_vllm_plugin/README.md
qwen35_vllm_plugin/PLUGIN_FLOW.md
```

## 统一启动脚本

大多数命令都走：

```bash
/workspace/project/RL-learning/vllm-test/scripts/run_qwen35_plugin.sh <mode>
```

支持的 mode：

- `online`：启动 `vllm serve`。
- `offline`：直接用 `vllm.LLM` 做一次离线推理。
- `benchmark-smoke`：普通 offline batch benchmark，每个 sample 是一个 vLLM 请求。
- `benchmark-merged-smoke`：把一个 logical batch 的文本和图片合并成一个 vLLM 请求，减少调度影响。
- `analyze`：汇总一个或多个 benchmark 输出目录。
- `analyze-timeline`：基于 `summary.csv` 和 `profile_records.jsonl` 生成 e2e 对账和 timeline。

脚本默认会设置：

```bash
VLLM_PLUGINS=qwen35_custom_model
```

## 离线推理

```bash
PROMPT="你好，用一句话介绍你自己。" \
/workspace/project/RL-learning/vllm-test/scripts/run_qwen35_plugin.sh offline
```

等价的显式命令：

```bash
VLLM_PLUGINS=qwen35_custom_model \
python /workspace/project/RL-learning/vllm-test/scripts/offline_infer.py \
  --model /share/models/Qwen3.5-2B \
  --prompt "你好，用一句话介绍你自己。"
```

## 在线服务

```bash
MODEL_PATH=/share/models/Qwen3.5-2B \
SERVED_MODEL_NAME=qwen3.5-2b \
HOST=0.0.0.0 \
PORT=8000 \
/workspace/project/RL-learning/vllm-test/scripts/run_qwen35_plugin.sh online
```

## 准备 Alpamayo 数据

从本地 Alpamayo PAI 数据导出 benchmark JSONL：

```bash
python /workspace/project/RL-learning/vllm-test/scripts/prepare_qwen35_dataset_from_alpamayo.py \
  --num-samples 200 \
  --output-dir /workspace/project/RL-learning/vllm-test/data/alpamayo_qwen35_200
```

导出的 JSONL 每行会保留原始 sample 的 prompt 和图片列表：

- `prompt`
- `images`
- `image_count`

脚本还会预计算供 `enable_mm_embeds=True` 使用的字段：

- `image_embeds`
- `image_grid_thw`
- `image_embeds_shape`
- `image_grid_thw_shape`
- `mm_min_pixels`
- `mm_max_pixels`
- `mm_do_rescale`

默认图片 processor 配置对齐 Alpamayo：

```text
do_rescale=True
min_pixels=163840
max_pixels=196608
```

如果只想快速生成轻量 smoke 数据，可以加：

```bash
--frames-per-clip 1 --num-samples 8
```

## 普通 Batch Benchmark

每个 sample 作为一个 vLLM 请求发送：

```bash
DATASET_JSONL=/workspace/project/RL-learning/vllm-test/data/alpamayo_qwen35_200/dataset.jsonl \
DATASET_LIMIT=64 \
BATCH_SIZES=1,2,4,8 \
REPEATS=3 \
WARMUP=1 \
OUTPUT_DIR=/workspace/project/RL-learning/vllm-test/outputs/alpamayo_qwen35_batch \
/workspace/project/RL-learning/vllm-test/scripts/run_qwen35_plugin.sh benchmark-smoke
```

适合观察真实 vLLM offline batching 下的吞吐。脚本会自动扫描每条样本的图片数量并设置 `limit_mm_per_prompt`。

## Merged-Request Benchmark

把同一个 logical batch 的所有文本和图片合并成一个请求发送：

```bash
DATASET_JSONL=/workspace/project/RL-learning/vllm-test/data/alpamayo_qwen35_200/dataset.jsonl \
DATASET_LIMIT=8 \
BATCH_SIZES=1,2,4,8 \
REPEATS=1 \
WARMUP=1 \
INPUT_MODE=tokenized \
ENABLE_VLLM_PYTHON_PROFILE=1 \
OUTPUT_DIR=/workspace/project/RL-learning/vllm-test/outputs/alpamayo_qwen35_merged \
/workspace/project/RL-learning/vllm-test/scripts/run_qwen35_plugin.sh benchmark-merged-smoke
```

这个模式用于减少多个请求调度带来的干扰，更适合拆分分析：

- prompt 构造时间
- 多模态 processor 时间
- ViT 时间
- LLM forward 时间
- vLLM engine/worker 时间
- e2e residual

`benchmark-merged-smoke` 默认 `INPUT_MODE=text`；如果要跳过 vLLM 内部文本 tokenize，可以显式设置 `INPUT_MODE=tokenized`。

## enable_mm_embeds 对照

如果 JSONL 每行已经有 `image_embeds` 和 `image_grid_thw`，可以开启：

```bash
ENABLE_MM_EMBEDS=1 \
DATASET_JSONL=/workspace/project/RL-learning/vllm-test/data/alpamayo_qwen35_200/dataset.jsonl \
DATASET_LIMIT=8 \
BATCH_SIZES=1,2,4,8 \
REPEATS=1 \
WARMUP=1 \
INPUT_MODE=tokenized \
OUTPUT_DIR=/workspace/project/RL-learning/vllm-test/outputs/alpamayo_qwen35_mm_embeds \
/workspace/project/RL-learning/vllm-test/scripts/run_qwen35_plugin.sh benchmark-merged-smoke
```

该模式会把预计算的视觉 embedding 作为 `multi_modal_data["image"]` 输入 vLLM，预期 `vit_calls=0`。它适合评估“跳过 ViT 和图片 processor 后还剩多少耗时”，不代表完整图片端到端性能。

## 输出文件

benchmark 输出目录通常包含：

- `run_config.json`：本次参数快照。
- `summary.csv`：每个 warmup/measured group 的原始汇总。
- `aggregate_summary.csv`：单次运行内按 batch size 聚合。
- `profile_records.jsonl`：插件写出的模型/vLLM profiling 原始记录。
- `accounting_summary.csv`：运行 `analyze-timeline` 后生成的 e2e 对账。
- `timeline.csv`：运行 `analyze-timeline` 后生成的 request timeline。

`summary.csv` 中常用字段：

- `e2e_ms`：一次 `llm.generate(...)` 的端到端 wall time。
- `vit_cuda_sum_ms`：当前 generate 内所有 `self.visual(...)` CUDA 时间总和。
- `llm_cuda_sum_ms`：底层 LLM forward CUDA 时间总和。
- `qwen_forward_cuda_sum_ms`：Qwen3.5 顶层 forward CUDA 时间总和。
- `engine_requests_per_s`：vLLM engine request 吞吐，merged 模式下通常是 1 个请求。
- `logical_items_per_s`：logical item 吞吐，merged 模式下更接近业务样本吞吐。
- `benchmark_prepare_mm_data_wall_ms`：benchmark 构造 `multi_modal_data` 的时间。
- `vllm_renderer_process_multimodal_wall_sum_ms`：vLLM renderer 多模态处理时间。

完整字段说明见：

```text
docs/benchmark/qwen35_mm_bs_benchmark.md
```

## 分析结果

汇总多个 benchmark 输出：

```bash
ANALYZE_INPUTS="/workspace/project/RL-learning/vllm-test/outputs/run1 /workspace/project/RL-learning/vllm-test/outputs/run2" \
ANALYZE_OUTPUT_DIR=/workspace/project/RL-learning/vllm-test/outputs/combined \
/workspace/project/RL-learning/vllm-test/scripts/run_qwen35_plugin.sh analyze
```

生成：

- `combined_by_bs_summary.csv`
- `combined_by_bs_summary.md`
- `recommendations.csv`

对单个 merged benchmark 做 timeline：

```bash
TIMELINE_OUTPUT_DIR=/workspace/project/RL-learning/vllm-test/outputs/alpamayo_qwen35_merged \
/workspace/project/RL-learning/vllm-test/scripts/run_qwen35_plugin.sh analyze-timeline
```

如果要生成静态 SVG / 交互 HTML timeline：

```bash
python /workspace/project/RL-learning/vllm-test/scripts/render_qwen35_request_timeline.py \
  /workspace/project/RL-learning/vllm-test/outputs/alpamayo_qwen35_merged/timeline.csv \
  --batch-size 2 \
  --output-html /workspace/project/RL-learning/vllm-test/outputs/alpamayo_qwen35_merged/bs2_timeline.html \
  --output-svg /workspace/project/RL-learning/vllm-test/outputs/alpamayo_qwen35_merged/bs2_timeline.svg
```

## OTel Trace

如果需要 vLLM OpenTelemetry trace 离线落盘，先启动 collector：

```bash
OUTPUT_DIR=/workspace/project/RL-learning/vllm-test/outputs/otel_traces \
bash /workspace/project/RL-learning/vllm-test/scripts/run_otel_file_collector.sh
```

另一个终端运行 merged benchmark：

```bash
OTLP_TRACES_ENDPOINT=http://localhost:4317 \
OTLP_TRACES_PROTOCOL=grpc \
COLLECT_DETAILED_TRACES=all \
/workspace/project/RL-learning/vllm-test/scripts/run_qwen35_plugin.sh benchmark-merged-smoke
```

collector 会把 trace 写到：

```text
<OUTPUT_DIR>/traces.json
```

详细说明见：

```text
docs/benchmark/otel_file_traces.md
```

## 当前结论入口

- [Qwen3.5 enable_mm_embeds 性能对比报告](/workspace/project/RL-learning/vllm-test/docs/benchmark/qwen35_enable_mm_embeds_perf_report.md)：当前主报告，基于 Alpamayo 数据、merged request、`INPUT_MODE=tokenized`，对比原图输入、不同 `OMP_NUM_THREADS` 和 `enable_mm_embeds`。
- [Qwen3.5 多模态 Batch Size 性能测试](/workspace/project/RL-learning/vllm-test/docs/benchmark/qwen35_mm_bs_benchmark.md)：benchmark 脚本、字段、历史 synthetic sweep 和分析命令说明。
- [Qwen3.5 多模态 forward 链路](/workspace/project/RL-learning/vllm-test/docs/model/qwen3_5_multimodal_forward.md)：说明 `pixel_values -> self.visual -> visual embeddings -> merge -> LLM forward` 在 vLLM 中的调用关系。

历史 synthetic 128-token benchmark 推荐过 `bs=48`，该结论只适用于 synthetic controlled sweep，不应直接替代当前真实 Alpamayo 数据下的结论。

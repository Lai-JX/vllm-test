# vLLM Qwen3.5 Plugin Testbed

这个目录用于非侵入式测试 `/share/models/Qwen3.5-2B` 在 vLLM 下的插件注册、离线推理和性能 profiling。

## 目录结构

- `qwen35_vllm_plugin/`：vLLM 插件工程，通过 `vllm.general_plugins` entry point 注册自定义 Qwen3.5 model class。
- `scripts/`：启动、离线推理、benchmark 和分析脚本。
- `docs/`：插件流程、模型 forward 链路和 benchmark 结论文档。
- `outputs/`：保留的 benchmark 结果。只保留可复现的正式结果和最新 smoke。

## 常用命令

在线服务：

```bash
/workspace/project/RL-learning/vllm-test/scripts/run_qwen35_plugin.sh online
```

普通 offline batch benchmark：

```bash
OUTPUT_DIR=/workspace/project/RL-learning/vllm-test/outputs/new_bs_sweep \
BATCH_SIZES=32,48,64 \
INPUT_LENS=128 \
REPEATS=3 \
WARMUP=1 \
/workspace/project/RL-learning/vllm-test/scripts/run_qwen35_plugin.sh benchmark-smoke
```

合并 logical batch 为单请求的 benchmark：

```bash
OUTPUT_DIR=/workspace/project/RL-learning/vllm-test/outputs/new_merged_sweep \
BATCH_SIZES=32,48,64 \
INPUT_LENS=128 \
INPUT_MODE=tokenized \
ENABLE_VLLM_PYTHON_PROFILE=1 \
REPEATS=3 \
WARMUP=1 \
/workspace/project/RL-learning/vllm-test/scripts/run_qwen35_plugin.sh benchmark-merged-smoke
```

merged benchmark 默认保持历史行为 `INPUT_MODE=text`；设置 `INPUT_MODE=tokenized` 后，合并后的单个请求会用 `prompt_token_ids` 发送给 vLLM。
设置 `ENABLE_VLLM_PYTHON_PROFILE=1` 后，会额外采集 vLLM Python 层关键函数的 wall time，用于解释模型 forward 外的开销。

对 merged benchmark 做 e2e 对账和 timeline 分析：

```bash
TIMELINE_OUTPUT_DIR=/workspace/project/RL-learning/vllm-test/outputs/new_merged_sweep \
/workspace/project/RL-learning/vllm-test/scripts/run_qwen35_plugin.sh analyze-timeline
```

会生成：

- `accounting_summary.csv`：每个 measured request 的 e2e 对账。
- `timeline.csv`：按相对时间排序的 benchmark / vLLM / model phase。

汇总分析：

```bash
ANALYZE_INPUTS="/workspace/project/RL-learning/vllm-test/outputs/run1 /workspace/project/RL-learning/vllm-test/outputs/run2" \
ANALYZE_OUTPUT_DIR=/workspace/project/RL-learning/vllm-test/outputs/combined \
/workspace/project/RL-learning/vllm-test/scripts/run_qwen35_plugin.sh analyze
```

保存 vLLM OTel trace 以便离线查看：

```bash
OUTPUT_DIR=/workspace/project/RL-learning/vllm-test/outputs/new_merged_sweep/otel_traces \
bash /workspace/project/RL-learning/vllm-test/scripts/run_otel_file_collector.sh
```

该脚本不依赖 Docker；如果本机没有 `otelcol-contrib`，会默认下载到 `vllm-test/.tools/otelcol-contrib/`。

然后在另一个终端给 merged benchmark 加：

```bash
OTLP_TRACES_ENDPOINT=http://localhost:4317 \
OTLP_TRACES_PROTOCOL=grpc \
COLLECT_DETAILED_TRACES=all \
/workspace/project/RL-learning/vllm-test/scripts/run_qwen35_plugin.sh benchmark-merged-smoke
```

详细说明见 [vLLM OTel Trace 落盘查看](/workspace/project/RL-learning/vllm-test/docs/benchmark/otel_file_traces.md)。

从 Alpamayo 本地 PAI 数据导出 Qwen3.5 benchmark JSONL：

```bash
python /workspace/project/RL-learning/vllm-test/scripts/prepare_qwen35_dataset_from_alpamayo.py \
  --num-samples 64 \
  --output-dir /workspace/project/RL-learning/vllm-test/data/alpamayo_qwen35
```

默认会保留 Alpamayo 原始 sample 的完整 vLLM prompt，并导出该 sample 对应的全部图片；如果只想临时做轻量 smoke，可以显式加 `--frames-per-clip 1`。

导出后可以这样跑真实数据：

```bash
DATASET_JSONL=/workspace/project/RL-learning/vllm-test/data/alpamayo_qwen35/dataset.jsonl \
DATASET_LIMIT=64 \
BATCH_SIZES=1,2,4,8 \
REPEATS=3 \
WARMUP=1 \
OUTPUT_DIR=/workspace/project/RL-learning/vllm-test/outputs/alpamayo_qwen35_bs_sweep \
/workspace/project/RL-learning/vllm-test/scripts/run_qwen35_plugin.sh benchmark-smoke
```

benchmark 不修改每条样本的 prompt；脚本会自动检测每条样本的图片数量并设置 `limit_mm_per_prompt`。每个请求仍会生成唯一 `multi_modal_uuids`，用于避免多模态 cache 复用。

benchmark 默认对齐 Alpamayo 原始多模态 processor 配置：`do_rescale=True`、`min_pixels=163840`、`max_pixels=196608`。需要覆盖时可设置 `MM_MIN_PIXELS`、`MM_MAX_PIXELS` 或 `DISABLE_MM_DO_RESCALE=1`。

benchmark 默认显式关闭 vLLM prefix cache；只有需要做 cache 对照实验时才设置 `ENABLE_PREFIX_CACHING=1`。

如果 JSONL 每行已经包含预计算视觉 embedding：

- `image_embeds`：`.pt` 文件路径
- `image_grid_thw`：`.pt` 文件路径

可以开启 embedding 输入，跳过图片预处理后的 ViT forward：

```bash
ENABLE_MM_EMBEDS=1 \
DATASET_JSONL=/workspace/project/RL-learning/vllm-test/data/alpamayo_qwen35_200/dataset.jsonl \
DATASET_LIMIT=1 \
BATCH_SIZES=1 \
REPEATS=1 \
WARMUP=1 \
OUTPUT_DIR=/workspace/project/RL-learning/vllm-test/outputs/alpamayo_qwen35_mm_embeds \
/workspace/project/RL-learning/vllm-test/scripts/run_qwen35_plugin.sh benchmark-merged-smoke
```

此模式会传 `enable_mm_embeds=True` 给 vLLM，并把每条样本的 `image_embeds` / `image_grid_thw` 作为 `multi_modal_data["image"]`。预期 `vit_calls=0`，适合单独观察 LLM forward 和 embedding 输入链路，不适合作为真实图片端到端结果。

## 当前结论

当前有两类结论文档：

- [Qwen3.5 enable_mm_embeds 性能对比报告](/workspace/project/RL-learning/vllm-test/docs/benchmark/qwen35_enable_mm_embeds_perf_report.md)：当前主报告，基于 Alpamayo 数据、merged request、`--input-mode tokenized`，对比原图输入、不同 `OMP_NUM_THREADS` 和 `enable_mm_embeds`。
- [Qwen3.5 多模态 Batch Size 性能测试](/workspace/project/RL-learning/vllm-test/docs/benchmark/qwen35_mm_bs_benchmark.md)：benchmark 脚本和历史 batch sweep 的说明。

历史 synthetic 128-token benchmark 推荐 `bs=48`：它已达到 `bs=64` 峰值吞吐的 99% 以上，同时模型侧总耗时明显低于 `bs=64`。该结论用于 synthetic controlled sweep，不应直接替代真实 Alpamayo 数据下的结论。

# Qwen3.5 多模态 Batch Size 性能测试

本文档记录 `/workspace/project/RL-learning/vllm-test` 下现有的离线 benchmark 链路，用于测试 Qwen3.5 多模态模型在不同 batch size 下的性能。当前推荐把 batch size 作为主要自变量；输入 token 数接口仍保留，主要用于 synthetic controlled sweep。

当前真实 Alpamayo 数据、merged request、`enable_mm_embeds` 对比的主报告见：

```text
/workspace/project/RL-learning/vllm-test/docs/benchmark/qwen35_enable_mm_embeds_perf_report.md
```

本文后半部分的 synthetic 结果是历史 controlled sweep，用于说明普通 offline batch 的平台点，不代表真实图片端到端链路的最终结论。

## 测试目标

主要观测指标：

- `vit_cuda_sum_ms`：当前 `llm.generate(...)` 调用内所有 `self.visual(...)` 记录的 CUDA 时间总和。
- `llm_cuda_sum_ms`：当前 `llm.generate(...)` 调用内所有底层 LLM forward 记录的 CUDA 时间总和。
- `qwen_forward_cuda_sum_ms`：当前 `llm.generate(...)` 调用内所有 `Qwen3_5ForConditionalGeneration.forward(...)` 顶层 CUDA 时间总和。
- `model_total_cuda_sum_ms`：当前脚本按 `vit_cuda_sum_ms + qwen_forward_cuda_sum_ms` 计算的模型侧总时间。
- `vit_cuda_first_ms` / `llm_cuda_first_ms` / `qwen_forward_cuda_first_ms`：当前 `llm.generate(...)` 调用内对应阶段第一条记录的 CUDA 时间，用于区分一次请求内多次 phase 调用时的首个阶段耗时。
- `model_total_cuda_first_ms`：当前脚本按 `vit_cuda_first_ms + qwen_forward_cuda_first_ms` 计算的首个模型侧阶段耗时。
- `e2e_ms`：一次 `llm.generate(...)` 的端到端耗时，包含调度、预处理、模型执行、采样与输出封装。
- `requests_per_s` / `total_requests_per_s`：普通 offline batch benchmark 的请求吞吐。
- `engine_requests_per_s` / `logical_items_per_s`：merged-request benchmark 的 engine request 吞吐和 logical item 吞吐。
- `prompt_tokens_per_s` / `total_prompt_tokens_per_s`：输入 token 吞吐。

输出文件：

- `summary.csv`：每个 warmup/measured batch 的原始汇总。
- `aggregate_summary.csv`：单次运行内按 batch size 聚合。
- `profile_records.jsonl`：插件写出的原始阶段耗时记录。
- `run_config.json`：本次运行的参数快照，analyzer 会优先用它识别 `max_num_batched_tokens` 等配置。

## 启动入口

统一入口是：

```bash
/workspace/project/RL-learning/vllm-test/scripts/run_qwen35_plugin.sh benchmark-smoke
```

脚本会设置：

```bash
export VLLM_PLUGINS=qwen35_custom_model
```

从而让 vLLM 加载 `qwen35_vllm_plugin` 插件。benchmark 脚本还会默认打开：

```bash
QWEN35_PLUGIN_PROFILE=1
QWEN35_PLUGIN_PROFILE_SYNC=1
```

这样插件会把 ViT、LLM forward、Qwen forward 的计时写入 `profile_records.jsonl`。

benchmark 脚本初始化 `vllm.LLM` 时会显式传入 `enable_prefix_caching=False`，避免 vLLM prefix cache 影响不同 batch size 的耗时对比。如果需要专门做 prefix cache 对照实验，可以设置 `ENABLE_PREFIX_CACHING=1`，或直接给 Python 脚本传 `--enable-prefix-caching`。

benchmark 脚本的全局 `mm_processor_kwargs` 默认对齐 `/share/models/Alpamayo-1.5-10B/config.json`：

```python
{
    "do_rescale": True,
    "min_pixels": 163840,
    "max_pixels": 196608,
}
```

如果需要做不同图像像素预算的对照实验，可以设置 `MM_MIN_PIXELS`、`MM_MAX_PIXELS` 或 `DISABLE_MM_DO_RESCALE=1`，也可以直接给 Python 脚本传 `--mm-min-pixels`、`--mm-max-pixels`、`--disable-mm-do-rescale`。

如果 JSONL 每行已经包含预计算视觉 embedding，也可以设置：

```bash
ENABLE_MM_EMBEDS=1
```

这会给 vLLM 传 `enable_mm_embeds=True`，并要求每条样本包含：

- `image_embeds`：`.pt` 文件路径。
- `image_grid_thw`：`.pt` 文件路径。

该模式会跳过 ViT，适合拆分分析 ViT 之后的链路；它不代表完整图片端到端性能。

## 真实数据集模式

推荐用真实数据集直接测试 batch size，不再强行控制 token 数。

```bash
DATASET_JSONL=/path/to/dataset.jsonl \
DATASET_LIMIT=256 \
BATCH_SIZES=32,40,48,64 \
REPEATS=3 \
WARMUP=1 \
OUTPUT_DIR=/workspace/project/RL-learning/vllm-test/outputs/qwen35_dataset_bs_sweep \
/workspace/project/RL-learning/vllm-test/scripts/run_qwen35_plugin.sh benchmark-smoke
```

JSONL 每行支持以下字段：

- 图片字段：`image`、`image_path`、`image_file`，或复数形式 `images`、`image_paths`、`image_files`。相对路径会按 JSONL 文件所在目录解析。
- 文本字段：`text`、`question`、`query` 或完整 `prompt`。
- 也可以直接提供 `prompt_token_ids`。

如果使用 `scripts/prepare_qwen35_dataset_from_alpamayo.py` 导出的 Alpamayo JSONL，每行会保留原始 vLLM prompt，并把同一个 sample 的全部图片写入 `images` 列表。`benchmark_qwen35_mm.py` 会在初始化 vLLM 前扫描 JSONL，自动设置 `limit_mm_per_prompt`，因此不需要手动调整每条样本的图片上限。

benchmark 不再向 prompt 追加 benchmark nonce，保证每个 sample 的文本与导出时一致。为了避免多模态 cache 复用，脚本仍会为每个请求的图片设置唯一 `multi_modal_uuids`。

如果只提供 `text/question/query`，脚本会自动包装成 Qwen3-VL 风格 prompt：

```text
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>{question}<|im_end|>
<|im_start|>assistant
```

## Synthetic Token 模式

保留 token 数接口，适合做受控输入长度实验：

```bash
BATCH_SIZES=1,2,4,8,16,32 \
INPUT_LENS=128,256,512,1024 \
REPEATS=3 \
WARMUP=1 \
OUTPUT_DIR=/workspace/project/RL-learning/vllm-test/outputs/qwen35_synthetic_len_sweep \
/workspace/project/RL-learning/vllm-test/scripts/run_qwen35_plugin.sh benchmark-smoke
```

默认 `INPUT_MODE=tokenized`，脚本直接构造 `prompt_token_ids`；也可以设置：

```bash
INPUT_MODE=text
```

让 vLLM 走文本 prompt 输入。

对于 `benchmark-merged-smoke` / `benchmark_qwen35_mm_merged_request.py`，默认保持历史行为 `INPUT_MODE=text`，也可以设置：

```bash
INPUT_MODE=tokenized
```

此时脚本会先把 logical batch 合并成单个 prompt，再把该 merged prompt 编码成 `prompt_token_ids` 作为一个 vLLM 请求发送；图片合并和 `multi_modal_uuids` 逻辑不变。

## 额外 vLLM Python 计时

如果 OTel 的 request-level 信息还不够，可以在 merged benchmark 中打开轻量 Python wall-time 计时：

```bash
ENABLE_VLLM_PYTHON_PROFILE=1 \
/workspace/project/RL-learning/vllm-test/scripts/run_qwen35_plugin.sh benchmark-merged-smoke
```

或直接传 Python 参数：

```bash
--enable-vllm-python-profile
```

这不会修改 vLLM 源码，而是通过插件在运行时给关键函数加 wrapper。原始记录会写入 `profile_records.jsonl`，并在 `summary.csv` / `aggregate_summary.csv` 中增加若干列，例如：

- `benchmark_build_prompt_wall_ms`
- `benchmark_prepare_mm_data_wall_ms`
- `benchmark_encode_prompt_wall_ms`
- `benchmark_llm_generate_wall_ms`
- `benchmark_cuda_sync_wall_ms`
- `benchmark_e2e_accounted_wall_ms`
- `benchmark_e2e_residual_wall_ms`
- `benchmark_run_total_wall_ms`
- `vllm_run_engine_wall_sum_ms`
- `vllm_add_completion_requests_wall_sum_ms`
- `vllm_preprocess_cmpl_wall_sum_ms`
- `vllm_renderer_render_cmpl_wall_sum_ms`
- `vllm_renderer_tokenize_prompts_wall_sum_ms`
- `vllm_renderer_process_for_engine_wall_sum_ms`
- `vllm_renderer_process_tokens_wall_sum_ms`
- `vllm_renderer_process_multimodal_wall_sum_ms`
- `vllm_engine_step_wall_sum_ms`
- `vllm_process_outputs_wall_sum_ms`
- `vllm_worker_execute_model_wall_sum_ms`
- `vllm_gpu_execute_model_wall_sum_ms`
- `vllm_execute_mm_encoder_wall_sum_ms`
- `vllm_gather_mm_embeddings_wall_sum_ms`

这些指标用于解释 `e2e_ms` 中模型 forward 之外的 Python/vLLM 调度链路开销；它们是 wall time，不能替代 torch profile 的 kernel/op 级分析，也不能替代插件已有的 `vit_cuda_*` / `llm_cuda_*` CUDA 计时。

进一步生成 e2e 对账和 timeline：

```bash
TIMELINE_OUTPUT_DIR=<output-dir> \
/workspace/project/RL-learning/vllm-test/scripts/run_qwen35_plugin.sh analyze-timeline
```

输出：

- `accounting_summary.csv`：按 request 汇总 `e2e_ms`、`benchmark_e2e_accounted_wall_ms`、`benchmark_e2e_residual_wall_ms`、`model_total_wall_sum_ms`、`e2e_minus_model_total_wall_ms`。
- `timeline.csv`：按 `benchmark:run_total` 起点排序展示 benchmark / vLLM / model phase 的 `start_rel_ms`、`end_rel_ms`、`wall_ms`。

当前正式示例输出：

```text
/workspace/project/RL-learning/vllm-test/outputs/perf_report_20260620_181529
```

该目录包含原图输入、不同 `OMP_NUM_THREADS`、`enable_mm_embeds` 的 merged-request 对比结果。详细解读见：

```text
/workspace/project/RL-learning/vllm-test/docs/benchmark/qwen35_enable_mm_embeds_perf_report.md
```

从 `accounting_summary.csv` 看，`e2e_ms` 可以被 benchmark 层的非重叠阶段基本闭合；如果 `e2e_ms` 明显大于 `renderer_mm_ms + vit_ms + llm_ms`，差值通常来自 vLLM 的 request add、engine step、worker execute、输出处理，以及 CUDA event 计时和 Python wall time 的口径差异。

如果要进一步拆 `vllm_preprocess_cmpl_wall_sum_ms`，新版本还会记录 renderer 内部阶段。重点看：

- `vllm_renderer_render_prompts_wall_sum_ms`
- `vllm_renderer_tokenize_prompts_wall_sum_ms`
- `vllm_renderer_process_for_engine_wall_sum_ms`
- `vllm_renderer_process_tokens_wall_sum_ms`
- `vllm_renderer_process_multimodal_wall_sum_ms`

其中最可疑的是 `vllm_renderer_process_multimodal_wall_sum_ms`，它对应 `Renderer._process_multimodal(...) -> mm_processor.apply(...)`，通常包含图片 resize / rescale / normalize / placeholder 对齐 / 多模态输入构造。

如果需要把 `timeline.csv` 渲染成更直观的请求时间线，可以使用：

```bash
python /workspace/project/RL-learning/vllm-test/scripts/render_qwen35_request_timeline.py \
  /workspace/project/RL-learning/vllm-test/outputs/perf_report_20260620_181529/raw_omp8/timeline.csv \
  --batch-size 2 \
  --group-idx 0 \
  --repeat-idx 0 \
  --output /workspace/project/RL-learning/vllm-test/docs/benchmark/assets/qwen35_enable_mm_embeds/raw_omp8_bs2_timeline.html \
  --svg-output /workspace/project/RL-learning/vllm-test/docs/benchmark/assets/qwen35_enable_mm_embeds/raw_omp8_bs2_timeline.svg
```

## 避免 Cache Hit

benchmark 已显式关闭 vLLM prefix cache，因此不再通过向 prompt 注入 nonce 来制造文本差异。为了避免多次发送相同图片而命中多模态 cache，脚本仍会给每次多模态输入构造唯一 UUID，并设置：

```python
multi_modal_uuids = {"image": mm_uuids}
```

这样即使图片内容相同，不同请求也不会复用相同的多模态 UUID。merged-request 场景下，一个 engine request 内可能包含多张图片，因此 `mm_uuids` 是按图片数量展开的列表。

脚本内还有全局唯一性校验：

- `mm_uuids_unique`：单个 batch 内多模态 UUID 是否唯一。
- `mm_uuids_unique_global`：本次运行中是否和之前所有 batch 的多模态 UUID 重复。
- 如果发现重复，会直接抛 `RuntimeError`，避免继续产出无效性能数据。

不建议删除 `multi_modal_uuids` 逻辑。只有在明确想观察多模态 cache 命中行为时，才应手动去掉这部分 UUID 设置。

## 分析多个 Sweep

多个输出目录可以用 analyze 模式汇总：

```bash
ANALYZE_INPUTS="/path/to/run1 /path/to/run2" \
ANALYZE_OUTPUT_DIR=/workspace/project/RL-learning/vllm-test/outputs/qwen35_bs_sweep_combined \
PLATEAU_RATIO=0.97 \
/workspace/project/RL-learning/vllm-test/scripts/run_qwen35_plugin.sh analyze
```

分析脚本会读取每个目录下的 `summary.csv`，只使用 `is_warmup=False` 的 measured rows，并生成：

- `combined_by_bs_summary.csv`
- `recommendations.csv`
- `combined_by_bs_summary.md`

分析会按 `config + source + target_prompt_tokens + input_mode + batch_size` 分组，避免把 dataset、synthetic、不同 token 长度或不同输入模式的结果混在一起。

`PLATEAU_RATIO=0.97` 的含义是：推荐吞吐达到最佳吞吐 97% 以上的最小 batch size。这个策略用于避免只为了很小的吞吐提升而选择过大的 batch size。

## 历史 Synthetic 结果

新版 synthetic 128-token prompt 已完成两轮正式 sweep，所有 measured row 都满足 `request_ids_unique_global=True`。

```text
/workspace/project/RL-learning/vllm-test/outputs/benchmark_qwen35_mm_bs_global_unique_combined/combined_by_bs_summary.md
```

合并 `bs=4,8,16,32,40,48,56,64` 后的结果：

| bs | groups | req/s | per_req_ms | vit_ms | llm_ms | model_total_ms | global_unique |
|---:|---:|---:|---:|---:|---:|---:|---|
| 4 | 2 | 19.26 | 51.91 | 51.83 | 121.59 | 179.87 | True |
| 8 | 2 | 31.97 | 31.28 | 39.28 | 174.75 | 220.39 | True |
| 16 | 2 | 39.79 | 25.13 | 84.59 | 248.52 | 341.93 | True |
| 24 | 3 | 45.01 | 22.22 | 129.21 | 357.49 | 493.99 | True |
| 32 | 5 | 47.76 | 20.94 | 133.53 | 450.68 | 590.14 | True |
| 40 | 5 | 47.19 | 21.19 | 198.06 | 561.25 | 769.16 | True |
| 48 | 5 | 51.03 | 19.60 | 210.38 | 663.23 | 879.62 | True |
| 56 | 3 | 50.04 | 19.98 | 276.42 | 762.93 | 1049.24 | True |
| 64 | 5 | 51.32 | 19.49 | 283.47 | 874.71 | 1167.17 | True |

按 `PLATEAU_RATIO=0.97` 的推荐结果：

| config | best bs | best req/s | recommended bs | recommended req/s |
|---|---:|---:|---:|---:|
| default | 64 | 51.32 | 48 | 51.03 |

结论：

- synthetic 场景下吞吐在 `bs=48` 后进入平台区。
- `bs=64` 的吞吐最高，但只比 `bs=48` 高约 0.6%。
- `bs=48` 已达到峰值 97% 以上，同时 `model_total_ms` 明显低于 `bs=64`，因此 synthetic controlled sweep 推荐 `bs=48`。
- `bs=32` 已经接近平台区，但合并结果约为 `47.76 req/s`，低于 `bs=48` 约 6.4%。

补充对照：旧的 `max_num_batched_tokens=16384` 试验没有带来明显收益，`bs=32` 更慢，`bs=48` 与 default 接近；synthetic 场景继续使用 default 配置。

## 建议的下一步

1. 真实 Alpamayo 数据下优先参考 `qwen35_enable_mm_embeds_perf_report.md`，该报告已经覆盖 `bs=1,2,4,8`、`OMP_NUM_THREADS=1/4/8` 和 `enable_mm_embeds`。
2. 如果要把趋势变成正式数字，建议把 `dataset-limit` 提高到几十或上百，并把 `repeats` 提高到 3。
3. 如果关注完整图片端到端性能，以原图输入路径为主；如果要拆分 ViT 之后的链路，再使用 `ENABLE_MM_EMBEDS=1`。
4. 普通 offline batch sweep 和 merged-request sweep 的吞吐字段不同，分析时不要混合比较 `total_requests_per_s` 与 `logical_items_per_s`。

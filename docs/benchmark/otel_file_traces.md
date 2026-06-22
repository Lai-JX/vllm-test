# vLLM OTel Trace 落盘查看

`benchmark_qwen35_mm_merged_request.py` 支持把 vLLM OpenTelemetry trace 发到 OTLP endpoint。OTel 本身不是直接写本地 trace 文件的工具；要像 torch trace 一样离线看，需要在旁边启动一个 OpenTelemetry Collector，把 trace 接住后写成 JSON 文件。

## 启动 Collector

开第一个终端：

```bash
OUTPUT_DIR=/workspace/project/RL-learning/vllm-test/outputs/cuda_graph_smoke_merge/otel_traces \
bash /workspace/project/RL-learning/vllm-test/scripts/run_otel_file_collector.sh
```

默认监听：

```text
http://localhost:4317  # OTLP gRPC
http://localhost:4318  # OTLP HTTP
```

默认输出：

```text
/workspace/project/RL-learning/vllm-test/outputs/cuda_graph_smoke_merge/otel_traces/traces.json
```

脚本优先使用本机 `otelcol-contrib`；如果本机没有，会默认下载固定版本的 `otelcol-contrib` 到 `/workspace/project/RL-learning/vllm-test/.tools/otelcol-contrib/` 后再启动。这里使用 contrib 版本是因为落盘需要 `file` exporter。默认版本由 `OTEL_COLLECTOR_VERSION` 控制，当前是 `0.154.0`。

如果机器不能联网，可以手动指定：

```bash
OTEL_COLLECTOR_BIN=/path/to/otelcol-contrib \
OUTPUT_DIR=/workspace/project/RL-learning/vllm-test/outputs/run1/otel_traces \
bash /workspace/project/RL-learning/vllm-test/scripts/run_otel_file_collector.sh
```

如果不想自动下载，可以设置：

```bash
OTEL_COLLECTOR_AUTO_DOWNLOAD=0 \
OUTPUT_DIR=/workspace/project/RL-learning/vllm-test/outputs/run1/otel_traces \
bash /workspace/project/RL-learning/vllm-test/scripts/run_otel_file_collector.sh
```

如果端口冲突，可以设置：

```bash
OTLP_GRPC_PORT=14317 \
OTLP_HTTP_PORT=14318 \
OUTPUT_DIR=/workspace/project/RL-learning/vllm-test/outputs/run1/otel_traces \
bash /workspace/project/RL-learning/vllm-test/scripts/run_otel_file_collector.sh
```

这时 benchmark endpoint 也要同步改成 `http://localhost:14317`。

collector 默认只打印启动和错误日志，不会逐条打印收到的 span。想在第一个终端看到收到了 trace，可以打开 debug exporter：

```bash
OTEL_COLLECTOR_DEBUG_EXPORTER=1 \
OUTPUT_DIR=/workspace/project/RL-learning/vllm-test/outputs/run1/otel_traces \
bash /workspace/project/RL-learning/vllm-test/scripts/run_otel_file_collector.sh
```

## 运行 Benchmark

开第二个终端，跑 merged benchmark：

```bash
python /workspace/project/RL-learning/vllm-test/scripts/benchmark_qwen35_mm_merged_request.py \
  --dataset-jsonl /workspace/project/RL-learning/vllm-test/data/alpamayo_qwen35_200/dataset.jsonl \
  --dataset-limit 1 \
  --batch-sizes 1 \
  --repeats 1 \
  --warmup 1 \
  --disable-enforce-eager \
  --output-dir /workspace/project/RL-learning/vllm-test/outputs/cuda_graph_smoke_merge \
  --otlp-traces-endpoint http://localhost:4317 \
  --otlp-traces-protocol grpc \
  --collect-detailed-traces all
```

也可以通过统一入口脚本传环境变量：

```bash
DATASET_JSONL=/workspace/project/RL-learning/vllm-test/data/alpamayo_qwen35_200/dataset.jsonl \
DATASET_LIMIT=1 \
BATCH_SIZES=1 \
REPEATS=1 \
WARMUP=1 \
OUTPUT_DIR=/workspace/project/RL-learning/vllm-test/outputs/cuda_graph_smoke_merge \
OTLP_TRACES_ENDPOINT=http://localhost:4317 \
OTLP_TRACES_PROTOCOL=grpc \
COLLECT_DETAILED_TRACES=all \
/workspace/project/RL-learning/vllm-test/scripts/run_qwen35_plugin.sh benchmark-merged-smoke
```

`run_config.json` 会记录本次是否开启 OTel，以及实际使用的 detailed traces 范围。

开启 OTel 时，benchmark 脚本会自动给 `vllm.LLM(...)` 传 `disable_log_stats=False`。这是必须的：vLLM 的 request-level tracing 会读取 `RequestState.stats`，而 offline `LLM` entrypoint 默认关闭 stats；如果不重新打开，可能在 `OutputProcessor.do_tracing()` 里触发 `AssertionError`。

## 离线查看

Collector 写出的文件是 OTel JSON。先确认有 span：

```bash
jq -c '.resourceSpans[]?.scopeSpans[]?.spans[]? | {name, traceId, spanId}' \
  /workspace/project/RL-learning/vllm-test/outputs/cuda_graph_smoke_merge/otel_traces/traces.json | head
```

按 span duration 排序：

```bash
jq -c '
  .resourceSpans[]?.scopeSpans[]?.spans[]?
  | {
      name,
      duration_ms: (((.endTimeUnixNano | tonumber) - (.startTimeUnixNano | tonumber)) / 1000000),
      traceId,
      spanId,
      parentSpanId
    }
' /workspace/project/RL-learning/vllm-test/outputs/cuda_graph_smoke_merge/otel_traces/traces.json \
  | jq -s 'sort_by(.duration_ms) | reverse | .[:30]'
```

查看 span attributes：

```bash
jq -c '
  .resourceSpans[]?.scopeSpans[]?.spans[]?
  | {
      name,
      duration_ms: (((.endTimeUnixNano | tonumber) - (.startTimeUnixNano | tonumber)) / 1000000),
      attributes: (.attributes // [])
    }
' /workspace/project/RL-learning/vllm-test/outputs/cuda_graph_smoke_merge/otel_traces/traces.json | head -n 20
```

一般重点看 request / scheduler / model worker 相关 span。OTel 更适合解释 `e2e_ms` 中模型 forward 之外的时间，例如排队、调度、输入处理、worker 执行和输出封装；torch profiler 更适合看 CUDA kernel 和 PyTorch op。

## 和 Torch Profile 的关系

- torch profile：保存到 `<output-dir>/torch_profile`，适合看执行层 kernel/op。
- OTel trace：保存到 `<output-dir>/otel_traces/traces.json`，适合看 vLLM 请求链路、调度和 worker 层 span。

两者可以同时开，但开销会更大。做正式性能数值时建议先不开 profiler；定位瓶颈时再开 OTel 或 torch profile。

如果只想解释本仓库 benchmark 的 `e2e_ms` 分解，优先使用 `--enable-vllm-python-profile` 生成的 `timeline.csv`，再用：

```bash
python /workspace/project/RL-learning/vllm-test/scripts/render_qwen35_request_timeline.py \
  <output-dir>/timeline.csv \
  --batch-size 2 \
  --group-idx 0 \
  --repeat-idx 0 \
  --output <output-dir>/request_timeline.html \
  --svg-output <output-dir>/request_timeline.svg
```

OTel 更适合看 vLLM 自带 tracing span；`timeline.csv` 更适合看当前脚本和插件记录到的 benchmark / vLLM / model phase 对账。

# Qwen3.5 vLLM Plugin

这个包用于在不修改 vLLM 源码的前提下替换 Qwen3.5 的模型实现，并在 benchmark 中记录 Qwen3.5 多模态链路的关键耗时。

插件通过 `pyproject.toml` 中的 entry point 暴露给 vLLM：

```toml
[project.entry-points."vllm.general_plugins"]
qwen35_custom_model = "qwen35_vllm_plugin:register"
```

详细发现和注册流程见 [PLUGIN_FLOW.md](PLUGIN_FLOW.md)。

## 安装

在当前目录安装 editable package：

```bash
cd /workspace/project/RL-learning/vllm-test/qwen35_vllm_plugin
pip install -e .
```

安装后，Python 环境会记录 `vllm.general_plugins` entry point。vLLM 启动时加载该 entry point，并调用 `qwen35_vllm_plugin.register()`。

## 加载方式

推荐显式指定插件名：

```bash
export VLLM_PLUGINS=qwen35_custom_model
```

这个插件默认是 opt-in 的：即使包已经安装，如果没有设置 `VLLM_PLUGINS=qwen35_custom_model` 或 `QWEN35_PLUGIN_ENABLE=1`，`register()` 也不会替换 Qwen3.5 模型注册。

可以用下面的命令验证是否替换成功：

```bash
VLLM_PLUGINS=qwen35_custom_model python -c "
from vllm.plugins import load_general_plugins
from vllm import ModelRegistry
load_general_plugins()
cls = ModelRegistry._try_load_model_cls('Qwen3_5ForConditionalGeneration')
print(cls.__module__ + ':' + cls.__name__)
"
```

期望输出包含：

```text
qwen35_vllm_plugin.model:CustomQwen35ForConditionalGeneration
```

## 启动入口

项目里已经提供统一脚本：

```bash
/workspace/project/RL-learning/vllm-test/scripts/run_qwen35_plugin.sh online
/workspace/project/RL-learning/vllm-test/scripts/run_qwen35_plugin.sh offline
/workspace/project/RL-learning/vllm-test/scripts/run_qwen35_plugin.sh benchmark-smoke
/workspace/project/RL-learning/vllm-test/scripts/run_qwen35_plugin.sh benchmark-merged-smoke
```

常用环境变量：

```bash
MODEL_PATH=/share/models/Qwen3.5-2B
MAX_MODEL_LEN=32768
GPU_MEMORY_UTILIZATION=0.9
DATASET_JSONL=/workspace/project/RL-learning/vllm-test/data/alpamayo_qwen35_200/dataset.jsonl
BATCH_SIZES=1,2,4,8
REPEATS=1
WARMUP=1
OUTPUT_DIR=/workspace/project/RL-learning/vllm-test/outputs/qwen35_run
```

普通离线推理：

```bash
VLLM_PLUGINS=qwen35_custom_model \
python /workspace/project/RL-learning/vllm-test/scripts/offline_infer.py \
  --model /share/models/Qwen3.5-2B \
  --prompt "你好，用一句话介绍你自己。"
```

在线服务：

```bash
VLLM_PLUGINS=qwen35_custom_model \
vllm serve /share/models/Qwen3.5-2B \
  --served-model-name qwen3.5-2b \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 32768
```

## 当前插件做了什么

`qwen35_vllm_plugin/__init__.py` 负责注册和可选的 vLLM Python wall-time profiling：

- `register()` 把 vLLM 的 `Qwen3_5ForConditionalGeneration` 映射替换为 `qwen35_vllm_plugin.model:CustomQwen35ForConditionalGeneration`。
- `QWEN35_PLUGIN_PROFILE_VLLM=1` 时，会给若干 vLLM Python 函数安装 wrapper，记录 request add、renderer、engine step、worker execute、MM encoder 等阶段 wall time。

`qwen35_vllm_plugin/model.py` 负责模型侧扩展：

- 继承 vLLM 原生 `Qwen3_5ForConditionalGeneration`，不复制整份 vLLM 模型实现。
- 覆盖顶层 `forward()`，记录 `qwen_forward` 和内部 `llm_forward`。
- 给 `self.visual` 注册 forward pre/post hook，记录 `vit_forward`。
- 可选给 `self.language_model.model` 注册 PyTorch forward hook，用于观察或修改语言模型输出。

## Profiling 输出

benchmark 脚本默认设置：

```bash
QWEN35_PLUGIN_PROFILE=1
QWEN35_PLUGIN_PROFILE_SYNC=1
QWEN35_PLUGIN_PROFILE_PATH=<output-dir>/profile_records.jsonl
```

模型侧会写入这些 phase：

- `vit_forward`：`self.visual(...)`，也就是视觉 encoder/ViT。
- `llm_forward`：`self.language_model.model(...)`，也就是底层 LLM forward。
- `qwen_forward`：`CustomQwen35ForConditionalGeneration.forward(...)` 顶层 forward。

每条记录包含：

- `wall_ms`
- `cuda_ms`
- `start_perf_counter` / `end_perf_counter`
- `start_unix` / `end_unix`
- `pid`
- `request_id`
- `output`

注意：模型实例通常在 vLLM worker 初始化时创建，`QWEN35_PLUGIN_REQUEST_ID` 不一定能反映每次 benchmark 请求。当前 benchmark 主要通过 `profile_records.jsonl` 的文件 offset 和时间范围把记录归到对应 `llm.generate(...)`，不要单独依赖模型侧记录里的 `request_id`。

## vLLM Python Profiling

merged benchmark 可以额外打开 vLLM Python wall-time 计时：

```bash
ENABLE_VLLM_PYTHON_PROFILE=1 \
/workspace/project/RL-learning/vllm-test/scripts/run_qwen35_plugin.sh benchmark-merged-smoke
```

等价于给 Python 脚本传：

```bash
--enable-vllm-python-profile
```

这会设置 `QWEN35_PLUGIN_PROFILE_VLLM=1`，插件会记录类似这些 phase：

- `vllm:LLM.generate`
- `vllm:LLM._preprocess_cmpl`
- `vllm:BaseRenderer.render_cmpl`
- `vllm:BaseRenderer._process_multimodal`
- `vllm:LLMEngine.step`
- `vllm:Worker.execute_model`
- `vllm:GPUModelRunner.execute_model`
- `vllm:GPUModelRunner._execute_mm_encoder`
- `vllm:GPUModelRunner._gather_mm_embeddings`

这些记录用于解释 `e2e_ms` 中模型 forward 之外的 Python/vLLM 调度和预处理开销。它们是 wall time，不是 CUDA kernel 时间。

## CUDA Graph 注意事项

插件的 CUDA event 计时会检测当前 stream 是否正在 CUDA graph capture：

- capture 中不会创建/record CUDA event，避免 `operation not permitted when stream is capturing`。
- 如果需要稳定的插件计时，benchmark 默认使用 eager 模式。
- 如果传 `--disable-enforce-eager` 开启 CUDA graph，模型侧 `cuda_ms` 可能为空或缺失，wall time 仍可作为参考。

## Feature Flags

- `VLLM_PLUGINS=qwen35_custom_model`：推荐的显式加载方式。
- `QWEN35_PLUGIN_ENABLE=1`：即使没有设置 `VLLM_PLUGINS`，也强制启用模型替换。
- `QWEN35_PLUGIN_DISABLE=1`：包已安装但跳过模型替换。
- `QWEN35_PLUGIN_LOG_FORWARD=1`：打印顶层 forward 输入/输出摘要。
- `QWEN35_PLUGIN_ENABLE_HOOK=1`：给 `self.language_model.model` 安装 PyTorch forward hook。
- `QWEN35_PLUGIN_PROFILE=1`：开启模型侧 profiling。
- `QWEN35_PLUGIN_PROFILE_PATH=/path/profile_records.jsonl`：指定 profiling 输出文件。
- `QWEN35_PLUGIN_PROFILE_SYNC=1`：同步 CUDA event，计时更稳定但会增加开销。
- `QWEN35_PLUGIN_PROFILE_VLLM=1`：开启 vLLM Python wall-time profiling。

## 修改自定义逻辑的位置

主要改 [qwen35_vllm_plugin/model.py](qwen35_vllm_plugin/model.py)：

- 在 `CustomQwen35ForConditionalGeneration.forward()` 前后添加顶层 forward 行为。
- 在 `_language_model_forward_hook()` 中观察或修改语言模型模块输出。
- 在 `_make_profile_pre_hook()` / `_make_profile_post_hook()` 中扩展模块级计时。

保持 `forward()` 的签名和返回类型与 vLLM 兼容，否则需要同步修改 vLLM 后续调用链。

benchmark 和分析流程见：

- [/workspace/project/RL-learning/vllm-test/docs/benchmark/qwen35_mm_bs_benchmark.md](/workspace/project/RL-learning/vllm-test/docs/benchmark/qwen35_mm_bs_benchmark.md)
- [/workspace/project/RL-learning/vllm-test/docs/benchmark/qwen35_enable_mm_embeds_perf_report.md](/workspace/project/RL-learning/vllm-test/docs/benchmark/qwen35_enable_mm_embeds_perf_report.md)

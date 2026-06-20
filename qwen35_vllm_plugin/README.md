# Qwen3.5 vLLM Plugin

This package demonstrates a non-invasive way to customize vLLM's Qwen3.5 model implementation. It does not modify the vLLM source tree. Instead, it registers a replacement model class through `vllm.general_plugins`.

For the detailed discovery and registration flow, see [PLUGIN_FLOW.md](PLUGIN_FLOW.md).

## Install

From this directory:

```bash
pip install -e .
```

## Run with vLLM

Use the entry-point name in `VLLM_PLUGINS` so vLLM only loads this plugin from the general plugin group:

```bash
VLLM_PLUGINS=qwen35_custom_model \
  vllm serve /share/models/Qwen3.5-2B \
  --served-model-name qwen3.5-2b \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 32768
```

## Offline vLLM

You can also test the plugin without starting an HTTP server:

```bash
python /workspace/project/RL-learning/vllm-test/scripts/offline_infer.py
```

Equivalent explicit form:

```bash
VLLM_PLUGINS=qwen35_custom_model \
  python /workspace/project/RL-learning/vllm-test/scripts/offline_infer.py \
  --model /share/models/Qwen3.5-2B \
  --prompt "你好，用一句话介绍你自己。"
```

This uses `vllm.LLM(...)` directly and does not expose `/v1/chat/completions`.

## Feature Flags

- `VLLM_PLUGINS=qwen35_custom_model`: recommended way to explicitly load this plugin.
- `QWEN35_PLUGIN_ENABLE=1`: force-enable the override even if `VLLM_PLUGINS` is unset.
- `QWEN35_PLUGIN_DISABLE=1`: keep the package installed but skip overriding vLLM's Qwen3.5 class.
- `QWEN35_PLUGIN_LOG_FORWARD=1`: log compact summaries of the top-level forward inputs and outputs.
- `QWEN35_PLUGIN_ENABLE_HOOK=1`: attach a PyTorch forward hook to `self.language_model.model`.

## Where to Add Custom Logic

Edit `qwen35_vllm_plugin/model.py`:

- Use `CustomQwen35ForConditionalGeneration.forward()` for pre/post behavior around the stock vLLM forward.
- Use `_language_model_forward_hook()` for observing or transforming the language model module output.

Keep the forward signature and return type compatible with vLLM unless you also update all downstream call sites.

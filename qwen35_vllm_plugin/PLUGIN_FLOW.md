# How vLLM Calls This Plugin

This document explains why this line in `pyproject.toml` is enough for vLLM to discover and call the plugin:

```toml
[project.entry-points."vllm.general_plugins"]
qwen35_custom_model = "qwen35_vllm_plugin:register"
```

## The Short Version

The plugin call path is:

```text
pip install -e qwen35_vllm_plugin
  -> Python records the package entry point metadata
  -> vLLM starts and calls load_general_plugins()
  -> vLLM asks Python for entry points in group "vllm.general_plugins"
  -> Python returns qwen35_custom_model = qwen35_vllm_plugin:register
  -> vLLM imports qwen35_vllm_plugin and calls register()
  -> register() overwrites ModelRegistry for Qwen3_5ForConditionalGeneration
  -> model config asks for Qwen3_5ForConditionalGeneration
  -> vLLM instantiates CustomQwen35ForConditionalGeneration
```

So vLLM does not scan this source directory directly. It discovers the plugin through Python package metadata created during installation.

## What the Entry Point Means

```toml
[project.entry-points."vllm.general_plugins"]
qwen35_custom_model = "qwen35_vllm_plugin:register"
```

This declares one Python packaging entry point. The pieces are:

- `vllm.general_plugins`: the entry point group name. vLLM explicitly loads this group during startup.
- `qwen35_custom_model`: the plugin name inside that group. This is the value used by `VLLM_PLUGINS=qwen35_custom_model`.
- `qwen35_vllm_plugin:register`: the import target. It means: import module `qwen35_vllm_plugin`, then get the callable named `register` from that module.

After installation, this metadata is available through `importlib.metadata.entry_points()`. Conceptually, vLLM can do this:

```python
from importlib.metadata import entry_points

for plugin in entry_points(group="vllm.general_plugins"):
    if plugin.name == "qwen35_custom_model":
        register_func = plugin.load()
        register_func()
```

The real vLLM implementation is in `vllm/plugins/__init__.py`. It also handles logging, filtering through `VLLM_PLUGINS`, and loading each plugin only once per process.

## Why `pip install -e .` Is Required

The `pyproject.toml` file by itself is only source code/configuration. Python does not automatically read every `pyproject.toml` on disk.

Running:

```bash
pip install -e .
```

installs the package in editable mode and writes package metadata into the active Python environment. That metadata includes the `vllm.general_plugins` entry point. Editable mode means code changes under this directory are used immediately, but entry point changes in `pyproject.toml` may require reinstalling the package.

You can verify discovery with:

```bash
python -c "from importlib.metadata import entry_points; print([(ep.name, ep.value) for ep in entry_points(group='vllm.general_plugins') if ep.name == 'qwen35_custom_model'])"
```

Expected output includes:

```text
('qwen35_custom_model', 'qwen35_vllm_plugin:register')
```

## What `register()` Does

The entry point calls `qwen35_vllm_plugin.register()`, implemented in `qwen35_vllm_plugin/__init__.py`. That function runs:

```python
ModelRegistry.register_model(
    "Qwen3_5ForConditionalGeneration",
    "qwen35_vllm_plugin.model:CustomQwen35ForConditionalGeneration",
)
```

This changes vLLM's architecture mapping:

```text
Qwen3_5ForConditionalGeneration
  -> qwen35_vllm_plugin.model.CustomQwen35ForConditionalGeneration
```

The local model at `/share/models/Qwen3.5-2B/` has this in `config.json`:

```json
"architectures": [
  "Qwen3_5ForConditionalGeneration"
]
```

When vLLM loads that model, it asks `ModelRegistry` which class implements `Qwen3_5ForConditionalGeneration`. Because the plugin replaced the mapping, vLLM instantiates the custom subclass from `model.py`.

## Why `VLLM_PLUGINS=qwen35_custom_model` Is Used

vLLM's default behavior is to load all installed plugins in a group when `VLLM_PLUGINS` is unset. This package deliberately makes the override opt-in so installing it does not silently change every Qwen3.5 run in the environment.

Use:

```bash
VLLM_PLUGINS=qwen35_custom_model vllm serve /share/models/Qwen3.5-2B ...
```

Then `register()` sees that `qwen35_custom_model` was explicitly selected and performs the override.

You can also force-enable it with:

```bash
QWEN35_PLUGIN_ENABLE=1 vllm serve /share/models/Qwen3.5-2B ...
```

And you can disable it even if installed with:

```bash
QWEN35_PLUGIN_DISABLE=1 vllm serve /share/models/Qwen3.5-2B ...
```

## End-to-End Verification

Without explicitly enabling the plugin, vLLM should still resolve the stock model class:

```bash
python -c "from vllm.plugins import load_general_plugins; from vllm import ModelRegistry; load_general_plugins(); cls = ModelRegistry._try_load_model_cls('Qwen3_5ForConditionalGeneration'); print(cls.__module__ + ':' + cls.__name__)"
```

Expected:

```text
vllm.model_executor.models.qwen3_5:Qwen3_5ForConditionalGeneration
```

With the plugin selected:

```bash
VLLM_PLUGINS=qwen35_custom_model python -c "from vllm.plugins import load_general_plugins; from vllm import ModelRegistry; load_general_plugins(); cls = ModelRegistry._try_load_model_cls('Qwen3_5ForConditionalGeneration'); print(cls.__module__ + ':' + cls.__name__)"
```

Expected:

```text
qwen35_vllm_plugin.model:CustomQwen35ForConditionalGeneration
```

## Mental Model

Think of the entry point as a named phone book entry installed into the Python environment:

```text
Group: vllm.general_plugins
Name:  qwen35_custom_model
Call:  qwen35_vllm_plugin.register
```

vLLM knows to open the `vllm.general_plugins` phone book. The plugin entry tells vLLM which function to call. That function then edits vLLM's model registry so Qwen3.5 resolves to our custom class.

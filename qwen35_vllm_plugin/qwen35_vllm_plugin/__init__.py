# SPDX-License-Identifier: Apache-2.0
"""vLLM general plugin entry point for custom Qwen3.5 behavior.

Installing this package exposes the ``qwen35_custom_model`` entry point under
``vllm.general_plugins``. vLLM loads that entry point during startup and calls
``register()``, which replaces the default Qwen3.5 architecture mapping with
our subclass.
"""

from __future__ import annotations

import os

from vllm.model_executor.models import ModelRegistry

_QWEN35_ARCH = "Qwen3_5ForConditionalGeneration"
_CUSTOM_MODEL = (
    "qwen35_vllm_plugin.model:CustomQwen35ForConditionalGeneration"
)


def register() -> None:
    """Register the custom Qwen3.5 model class with vLLM.

    vLLM loads general plugins in multiple processes, so this function must be
    safe to call more than once. ``register_model`` overwrites the architecture
    mapping when it already exists, which is the behavior we want here. Set
    ``QWEN35_PLUGIN_DISABLE=1`` to leave the stock vLLM class untouched while
    keeping the package installed.
    """

    if os.getenv("QWEN35_PLUGIN_DISABLE", "0") == "1":
        return

    explicit_plugins = os.getenv("VLLM_PLUGINS")
    explicitly_selected = (
        explicit_plugins is not None
        and "qwen35_custom_model" in {
            item.strip() for item in explicit_plugins.split(",") if item.strip()
        }
    )
    explicitly_enabled = os.getenv("QWEN35_PLUGIN_ENABLE", "0") == "1"

    # vLLM loads every installed general plugin when VLLM_PLUGINS is unset.
    # Keep this package opt-in so installing it does not silently alter every
    # Qwen3.5 run in the environment.
    if not explicitly_selected and not explicitly_enabled:
        return

    ModelRegistry.register_model(_QWEN35_ARCH, _CUSTOM_MODEL)

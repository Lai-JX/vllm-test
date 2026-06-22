# SPDX-License-Identifier: Apache-2.0
"""vLLM general plugin entry point for custom Qwen3.5 behavior.

Installing this package exposes the ``qwen35_custom_model`` entry point under
``vllm.general_plugins``. vLLM loads that entry point during startup and calls
``register()``, which replaces the default Qwen3.5 architecture mapping with
our subclass.
"""

from __future__ import annotations

import functools
import json
import os
import time
from pathlib import Path
from typing import Any

from vllm.logger import init_logger
from vllm.model_executor.models import ModelRegistry

_QWEN35_ARCH = "Qwen3_5ForConditionalGeneration"
_CUSTOM_MODEL = (
    "qwen35_vllm_plugin.model:CustomQwen35ForConditionalGeneration"
)
_PATCHED_ATTR = "_qwen35_vllm_profile_wrapped"
logger = init_logger(__name__)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def _write_vllm_profile_record(
    phase: str,
    wall_ms: float,
    output: Any = None,
    *,
    start_perf_counter: float | None = None,
    end_perf_counter: float | None = None,
    start_unix: float | None = None,
    end_unix: float | None = None,
) -> None:
    profile_path = os.getenv("QWEN35_PLUGIN_PROFILE_PATH")
    if not profile_path:
        return
    record = {
        "phase": phase,
        "request_id": os.getenv("QWEN35_PLUGIN_REQUEST_ID", ""),
        "pid": os.getpid(),
        "wall_ms": wall_ms,
        "cuda_ms": None,
        "output": _summarize_output(output),
        "start_perf_counter": start_perf_counter,
        "end_perf_counter": end_perf_counter,
        "start_unix": start_unix,
        "end_unix": end_unix,
        "time_unix": time.time(),
    }
    Path(profile_path).parent.mkdir(parents=True, exist_ok=True)
    with open(profile_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True) + "\n")


def _summarize_output(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, (list, tuple)):
        return f"{type(value).__name__}(len={len(value)})"
    if isinstance(value, dict):
        return f"dict(len={len(value)})"
    shape = getattr(value, "shape", None)
    if shape is not None:
        return f"{type(value).__name__}(shape={tuple(shape)})"
    return type(value).__name__


def _wrap_method(cls: type[Any], method_name: str, phase: str) -> None:
    original = getattr(cls, method_name, None)
    if original is None or getattr(original, _PATCHED_ATTR, False):
        return

    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        start = time.perf_counter()
        start_unix = time.time()
        try:
            output = original(self, *args, **kwargs)
        except Exception:
            end = time.perf_counter()
            end_unix = time.time()
            _write_vllm_profile_record(
                f"{phase}:exception",
                (end - start) * 1000.0,
                start_perf_counter=start,
                end_perf_counter=end,
                start_unix=start_unix,
                end_unix=end_unix,
            )
            raise
        end = time.perf_counter()
        end_unix = time.time()
        _write_vllm_profile_record(
            phase,
            (end - start) * 1000.0,
            output,
            start_perf_counter=start,
            end_perf_counter=end,
            start_unix=start_unix,
            end_unix=end_unix,
        )
        return output

    setattr(wrapped, _PATCHED_ATTR, True)
    setattr(cls, method_name, wrapped)


def _install_vllm_profile_patches() -> None:
    if not _env_flag("QWEN35_PLUGIN_PROFILE_VLLM"):
        return

    patched: list[str] = []
    patch_specs: tuple[tuple[str, str, str], ...] = (
        ("vllm.entrypoints.llm", "LLM", "generate"),
        ("vllm.entrypoints.llm", "LLM", "_run_completion"),
        ("vllm.entrypoints.llm", "LLM", "_add_completion_requests"),
        ("vllm.entrypoints.llm", "LLM", "_render_and_add_requests"),
        ("vllm.entrypoints.llm", "LLM", "_preprocess_cmpl_one"),
        ("vllm.entrypoints.llm", "LLM", "_preprocess_cmpl"),
        ("vllm.renderers.base", "BaseRenderer", "render_cmpl"),
        ("vllm.renderers.base", "BaseRenderer", "render_prompts"),
        ("vllm.renderers.base", "BaseRenderer", "tokenize_prompts"),
        ("vllm.renderers.base", "BaseRenderer", "process_for_engine"),
        ("vllm.renderers.base", "BaseRenderer", "_process_tokens"),
        ("vllm.renderers.base", "BaseRenderer", "_process_multimodal"),
        ("vllm.entrypoints.llm", "LLM", "_add_request"),
        ("vllm.entrypoints.llm", "LLM", "_run_engine"),
        ("vllm.v1.engine.llm_engine", "LLMEngine", "add_request"),
        ("vllm.v1.engine.llm_engine", "LLMEngine", "step"),
        ("vllm.v1.engine.output_processor", "OutputProcessor", "add_request"),
        ("vllm.v1.engine.output_processor", "OutputProcessor", "process_outputs"),
        ("vllm.v1.worker.gpu_model_runner", "GPUModelRunner", "_execute_mm_encoder"),
        ("vllm.v1.worker.gpu_model_runner", "GPUModelRunner", "_gather_mm_embeddings"),
        ("vllm.v1.worker.gpu_model_runner", "GPUModelRunner", "execute_model"),
        ("vllm.v1.worker.gpu_worker", "Worker", "execute_model"),
    )

    for module_name, class_name, method_name in patch_specs:
        try:
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)
            phase = f"vllm:{class_name}.{method_name}"
            _wrap_method(cls, method_name, phase)
            patched.append(phase)
        except Exception:
            logger.exception(
                "Failed to install Qwen3.5 vLLM profile patch for %s.%s",
                module_name,
                method_name,
            )

    if patched:
        logger.info("Qwen3.5 vLLM Python profiling enabled: %s", ", ".join(patched))


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

    _install_vllm_profile_patches()
    ModelRegistry.register_model(_QWEN35_ARCH, _CUSTOM_MODEL)

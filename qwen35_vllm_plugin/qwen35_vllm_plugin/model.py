# SPDX-License-Identifier: Apache-2.0
"""Custom Qwen3.5 model implementation loaded through vLLM's plugin system.

This module intentionally subclasses the in-tree vLLM implementation instead
of copying it. Put project-specific behavior in the marked sections below so
upstream vLLM changes remain easy to absorb.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration
from vllm.sequence import IntermediateTensors

logger = init_logger(__name__)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def _cuda_event_pair() -> tuple[torch.cuda.Event | None, torch.cuda.Event | None]:
    if torch.cuda.is_available() and not _is_cuda_graph_capturing():
        return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    return None, None


def _is_cuda_graph_capturing() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:
        return False


def _record_cuda_event(event: torch.cuda.Event | None) -> bool:
    if event is None or _is_cuda_graph_capturing():
        return False
    event.record()
    return True


def _maybe_cuda_synchronize(enabled: bool) -> None:
    if enabled and torch.cuda.is_available() and not _is_cuda_graph_capturing():
        torch.cuda.synchronize()


class CustomQwen35ForConditionalGeneration(Qwen3_5ForConditionalGeneration):
    """Drop-in replacement for vLLM's Qwen3_5ForConditionalGeneration.

    The class demonstrates three non-invasive extension points:
    1. Override the top-level ``forward`` while preserving vLLM's signature.
    2. Optionally attach a PyTorch forward hook to the language model module.
    3. Optionally profile ViT/LLM/Qwen forward timings into JSONL.

    Environment flags:
    - ``QWEN35_PLUGIN_LOG_FORWARD=1`` logs top-level input/output summaries.
    - ``QWEN35_PLUGIN_ENABLE_HOOK=1`` enables the PyTorch module hook example.
    - ``QWEN35_PLUGIN_PROFILE=1`` enables timing records.
    - ``QWEN35_PLUGIN_PROFILE_PATH=/path/file.jsonl`` selects the output file.
    - ``QWEN35_PLUGIN_PROFILE_SYNC=1`` synchronizes CUDA events for stable timing.
    """

    def __init__(self, *, vllm_config, prefix: str = "model"):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        self._log_forward = _env_flag("QWEN35_PLUGIN_LOG_FORWARD")
        self._enable_hook = _env_flag("QWEN35_PLUGIN_ENABLE_HOOK")
        self._profile_enabled = _env_flag("QWEN35_PLUGIN_PROFILE")
        self._profile_path = os.getenv("QWEN35_PLUGIN_PROFILE_PATH")
        self._profile_sync = _env_flag("QWEN35_PLUGIN_PROFILE_SYNC", True)
        self._profile_request_id = os.getenv("QWEN35_PLUGIN_REQUEST_ID", "")
        self._profile_stacks: dict[str, list[tuple[float, torch.cuda.Event | None]]] = {
            "vit_forward": [],
        }

        if self._enable_hook:
            # Keep a reference so the hook can be removed later if needed.
            self._qwen35_plugin_hook_handle = (
                self.language_model.model.register_forward_hook(
                    self._language_model_forward_hook
                )
            )
            logger.info("Qwen3.5 custom plugin forward hook enabled.")

        if self._profile_enabled:
            if not self._profile_path:
                raise ValueError(
                    "QWEN35_PLUGIN_PROFILE=1 requires "
                    "QWEN35_PLUGIN_PROFILE_PATH to be set."
                )
            Path(self._profile_path).parent.mkdir(parents=True, exist_ok=True)
            self._qwen35_vit_pre_handle = self.visual.register_forward_pre_hook(
                self._make_profile_pre_hook("vit_forward")
            )
            self._qwen35_vit_post_handle = self.visual.register_forward_hook(
                self._make_profile_post_hook("vit_forward")
            )
            logger.info(
                "Qwen3.5 profiling enabled. Writing records to %s", self._profile_path
            )

    def _language_model_forward_hook(self, module, inputs, output):
        """Observe or transform the language model output.

        Returning ``output`` preserves default behavior. Replace this method's
        body if the hook should collect tensors, export stats, or alter the
        hidden states. Be careful: this hook runs inside vLLM's hot path.
        """

        if self._log_forward:
            logger.info(
                "Qwen3.5 hook: module=%s output=%s",
                module.__class__.__name__,
                self._summarize_value(output),
            )
        return output

    def _make_profile_pre_hook(self, phase: str):
        def hook(module, inputs):
            start_event, _ = _cuda_event_pair()
            if not _record_cuda_event(start_event):
                start_event = None
            self._profile_stacks[phase].append((time.perf_counter(), start_event))

        return hook

    def _make_profile_post_hook(self, phase: str):
        def hook(module, inputs, output):
            if not self._profile_stacks[phase]:
                return output
            start_wall, start_event = self._profile_stacks[phase].pop()
            _, end_event = _cuda_event_pair()
            if not _record_cuda_event(end_event):
                end_event = None
            _maybe_cuda_synchronize(self._profile_sync and end_event is not None)
            wall_ms = (time.perf_counter() - start_wall) * 1000.0
            cuda_ms = None
            if start_event is not None and end_event is not None:
                cuda_ms = float(start_event.elapsed_time(end_event))
            self._write_profile_record(phase, wall_ms, cuda_ms, output)
            return output

        return hook

    def _write_profile_record(
        self,
        phase: str,
        wall_ms: float,
        cuda_ms: float | None,
        output: Any = None,
    ) -> None:
        if not self._profile_enabled or not self._profile_path:
            return
        record = {
            "phase": phase,
            "request_id": self._profile_request_id,
            "pid": os.getpid(),
            "wall_ms": wall_ms,
            "cuda_ms": cuda_ms,
            "output": self._summarize_value(output),
            "time_unix": time.time(),
        }
        with open(self._profile_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        """Run Qwen3.5 forward with a project-specific extension shell."""

        if self._log_forward:
            logger.info(
                "Qwen3.5 custom forward: input_ids=%s positions=%s "
                "intermediate_tensors=%s inputs_embeds=%s extra_keys=%s",
                self._summarize_value(input_ids),
                self._summarize_value(positions),
                intermediate_tensors is not None,
                self._summarize_value(inputs_embeds),
                sorted(kwargs.keys()),
            )

        start_wall = time.perf_counter()
        start_event, end_event = _cuda_event_pair()
        if not _record_cuda_event(start_event):
            start_event = None

        if intermediate_tensors is not None:
            inputs_embeds = None

        llm_start_wall = time.perf_counter()
        llm_start_event, llm_end_event = _cuda_event_pair()
        if not _record_cuda_event(llm_start_event):
            llm_start_event = None

        output = self.language_model.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

        if not _record_cuda_event(llm_end_event):
            llm_end_event = None
        _maybe_cuda_synchronize(self._profile_sync and llm_end_event is not None)
        if self._profile_enabled:
            llm_wall_ms = (time.perf_counter() - llm_start_wall) * 1000.0
            llm_cuda_ms = None
            if llm_start_event is not None and llm_end_event is not None:
                llm_cuda_ms = float(llm_start_event.elapsed_time(llm_end_event))
            self._write_profile_record(
                "llm_forward", llm_wall_ms, llm_cuda_ms, output
            )

        if not _record_cuda_event(end_event):
            end_event = None
        _maybe_cuda_synchronize(self._profile_sync and end_event is not None)
        if self._profile_enabled:
            wall_ms = (time.perf_counter() - start_wall) * 1000.0
            cuda_ms = None
            if start_event is not None and end_event is not None:
                cuda_ms = float(start_event.elapsed_time(end_event))
            self._write_profile_record("qwen_forward", wall_ms, cuda_ms, output)

        if self._log_forward:
            logger.info("Qwen3.5 custom forward output=%s", self._summarize_value(output))

        return output

    @staticmethod
    def _summarize_value(value: Any) -> str:
        if value is None:
            return "None"
        if isinstance(value, torch.Tensor):
            return (
                f"Tensor(shape={tuple(value.shape)}, dtype={value.dtype}, "
                f"device={value.device})"
            )
        if isinstance(value, (tuple, list)):
            return f"{type(value).__name__}(len={len(value)})"
        return type(value).__name__

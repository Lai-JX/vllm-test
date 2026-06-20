#!/usr/bin/env python3
"""Offline vLLM inference test for the Qwen3.5 plugin.

Run from any directory after installing this package with `pip install -e .`:

    python offline_infer.py

The script uses `vllm.LLM` directly, so it does not start an HTTP server.
"""

from __future__ import annotations

import argparse
import os
import random

# This must be set before importing vLLM modules that may load plugins.
os.environ.setdefault("VLLM_PLUGINS", "qwen35_custom_model")

from vllm import LLM, SamplingParams  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="/share/models/Qwen3.5-2B",
        help="Local model path or Hugging Face model id.",
    )
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--prompt",
        default=None,
        help="User prompt for chat inference.",
    )
    input_group.add_argument(
        "--input-len",
        type=int,
        default=None,
        help=(
            "Construct a random token-id prompt with exactly this many input "
            "tokens. Mutually exclusive with --prompt."
        ),
    )
    parser.add_argument(
        "--input-seed",
        type=int,
        default=0,
        help="Random seed used with --input-len.",
    )
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument(
        "--log-forward",
        action="store_true",
        help="Enable compact forward logging in the plugin.",
    )
    parser.add_argument(
        "--enable-hook",
        action="store_true",
        help="Enable the plugin's PyTorch forward hook example.",
    )
    args = parser.parse_args()
    if args.input_len is not None and args.input_len <= 0:
        parser.error("--input-len must be a positive integer.")
    return args


def make_random_prompt_token_ids(tokenizer, input_len: int,
                                 seed: int) -> list[int]:
    vocab = tokenizer.get_vocab()
    special_token_ids = set(tokenizer.all_special_ids)
    candidate_token_ids = [
        token_id for token_id in vocab.values()
        if token_id not in special_token_ids
    ]

    if not candidate_token_ids:
        raise RuntimeError("Tokenizer has no non-special tokens to sample from.")

    rng = random.Random(seed)
    return [rng.choice(candidate_token_ids) for _ in range(input_len)]


def main() -> None:
    args = parse_args()

    if args.log_forward:
        os.environ["QWEN35_PLUGIN_LOG_FORWARD"] = "1"
    if args.enable_hook:
        os.environ["QWEN35_PLUGIN_ENABLE_HOOK"] = "1"

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    if args.input_len is None:
        prompt = args.prompt or "你好，用一句话介绍你自己。"
        messages = [[{"role": "user", "content": prompt}]]
        outputs = llm.chat(
            messages,
            sampling_params=sampling_params,
            use_tqdm=False,
        )
    else:
        prompt_token_ids = make_random_prompt_token_ids(
            llm.get_tokenizer(),
            args.input_len,
            args.input_seed,
        )
        assert len(prompt_token_ids) == args.input_len
        outputs = llm.generate(
            prompts=[{"prompt_token_ids": prompt_token_ids}],
            sampling_params=sampling_params,
            use_tqdm=False,
        )

    for output in outputs:
        print(output.outputs[0].text)


if __name__ == "__main__":
    main()

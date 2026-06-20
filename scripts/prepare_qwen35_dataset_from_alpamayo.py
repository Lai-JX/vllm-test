#!/usr/bin/env python3
"""Export Alpamayo samples into the JSONL format used by Qwen3.5 benchmarks.

This follows the dataset loading path used by
`vllm-omni/profiler/batch_sweep_cont_inproc_tokenized_warmup.py`: it reads clip
ids from the local PAI interface, builds Alpamayo model inputs, extracts the
original prompt text and image frames from `multi_modal_data["image"]`, saves
frames as PNG files, and writes a JSONL that can be consumed by the Qwen3.5
benchmark scripts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoConfig

RL_ROOT = Path("/workspace/project/RL-learning")
OMNI = RL_ROOT / "vllm-omni"
VERL_ROOT = RL_ROOT / "verl"
MY_EXAMPLE2_ROOT = VERL_ROOT / "my_example2"

for path in [
    OMNI,
    RL_ROOT / "alpamayo1.5" / "src",
    VERL_ROOT / "my_example" / "alpamayo" / "src",
    RL_ROOT / "verl-liming" / "my_example" / "alpamayo" / "src",
    VERL_ROOT,
    MY_EXAMPLE2_ROOT,
]:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from my_example2.src.inputs.alpamayo_batch import (  # noqa: E402
    build_alpamayo_sample,
    build_rollout_preprocess_fn,
    build_rollout_processor,
    build_traj_fuser,
    sample_to_vllm_prompt,
    validate_local_pai_dir,
)
from my_example2.src.models.register_vla_models import register_vla_models  # noqa: E402


DEFAULT_ALPAMAYO_MODEL = "/share/models/Alpamayo-1.5-10B"
DEFAULT_PAI_LOCAL_DIR = "/share/datasets/Alpamayo_pai_av_big/"
DEFAULT_OUTPUT_DIR = "/workspace/project/RL-learning/vllm-test/data/alpamayo_qwen35"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpamayo-model", default=DEFAULT_ALPAMAYO_MODEL)
    parser.add_argument("--pai-local-dir", default=DEFAULT_PAI_LOCAL_DIR)
    parser.add_argument(
        "--chunk-id",
        type=int,
        default=3116,
        help="Select clip ids from avdi.clip_index where chunk equals this value.",
    )
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--t0-us", type=int, default=5_100_000)
    parser.add_argument(
        "--frames-per-clip",
        type=int,
        default=0,
        help="0 means export all images from the original Alpamayo sample.",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-jsonl", default=None)
    parser.add_argument(
        "--text-mode",
        choices=("original_prompt", "tokenized_text"),
        default="original_prompt",
        help=(
            "original_prompt decodes the final Alpamayo vLLM prompt ids. "
            "tokenized_text uses sample['tokenized_data']['text'] before "
            "trajectory-token fusion."
        ),
    )
    return parser.parse_args()


def build_local_pai_interface(local_dir: str) -> Any:
    from alpamayo_r1.data.pai_utils import PhysicalAIAVDatasetLocalInterface

    return PhysicalAIAVDatasetLocalInterface(local_dir=local_dir)


def build_components(args: argparse.Namespace) -> dict[str, Any]:
    register_vla_models()
    import alpamayo1_5.models.alpamayo1_5  # noqa: F401

    validate_local_pai_dir(args.pai_local_dir)
    model_config = AutoConfig.from_pretrained(args.alpamayo_model,
                                             trust_remote_code=True)
    processor = build_rollout_processor(model_config)
    preprocess_fn = build_rollout_preprocess_fn(model_config)
    traj_fuser = build_traj_fuser(model_config)
    avdi = build_local_pai_interface(args.pai_local_dir)
    return {
        "model_config": model_config,
        "processor": processor,
        "preprocess_fn": preprocess_fn,
        "traj_fuser": traj_fuser,
        "avdi": avdi,
    }


def select_clip_ids(avdi: Any, chunk_id: int, start_index: int,
                    num_samples: int) -> list[str]:
    clip_index = avdi.clip_index
    selected = list(clip_index[clip_index.chunk == chunk_id].index)
    if start_index:
        selected = selected[start_index:]
    if num_samples > 0:
        selected = selected[:num_samples]
    if not selected:
        raise ValueError(
            f"No clip ids found for chunk={chunk_id}, start_index={start_index}")
    return [str(clip_id) for clip_id in selected]


def tensor_to_pil(value: torch.Tensor) -> Image.Image:
    tensor = value.detach().cpu()
    if tensor.ndim == 4 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.ndim != 3:
        raise ValueError(f"Expected image tensor with 3 dims, got {tuple(tensor.shape)}")

    array = tensor.float().numpy()
    if array.shape[0] in {1, 3, 4}:
        array = np.moveaxis(array, 0, -1)
    if array.shape[-1] == 1:
        array = array[..., 0]

    if array.dtype != np.uint8:
        if np.nanmax(array) <= 1.5:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)

    if array.ndim == 2:
        return Image.fromarray(array, mode="L").convert("RGB")
    return Image.fromarray(array[..., :3]).convert("RGB")


def to_pil_image(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, torch.Tensor):
        return tensor_to_pil(value)
    if isinstance(value, np.ndarray):
        array = value
        if array.ndim == 3 and array.shape[0] in {1, 3, 4}:
            array = np.moveaxis(array, 0, -1)
        if array.dtype != np.uint8:
            if np.nanmax(array) <= 1.5:
                array = array * 255.0
            array = np.clip(array, 0, 255).astype(np.uint8)
        return Image.fromarray(array[..., :3]).convert("RGB")
    raise TypeError(f"Unsupported image value type: {type(value)!r}")


def build_sample_and_model_input(clip_id: str, args: argparse.Namespace,
                                 components: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    sample = build_alpamayo_sample(
        clip_id,
        args.t0_us,
        components["preprocess_fn"],
        components["avdi"],
    )
    model_input = sample_to_vllm_prompt(
        sample,
        components["processor"].tokenizer,
        components["model_config"],
        components["traj_fuser"],
    )
    return sample, model_input


def extract_original_text(sample: dict[str, Any], model_input: dict[str, Any],
                          components: dict[str, Any], mode: str) -> str:
    if mode == "tokenized_text":
        text = (sample.get("tokenized_data") or {}).get("text")
        if text is None:
            raise KeyError("sample['tokenized_data']['text'] is missing")
        return str(text)

    prompt_ids = model_input.get("prompt_token_ids")
    if not prompt_ids:
        raise KeyError("model_input['prompt_token_ids'] is missing")
    return components["processor"].tokenizer.decode(
        list(prompt_ids),
        skip_special_tokens=False,
    )


def extract_images(model_input: dict[str, Any]) -> list[Image.Image]:
    mm_data = model_input.get("multi_modal_data") or {}
    images = mm_data.get("image") or []
    if isinstance(images, (Image.Image, torch.Tensor, np.ndarray)):
        images = [images]
    return [to_pil_image(image) for image in images]


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl = Path(args.output_jsonl or output_dir / "dataset.jsonl")

    components = build_components(args)
    clip_ids = select_clip_ids(
        components["avdi"],
        args.chunk_id,
        args.start_index,
        args.num_samples,
    )

    rows = []
    for clip_pos, clip_id in enumerate(clip_ids):
        print(f"[{clip_pos + 1}/{len(clip_ids)}] clip={clip_id}")
        sample, model_input = build_sample_and_model_input(clip_id, args, components)
        original_text = extract_original_text(
            sample,
            model_input,
            components,
            args.text_mode,
        )
        images = extract_images(model_input)
        if not images:
            print(f"  skip: no images extracted for clip={clip_id}")
            continue

        selected_images = images if args.frames_per_clip <= 0 else images[:args.frames_per_clip]
        image_paths = []
        for frame_index, image in enumerate(selected_images):
            image_name = f"{clip_pos:05d}_{clip_id[:12]}_image{frame_index:02d}.png"
            image_path = image_dir / image_name
            image.save(image_path)
            image_paths.append(str(image_path))

        row = {
            "id": clip_id,
            "clip_id": clip_id,
            "t0_us": args.t0_us,
            "image_count": len(image_paths),
            "images": image_paths,
            "prompt": original_text,
            "text_source": args.text_mode,
        }
        rows.append(row)

    if not rows:
        raise RuntimeError("No rows were exported.")

    with output_jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} rows -> {output_jsonl}")
    print(f"Images -> {image_dir}")


if __name__ == "__main__":
    main()

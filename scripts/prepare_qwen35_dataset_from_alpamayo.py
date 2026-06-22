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
import os
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
DEFAULT_QWEN35_MODEL = "/share/models/Qwen3.5-2B"
DEFAULT_PAI_LOCAL_DIR = "/share/datasets/Alpamayo_pai_av_big/"
DEFAULT_OUTPUT_DIR = "/workspace/project/RL-learning/vllm-test/data/alpamayo_qwen35"
ALPAMAYO_MIN_PIXELS = 163840
ALPAMAYO_MAX_PIXELS = 196608


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpamayo-model", default=DEFAULT_ALPAMAYO_MODEL)
    parser.add_argument(
        "--model",
        default=DEFAULT_QWEN35_MODEL,
        help="Qwen3.5 model used to precompute image_grid_thw and image_embeds.",
    )
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
        "--embeds-dir",
        default=None,
        help="Directory for .pt embedding files. Defaults to OUTPUT_DIR/image_embeds.",
    )
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument(
        "--mm-min-pixels",
        type=int,
        default=ALPAMAYO_MIN_PIXELS,
        help="Minimum image pixels for the Qwen3.5 multimodal processor.",
    )
    parser.add_argument(
        "--mm-max-pixels",
        type=int,
        default=ALPAMAYO_MAX_PIXELS,
        help="Maximum image pixels for the Qwen3.5 multimodal processor.",
    )
    parser.add_argument(
        "--disable-mm-do-rescale",
        action="store_true",
        help="Do not pass do_rescale=True to the Qwen3.5 multimodal processor.",
    )
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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp_path.replace(path)


def _safe_stem(value: Any, fallback: str) -> str:
    raw = str(value or fallback)
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in raw)
    return safe[:80] or fallback


def _resolve_existing_paths(dataset_path: Path, raw_paths: Any) -> list[Path]:
    if raw_paths is None:
        return []
    paths = raw_paths if isinstance(raw_paths, list) else [raw_paths]
    resolved = []
    for raw_path in paths:
        path = Path(str(raw_path))
        if not path.is_absolute():
            path = dataset_path.parent / path
        resolved.append(path)
    return resolved


def resolve_row_image_paths(dataset_path: Path, row: dict[str, Any]) -> list[Path]:
    raw = (
        row.get("images")
        or row.get("image_paths")
        or row.get("image_files")
        or row.get("image")
        or row.get("image_path")
        or row.get("image_file")
    )
    paths = _resolve_existing_paths(dataset_path, raw)
    if not paths:
        raise ValueError(
            f"row {row.get('id') or row.get('clip_id') or '<unknown>'} has no images"
        )
    return paths


def _resolve_existing_path(dataset_path: Path, raw_path: Any) -> Path:
    path = Path(str(raw_path))
    if not path.is_absolute():
        path = dataset_path.parent / path
    return path


def _row_has_existing_embedding_files(
    dataset_path: Path,
    row: dict[str, Any],
) -> bool:
    image_embeds = row.get("image_embeds")
    image_grid_thw = row.get("image_grid_thw")
    if not image_embeds or not image_grid_thw:
        return False
    return (
        _resolve_existing_path(dataset_path, image_embeds).exists()
        and _resolve_existing_path(dataset_path, image_grid_thw).exists()
    )


def build_qwen35_embedding_components(args: argparse.Namespace) -> tuple[Any, Any]:
    os.environ.setdefault("VLLM_PLUGINS", "qwen35_custom_model")
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    from transformers import AutoProcessor
    from vllm import LLM

    processor_kwargs = {
        "trust_remote_code": True,
        "min_pixels": args.mm_min_pixels,
        "max_pixels": args.mm_max_pixels,
    }
    try:
        processor = AutoProcessor.from_pretrained(args.model, **processor_kwargs)
    except TypeError:
        processor = AutoProcessor.from_pretrained(
            args.model,
            trust_remote_code=True,
        )

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
    )
    return processor, llm


def compute_image_embedding_files(
    *,
    image_paths: list[Path],
    image_embeds_path: Path,
    image_grid_thw_path: Path,
    processor: Any,
    llm: Any,
    mm_min_pixels: int,
    mm_max_pixels: int,
    mm_do_rescale: bool,
) -> dict[str, Any]:
    images = [Image.open(path).convert("RGB") for path in image_paths]
    preprocess_result = processor.image_processor.preprocess(
        images=images,
        do_rescale=mm_do_rescale,
        min_pixels=mm_min_pixels,
        max_pixels=mm_max_pixels,
        return_tensors="pt",
    ).data
    pixel_values = preprocess_result["pixel_values"]
    image_grid_thw = preprocess_result["image_grid_thw"]

    def get_image_embeds(model):
        with torch.no_grad():
            visual = model.visual
            pixel_values_on_device = pixel_values.to(
                visual.device,
                dtype=visual.dtype,
            )
            return visual(pixel_values_on_device, grid_thw=image_grid_thw).cpu()

    image_embeds = torch.cat(llm.apply_model(get_image_embeds), dim=0)

    image_embeds_path.parent.mkdir(parents=True, exist_ok=True)
    image_grid_thw_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(image_embeds, image_embeds_path)
    torch.save(image_grid_thw.cpu(), image_grid_thw_path)

    return {
        "image_embeds": str(image_embeds_path),
        "image_grid_thw": str(image_grid_thw_path),
        "image_embeds_shape": list(image_embeds.shape),
        "image_grid_thw_shape": list(image_grid_thw.shape),
        "mm_min_pixels": mm_min_pixels,
        "mm_max_pixels": mm_max_pixels,
        "mm_do_rescale": mm_do_rescale,
    }


def add_qwen35_image_embeddings(
    rows: list[dict[str, Any]],
    *,
    dataset_path: Path,
    embeds_dir: Path,
    args: argparse.Namespace,
) -> int:
    pending = [
        (idx, row)
        for idx, row in enumerate(rows)
        if not _row_has_existing_embedding_files(dataset_path, row)
    ]
    if not pending:
        return 0

    processor = None
    llm = None
    updated = 0
    for idx, row in pending:
        row_id = row.get("id") or row.get("clip_id") or row.get("cid")
        stem = f"{idx:05d}_{_safe_stem(row_id, f'row{idx:05d}')}"
        image_embeds_path = embeds_dir / f"{stem}_image_embeds.pt"
        image_grid_thw_path = embeds_dir / f"{stem}_image_grid_thw.pt"

        if image_embeds_path.exists() and image_grid_thw_path.exists():
            image_embeds = torch.load(image_embeds_path, map_location="cpu")
            image_grid_thw = torch.load(image_grid_thw_path, map_location="cpu")
            row.update(
                {
                    "image_embeds": str(image_embeds_path),
                    "image_grid_thw": str(image_grid_thw_path),
                    "image_embeds_shape": list(image_embeds.shape),
                    "image_grid_thw_shape": list(image_grid_thw.shape),
                    "mm_min_pixels": args.mm_min_pixels,
                    "mm_max_pixels": args.mm_max_pixels,
                    "mm_do_rescale": not args.disable_mm_do_rescale,
                }
            )
        else:
            if processor is None or llm is None:
                processor, llm = build_qwen35_embedding_components(args)
            image_paths = resolve_row_image_paths(dataset_path, row)
            row.update(
                compute_image_embedding_files(
                    image_paths=image_paths,
                    image_embeds_path=image_embeds_path,
                    image_grid_thw_path=image_grid_thw_path,
                    processor=processor,
                    llm=llm,
                    mm_min_pixels=args.mm_min_pixels,
                    mm_max_pixels=args.mm_max_pixels,
                    mm_do_rescale=not args.disable_mm_do_rescale,
                )
            )
        updated += 1
        print(f"  added embeddings [{updated}/{len(pending)}] row={row_id or idx}")

    return updated


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


def build_sample_and_model_input(
    clip_id: str,
    args: argparse.Namespace,
    components: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
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
    output_jsonl = Path(args.output_jsonl or output_dir / "dataset.jsonl")
    embeds_dir = Path(args.embeds_dir or output_dir / "image_embeds")

    if output_jsonl.exists():
        rows = read_jsonl(output_jsonl)
        if not rows:
            raise RuntimeError(f"Existing JSONL is empty: {output_jsonl}")
        print(f"Found existing JSONL with {len(rows)} rows -> {output_jsonl}")
        updated = add_qwen35_image_embeddings(
            rows,
            dataset_path=output_jsonl,
            embeds_dir=embeds_dir,
            args=args,
        )
        write_jsonl(output_jsonl, rows)
        print(f"Updated {updated} rows with image_grid_thw/image_embeds.")
        print(f"JSONL -> {output_jsonl}")
        print(f"Embeddings -> {embeds_dir}")
        return

    image_dir.mkdir(parents=True, exist_ok=True)

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

    add_qwen35_image_embeddings(
        rows,
        dataset_path=output_jsonl,
        embeds_dir=embeds_dir,
        args=args,
    )
    write_jsonl(output_jsonl, rows)

    print(f"Wrote {len(rows)} rows -> {output_jsonl}")
    print(f"Images -> {image_dir}")
    print(f"Embeddings -> {embeds_dir}")


if __name__ == "__main__":
    main()

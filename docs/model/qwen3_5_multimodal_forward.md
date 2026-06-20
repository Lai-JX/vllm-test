# vLLM 中 Qwen3.5 多模态 Forward 路径

本文总结当前本地 vLLM 代码里 dense Qwen3.5 的多模态 forward 路径。主线是 vLLM V1 GPU runner 的这条调用链：

```text
vllm/v1/worker/gpu_model_runner.py
  -> self._execute_mm_encoder(...)
      -> model.embed_multimodal(...)
          -> Qwen3VLForConditionalGeneration.embed_multimodal(...)
              -> self._process_image_input(...)
                  -> self.visual(pixel_values, grid_thw=grid_thw)
              -> self._process_video_input(...)
                  -> self.visual(pixel_values_videos, grid_thw=grid_thw)
  -> self._gather_mm_embeddings(...)
  -> model.embed_input_ids(..., multimodal_embeddings=..., is_multimodal=...)
      -> Qwen3_5ForConditionalGeneration.embed_input_ids(...)
          -> _merge_multimodal_embeddings(...)
  -> gpu_model_runner._call_model(..., inputs_embeds=inputs_embeds)
      -> self.model(..., inputs_embeds=inputs_embeds)
          -> Qwen3_5ForConditionalGeneration.forward(...)
              -> self.language_model.model(..., inputs_embeds=inputs_embeds)
```

简化后就是：

```text
pixel_values / pixel_values_videos
  -> self.visual(...)  # ViT
  -> visual embeddings
  -> merge into text embeddings
  -> LLM forward
```

最关键的一点是：`self.visual(...)` **不是在** `Qwen3_5ForConditionalGeneration.forward()` 里调用的。它发生得更早，由 vLLM worker 通过 `model.embed_multimodal(...)` 触发。等执行到 Qwen3.5 自己的 `forward(...)` 时，图片/视频特征已经被编码并合并进 `inputs_embeds` 了。

这条链路对应的源码检查点：

- `vllm/v1/worker/gpu_model_runner.py` 里的 `_execute_mm_encoder(...)` 调用 `model.embed_multimodal(...)`。
- `vllm/model_executor/models/qwen3_vl.py` 里的 `Qwen3VLForConditionalGeneration.embed_multimodal(...)` 分发到 `_process_image_input(...)` 和 `_process_video_input(...)`。
- 当输入是原始像素时，`_process_image_input(...)` 和 `_process_video_input(...)` 会直接调用 `self.visual(...)`。
- `gpu_model_runner.py` 随后带着 `multimodal_embeddings` 和 `is_multimodal` 调用 `self.model.embed_input_ids(...)`。
- `vllm/model_executor/models/qwen3_5.py` 里的 Qwen3.5 `embed_input_ids(...)` 调用 `_merge_multimodal_embeddings(...)`。
- `gpu_model_runner._call_model(...)` 调用 `self.model(...)`，进入 `Qwen3_5ForConditionalGeneration.forward(...)`。

## 涉及文件

- `vllm/model_executor/models/qwen3_5.py`
  - 定义 `Qwen3_5ForConditionalGeneration`。
  - 创建 `self.visual` 和 `self.language_model`。
  - 定义 Qwen3.5 自己的 `embed_input_ids(...)` 和 `forward(...)`。

- `vllm/model_executor/models/qwen3_vl.py`
  - 定义 `Qwen3VLForConditionalGeneration`。
  - 提供 Qwen3.5 继承使用的多模态方法，包括 `embed_multimodal(...)`、`_process_image_input(...)` 和 `_process_video_input(...)`。

- `vllm/v1/worker/gpu_model_runner.py`
  - 在模型 forward 前调用 `model.embed_multimodal(...)`。
  - 调用 `model.embed_input_ids(...)`，把视觉 embedding 合并进 token embedding。
  - 通过 `_call_model(...)` 调用 `self.model(...)`，最终进入 `Qwen3_5ForConditionalGeneration.forward(...)`。

## 类继承关系

Dense Qwen3.5 的类定义是：

```python
class Qwen3_5ForConditionalGeneration(Qwen3VLForConditionalGeneration, IsHybrid):
    ...
```

因此 Qwen3.5 继承了 `Qwen3VLForConditionalGeneration` 的多模态辅助方法。尤其是：Qwen3.5 自己没有定义 `embed_multimodal(...)`，它继承自父类：

```text
Qwen3_5ForConditionalGeneration
  -> Qwen3VLForConditionalGeneration
      -> embed_multimodal(...)
      -> _process_image_input(...)
      -> _process_video_input(...)
```

运行时，父类方法里访问 `self.visual` 时，`self` 仍然是 Qwen3.5 实例。

## 1. Qwen3.5 创建 ViT 和 LLM

在 `qwen3_5.py` 中，`Qwen3_5ForConditionalGeneration.__init__` 同时创建视觉塔和语言模型：

```python
with self._mark_tower_model(vllm_config, {"image", "video"}):
    self.visual = Qwen3_VisionTransformer(
        config.vision_config,
        norm_eps=getattr(config, "rms_norm_eps", 1e-6),
        quant_config=quant_config,
        prefix=maybe_prefix(prefix, "visual"),
    )

with self._mark_language_model(vllm_config):
    self.language_model = Qwen3_5ForCausalLM(
        vllm_config=vllm_config,
        prefix=maybe_prefix(prefix, "language_model"),
    )
```

也就是说，Qwen3.5 模型对象上有两个核心子模块：

```text
self.visual          # Qwen3_VisionTransformer，用来编码图片/视频
self.language_model  # Qwen3_5ForCausalLM，用来做 LLM forward/logits
```

## 2. vLLM worker 调用 `embed_multimodal(...)`

在 `gpu_model_runner.py` 中，LLM forward 前，vLLM 会先判断模型是否支持多模态输入。如果支持，就先运行多模态 encoder：

```python
if self.supports_mm_inputs and is_first_rank and not is_encoder_decoder:
    # Run the multimodal encoder if any.
    with self.maybe_get_ec_connector_output(
        scheduler_output,
        encoder_cache=self.encoder_cache,
    ) as ec_connector_output:
        self._execute_mm_encoder(scheduler_output)
        mm_embeds, is_mm_embed = self._gather_mm_embeddings(scheduler_output)

    inputs_embeds_scheduled = self.model.embed_input_ids(
        self.input_ids.gpu[:num_scheduled_tokens],
        multimodal_embeddings=mm_embeds,
        is_multimodal=is_mm_embed,
    )
```

真正触发多模态 encoder 的直接调用发生在 `_execute_mm_encoder(...)` 里：

```python
batch_outputs = model.embed_multimodal(**mm_kwargs_batch)
```

在某些视频 pruning 路径里，它会做 micro-batch，然后调用：

```python
micro_batch_outputs = model.embed_multimodal(
    **micro_batch_mm_inputs
)
```

另外，如果走分离的 multimodal encoder runner 路径，`vllm/v1/worker/gpu/mm/encoder_runner.py` 里也会调用：

```python
batch_outputs = self.model.embed_multimodal(**mm_kwargs_batch)
```

所以，真正触发视觉编码的第一层调用者是 worker，不是 `Qwen3_5ForConditionalGeneration.forward(...)`。

## 3. `embed_multimodal(...)` 分发到图片或视频处理

Qwen3.5 继承了 `Qwen3VLForConditionalGeneration` 的 `embed_multimodal(...)`：

```python
def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings | None:
    mm_input_by_modality = self._parse_and_validate_multimodal_inputs(**kwargs)
    if not mm_input_by_modality:
        return None

    # The result multimodal_embeddings is tuple of tensors, with each
    # tensor corresponding to a multimodal data item (image or video).
    multimodal_embeddings: list[torch.Tensor] = []

    # NOTE: It is important to iterate over the keys in this dictionary
    # to preserve the order of the modalities.
    for modality in mm_input_by_modality:
        multimodal_input = mm_input_by_modality[modality]
        if modality == "image":
            image_embeddings = self._process_image_input(multimodal_input)
            image_embeddings = self._postprocess_image_embeds_evs(
                image_embeddings, multimodal_input
            )
            multimodal_embeddings.extend(image_embeddings)
        if modality == "video":
            video_embeddings = self._process_video_input(multimodal_input)
            if self.is_multimodal_pruning_enabled:
                video_embeddings = self._postprocess_video_embeds_evs(
                    video_embeddings, multimodal_input
                )
            multimodal_embeddings.extend(video_embeddings)

    embeddings_tuple = tuple(multimodal_embeddings)
    return embeddings_tuple
```

对 Qwen3.5 来说，`self.is_multimodal_pruning_enabled` 被设置为 `False`，所以通常不会走 EVS pruning 分支。

## 4. 图片路径：`pixel_values -> self.visual(...)`

图片输入由 `_process_image_input(...)` 处理，这个方法继承自 `Qwen3VLForConditionalGeneration`：

```python
def _process_image_input(
    self, image_input: Qwen2_5_VLImageInputs
) -> tuple[torch.Tensor, ...]:
    grid_thw = image_input["image_grid_thw"]
    assert grid_thw.ndim == 2

    if image_input["type"] == "image_embeds":
        image_embeds = image_input["image_embeds"].type(self.visual.dtype)
    else:
        pixel_values = image_input["pixel_values"].type(self.visual.dtype)
        if self.use_data_parallel:
            return run_dp_sharded_mrope_vision_model(
                self.visual, pixel_values, grid_thw.tolist(), rope_type="rope_3d"
            )
        else:
            image_embeds = self.visual(pixel_values, grid_thw=grid_thw)

    # Split concatenated embeddings for each image item.
    merge_size = self.visual.spatial_merge_size
    sizes = (grid_thw.prod(-1) // merge_size // merge_size).tolist()
    return image_embeds.split(sizes)
```

非 data parallel 的图片路径核心就是：

```python
image_embeds = self.visual(pixel_values, grid_thw=grid_thw)
```

也就是：

```text
pixel_values
  -> 转成 self.visual.dtype
  -> self.visual(pixel_values, grid_thw=image_grid_thw)
  -> image_embeds
  -> 按图片 item split
```

如果请求里已经提供了 `image_embeds`，就会跳过 `self.visual(...)`。

## 5. 视频路径：`pixel_values_videos -> self.visual(...)`

视频输入路径类似：

```python
def _process_video_input(
    self, video_input: Qwen2_5_VLVideoInputs
) -> tuple[torch.Tensor, ...]:
    grid_thw = video_input["video_grid_thw"]
    assert grid_thw.ndim == 2

    if video_input["type"] == "video_embeds":
        video_embeds = video_input["video_embeds"].type(self.visual.dtype)
    else:
        pixel_values_videos = video_input["pixel_values_videos"].type(
            self.visual.dtype
        )
        if self.use_data_parallel:
            grid_thw_list = grid_thw.tolist()
            return run_dp_sharded_mrope_vision_model(
                self.visual, pixel_values_videos, grid_thw_list, rope_type="rope_3d"
            )
        else:
            video_embeds = self.visual(pixel_values_videos, grid_thw=grid_thw)

    # Split concatenated embeddings for each video item.
    merge_size = self.visual.spatial_merge_size
    sizes = (grid_thw.prod(-1) // merge_size // merge_size).tolist()
    return video_embeds.split(sizes)
```

非 data parallel 的视频路径核心是：

```python
video_embeds = self.visual(pixel_values_videos, grid_thw=grid_thw)
```

也就是：

```text
pixel_values_videos
  -> 转成 self.visual.dtype
  -> self.visual(pixel_values_videos, grid_thw=video_grid_thw)
  -> video_embeds
  -> 按 video item split
```

如果请求里已经提供了 `video_embeds`，就会跳过 `self.visual(...)`。

## 6. worker 收集视觉 embeddings

`_execute_mm_encoder(...)` 完成后，vLLM 会收集视觉 embedding，并构造一个布尔 mask，标记哪些 token 位置应该被视觉 embedding 替换：

```python
mm_embeds, is_mm_embed = self._gather_mm_embeddings(scheduler_output)
```

概念上可以理解为：

```text
mm_embeds    # self.visual(...) 返回的 image/video embeddings
is_mm_embed  # 哪些位置需要替换成视觉 embeddings 的 mask
```

## 7. Qwen3.5 将视觉 embeddings 合并进文本 embeddings

worker 接着调用 Qwen3.5 的 `embed_input_ids(...)`：

```python
inputs_embeds_scheduled = self.model.embed_input_ids(
    self.input_ids.gpu[:num_scheduled_tokens],
    multimodal_embeddings=mm_embeds,
    is_multimodal=is_mm_embed,
)
```

Qwen3.5 在 `qwen3_5.py` 中定义了自己的 `embed_input_ids(...)`：

```python
def embed_input_ids(
    self,
    input_ids: torch.Tensor,
    multimodal_embeddings: MultiModalEmbeddings | None = None,
    *,
    is_multimodal: torch.Tensor | None = None,
) -> torch.Tensor:
    inputs_embeds = self._embed_text_input_ids(
        input_ids,
        self.language_model.embed_input_ids,
        is_multimodal=is_multimodal,
    )

    if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
        return inputs_embeds

    is_multimodal = _require_is_multimodal(is_multimodal)

    inputs_embeds = _merge_multimodal_embeddings(
        inputs_embeds=inputs_embeds,
        multimodal_embeddings=multimodal_embeddings,
        is_multimodal=is_multimodal,
    )

    return inputs_embeds
```

这一步做了两件事：

```text
input_ids
  -> self.language_model.embed_input_ids(...)
  -> 文本 token embeddings

文本 token embeddings + 视觉 embeddings + is_multimodal mask
  -> _merge_multimodal_embeddings(...)
  -> 最终 inputs_embeds
```

也就是说，这个方法返回后，`inputs_embeds` 中的图片/视频 placeholder 位置已经被替换成视觉 embeddings。

## 8. worker 带着 `inputs_embeds` 调用 Qwen3.5 forward

准备好 `inputs_embeds` 后，runner 会通过 `_call_model(...)` 调用模型 forward：

```python
return self.model(
    input_ids=input_ids,
    positions=positions,
    intermediate_tensors=intermediate_tensors,
    inputs_embeds=inputs_embeds,
    **model_kwargs,
)
```

对多模态模型来说，关键输入已经是 `inputs_embeds`，因为视觉 embedding 已经合并进去了。

## 9. Qwen3.5 forward 执行 LLM

Qwen3.5 自己的 forward 很短：

```python
def forward(
    self,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    intermediate_tensors: IntermediateTensors | None = None,
    inputs_embeds: torch.Tensor | None = None,
    **kwargs: object,
) -> torch.Tensor | IntermediateTensors:
    """Run forward pass for Qwen3.5."""

    if intermediate_tensors is not None:
        inputs_embeds = None

    hidden_states = self.language_model.model(
        input_ids=input_ids,
        positions=positions,
        intermediate_tensors=intermediate_tensors,
        inputs_embeds=inputs_embeds,
    )

    return hidden_states
```

如果这是一个多模态请求，此时 `inputs_embeds` 已经包含视觉 embedding。LLM 不会再次调用 `self.visual(...)`。

## 端到端 dense Qwen3.5 多模态路径

最终，dense Qwen3.5 的多模态路径可以概括为：

```text
请求包含 image/video 数据
  -> scheduler/model runner 提取 multimodal kwargs
  -> gpu_model_runner._execute_mm_encoder(...)
  -> model.embed_multimodal(...)
      -> Qwen3VLForConditionalGeneration.embed_multimodal(...)
          -> _process_image_input(...)
              -> self.visual(pixel_values, grid_thw=image_grid_thw)
          -> _process_video_input(...)
              -> self.visual(pixel_values_videos, grid_thw=video_grid_thw)
  -> visual embeddings 返回给 model runner
  -> gpu_model_runner._gather_mm_embeddings(...)
  -> model.embed_input_ids(input_ids, multimodal_embeddings, is_multimodal)
      -> text embeddings
      -> _merge_multimodal_embeddings(...)
      -> merged inputs_embeds
  -> gpu_model_runner._call_model(...)
  -> Qwen3_5ForConditionalGeneration.forward(...)
      -> self.language_model.model(..., inputs_embeds=inputs_embeds)
  -> Qwen3.5 LLM forward
  -> hidden_states
  -> compute_logits / sampling outside this forward
```

## 关键结论

`self.visual(...)` 由继承自 Qwen3-VL 的多模态 encoder 方法调用，而这些方法由 vLLM worker 在 Qwen3.5 常规 LLM forward 之前触发。

所以，只看 `Qwen3_5ForConditionalGeneration.forward(...)` 找不到 ViT 调用是正常的：等 `forward(...)` 收到 `inputs_embeds` 时，ViT 阶段已经完成了。

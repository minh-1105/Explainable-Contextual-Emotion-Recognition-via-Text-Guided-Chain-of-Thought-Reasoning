"""Load Qwen3-VL and run batched greedy generation on image-prompt pairs."""

from __future__ import annotations

import torch


def load_qwen(model_id: str, max_pixels: int, device: str = "cuda"):
    """Load processor and model in bf16 (fp16 before Ampere) with left padding."""
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    processor = AutoProcessor.from_pretrained(model_id, max_pixels=max_pixels)
    processor.tokenizer.padding_side = "left"
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=dtype, attn_implementation="sdpa", low_cpu_mem_usage=True,
    ).to(device).eval()
    return processor, model


@torch.inference_mode()
def ask_qwen_batch(processor, model, images, prompts, max_new_tokens: int):
    """Generate one greedy answer per image-prompt pair.

    Returns the stripped texts and, per answer, whether it hit ``max_new_tokens``
    without an end or padding token.
    """
    messages = [
        [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        for image, prompt in zip(images, prompts, strict=True)
    ]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt", padding=True,
    )
    inputs.pop("token_type_ids", None)
    inputs = inputs.to(model.device)
    output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated = output[:, inputs["input_ids"].shape[1]:]
    texts = processor.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    eos = model.generation_config.eos_token_id
    stop = torch.tensor(
        (eos if isinstance(eos, list) else [eos]) + [processor.tokenizer.pad_token_id], device=generated.device
    )
    truncated = (~torch.isin(generated, stop).any(dim=1)).tolist()
    return [text.strip() for text in texts], truncated


__all__ = ["ask_qwen_batch", "load_qwen"]

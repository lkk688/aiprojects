#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
merge_qwen_lora.py

Merge a LoRA adapter into a base causal LM or VLM and save a standalone merged model.

Example:
python merge_qwen_lora.py \
  --base_model Qwen/Qwen2-VL-7B-Instruct \
  --adapter_path output/qwenvl_sft_lora \
  --output_path output/qwenvl_sft_merged
"""

import os
import gc
import json
import time
import argparse
from typing import Optional

import torch
from transformers import AutoConfig


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def is_bf16_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability(0)
    return major >= 8


def print_header(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def check_if_vision_model(base_model: str, trust_remote_code: bool = True) -> bool:
    """Dynamically determine if the base model contains a vision tower."""
    try:
        config = AutoConfig.from_pretrained(base_model, trust_remote_code=trust_remote_code)
        arch = config.architectures[0].lower() if config.architectures else ""
        # Check for common VLM architecture keywords
        return any(keyword in arch for keyword in ["vl", "vision", "llava", "qwen2vl", "idefics"])
    except Exception as e:
        print(f"[WARN] Could not inspect config for vision components: {e}")
        return False


def try_merge_with_transformers_peft(
    base_model: str,
    adapter_path: str,
    output_path: str,
    torch_dtype: Optional[torch.dtype],
    device_map: str = "auto",
    trust_remote_code: bool = True,
    is_vision: bool = False,
):
    from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM, AutoModelForImageTextToText
    from peft import PeftModel

    print_header(f"Loading processor & tokenizer (Vision mode: {is_vision})")
    
    # Always attempt to load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=trust_remote_code)
    
    # If it's a vision model, we must grab the Processor to ensure the vision encoder has an input pipeline
    processor = None
    if is_vision:
        try:
            processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=trust_remote_code)
        except Exception as e:
            print(f"[WARN] Failed to load AutoProcessor. Vision inputs may fail later.\n{e}")

    print_header("Loading base model")
    # Route to the correct architecture class
    ModelClass = AutoModelForImageTextToText if is_vision else AutoModelForCausalLM
    
    model = ModelClass.from_pretrained(
        base_model,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
    )

    print_header("Loading LoRA adapter")
    model = PeftModel.from_pretrained(model, adapter_path)

    print_header("Merging LoRA into base model")
    merged_model = model.merge_and_unload()

    print_header(f"Saving merged model to {output_path}")
    ensure_dir(output_path)
    
    # Save the model in 2GB shards to prevent RAM OOMs on reload
    merged_model.save_pretrained(output_path, safe_serialization=True, max_shard_size="2GB")
    tokenizer.save_pretrained(output_path)
    if processor is not None:
        processor.save_pretrained(output_path)

    del model
    del merged_model
    del tokenizer
    if processor: del processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def try_merge_with_unsloth(
    base_model: str,
    adapter_path: str,
    output_path: str,
    max_seq_length: int = 4096,
    load_in_4bit: bool = False,
    is_vision: bool = False,
):
    # Conditionally load Unsloth vision vs text classes
    if is_vision:
        from unsloth import FastVisionModel as UnslothModel
    else:
        from unsloth import FastLanguageModel as UnslothModel
        
    from peft import PeftModel

    dtype = torch.bfloat16 if is_bf16_supported() else torch.float16

    print_header(f"Loading base model with Unsloth (Vision mode: {is_vision})")
    model, tokenizer = UnslothModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )

    print_header("Loading LoRA adapter with PEFT")
    model = PeftModel.from_pretrained(model, adapter_path)

    print_header("Merging LoRA into base model")
    merged_model = model.merge_and_unload()

    print_header(f"Saving merged model to {output_path}")
    ensure_dir(output_path)
    
    merged_model.save_pretrained(output_path, safe_serialization=True, max_shard_size="2GB")
    
    # Unsloth's tokenizer object acts as a Processor for VLMs
    tokenizer.save_pretrained(output_path)

    del model
    del merged_model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument("--base_model", type=str, required=True, help="Base model path or HF model id")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to LoRA adapter directory")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save merged model")
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--force_unsloth", action="store_true", help="Force Unsloth loading path")
    parser.add_argument("--cpu", action="store_true", help="Load/merge on CPU (slower, lower GPU usage)")
    args = parser.parse_args()

    if not os.path.exists(args.adapter_path):
        raise FileNotFoundError(f"Adapter path not found: {args.adapter_path}")

    ensure_dir(args.output_path)
    
    # Autodetect if we are merging a multimodal model
    is_vision = check_if_vision_model(args.base_model)

    print_header("Merge configuration")
    print(json.dumps({
        "base_model": args.base_model,
        "adapter_path": args.adapter_path,
        "output_path": args.output_path,
        "max_seq_length": args.max_seq_length,
        "force_unsloth": args.force_unsloth,
        "cpu": args.cpu,
        "is_vision": is_vision,
    }, indent=2))

    torch_dtype = None
    if not args.cpu:
        torch_dtype = torch.bfloat16 if is_bf16_supported() else torch.float16
    else:
        torch_dtype = torch.float32

    start_time = time.time()
    success = False
    last_error = None

    if not args.force_unsloth:
        try:
            try_merge_with_transformers_peft(
                base_model=args.base_model,
                adapter_path=args.adapter_path,
                output_path=args.output_path,
                torch_dtype=torch_dtype,
                device_map="cpu" if args.cpu else "auto",
                trust_remote_code=True,
                is_vision=is_vision,
            )
            success = True
        except Exception as e:
            last_error = e
            print(f"[WARN] Transformers+PEFT merge failed:\n{e}")

    if not success:
        try:
            try_merge_with_unsloth(
                base_model=args.base_model,
                adapter_path=args.adapter_path,
                output_path=args.output_path,
                max_seq_length=args.max_seq_length,
                load_in_4bit=False,
                is_vision=is_vision,
            )
            success = True
        except Exception as e:
            last_error = e
            print(f"[WARN] Unsloth merge failed:\n{e}")

    if not success:
        raise RuntimeError(f"LoRA merge failed. Last error:\n{last_error}")

    elapsed = time.time() - start_time
    print_header("Merge complete")
    print(f"Merged model saved to: {os.path.abspath(args.output_path)}")
    print(f"Elapsed time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()

"""
python CodeAgent/merge_qwen_lora.py \
  --base_model Qwen/Qwen3.5-9B \
  --adapter_path output/qwen9b_sft_lora \
  --output_path output/qwen9b_sft_merged
"""
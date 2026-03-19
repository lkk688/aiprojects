#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gc
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel

def recover_and_merge():
    base_model_id = "Qwen/Qwen3.5-9B"
    adapter_path = "output/qwen9b_sft_lora"
    output_path = "output/qwen9b_vlm_sft_merged"

    print(f"[*] 1. Loading original full VLM base model: {base_model_id}...")
    # Forcing AutoModelForImageTextToText prevents the vision tower from being dropped
    model = AutoModelForImageTextToText.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"[*] 2. Loading original processor...")
    processor = AutoProcessor.from_pretrained(base_model_id, trust_remote_code=True)

    print(f"[*] 3. Grafting SFT LoRA onto the VLM text backbone...")
    # PEFT will map the q_proj/v_proj weights directly to the LLM layers inside the VLM
    model = PeftModel.from_pretrained(model, adapter_path)

    print("[*] 4. Merging weights...")
    merged_model = model.merge_and_unload()

    print(f"[*] 5. Saving recovered multimodal model to: {output_path}")
    merged_model.save_pretrained(output_path, safe_serialization=True, max_shard_size="2GB")
    processor.save_pretrained(output_path)

    print("[+] Done! The vision tower is fully restored.")

    # Cleanup
    del model
    del merged_model
    del processor
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    recover_and_merge()
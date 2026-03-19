#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def test_merged_text_model(model_path: str):
    print(f"[*] Loading merged text model from: {model_path}")
    
    # 1. Load the Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # 2. Load the Model using the correct causal LM class
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    print("[+] Model and Tokenizer loaded successfully!")

    # 3. Format the Prompt
    messages = [
        {"role": "system", "content": "You are an expert software engineer."},
        {"role": "user", "content": "Write a clean, efficient Python function to calculate the factorial of a number."}
    ]

    print("[*] Processing text inputs...")
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    print("[*] Generating response...")
    
    # 4. Generate Output
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=256)

    # 5. Decode and format the output
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    output_text = tokenizer.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    print("\n" + "="*50)
    print("MODEL RESPONSE:")
    print("="*50)
    print(output_text[0].strip())
    print("="*50)

if __name__ == "__main__":
    # Pointing to your actual merged text model
    MERGED_MODEL_DIR = "output/qwen9b_sft_merged" 
    
    test_merged_text_model(MERGED_MODEL_DIR)
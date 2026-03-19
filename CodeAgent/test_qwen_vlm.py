#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info

def test_merged_vlm(model_path: str):
    print(f"[*] Loading merged model from: {model_path}")
    
    # 1. Load the Processor (Handles both Tokenization AND Image Processing)
    try:
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    except Exception as e:
        raise RuntimeError(f"Failed to load processor. The vision tower might not have saved correctly.\n{e}")

    # 2. Load the Model
    # Using AutoModelForImageTextToText ensures the vision weights are loaded
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    print("[+] Model and Processor loaded successfully!")

    # 3. Format the Multimodal Prompt
    # Qwen2-VL uses a list of dicts for the 'content' block to separate images and text
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image", 
                    "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
                },
                {
                    "type": "text", 
                    "text": "Describe this image in detail. What kind of dog is this?"
                },
            ],
        }
    ]

    print("[*] Processing image and text inputs...")
    
    # Apply ChatML template to the text portion
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    # Extract and process the image into pixel values
    image_inputs, video_inputs = process_vision_info(messages)
    
    # Pack everything into the final tensors
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    print("[*] Generating response...")
    
    # 4. Generate Output
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=128)

    # 5. Decode and format the output
    # Slice the generated_ids to remove the input prompt tokens from the printed output
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    print("\n" + "="*50)
    print("MODEL RESPONSE:")
    print("="*50)
    print(output_text[0])
    print("="*50)

if __name__ == "__main__":
    # Change this to the exact output path you used in your merge script
    #MERGED_MODEL_DIR = "output/qwen9b_sft_merged" 
    MERGED_MODEL_DIR = "output/qwen9b_vlm_sft_merged"
    
    test_merged_vlm(MERGED_MODEL_DIR)

"""
pip install qwen-vl-utils torchvision
"""
import os
import sys
import subprocess
import torch
import re

def install_dependencies():
    """
    Installs necessary dependencies for Unsloth and Qwen Coder finetuning.
    """
    print("Installing dependencies...")
    packages = [
        "unsloth",
        "unsloth_zoo",
        "bitsandbytes",
        "accelerate",
        "peft",
        "trl",
        "triton",
        "transformers==4.56.2",
        "sentencepiece",
        "protobuf",
        "datasets==4.3.0",
        "huggingface_hub>=0.34.0",
        "hf_transfer"
    ]
    
    try:
        v = re.match(r'[\d]{1,}\.[\d]{1,}', str(torch.__version__)).group(0)
        xformers_ver = {'2.10':'0.0.34','2.9':'0.0.33.post1','2.8':'0.0.32.post2'}.get(v, "0.0.34")
        packages.append(f"xformers=={xformers_ver}")
    except Exception as e:
        print(f"Could not determine specific xformers version: {e}. Skipping explicit xformers version pin.")

    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "--no-cache-dir"] + packages)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-deps", "trl==0.22.2"])
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        sys.exit(1)
    print("Dependencies installed successfully.")

def formatting_prompts_func(examples, tokenizer):
    """
    Formats the Evol-Instruct-Code dataset (User/Assistant style) into the chat template.
    Maps 'instruction' -> User, 'output' -> Assistant.
    """
    convos = []
    texts = []
    
    # Check column names
    # Evol-Instruct typically has 'instruction', 'output'
    instructions = examples.get("instruction", [])
    outputs = examples.get("output", [])
    
    for instruction, output in zip(instructions, outputs):
        conversation = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": output},
        ]
        convos.append(conversation)
        
    # Apply standard chat template
    return {
        "text": [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
    }

def main():
    # --- Imports ---
    try:
        from unsloth import FastLanguageModel
        from unsloth import is_bfloat16_supported
        from trl import SFTTrainer, SFTConfig
        from transformers import TextStreamer
        from datasets import load_dataset
    except ImportError:
        install_dependencies()
        from unsloth import FastLanguageModel
        from unsloth import is_bfloat16_supported
        from trl import SFTTrainer, SFTConfig
        from transformers import TextStreamer
        from datasets import load_dataset

    # --- Configuration ---
    # H100 Optimization Settings
    MAX_SEQ_LENGTH = 4096 # Increased for Coding tasks
    DTYPE = None
    LOAD_IN_4BIT = True
    
    # Training Params
    BATCH_SIZE = 16
    GRADIENT_ACCUMULATION_STEPS = 1
    LEARNING_RATE = 2e-5 # Lower LR for Instruct model preservation
    EPOCHS = 1 # Full epoch for longer training

    print("Loading Qwen2.5-Coder-14B-Instruct model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Qwen2.5-Coder-14B-Instruct",
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = DTYPE,
        load_in_4bit = LOAD_IN_4BIT,
    )

    print("Adding LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, # Efficient rank
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
        use_rslora = False,
        loftq_config = None,
    )

    print("Loading Evol-Instruct-Code-80k-v1 dataset...")
    # Using specific high quality code instruction dataset
    dataset = load_dataset("nickrosh/Evol-Instruct-Code-80k-v1", split = "train")
    
    print("Formatting dataset...")
    # We map formatting on the fly during training usually, but SFTTrainer likes a 'text' column if packing isn't used 
    # or we can pass a formatting func. Unsloth encourages pre-formatting to 'text'.
    dataset = dataset.map(lambda x: formatting_prompts_func(x, tokenizer), batched = True)
    
    print(f"Dataset Size: {len(dataset)}")

    print("Starting Training (This may take a while)...")
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = MAX_SEQ_LENGTH,
        dataset_num_proc = 2,
        packing = False, # Can set to True for speedup if sequences are short
        args = SFTConfig(
            per_device_train_batch_size = BATCH_SIZE,
            gradient_accumulation_steps = GRADIENT_ACCUMULATION_STEPS,
            warmup_steps = 100, # Increased warmup for full epoch
            num_train_epochs = EPOCHS,
            # max_steps = -1, # Use epochs
            learning_rate = LEARNING_RATE,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 10,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "outputs",
            report_to = "none",
        ),
    )

    # Show memory stats
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    trainer_stats = trainer.train()
    
    # Final Stats
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    print(f"{trainer_stats.metrics['train_runtime']/60:.2f} minutes used for training.")
    print(f"Peak reserved memory = {used_memory} GB.")

    # --- Inference Test ---
    print("\n--- Running Inference Test (Coding) ---")
    FastLanguageModel.for_inference(model)
    
    messages = [
        {"role": "user", "content": "Write a Python function to calculate the Fibonacci sequence using dynamic programming."}
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize = True,
        add_generation_prompt = True,
        return_tensors = "pt",
    ).to("cuda")

    outputs = model.generate(
        input_ids = inputs,
        max_new_tokens = 512,
        temperature = 0.2, # Lower temperature for coding
        use_cache = True,
    )
    print(tokenizer.batch_decode(outputs)[0])

    print("\nSaving model to 'qwen_coder_lora'...")
    model.save_pretrained("qwen_coder_lora")
    tokenizer.save_pretrained("qwen_coder_lora")

if __name__ == "__main__":
    main()

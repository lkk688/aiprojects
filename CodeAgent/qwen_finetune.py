import os
import sys
import subprocess
import torch

def install_dependencies():
    """
    Installs necessary dependencies for Unsloth and Qwen finetuning.
    This mimics the installation cells in the Colab notebook.
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
    
    # Check for specific xformers version based on torch version if possible, 
    # but for simplicity and robustness on H100 with likely recent torch, 
    # we'll let unsloth handle its deps or install a compatible one.
    # The notebook logic for xformers:
    try:
        import re
        v = re.match(r'[\d]{1,}\.[\d]{1,}', str(torch.__version__)).group(0)
        xformers_ver = {'2.10':'0.0.34','2.9':'0.0.33.post1','2.8':'0.0.32.post2'}.get(v, "0.0.34")
        packages.append(f"xformers=={xformers_ver}")
    except Exception as e:
        print(f"Could not determine specific xformers version: {e}. Skipping explicit xformers version pin.")

    try:
        # Using pip to install. 
        # Note: In a real server script, you might want to check if they are already installed.
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "--no-cache-dir"] + packages)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-deps", "trl==0.22.2"])
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        sys.exit(1)
    print("Dependencies installed successfully.")

def main():
    # --- Configuration ---
    # H100 Optimization Settings
    MAX_SEQ_LENGTH = 2048
    DTYPE = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    LOAD_IN_4BIT = True # Use 4bit quantization to reduce memory usage. Can be False for H100 if we want higher precision, but 4bit + LoRA is standard unsloth flow.
    
    # H100 typical settings
    BATCH_SIZE = 16 # Increased from 2 (Colab) to 16 for H100
    GRADIENT_ACCUMULATION_STEPS = 1 # Reduced from 4 (Colab) since we have larger Batch Size
    # Note: 16 * 1 = 16 effective batch size. Colab was 2 * 4 = 8.
    
    # --- Imports ---
    try:
        from unsloth import FastLanguageModel, is_bfloat16_supported
        from trl import SFTTrainer, SFTConfig
        from transformers import TextStreamer
        from datasets import load_dataset, Dataset
        import pandas as pd
    except ImportError:
        print("Required packages not found. Attempting to install...")
        install_dependencies()
        from unsloth import FastLanguageModel, is_bfloat16_supported
        from trl import SFTTrainer, SFTConfig
        from transformers import TextStreamer
        from datasets import load_dataset, Dataset
        import pandas as pd

    # --- 1. Load Model ---
    print("Loading Qwen3-14B model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Qwen3-14B",
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = DTYPE,
        load_in_4bit = LOAD_IN_4BIT,
    )

    # --- 2. Add LoRA Adapters ---
    print("Adding LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r = 32, # Choose any number > 0! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 32,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )

    # --- 3. Data Preparation ---
    print("Loading and preparing datasets...")
    # Load default datasets from the notebook
    reasoning_dataset = load_dataset("unsloth/OpenMathReasoning-mini", split = "cot")
    non_reasoning_dataset = load_dataset("mlabonne/FineTome-100k", split = "train")

    # Format Reasoning Dataset
    def generate_conversation(examples):
        problems  = examples["problem"]
        solutions = examples["generated_solution"]
        conversations = []
        for problem, solution in zip(problems, solutions):
            conversations.append([
                {"role" : "user",      "content" : problem},
                {"role" : "assistant", "content" : solution},
            ])
        return { "conversations": conversations, }

    reasoning_conversations = tokenizer.apply_chat_template(
        list(reasoning_dataset.map(generate_conversation, batched = True)["conversations"]),
        tokenize = False,
    )

    # Format Non-Reasoning Dataset (Standardize ShareGPT)
    from unsloth.chat_templates import standardize_sharegpt
    dataset = standardize_sharegpt(non_reasoning_dataset)
    
    non_reasoning_conversations = tokenizer.apply_chat_template(
        list(dataset["conversations"]),
        tokenize = False,
    )

    # Mix Datasets (25% Non-Reasoning)
    chat_percentage = 0.25
    print(f"Mixing datasets with {chat_percentage*100}% non-reasoning data...")
    
    non_reasoning_subset = pd.Series(non_reasoning_conversations)
    non_reasoning_subset = non_reasoning_subset.sample(
        int(len(reasoning_conversations)*(chat_percentage/(1 - chat_percentage))),
        random_state = 2407,
    )
    
    data = pd.concat([
        pd.Series(reasoning_conversations),
        pd.Series(non_reasoning_subset)
    ])
    data.name = "text"
    
    combined_dataset = Dataset.from_pandas(pd.DataFrame(data))
    combined_dataset = combined_dataset.shuffle(seed = 3407)
    
    print(f"Total training examples: {len(combined_dataset)}")

    # --- 4. Training ---
    print("Starting training on H100...")
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = combined_dataset,
        eval_dataset = None,
        args = SFTConfig(
            dataset_text_field = "text",
            max_seq_length = MAX_SEQ_LENGTH,
            per_device_train_batch_size = BATCH_SIZE, # 16 for H100
            gradient_accumulation_steps = GRADIENT_ACCUMULATION_STEPS, # 1 for H100
            warmup_steps = 5,
            # num_train_epochs = 1, # Set this for 1 full training run.
            max_steps = 30, # Keeping notebook default for quick demo, increase for real training
            learning_rate = 2e-4,
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.001,
            lr_scheduler_type = "linear",
            seed = 3407,
            report_to = "none",
            bf16 = is_bfloat16_supported(), # Enable BF16 for H100
            fp16 = not is_bfloat16_supported(),
        ),
    )

    # Show memory stats
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    trainer_stats = trainer.train()

    # Show final stats
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")

    # --- 5. Inference ---
    print("\n--- Running Inference (Without Thinking) ---")
    FastLanguageModel.for_inference(model) # Enable native 2x faster inference
    
    messages = [
        {"role" : "user", "content" : "Solve (x + 2)^2 = 0."}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize = False,
        add_generation_prompt = True,
        enable_thinking = False, # Disable thinking
    )
    
    model.generate(
        **tokenizer(text, return_tensors = "pt").to("cuda"),
        max_new_tokens = 256,
        temperature = 0.7, top_p = 0.8, top_k = 20,
        streamer = TextStreamer(tokenizer, skip_prompt = True),
    )
    
    print("\n\n--- Running Inference (With Thinking) ---")
    text = tokenizer.apply_chat_template(
        messages,
        tokenize = False,
        add_generation_prompt = True,
        enable_thinking = True, # Enable thinking
    )
    
    model.generate(
        **tokenizer(text, return_tensors = "pt").to("cuda"),
        max_new_tokens = 1024,
        temperature = 0.6, top_p = 0.95, top_k = 20,
        streamer = TextStreamer(tokenizer, skip_prompt = True),
    )

    # --- 6. Saving ---
    print("\nSaving model to 'qwen_lora'...")
    model.save_pretrained("qwen_lora")
    tokenizer.save_pretrained("qwen_lora")
    print("Model saved successfully.")

    # Optional: Merge to 16bit and save (Commented out)
    # model.save_pretrained_merged("qwen_finetune_16bit", tokenizer, save_method = "merged_16bit")
    # model.save_pretrained_gguf("qwen_finetune", tokenizer, quantization_method = "q4_k_m")

if __name__ == "__main__":
    main()

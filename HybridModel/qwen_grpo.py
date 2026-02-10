import os
import sys
import subprocess
import torch
import re

def install_dependencies():
    """
    Installs necessary dependencies for Unsloth and Qwen GRPO finetuning.
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
        "hf_transfer", 
        "vllm" # Added for GRPO fast inference
    ]
    
    # Check for specific xformers version based on torch version if possible
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

# --- Globals for Reward Functions ---
reasoning_start = "<start_working_out>"
reasoning_end   = "<end_working_out>"
solution_start  = "<SOLUTION>"
solution_end    = "</SOLUTION>"

def extract_hash_answer(text):
    # if "####" not in text: return None
    # return text.split("####")[1].strip()
    return text

def get_match_format():
    # Helper to clean up global scope use
    import re
    # Add optional EOS token matching
    # We need tokenizer here, but it's not global yet. 
    # Logic in notebook compiles this regex globally.
    # We will compile it inside functions or pass it.
    pass

# --- 1. SFT Stage Functions ---

def format_dataset_sft(x, system_prompt):
    expected_answer = x["expected_answer"]
    problem = x["problem"]

    # Remove generated <think> and </think>
    thoughts = x["generated_solution"]
    thoughts = thoughts.replace("<think>", "").replace("</think>", "")

    # Strip newlines on left and right
    thoughts = thoughts.strip()
    # Add our custom formatting
    final_prompt = \
        reasoning_start + thoughts + reasoning_end + \
        solution_start + expected_answer + solution_end
    return [
        {"role" : "system",    "content" : system_prompt},
        {"role" : "user",      "content" : problem},
        {"role" : "assistant", "content" : final_prompt},
    ]

# --- 2. GRPO Reward Functions ---

def match_format_exactly(completions, **kwargs):
    import re
    scores = []
    
    # Regex for exact format
    # Note: Regex depends on tokenizer.eos_token which we need to access.
    # We'll re-compile it here for safety or assume standard EOS.
    # For simplicity, we assume generic structure first or pass it if possible.
    # In TRL, reward funcs get a lot of kwargs.
    
    # We'll use a simplified regex that essentially checks structure
    # logic from notebook:
    # solution_end_regex = r"</SOLUTION>[\s]{0,}" + "(?:" + re.escape(tokenizer.eos_token) + ")?"
    # match_format = re.compile(rf"{reasoning_end}.*?{solution_start}(.+?){solution_end_regex}rf"[\s]{{0,}}$", flags = re.MULTILINE | re.DOTALL)
    
    # Since we don't have tokenizer globally easily in this scope without implicit global, 
    # we'll approximate the EOS part or assume strict string ending.
    
    pattern = rf"{reasoning_end}.*?{solution_start}(.+?){solution_end}"
    match_format = re.compile(pattern, flags = re.MULTILINE | re.DOTALL)

    for completion in completions:
        score = 0
        response = completion[0]["content"]
        if match_format.search(response) is not None: 
            score += 3.0
        scores.append(score)
    return scores

def match_format_approximately(completions, **kwargs):
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        # score += 0.5 if response.count(reasoning_start) == 1 else -1.0
        score += 0.5 if response.count(reasoning_end)   == 1 else -1.0
        score += 0.5 if response.count(solution_start)  == 1 else -1.0
        score += 0.5 if response.count(solution_end)    == 1 else -1.0
        scores.append(score)
    return scores

def check_answer(prompts, completions, answer, **kwargs):
    question = prompts[0][-1]["content"]
    responses = [completion[0]["content"] for completion in completions]
    
    # Re-using the regex logic
    pattern = rf"{reasoning_end}.*?{solution_start}(.+?){solution_end}"
    match_format = re.compile(pattern, flags = re.MULTILINE | re.DOTALL)

    extracted_responses = [
        guess.group(1)
        if (guess := match_format.search(r)) is not None else None \
        for r in responses
    ]

    scores = []
    for guess, true_answer in zip(extracted_responses, answer):
        score = 0
        if guess is None:
            scores.append(-2.0)
            continue
        if guess == true_answer:
            score += 5.0
        elif guess.strip() == true_answer.strip():
            score += 3.5
        else:
            try:
                ratio = float(guess) / float(true_answer)
                if   ratio >= 0.9 and ratio <= 1.1: score += 2.0
                elif ratio >= 0.8 and ratio <= 1.2: score += 1.5
                else: score -= 2.5
            except:
                score -= 4.5
        scores.append(score)
    return scores

def check_numbers(prompts, completions, answer, **kwargs):
    # global PRINTED_TIMES
    # global PRINT_EVERY_STEPS
    # For script, we might skip printing or just logging
    
    responses = [completion[0]["content"] for completion in completions]
    
    match_numbers = re.compile(
        solution_start + r".*?[\s]{0,}([-]?[\d\.\,]{1,})",
        flags = re.MULTILINE | re.DOTALL
    )

    extracted_responses = [
        guess.group(1)
        if (guess := match_numbers.search(r)) is not None else None \
        for r in responses
    ]

    scores = []
    for guess, true_answer in zip(extracted_responses, answer):
        if guess is None:
            scores.append(-2.5)
            continue
        try:
            true_answer_val = float(true_answer.strip())
            guess_val       = float(guess.strip().replace(",", ""))
            scores.append(3.5 if guess_val == true_answer_val else -1.5)
        except:
            scores.append(0)
            continue
    return scores


def main():
    # --- Imports & Setup ---
    try:
        from unsloth import FastLanguageModel, PatchFastRL
        from unsloth import is_bfloat16_supported
        from trl import SFTTrainer, SFTConfig, GRPOConfig, GRPOTrainer
        from transformers import TextStreamer
        from datasets import load_dataset, Dataset
        import pandas as pd
        import numpy as np
        from vllm import SamplingParams
    except ImportError:
        install_dependencies()
        from unsloth import FastLanguageModel, PatchFastRL
        from unsloth import is_bfloat16_supported
        from trl import SFTTrainer, SFTConfig, GRPOConfig, GRPOTrainer
        from transformers import TextStreamer
        from datasets import load_dataset, Dataset
        import pandas as pd
        import numpy as np
        from vllm import SamplingParams

    PatchFastRL("GRPO", FastLanguageModel)
    
    # Config
    MAX_SEQ_LENGTH = 2048 # Can increase for longer reasoning
    LORA_RANK = 32
    # H100 settings
    GPU_MEMORY_UTILIZATION = 0.9 
    
    print("Loading Qwen3-4B-Base model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Qwen3-4B-Base",
        max_seq_length = MAX_SEQ_LENGTH,
        load_in_4bit = False, # False for LoRA 16bit / H100 optimization
        fast_inference = True, # Enable vllm fast inference
        max_lora_rank = LORA_RANK,
        gpu_memory_utilization = GPU_MEMORY_UTILIZATION,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = LORA_RANK,
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha = LORA_RANK*2,
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
    )

    # --- Setup System Prompt & Chat Template ---
    system_prompt = f"""You are given a problem.
Think about the problem and provide your working out.
Place it between {reasoning_start} and {reasoning_end}.
Then, provide your solution between {solution_start}{solution_end}"""

    # Custom Chat Template
    chat_template = \
    "{% if messages[0]['role'] == 'system' %}"\
        "{{ messages[0]['content'] + eos_token }}"\
        "{% set loop_messages = messages[1:] %}"\
    "{% else %}"\
        "{{ '{system_prompt}' + eos_token }}"\
        "{% set loop_messages = messages %}"\
    "{% endif %}"\
    "{% for message in loop_messages %}"\
        "{% if message['role'] == 'user' %}"\
            "{{ message['content'] }}"\
        "{% elif message['role'] == 'assistant' %}"\
            "{{ message['content'] + eos_token }}"\
        "{% endif %}"\
    "{% endfor %}"\
    "{% if add_generation_prompt %}{{ '{reasoning_start}' }}"\
    "{% endif %}"
    
    chat_template = chat_template.replace("'{system_prompt}'", f"'{system_prompt}'") \
                                 .replace("'{reasoning_start}'", f"'{reasoning_start}'")
    tokenizer.chat_template = chat_template

    # ==========================================
    # STAGE 1: SFT (Cold Start)
    # ==========================================
    print("\n=== Stage 1: SFT (Cold Start) ===")
    print("Loading OpenMathReasoning-mini dataset...")
    
    dataset_sft = load_dataset("unsloth/OpenMathReasoning-mini", split = "cot")
    dataset_sft = dataset_sft.to_pandas()[["expected_answer", "problem", "generated_solution"]]
    
    # Filter for numbers
    is_number = pd.to_numeric(pd.Series(dataset_sft["expected_answer"]), errors = "coerce").notnull()
    dataset_sft = dataset_sft.iloc[np.where(is_number)[0]]
    
    # Format
    dataset_sft["Messages"] = dataset_sft.apply(lambda x: format_dataset_sft(x, system_prompt), axis = 1)
    
    # Length Filtering
    dataset_sft["N"] = dataset_sft["Messages"].apply(lambda x: len(tokenizer.apply_chat_template(x)))
    dataset_sft = dataset_sft.loc[dataset_sft["N"] <= MAX_SEQ_LENGTH/2].copy()
    
    dataset_sft_final = Dataset.from_pandas(dataset_sft)
    dataset_sft_final = dataset_sft_final.map(lambda x: {"text": tokenizer.apply_chat_template(x["Messages"], tokenize=False)})

    print("Starting SFT Training...")
    trainer_sft = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset_sft_final,
        args = SFTConfig(
            dataset_text_field = "text",
            per_device_train_batch_size = 16, # H100 optimized
            gradient_accumulation_steps = 1,
            warmup_steps = 5,
            num_train_epochs = 1, 
            learning_rate = 2e-4,
            logging_steps = 10,
            optim = "adamw_8bit",
            weight_decay = 0.001,
            lr_scheduler_type = "linear",
            seed = 3407,
            report_to = "none",
            bf16 = is_bfloat16_supported(),
        ),
    )
    trainer_sft.train()
    print("SFT Training Complete.")
    
    # Save SFT model mainly for checkpointing
    model.save_lora("qwen_sft_lora")
    
    # Clean up memory
    import gc
    del dataset_sft, dataset_sft_final, trainer_sft
    torch.cuda.empty_cache()
    gc.collect()

    # ==========================================
    # STAGE 2: GRPO
    # ==========================================
    print("\n=== Stage 2: GRPO ===")
    print("Loading DAPO-Math-17k-Processed dataset...")
    
    dataset_grpo = load_dataset("open-r1/DAPO-Math-17k-Processed", "en", split = "train")
    
    # Preprocess GRPO dataset
    dataset_grpo = dataset_grpo.map(lambda x: {
        "prompt" : [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": x["prompt"]},
        ],
        "answer": extract_hash_answer(x["solution"]),
    })
    
    # Length filtering for GRPO
    tokenized = dataset_grpo.map(
        lambda x: {"tokens" : tokenizer.apply_chat_template(x["prompt"], add_generation_prompt = True, tokenize = True)},
        batched = True,
    )
    tokenized = tokenized.map(lambda x: {"L" : len(x["tokens"])})
    maximum_length = int(np.quantile(tokenized["L"], 0.9))
    dataset_grpo = dataset_grpo.select(np.where(np.array(tokenized["L"]) <= maximum_length)[0])
    
    print(f"GRPO Max Prompt Length: {maximum_length}")
    
    # GRPO Config
    max_prompt_length = maximum_length + 1
    max_completion_length = MAX_SEQ_LENGTH - max_prompt_length
    
    # Config for H100
    # num_generations = 8 (increase from 4 for H100 to get better gradients)
    # per_device_train_batch_size = 1 (typically 1 for generation tasks to avoid OOM with large contexts)
    # gradient_accumulation_steps = 4 
    
    vllm_sampling_params = SamplingParams(
        min_p = 0.1,
        top_p = 1.0,
        top_k = -1,
        seed = 3407,
        stop = [tokenizer.eos_token],
        include_stop_str_in_output = True,
    )

    training_args = GRPOConfig(
        vllm_sampling_params = vllm_sampling_params,
        temperature = 1.0,
        learning_rate = 5e-6,
        weight_decay = 0.001,
        warmup_ratio = 0.1,
        lr_scheduler_type = "linear",
        optim = "adamw_8bit",
        logging_steps = 1,
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4, 
        num_generations = 8, 
        max_prompt_length = max_prompt_length,
        max_completion_length = max_completion_length,
        max_steps = 100, # Adjustable
        save_steps = 100,
        report_to = "none",
        output_dir = "grpo_outputs",
        bf16 = is_bfloat16_supported(),
    )

    print("Starting GRPO Training...")
    trainer_grpo = GRPOTrainer(
        model = model,
        processing_class = tokenizer,
        reward_funcs = [
            match_format_exactly,
            match_format_approximately,
            check_answer,
            check_numbers,
        ],
        args = training_args,
        train_dataset = dataset_grpo,
    )
    trainer_grpo.train()
    print("GRPO Training Complete.")

    # Save GRPO model
    print("Saving GRPO model...")
    model.save_lora("qwen_grpo_lora")
    
    # Inference Test
    print("\n--- Running Inference Test ---")
    text = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": "What is the sqrt of 101?"},
        ],
        add_generation_prompt = True,
        tokenize = False,
    )
    
    sampling_params_inf = SamplingParams(
        temperature = 0.8,
        top_k = 50,
        max_tokens = 1024,
    )
    
    # Fast inference with vLLM
    # Note: when using Unsloth + GRPO, the model object might be wrapped differently. 
    # But code in notebook uses `model.fast_generate` directly.
    try:
        output = model.fast_generate(
            text,
            sampling_params = sampling_params_inf,
            lora_request = model.load_lora("qwen_grpo_lora"),
        )[0].outputs[0].text
        print("Generated Output:")
        print(output)
    except Exception as e:
        print(f"Inference failed or using standard generation: {e}")

if __name__ == "__main__":
    main()

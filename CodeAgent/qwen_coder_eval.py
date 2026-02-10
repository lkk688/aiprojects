import os
import sys
import subprocess
import torch
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

# Safety warning for code execution
print("WARNING: This script executes generated code for HumanEval. Run within a sandboxed environment if possible.")

def install_dependencies():
    """Installs evaluation dependencies."""
    print("Installing evaluation dependencies...")
    packages = [
        "unsloth", "transformers", "datasets", "matplotlib", "seaborn", "scipy", "numpy", "tqdm"
    ]
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade"] + packages)
    except Exception as e:
        print(f"Dependency install failed: {e}")

# --- Metrics Implementation ---

def plot_results(base_res, fine_res, save_path="qwen_eval_comparison.png"):
    sns.set_theme(style="whitegrid")
    
    # Define metrics categories for better subplot organization if needed
    # For now, separate plots or a grouped bar chart if metrics valid
    
    # Filter valid keys present in both
    keys = [k for k in base_res.keys() if k in fine_res.keys()]
    if not keys: 
        print("No overlapping metrics to plot.")
        return

    n_metrics = len(keys)
    fig, axes = plt.subplots(1, n_metrics, figsize=(min(n_metrics * 5, 20), 6))
    if n_metrics == 1: axes = [axes]

    colors = sns.color_palette("muted")
    
    for i, metric in enumerate(keys):
        ax = axes[i]
        val_base = base_res.get(metric, 0)
        val_fine = fine_res.get(metric, 0)
        
        # Prepare data
        x = ["Base Model", "Finetuned"]
        y = [val_base, val_fine]
        
        # Plot
        bars = ax.bar(x, y, color=[colors[0], colors[1]], alpha=0.8, width=0.6)
        
        # Style
        ax.set_title(metric.replace("_", " "), fontsize=14, fontweight='bold', pad=15)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + (height * 0.01),
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.suptitle("Qwen2.5-Coder-14B: Base vs Finetuned Evaluation", fontsize=16, y=1.05)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Professional comparison plot saved to {save_path}")

def calculate_perplexity(model, tokenizer, dataset, max_length=2048, n_samples=100):
    """Calculates PPL on a specific dataset sample."""
    print(f"Calculating Perplexity (n={n_samples})...")
    model.eval()
    nalls = []
    
    # Take a sample
    if hasattr(dataset, "select"):
        subset = dataset.select(range(min(len(dataset), n_samples)))
    else:
        subset = dataset[:n_samples]
        
    for example in tqdm(subset, desc="PPL"):
        # Try multiple common column names
        text = ""
        for col in ["text", "content", "instruction", "input"]:
            if example.get(col):
                text = example[col]
                # If instruction, append output if exists for full context PPL
                if col == "instruction" and example.get("output"):
                    text += "\n" + example["output"]
                break
        
        if not text: 
            continue
        
        encodings = tokenizer(text, return_tensors="pt")
        # Ensure we don't exceed model limits or GPU memory
        input_ids = encodings.input_ids[:, :max_length].to("cuda")
        target_ids = input_ids.clone()
        
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            nalls.append(outputs.loss)

    if not nalls:
        return float('inf')

    ppl = torch.exp(torch.stack(nalls).mean())
    return ppl.item()


def calculate_induction_score(model, tokenizer, seq_len=256, n_samples=20):
    """
    Estimates Induction capabilities.
    Task: Repeat a random sequence of tokens.
    Score: Accuracy of predicting the repeated sequence.
    """
    print("Calculating Induction Score...")
    model.eval()
    scores = []
    
    for _ in range(n_samples):
        # Generate random tokens (simulating abstract symbols)
        random_ids = torch.randint(1000, 10000, (1, seq_len)).to("cuda")
        
        # Context: [Seq] [Seq]
        # We want to see if model predicts [Seq] given [Seq]
        context = torch.cat([random_ids, random_ids[:, :-1]], dim=1) # Leave last one for prediction check? 
        # Actually simpler: Context A B C ... A B C ...
        # Check PPL or Accuracy on second half.
        
        # Let's do raw accuracy of next token prediction for the second half
        vocab_size = model.config.vocab_size
        
        # Create a repeating sequence: A ... A (we want to see if it copies)
        # Input: [Random Sequence A]
        # Target: [Random Sequence A]
        
        # We construct input as [A] then asking for [A] is standard next token prediction if we give it [A] [A]
        # But induction head usually refers to [A] ... [B] ... [A] -> [B]
        # Simplified test: "Repeat this sequence: <random_seq>"
        pass # Implementation detail: usually we just look at loss on the repeated part.
        
        # Input: [Rand] [Rand]
        input_ids = torch.cat([random_ids, random_ids], dim=1)
        target_ids = input_ids.clone()
        # Mask out the first half targets
        target_ids[:, :seq_len] = -100
        
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            # This is just PPL of repetition. Lower is better. 
            # We convert to a 0-1 score roughly related to confidence.
            # Negative Log Likelihood
            nll = outputs.loss.item()
            scores.append(nll)
            
    # Lower NLL is better induction. Return 1/NLL or just NLL.
    # Let's return NLL.
    return np.mean(scores)

def retrieve_passkey(model, tokenizer, context_length=8192, needle_depth=0.5):
    """
    Needle in a Haystack test.
    Inserts a passkey at `needle_depth` (0.0 to 1.0) in a context of `context_length`.
    """
    print(f"Testing Passkey Retrieval (Ctx={context_length}, Depth={needle_depth:.2f})...")
    
    passkey = random.randint(10000, 99999)
    passkey_str = str(passkey)
    
    # Filler text
    filler = "The sun sets in the west. " * (context_length // 6) # Approximation
    tokens = tokenizer.encode(filler)
    
    # Trim to make space
    insert_idx = int(len(tokens) * needle_depth)
    
    needle_text = f"\nThe secret passkey is {passkey_str}.\n"
    needle_tokens = tokenizer.encode(needle_text)
    
    final_tokens = tokens[:insert_idx] + needle_tokens + tokens[insert_idx:]
    final_tokens = final_tokens[:context_length] # Clip if strictly needed
    
    # Prompt
    prompt_ids = torch.tensor([final_tokens]).to("cuda")
    prompt_str = tokenizer.decode(final_tokens)
    
    question = "\nWhat is the secret passkey?"
    input_text = prompt_str + question
    
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=20, 
            temperature=0.1
        )
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:])
    # print(f"Response: {response}")
    
    if passkey_str in response:
        return 1.0
    return 0.0

def eval_humaneval_subset(model, tokenizer, n_problems=20):
    """
    Evaluates on a subset of HumanEval.
    Uses basic functional correctness check.
    """
    print(f"Running HumanEval (Subset n={n_problems})...")
    from datasets import load_dataset
    ds = load_dataset("openai_humaneval", split="test")
    
    # Take random or first N
    ds = ds.select(range(n_problems))
    
    passed = 0
    
    for example in tqdm(ds, desc="HumanEval"):
        prompt = example["prompt"]
        test_code = example["test"]
        entry_point = example["entry_point"]
        
        # Generate
        messages = [{"role": "user", "content": f"Complete the following Python code:\n{prompt}"}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=512, 
                temperature=0.2
            )
        
        gen_code = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # Extract code block
        if "```python" in gen_code:
            gen_code = gen_code.split("```python")[1].split("```")[0]
        elif "```" in gen_code:
            gen_code = gen_code.split("```")[1].split("```")[0]
            
        # Combine prompt + generation + test
        # Note: HumanEval prompt is usually a function signature.
        # Ideally we just take the function body.
        # Simple heuristic: concat prediction to imports.
        # For strict HumanEval, we usually just run the prompt + gen.
        
        full_code = f"import math\nfrom typing import List, Dict, Tuple, Optional\n\n{prompt}\n{gen_code}\n\n{test_code}\n\ncheck({entry_point})"
        
        # Execute in separate process
        if run_unsafe_code(full_code):
            passed += 1
            
    return (passed / n_problems) * 100

def run_unsafe_code(code):
    """Runs code in simple sub-process. VERY UNSAFE. Use with caution."""
    # We wrap in a try-except block script
    enc_code = code.replace('"', '\\"') # Simple escaping
    
    wrapper = f"""
try:
{'\n'.join(['    ' + line for line in code.splitlines()])}
    print("SUCCESS")
except Exception:
    print("FAILURE")
"""
    try:
        # 3 second timeout
        result = subprocess.run(
            [sys.executable, "-c", wrapper], 
            capture_output=True, 
            text=True, 
            timeout=3
        )
        return "SUCCESS" in result.stdout
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False

# --- Main Evaluation Loop ---

def evaluate_model(model_name, is_adapter=False):
    results = {}
    
    try:
        from unsloth import FastLanguageModel
        from datasets import load_dataset
        
        print(f"\nLoading Model: {model_name}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_name,
            max_seq_length = 8192,
            load_in_4bit = True,
        )
        FastLanguageModel.for_inference(model)
        
        # 1. PPL (WikiText2)
        wt2 = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        results["PPL_WikiText"] = calculate_perplexity(model, tokenizer, wt2, n_samples=50)

        # 2. PPL (Coding) - Repurpose Evol-Code test split if available, or just take some
        evol = load_dataset("nickrosh/Evol-Instruct-Code-80k-v1", split="train") # Use first few as proxy
        results["PPL_Code"] = calculate_perplexity(model, tokenizer, evol, n_samples=50)
        
        # 3. Induction (NLL)
        results["Induction_NLL"] = calculate_induction_score(model, tokenizer)
        
        # 4. Passkey
        results["Passkey_Acc"] = retrieve_passkey(model, tokenizer, context_length=4096)
        
        # 5. HumanEval
        results["HumanEval_Pass@1"] = eval_humaneval_subset(model, tokenizer, n_problems=20)
        
        del model, tokenizer
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"Eval failed for {model_name}: {e}")
        
    return results

# Plotting handled above
def plot_results_dummy(base, fine): pass


def main():
    # Install if needed
    try:
        import unsloth
    except ImportError:
        install_dependencies()

    # 1. Evaluate Base
    base_model = "unsloth/Qwen2.5-Coder-14B-Instruct"
    print(f"Evaluating Base Model: {base_model}")
    start_results = evaluate_model(base_model)
    print("Base Results:", json.dumps(start_results, indent=2))
    
    # 2. Evaluate Finetuned (Adapter)
    # Note: Loading adapter requires loading base then patching.
    # But `evaluate_model` loads via `FastLanguageModel.from_pretrained`.
    # If "qwen_coder_lora" is a local folder with adapter, Unsloth usually handles it 
    # if we pass the folder name, it merges on load or loads adapter.
    fine_model = "qwen_coder_lora" 
    print(f"Evaluating Finetuned Model: {fine_model}")
    end_results = evaluate_model(fine_model)
    print("Finetuned Results:", json.dumps(end_results, indent=2))
    
    # 3. Plot
    plot_results(start_results, end_results)
    
    # Save raw data
    with open("eval_results.json", "w") as f:
        json.dump({"base": start_results, "finetuned": end_results}, f, indent=2)

if __name__ == "__main__":
    main()

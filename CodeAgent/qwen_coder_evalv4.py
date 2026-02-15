#!/usr/bin/env python3
"""
coder_eval_v4.py

A comprehensive LLM Coder evaluation script that merges functionalities from
qwen_coder_eval_v3.py and advanced_coder_eval.py.

New Features:
- Evaluates models from Hugging Face (HF), local folders, or OpenAI-compatible APIs.
- Simple model specification via command line.
- Evaluates models sequentially to conserve GPU memory.
- Integrates advanced evaluations: EvalPlus, lm-eval-harness.
- Provides a comprehensive, professional-grade evaluation for comparing different coder models.
- Robust execution: Skips individual evaluations that fail and continues.
- Markdown reporting: Generates a summary report in Markdown format.

Evaluations included:
- Perplexity (PPL) on WikiText2 and MBPP text (for local/HF models).
- Induction NLL (repeat-half task) (for local/HF models).
- Passkey retrieval accuracy (for local/HF models).
- HumanEval pass@1 (from original script and from EvalPlus).
- MBPP pass@1 (from original script and from EvalPlus).
- lm-eval-harness for various coding and general-purpose benchmarks.

SECURITY WARNING:
This script executes model-generated code. Run it in a sandboxed or containerized environment.

Example Usage:

# Evaluate a HF model, a local model, and an API model
python CodeAgent/qwen_coder_evalv4.py \
  --models "unsloth/Qwen2.5-Coder-14B-Instruct" "./qwen_coder_lora" "openai:gpt-4o" \
  --run_all --make_figures \
  --out_dir ./eval_results

"""

import os
import sys
import json
import math
import time
import random
import argparse
import subprocess
import shutil
import re
import tempfile
from typing import Dict, Any, List, Tuple, Optional, Callable
from dataclasses import dataclass

import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

# -----------------------------
# Utilities
# -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def now_ts() -> str:
    return time.strftime("%Y%m%d-%H%M%S")

def is_bf16_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability(0)
    return major >= 8

def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p

def which_or_raise(cmd: str, install_hint: str) -> str:
    path = shutil.which(cmd)
    if not path:
        raise RuntimeError(f"Missing command `{cmd}`.\nInstall hint:\n{install_hint}")
    return path

def run_cmd(
    cmd: List[str],
    cwd: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    tee_to: Optional[str] = None,
    timeout: Optional[int] = None,
) -> subprocess.CompletedProcess:
    """Run subprocess, optionally tee stdout+stderr to file."""
    print(f"\n>> {' '.join(cmd)}")
    timeout = timeout or 3600  # Set a long default timeout
    p = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
    )
    if tee_to:
        ensure_dir(os.path.dirname(tee_to))
        with open(tee_to, "w", encoding="utf-8") as f:
            f.write(p.stdout)
    if p.returncode != 0:
        print(f"[WARN] Command failed (code={p.returncode}): {' '.join(cmd)}")
        print("====== STDOUT/STDERR ======")
        print(p.stdout[-4000:])
        print("===========================")
    return p

# -----------------------------
# Model Specification
# -----------------------------
@dataclass
class ModelSpec:
    name: str
    path: str
    type: str
    api_base: Optional[str] = None
    api_key_env: str = "OPENAI_API_KEY"
    tokenizer_path: str = "unsloth/Qwen2.5-Coder-14B-Instruct"
    is_local_api: bool = False

def parse_model_spec(model_str: str, args: argparse.Namespace) -> ModelSpec:
    """
    Determines model type and creates a ModelSpec.
    - openai:model_name -> API model
    - /path/to/dir -> local model
    - org/model -> HF model
    """
    if model_str.startswith("openai:"):
        model_name = model_str.split(":", 1)[1]
        is_local = bool(args.api_base and ("127.0.0.1" in args.api_base or "localhost" in args.api_base))
        return ModelSpec(
            name=model_name,
            path=model_name,
            type="api",
            api_base=args.api_base,
            api_key_env=args.api_key_env,
            is_local_api=is_local,
        )
    elif os.path.isdir(model_str):
        model_name = os.path.basename(model_str.rstrip('/'))
        return ModelSpec(
            name=model_name,
            path=model_str,
            type="local"
        )
    else: # Assume HF model
        model_name = model_str.split("/")[-1]
        return ModelSpec(
            name=model_name,
            path=model_str,
            type="hf"
        )

# -----------------------------
# Model Loading & Generation
# -----------------------------
def load_model_and_tokenizer(
    model_name_or_path: str,
    max_seq_length: int,
    device: str = "cuda",
    use_4bit: bool = False,
) -> Tuple[torch.nn.Module, Any]:
    """Loads a model and tokenizer, preferring Unsloth."""
    try:
        from unsloth import FastLanguageModel
        dtype = torch.bfloat16 if (device.startswith("cuda") and is_bf16_supported()) else torch.float16
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name_or_path,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=use_4bit,
        )
        FastLanguageModel.for_inference(model)
        model.to(device)
        model.eval()
        return model, tokenizer
    except ImportError:
        print("[WARN] Unsloth not found. Falling back to standard Transformers.")
        from transformers import AutoTokenizer, AutoModelForCausalLM
        dtype = torch.bfloat16 if (device.startswith("cuda") and is_bf16_supported()) else torch.float16
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
            device_map="auto" if device.startswith("cuda") else None,
            trust_remote_code=True,
        )
        model.eval()
        return model, tokenizer

def create_hf_generator(model, tokenizer, device, max_new_tokens, do_sample=False) -> Callable[[str], str]:
    @torch.no_grad()
    def generator(prompt_text: str) -> str:
        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
        gen = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=0.0 if not do_sample else 0.2,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        return tokenizer.decode(gen[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return generator

def create_api_generator(model_spec: ModelSpec, max_new_tokens: int) -> Callable[[str], str]:
    import openai
    api_key = os.getenv(model_spec.api_key_env)
    if not api_key:
        if model_spec.is_local_api:
            api_key = "none"  # Dummy key for local servers like llama.cpp
        else:
            raise ValueError(f"API key not found. Please set the {model_spec.api_key_env} environment variable.")

    client = openai.OpenAI(api_key=api_key, base_url=model_spec.api_base)

    def generator(prompt_text: str) -> str:
        if prompt_text.endswith("<|im_start|>assistant\n"):
             prompt_text = prompt_text.removesuffix("<|im_start|>assistant\n")

        response = client.chat.completions.create(
            model=model_spec.path,
            messages=[{"role": "user", "content": prompt_text}],
            max_tokens=max_new_tokens,
            temperature=0.0,
            n=1,
            stop=None,
        )
        return response.choices[0].message.content or ""
    return generator

# -----------------------------
# Performance Benchmarking
# -----------------------------
def get_gpu_memory_usage(device: str = "cuda") -> float:
    """Returns peak GPU memory usage in GB."""
    if not torch.cuda.is_available():
        return 0.0
    # Gets the peak memory in bytes and converts to GB.
    peak_mem_bytes = torch.cuda.max_memory_allocated(device)
    return round(peak_mem_bytes / (1024**3), 2)

@torch.no_grad()
def benchmark_generation(model, tokenizer, args) -> Dict[str, float]:
    """Measures token generation speed and memory usage for a model."""
    if not args.device.startswith("cuda"):
        print("[INFO] Skipping generation benchmark on non-GPU device.")
        return {}

    torch.cuda.reset_peak_memory_stats(args.device)
    torch.cuda.empty_cache()

    prompt = "def fib(n):"  # Simple prompt to start generation
    inputs = tokenizer(prompt, return_tensors="pt").to(args.device)

    # Warmup run
    model.generate(**inputs, max_new_tokens=16, use_cache=True, pad_token_id=tokenizer.eos_token_id)
    torch.cuda.synchronize()

    start_time = time.time()
    generated_ids = model.generate(**inputs, max_new_tokens=args.benchmark_max_new_tokens, use_cache=True, pad_token_id=tokenizer.eos_token_id)
    torch.cuda.synchronize()
    end_time = time.time()

    num_new_tokens = generated_ids.shape[1] - inputs["input_ids"].shape[1]
    duration = end_time - start_time
    
    tokens_per_sec = (num_new_tokens / duration) if duration > 0 else 0
    peak_memory_gb = get_gpu_memory_usage(args.device)

    return {
        "tokens_per_sec": round(tokens_per_sec, 2),
        "peak_memory_gb": peak_memory_gb,
    }

# -----------------------------
# Evals
# -----------------------------
@torch.no_grad()
def perplexity_sliding_window(model, tokenizer, dataset_name: str, args: argparse.Namespace) -> float:
    from datasets import load_dataset
    split_map = {"wikitext": "test", "mbpp": "test"}
    text_key_map = {"wikitext": "text", "mbpp": "prompt"}
    dataset = load_dataset(dataset_name, "wikitext-2-raw-v1" if dataset_name == "wikitext" else "sanitized", split=split_map[dataset_name])
    texts = [d[text_key_map[dataset_name]] for d in dataset if d[text_key_map[dataset_name]]]
    
    model.eval()
    total_nll, total_tokens = 0.0, 0
    for text in tqdm(texts, desc=f"PPL ({dataset_name})"):
        enc = tokenizer(text, return_tensors="pt", add_special_tokens=True)
        input_ids = enc["input_ids"][0].to(args.device)
        if input_ids.numel() < 2: continue
        for start in range(0, input_ids.numel(), args.ppl_stride):
            end = min(start + args.ppl_max_len, input_ids.numel())
            window = input_ids[start:end]
            labels = window.clone()
            if start > 0:
                overlap = max(0, min(args.ppl_max_len - args.ppl_stride, labels.numel()))
                labels[:overlap] = -100
            out = model(window.unsqueeze(0), labels=labels.unsqueeze(0))
            contrib = (labels != -100).sum().item()
            if contrib > 0:
                total_nll += out.loss.item() * contrib
                total_tokens += contrib
            if end == input_ids.numel(): break
    return float(math.exp(total_nll / total_tokens)) if total_tokens > 0 else float("inf")

@torch.no_grad()
def induction_repeat_nll(model, tokenizer, args: argparse.Namespace) -> float:
    model.eval()
    nlls = []
    vocab_size = getattr(model.config, "vocab_size", 50257)
    hi, lo = min(int(vocab_size), 50000), 1000
    for _ in tqdm(range(args.induction_samples), desc="Induction"):
        A = torch.randint(low=lo, high=hi, size=(1, args.induction_seq_len), device=args.device)
        inp = torch.cat([A, A], dim=1)
        labels = inp.clone()
        labels[:, :args.induction_seq_len] = -100
        out = model(inp, labels=labels)
        nlls.append(float(out.loss.item()))
    return float(np.mean(nlls)) if nlls else float("inf")

@torch.no_grad()
def passkey_retrieval_acc(model, tokenizer, args: argparse.Namespace) -> float:
    model.eval()
    hits = 0
    filler_ids = tokenizer.encode("The sun sets in the west. ", add_special_tokens=False)

    for _ in tqdm(range(args.passkey_trials), desc="Passkey"):
        passkey = random.randint(10000, 99999)
        needle = f"\nThe secret passkey is {passkey}.\n"
        budget = args.passkey_ctx - (len(tokenizer.encode(needle)) + 80)
        filler = (filler_ids * (budget // len(filler_ids) + 1))[:budget]
        
        insert_at = int(len(filler) * args.passkey_depth)
        ctx_text = tokenizer.decode(filler[:insert_at] + tokenizer.encode(needle) + filler[insert_at:])
        question = "\nWhat is the secret passkey? Answer with ONLY the number."
        
        messages = [{"role": "user", "content": ctx_text + question}]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt_text, return_tensors="pt").to(args.device)
        
        gen = model.generate(**inputs, max_new_tokens=64, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        out_txt = tokenizer.decode(gen[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        
        if str(passkey) in "".join(filter(str.isdigit, out_txt)):
            hits += 1
            
    return float(hits / args.passkey_trials) if args.passkey_trials > 0 else 0.0

def extract_code_from_generation(gen_text: str) -> str:
    if "```python" in gen_text:
        return gen_text.split("```python", 1)[1].split("```", 1)[0].strip()
    if "```" in gen_text:
        return gen_text.split("```", 1)[1].split("```", 1)[0].strip()
    return gen_text.strip()

def run_code_subprocess(code: str, timeout_s: float = 10.0) -> bool:
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "prog.py")
        with open(path, "w", encoding="utf-8") as f: f.write(code)
        try:
            res = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=timeout_s)
            return res.returncode == 0
        except (subprocess.TimeoutExpired, Exception):
            return False

def humaneval_pass_at_1(generator_fn: Callable, tokenizer: Any, n: int) -> float:
    from datasets import load_dataset
    ds = load_dataset("openai_humaneval", split="test")
    n = min(n, len(ds))
    ds = ds.select(range(n))
    passed = 0
    for ex in tqdm(ds, desc="HumanEval"):
        messages = [{"role": "user", "content": f"Complete the following Python function.\nReturn only code.\n\n{ex['prompt']}"}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        gen_text = generator_fn(text)
        code_pred = extract_code_from_generation(gen_text)
        full_code = f"import math\nfrom typing import *\n\n{ex['prompt']}{code_pred}\n\n{ex['test']}\n\ncheck({ex['entry_point']})\n"
        if run_code_subprocess(full_code): passed += 1
    return (100.0 * passed / n) if n > 0 else 0.0

def mbpp_pass_at_1(generator_fn: Callable, tokenizer: Any, n: int) -> float:
    from datasets import load_dataset
    ds = load_dataset("mbpp", "sanitized", split="test")
    n = min(n, len(ds))
    ds = ds.select(range(n))
    passed = 0
    for ex in tqdm(ds, desc="MBPP"):
        user_prompt = f"Write a Python function to solve the following problem.\nReturn only code.\n\n{ex['prompt']}"
        messages = [{"role": "user", "content": user_prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        gen_text = generator_fn(text)
        code_pred = extract_code_from_generation(gen_text)
        full_code = f"import math\nfrom typing import *\n\n{code_pred}\n\n" + "\n".join(ex["test_list"])
        if run_code_subprocess(full_code): passed += 1
    return (100.0 * passed / n) if n > 0 else 0.0

def run_evalplus(spec: ModelSpec, dataset: str, parallel: int, out_dir: str) -> Dict:
    log_path = os.path.join(out_dir, f"evalplus_{spec.name.replace('/', '_')}_{dataset}.log")
    cmd = ["evalplus.evaluate", "--dataset", dataset, "--greedy"]
    if spec.type == "api":
        cmd.extend(["--model", spec.path, "--backend", "oai"])
        if spec.api_base: cmd.extend(["--base-url", spec.api_base])
    else:
        cmd.extend(["--model", spec.path, "--backend", "hf"])
    if parallel > 0: cmd.extend(["--parallel", str(parallel)])
    
    env = os.environ.copy()
    if spec.is_local_api and "OPENAI_API_KEY" not in env:
        env["OPENAI_API_KEY"] = "none"
        
    p = run_cmd(cmd, tee_to=log_path, env=env)
    if p.returncode != 0: return {"pass@1": "failed", "log": log_path}
    
    m = re.search(r"pass@1\s*:\s*([0-9.]+)", p.stdout)
    return {"pass@1": float(m.group(1)) * 100.0 if m else 0.0, "log": log_path}

def run_lm_eval(spec: ModelSpec, tasks: str, bs: int, dev: str, out_dir: str) -> Dict:
    out_json_path = os.path.join(out_dir, f"lm_eval_{spec.name.replace('/', '_')}.json")
    log_path = out_json_path.replace(".json", ".log")
    
    cmd = ["lm_eval", "--tasks", tasks, "--output_path", out_json_path, "--confirm_run_unsafe_code"]
    
    if spec.type == "api":
        if spec.is_local_api:
            # Use local-completions for local API endpoints, as they are often not chat-tuned
            # and might be served by tools like llama.cpp or vLLM at a completion endpoint.
            model_args_parts = [f"model={spec.path}", "num_concurrent=16"]
            if spec.api_base:
                # lm-eval's local-completions wants the full completions URL. This handles
                # base URLs with or without a trailing /v1.
                api_base = spec.api_base.strip('/')
                if api_base.endswith('/v1'):
                    completions_url = api_base + "/completions"
                else:
                    completions_url = api_base + "/v1/completions"
                model_args_parts.append(f"base_url={completions_url}")
            
            cmd.extend([
                "--model", "local-completions",
                "--model_args", ",".join(model_args_parts),
                "--batch_size", str(bs)
            ])
        else:
            model_args_parts = [f"model={spec.path}"]
            if spec.api_base:
                model_args_parts.append(f"base_url={spec.api_base}")
            cmd.extend(["--model", "openai", "--model_args", ",".join(model_args_parts)])
    else:
        cmd.extend(["--model", "hf", "--model_args", f"pretrained={spec.path}", "--device", dev, "--batch_size", str(bs)])
    
    env = os.environ.copy()
    env["LM_EVAL_ALLOW_CODE_EXECUTION"] = "1"
    env["HF_ALLOW_CODE_EVAL"] = "1"
    if spec.is_local_api and "OPENAI_API_KEY" not in env:
        env["OPENAI_API_KEY"] = "none"

    p = run_cmd(cmd, tee_to=log_path, env=env)
    
    if p.returncode != 0:
        print(f"[WARN] lm-eval command failed. Trying to parse from stdout. See log: {log_path}")
    
    # Try parsing from file first
    if os.path.exists(out_json_path):
        with open(out_json_path) as f:
            try:
                return json.load(f).get("results", {})
            except json.JSONDecodeError:
                print(f"[WARN] Could not parse lm-eval JSON from file {out_json_path}.")

    # Fallback to stdout
    try:
        json_start = p.stdout.find('{')
        if json_start != -1:
            json_str = p.stdout[json_start:]
            # The stdout might contain more than just the JSON, so we need to find the end of it.
            brace_count = 0
            end_pos = -1
            for i, char in enumerate(json_str):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                if brace_count == 0:
                    end_pos = i + 1
                    break
            if end_pos != -1:
                json_str = json_str[:end_pos]

            results = json.loads(json_str)
            with open(out_json_path, "w") as f:
                json.dump(results, f, indent=2)
            return results.get("results", {})
    except (json.JSONDecodeError, IndexError):
        print(f"[WARN] Could not parse lm-eval JSON output from stdout for {spec.name}.")

    return {}

# -----------------------------
# Reporting
# -----------------------------
def plot_results(all_results: Dict[str, Dict], out_dir: str):
    if not all_results: return
    model_names = list(all_results.keys())
    metric_keys = sorted({k for res in all_results.values() for k, v in res.items() if isinstance(v, (int, float))})
    if not metric_keys: return

    fig, axes = plt.subplots(1, len(metric_keys), figsize=(4 * len(metric_keys), 6), squeeze=False)
    for i, key in enumerate(metric_keys):
        ax = axes[0, i]
        values = [all_results[name].get(key, 0) for name in model_names]
        ax.bar(model_names, values)
        ax.set_title(key.replace("_", " "), fontsize=10)
        ax.set_xticklabels(model_names, rotation=60, ha="right")
        for j, v in enumerate(values): ax.text(j, v, f"{v:.2f}", ha="center", va="bottom")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "comparison_summary.png"), dpi=200)
    print(f"\n[OK] Saved summary plot to: {os.path.join(out_dir, 'comparison_summary.png')}")

def generate_markdown_report(all_results: Dict[str, Dict], out_dir: str):
    if not all_results: return
    model_names, metric_keys = list(all_results.keys()), sorted({k for res in all_results.values() for k in res})
    
    md = ["# Evaluation Report\n"]
    md.append("| Model | " + " | ".join(metric_keys) + " |")
    md.append("|:---| " + " | ".join([":---:"] * len(metric_keys)) + " |")
    for name in model_names:
        row = [f"| {name} "]
        for key in metric_keys:
            val = all_results[name].get(key, "N/A")
            row.append(f" {val:.2f} " if isinstance(val, float) else f" {val} ")
        md.append(" | ".join(row) + " |")
    
    with open(os.path.join(out_dir, "report.md"), "w") as f: f.write("\n".join(md))
    print(f"[OK] Saved Markdown report to: {os.path.join(out_dir, 'report.md')}")

# -----------------------------
# Main Loop
# -----------------------------
def evaluate_model(spec: ModelSpec, args: argparse.Namespace) -> Dict[str, Any]:
    print(f"\n{'='*20} Evaluating: {spec.name} ({spec.path}) {'='*20}")

    original_no_proxy = os.environ.get('NO_PROXY')
    original_lower_no_proxy = os.environ.get('no_proxy')
    if spec.is_local_api:
        # Append localhost to NO_PROXY to bypass proxy for local server,
        # without breaking connections to the outside world (e.g., huggingface.co)
        no_proxy_parts = original_no_proxy.split(',') if original_no_proxy else []
        if '127.0.0.1' not in no_proxy_parts:
            no_proxy_parts.append('127.0.0.1')
        if 'localhost' not in no_proxy_parts:
            no_proxy_parts.append('localhost')
        new_no_proxy = ','.join(no_proxy_parts)
        os.environ['NO_PROXY'] = new_no_proxy
        os.environ['no_proxy'] = new_no_proxy
        print(f"[INFO] Temporarily setting NO_PROXY/no_proxy='{new_no_proxy}' for local API evaluation.")

    try:
        results: Dict[str, Any] = {}

        try:
            from transformers import AutoTokenizer
            prompt_tokenizer = AutoTokenizer.from_pretrained(spec.tokenizer_path, trust_remote_code=True)
        except Exception as e:
            print(f"[ERROR] Failed to load tokenizer for {spec.name}: {e}"); return {"error": "Tokenizer loading failed"}

        gen_fn: Optional[Callable] = None
        model: Optional[torch.nn.Module] = None

        if spec.type in ["hf", "local"]:
            try:
                model, tok = load_model_and_tokenizer(spec.path, args.max_seq_length, args.device, args.use_4bit_for_eval)
                prompt_tokenizer = tok
                gen_fn = create_hf_generator(model, tok, args.device, args.humaneval_max_new_tokens)
            except Exception as e:
                print(f"[ERROR] Failed to load model {spec.name}: {e}")
                results["model_loading"] = "failed"
        elif spec.type == "api":
            try:
                gen_fn = create_api_generator(spec, args.humaneval_max_new_tokens)
            except Exception as e:
                print(f"[ERROR] Failed to create API generator for {spec.name}: {e}")
                results["generator_creation"] = "failed"

        if model:
            if args.run_ppl:
                try:
                    results["PPL_WikiText"] = perplexity_sliding_window(model, prompt_tokenizer, "wikitext", args)
                    results["PPL_MBPP"] = perplexity_sliding_window(model, prompt_tokenizer, "mbpp", args)
                except Exception as e:
                    print(f"[WARN] PPL evaluation failed for {spec.name}: {e}")
                    results["PPL"] = "failed"
            if args.run_induction:
                try:
                    results["Induction_NLL"] = induction_repeat_nll(model, prompt_tokenizer, args)
                except Exception as e:
                    print(f"[WARN] Induction evaluation failed for {spec.name}: {e}")
                    results["Induction_NLL"] = "failed"
            if args.run_passkey:
                try:
                    results["Passkey_Acc"] = passkey_retrieval_acc(model, prompt_tokenizer, args)
                except Exception as e:
                    print(f"[WARN] Passkey evaluation failed for {spec.name}: {e}")
                    results["Passkey_Acc"] = "failed"
            del model
            if torch.cuda.is_available(): torch.cuda.empty_cache()

        if not gen_fn:
            print(f"[WARN] No generator for {spec.name}, skipping code-gen evals.")
            return results

        if args.run_humaneval:
            try:
                results["HumanEval_pass@1"] = humaneval_pass_at_1(gen_fn, prompt_tokenizer, args.humaneval_n)
            except Exception as e:
                print(f"[WARN] HumanEval pass@1 failed for {spec.name}: {e}")
                results["HumanEval_pass@1"] = "failed"
        if args.run_mbpp:
            try:
                results["MBPP_pass@1"] = mbpp_pass_at_1(gen_fn, prompt_tokenizer, args.mbpp_n)
            except Exception as e:
                print(f"[WARN] MBPP pass@1 failed for {spec.name}: {e}")
                results["MBPP_pass@1"] = "failed"

        raw_out_dir = ensure_dir(os.path.join(args.out_dir, "raw_outputs"))
        if args.run_evalplus:
            try:
                results["EvalPlus_HumanEval"] = run_evalplus(spec, "humaneval", args.evalplus_parallel, raw_out_dir).get("pass@1")
                results["EvalPlus_MBPP"] = run_evalplus(spec, "mbpp", args.evalplus_parallel, raw_out_dir).get("pass@1")
            except Exception as e:
                print(f"[WARN] EvalPlus failed for {spec.name}: {e}")
                results["EvalPlus"] = "failed"
                
        if args.run_lm_eval:
            try:
                lm_res = run_lm_eval(spec, args.lm_eval_tasks, args.lm_eval_batch_size, args.device, raw_out_dir)
                for task, res in lm_res.items():
                    metric = next((v for k, v in res.items() if "acc" in k or "pass" in k), None)
                    if metric is not None:
                        results[f"lm-eval_{task}"] = round(metric * 100, 2)
            except Exception as e:
                print(f"[WARN] lm-eval failed for {spec.name}: {e}")
                results["lm-eval"] = "failed"

        return results
    finally:
        if spec.is_local_api:
            if original_no_proxy is None:
                if 'NO_PROXY' in os.environ:
                    del os.environ['NO_PROXY']
            else:
                os.environ['NO_PROXY'] = original_no_proxy
            
            if original_lower_no_proxy is None:
                if 'no_proxy' in os.environ:
                    del os.environ['no_proxy']
            else:
                os.environ['no_proxy'] = original_lower_no_proxy
            print("[INFO] Restored original NO_PROXY/no_proxy setting.")

def main():
    parser = argparse.ArgumentParser(description="Comprehensive Coder Model Evaluation Script", formatter_class=argparse.RawTextHelpFormatter)
    
    # Core
    parser.add_argument("--models", nargs='+', default=["unsloth/Qwen2.5-Coder-14B-Instruct", "qwen_coder_lora_sft_v3_2"], help="Models to evaluate. Can be HF path, local folder, or 'openai:model_name'.")
    parser.add_argument("--out_dir", type=str, default=f"eval_results_{now_ts()}", help="Output directory for results, logs, and plots.")
    parser.add_argument("--device", type=str, default="cuda", help="Device for local model inference (e.g., 'cuda', 'cpu').")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--max_seq_length", type=int, default=4096, help="Max sequence length for model loading.")
    parser.add_argument("--use_4bit_for_eval", action="store_true", help="Use 4-bit quantization for local models.")
    
    # API specific
    parser.add_argument("--api_base", type=str, default=None, help="Base URL for OpenAI-compatible API.")
    parser.add_argument("--api_key_env", type=str, default="OPENAI_API_KEY", help="Environment variable for API key.")

    # Evaluation selectors
    eval_group = parser.add_argument_group('Evaluation Selectors')
    eval_group.add_argument("--run_all", action="store_true", help="Run all available evaluations.")
    eval_group.add_argument("--run_benchmark", action="store_true", help="Run throughput and memory benchmark.")
    eval_group.add_argument("--run_ppl", action="store_true", help="Run Perplexity evaluations.")
    eval_group.add_argument("--run_induction", action="store_true", help="Run Induction Head evaluation.")
    eval_group.add_argument("--run_passkey", action="store_true", help="Run Passkey Retrieval evaluation.")
    eval_group.add_argument("--run_humaneval", action="store_true", help="Run HumanEval pass@1.")
    eval_group.add_argument("--run_mbpp", action="store_true", help="Run MBPP pass@1.")
    eval_group.add_argument("--run_evalplus", action="store_true", help="Run EvalPlus (HumanEval+, MBPP+).")
    eval_group.add_argument("--run_lm_eval", action="store_true", help="Run lm-evaluation-harness.")
    
    # Reporting
    parser.add_argument("--make_figures", action="store_true", help="Generate plots from results.")

    # Evaluation parameters
    param_group = parser.add_argument_group('Evaluation Parameters')
    param_group.add_argument("--humaneval_n", type=int, default=164, help="Number of problems for HumanEval.")
    param_group.add_argument("--humaneval_max_new_tokens", type=int, default=512, help="Max new tokens for HumanEval generation.")
    param_group.add_argument("--mbpp_n", type=int, default=399, help="Number of problems for MBPP.")
    param_group.add_argument("--evalplus_parallel", type=int, default=8, help="Number of parallel workers for EvalPlus.")
    param_group.add_argument("--lm_eval_tasks", type=str, default=None, help="Comma-separated tasks for lm-eval-harness. Defaults to coding tasks.")
    param_group.add_argument("--lm_eval_backend", type=str, default="hf", choices=["hf", "vllm"], help="Backend for lm-eval-harness for local models.")
    param_group.add_argument("--lm_eval_batch_size", type=int, default=8)
    param_group.add_argument("--lm_eval_timeout", type=int, default=7200, help="Timeout in seconds for lm-eval command.")
    param_group.add_argument("--benchmark_max_new_tokens", type=int, default=256)
    param_group.add_argument("--ppl_max_len", type=int, default=2048)
    param_group.add_argument("--ppl_stride", type=int, default=512)
    param_group.add_argument("--induction_seq_len", type=int, default=256)
    param_group.add_argument("--induction_samples", type=int, default=20)
    param_group.add_argument("--passkey_ctx", type=int, default=4096)
    param_group.add_argument("--passkey_depth", type=float, default=0.5)
    param_group.add_argument("--passkey_trials", type=int, default=10)


    args = parser.parse_args()

    if args.run_all:
        args.run_benchmark = args.run_ppl = args.run_induction = args.run_passkey = args.run_humaneval = args.run_mbpp = args.run_evalplus = args.run_lm_eval = True

    if args.run_lm_eval and not args.lm_eval_tasks:
        args.lm_eval_tasks = "humaneval"

    set_seed(args.seed)
    ensure_dir(args.out_dir)

    model_specs = [parse_model_spec(m, args) for m in args.models]
        
    all_results = {}
    for spec in model_specs:
        try:
            model_results = evaluate_model(spec, args)
            all_results[spec.name] = {k: v for k, v in model_results.items() if v is not None}
            print(f"\n--- Results for {spec.name} ---")
            print(json.dumps(all_results[spec.name], indent=2))
        except Exception as e:
            print(f"\n[CATASTROPHIC ERROR] Evaluation failed for {spec.name}: {e}")
            all_results[spec.name] = {"status": "catastrophic failure"}

    results_path = os.path.join(args.out_dir, "summary.json")
    with open(results_path, "w") as f: json.dump({"meta": {k:v for k,v in vars(args).items() if not k.startswith('_')}, "results": all_results}, f, indent=2)
    print(f"\n[OK] All evaluations complete. Summary saved to: {results_path}")

    if args.make_figures: plot_results(all_results, args.out_dir)
    generate_markdown_report(all_results, args.out_dir)

if __name__ == "__main__":
    main()
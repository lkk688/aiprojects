#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
coder_eval_v5.py

Upgraded comprehensive coder evaluation script.

Major improvements over v4:
- Unified result schema with per-task metadata
- Resume support and per-model JSON outputs
- Safer code execution in isolated subprocesses
- Adds compile_rate / exec_rate / avg_gen_chars for code-gen tasks
- More robust API generation with retries
- Better lm-eval / EvalPlus logging and parsing
- CSV + Markdown + JSON summary outputs
- Local model throughput + GPU peak memory benchmark
- Cleaner prompt handling and code extraction
- Better failure isolation

Supported model specs:
- HF model:        "Qwen/Qwen2.5-Coder-14B-Instruct"
- Local folder:    "./my_lora_or_model"
- API model:       "openai:gpt-4o" with optional --api_base
"""

import os
import sys
import csv
import io
import gc
import json
import math
import time
import shutil
import signal
import random
import argparse
import subprocess
import tempfile
import traceback
import re
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Tuple, Optional, Callable

import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt


# ============================================================
# Utilities
# ============================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def now_ts() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def is_bf16_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability(0)
    return major >= 8


def sanitize_name(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", s)


def json_dump(obj: Any, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def json_load(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def tail_text(s: str, n: int = 4000) -> str:
    return s[-n:] if isinstance(s, str) else ""


def run_cmd(
    cmd: List[str],
    cwd: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    tee_to: Optional[str] = None,
    timeout: Optional[int] = None,
) -> subprocess.CompletedProcess:
    print(f"\n>> {' '.join(cmd)}")
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
        print("====== STDOUT/STDERR (tail) ======")
        print(tail_text(p.stdout))
        print("==================================")
    return p


# ============================================================
# Model spec
# ============================================================
@dataclass
class ModelSpec:
    raw: str
    name: str
    path: str
    type: str                     # hf | local | api
    api_base: Optional[str] = None
    api_key_env: str = "OPENAI_API_KEY"
    tokenizer_path: Optional[str] = None
    is_local_api: bool = False


def parse_model_spec(model_str: str, args: argparse.Namespace) -> ModelSpec:
    if model_str.startswith("openai:"):
        model_name = model_str.split(":", 1)[1]
        is_local = bool(args.api_base and ("127.0.0.1" in args.api_base or "localhost" in args.api_base))
        return ModelSpec(
            raw=model_str,
            name=model_name,
            path=model_name,
            type="api",
            api_base=args.api_base,
            api_key_env=args.api_key_env,
            tokenizer_path=args.prompt_tokenizer,
            is_local_api=is_local,
        )
    elif os.path.isdir(model_str):
        model_name = os.path.basename(model_str.rstrip("/"))
        return ModelSpec(
            raw=model_str,
            name=model_name,
            path=model_str,
            type="local",
            tokenizer_path=args.prompt_tokenizer or model_str,
        )
    else:
        model_name = model_str.split("/")[-1]
        return ModelSpec(
            raw=model_str,
            name=model_name,
            path=model_str,
            type="hf",
            tokenizer_path=args.prompt_tokenizer or model_str,
        )


# ============================================================
# Loading / generation
# ============================================================
def load_prompt_tokenizer(tokenizer_path: str):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    return tok


def load_model_and_tokenizer(
    model_name_or_path: str,
    max_seq_length: int,
    device: str = "cuda",
    use_4bit: bool = False,
):
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
        return model, tokenizer, "unsloth"
    except Exception:
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
        return model, tokenizer, "transformers"


def cleanup_local_model(model=None, tokenizer=None):
    try:
        del model
    except Exception:
        pass
    try:
        del tokenizer
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def create_hf_generator(model, tokenizer, device: str, max_new_tokens: int, do_sample: bool = False):
    @torch.no_grad()
    def generator(prompt_text: str) -> Dict[str, Any]:
        t0 = time.time()
        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
        gen = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=0.0 if not do_sample else 0.2,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        out_ids = gen[0, inputs["input_ids"].shape[1]:]
        out_text = tokenizer.decode(out_ids, skip_special_tokens=True)
        dt = time.time() - t0
        return {
            "text": out_text,
            "prompt_tokens": int(inputs["input_ids"].shape[1]),
            "completion_tokens": int(out_ids.shape[0]),
            "elapsed_seconds": round(dt, 4),
        }
    return generator


def create_api_generator(model_spec: ModelSpec, max_new_tokens: int, temperature: float = 0.0):
    import openai

    api_key = os.getenv(model_spec.api_key_env)
    if not api_key:
        if model_spec.is_local_api:
            api_key = "none"
        else:
            raise ValueError(f"Missing API key in env: {model_spec.api_key_env}")

    client = openai.OpenAI(api_key=api_key, base_url=model_spec.api_base)

    def generator(prompt_text: str) -> Dict[str, Any]:
        # Retry because local OpenAI-compatible servers can occasionally fail under load
        last_err = None
        for attempt in range(3):
            try:
                t0 = time.time()
                response = client.chat.completions.create(
                    model=model_spec.path,
                    messages=[{"role": "user", "content": prompt_text}],
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    n=1,
                )
                dt = time.time() - t0
                text = response.choices[0].message.content or ""
                usage = getattr(response, "usage", None)
                return {
                    "text": text,
                    "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                    "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
                    "elapsed_seconds": round(dt, 4),
                }
            except Exception as e:
                last_err = e
                time.sleep(1.0 * (attempt + 1))
        raise RuntimeError(f"API generation failed after retries: {last_err}")

    return generator


# ============================================================
# Prompt helpers
# ============================================================
def apply_chat_prompt(tokenizer, user_text: str) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": user_text}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass
    return user_text


def extract_code_from_generation(gen_text: str) -> str:
    if not isinstance(gen_text, str):
        return ""

    # Prefer python fenced block
    m = re.search(r"```python\s*(.*?)```", gen_text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # Then any fenced block
    m = re.search(r"```(.*?)```", gen_text, flags=re.DOTALL)
    if m:
        return m.group(1).strip()

    # Strip obvious prose before first def/class/import if present
    lines = gen_text.strip().splitlines()
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith(("def ", "class ", "import ", "from ", "@")):
            return "\n".join(lines[i:]).strip()

    return gen_text.strip()


# ============================================================
# Safer code execution
# ============================================================
def _limit_resources_posix(cpu_seconds: int, memory_mb: int):
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        mem_bytes = memory_mb * 1024 * 1024
        # Address space limit
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
        # No core dump
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        # Limit files
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    except Exception:
        pass


def run_code_subprocess(
    code: str,
    timeout_s: float = 8.0,
    cpu_seconds: int = 8,
    memory_mb: int = 1024,
) -> Dict[str, Any]:
    """
    Returns:
      {
        "compiled": bool,
        "passed": bool,
        "returncode": int,
        "stdout": str,
        "stderr": str
      }
    """
    try:
        compile(code, "<generated>", "exec")
        compiled = True
    except Exception as e:
        return {
            "compiled": False,
            "passed": False,
            "returncode": -999,
            "stdout": "",
            "stderr": f"compile_error: {e}",
        }

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "prog.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)

        preexec_fn = None
        if os.name == "posix":
            preexec_fn = lambda: _limit_resources_posix(cpu_seconds=cpu_seconds, memory_mb=memory_mb)

        try:
            res = subprocess.run(
                [sys.executable, "-I", path],
                capture_output=True,
                text=True,
                timeout=timeout_s,
                cwd=td,
                preexec_fn=preexec_fn,
            )
            return {
                "compiled": True,
                "passed": (res.returncode == 0),
                "returncode": int(res.returncode),
                "stdout": res.stdout,
                "stderr": res.stderr,
            }
        except subprocess.TimeoutExpired:
            return {
                "compiled": True,
                "passed": False,
                "returncode": -998,
                "stdout": "",
                "stderr": "timeout",
            }
        except Exception as e:
            return {
                "compiled": True,
                "passed": False,
                "returncode": -997,
                "stdout": "",
                "stderr": f"runtime_error: {e}",
            }


# ============================================================
# Benchmarks
# ============================================================
def get_gpu_peak_mem_gb(device: str = "cuda") -> float:
    if not torch.cuda.is_available():
        return 0.0
    return round(torch.cuda.max_memory_allocated(device) / (1024 ** 3), 3)


@torch.no_grad()
def benchmark_generation(model, tokenizer, args) -> Dict[str, Any]:
    if not args.device.startswith("cuda"):
        return {"status": "skipped_non_gpu"}

    torch.cuda.reset_peak_memory_stats(args.device)
    torch.cuda.empty_cache()

    prompt = "def fib(n):"
    inputs = tokenizer(prompt, return_tensors="pt").to(args.device)

    # warmup
    _ = model.generate(
        **inputs,
        max_new_tokens=16,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    torch.cuda.synchronize()

    t0 = time.time()
    out = model.generate(
        **inputs,
        max_new_tokens=args.benchmark_max_new_tokens,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    torch.cuda.synchronize()
    dt = time.time() - t0

    new_tokens = int(out.shape[1] - inputs["input_ids"].shape[1])
    tok_per_sec = (new_tokens / dt) if dt > 0 else 0.0

    return {
        "status": "ok",
        "prompt_chars": len(prompt),
        "new_tokens": new_tokens,
        "elapsed_seconds": round(dt, 4),
        "tokens_per_sec": round(tok_per_sec, 3),
        "peak_memory_gb": get_gpu_peak_mem_gb(args.device),
    }


# ============================================================
# Local-model-only evals
# ============================================================
@torch.no_grad()
def perplexity_sliding_window(model, tokenizer, dataset_name: str, args: argparse.Namespace) -> Dict[str, Any]:
    from datasets import load_dataset

    if dataset_name == "wikitext":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        texts = [x["text"] for x in ds if x["text"] and x["text"].strip()]
    elif dataset_name == "mbpp":
        ds = load_dataset("mbpp", "sanitized", split="test")
        # Prefer the actual task description field if present
        key = "prompt" if "prompt" in ds.column_names else "text"
        texts = [x[key] for x in ds if x[key] and str(x[key]).strip()]
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")

    total_nll = 0.0
    total_tokens = 0
    per_doc = 0

    model.eval()
    for text in tqdm(texts, desc=f"PPL ({dataset_name})"):
        enc = tokenizer(text, return_tensors="pt", add_special_tokens=True)
        input_ids = enc["input_ids"][0].to(args.device)
        if input_ids.numel() < 2:
            continue

        for start in range(0, input_ids.numel(), args.ppl_stride):
            end = min(start + args.ppl_max_len, input_ids.numel())
            window = input_ids[start:end]
            labels = window.clone()

            if start > 0:
                overlap = max(0, min(args.ppl_max_len - args.ppl_stride, labels.numel()))
                labels[:overlap] = -100

            out = model(window.unsqueeze(0), labels=labels.unsqueeze(0))
            contrib = int((labels != -100).sum().item())
            if contrib > 0:
                total_nll += float(out.loss.item()) * contrib
                total_tokens += contrib
            if end == input_ids.numel():
                break

        per_doc += 1

    ppl = float(math.exp(total_nll / total_tokens)) if total_tokens > 0 else float("inf")
    return {
        "status": "ok",
        "dataset": dataset_name,
        "docs": per_doc,
        "tokens": total_tokens,
        "ppl": ppl,
    }


@torch.no_grad()
def induction_repeat_nll(model, tokenizer, args: argparse.Namespace) -> Dict[str, Any]:
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

    return {
        "status": "ok",
        "samples": len(nlls),
        "mean_nll": float(np.mean(nlls)) if nlls else float("inf"),
        "std_nll": float(np.std(nlls)) if nlls else float("inf"),
    }


@torch.no_grad()
def passkey_retrieval_acc(model, tokenizer, args: argparse.Namespace) -> Dict[str, Any]:
    model.eval()
    hits = 0

    filler_ids = tokenizer.encode("The sun sets in the west. ", add_special_tokens=False)

    for _ in tqdm(range(args.passkey_trials), desc="Passkey"):
        passkey = random.randint(10000, 99999)
        needle = f"\nThe secret passkey is {passkey}.\n"

        budget = args.passkey_ctx - (len(tokenizer.encode(needle)) + 80)
        budget = max(budget, 0)
        filler = (filler_ids * (budget // max(1, len(filler_ids)) + 1))[:budget]

        insert_at = int(len(filler) * args.passkey_depth)
        ctx_ids = filler[:insert_at] + tokenizer.encode(needle, add_special_tokens=False) + filler[insert_at:]
        ctx_text = tokenizer.decode(ctx_ids)
        question = "\nWhat is the secret passkey? Answer with ONLY the number."

        prompt_text = apply_chat_prompt(tokenizer, ctx_text + question)
        inputs = tokenizer(prompt_text, return_tensors="pt").to(args.device)

        gen = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        out_txt = tokenizer.decode(gen[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        predicted_digits = "".join(ch for ch in out_txt if ch.isdigit())

        if str(passkey) in predicted_digits:
            hits += 1

    return {
        "status": "ok",
        "trials": args.passkey_trials,
        "accuracy": float(hits / args.passkey_trials) if args.passkey_trials > 0 else 0.0,
    }


# ============================================================
# Code-generation evals
# ============================================================
def humaneval_pass_at_1(generator_fn: Callable, tokenizer: Any, n: int, args: argparse.Namespace) -> Dict[str, Any]:
    from datasets import load_dataset

    ds = load_dataset("openai_humaneval", split="test")
    n = min(n, len(ds))
    ds = ds.select(range(n))

    passed = 0
    compiled = 0
    generated_chars = []
    exec_fail = 0

    for ex in tqdm(ds, desc="HumanEval"):
        user_prompt = (
            "Complete the following Python function.\n"
            "Return ONLY code.\n\n"
            f"{ex['prompt']}"
        )
        prompt_text = apply_chat_prompt(tokenizer, user_prompt)
        gen = generator_fn(prompt_text)
        gen_text = gen["text"]
        code_pred = extract_code_from_generation(gen_text)
        generated_chars.append(len(code_pred))

        full_code = (
            "import math\nfrom typing import *\n\n"
            f"{ex['prompt']}{code_pred}\n\n"
            f"{ex['test']}\n\n"
            f"check({ex['entry_point']})\n"
        )

        res = run_code_subprocess(
            full_code,
            timeout_s=args.exec_timeout_s,
            cpu_seconds=args.exec_cpu_seconds,
            memory_mb=args.exec_memory_mb,
        )
        if res["compiled"]:
            compiled += 1
        if res["passed"]:
            passed += 1
        else:
            if res["compiled"]:
                exec_fail += 1

    return {
        "status": "ok",
        "samples": n,
        "pass@1": (100.0 * passed / n) if n > 0 else 0.0,
        "compile_rate": (100.0 * compiled / n) if n > 0 else 0.0,
        "exec_rate": (100.0 * passed / compiled) if compiled > 0 else 0.0,
        "avg_gen_chars": float(np.mean(generated_chars)) if generated_chars else 0.0,
        "exec_failures_after_compile": exec_fail,
    }


def mbpp_pass_at_1(generator_fn: Callable, tokenizer: Any, n: int, args: argparse.Namespace) -> Dict[str, Any]:
    from datasets import load_dataset

    ds = load_dataset("mbpp", "sanitized", split="test")
    n = min(n, len(ds))
    ds = ds.select(range(n))

    passed = 0
    compiled = 0
    generated_chars = []
    exec_fail = 0

    prompt_key = "prompt" if "prompt" in ds.column_names else "text"

    for ex in tqdm(ds, desc="MBPP"):
        user_prompt = (
            "Write a Python function that solves the following problem.\n"
            "Return ONLY code.\n\n"
            f"{ex[prompt_key]}"
        )
        prompt_text = apply_chat_prompt(tokenizer, user_prompt)
        gen = generator_fn(prompt_text)
        gen_text = gen["text"]
        code_pred = extract_code_from_generation(gen_text)
        generated_chars.append(len(code_pred))

        full_code = (
            "import math\nfrom typing import *\n\n"
            f"{code_pred}\n\n"
            + "\n".join(ex["test_list"])
        )

        res = run_code_subprocess(
            full_code,
            timeout_s=args.exec_timeout_s,
            cpu_seconds=args.exec_cpu_seconds,
            memory_mb=args.exec_memory_mb,
        )
        if res["compiled"]:
            compiled += 1
        if res["passed"]:
            passed += 1
        else:
            if res["compiled"]:
                exec_fail += 1

    return {
        "status": "ok",
        "samples": n,
        "pass@1": (100.0 * passed / n) if n > 0 else 0.0,
        "compile_rate": (100.0 * compiled / n) if n > 0 else 0.0,
        "exec_rate": (100.0 * passed / compiled) if compiled > 0 else 0.0,
        "avg_gen_chars": float(np.mean(generated_chars)) if generated_chars else 0.0,
        "exec_failures_after_compile": exec_fail,
    }


# ============================================================
# External harnesses
# ============================================================
def parse_evalplus_pass(stdout_text: str) -> Optional[float]:
    for pat in [
        r"pass@1\s*:\s*([0-9.]+)",
        r"pass_at_1\s*[:=]\s*([0-9.]+)",
    ]:
        m = re.search(pat, stdout_text)
        if m:
            val = float(m.group(1))
            return val * 100.0 if val <= 1.0 else val
    return None


def run_evalplus(spec: ModelSpec, dataset: str, parallel: int, out_dir: str, timeout: int = 7200) -> Dict[str, Any]:
    log_path = os.path.join(out_dir, f"evalplus_{sanitize_name(spec.name)}_{dataset}.log")
    cmd = ["evalplus.evaluate", "--dataset", dataset, "--greedy"]

    if spec.type == "api":
        cmd.extend(["--model", spec.path, "--backend", "oai"])
        if spec.api_base:
            cmd.extend(["--base-url", spec.api_base])
    else:
        cmd.extend(["--model", spec.path, "--backend", "hf"])

    if parallel > 0:
        cmd.extend(["--parallel", str(parallel)])

    env = os.environ.copy()
    if spec.is_local_api and "OPENAI_API_KEY" not in env:
        env["OPENAI_API_KEY"] = "none"

    p = run_cmd(cmd, tee_to=log_path, env=env, timeout=timeout)
    score = parse_evalplus_pass(p.stdout)

    return {
        "status": "ok" if p.returncode == 0 else "failed",
        "dataset": dataset,
        "pass@1": score,
        "returncode": p.returncode,
        "log": log_path,
    }


def parse_lm_eval_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        data = json_load(path)
        return data.get("results", {})
    except Exception:
        return {}


def run_lm_eval(spec: ModelSpec, tasks: str, bs: int, dev: str, out_dir: str, timeout: int = 7200) -> Dict[str, Any]:
    out_json_path = os.path.join(out_dir, f"lm_eval_{sanitize_name(spec.name)}.json")
    log_path = out_json_path.replace(".json", ".log")

    cmd = [
        "lm_eval",
        "--tasks", tasks,
        "--output_path", out_json_path,
        "--confirm_run_unsafe_code",
    ]

    if spec.type == "api":
        if spec.is_local_api:
            model_args_parts = [f"model={spec.path}", "num_concurrent=16"]
            if spec.api_base:
                api_base = spec.api_base.strip("/")
                completions_url = api_base + ("/completions" if api_base.endswith("/v1") else "/v1/completions")
                model_args_parts.append(f"base_url={completions_url}")
            cmd.extend([
                "--model", "local-completions",
                "--model_args", ",".join(model_args_parts),
                "--batch_size", str(bs),
            ])
        else:
            model_args_parts = [f"model={spec.path}"]
            if spec.api_base:
                model_args_parts.append(f"base_url={spec.api_base}")
            cmd.extend(["--model", "openai", "--model_args", ",".join(model_args_parts)])
    else:
        cmd.extend([
            "--model", "hf",
            "--model_args", f"pretrained={spec.path}",
            "--device", dev,
            "--batch_size", str(bs),
        ])

    env = os.environ.copy()
    env["LM_EVAL_ALLOW_CODE_EXECUTION"] = "1"
    env["HF_ALLOW_CODE_EVAL"] = "1"
    if spec.is_local_api and "OPENAI_API_KEY" not in env:
        env["OPENAI_API_KEY"] = "none"

    p = run_cmd(cmd, tee_to=log_path, env=env, timeout=timeout)
    results = parse_lm_eval_json(out_json_path)

    parsed = {}
    for task, res in results.items():
        parsed[task] = {}
        for k, v in res.items():
            if isinstance(v, (int, float)):
                parsed[task][k] = v

    return {
        "status": "ok" if p.returncode == 0 else "failed",
        "tasks": tasks,
        "results": parsed,
        "returncode": p.returncode,
        "log": log_path,
        "json": out_json_path if os.path.exists(out_json_path) else None,
    }


# ============================================================
# Reporting helpers
# ============================================================
def flatten_results_for_summary(per_model_results: Dict[str, Any]) -> Dict[str, Any]:
    flat = {}
    for key, val in per_model_results.items():
        if isinstance(val, dict):
            if "pass@1" in val and isinstance(val["pass@1"], (int, float)):
                flat[f"{key}_pass@1"] = round(float(val["pass@1"]), 4)
            if "compile_rate" in val and isinstance(val["compile_rate"], (int, float)):
                flat[f"{key}_compile_rate"] = round(float(val["compile_rate"]), 4)
            if "exec_rate" in val and isinstance(val["exec_rate"], (int, float)):
                flat[f"{key}_exec_rate"] = round(float(val["exec_rate"]), 4)
            if "accuracy" in val and isinstance(val["accuracy"], (int, float)):
                flat[f"{key}_accuracy"] = round(float(val["accuracy"]), 6)
            if "ppl" in val and isinstance(val["ppl"], (int, float)):
                flat[f"{key}_ppl"] = round(float(val["ppl"]), 6)
            if "mean_nll" in val and isinstance(val["mean_nll"], (int, float)):
                flat[f"{key}_mean_nll"] = round(float(val["mean_nll"]), 6)
            if "tokens_per_sec" in val and isinstance(val["tokens_per_sec"], (int, float)):
                flat[f"{key}_tokens_per_sec"] = round(float(val["tokens_per_sec"]), 4)
            if "peak_memory_gb" in val and isinstance(val["peak_memory_gb"], (int, float)):
                flat[f"{key}_peak_memory_gb"] = round(float(val["peak_memory_gb"]), 4)
            if key == "lm_eval" and isinstance(val.get("results"), dict):
                for task, task_res in val["results"].items():
                    for mk, mv in task_res.items():
                        if isinstance(mv, (int, float)) and ("acc" in mk or "pass" in mk or "exact_match" in mk):
                            flat[f"lm_eval_{task}_{mk}"] = round(float(mv) * 100.0 if mv <= 1.0 else float(mv), 4)
        elif isinstance(val, (int, float)):
            flat[key] = round(float(val), 6)
        else:
            flat[key] = val
    return flat


def save_summary_csv(all_results: Dict[str, Dict[str, Any]], out_dir: str):
    rows = []
    all_keys = set()
    for model_name, res in all_results.items():
        flat = flatten_results_for_summary(res)
        flat["model"] = model_name
        rows.append(flat)
        all_keys.update(flat.keys())

    keys = ["model"] + sorted(k for k in all_keys if k != "model")
    path = os.path.join(out_dir, "summary.csv")
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"[OK] Saved CSV summary to: {path}")


def generate_markdown_report(all_results: Dict[str, Dict[str, Any]], out_dir: str):
    rows = []
    all_keys = set()
    for model_name, res in all_results.items():
        flat = flatten_results_for_summary(res)
        flat["model"] = model_name
        rows.append(flat)
        all_keys.update(flat.keys())

    keys = ["model"] + sorted(k for k in all_keys if k != "model")
    md = ["# Evaluation Report", ""]

    md.append("| " + " | ".join(keys) + " |")
    md.append("|" + "|".join([":---"] * len(keys)) + "|")

    for row in rows:
        vals = []
        for k in keys:
            v = row.get(k, "N/A")
            if isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        md.append("| " + " | ".join(vals) + " |")

    path = os.path.join(out_dir, "report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"[OK] Saved Markdown report to: {path}")


def plot_results(all_results: Dict[str, Dict[str, Any]], out_dir: str):
    rows = []
    for model_name, res in all_results.items():
        flat = flatten_results_for_summary(res)
        flat["model"] = model_name
        rows.append(flat)

    if not rows:
        return

    # Keep only numeric metrics
    metric_keys = sorted({
        k for r in rows for k, v in r.items()
        if k != "model" and isinstance(v, (int, float))
    })
    if not metric_keys:
        return

    ncols = min(4, len(metric_keys))
    nrows = int(math.ceil(len(metric_keys) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 4.2 * nrows), squeeze=False)

    model_names = [r["model"] for r in rows]

    for idx, key in enumerate(metric_keys):
        ax = axes[idx // ncols][idx % ncols]
        vals = [r.get(key, 0.0) for r in rows]
        ax.bar(model_names, vals)
        ax.set_title(key, fontsize=9)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        for j, v in enumerate(vals):
            if isinstance(v, (int, float)):
                ax.text(j, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    # hide unused axes
    for idx in range(len(metric_keys), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    fig.tight_layout()
    path = os.path.join(out_dir, "comparison_summary.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"[OK] Saved summary plot to: {path}")


# ============================================================
# Evaluation core
# ============================================================
def evaluate_model(spec: ModelSpec, args: argparse.Namespace) -> Dict[str, Any]:
    print(f"\n{'=' * 24} Evaluating: {spec.name} ({spec.path}) {'=' * 24}")
    raw_out_dir = ensure_dir(os.path.join(args.out_dir, "raw_outputs"))
    model_json_path = os.path.join(raw_out_dir, f"{sanitize_name(spec.name)}.json")

    if args.resume and os.path.exists(model_json_path):
        print(f"[INFO] Resuming from existing result: {model_json_path}")
        return json_load(model_json_path)

    original_no_proxy = os.environ.get("NO_PROXY")
    original_no_proxy_lower = os.environ.get("no_proxy")

    if spec.is_local_api:
        no_proxy_parts = original_no_proxy.split(",") if original_no_proxy else []
        if "127.0.0.1" not in no_proxy_parts:
            no_proxy_parts.append("127.0.0.1")
        if "localhost" not in no_proxy_parts:
            no_proxy_parts.append("localhost")
        joined = ",".join(no_proxy_parts)
        os.environ["NO_PROXY"] = joined
        os.environ["no_proxy"] = joined

    results: Dict[str, Any] = {
        "model_name": spec.name,
        "model_path": spec.path,
        "model_type": spec.type,
        "tokenizer_path": spec.tokenizer_path,
        "started_at": now_ts(),
    }

    model = None
    tokenizer = None
    prompt_tokenizer = None
    gen_fn = None

    try:
        prompt_tokenizer = load_prompt_tokenizer(spec.tokenizer_path or spec.path)
    except Exception as e:
        results["tokenizer_load_error"] = str(e)
        json_dump(results, model_json_path)
        return results

    try:
        if spec.type in {"hf", "local"}:
            try:
                model, tokenizer, backend = load_model_and_tokenizer(
                    spec.path,
                    args.max_seq_length,
                    args.device,
                    args.use_4bit_for_eval,
                )
                results["local_backend"] = backend
                prompt_tokenizer = tokenizer
                gen_fn = create_hf_generator(
                    model, tokenizer, args.device, args.humaneval_max_new_tokens, do_sample=False
                )
            except Exception as e:
                results["model_loading"] = {"status": "failed", "error": str(e)}

        elif spec.type == "api":
            try:
                gen_fn = create_api_generator(spec, args.humaneval_max_new_tokens, temperature=0.0)
            except Exception as e:
                results["generator_creation"] = {"status": "failed", "error": str(e)}

        # local-only benchmarks
        if model is not None and args.run_benchmark:
            try:
                results["benchmark"] = benchmark_generation(model, prompt_tokenizer, args)
            except Exception as e:
                results["benchmark"] = {"status": "failed", "error": str(e)}

        if model is not None and args.run_ppl:
            try:
                results["ppl_wikitext"] = perplexity_sliding_window(model, prompt_tokenizer, "wikitext", args)
            except Exception as e:
                results["ppl_wikitext"] = {"status": "failed", "error": str(e)}
            try:
                results["ppl_mbpp"] = perplexity_sliding_window(model, prompt_tokenizer, "mbpp", args)
            except Exception as e:
                results["ppl_mbpp"] = {"status": "failed", "error": str(e)}

        if model is not None and args.run_induction:
            try:
                results["induction"] = induction_repeat_nll(model, prompt_tokenizer, args)
            except Exception as e:
                results["induction"] = {"status": "failed", "error": str(e)}

        if model is not None and args.run_passkey:
            try:
                results["passkey"] = passkey_retrieval_acc(model, prompt_tokenizer, args)
            except Exception as e:
                results["passkey"] = {"status": "failed", "error": str(e)}

        if gen_fn is None:
            results["codegen_status"] = "skipped_no_generator"
            json_dump(results, model_json_path)
            return results

        if args.run_humaneval:
            print("Running Human Eval...")
            try:
                results["humaneval"] = humaneval_pass_at_1(gen_fn, prompt_tokenizer, args.humaneval_n, args)
            except Exception as e:
                results["humaneval"] = {"status": "failed", "error": str(e)}

        if args.run_mbpp:
            print("Running MBPP...")
            try:
                results["mbpp"] = mbpp_pass_at_1(gen_fn, prompt_tokenizer, args.mbpp_n, args)
            except Exception as e:
                results["mbpp"] = {"status": "failed", "error": str(e)}

        # ============================================================
        # FIX: 释放主进程的显存，给子进程 (EvalPlus / LM-Eval) 腾出空间
        # ============================================================
        if (args.run_evalplus or args.run_lm_eval) and model is not None:
            print("[INFO] Clearing VRAM before running external subprocesses...")
            cleanup_local_model(model, tokenizer)
            model = None
            tokenizer = None
            prompt_tokenizer = None
            gen_fn = None


        if args.run_evalplus:
            print("Running Eval Plus...")
            try:
                results["evalplus_humaneval"] = run_evalplus(
                    spec, "humaneval", args.evalplus_parallel, raw_out_dir, timeout=args.evalplus_timeout
                )
            except Exception as e:
                results["evalplus_humaneval"] = {"status": "failed", "error": str(e)}

            try:
                results["evalplus_mbpp"] = run_evalplus(
                    spec, "mbpp", args.evalplus_parallel, raw_out_dir, timeout=args.evalplus_timeout
                )
            except Exception as e:
                results["evalplus_mbpp"] = {"status": "failed", "error": str(e)}

        if args.run_lm_eval:
            try:
                results["lm_eval"] = run_lm_eval(
                    spec,
                    args.lm_eval_tasks,
                    args.lm_eval_batch_size,
                    args.device,
                    raw_out_dir,
                    timeout=args.lm_eval_timeout,
                )
            except Exception as e:
                results["lm_eval"] = {"status": "failed", "error": str(e)}

        results["finished_at"] = now_ts()
        json_dump(results, model_json_path)
        return results

    finally:
        cleanup_local_model(model, tokenizer)
        if spec.is_local_api:
            if original_no_proxy is None:
                os.environ.pop("NO_PROXY", None)
            else:
                os.environ["NO_PROXY"] = original_no_proxy

            if original_no_proxy_lower is None:
                os.environ.pop("no_proxy", None)
            else:
                os.environ["no_proxy"] = original_no_proxy_lower


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive coder model evaluation script",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # core
    parser.add_argument("--models", nargs="+", required=True,
                        help="HF model, local model folder, or openai:model_name")
    parser.add_argument("--out_dir", type=str, default=f"eval_results_{now_ts()}")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--use_4bit_for_eval", action="store_true")
    parser.add_argument("--prompt_tokenizer", type=str, default=None,
                        help="Optional tokenizer path used for prompt formatting, especially for API models.")
    parser.add_argument("--resume", action="store_true", help="Reuse per-model JSON results if already present.")

    # API
    parser.add_argument("--api_base", type=str, default=None)
    parser.add_argument("--api_key_env", type=str, default="OPENAI_API_KEY")

    # selectors
    parser.add_argument("--run_all", action="store_true")
    parser.add_argument("--run_benchmark", action="store_true")
    parser.add_argument("--run_ppl", action="store_true")
    parser.add_argument("--run_induction", action="store_true")
    parser.add_argument("--run_passkey", action="store_true")
    parser.add_argument("--run_humaneval", action="store_true")
    parser.add_argument("--run_mbpp", action="store_true")
    parser.add_argument("--run_evalplus", action="store_true")
    parser.add_argument("--run_lm_eval", action="store_true")

    # reporting
    parser.add_argument("--make_figures", action="store_true")

    # parameters
    parser.add_argument("--humaneval_n", type=int, default=164)
    parser.add_argument("--humaneval_max_new_tokens", type=int, default=512)
    parser.add_argument("--mbpp_n", type=int, default=399)
    parser.add_argument("--benchmark_max_new_tokens", type=int, default=256)

    parser.add_argument("--ppl_max_len", type=int, default=2048)
    parser.add_argument("--ppl_stride", type=int, default=512)

    parser.add_argument("--induction_seq_len", type=int, default=256)
    parser.add_argument("--induction_samples", type=int, default=20)

    parser.add_argument("--passkey_ctx", type=int, default=4096)
    parser.add_argument("--passkey_depth", type=float, default=0.5)
    parser.add_argument("--passkey_trials", type=int, default=10)

    parser.add_argument("--exec_timeout_s", type=float, default=8.0)
    parser.add_argument("--exec_cpu_seconds", type=int, default=8)
    parser.add_argument("--exec_memory_mb", type=int, default=1024)

    parser.add_argument("--evalplus_parallel", type=int, default=8)
    parser.add_argument("--evalplus_timeout", type=int, default=7200)

    parser.add_argument("--lm_eval_tasks", type=str, default=None)
    parser.add_argument("--lm_eval_batch_size", type=int, default=8)
    parser.add_argument("--lm_eval_timeout", type=int, default=7200)

    args = parser.parse_args()

    if args.run_all:
        args.run_benchmark = True
        args.run_ppl = True
        args.run_induction = True
        args.run_passkey = True
        args.run_humaneval = True
        args.run_mbpp = True
        args.run_evalplus = True
        args.run_lm_eval = True

    if args.run_lm_eval and not args.lm_eval_tasks:
        args.lm_eval_tasks = "humaneval"

    ensure_dir(args.out_dir)
    ensure_dir(os.path.join(args.out_dir, "raw_outputs"))
    set_seed(args.seed)

    model_specs = [parse_model_spec(m, args) for m in args.models]
    all_results: Dict[str, Dict[str, Any]] = {}

    for spec in model_specs:
        try:
            res = evaluate_model(spec, args)
            all_results[spec.name] = res
            print(f"\n--- Results for {spec.name} ---")
            print(json.dumps(flatten_results_for_summary(res), indent=2))
        except Exception as e:
            all_results[spec.name] = {
                "status": "catastrophic_failure",
                "error": str(e),
                "traceback": traceback.format_exc(),
            }

    summary = {
        "meta": {
            "args": {k: v for k, v in vars(args).items() if not k.startswith("_")},
            "timestamp": now_ts(),
        },
        "results": all_results,
    }

    summary_path = os.path.join(args.out_dir, "summary.json")
    json_dump(summary, summary_path)
    print(f"\n[OK] All evaluations complete. Summary saved to: {summary_path}")

    save_summary_csv(all_results, args.out_dir)
    generate_markdown_report(all_results, args.out_dir)

    if args.make_figures:
        plot_results(all_results, args.out_dir)


if __name__ == "__main__":
    main()

"""
python coder_eval_v5.py \
  --models \
    Qwen/Qwen2.5-Coder-14B-Instruct \
    ./output/qwen9b_sft_merged \
  --out_dir ./eval_compare_base_vs_sft \
  --run_humaneval \
  --run_mbpp \
  --run_evalplus \
  --run_lm_eval \
  --make_figures

python CodeAgent/qwen_coder_evalv5.py \
  --models \
    Qwen/Qwen3.5-9B \
    output/qwen9b_sft_merged \
  --out_dir eval_base_vs_sft \
  --run_humaneval \
  --run_mbpp \
  --run_evalplus \
  --run_lm_eval \
  --make_figures
"""

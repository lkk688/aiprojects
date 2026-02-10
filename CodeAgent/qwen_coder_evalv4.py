#!/usr/bin/env python3
"""
coder_eval_v3.py

Adds:
- HumanEval pass@1 (subset or full 164)
- MBPP pass@1 (exec unit tests from dataset)
Keeps:
- PPL WikiText2 (sliding window)
- PPL MBPP text (renamed to PPL_MBPP_Text for clarity)
- Induction repeat-half NLL
- Passkey retrieval (chat-template aligned, token-accurate)

SECURITY WARNING:
This script executes model-generated code (HumanEval + MBPP). Run inside a sandbox/container.

Example:
  python coder_eval_v3.py \
    --base unsloth/Qwen2.5-Coder-14B-Instruct \
    --fine ./qwen_coder_lora \
    --humaneval_n 164 \
    --mbpp_n 200 \
    --out eval_results_v3.json \
    --plot qwen_eval_comparison_v3.png

Notes:
- For trustworthy PPL, evaluate in bf16/fp16 (default). --use_4bit_for_eval makes PPL less reliable.
- For pass@1 stability, decoding is deterministic (do_sample=False, temperature=0).
"""

import os
import sys
import json
import math
import time
import random
import argparse
import subprocess
from typing import Dict, Any, List, Tuple, Optional

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
    try:
        from unsloth import is_bfloat16_supported
        return bool(is_bfloat16_supported())
    except Exception:
        if not torch.cuda.is_available():
            return False
        major, _minor = torch.cuda.get_device_capability(0)
        return major >= 8  # Ampere+


# -----------------------------
# Model loading
# -----------------------------
def load_model_and_tokenizer(
    model_name_or_path: str,
    max_seq_length: int,
    device: str = "cuda",
    use_4bit: bool = False,
) -> Tuple[torch.nn.Module, Any]:
    """
    Prefer Unsloth loader (works well with adapter folders).
    Falls back to Transformers if needed.
    """
    try:
        from unsloth import FastLanguageModel

        dtype = torch.bfloat16 if (device.startswith("cuda") and is_bf16_supported()) else torch.float16
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name_or_path,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=use_4bit,
        )
        try:
            FastLanguageModel.for_inference(model)
        except Exception:
            pass

        model.to(device)
        model.eval()
        return model, tokenizer

    except Exception as e:
        print(f"[WARN] Unsloth load failed: {e}\nFalling back to Transformers...")

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


# -----------------------------
# Perplexity (sliding window)
# -----------------------------
@torch.no_grad()
def perplexity_sliding_window(
    model,
    tokenizer,
    texts: List[str],
    device: str,
    max_length: int,
    stride: int,
) -> float:
    model.eval()
    total_nll = 0.0
    total_tokens = 0

    for text in tqdm(texts, desc="PPL"):
        enc = tokenizer(text, return_tensors="pt", add_special_tokens=True)
        input_ids = enc["input_ids"][0].to(device)

        if input_ids.numel() < 2:
            continue

        for start in range(0, input_ids.numel(), stride):
            end = min(start + max_length, input_ids.numel())
            window = input_ids[start:end]

            labels = window.clone()
            if start > 0:
                overlap = max_length - stride
                overlap = max(0, min(overlap, labels.numel()))
                labels[:overlap] = -100

            out = model(window.unsqueeze(0), labels=labels.unsqueeze(0))
            contrib = (labels != -100).sum().item()
            if contrib > 0:
                total_nll += out.loss.item() * contrib
                total_tokens += contrib

            if end == input_ids.numel():
                break

    if total_tokens == 0:
        return float("inf")

    return float(math.exp(total_nll / total_tokens))


def sample_texts_from_dataset(ds, text_cols: List[str], n_samples: int, seed: int) -> List[str]:
    rng = random.Random(seed)
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    idxs = idxs[: min(n_samples, len(idxs))]

    texts: List[str] = []
    for i in idxs:
        ex = ds[i]
        t = ""
        for c in text_cols:
            if c in ex and ex[c]:
                t = ex[c]
                break
        if t:
            texts.append(t)
    return texts


# -----------------------------
# Induction NLL (repeat-half)
# -----------------------------
@torch.no_grad()
def induction_repeat_nll(
    model,
    vocab_size: int,
    seq_len: int,
    n_samples: int,
    device: str,
) -> float:
    model.eval()
    nlls = []

    hi = min(int(vocab_size), 50000)
    lo = 1000 if hi > 2000 else 0

    for _ in tqdm(range(n_samples), desc="Induction"):
        A = torch.randint(low=lo, high=hi, size=(1, seq_len), device=device)
        inp = torch.cat([A, A], dim=1)
        labels = inp.clone()
        labels[:, :seq_len] = -100
        out = model(inp, labels=labels)
        nlls.append(float(out.loss.item()))

    return float(np.mean(nlls)) if nlls else float("inf")


# -----------------------------
# Passkey retrieval (chat-aligned)
# -----------------------------
@torch.no_grad()
def passkey_retrieval_acc(
    model,
    tokenizer,
    context_tokens: int,
    needle_depth: float,
    n_trials: int,
    device: str,
    max_new_tokens: int = 64,
    min_new_tokens: int = 4,
    use_chat_template: bool = True,
    debug_trials: int = 0,
) -> float:
    model.eval()
    hits = 0

    filler_sentence = "The sun sets in the west. "
    filler_ids = tokenizer.encode(filler_sentence, add_special_tokens=False)

    for t in tqdm(range(n_trials), desc="Passkey"):
        passkey = random.randint(10000, 99999)
        needle = f"\nThe secret passkey is {passkey}.\n"
        needle_ids = tokenizer.encode(needle, add_special_tokens=False)

        question = "\nWhat is the secret passkey? Answer with ONLY the number."
        q_ids = tokenizer.encode(question, add_special_tokens=False)

        budget = context_tokens - (len(needle_ids) + len(q_ids) + 16)
        budget = max(budget, 256)
        reps = max(1, budget // max(1, len(filler_ids)))
        filler = (filler_ids * reps)[:budget]

        insert_at = int(len(filler) * needle_depth)
        full_ctx_ids = filler[:insert_at] + needle_ids + filler[insert_at:]
        ctx_text = tokenizer.decode(full_ctx_ids, skip_special_tokens=True)

        if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": ctx_text + question}]
            prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
            prompt_len = inputs["input_ids"].shape[1]
            gen = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                do_sample=False,
                temperature=0.0,
                use_cache=True,
                pad_token_id=getattr(tokenizer, "eos_token_id", None),
            )
            out_ids = gen[0, prompt_len:].tolist()
        else:
            prompt_ids = torch.tensor([full_ctx_ids + q_ids], device=device)
            gen = model.generate(
                input_ids=prompt_ids,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                do_sample=False,
                temperature=0.0,
                use_cache=True,
                pad_token_id=getattr(tokenizer, "eos_token_id", None),
            )
            out_ids = gen[0, prompt_ids.shape[1]:].tolist()

        out_txt = tokenizer.decode(out_ids, skip_special_tokens=True)

        digits = "".join([ch if ch.isdigit() else " " for ch in out_txt]).split()
        pred = digits[0] if digits else ""

        if t < debug_trials:
            print(f"[DEBUG] GT={passkey} pred={pred} raw={repr(out_txt[:200])}")

        if pred == str(passkey):
            hits += 1

    return float(hits / n_trials) if n_trials > 0 else 0.0


# -----------------------------
# Code extraction + execution
# -----------------------------
def extract_code_from_generation(gen_text: str) -> str:
    t = gen_text
    if "```python" in t:
        t = t.split("```python", 1)[1].split("```", 1)[0]
        return t.strip()
    if "```" in t:
        t = t.split("```", 1)[1].split("```", 1)[0]
        return t.strip()
    return t.strip()


# def run_code_subprocess(code: str, timeout_s: float = 8.0) -> bool:
#     """
#     Execute code in a subprocess. Still unsafe; use sandbox.
#     Returns True if prints SUCCESS and no FAILURE.
#     """
#     wrapper = (
#         "import sys\n"
#         "try:\n"
#         + "\n".join("    " + line for line in code.splitlines())
#         + "\n    print('SUCCESS')\n"
#         "except Exception as e:\n"
#         "    print('FAILURE')\n"
#     )
#     try:
#         res = subprocess.run(
#             [sys.executable, "-c", wrapper],
#             capture_output=True,
#             text=True,
#             timeout=timeout_s,
#         )
#         return ("SUCCESS" in res.stdout) and ("FAILURE" not in res.stdout)
#     except subprocess.TimeoutExpired:
#         return False
#     except Exception:
#         return False
import tempfile
import textwrap
import subprocess
import sys
import os

def run_code_subprocess(code: str, timeout_s: float = 10.0) -> bool:
    """
    Run code by writing it to a temporary .py file and executing it.
    This avoids indentation corruption from the previous wrapper approach.
    Returns True if exit code == 0.
    """
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "prog.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
            f.write("\n")
        try:
            res = subprocess.run(
                [sys.executable, path],
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
            return res.returncode == 0
        except subprocess.TimeoutExpired:
            return False
        except Exception:
            return False

# -----------------------------
# HumanEval pass@1
# -----------------------------
@torch.no_grad()
def humaneval_pass_at_1(
    model,
    tokenizer,
    n_problems: int,
    device: str,
    max_new_tokens: int = 512,
) -> float:
    from datasets import load_dataset
    ds = load_dataset("openai_humaneval", split="test")
    n = min(n_problems, len(ds))
    ds = ds.select(range(n))

    passed = 0

    for ex in tqdm(ds, desc="HumanEval"):
        prompt = ex["prompt"]
        test_code = ex["test"]
        entry_point = ex["entry_point"]

        # Chat prompt
        if hasattr(tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": f"Complete the following Python function.\nReturn only code.\n\n{prompt}"}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = f"Complete the following Python function. Return only code.\n\n{prompt}\n"

        inputs = tokenizer(text, return_tensors="pt").to(device)

        gen = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            use_cache=True,
            pad_token_id=getattr(tokenizer, "eos_token_id", None),
        )

        gen_text = tokenizer.decode(gen[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        code_pred = extract_code_from_generation(gen_text)

        full = (
            "import math\n"
            "from typing import *\n\n"
            f"{prompt}\n"
            f"{code_pred}\n\n"
            f"{test_code}\n\n"
            f"check({entry_point})\n"
        )

        if run_code_subprocess(full, timeout_s=10.0):
            passed += 1

    return float(100.0 * passed / n) if n > 0 else 0.0


# # -----------------------------
# # MBPP pass@1
# # -----------------------------
# @torch.no_grad()
# def mbpp_pass_at_1(
#     model,
#     tokenizer,
#     n_problems: int,
#     device: str,
#     max_new_tokens: int = 512,
# ) -> float:
#     """
#     Evaluates MBPP by generating a function solution and running MBPP's tests.

#     Dataset: "mbpp", split="test"
#     Fields typically include:
#       - "text": problem statement
#       - "code": reference solution (not used)
#       - "test_list": list of assert statements (strings)
#       - "task_id": identifier

#     We prompt the model to write Python code only.
#     We then run:
#       - model code
#       - each assert in test_list

#     This is still a simplified harness.
#     """
#     from datasets import load_dataset
#     ds = load_dataset("mbpp", split="test")
#     n = min(n_problems, len(ds))
#     ds = ds.select(range(n))

#     passed = 0

#     for ex in tqdm(ds, desc="MBPP"):
#         text = ex.get("text", "")
#         tests = ex.get("test_list", []) or []

#         # Build prompt; MBPP does not provide entry_point reliably, so we just ask for code.
#         user_prompt = (
#             "Write a correct Python solution.\n"
#             "Return ONLY Python code (no explanations).\n\n"
#             f"Problem:\n{text}\n"
#         )

#         if hasattr(tokenizer, "apply_chat_template"):
#             messages = [{"role": "user", "content": user_prompt}]
#             prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#         else:
#             prompt = user_prompt

#         inputs = tokenizer(prompt, return_tensors="pt").to(device)

#         gen = model.generate(
#             **inputs,
#             max_new_tokens=max_new_tokens,
#             do_sample=False,
#             temperature=0.0,
#             use_cache=True,
#             pad_token_id=getattr(tokenizer, "eos_token_id", None),
#         )

#         gen_text = tokenizer.decode(gen[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
#         code_pred = extract_code_from_generation(gen_text)

#         # Assemble runnable script
#         # Put tests after code; tests are usually assert statements.
#         test_block = "\n".join(tests)
#         full = (
#             "import math\n"
#             "from typing import *\n\n"
#             f"{code_pred}\n\n"
#             f"{test_block}\n"
#         )

#         if run_code_subprocess(full, timeout_s=10.0):
#             passed += 1

#     return float(100.0 * passed / n) if n > 0 else 0.0

# -----------------------------
# MBPP pass@1 (FIXED)
# -----------------------------
# -----------------------------
# MBPP pass@1 (UPGRADED)
# -----------------------------
def run_code_subprocess_verbose(code: str, timeout_s: float = 10.0):
    import tempfile, os, subprocess, sys
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "prog.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
            f.write("\n")
        try:
            res = subprocess.run(
                [sys.executable, path],
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
            return (res.returncode == 0, res.stdout, res.stderr)
        except subprocess.TimeoutExpired:
            return (False, "", "TIMEOUT")
        except Exception as e:
            return (False, "", f"EXCEPTION: {e}")
        
@torch.no_grad()
def mbpp_pass_at_1(
    model,
    tokenizer,
    n_problems: int,
    device: str,
    max_new_tokens: int = 512,
    timeout_s: float = 10.0,
    debug_n: int = 3,                  # print first N failures
    failures_jsonl: str = "",          # optional: save failures for inspection
    tests_in_prompt: int = 5,          # include first K asserts in prompt (recommend 3-10)
    retry_on_fail: bool = True,        # do one extra attempt with error feedback
) -> float:
    """
    MBPP pass@1 with:
      - safe execution (file-based runner recommended)
      - entry-point inference primarily from asserts
      - prompt includes MBPP tests (reduces I/O mismatches)
      - records stderr tail for debugging / analysis
      - optional 1 retry with error feedback

    Requires:
      - extract_code_from_generation
      - run_code_subprocess_verbose (recommended) OR run_code_subprocess + a wrapper below
    """
    import re
    import json
    from datasets import load_dataset

    # ---------- helpers ----------
    def infer_entry_point(problem_text: str, tests: List[str]) -> str:
        joined = "\n".join(tests)
        # Most reliable: assert foo(...)
        m = re.search(r"assert\s+([A-Za-z_]\w*)\s*\(", joined)
        if m:
            return m.group(1)

        # fallback from problem statement
        m = re.search(r"\bfunction\s+([A-Za-z_]\w*)\b", problem_text)
        if m:
            return m.group(1)
        m = re.search(r"\bnamed\s+([A-Za-z_]\w*)\b", problem_text)
        if m:
            return m.group(1)
        return ""

    def build_prompt(problem_text: str, entry_point: str, tests: List[str], extra_feedback: str = "") -> str:
        # include a few tests to enforce I/O + exact output format
        tblock = ""
        if tests:
            tblock = "\n".join(tests[:max(0, int(tests_in_prompt))])
        if entry_point:
            base = (
                "Return ONLY valid Python code. No markdown fences. No explanation.\n"
                f"You MUST implement a function named `{entry_point}` exactly.\n"
                "Do not change the function name.\n"
                "You may define helper functions if needed.\n"
                "Do not include any tests or main code.\n\n"
                f"Problem:\n{problem_text}\n\n"
            )
        else:
            base = (
                "Return ONLY valid Python code. No markdown fences. No explanation.\n"
                "You may define helper functions if needed.\n"
                "Do not include any tests or main code.\n\n"
                f"Problem:\n{problem_text}\n\n"
            )

        if tblock:
            base += "Your solution MUST satisfy these tests:\n" + tblock + "\n\n"

        if extra_feedback:
            base += "Previous attempt failed with:\n" + extra_feedback.strip() + "\n\n"

        base += "Now output ONLY the Python code:\n"
        return base

    def generate_code(user_prompt: str) -> str:
        if hasattr(tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": user_prompt}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = user_prompt

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        gen = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            use_cache=True,
            pad_token_id=getattr(tokenizer, "eos_token_id", None),
        )
        gen_text = tokenizer.decode(gen[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return extract_code_from_generation(gen_text)

    # If you already replaced run_code_subprocess with file-based version, keep it.
    # But to capture errors we want a verbose runner.
    def run_verbose(code: str, timeout_s: float):
        """
        Uses run_code_subprocess_verbose if available; otherwise falls back to run_code_subprocess.
        Returns: ok(bool), stdout(str), stderr(str)
        """
        if "run_code_subprocess_verbose" in globals():
            return run_code_subprocess_verbose(code, timeout_s=timeout_s)
        # fallback (no stderr)
        ok = run_code_subprocess(code, timeout_s=timeout_s)
        return ok, "", "(no stderr captured; define run_code_subprocess_verbose to see errors)"

    # Richer prelude to reduce import-related failures
    prelude = (
        "import math\n"
        "import re\n"
        "import string\n"
        "import itertools\n"
        "import functools\n"
        "import collections\n"
        "import statistics\n"
        "from typing import *\n\n"
    )

    # ---------- load dataset ----------
    ds = load_dataset("mbpp", split="test")
    n = min(n_problems, len(ds))
    ds = ds.select(range(n))

    passed = 0
    failures = []
    shown = 0

    for ex in tqdm(ds, desc="MBPP"):
        problem = ex.get("text", "") or ""
        tests = ex.get("test_list", []) or []
        task_id = ex.get("task_id", None)

        entry = infer_entry_point(problem, tests)
        test_block = "\n".join(tests)

        # --- Attempt 1 ---
        prompt1 = build_prompt(problem, entry, tests)
        code1 = generate_code(prompt1)
        full1 = f"{prelude}{code1}\n\n{test_block}\n"
        ok1, out1, err1 = run_verbose(full1, timeout_s=timeout_s)

        if ok1:
            passed += 1
            continue

        # --- Attempt 2 (optional retry with feedback) ---
        ok2 = False
        code2 = ""
        err2 = ""
        if retry_on_fail:
            # include tail of stderr (common cases: AssertionError/TypeError/NameError)
            feedback = (err1 or "").strip()
            feedback_tail = feedback[-1200:] if feedback else ""
            # also include first test to focus
            first_test = tests[0] if tests else ""
            extra = ""
            if first_test:
                extra += f"First failing test (at least):\n{first_test}\n\n"
            if feedback_tail:
                extra += f"Error output:\n{feedback_tail}\n"

            prompt2 = build_prompt(problem, entry, tests, extra_feedback=extra)
            code2 = generate_code(prompt2)
            full2 = f"{prelude}{code2}\n\n{test_block}\n"
            ok2, out2, err2 = run_verbose(full2, timeout_s=timeout_s)

            if ok2:
                passed += 1
                continue

        # record failure
        if failures_jsonl:
            failures.append({
                "task_id": task_id,
                "entry_point": entry,
                "problem": problem,
                "tests": tests,
                "attempt1": {
                    "code_head": code1[:800],
                    "stderr_tail": (err1 or "")[-2000:],
                },
                "attempt2": {
                    "enabled": bool(retry_on_fail),
                    "code_head": code2[:800] if code2 else "",
                    "stderr_tail": (err2 or "")[-2000:] if err2 else "",
                },
            })

        if debug_n and shown < debug_n:
            shown += 1
            print("\n[MBPP FAIL]")
            print("task_id:", task_id, "entry_point:", entry)
            print("problem:", problem[:260])
            print("first_test:", tests[0] if tests else "<none>")
            print("code_pred_head:", code1[:260])
            if err1:
                print("stderr_tail:", err1[-600:])

    if failures_jsonl and failures:
        with open(failures_jsonl, "w", encoding="utf-8") as f:
            for row in failures:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return float(100.0 * passed / n) if n > 0 else 0.0


# -----------------------------
# Plotting
# -----------------------------
def plot_comparison(base: Dict[str, float], fine: Dict[str, float], save_path: str):
    keys = [k for k in base.keys() if k in fine.keys()]
    if not keys:
        print("[WARN] No overlapping metrics to plot.")
        return

    order = [
        "HumanEval_Pass@1",
        "MBPP_Pass@1",
        "Induction_Repeat_NLL",
        "PPL_MBPP_Text",
        "PPL_WikiText2",
        "Passkey_Acc",
    ]
    keys_sorted = [k for k in order if k in keys] + [k for k in sorted(keys) if k not in order]

    fig_w = min(5 * len(keys_sorted), 26)
    fig, axes = plt.subplots(1, len(keys_sorted), figsize=(fig_w, 5))
    if len(keys_sorted) == 1:
        axes = [axes]

    for ax, k in zip(axes, keys_sorted):
        ax.bar(["Base", "Finetuned"], [base[k], fine[k]])
        ax.set_title(k.replace("_", " "))
        for i, v in enumerate([base[k], fine[k]]):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom")
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    print(f"[OK] Saved plot to: {save_path}")


# -----------------------------
# Evaluation wrapper
# -----------------------------
def evaluate_one(
    model_name_or_path: str,
    device: str,
    max_seq_length: int,
    ppl_max_len: int,
    ppl_stride: int,
    ppl_samples: int,
    induction_seq_len: int,
    induction_samples: int,
    passkey_ctx: int,
    passkey_depth: float,
    passkey_trials: int,
    humaneval_n: int,
    humaneval_max_new_tokens: int,
    mbpp_n: int,
    mbpp_max_new_tokens: int,
    use_4bit_for_eval: bool,
    seed: int,
) -> Dict[str, float]:
    set_seed(seed)

    model, tokenizer = load_model_and_tokenizer(
        model_name_or_path=model_name_or_path,
        max_seq_length=max_seq_length,
        device=device,
        use_4bit=use_4bit_for_eval,
    )

    from datasets import load_dataset

    results: Dict[str, float] = {}

    # PPL WikiText2
    wt2 = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    wt_texts = sample_texts_from_dataset(wt2, ["text"], ppl_samples, seed=seed + 1)
    results["PPL_WikiText2"] = perplexity_sliding_window(
        model, tokenizer, wt_texts, device=device, max_length=ppl_max_len, stride=ppl_stride
    )

    # PPL on MBPP text (clarity: it's the text prompt, not solve rate)
    mbpp = load_dataset("mbpp", split="test")
    mbpp_texts = sample_texts_from_dataset(mbpp, ["text"], ppl_samples, seed=seed + 2)
    results["PPL_MBPP_Text"] = perplexity_sliding_window(
        model, tokenizer, mbpp_texts, device=device, max_length=ppl_max_len, stride=ppl_stride
    )

    # Induction
    vocab_size = int(getattr(getattr(model, "config", None), "vocab_size", 100000))
    results["Induction_Repeat_NLL"] = induction_repeat_nll(
        model=model,
        vocab_size=vocab_size,
        seq_len=induction_seq_len,
        n_samples=induction_samples,
        device=device,
    )

    # Passkey
    results["Passkey_Acc"] = passkey_retrieval_acc(
        model=model,
        tokenizer=tokenizer,
        context_tokens=passkey_ctx,
        needle_depth=passkey_depth,
        n_trials=passkey_trials,
        device=device,
        use_chat_template=True,
    )

    # HumanEval
    results["HumanEval_Pass@1"] = humaneval_pass_at_1(
        model=model,
        tokenizer=tokenizer,
        n_problems=humaneval_n,
        device=device,
        max_new_tokens=humaneval_max_new_tokens,
    )

    # MBPP solve rate
    results["MBPP_Pass@1"] = mbpp_pass_at_1(
        model=model,
        tokenizer=tokenizer,
        n_problems=mbpp_n,
        device=device,
        max_new_tokens=mbpp_max_new_tokens,
    )

    # cleanup
    del model, tokenizer
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, default="unsloth/Qwen2.5-Coder-14B-Instruct")
    parser.add_argument("--fine", type=str, default="./qwen_coder_lora")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=3407)

    parser.add_argument("--max_seq_length", type=int, default=8192)

    # PPL
    parser.add_argument("--ppl_max_len", type=int, default=2048)
    parser.add_argument("--ppl_stride", type=int, default=512)
    parser.add_argument("--ppl_samples", type=int, default=32)

    # Induction
    parser.add_argument("--induction_seq_len", type=int, default=256)
    parser.add_argument("--induction_samples", type=int, default=20)

    # Passkey
    parser.add_argument("--passkey_ctx", type=int, default=4096)
    parser.add_argument("--passkey_depth", type=float, default=0.5)
    parser.add_argument("--passkey_trials", type=int, default=10)

    # HumanEval
    parser.add_argument("--humaneval_n", type=int, default=164)
    parser.add_argument("--humaneval_max_new_tokens", type=int, default=512)

    # MBPP solve
    parser.add_argument("--mbpp_n", type=int, default=200)
    parser.add_argument("--mbpp_max_new_tokens", type=int, default=512)

    # Output
    parser.add_argument("--out", type=str, default="eval_results_v3.json")
    parser.add_argument("--plot", type=str, default="qwen_eval_comparison_v3.png")

    # Precision for eval
    parser.add_argument(
        "--use_4bit_for_eval",
        action="store_true",
        help="Evaluate in 4-bit (faster, but PPL less reliable).",
    )

    args = parser.parse_args()

    print("WARNING: This script executes generated code for HumanEval + MBPP. Run in a sandbox.\n")

    print(f"Evaluating BASE: {args.base}")
    base_res = evaluate_one(
        model_name_or_path=args.base,
        device=args.device,
        max_seq_length=args.max_seq_length,
        ppl_max_len=args.ppl_max_len,
        ppl_stride=args.ppl_stride,
        ppl_samples=args.ppl_samples,
        induction_seq_len=args.induction_seq_len,
        induction_samples=args.induction_samples,
        passkey_ctx=args.passkey_ctx,
        passkey_depth=args.passkey_depth,
        passkey_trials=args.passkey_trials,
        humaneval_n=args.humaneval_n,
        humaneval_max_new_tokens=args.humaneval_max_new_tokens,
        mbpp_n=args.mbpp_n,
        mbpp_max_new_tokens=args.mbpp_max_new_tokens,
        use_4bit_for_eval=args.use_4bit_for_eval,
        seed=args.seed,
    )
    print(json.dumps(base_res, indent=2))

    print(f"\nEvaluating FINETUNED: {args.fine}")
    fine_res = evaluate_one(
        model_name_or_path=args.fine,
        device=args.device,
        max_seq_length=args.max_seq_length,
        ppl_max_len=args.ppl_max_len,
        ppl_stride=args.ppl_stride,
        ppl_samples=args.ppl_samples,
        induction_seq_len=args.induction_seq_len,
        induction_samples=args.induction_samples,
        passkey_ctx=args.passkey_ctx,
        passkey_depth=args.passkey_depth,
        passkey_trials=args.passkey_trials,
        humaneval_n=args.humaneval_n,
        humaneval_max_new_tokens=args.humaneval_max_new_tokens,
        mbpp_n=args.mbpp_n,
        mbpp_max_new_tokens=args.mbpp_max_new_tokens,
        use_4bit_for_eval=args.use_4bit_for_eval,
        seed=args.seed,
    )
    print(json.dumps(fine_res, indent=2))

    payload = {
        "meta": {
            "timestamp": now_ts(),
            "base": args.base,
            "fine": args.fine,
            "device": args.device,
            "use_4bit_for_eval": bool(args.use_4bit_for_eval),
            "ppl": {"max_len": args.ppl_max_len, "stride": args.ppl_stride, "samples": args.ppl_samples},
            "induction": {"seq_len": args.induction_seq_len, "samples": args.induction_samples},
            "passkey": {"ctx": args.passkey_ctx, "depth": args.passkey_depth, "trials": args.passkey_trials},
            "humaneval": {"n": args.humaneval_n, "max_new_tokens": args.humaneval_max_new_tokens},
            "mbpp": {"n": args.mbpp_n, "max_new_tokens": args.mbpp_max_new_tokens},
        },
        "base": base_res,
        "finetuned": fine_res,
    }

    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[OK] Saved results to: {args.out}")

    plot_comparison(base_res, fine_res, args.plot)


if __name__ == "__main__":
    main()
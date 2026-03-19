#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Robust SFT Trainer for Qwen Code Models (LoRA / Unsloth)

Design goals:
1) Completion-first pipeline to protect code completion ability.
2) Exact-quota mixing so instruction data never overwhelms completion data.
3) Strong schema handling across common Python/code datasets.
4) Multiprocessing-safe completion extraction.
5) Practical CLI configurability for Qwen 9B / 14B / 32B / other Qwen models.

Recommended usage:
python train_qwen_sft.py \
  --base_model YOUR_QWEN_9B_MODEL_ID \
  --output_lora qwen9b_sft_lora \
  --output_dir outputs_qwen9b_sft

Notes:
- Replace --base_model with your actual Qwen 9B HF model ID.
- This script does NOT use HumanEval test prompts.
- MBPP uses train split only.
"""

import os
import sys
import re
import ast
import json
import math
import time
import hashlib
import random
import argparse
import subprocess
import multiprocessing as mp
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

# -----------------------------------------
# Environment guards
# -----------------------------------------
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

import torch

# Delay heavy imports until needed where possible.


# =====================================================================
# Dependency Management
# =====================================================================
def install_dependencies() -> None:
    print("[*] Installing dependencies...")
    packages = [
        "unsloth==2026.1.4",
        "unsloth_zoo==2026.1.4",
        "bitsandbytes",
        "accelerate",
        "peft",
        "trl==0.22.2",
        "triton",
        "transformers==4.57.3",
        "sentencepiece",
        "protobuf",
        "datasets==4.3.0",
        "huggingface_hub>=0.34.0",
        "hf_transfer",
    ]
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--upgrade", "--no-cache-dir"] + packages
    )
    print("[+] Dependencies installed.")


def lazy_import_training_stack():
    try:
        from unsloth import FastLanguageModel, is_bfloat16_supported
        from trl import SFTTrainer
        from transformers import TrainingArguments
        from datasets import load_dataset, Dataset
        return FastLanguageModel, is_bfloat16_supported, SFTTrainer, TrainingArguments, load_dataset, Dataset
    except Exception:
        install_dependencies()
        from unsloth import FastLanguageModel, is_bfloat16_supported
        from trl import SFTTrainer
        from transformers import TrainingArguments
        from datasets import load_dataset, Dataset
        return FastLanguageModel, is_bfloat16_supported, SFTTrainer, TrainingArguments, load_dataset, Dataset


# =====================================================================
# General Utilities
# =====================================================================
def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def json_dump(obj: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def make_training_args(TrainingArguments, **kwargs):
    """
    Compatibility shim across transformer versions.
    """
    try:
        return TrainingArguments(**kwargs)
    except TypeError as e:
        msg = str(e)
        patched = dict(kwargs)

        if "evaluation_strategy" in patched and "evaluation_strategy" in msg:
            patched["eval_strategy"] = patched.pop("evaluation_strategy")
        if "logging_strategy" in patched and "logging_strategy" in msg:
            patched["log_strategy"] = patched.pop("logging_strategy")
        return TrainingArguments(**patched)


# =====================================================================
# Text / Code Cleaning
# =====================================================================
_FENCE = "`" * 3
_CODE_FENCE_RE = re.compile(
    rf"{_FENCE}(?:python|py)?\s*(.*?){_FENCE}",
    re.DOTALL | re.IGNORECASE,
)

_COMMON_PREFACE_RE = re.compile(
    r"^\s*(Sure|Here(?:'|’)s|Here is|Below is|Certainly|Absolutely).*?\n",
    re.IGNORECASE,
)

def strip_to_code_only(text: str) -> str:
    """
    Lightly normalize assistant outputs into raw code.
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    t = text.strip()

    m = _CODE_FENCE_RE.search(t)
    if m:
        t = m.group(1).strip()

    t = _COMMON_PREFACE_RE.sub("", t).strip()

    return (t + "\n") if t else ""


def looks_like_python_code(text: str) -> bool:
    if not isinstance(text, str):
        return False
    t = text.strip()
    if len(t) < 20:
        return False
    hints = (
        "def ", "class ", "import ", "from ", "return ",
        "if __name__ ==", "try:", "except ", "lambda ",
    )
    score = sum(int(h in t) for h in hints)
    return score >= 1


def safe_parse_python(text: str) -> bool:
    if not isinstance(text, str) or not text.strip():
        return False
    try:
        ast.parse(text)
        return True
    except SyntaxError:
        return False


def trim_long_code_sample(code: str, max_chars: int) -> str:
    """
    Prevent very long script-like samples from dominating training.
    Prefer preserving the beginning where definitions/imports usually live.
    """
    if not isinstance(code, str):
        return ""
    code = code.strip()
    if len(code) <= max_chars:
        return code + "\n"
    return code[:max_chars].rstrip() + "\n"


# =====================================================================
# Prompt Builders
# =====================================================================
def prompt_code_completion(prefix: str) -> str:
    return (
        "Complete the following Python code.\n"
        "Rules:\n"
        "- Return ONLY valid Python code (the continuation).\n"
        "- No markdown.\n"
        "- No explanation.\n\n"
        "Code:\n"
        f"{prefix}"
    )


def prompt_instruction_to_code(instr: str, sys_msg: str = "") -> str:
    base = (
        "Write correct Python code for the following request.\n"
        "Rules:\n"
        "- Return ONLY valid Python code.\n"
        "- No markdown fences.\n"
        "- No explanations.\n\n"
    )
    if sys_msg and sys_msg.strip():
        base = f"System:\n{sys_msg.strip()}\n\n" + base
    return base + f"Request:\n{instr}\n"


def prompt_mbpp(problem: str, entry_point: str = "") -> str:
    req = f"- You MUST implement a function named `{entry_point}` exactly.\n" if entry_point else ""
    return (
        "Write a correct Python solution.\n"
        "Rules:\n"
        "- Return ONLY valid Python code.\n"
        "- No markdown.\n"
        "- No explanation.\n"
        f"{req}\n"
        f"Problem:\n{problem}\n"
    )


def prompt_patch(problem: str, fail_log_tail: str, repo_summary: str = "") -> str:
    return (
        "You are a software engineer fixing a real repository bug.\n"
        "Rules:\n"
        "- Output ONLY a unified diff patch.\n"
        "- Start with: diff --git a/... b/...\n"
        "- Do NOT include markdown or explanations.\n"
        "- Keep the patch minimal.\n"
        "- Make tests pass.\n\n"
        f"Issue:\n{problem}\n\n"
        f"Repository summary:\n{repo_summary}\n\n"
        f"Failing log (tail):\n{fail_log_tail}\n\n"
        "Now output the patch ONLY:\n"
    )


# =====================================================================
# Chat Formatting
# =====================================================================
def make_chat_text(tokenizer, user: str, assistant: str) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        convo = [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
        return tokenizer.apply_chat_template(
            convo, tokenize=False, add_generation_prompt=False
        )
    return f"User:\n{user}\n\nAssistant:\n{assistant}"


# =====================================================================
# Dataset Schema Helpers
# =====================================================================
def infer_entry_point_from_mbpp_tests(tests: List[str], problem_text: str) -> str:
    joined = "\n".join(tests or [])
    m = re.search(r"assert\s+([A-Za-z_]\w*)\s*\(", joined)
    if m:
        return m.group(1)
    m = re.search(r"\bfunction\s+([A-Za-z_]\w*)\b", problem_text or "")
    if m:
        return m.group(1)
    m = re.search(r"\bnamed\s+([A-Za-z_]\w*)\b", problem_text or "")
    if m:
        return m.group(1)
    return ""


def load_patch_sft_jsonl(load_dataset_fn, path: str):
    if not path or not os.path.exists(path):
        return None
    ds = load_dataset_fn("json", data_files=path, split="train")
    if "prompt" not in ds.column_names or "patch" not in ds.column_names:
        print(f"[WARN] patch_sft.jsonl missing prompt/patch fields: {ds.column_names}")
        return None
    return ds


def dataset_has_columns(ds, cols: List[str]) -> bool:
    return all(c in ds.column_names for c in cols)


def choose_first_present(ex: Dict[str, Any], keys: List[str], default: Any = "") -> Any:
    for k in keys:
        if k in ex and ex[k] is not None:
            return ex[k]
    return default


# =====================================================================
# Completion Extraction
# =====================================================================
def extract_function_completion_pairs(
    code: str,
    seed: int,
    min_func_lines: int = 12,
    min_suffix_chars: int = 80,
    max_pairs_per_code: int = 2,
) -> List[Tuple[str, str]]:
    """
    AST-based split inside function bodies.
    """
    if not isinstance(code, str) or len(code) < 200:
        return []

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    lines = code.splitlines(keepends=True)
    if not lines:
        return []

    rng = random.Random(seed ^ (len(code) * 1315423911 & 0xFFFFFFFF))

    candidates = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start = node.lineno - 1
            end = getattr(node, "end_lineno", node.lineno + len(node.body))
            if (end - start) >= min_func_lines and len(node.body) >= 2:
                candidates.append(node)

    if not candidates:
        return []

    rng.shuffle(candidates)
    pairs: List[Tuple[str, str]] = []

    for node in candidates:
        body_nodes = node.body
        if len(body_nodes) < 2:
            continue

        low = max(1, int(len(body_nodes) * 0.3))
        high = max(low + 1, int(len(body_nodes) * 0.8))
        if high <= low:
            continue

        split_idx = rng.randint(low, high - 1)
        split_stmt = body_nodes[split_idx]
        cut_lineno = split_stmt.lineno - 1

        if cut_lineno <= 0 or cut_lineno >= len(lines):
            continue

        prefix = "".join(lines[:cut_lineno])
        suffix = "".join(lines[cut_lineno:])

        if len(suffix.strip()) < min_suffix_chars:
            continue

        pairs.append((prefix, suffix))
        if len(pairs) >= max_pairs_per_code:
            break

    return pairs


def extract_completion_pairs_with_fallback(
    code: str,
    seed: int,
    min_func_lines: int,
    min_suffix_chars: int,
    max_pairs_per_code: int,
) -> List[Tuple[str, str]]:
    code = strip_to_code_only(code)
    if not looks_like_python_code(code):
        return []

    pairs = extract_function_completion_pairs(
        code=code,
        seed=seed,
        min_func_lines=min_func_lines,
        min_suffix_chars=min_suffix_chars,
        max_pairs_per_code=max_pairs_per_code,
    )
    if pairs:
        return pairs

    lines = code.splitlines(keepends=True)
    if len(lines) < min_func_lines:
        return []

    rng = random.Random(seed ^ (len(lines) * 2654435761 & 0xFFFFFFFF))
    fallback_pairs = []

    trials = min(2, max_pairs_per_code)
    for _ in range(trials):
        lo = max(1, int(len(lines) * 0.2))
        hi = max(lo + 1, int(len(lines) * 0.8))
        if hi <= lo:
            break
        cut = rng.randint(lo, hi - 1)
        prefix = "".join(lines[:cut])
        suffix = "".join(lines[cut:])
        if len(suffix.strip()) >= min_suffix_chars:
            fallback_pairs.append((prefix, suffix))

    return fallback_pairs[:max_pairs_per_code]


def process_completion_batch_raw(
    examples: Dict[str, List[Any]],
    seed: int,
    min_lines: int,
    min_chars: int,
    max_pairs_per_code: int,
) -> Dict[str, List[str]]:
    texts = []
    raw_codes = examples.get("raw_code", [])
    for idx, code in enumerate(raw_codes):
        if not isinstance(code, str):
            continue
        local_seed = seed + idx * 9973
        pairs = extract_completion_pairs_with_fallback(
            code=code,
            seed=local_seed,
            min_func_lines=min_lines,
            min_suffix_chars=min_chars,
            max_pairs_per_code=max_pairs_per_code,
        )
        for prefix, suffix in pairs:
            texts.append(json.dumps({"user": prompt_code_completion(prefix), "assistant": suffix}, ensure_ascii=False))
    return {"text_json": texts}


# =====================================================================
# Dataset Text Validation / Cleanup
# =====================================================================
def validate_nonempty_text(ex: Dict[str, Any], min_len: int = 32) -> bool:
    t = ex.get("text", "")
    return isinstance(t, str) and len(t.strip()) >= min_len


def dedupe_texts_preserve_order(texts: List[str]) -> List[str]:
    seen = set()
    out = []
    for t in texts:
        h = sha1_text(t)
        if h in seen:
            continue
        seen.add(h)
        out.append(t)
    return out


# =====================================================================
# Dataset Formatting
# =====================================================================
def format_magicoder_row(ex: Dict[str, Any], tokenizer, max_code_chars: int) -> Optional[str]:
    problem = choose_first_present(ex, ["problem", "instruction"], "")
    solution = choose_first_present(ex, ["solution", "output", "answer"], "")
    lang = choose_first_present(ex, ["lang", "language"], "")

    if not problem or not solution:
        return None
    if lang and "python" not in str(lang).lower():
        return None

    code = trim_long_code_sample(strip_to_code_only(solution), max_code_chars)
    if not looks_like_python_code(code):
        return None

    user = prompt_instruction_to_code(problem)
    return make_chat_text(tokenizer, user, code)


def format_evol_row(ex: Dict[str, Any], tokenizer, max_code_chars: int) -> Optional[str]:
    instr = choose_first_present(ex, ["instruction", "problem", "prompt"], "")
    out = choose_first_present(ex, ["output", "solution", "answer"], "")
    if not instr or not out:
        return None

    code = trim_long_code_sample(strip_to_code_only(out), max_code_chars)
    if not looks_like_python_code(code):
        return None

    user = prompt_instruction_to_code(instr)
    return make_chat_text(tokenizer, user, code)


def format_py_instr_row(ex: Dict[str, Any], tokenizer, max_code_chars: int) -> Optional[str]:
    """
    Robust handling for jtatman/python-code-dataset-500k:
    - instruction/output style
    - raw code corpora style
    """
    instr = choose_first_present(ex, ["instruction", "prompt", "problem"], "")
    sysm = choose_first_present(ex, ["system"], "")
    out = choose_first_present(ex, ["output", "answer", "solution"], "")

    if instr and out:
        code = trim_long_code_sample(strip_to_code_only(out), max_code_chars)
        if looks_like_python_code(code):
            user = prompt_instruction_to_code(instr, sysm)
            return make_chat_text(tokenizer, user, code)

    raw_code = choose_first_present(ex, ["text", "code", "content", "body"], "")
    raw_code = trim_long_code_sample(strip_to_code_only(raw_code), max_code_chars)
    if looks_like_python_code(raw_code):
        prefix = (
            "Write Python code equivalent to the following task.\n"
            "Rules:\n"
            "- Return ONLY valid Python code.\n"
            "- No markdown.\n"
            "- No explanation.\n\n"
            "Task:\nImplement the code.\n"
        )
        return make_chat_text(tokenizer, prefix, raw_code)

    return None


def format_mbpp_row(ex: Dict[str, Any], tokenizer, max_code_chars: int) -> Optional[str]:
    problem = ex.get("text", "") or ""
    tests = ex.get("test_list", []) or []
    code = trim_long_code_sample(strip_to_code_only(ex.get("code", "") or ""), max_code_chars)

    if not problem or not code:
        return None
    if not looks_like_python_code(code):
        return None

    entry = infer_entry_point_from_mbpp_tests(tests, problem)
    user = prompt_mbpp(problem, entry_point=entry)
    return make_chat_text(tokenizer, user, code)


def format_patch_row(ex: Dict[str, Any], tokenizer) -> Optional[str]:
    prompt = ex.get("prompt", "") or ""
    patch = (ex.get("patch", "") or "").strip()
    if not prompt or not patch:
        return None
    if ("diff --git" not in patch) and (not patch.startswith("---")):
        return None
    return make_chat_text(tokenizer, prompt, patch + "\n")


# =====================================================================
# Config
# =====================================================================
@dataclass
class CFG:
    base_model: str
    output_lora: str
    output_dir: str
    seed: int = 3407

    max_seq_length: int = 4096
    load_in_4bit: bool = True

    per_device_bs: int = 4
    grad_acc: int = 4
    epochs: float = 1.0
    lr: float = 7.0e-6
    wd: float = 0.01
    warmup_ratio: float = 0.03

    eval_steps: int = 250
    save_steps: int = 500
    save_total_limit: int = 2
    logging_steps: int = 20

    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0

    packing: bool = True
    neftune_noise_alpha: float = 5.0

    # raw dataset subsample sizes
    n_magicoder: int = 12000
    n_evol: int = 12000
    n_py_instr: int = 60000
    n_mbpp_train: int = 2000

    # final mix quotas
    mix_completion: int = 42000
    mix_mbpp: int = 12000
    mix_evol: int = 8000
    mix_magicoder: int = 6000
    mix_py_instr: int = 6000
    mix_patch: int = 0
    max_mixed: int = 80000

    # completion extraction
    completion_min_lines: int = 12
    completion_min_suffix_chars: int = 60
    completion_max_pairs_per_code: int = 2

    # sample shaping
    max_code_chars: int = 12000
    min_text_len: int = 32

    # data
    patch_jsonl: str = "data/patch_sft.jsonl"

    # multiprocessing
    num_proc: int = max(1, mp.cpu_count() - 2)


# =====================================================================
# CLI
# =====================================================================
def parse_args() -> CFG:
    parser = argparse.ArgumentParser(description="Robust Qwen SFT Trainer")

    parser.add_argument("--base_model", type=str, required=True, help="HF model id of your Qwen 9B/base model")
    parser.add_argument("--output_lora", type=str, default="qwen_sft_lora")
    parser.add_argument("--output_dir", type=str, default="outputs_qwen_sft")

    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--no_load_in_4bit", action="store_true")

    parser.add_argument("--per_device_bs", type=int, default=4)
    parser.add_argument("--grad_acc", type=int, default=4)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=7.0e-6)
    parser.add_argument("--wd", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)

    parser.add_argument("--eval_steps", type=int, default=250)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--logging_steps", type=int, default=20)

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.0)

    parser.add_argument("--n_magicoder", type=int, default=12000)
    parser.add_argument("--n_evol", type=int, default=12000)
    parser.add_argument("--n_py_instr", type=int, default=60000)
    parser.add_argument("--n_mbpp_train", type=int, default=2000)

    parser.add_argument("--mix_completion", type=int, default=42000)
    parser.add_argument("--mix_mbpp", type=int, default=12000)
    parser.add_argument("--mix_evol", type=int, default=8000)
    parser.add_argument("--mix_magicoder", type=int, default=6000)
    parser.add_argument("--mix_py_instr", type=int, default=6000)
    parser.add_argument("--mix_patch", type=int, default=0)
    parser.add_argument("--max_mixed", type=int, default=80000)

    parser.add_argument("--completion_min_lines", type=int, default=12)
    parser.add_argument("--completion_min_suffix_chars", type=int, default=60)
    parser.add_argument("--completion_max_pairs_per_code", type=int, default=2)

    parser.add_argument("--patch_jsonl", type=str, default="data/patch_sft.jsonl")
    parser.add_argument("--num_proc", type=int, default=max(1, mp.cpu_count() - 2))
    parser.add_argument("--max_code_chars", type=int, default=12000)
    parser.add_argument("--min_text_len", type=int, default=32)

    args = parser.parse_args()

    load_in_4bit = True
    if args.no_load_in_4bit:
        load_in_4bit = False
    elif args.load_in_4bit:
        load_in_4bit = True

    return CFG(
        base_model=args.base_model,
        output_lora=args.output_lora,
        output_dir=args.output_dir,
        seed=args.seed,
        max_seq_length=args.max_seq_length,
        load_in_4bit=load_in_4bit,
        per_device_bs=args.per_device_bs,
        grad_acc=args.grad_acc,
        epochs=args.epochs,
        lr=args.lr,
        wd=args.wd,
        warmup_ratio=args.warmup_ratio,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        n_magicoder=args.n_magicoder,
        n_evol=args.n_evol,
        n_py_instr=args.n_py_instr,
        n_mbpp_train=args.n_mbpp_train,
        mix_completion=args.mix_completion,
        mix_mbpp=args.mix_mbpp,
        mix_evol=args.mix_evol,
        mix_magicoder=args.mix_magicoder,
        mix_py_instr=args.mix_py_instr,
        mix_patch=args.mix_patch,
        max_mixed=args.max_mixed,
        completion_min_lines=args.completion_min_lines,
        completion_min_suffix_chars=args.completion_min_suffix_chars,
        completion_max_pairs_per_code=args.completion_max_pairs_per_code,
        patch_jsonl=args.patch_jsonl,
        num_proc=max(1, args.num_proc),
        max_code_chars=args.max_code_chars,
        min_text_len=args.min_text_len,
    )


# =====================================================================
# Main
# =====================================================================
def main():
    cfg = parse_args()
    ensure_dir(cfg.output_dir)
    seed_everything(cfg.seed)

    FastLanguageModel, is_bfloat16_supported, SFTTrainer, TrainingArguments, load_dataset, Dataset = lazy_import_training_stack()

    print(f"[*] Config:\n{json.dumps(asdict(cfg), ensure_ascii=False, indent=2)}")
    json_dump(asdict(cfg), os.path.join(cfg.output_dir, "config.json"))

    # -------------------------------------------------
    # Load model/tokenizer
    # -------------------------------------------------
    print(f"[*] Loading model: {cfg.base_model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.base_model,
        max_seq_length=cfg.max_seq_length,
        dtype=None,
        load_in_4bit=cfg.load_in_4bit,
    )

    print("[*] Applying LoRA...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora_r,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=cfg.seed,
        use_rslora=False,
        loftq_config=None,
    )

    # -------------------------------------------------
    # Load raw datasets
    # -------------------------------------------------
    print("[*] Loading datasets...")
    ds_magic = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train", trust_remote_code=False)
    if len(ds_magic) > cfg.n_magicoder:
        ds_magic = ds_magic.shuffle(seed=cfg.seed).select(range(cfg.n_magicoder))

    ds_evol = load_dataset("nickrosh/Evol-Instruct-Code-80k-v1", split="train", trust_remote_code=False)
    if len(ds_evol) > cfg.n_evol:
        ds_evol = ds_evol.shuffle(seed=cfg.seed).select(range(cfg.n_evol))

    ds_py = load_dataset("jtatman/python-code-dataset-500k", split="train", trust_remote_code=False)
    if len(ds_py) > cfg.n_py_instr:
        ds_py = ds_py.shuffle(seed=cfg.seed).select(range(cfg.n_py_instr))

    ds_mbpp = load_dataset("mbpp", split="train", trust_remote_code=False)
    if len(ds_mbpp) > cfg.n_mbpp_train:
        ds_mbpp = ds_mbpp.shuffle(seed=cfg.seed).select(range(cfg.n_mbpp_train))

    ds_patch = load_patch_sft_jsonl(load_dataset, cfg.patch_jsonl)
    if ds_patch is None or len(ds_patch) == 0:
        ds_patch = None
        print("[WARN] patch bucket disabled.")
    else:
        cfg.mix_patch = min(cfg.mix_patch, len(ds_patch))
        print(f"[OK] patch bucket loaded: {len(ds_patch)} rows")

    print("[DEBUG] magic columns:", ds_magic.column_names)
    print("[DEBUG] evol columns :", ds_evol.column_names)
    print("[DEBUG] py columns   :", ds_py.column_names)
    print("[DEBUG] mbpp columns :", ds_mbpp.column_names)
    if ds_patch is not None:
        print("[DEBUG] patch columns:", ds_patch.column_names)

    # -------------------------------------------------
    # Phase 1: Build completion bucket FIRST
    # -------------------------------------------------
    print(f"[*] Building completion bucket with {cfg.num_proc} processes...")

    raw_codes: List[str] = []

    if "solution" in ds_magic.column_names:
        raw_codes.extend([x for x in ds_magic["solution"] if isinstance(x, str)])
    elif "output" in ds_magic.column_names:
        raw_codes.extend([x for x in ds_magic["output"] if isinstance(x, str)])

    if "output" in ds_evol.column_names:
        raw_codes.extend([x for x in ds_evol["output"] if isinstance(x, str)])
    elif "solution" in ds_evol.column_names:
        raw_codes.extend([x for x in ds_evol["solution"] if isinstance(x, str)])

    if "code" in ds_mbpp.column_names:
        raw_codes.extend([x for x in ds_mbpp["code"] if isinstance(x, str)])

    ds_raw_codes = Dataset.from_dict({"raw_code": raw_codes})

    ds_completion_json = ds_raw_codes.map(
        process_completion_batch_raw,
        batched=True,
        batch_size=1000,
        num_proc=cfg.num_proc,
        remove_columns=["raw_code"],
        fn_kwargs=dict(
            seed=cfg.seed,
            min_lines=cfg.completion_min_lines,
            min_chars=cfg.completion_min_suffix_chars,
            max_pairs_per_code=cfg.completion_max_pairs_per_code,
        ),
        desc="Extracting completion pairs",
    )

    completion_pairs = ds_completion_json["text_json"] if "text_json" in ds_completion_json.column_names else []
    completion_texts = []
    for item in completion_pairs:
        try:
            obj = json.loads(item)
            user = obj["user"]
            assistant = trim_long_code_sample(obj["assistant"], cfg.max_code_chars)
            completion_texts.append(make_chat_text(tokenizer, user, assistant))
        except Exception:
            continue

    completion_texts = [t for t in completion_texts if isinstance(t, str) and len(t) >= cfg.min_text_len]
    completion_texts = dedupe_texts_preserve_order(completion_texts)
    ds_completion = Dataset.from_dict({"text": completion_texts})

    print(f"[+] completion bucket size: {len(ds_completion)}")

    if len(ds_completion) == 0:
        raise RuntimeError("Completion bucket is empty. Aborting because completion must dominate this SFT.")

    # -------------------------------------------------
    # Phase 2: Format SFT instruction buckets
    # -------------------------------------------------
    print("[*] Formatting SFT instruction buckets...")

    magic_texts = []
    for ex in ds_magic:
        t = format_magicoder_row(ex, tokenizer, cfg.max_code_chars)
        if t and len(t) >= cfg.min_text_len:
            magic_texts.append(t)

    evol_texts = []
    for ex in ds_evol:
        t = format_evol_row(ex, tokenizer, cfg.max_code_chars)
        if t and len(t) >= cfg.min_text_len:
            evol_texts.append(t)

    py_texts = []
    for ex in ds_py:
        t = format_py_instr_row(ex, tokenizer, cfg.max_code_chars)
        if t and len(t) >= cfg.min_text_len:
            py_texts.append(t)

    mbpp_texts = []
    for ex in ds_mbpp:
        t = format_mbpp_row(ex, tokenizer, cfg.max_code_chars)
        if t and len(t) >= cfg.min_text_len:
            mbpp_texts.append(t)

    patch_texts = []
    if ds_patch is not None:
        for ex in ds_patch:
            t = format_patch_row(ex, tokenizer)
            if t and len(t) >= cfg.min_text_len:
                patch_texts.append(t)

    # de-dup
    magic_texts = dedupe_texts_preserve_order(magic_texts)
    evol_texts = dedupe_texts_preserve_order(evol_texts)
    py_texts = dedupe_texts_preserve_order(py_texts)
    mbpp_texts = dedupe_texts_preserve_order(mbpp_texts)
    patch_texts = dedupe_texts_preserve_order(patch_texts)

    ds_magic_fmt = Dataset.from_dict({"text": magic_texts})
    ds_evol_fmt = Dataset.from_dict({"text": evol_texts})
    ds_py_fmt = Dataset.from_dict({"text": py_texts})
    ds_mbpp_fmt = Dataset.from_dict({"text": mbpp_texts})
    ds_patch_fmt = Dataset.from_dict({"text": patch_texts}) if patch_texts else None

    print("[+] bucket sizes after formatting:")
    print(f"    completion: {len(ds_completion)}")
    print(f"    mbpp      : {len(ds_mbpp_fmt)}")
    print(f"    evol      : {len(ds_evol_fmt)}")
    print(f"    magicoder : {len(ds_magic_fmt)}")
    print(f"    py_instr  : {len(ds_py_fmt)}")
    print(f"    patch     : {len(ds_patch_fmt) if ds_patch_fmt is not None else 0}")

    # -------------------------------------------------
    # Phase 3: Fixed-count exact quota mixing
    # -------------------------------------------------
    def sample_bucket_texts(ds, k: int, seed: int) -> List[str]:
        if ds is None or len(ds) == 0 or k <= 0:
            return []
        rng = random.Random(seed)
        if len(ds) >= k:
            picked = ds.shuffle(seed=seed).select(range(k))
            return list(picked["text"])
        out = list(ds["text"])
        while len(out) < k:
            out.append(ds[rng.randrange(len(ds))]["text"])
        return out[:k]

    mix_plan = [
        ("completion", ds_completion, min(cfg.mix_completion, len(ds_completion))),
        ("mbpp", ds_mbpp_fmt, min(cfg.mix_mbpp, len(ds_mbpp_fmt))),
        ("evol", ds_evol_fmt, min(cfg.mix_evol, len(ds_evol_fmt))),
        ("magicoder", ds_magic_fmt, min(cfg.mix_magicoder, len(ds_magic_fmt))),
        ("py_instr", ds_py_fmt, min(cfg.mix_py_instr, len(ds_py_fmt))),
    ]
    if ds_patch_fmt is not None and cfg.mix_patch > 0:
        mix_plan.append(("patch", ds_patch_fmt, min(cfg.mix_patch, len(ds_patch_fmt))))

    total_target = sum(k for _, _, k in mix_plan)
    if total_target > cfg.max_mixed:
        overflow = total_target - cfg.max_mixed
        shrink_order = ["py_instr", "magicoder", "evol", "mbpp", "patch"]
        temp = [[n, d, k] for n, d, k in mix_plan]
        idx = {x[0]: i for i, x in enumerate(temp)}
        for name in shrink_order:
            if overflow <= 0:
                break
            if name in idx:
                i = idx[name]
                reducible = max(0, temp[i][2] - 1000)
                dec = min(reducible, overflow)
                temp[i][2] -= dec
                overflow -= dec
        mix_plan = [(n, d, int(k)) for n, d, k in temp if int(k) > 0]

    mixed_texts: List[str] = []
    mix_info = []
    for i, (name, ds, k) in enumerate(mix_plan):
        picked = sample_bucket_texts(ds, k, cfg.seed + i * 101)
        mixed_texts.extend(picked)
        mix_info.append({
            "bucket": name,
            "target": k,
            "available": len(ds) if ds is not None else 0,
            "used": len(picked),
        })

    mixed_texts = dedupe_texts_preserve_order(mixed_texts)
    random.Random(cfg.seed).shuffle(mixed_texts)

    mixed_ds = Dataset.from_dict({"text": mixed_texts}).shuffle(seed=cfg.seed)
    split = mixed_ds.train_test_split(test_size=0.02, seed=cfg.seed)
    train_ds = split["train"]
    eval_ds = split["test"]

    print("[+] Mix plan:")
    for row in mix_info:
        print(f"    {row['bucket']:10s} used={row['used']:6d} available={row['available']:6d}")
    print(f"[+] Final dataset: total={len(mixed_ds)} train={len(train_ds)} eval={len(eval_ds)}")

    json_dump(
        {
            "config": asdict(cfg),
            "mix_plan": mix_info,
            "sizes": {
                "completion": len(ds_completion),
                "mbpp": len(ds_mbpp_fmt),
                "evol": len(ds_evol_fmt),
                "magicoder": len(ds_magic_fmt),
                "py_instr": len(ds_py_fmt),
                "patch": len(ds_patch_fmt) if ds_patch_fmt is not None else 0,
                "mixed_total": len(mixed_ds),
                "train": len(train_ds),
                "eval": len(eval_ds),
            },
        },
        os.path.join(cfg.output_dir, "dataset_report.json"),
    )

    # -------------------------------------------------
    # Phase 4: Training
    # -------------------------------------------------
    bf16_ok = is_bfloat16_supported()

    training_args = make_training_args(
        TrainingArguments,
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.per_device_bs,
        gradient_accumulation_steps=cfg.grad_acc,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.lr,
        warmup_ratio=cfg.warmup_ratio,
        logging_steps=cfg.logging_steps,
        evaluation_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        weight_decay=cfg.wd,
        lr_scheduler_type="linear",
        optim="adamw_8bit",
        bf16=bf16_ok,
        fp16=not bf16_ok,
        report_to="none",
        seed=cfg.seed,
        torch_compile=False,
        neftune_noise_alpha=cfg.neftune_noise_alpha,
    )

    print("[*] Starting SFT training...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        dataset_text_field="text",
        max_seq_length=cfg.max_seq_length,
        packing=cfg.packing,
        args=training_args,
    )

    start_time = time.time()
    trainer.train()
    train_seconds = time.time() - start_time

    print(f"[+] Training finished in {train_seconds/60:.2f} minutes")

    # -------------------------------------------------
    # Save
    # -------------------------------------------------
    print(f"[*] Saving LoRA to: {cfg.output_lora}")
    model.save_pretrained(cfg.output_lora)
    tokenizer.save_pretrained(cfg.output_lora)

    json_dump(
        {
            "base_model": cfg.base_model,
            "output_lora": cfg.output_lora,
            "train_seconds": train_seconds,
        },
        os.path.join(cfg.output_dir, "train_summary.json"),
    )

    print("[+] Done.")


if __name__ == "__main__":
    main()

"""
python CodeAgent/qwen_coder_sft_v4.py  \
    --base_model Qwen/Qwen3.5-9B \
    --output_lora output/qwen9b_sft_lora \
    --output_dir output/qwen9b_sft
    --max_seq_length 4096
    --per_device_bs 4
    --grad_acc 4
    --lr 7e-6
    --epochs 1.0
    --mix_completion 42000
    --mix_mbpp 12000
    --mix_evol 8000
    --mix_magicoder 6000
    --mix_py_instr 6000
"""

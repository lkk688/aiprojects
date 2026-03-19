#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Robust SFT Trainer for Qwen Code Models (LoRA / Unsloth) — v5

New in v5 vs v4:
1. Loss Masking   — DataCollatorForCompletionOnlyLM masks all prompt/system/user
                    tokens to -100; only assistant response tokens contribute to loss.
                    Automatically disables sequence packing (incompatible with the
                    response-boundary collator).  Controllable via --no_loss_masking.
2. NEFTune        — neftune_noise_alpha is now a first-class CLI flag (--neftune_alpha).
                    Default 5.0 per the NEFTune paper.  Set to 0 to disable.
3. Multi-model    — Tested defaults documented for 7B / 14B / 27B / 32B Qwen variants.
                    Use --base_model Qwen/Qwen3.5-27B as a ready-to-run 27B example.

Design goals (unchanged from v4):
1) Completion-first pipeline to protect code completion ability.
2) Exact-quota mixing so instruction data never overwhelms completion data.
3) Strong schema handling across common Python/code datasets.
4) Multiprocessing-safe completion extraction.
5) Practical CLI configurability.

Multi-model quick-start examples
──────────────────────────────────
# Qwen3.5-7B  (fast, fits 24 GB GPU)
python CodeAgent/qwen_coder_sft_v5.py \
  --base_model Qwen/Qwen3.5-7B \
  --output_lora output/qwen7b_sft_lora \
  --output_dir  output/qwen7b_sft \
  --per_device_bs 4 --grad_acc 4 --lr 7e-6

# Qwen3.5-14B
python CodeAgent/qwen_coder_sft_v5.py \
  --base_model Qwen/Qwen3.5-14B \
  --output_lora output/qwen14b_sft_lora \
  --output_dir  output/qwen14b_sft \
  --per_device_bs 2 --grad_acc 8 --lr 5e-6

# Qwen3.5-27B  (inspired by finetune_qwen_27b.py)
python CodeAgent/qwen_coder_sft_v5.py \
  --base_model Qwen/Qwen3.5-27B \
  --output_lora output/qwen27b_sft_lora \
  --output_dir  output/qwen27b_sft \
  --per_device_bs 2 --grad_acc 8 \
  --lr 2e-4 --lora_r 16 --lora_alpha 32 \
  --neftune_alpha 5.0

# Qwen3.5-32B  (tight VRAM, reduce seq length)
python CodeAgent/qwen_coder_sft_v5.py \
  --base_model Qwen/Qwen3.5-32B \
  --output_lora output/qwen32b_sft_lora \
  --output_dir  output/qwen32b_sft \
  --per_device_bs 1 --grad_acc 16 \
  --lr 2e-4 --lora_r 16 --lora_alpha 32 \
  --max_seq_length 2048 --neftune_alpha 5.0

# Disable loss masking or NEFTune individually
python CodeAgent/qwen_coder_sft_v5.py \
  --base_model Qwen/Qwen3.5-7B \
  --output_lora output/test_lora \
  --output_dir  output/test \
  --no_loss_masking --neftune_alpha 0

  
4 new formatter functions:

format_glaive_tool_call_row() — parses glaive's USER/ASSISTANT/<functioncall> format into ChatML with <tool_call> tags
format_ultrachat_row() — HuggingFaceH4/ultrachat_200k multi-turn → apply_chat_template
format_gsm8k_row() — GSM8K step-by-step reasoning, encourages Python verification
load_private_sft_texts() — reads sft_all.jsonl and applies chat template

3 new dataset buckets (all fail-safe — gracefully skips if unavailable):

Dataset	CFG default	Mix quota
glaiveai/glaive-function-calling-v2	n_tool_calling=10000	mix_tool_calling=3000
HuggingFaceH4/ultrachat_200k	n_instruction=10000	mix_instruction=2500
openai/gsm8k	n_math_logic=8000	mix_math_logic=2000
sft_all.jsonl (private)	path via CLI	mix_private_sft=8000

# Step 1: generate private SFT from your repos
python CodeAgent/prepare_private_datav3.py \
    --github_repos https://github.com/you/repo1 https://github.com/you/repo2 \
    --local_dirs /path/to/myrepo \
    --output_dir data/private_v3

# Step 2: train — Qwen3.5-9B (recommended, fits any 40 GiB+ GPU)
python CodeAgent/qwen_coder_sft_v5.py \
    --base_model Qwen/Qwen3.5-9B \
    --output_lora output/qwen9b_sft_lora \
    --output_dir  output/qwen9b_sft \
    --per_device_bs 4 --grad_acc 4 \
    --lr 2e-4 --lora_r 16 --lora_alpha 32 \
    --neftune_alpha 5.0 \
    --private_sft_jsonl data/private_v3/sft_all.jsonl \
    --mix_private_sft 8000

# Step 2 (alt): Qwen3.5-27B — requires 90+ GiB GPU; use per_device_bs=1
python CodeAgent/qwen_coder_sft_v5.py \
    --base_model Qwen/Qwen3.5-27B \
    --output_lora output/qwen27b_sft_lora \
    --output_dir  output/qwen27b_sft \
    --per_device_bs 1 --grad_acc 16 --max_seq_length 2048 \
    --lr 2e-4 --lora_r 16 --lora_alpha 32 \
    --neftune_alpha 5.0 \
    --private_sft_jsonl data/private_v3/sft_all.jsonl \
    --mix_private_sft 8000
"""

import os
import sys

# Reduce CUDA allocator fragmentation — set before any torch/CUDA import.
# Particularly important for large models (27B+) where many temporary tensors
# cause fragmentation that triggers OOM even when enough total VRAM is free.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Tuple
from unsloth import FastLanguageModel, is_bfloat16_supported
# -----------------------------------------
# Environment guards
# -----------------------------------------
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

import torch


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
    # DataCollatorForCompletionOnlyLM is NOT imported here because its location
    # changed across TRL versions; it is resolved lazily in
    # build_response_mask_collator() with multi-path fallback.
    try:
        from unsloth import FastLanguageModel, is_bfloat16_supported
        from trl import SFTTrainer
        from transformers import TrainingArguments
        from datasets import load_dataset, Dataset
        return (FastLanguageModel, is_bfloat16_supported,
                SFTTrainer, TrainingArguments, load_dataset, Dataset)
    except Exception:
        install_dependencies()
        from unsloth import FastLanguageModel, is_bfloat16_supported
        from trl import SFTTrainer
        from transformers import TrainingArguments
        from datasets import load_dataset, Dataset
        return (FastLanguageModel, is_bfloat16_supported,
                SFTTrainer, TrainingArguments, load_dataset, Dataset)


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
    """Compatibility shim across transformer versions."""
    import inspect
    valid_params = set(inspect.signature(TrainingArguments.__init__).parameters)
    # evaluation_strategy → eval_strategy rename
    if "evaluation_strategy" in kwargs and "evaluation_strategy" not in valid_params:
        kwargs["eval_strategy"] = kwargs.pop("evaluation_strategy")
    # logging_strategy → log_strategy rename
    if "logging_strategy" in kwargs and "logging_strategy" not in valid_params:
        kwargs["log_strategy"] = kwargs.pop("logging_strategy")
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
    r"^\s*(Sure|Here(?:'|')s|Here is|Below is|Certainly|Absolutely).*?\n",
    re.IGNORECASE,
)


def strip_to_code_only(text: str) -> str:
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
    return sum(int(h in t) for h in hints) >= 1


def safe_parse_python(text: str) -> bool:
    if not isinstance(text, str) or not text.strip():
        return False
    try:
        ast.parse(text)
        return True
    except SyntaxError:
        return False


def trim_long_code_sample(code: str, max_chars: int) -> str:
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
            texts.append(json.dumps(
                {"user": prompt_code_completion(prefix), "assistant": suffix},
                ensure_ascii=False,
            ))
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


def pre_tokenize_dataset(ds, tokenizer, max_seq_length: int, num_proc: int = 4):
    """
    Tokenize a Dataset whose sole column is 'text' (pre-formatted ChatML strings)
    into input_ids / attention_mask / labels.

    This bypasses TRL's internal tokenization entirely and therefore works
    identically across all TRL versions regardless of the tokenizer /
    processing_class API rename that happened in TRL 0.22.

    For VL processors (Qwen3-VL, LLaVA, etc.) unsloth patches __call__ to
    intercept vision inputs; calling tokenizer(text_batch) positionally maps
    the batch to 'images=' in the VL processor signature, crashing with
    "Incorrect image source".  We extract the underlying text-only tokenizer
    via tokenizer.tokenizer (present on all HF processor classes) to bypass
    the vision pipeline entirely.  For plain text tokenizers the attribute
    lookup falls back to the tokenizer itself, so this is safe in all cases.
    """
    # Unwrap VL processor → plain text tokenizer
    text_tok = getattr(tokenizer, "tokenizer", tokenizer)

    def _tok(batch):
        enc = text_tok(
            batch["text"],
            truncation=True,
            max_length=max_seq_length,
            padding=False,
            return_attention_mask=True,
        )
        enc["labels"] = [list(ids) for ids in enc["input_ids"]]
        return enc

    return ds.map(
        _tok,
        batched=True,
        remove_columns=["text"],
        num_proc=min(num_proc, 8),
        desc="Tokenizing",
    )


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

    return make_chat_text(tokenizer, prompt_instruction_to_code(problem), code)


def format_evol_row(ex: Dict[str, Any], tokenizer, max_code_chars: int) -> Optional[str]:
    instr = choose_first_present(ex, ["instruction", "problem", "prompt"], "")
    out = choose_first_present(ex, ["output", "solution", "answer"], "")
    if not instr or not out:
        return None

    code = trim_long_code_sample(strip_to_code_only(out), max_code_chars)
    if not looks_like_python_code(code):
        return None

    return make_chat_text(tokenizer, prompt_instruction_to_code(instr), code)


def format_py_instr_row(ex: Dict[str, Any], tokenizer, max_code_chars: int) -> Optional[str]:
    instr = choose_first_present(ex, ["instruction", "prompt", "problem"], "")
    sysm = choose_first_present(ex, ["system"], "")
    out = choose_first_present(ex, ["output", "answer", "solution"], "")

    if instr and out:
        code = trim_long_code_sample(strip_to_code_only(out), max_code_chars)
        if looks_like_python_code(code):
            return make_chat_text(tokenizer, prompt_instruction_to_code(instr, sysm), code)

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
    return make_chat_text(tokenizer, prompt_mbpp(problem, entry_point=entry), code)


def format_patch_row(ex: Dict[str, Any], tokenizer) -> Optional[str]:
    prompt = ex.get("prompt", "") or ""
    patch = (ex.get("patch", "") or "").strip()
    if not prompt or not patch:
        return None
    if ("diff --git" not in patch) and (not patch.startswith("---")):
        return None
    return make_chat_text(tokenizer, prompt, patch + "\n")


# =====================================================================
# Tool calling / Instruction following / Math formatters  (v5 NEW)
# =====================================================================

def _glaive_to_chattext(system: str, chat: str, tokenizer) -> Optional[str]:
    """
    Parse glaive-function-calling-v2 format into ChatML.

    glaive format example:
        SYSTEM: You are a helpful assistant with access to ...
        USER: What is the weather ...
        ASSISTANT: <functioncall> {"name": "get_weather", ...}
        FUNCTION RESPONSE: {"temperature": ...}
        ASSISTANT: The weather is ...
    """
    import re
    messages = []
    if system and system.strip():
        messages.append({"role": "system", "content": system.strip()})

    # Split on turn markers; keep the marker in each chunk
    parts = re.split(r"(USER:|ASSISTANT:|FUNCTION RESPONSE:)", chat)
    role = None
    for part in parts:
        part = part.strip()
        if part == "USER:":
            role = "user"
        elif part == "ASSISTANT:":
            role = "assistant"
        elif part == "FUNCTION RESPONSE:":
            role = "tool"
        elif part and role:
            # Convert <functioncall> JSON to <tool_call> tag
            content = re.sub(
                r"<functioncall>\s*(\{.*?\})\s*",
                lambda m: f"<tool_call>{m.group(1)}</tool_call>",
                part,
                flags=re.DOTALL,
            )
            messages.append({"role": role, "content": content.strip()})
            role = None

    if len(messages) < 2:
        return None
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    except Exception:
        return None


def format_glaive_tool_call_row(ex: Dict[str, Any], tokenizer) -> Optional[str]:
    system = ex.get("system", "") or ""
    chat   = ex.get("chat",   "") or ""
    if not chat.strip():
        return None
    return _glaive_to_chattext(system, chat, tokenizer)


def format_ultrachat_row(ex: Dict[str, Any], tokenizer, max_chars: int) -> Optional[str]:
    """HuggingFaceH4/ultrachat_200k — already a list of {role, content} dicts."""
    messages = ex.get("messages", []) or []
    if not messages:
        return None
    # Trim very long turns
    trimmed = []
    for msg in messages:
        content = (msg.get("content", "") or "")[:max_chars]
        trimmed.append({"role": msg.get("role", "user"), "content": content})
    try:
        return tokenizer.apply_chat_template(
            trimmed, tokenize=False, add_generation_prompt=False
        )
    except Exception:
        return None


def format_gsm8k_row(ex: Dict[str, Any], tokenizer) -> Optional[str]:
    """
    openai/gsm8k — question + chain-of-thought answer.
    Strips the '#### N' final-answer marker and keeps the reasoning steps.
    Encourages the model to optionally verify with Python.
    """
    question = (ex.get("question", "") or "").strip()
    answer   = (ex.get("answer",   "") or "").strip()
    if not question or not answer:
        return None

    # Strip the '#### <number>' final answer line
    answer_body = re.sub(r"\s*####.*$", "", answer, flags=re.MULTILINE).strip()

    system_msg = (
        "You are a precise math tutor. Solve step by step. "
        "You may write Python code to verify your answer."
    )
    user_msg   = question
    asst_msg   = answer_body

    messages = [
        {"role": "system",    "content": system_msg},
        {"role": "user",      "content": user_msg},
        {"role": "assistant", "content": asst_msg},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    except Exception:
        return None


def load_private_sft_texts(
    jsonl_path: str,
    tokenizer,
    max_chars: int,
    min_len: int,
) -> List[str]:
    """
    Load sft_all.jsonl produced by prepare_private_datav3.py.
    Each line: {"user": "...", "assistant": "...", "type": "...", "source": "..."}
    """
    if not jsonl_path or not os.path.isfile(jsonl_path):
        return []
    texts: List[str] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            user_msg = (obj.get("user", "") or "")[:max_chars]
            asst_msg = (obj.get("assistant", "") or "")[:max_chars]
            if not user_msg or not asst_msg:
                continue
            messages = [
                {"role": "user",      "content": user_msg},
                {"role": "assistant", "content": asst_msg},
            ]
            try:
                t = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
            except Exception:
                t = make_chat_text(tokenizer, user_msg, asst_msg)
            if t and len(t) >= min_len:
                texts.append(t)
    print(f"[+] Loaded {len(texts)} private SFT samples from {jsonl_path}")
    return texts


# =====================================================================
# Loss Masking  (v5 NEW)
# =====================================================================
def get_response_template_ids(tokenizer, response_template: str) -> List[int]:
    """
    Tokenize the response template string to obtain its token IDs.

    For Qwen ChatML the default template is '<|im_start|>assistant\\n'.
    Special tokens like <|im_start|> are always encoded consistently, so
    tokenizing in isolation is safe.
    """
    ids = tokenizer.encode(response_template, add_special_tokens=False)
    return ids


def _import_completion_collator():
    """
    Resolve DataCollatorForCompletionOnlyLM across TRL versions.

    TRL moved the class between releases:
      < 0.15  : from trl import DataCollatorForCompletionOnlyLM
      0.15–0.21: from trl.trainer import DataCollatorForCompletionOnlyLM
      0.22+   : from trl.trainer.utils import DataCollatorForCompletionOnlyLM
                 (also re-exported by trl.trainer in some builds)
    """
    candidates = [
        ("trl",               "DataCollatorForCompletionOnlyLM"),
        ("trl.trainer",       "DataCollatorForCompletionOnlyLM"),
        ("trl.trainer.utils", "DataCollatorForCompletionOnlyLM"),
        ("trl.data_utils",    "DataCollatorForCompletionOnlyLM"),
    ]
    for module_path, attr in candidates:
        try:
            import importlib
            mod = importlib.import_module(module_path)
            cls = getattr(mod, attr, None)
            if cls is not None:
                return cls
        except ImportError:
            continue
    return None


def build_response_mask_collator(tokenizer, response_template: str):
    """
    Build a DataCollatorForCompletionOnlyLM that sets all non-response
    token labels to -100, so the loss is computed only on the assistant
    response portion of each training example.

    Requirements
    ────────────
    - trl (any version with DataCollatorForCompletionOnlyLM)
    - packing must be False in the SFTTrainer (sequences need individual
      boundaries to detect the response template reliably)

    Falls back to None (full-sequence loss) if the class is unavailable
    or the template tokenises to an empty sequence.
    """
    DataCollatorForCompletionOnlyLM = _import_completion_collator()
    if DataCollatorForCompletionOnlyLM is None:
        print("[WARN] DataCollatorForCompletionOnlyLM not found in any trl submodule. "
              "Falling back to full-sequence loss.")
        return None

    ids = get_response_template_ids(tokenizer, response_template)
    if not ids:
        print(f"[WARN] Response template {repr(response_template)} tokenises to empty. "
              "Falling back to full-sequence loss.")
        return None

    print(f"[*] Loss masking ON  — response template : {repr(response_template)}")
    print(f"[*]                  — template token IDs: {ids}")

    collator = DataCollatorForCompletionOnlyLM(
        response_template=ids,
        tokenizer=tokenizer,
    )
    return collator


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

    # Defaults tuned for Qwen3.5-9B on a single 80-94 GiB GPU.
    # For 27B models reduce to per_device_bs=1 and increase grad_acc accordingly.
    per_device_bs: int = 2
    grad_acc: int = 8
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

    # ── NEFTune (v5: now a first-class CLI flag) ──────────────────────
    # Adds uniform noise to embedding vectors during the forward pass,
    # improving generalisation especially on small instruction datasets.
    # Set to 0.0 to disable.  Recommended range: 5–15.
    neftune_noise_alpha: float = 5.0

    # ── Loss Masking (v5 NEW) ─────────────────────────────────────────
    # When True, only the assistant response tokens contribute to the
    # cross-entropy loss.  Prompt / system / user tokens are masked to
    # -100.  Requires packing=False (set automatically).
    use_loss_masking: bool = True
    # ChatML assistant-turn start marker used by all Qwen2/Qwen3 models.
    # Override with --response_template if you use a different format.
    response_template: str = "<|im_start|>assistant\n"

    # raw dataset subsample sizes
    n_magicoder: int = 12000
    n_evol: int = 12000
    n_py_instr: int = 60000
    n_mbpp_train: int = 2000
    n_tool_calling: int = 10000   # glaiveai/glaive-function-calling-v2
    n_math_logic: int = 8000      # openai/gsm8k
    n_instruction: int = 10000    # HuggingFaceH4/ultrachat_200k

    # final mix quotas
    mix_completion: int = 42000
    mix_mbpp: int = 12000
    mix_evol: int = 8000
    mix_magicoder: int = 6000
    mix_py_instr: int = 6000
    mix_patch: int = 0
    mix_tool_calling: int = 3000   # agent tool-use
    mix_math_logic: int = 2000     # step-by-step reasoning
    mix_instruction: int = 2500    # general instruction following
    mix_private_sft: int = 8000    # code navigation (prepare_private_datav3.py)
    max_mixed: int = 100000

    # completion extraction
    completion_min_lines: int = 12
    completion_min_suffix_chars: int = 60
    completion_max_pairs_per_code: int = 2

    # sample shaping
    max_code_chars: int = 12000
    min_text_len: int = 32

    # data
    patch_jsonl: str = "data/patch_sft.jsonl"
    private_sft_jsonl: str = ""   # sft_all.jsonl from prepare_private_datav3.py

    # multiprocessing
    num_proc: int = max(1, mp.cpu_count() - 2)


# =====================================================================
# CLI
# =====================================================================
def parse_args() -> CFG:
    parser = argparse.ArgumentParser(
        description="Robust Qwen SFT Trainer v5 (loss masking + NEFTune)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Model ────────────────────────────────────────────────────────
    parser.add_argument("--base_model", type=str, required=True,
                        help="HF model id, e.g. Qwen/Qwen3.5-27B")
    parser.add_argument("--output_lora", type=str, default="qwen_sft_lora")
    parser.add_argument("--output_dir",  type=str, default="outputs_qwen_sft")

    # ── General ──────────────────────────────────────────────────────
    parser.add_argument("--seed",           type=int,   default=3407)
    parser.add_argument("--max_seq_length", type=int,   default=4096)
    parser.add_argument("--load_in_4bit",    action="store_true")
    parser.add_argument("--no_load_in_4bit", action="store_true",
                        help="Disable 4-bit quantisation (requires more VRAM)")

    # ── Training hyperparameters ─────────────────────────────────────
    parser.add_argument("--per_device_bs",  type=int,   default=2)
    parser.add_argument("--grad_acc",       type=int,   default=8)
    parser.add_argument("--epochs",         type=float, default=1.0)
    parser.add_argument("--lr",             type=float, default=7.0e-6)
    parser.add_argument("--wd",             type=float, default=0.01)
    parser.add_argument("--warmup_ratio",   type=float, default=0.03)
    parser.add_argument("--eval_steps",     type=int,   default=250)
    parser.add_argument("--save_steps",     type=int,   default=500)
    parser.add_argument("--logging_steps",  type=int,   default=20)

    # ── LoRA ─────────────────────────────────────────────────────────
    parser.add_argument("--lora_r",       type=int,   default=16)
    parser.add_argument("--lora_alpha",   type=int,   default=16,
                        help="LoRA alpha.  27B/32B typically use 32 (= 2×r).")
    parser.add_argument("--lora_dropout", type=float, default=0.0)

    # ── NEFTune (v5) ─────────────────────────────────────────────────
    parser.add_argument("--neftune_alpha", type=float, default=5.0,
                        help="NEFTune noise alpha.  Adds uniform noise to "
                             "embeddings during training for better "
                             "generalisation.  Set to 0 to disable.")

    # ── Loss Masking (v5 NEW) ─────────────────────────────────────────
    parser.add_argument("--use_loss_masking", action="store_true", default=True,
                        help="Compute loss only on assistant response tokens "
                             "(default: enabled).")
    parser.add_argument("--no_loss_masking", action="store_true",
                        help="Disable response-only loss masking and train on "
                             "the full sequence.")
    parser.add_argument("--response_template", type=str,
                        default="<|im_start|>assistant\n",
                        help="Token string that marks the start of each "
                             "assistant turn.  Qwen ChatML default works for "
                             "all Qwen2/Qwen3 family models.")

    # ── Dataset mix ──────────────────────────────────────────────────
    parser.add_argument("--n_magicoder",  type=int, default=12000)
    parser.add_argument("--n_evol",       type=int, default=12000)
    parser.add_argument("--n_py_instr",   type=int, default=60000)
    parser.add_argument("--n_mbpp_train", type=int, default=2000)

    parser.add_argument("--mix_completion", type=int, default=42000)
    parser.add_argument("--mix_mbpp",       type=int, default=12000)
    parser.add_argument("--mix_evol",       type=int, default=8000)
    parser.add_argument("--mix_magicoder",  type=int, default=6000)
    parser.add_argument("--mix_py_instr",   type=int, default=6000)
    parser.add_argument("--mix_patch",      type=int, default=0)
    parser.add_argument("--max_mixed",      type=int, default=80000)

    parser.add_argument("--completion_min_lines",          type=int, default=12)
    parser.add_argument("--completion_min_suffix_chars",   type=int, default=60)
    parser.add_argument("--completion_max_pairs_per_code", type=int, default=2)

    parser.add_argument("--patch_jsonl",  type=str, default="data/patch_sft.jsonl")
    parser.add_argument("--num_proc",     type=int, default=max(1, mp.cpu_count() - 2))
    parser.add_argument("--max_code_chars", type=int, default=12000)
    parser.add_argument("--min_text_len",   type=int, default=32)

    # ── New dataset buckets ───────────────────────────────────────────
    parser.add_argument("--n_tool_calling", type=int, default=10000,
                        help="Rows from glaive-function-calling-v2.")
    parser.add_argument("--n_math_logic",   type=int, default=8000,
                        help="Rows from openai/gsm8k.")
    parser.add_argument("--n_instruction",  type=int, default=10000,
                        help="Rows from HuggingFaceH4/ultrachat_200k.")
    parser.add_argument("--mix_tool_calling", type=int, default=3000)
    parser.add_argument("--mix_math_logic",   type=int, default=2000)
    parser.add_argument("--mix_instruction",  type=int, default=2500)
    parser.add_argument("--mix_private_sft",  type=int, default=8000,
                        help="Quota for code-navigation SFT from prepare_private_datav3.py.")
    parser.add_argument("--private_sft_jsonl", type=str, default="",
                        help="Path to sft_all.jsonl from prepare_private_datav3.py.")

    args = parser.parse_args()

    # Resolve load_in_4bit
    load_in_4bit = True
    if args.no_load_in_4bit:
        load_in_4bit = False
    elif args.load_in_4bit:
        load_in_4bit = True

    # Resolve loss masking
    use_loss_masking = not args.no_loss_masking

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
        neftune_noise_alpha=args.neftune_alpha,
        use_loss_masking=use_loss_masking,
        response_template=args.response_template,
        n_magicoder=args.n_magicoder,
        n_evol=args.n_evol,
        n_py_instr=args.n_py_instr,
        n_mbpp_train=args.n_mbpp_train,
        n_tool_calling=args.n_tool_calling,
        n_math_logic=args.n_math_logic,
        n_instruction=args.n_instruction,
        mix_completion=args.mix_completion,
        mix_mbpp=args.mix_mbpp,
        mix_evol=args.mix_evol,
        mix_magicoder=args.mix_magicoder,
        mix_py_instr=args.mix_py_instr,
        mix_patch=args.mix_patch,
        mix_tool_calling=args.mix_tool_calling,
        mix_math_logic=args.mix_math_logic,
        mix_instruction=args.mix_instruction,
        mix_private_sft=args.mix_private_sft,
        max_mixed=args.max_mixed,
        completion_min_lines=args.completion_min_lines,
        completion_min_suffix_chars=args.completion_min_suffix_chars,
        completion_max_pairs_per_code=args.completion_max_pairs_per_code,
        patch_jsonl=args.patch_jsonl,
        private_sft_jsonl=args.private_sft_jsonl,
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

    (FastLanguageModel, is_bfloat16_supported,
     SFTTrainer, TrainingArguments, load_dataset, Dataset) = lazy_import_training_stack()

    print(f"[*] Config:\n{json.dumps(asdict(cfg), ensure_ascii=False, indent=2)}")
    json_dump(asdict(cfg), os.path.join(cfg.output_dir, "config.json"))

    # -------------------------------------------------
    # Load model / tokenizer
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
    ds_magic = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train",
                            trust_remote_code=False)
    if len(ds_magic) > cfg.n_magicoder:
        ds_magic = ds_magic.shuffle(seed=cfg.seed).select(range(cfg.n_magicoder))

    ds_evol = load_dataset("nickrosh/Evol-Instruct-Code-80k-v1", split="train",
                           trust_remote_code=False)
    if len(ds_evol) > cfg.n_evol:
        ds_evol = ds_evol.shuffle(seed=cfg.seed).select(range(cfg.n_evol))

    ds_py = load_dataset("jtatman/python-code-dataset-500k", split="train",
                         trust_remote_code=False)
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

    # ── Tool calling (glaiveai/glaive-function-calling-v2) ────────────
    ds_glaive = None
    if cfg.n_tool_calling > 0:
        try:
            ds_glaive = load_dataset(
                "glaiveai/glaive-function-calling-v2", split="train",
                trust_remote_code=False,
            )
            if len(ds_glaive) > cfg.n_tool_calling:
                ds_glaive = ds_glaive.shuffle(seed=cfg.seed).select(range(cfg.n_tool_calling))
            print(f"[OK] glaive tool-calling: {len(ds_glaive)} rows")
        except Exception as e:
            print(f"[WARN] glaive-function-calling-v2 unavailable: {e}")
            ds_glaive = None

    # ── Instruction following (HuggingFaceH4/ultrachat_200k) ─────────
    ds_ultrachat = None
    if cfg.n_instruction > 0:
        try:
            ds_ultrachat = load_dataset(
                "HuggingFaceH4/ultrachat_200k", split="train_sft",
                trust_remote_code=False,
            )
            if len(ds_ultrachat) > cfg.n_instruction:
                ds_ultrachat = ds_ultrachat.shuffle(seed=cfg.seed).select(range(cfg.n_instruction))
            print(f"[OK] ultrachat_200k instruction: {len(ds_ultrachat)} rows")
        except Exception as e:
            print(f"[WARN] ultrachat_200k unavailable: {e}")
            ds_ultrachat = None

    # ── Math / logic reasoning (openai/gsm8k) ─────────────────────────
    ds_gsm8k = None
    if cfg.n_math_logic > 0:
        try:
            ds_gsm8k = load_dataset(
                "openai/gsm8k", "main", split="train",
                trust_remote_code=False,
            )
            if len(ds_gsm8k) > cfg.n_math_logic:
                ds_gsm8k = ds_gsm8k.shuffle(seed=cfg.seed).select(range(cfg.n_math_logic))
            print(f"[OK] gsm8k math/logic: {len(ds_gsm8k)} rows")
        except Exception as e:
            print(f"[WARN] gsm8k unavailable: {e}")
            ds_gsm8k = None

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

    completion_pairs = (
        ds_completion_json["text_json"]
        if "text_json" in ds_completion_json.column_names else []
    )
    completion_texts = []
    for item in completion_pairs:
        try:
            obj = json.loads(item)
            user = obj["user"]
            assistant = trim_long_code_sample(obj["assistant"], cfg.max_code_chars)
            completion_texts.append(make_chat_text(tokenizer, user, assistant))
        except Exception:
            continue

    completion_texts = [
        t for t in completion_texts
        if isinstance(t, str) and len(t) >= cfg.min_text_len
    ]
    completion_texts = dedupe_texts_preserve_order(completion_texts)
    ds_completion = Dataset.from_dict({"text": completion_texts})

    print(f"[+] completion bucket size: {len(ds_completion)}")

    if len(ds_completion) == 0:
        raise RuntimeError(
            "Completion bucket is empty. Aborting because completion must dominate this SFT."
        )

    # -------------------------------------------------
    # Phase 2: Format SFT instruction buckets
    # -------------------------------------------------
    print("[*] Formatting SFT instruction buckets...")

    magic_texts = [
        t for ex in ds_magic
        if (t := format_magicoder_row(ex, tokenizer, cfg.max_code_chars))
        and len(t) >= cfg.min_text_len
    ]
    evol_texts = [
        t for ex in ds_evol
        if (t := format_evol_row(ex, tokenizer, cfg.max_code_chars))
        and len(t) >= cfg.min_text_len
    ]
    py_texts = [
        t for ex in ds_py
        if (t := format_py_instr_row(ex, tokenizer, cfg.max_code_chars))
        and len(t) >= cfg.min_text_len
    ]
    mbpp_texts = [
        t for ex in ds_mbpp
        if (t := format_mbpp_row(ex, tokenizer, cfg.max_code_chars))
        and len(t) >= cfg.min_text_len
    ]
    patch_texts = []
    if ds_patch is not None:
        patch_texts = [
            t for ex in ds_patch
            if (t := format_patch_row(ex, tokenizer))
            and len(t) >= cfg.min_text_len
        ]

    magic_texts  = dedupe_texts_preserve_order(magic_texts)
    evol_texts   = dedupe_texts_preserve_order(evol_texts)
    py_texts     = dedupe_texts_preserve_order(py_texts)
    mbpp_texts   = dedupe_texts_preserve_order(mbpp_texts)
    patch_texts  = dedupe_texts_preserve_order(patch_texts)

    ds_magic_fmt  = Dataset.from_dict({"text": magic_texts})
    ds_evol_fmt   = Dataset.from_dict({"text": evol_texts})
    ds_py_fmt     = Dataset.from_dict({"text": py_texts})
    ds_mbpp_fmt   = Dataset.from_dict({"text": mbpp_texts})
    ds_patch_fmt  = Dataset.from_dict({"text": patch_texts}) if patch_texts else None

    # ── Tool calling ──────────────────────────────────────────────────
    tool_call_texts: List[str] = []
    if ds_glaive is not None:
        for ex in ds_glaive:
            t = format_glaive_tool_call_row(ex, tokenizer)
            if t and len(t) >= cfg.min_text_len:
                tool_call_texts.append(t)
    tool_call_texts = dedupe_texts_preserve_order(tool_call_texts)
    ds_tool_call_fmt = Dataset.from_dict({"text": tool_call_texts})

    # ── Instruction following ─────────────────────────────────────────
    instruction_texts: List[str] = []
    if ds_ultrachat is not None:
        for ex in ds_ultrachat:
            t = format_ultrachat_row(ex, tokenizer, cfg.max_code_chars)
            if t and len(t) >= cfg.min_text_len:
                instruction_texts.append(t)
    instruction_texts = dedupe_texts_preserve_order(instruction_texts)
    ds_instruction_fmt = Dataset.from_dict({"text": instruction_texts})

    # ── Math / logic reasoning ────────────────────────────────────────
    math_texts: List[str] = []
    if ds_gsm8k is not None:
        for ex in ds_gsm8k:
            t = format_gsm8k_row(ex, tokenizer)
            if t and len(t) >= cfg.min_text_len:
                math_texts.append(t)
    math_texts = dedupe_texts_preserve_order(math_texts)
    ds_math_fmt = Dataset.from_dict({"text": math_texts})

    # ── Private SFT (code navigation) ────────────────────────────────
    private_texts = load_private_sft_texts(
        cfg.private_sft_jsonl, tokenizer, cfg.max_code_chars, cfg.min_text_len
    )
    private_texts = dedupe_texts_preserve_order(private_texts)
    ds_private_fmt = Dataset.from_dict({"text": private_texts})

    print("[+] bucket sizes after formatting:")
    print(f"    completion   : {len(ds_completion)}")
    print(f"    mbpp         : {len(ds_mbpp_fmt)}")
    print(f"    evol         : {len(ds_evol_fmt)}")
    print(f"    magicoder    : {len(ds_magic_fmt)}")
    print(f"    py_instr     : {len(ds_py_fmt)}")
    print(f"    patch        : {len(ds_patch_fmt) if ds_patch_fmt is not None else 0}")
    print(f"    tool_calling : {len(ds_tool_call_fmt)}")
    print(f"    instruction  : {len(ds_instruction_fmt)}")
    print(f"    math_logic   : {len(ds_math_fmt)}")
    print(f"    private_sft  : {len(ds_private_fmt)}")

    # -------------------------------------------------
    # Phase 3: Fixed-count exact quota mixing
    # -------------------------------------------------
    def sample_bucket_texts(ds, k: int, seed: int) -> List[str]:
        if ds is None or len(ds) == 0 or k <= 0:
            return []
        rng = random.Random(seed)
        if len(ds) >= k:
            return list(ds.shuffle(seed=seed).select(range(k))["text"])
        out = list(ds["text"])
        while len(out) < k:
            out.append(ds[rng.randrange(len(ds))]["text"])
        return out[:k]

    mix_plan = [
        ("completion",   ds_completion,      min(cfg.mix_completion,   len(ds_completion))),
        ("mbpp",         ds_mbpp_fmt,         min(cfg.mix_mbpp,         len(ds_mbpp_fmt))),
        ("evol",         ds_evol_fmt,         min(cfg.mix_evol,         len(ds_evol_fmt))),
        ("magicoder",    ds_magic_fmt,        min(cfg.mix_magicoder,    len(ds_magic_fmt))),
        ("py_instr",     ds_py_fmt,           min(cfg.mix_py_instr,     len(ds_py_fmt))),
        ("tool_calling", ds_tool_call_fmt,    min(cfg.mix_tool_calling, len(ds_tool_call_fmt))),
        ("math_logic",   ds_math_fmt,         min(cfg.mix_math_logic,   len(ds_math_fmt))),
        ("instruction",  ds_instruction_fmt,  min(cfg.mix_instruction,  len(ds_instruction_fmt))),
        ("private_sft",  ds_private_fmt,      min(cfg.mix_private_sft,  len(ds_private_fmt))),
    ]
    if ds_patch_fmt is not None and cfg.mix_patch > 0:
        mix_plan.append(("patch", ds_patch_fmt, min(cfg.mix_patch, len(ds_patch_fmt))))

    total_target = sum(k for _, _, k in mix_plan)
    if total_target > cfg.max_mixed:
        overflow = total_target - cfg.max_mixed
        # Shrink least-critical buckets first; preserve completion and private_sft
        shrink_order = [
            "py_instr", "magicoder", "evol", "instruction",
            "math_logic", "tool_calling", "mbpp", "patch", "private_sft",
        ]
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
            "bucket":    name,
            "target":    k,
            "available": len(ds) if ds is not None else 0,
            "used":      len(picked),
        })

    mixed_texts = dedupe_texts_preserve_order(mixed_texts)
    random.Random(cfg.seed).shuffle(mixed_texts)

    mixed_ds = Dataset.from_dict({"text": mixed_texts}).shuffle(seed=cfg.seed)
    split    = mixed_ds.train_test_split(test_size=0.02, seed=cfg.seed)
    train_ds = split["train"]
    eval_ds  = split["test"]

    print("[+] Mix plan:")
    for row in mix_info:
        print(f"    {row['bucket']:10s} used={row['used']:6d} available={row['available']:6d}")
    print(f"[+] Final dataset: total={len(mixed_ds)} train={len(train_ds)} eval={len(eval_ds)}")

    json_dump(
        {
            "config":   asdict(cfg),
            "mix_plan": mix_info,
            "sizes": {
                "completion":  len(ds_completion),
                "mbpp":        len(ds_mbpp_fmt),
                "evol":        len(ds_evol_fmt),
                "magicoder":   len(ds_magic_fmt),
                "py_instr":    len(ds_py_fmt),
                "patch":       len(ds_patch_fmt) if ds_patch_fmt is not None else 0,
                "mixed_total": len(mixed_ds),
                "train":       len(train_ds),
                "eval":        len(eval_ds),
            },
        },
        os.path.join(cfg.output_dir, "dataset_report.json"),
    )

    # -------------------------------------------------
    # Phase 4: Build loss-masking collator (v5)
    # -------------------------------------------------
    # Loss masking: only response tokens contribute to the cross-entropy loss.
    # DataCollatorForCompletionOnlyLM scans each tokenised example for the
    # response_template token sequence and sets labels=-100 for everything
    # before it.  Packing must be disabled so per-example boundaries are intact.
    #
    # NEFTune: adds small uniform noise to the embedding layer during the
    # forward pass, regularising the model and improving instruction following.
    # Enabled via neftune_noise_alpha in TrainingArguments (HF native support).

    data_collator   = None
    effective_packing = cfg.packing

    if cfg.use_loss_masking:
        data_collator = build_response_mask_collator(tokenizer, cfg.response_template)
        if data_collator is not None:
            effective_packing = False  # required for response-boundary detection
            print("[*] Sequence packing disabled (required by loss masking).")
        else:
            print("[WARN] Loss masking collator unavailable; falling back to full-sequence loss.")
    else:
        print("[*] Loss masking disabled — training on full sequence.")

    neftune_log = (
        f"{cfg.neftune_noise_alpha} (active)" if cfg.neftune_noise_alpha > 0
        else "0.0 (disabled)"
    )
    print(f"[*] NEFTune noise alpha : {neftune_log}")

    # -------------------------------------------------
    # Phase 5: Training
    # -------------------------------------------------
    bf16_ok = is_bfloat16_supported()

    # Compute warmup_steps from ratio now that dataset size is known, avoiding
    # the deprecation warning from passing warmup_ratio to TrainingArguments.
    _steps_per_epoch = max(1, len(train_ds) // (cfg.per_device_bs * cfg.grad_acc))
    _total_steps = int(_steps_per_epoch * cfg.epochs)
    _warmup_steps = max(1, int(_total_steps * cfg.warmup_ratio))

    training_args = make_training_args(
        TrainingArguments,
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.per_device_bs,
        gradient_accumulation_steps=cfg.grad_acc,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.lr,
        warmup_steps=_warmup_steps,
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
        max_grad_norm=1.0,
        dataloader_pin_memory=False,   # avoids pinned-memory overhead on large models
        # NEFTune: natively supported by HuggingFace Trainer.
        # Adds uniform noise U(-alpha/sqrt(L*d), alpha/sqrt(L*d)) to embeddings.
        neftune_noise_alpha=cfg.neftune_noise_alpha if cfg.neftune_noise_alpha > 0 else None,
    )

    # -------------------------------------------------
    # Pre-tokenize (always).
    # TRL 0.22+ SFTDataCollator (used when data_collator=None) only accepts its
    # own conversational/text format, not pre-tokenized input_ids.  We bypass it
    # by (a) pre-tokenizing ourselves and (b) supplying DataCollatorForSeq2Seq
    # which correctly pads input_ids / attention_mask / labels.
    # VL processors (Qwen3-VL etc.) are unwrapped inside pre_tokenize_dataset.
    # -------------------------------------------------
    print("[*] Pre-tokenizing datasets...")
    # text_tok is the plain text tokenizer (unwrapped from any VL processor)
    text_tok = getattr(tokenizer, "tokenizer", tokenizer)
    train_ds = pre_tokenize_dataset(train_ds, tokenizer, cfg.max_seq_length, cfg.num_proc)
    if eval_ds is not None:
        eval_ds = pre_tokenize_dataset(eval_ds, tokenizer, cfg.max_seq_length, cfg.num_proc)
    effective_packing = False   # packing requires raw text; not applicable here

    # Build a proper collator for pre-tokenized data when loss masking is not active.
    # DataCollatorForSeq2Seq pads input_ids / attention_mask / labels and is
    # compatible with all transformers / TRL / unsloth versions.
    if data_collator is None:
        from transformers import DataCollatorForSeq2Seq
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=text_tok,
            model=None,
            label_pad_token_id=-100,
            pad_to_multiple_of=8,
        )
        print("[*] Using DataCollatorForSeq2Seq (full-sequence loss).")

    print("[*] Starting SFT training...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        max_seq_length=cfg.max_seq_length,
        packing=effective_packing,
        data_collator=data_collator,
        args=training_args,
    )

    start_time = time.time()
    trainer.train()
    train_seconds = time.time() - start_time

    print(f"[+] Training finished in {train_seconds / 60:.2f} minutes")

    # -------------------------------------------------
    # Save
    # -------------------------------------------------
    print(f"[*] Saving LoRA to: {cfg.output_lora}")
    model.save_pretrained(cfg.output_lora)
    tokenizer.save_pretrained(cfg.output_lora)

    json_dump(
        {
            "base_model":        cfg.base_model,
            "output_lora":       cfg.output_lora,
            "train_seconds":     train_seconds,
            "use_loss_masking":  cfg.use_loss_masking,
            "neftune_alpha":     cfg.neftune_noise_alpha,
            "effective_packing": effective_packing,
        },
        os.path.join(cfg.output_dir, "train_summary.json"),
    )

    print("[+] Done.")


if __name__ == "__main__":
    main()


"""
python CodeAgent/qwen_coder_sft_v5.py \
    --base_model Qwen/Qwen3.5-9B \
    --output_lora output/qwen9b_sft_lora \
    --output_dir output/qwen9b_sft \
    --per_device_bs 2 --grad_acc 8 \
    --private_sft_jsonl data/private_v1/sft_all.jsonl


# Qwen3.5-27B
python CodeAgent/qwen_coder_sft_v5.py \
  --base_model Qwen/Qwen3.5-27B \
  --output_lora output/qwen27b_sft_lora \
  --output_dir  output/qwen27b_sft \
  --per_device_bs 2 --grad_acc 8 \
  --lr 2e-4 --lora_r 16 --lora_alpha 32 \
  --neftune_alpha 5.0

Disable either feature independently
--no_loss_masking        # full-sequence loss, packing re-enabled
--neftune_alpha 0        # no NEFTune noise
"""
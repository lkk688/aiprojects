#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SFT v3.1 (fix HumanEval collapse) for Qwen2.5-Coder-14B-Instruct (Unsloth)

Key changes vs your v3:
1) REAL completion bucket (HumanEval-style):
   - We extract function-level prefix/suffix from raw Python code corpora.
   - This is the most important fix for HumanEval regressions.

2) Reduce “instruction-only” dominance:
   - We keep Evol/Magicoder but ensure they don’t swamp completion.
   - We also prevent long “script-like” outputs from overpowering completion.

3) Stronger dataset schema handling:
   - jtatman/python-code-dataset-500k can have varying columns; we handle both:
     - raw code corpora (text/code/content/...)
     - instruction/output (system/instruction/output)

4) Robust mixing:
   - Buckets are dropped if empty or missing "text"
   - We enforce target mix by sampling fixed counts per bucket (not purely probabilistic)
     so you don't accidentally train on mostly one bucket.

5) Keeps datasets==4.3.0 for Unsloth 2026.1.4 compatibility.

Notes:
- You SHOULD NOT train on HumanEval test prompts. This script does not.
- MBPP uses split="train" and uses reference "code" as supervised signal.
- Patch bucket is optional (your own patch_sft.jsonl).

"""

import os
import sys
import re
import json
import random
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# Avoid compile memory spikes
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")

import torch


# -----------------------
# Dependency management
# -----------------------
def install_dependencies():
    print("Installing dependencies...")
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
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "--no-cache-dir"] + packages)
    print("Done.")


# -----------------------
# Helpers: TrainingArguments compat
# -----------------------
def make_training_args(TrainingArguments, **kwargs):
    """
    Transformers TrainingArguments API compatibility shim.
    Some installs accept eval_strategy instead of evaluation_strategy.
    """
    try:
        return TrainingArguments(**kwargs)
    except TypeError as e:
        msg = str(e)
        if "evaluation_strategy" in kwargs and "evaluation_strategy" in msg:
            kwargs["eval_strategy"] = kwargs.pop("evaluation_strategy")
        if "logging_strategy" in kwargs and "logging_strategy" in msg:
            kwargs["log_strategy"] = kwargs.pop("logging_strategy")
        return TrainingArguments(**kwargs)


# -----------------------
# Code-only cleaning
# -----------------------
_CODE_FENCE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)

def strip_to_code_only(text: str) -> str:
    """
    Extract code from a string.
    - Prefer fenced blocks.
    - Remove obvious prose prefaces.
    - Keep result as-is (do NOT over-filter; filtering can break logic).
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    t = text.strip()
    m = _CODE_FENCE_RE.search(t)
    if m:
        t = m.group(1).strip()

    # Remove common assistant preface lines (light touch)
    t = re.sub(r"^\s*(Sure|Here(?:'|’)s|Here is|Below is).*?\n", "", t, flags=re.IGNORECASE).strip()

    # Ensure trailing newline for training stability
    return (t + "\n") if t else ""


def looks_like_python_code(text: str) -> bool:
    if not isinstance(text, str):
        return False
    t = text.strip()
    if len(t) < 20:
        return False
    return any(k in t for k in ("def ", "class ", "import ", "from "))


# -----------------------
# Chat formatting
# -----------------------
def make_chat_text(tokenizer, user: str, assistant: str) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        convo = [{"role": "user", "content": user},
                 {"role": "assistant", "content": assistant}]
        return tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
    return f"User:\n{user}\n\nAssistant:\n{assistant}"


# -----------------------
# Prompt builders
# -----------------------
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


def prompt_instruction_to_code(instr: str) -> str:
    return (
        "Write correct Python code for the following request.\n"
        "Rules:\n"
        "- Return ONLY valid Python code.\n"
        "- No markdown.\n"
        "- No explanation.\n\n"
        f"Request:\n{instr}\n"
    )


def prompt_mbpp(problem: str, entry_point: str = "") -> str:
    if entry_point:
        return (
            "Write a correct Python solution.\n"
            "Rules:\n"
            "- Return ONLY valid Python code.\n"
            "- No markdown.\n"
            "- No explanation.\n"
            f"- You MUST implement a function named `{entry_point}` exactly.\n\n"
            f"Problem:\n{problem}\n"
        )
    return (
        "Write a correct Python solution.\n"
        "Rules:\n"
        "- Return ONLY valid Python code.\n"
        "- No markdown.\n"
        "- No explanation.\n\n"
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


# -----------------------
# Dataset helpers
# -----------------------
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


def load_patch_sft_jsonl(path: str):
    from datasets import load_dataset
    if not path or not os.path.exists(path):
        return None
    ds = load_dataset("json", data_files=path, split="train")
    if "prompt" not in ds.column_names or "patch" not in ds.column_names:
        print(f"[WARN] patch_sft.jsonl missing prompt/patch fields: {ds.column_names}")
        return None
    return ds


def nonempty_text(ds, min_len: int = 32):
    if "text" not in ds.column_names:
        return ds
    return ds.filter(lambda x: isinstance(x.get("text", None), str) and len(x["text"]) >= min_len)


def drop_if_bad(name: str, ds):
    if ds is None:
        return None
    if "text" not in ds.column_names:
        print(f"[WARN] bucket '{name}' has no 'text' column -> DROPPED. columns={ds.column_names}")
        return None
    if len(ds) == 0:
        print(f"[WARN] bucket '{name}' empty -> DROPPED.")
        return None
    return ds


# -----------------------
# Completion extraction (HumanEval-style)
# -----------------------
def extract_function_completion_pairs(
    code: str,
    rng: random.Random,
    min_func_lines: int = 12,
    min_suffix_chars: int = 80,
) -> List[Tuple[str, str]]:
    """
    Convert a Python file into (prefix, suffix) completion pairs by slicing inside function bodies.

    Heuristics:
    - Find lines starting with 'def ' or 'class '.
    - For 'def', take a window until next top-level def/class (indent == 0).
    - Only keep functions with enough lines.
    - Slice somewhere in the middle of the function to create a completion task.
    """
    if not isinstance(code, str):
        return []
    text = code.strip("\n")
    if len(text) < 300:
        return []

    lines = text.splitlines()
    if len(lines) < 30:
        return []

    # collect candidate top-level def blocks
    def_starts = []
    for i, ln in enumerate(lines):
        if ln.startswith("def ") or ln.startswith("class "):
            def_starts.append(i)

    if not def_starts:
        return []

    pairs: List[Tuple[str, str]] = []

    # find blocks by next top-level def/class
    for si, start in enumerate(def_starts):
        end = def_starts[si + 1] if si + 1 < len(def_starts) else len(lines)
        block = lines[start:end]

        # prefer functions, but allow small classes that contain defs (rare)
        if not block or not (block[0].startswith("def ") or block[0].startswith("class ")):
            continue

        # must include at least some indented body lines
        if len(block) < min_func_lines:
            continue

        # For class blocks, skip unless it contains a def inside (completion inside class is okay but harder)
        if block[0].startswith("class "):
            # require at least one method
            if not any(("def " in ln and ln.startswith("    def ")) for ln in block[1:]):
                continue

        # pick a cut point inside the block, not too early
        # choose a cut between 35% and 70% of block length
        lo = max(5, int(len(block) * 0.35))
        hi = max(lo + 1, int(len(block) * 0.70))
        if hi <= lo:
            continue
        cut = rng.randint(lo, hi)

        prefix_lines = lines[:start] + block[:cut]
        suffix_lines = block[cut:] + lines[end:]  # continuation can include rest of file; OK

        prefix = "\n".join(prefix_lines).rstrip() + "\n"
        suffix = "\n".join(suffix_lines).lstrip() + "\n"

        if len(suffix.strip()) < min_suffix_chars:
            continue

        pairs.append((prefix, suffix))

        # limit per file
        if len(pairs) >= 2:
            break

    return pairs


def main():
    try:
        from unsloth import FastLanguageModel, is_bfloat16_supported
        from trl import SFTTrainer
        from transformers import TrainingArguments
        from datasets import load_dataset, Dataset
    except Exception:
        install_dependencies()
        from unsloth import FastLanguageModel, is_bfloat16_supported
        from trl import SFTTrainer
        from transformers import TrainingArguments
        from datasets import load_dataset, Dataset

    @dataclass
    class CFG:
        base_model: str = "unsloth/Qwen2.5-Coder-14B-Instruct"
        output_lora: str = "qwen_coder_lora_sft_v3_1"
        seed: int = 3407

        max_seq_length: int = 2048
        load_in_4bit: bool = True

        # H100-safe defaults
        per_device_bs: int = 4
        grad_acc: int = 4
        epochs: float = 1.0
        lr: float = 7.0e-6            # lower LR to reduce behavior drift
        wd: float = 0.01

        eval_steps: int = 250
        save_steps: int = 500

        # Raw subsampling sizes
        n_magicoder: int = 12000
        n_evol: int = 12000
        n_pycode_raw: int = 60000
        n_mbpp_train: int = 2000

        patch_jsonl: str = "data/patch_sft.jsonl"
        packing: bool = True

        # FINAL MIX TARGET COUNTS (this matters most)
        # Make completion dominate to protect HumanEval
        mix_completion: int = 42000
        mix_mbpp: int = 12000
        mix_evol: int = 8000
        mix_magicoder: int = 6000
        mix_patch: int = 0             # will be set if patch jsonl exists

        # cap
        max_mixed: int = 80000

    cfg = CFG()
    rng = random.Random(cfg.seed)
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    print(f"Loading model: {cfg.base_model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.base_model,
        max_seq_length=cfg.max_seq_length,
        dtype=None,
        load_in_4bit=cfg.load_in_4bit,
    )

    print("Adding LoRA...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=cfg.seed,
        use_rslora=False,
        loftq_config=None,
    )

    # -----------------------
    # Load datasets
    # -----------------------
    print("Loading datasets...")

    # Magicoder (instruction->solution)
    magic = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train", trust_remote_code=False)
    if len(magic) > cfg.n_magicoder:
        magic = magic.shuffle(seed=cfg.seed).select(range(cfg.n_magicoder))

    # Evol Instruct Code
    evol = load_dataset("nickrosh/Evol-Instruct-Code-80k-v1", split="train", trust_remote_code=False)
    if len(evol) > cfg.n_evol:
        evol = evol.shuffle(seed=cfg.seed).select(range(cfg.n_evol))

    # Python code corpus (raw)
    pycode = load_dataset("jtatman/python-code-dataset-500k", split="train", trust_remote_code=False)
    if len(pycode) > cfg.n_pycode_raw:
        pycode = pycode.shuffle(seed=cfg.seed).select(range(cfg.n_pycode_raw))

    # MBPP train
    mbpp = load_dataset("mbpp", split="train", trust_remote_code=False)
    if len(mbpp) > cfg.n_mbpp_train:
        mbpp = mbpp.shuffle(seed=cfg.seed).select(range(cfg.n_mbpp_train))

    # Patch SFT
    patch_ds = load_patch_sft_jsonl(cfg.patch_jsonl)
    if patch_ds is None or len(patch_ds) == 0:
        patch_ds = None
        print("[WARN] No patch_sft.jsonl found or empty; patch bucket disabled.")
    else:
        cfg.mix_patch = min(4000, len(patch_ds))
        print(f"[OK] Loaded patch SFT: {cfg.patch_jsonl} ({len(patch_ds)} rows), mix_patch={cfg.mix_patch}")

    # -----------------------
    # Format buckets to {"text"}
    # -----------------------
    def fmt_magicoder(examples):
        outs = []
        problems = examples.get("problem", [])
        solutions = examples.get("solution", [])
        langs = examples.get("lang", None)

        for i, (p, s) in enumerate(zip(problems, solutions)):
            if not p or not s:
                continue
            if langs is not None:
                lg = langs[i]
                if lg and ("python" not in str(lg).lower()):
                    continue

            s_code = strip_to_code_only(s)
            if not looks_like_python_code(s_code):
                continue

            user = prompt_instruction_to_code(p)
            outs.append(make_chat_text(tokenizer, user, s_code))
        return {"text": outs}

    def fmt_evol(examples):
        outs = []
        instrs = examples.get("instruction", [])
        outs_raw = examples.get("output", [])
        for instr, out in zip(instrs, outs_raw):
            if not instr or not out:
                continue
            out_code = strip_to_code_only(out)
            if not looks_like_python_code(out_code):
                continue
            user = prompt_instruction_to_code(instr)
            outs.append(make_chat_text(tokenizer, user, out_code))
        return {"text": outs}

    def fmt_mbpp_row(ex):
        problem = ex.get("text", "") or ""
        tests = ex.get("test_list", []) or []
        code = strip_to_code_only(ex.get("code", "") or "")
        if not problem or not code:
            return None
        if not looks_like_python_code(code):
            return None
        entry = infer_entry_point_from_mbpp_tests(tests, problem)
        user = prompt_mbpp(problem, entry_point=entry)
        return make_chat_text(tokenizer, user, code)

    def fmt_patch(examples):
        outs = []
        prompts = examples.get("prompt", [])
        patches = examples.get("patch", [])
        for p, d in zip(prompts, patches):
            if not p or not d:
                continue
            d = d.strip() + "\n"
            if ("diff --git" not in d) and (not d.startswith("---")):
                continue
            outs.append(make_chat_text(tokenizer, p, d))
        return {"text": outs}

    def get_pycode_raw_strings(examples) -> List[str]:
        # If it looks like instruction/output, treat separately (we will NOT use it as completion)
        if "instruction" in examples and "output" in examples:
            return []

        for k in ["code", "content", "text", "file_content", "source", "src"]:
            if k in examples:
                vals = examples.get(k, [])
                if isinstance(vals, list):
                    return vals
        return []

    # Map instruction buckets
    magic = magic.map(fmt_magicoder, batched=True, remove_columns=magic.column_names)
    evol  = evol.map(fmt_evol, batched=True, remove_columns=evol.column_names)

    # MBPP row-by-row (schema is stable but safer this way)
    from datasets import Dataset
    mbpp_texts = []
    for ex in mbpp:
        t = fmt_mbpp_row(ex)
        if t is not None:
            mbpp_texts.append(t)
    mbpp = Dataset.from_dict({"text": mbpp_texts})

    # Patch bucket
    if patch_ds is not None:
        patch_ds = patch_ds.map(fmt_patch, batched=True, remove_columns=patch_ds.column_names)
    else:
        patch_ds = Dataset.from_dict({"text": []})

    # Completion bucket: derive from raw pycode
    # We'll build it from scratch because we need prefix/suffix slicing.
    completion_texts: List[str] = []
    # batch iterate through pycode with column handling
    # (convert to python list access safely; datasets supports iteration)
    # Attempt to find a usable field name once:
    py_cols = list(pycode.column_names)

    # Determine which single column to use for raw code if present
    raw_col = None
    for k in ["code", "content", "text", "file_content", "source", "src"]:
        if k in py_cols:
            raw_col = k
            break

    if raw_col is None:
        # If the dataset is instruction/output style, we refuse to use it for completion
        print(f"[WARN] pycode has no raw code column among {py_cols}. Completion bucket may be empty.")
    else:
        for ex in pycode:
            code = ex.get(raw_col, None)
            if not isinstance(code, str):
                continue
            if len(code) < 300:
                continue
            pairs = extract_function_completion_pairs(code, rng=rng)
            for prefix, suffix in pairs:
                user = prompt_code_completion(prefix)
                completion_texts.append(make_chat_text(tokenizer, user, suffix))
            # keep it bounded
            if len(completion_texts) >= (cfg.mix_completion * 2):
                break

    completion = Dataset.from_dict({"text": completion_texts})

    # Filter empties
    magic = nonempty_text(magic, 32)
    evol = nonempty_text(evol, 32)
    mbpp = nonempty_text(mbpp, 32)
    patch_ds = nonempty_text(patch_ds, 32)
    completion = nonempty_text(completion, 32)

    # Drop bad buckets
    magic = drop_if_bad("magicoder", magic)
    evol = drop_if_bad("evol", evol)
    mbpp = drop_if_bad("mbpp", mbpp)
    completion = drop_if_bad("completion", completion)
    patch_ds = drop_if_bad("patch", patch_ds)

    print("\nSizes after formatting:")
    print("  completion:", (len(completion) if completion is not None else 0))
    print("  mbpp      :", (len(mbpp) if mbpp is not None else 0))
    print("  evol      :", (len(evol) if evol is not None else 0))
    print("  magicoder :", (len(magic) if magic is not None else 0))
    print("  patch     :", (len(patch_ds) if patch_ds is not None else 0))

    if completion is None or len(completion) == 0:
        raise RuntimeError("Completion bucket is empty. This will likely collapse HumanEval again. Fix pycode raw column.")

    # -----------------------
    # Build MIX with fixed counts (prevents accidental dominance)
    # -----------------------
    mix_plan: List[Tuple[str, Dataset, int]] = []
    mix_plan.append(("completion", completion, min(cfg.mix_completion, len(completion))))
    if mbpp is not None and len(mbpp) > 0:
        mix_plan.append(("mbpp", mbpp, min(cfg.mix_mbpp, len(mbpp))))
    if evol is not None and len(evol) > 0:
        mix_plan.append(("evol", evol, min(cfg.mix_evol, len(evol))))
    if magic is not None and len(magic) > 0:
        mix_plan.append(("magicoder", magic, min(cfg.mix_magicoder, len(magic))))
    if patch_ds is not None and len(patch_ds) > 0 and cfg.mix_patch > 0:
        mix_plan.append(("patch", patch_ds, min(cfg.mix_patch, len(patch_ds))))

    # If total > max_mixed, scale down non-completion buckets first
    total_target = sum(n for _, _, n in mix_plan)
    if total_target > cfg.max_mixed:
        overflow = total_target - cfg.max_mixed
        # reduce in this order: magicoder, evol, mbpp, patch (keep completion intact if possible)
        order = ["magicoder", "evol", "mbpp", "patch"]
        new_plan = []
        for name, ds, n in mix_plan:
            new_plan.append([name, ds, n])
        name_to_idx = {x[0]: i for i, x in enumerate(new_plan)}
        for nm in order:
            if overflow <= 0:
                break
            if nm in name_to_idx:
                i = name_to_idx[nm]
                reducible = max(0, new_plan[i][2] - 1000)
                dec = min(reducible, overflow)
                new_plan[i][2] -= dec
                overflow -= dec
        mix_plan = [(a, b, int(c)) for a, b, c in new_plan if int(c) > 0]

    # Materialize mix
    mixed_texts: List[str] = []
    for name, ds, k in mix_plan:
        if k <= 0:
            continue
        # sample with replacement if bucket smaller than k
        if len(ds) >= k:
            picked = ds.shuffle(seed=cfg.seed).select(range(k))
            mixed_texts.extend(picked["text"])
        else:
            # replacement
            for _ in range(k):
                idx = rng.randrange(len(ds))
                mixed_texts.append(ds[idx]["text"])

    rng.shuffle(mixed_texts)
    mixed_ds = Dataset.from_dict({"text": mixed_texts}).shuffle(seed=cfg.seed)

    split = mixed_ds.train_test_split(test_size=0.02, seed=cfg.seed)
    train_ds = split["train"]
    eval_ds = split["test"]

    print("\nMix plan:")
    for name, ds, k in mix_plan:
        print(f"  {name:10s} k={k:6d} (bucket_size={len(ds)})")
    print(f"Mixed dataset size: {len(mixed_ds)} | Train: {len(train_ds)} | Eval: {len(eval_ds)}")

    # -----------------------
    # Train
    # -----------------------
    training_args = make_training_args(
        TrainingArguments,
        output_dir="outputs_sft_v3_1",
        per_device_train_batch_size=cfg.per_device_bs,
        gradient_accumulation_steps=cfg.grad_acc,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.lr,
        warmup_ratio=0.03,
        logging_steps=20,
        evaluation_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=2,
        weight_decay=cfg.wd,
        lr_scheduler_type="linear",
        optim="adamw_8bit",
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        report_to="none",
        seed=cfg.seed,
        torch_compile=False,
    )

    print("\nStarting SFT v3.1 training...")
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

    trainer.train()

    print(f"\nSaving LoRA to: {cfg.output_lora}")
    model.save_pretrained(cfg.output_lora)
    tokenizer.save_pretrained(cfg.output_lora)
    print("Done.")


if __name__ == "__main__":
    main()
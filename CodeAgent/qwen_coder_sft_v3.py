#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SFT v3 for Qwen2.5-Coder-14B-Instruct (Unsloth):
- Fixes HumanEval collapse by training code-only format discipline + completion style.
- Uses python-code-dataset-500k for completion/continuation.
- Uses MBPP train for test-driven function synthesis style.
- Uses Evol-Instruct-Code but filters/forces code-only answers.
- Optional: patch_sft.jsonl (your own SWE-bench-style patch episodes).

IMPORTANT:
- Keep datasets==4.3.0 for Unsloth 2026.1.4 compatibility.
"""

import os
import sys
import re
import json
import random
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# ---- strongly recommended in your environment ----
# Avoid compile memory spikes on first steps
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
    Turn model/dataset output into pure Python code.
    - Prefer content inside ``` ``` fences.
    - Otherwise remove common prefaces.
    """
    if not text:
        return ""

    m = _CODE_FENCE_RE.search(text)
    if m:
        text = m.group(1)

    # Remove common assistant prefaces
    text = re.sub(r"^\s*(Sure|Here(?:'|’)s|Here is|Below is|Let's|Let’s).*?\n", "", text, flags=re.IGNORECASE)
    text = text.strip()

    # If it still contains obvious prose lines, keep only lines that look like code-ish.
    lines = text.splitlines()
    # If majority lines look code-like, keep all; else filter
    codeish = 0
    for ln in lines:
        s = ln.strip()
        if (
            s.startswith(("def ", "class ", "import ", "from ", "@", "if ", "for ", "while ", "try:", "return ", "#"))
            or s.endswith(":")
            or "(" in s
            or "=" in s
        ):
            codeish += 1
    if lines and codeish / max(1, len(lines)) < 0.35:
        # Filter aggressively
        kept = []
        for ln in lines:
            s = ln.strip()
            if (
                s.startswith(("def ", "class ", "import ", "from ", "@", "#"))
                or "return" in s
                or "=" in s
                or "(" in s
                or s.endswith(":")
            ):
                kept.append(ln)
        text = "\n".join(kept).strip()

    return text.strip() + "\n" if text else ""


def looks_like_python_code(text: str) -> bool:
    t = strip_to_code_only(text)
    if not t:
        return False
    # very lightweight heuristics
    return ("def " in t) or ("class " in t) or ("import " in t) or ("from " in t)


# -----------------------
# Prompt builders
# -----------------------
def make_chat_text(tokenizer, user: str, assistant: str) -> str:
    convo = [
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]
    return tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)


def prompt_code_completion(prefix: str) -> str:
    return (
        "You are a coding assistant.\n"
        "Complete the following Python code.\n"
        "Rules:\n"
        "- Return ONLY valid Python code.\n"
        "- No markdown fences.\n"
        "- No explanations.\n\n"
        "Code to complete:\n"
        f"{prefix}\n"
    )


def prompt_mbpp(problem: str, entry_point: str = "") -> str:
    if entry_point:
        return (
            "Write a correct Python solution.\n"
            "Rules:\n"
            "- Return ONLY valid Python code.\n"
            "- No markdown fences.\n"
            "- No explanations.\n"
            f"- You MUST implement a function named `{entry_point}` exactly.\n\n"
            f"Problem:\n{problem}\n"
        )
    return (
        "Write a correct Python solution.\n"
        "Rules:\n"
        "- Return ONLY valid Python code.\n"
        "- No markdown fences.\n"
        "- No explanations.\n\n"
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
# Dataset loading & formatting
# -----------------------
def infer_entry_point_from_mbpp_tests(tests: List[str], problem_text: str) -> str:
    joined = "\n".join(tests or [])
    m = re.search(r"assert\s+([A-Za-z_]\w*)\s*\(", joined)
    if m:
        return m.group(1)
    m = re.search(r"\bfunction\s+([A-Za-z_]\w*)\b", problem_text)
    if m:
        return m.group(1)
    m = re.search(r"\bnamed\s+([A-Za-z_]\w*)\b", problem_text)
    if m:
        return m.group(1)
    return ""


def load_patch_sft_jsonl(path: str):
    from datasets import load_dataset
    if not path or not os.path.exists(path):
        return None
    ds = load_dataset("json", data_files=path, split="train")
    # must contain prompt+patch
    if "prompt" not in ds.column_names or "patch" not in ds.column_names:
        print(f"[WARN] patch_sft.jsonl missing prompt/patch fields: {ds.column_names}")
        return None
    return ds


def main():
    import os, sys, json, random
    from dataclasses import dataclass
    from typing import List, Tuple

    import torch

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
        output_lora: str = "qwen_coder_lora_sft_v3"
        seed: int = 3407

        max_seq_length: int = 2048
        load_in_4bit: bool = True

        per_device_bs: int = 4
        grad_acc: int = 4
        epochs: float = 1.0
        lr: float = 1.0e-5
        wd: float = 0.01

        eval_steps: int = 250
        save_steps: int = 500

        n_instruct: int = 20000
        n_evol_code: int = 20000
        n_python_bucket: int = 30000
        n_mbpp_train: int = 2000

        patch_jsonl: str = "data/patch_sft.jsonl"
        packing: bool = True

        max_mixed: int = 80000  # cap for sanity

    cfg = CFG()
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
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=cfg.seed,
        use_rslora=False,
        loftq_config=None,
    )

    # -----------------------
    # Helpers
    # -----------------------
    def nonempty(ds: Dataset, min_len: int = 32) -> Dataset:
        if "text" not in ds.column_names:
            print(f"[WARN] dataset missing 'text' column. columns={ds.column_names}. Keeping as-is.")
            return ds
        return ds.filter(lambda x: isinstance(x.get("text", None), str) and len(x["text"]) > min_len)

    def drop_if_bad(name: str, ds: Dataset) -> Dataset | None:
        if ds is None:
            return None
        if "text" not in ds.column_names:
            print(f"[WARN] bucket '{name}' has no 'text' column after formatting -> DROPPED. columns={ds.column_names}")
            return None
        if len(ds) == 0:
            print(f"[WARN] bucket '{name}' is empty -> DROPPED.")
            return None
        return ds

    def make_chat_text(user: str, assistant: str) -> str:
        # Use tokenizer chat template if available
        if hasattr(tokenizer, "apply_chat_template"):
            msgs = [{"role": "user", "content": user},
                    {"role": "assistant", "content": assistant}]
            return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        return f"User:\n{user}\n\nAssistant:\n{assistant}"

    def strip_to_code_only(s: str) -> str:
        if not isinstance(s, str):
            return ""
        t = s.strip()
        # strip fenced blocks
        if "```" in t:
            # prefer python fences
            if "```python" in t:
                try:
                    return t.split("```python", 1)[1].split("```", 1)[0].strip()
                except Exception:
                    pass
            try:
                return t.split("```", 1)[1].split("```", 1)[0].strip()
            except Exception:
                pass
        return t

    def looks_like_python_code(s: str) -> bool:
        if not s or not isinstance(s, str):
            return False
        x = s.strip()
        if len(x) < 20:
            return False
        needles = ["def ", "class ", "import ", "from "]
        return any(n in x for n in needles)

    def prompt_code_completion(prefix: str) -> str:
        return (
            "Complete the following Python code.\n"
            "Rules: return ONLY the continuation code (no markdown, no explanation).\n\n"
            f"{prefix}"
        )

    def infer_entry_point_from_mbpp_tests(tests: List[str], problem: str) -> str:
        import re
        joined = "\n".join(tests or [])
        m = re.search(r"assert\s+([A-Za-z_]\w*)\s*\(", joined)
        if m:
            return m.group(1)
        m = re.search(r"\bfunction\s+([A-Za-z_]\w*)\b", problem or "")
        if m:
            return m.group(1)
        m = re.search(r"\bnamed\s+([A-Za-z_]\w*)\b", problem or "")
        if m:
            return m.group(1)
        return ""

    def prompt_mbpp(problem: str, entry_point: str = "") -> str:
        if entry_point:
            return (
                "Write a correct Python solution.\n"
                "Rules: return ONLY Python code, no markdown, no explanation.\n"
                f"You MUST implement a function named `{entry_point}` exactly.\n\n"
                f"Problem:\n{problem}\n"
            )
        return (
            "Write a correct Python solution.\n"
            "Rules: return ONLY Python code, no markdown, no explanation.\n\n"
            f"Problem:\n{problem}\n"
        )

    def load_patch_sft_jsonl(path: str) -> Dataset | None:
        if not path or (not os.path.exists(path)):
            return None
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
        if not rows:
            return None
        # expect {"prompt":..., "patch":...}
        prompt = [r.get("prompt", "") for r in rows]
        patch = [r.get("patch", "") for r in rows]
        return Dataset.from_dict({"prompt": prompt, "patch": patch})

    # -----------------------
    # Load datasets
    # -----------------------
    print("Loading datasets...")

    instruct = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train", trust_remote_code=False)
    if len(instruct) > cfg.n_instruct:
        instruct = instruct.shuffle(seed=cfg.seed).select(range(cfg.n_instruct))

    evol = load_dataset("nickrosh/Evol-Instruct-Code-80k-v1", split="train", trust_remote_code=False)
    if len(evol) > cfg.n_evol_code:
        evol = evol.shuffle(seed=cfg.seed).select(range(cfg.n_evol_code))

    pycode = load_dataset("jtatman/python-code-dataset-500k", split="train", trust_remote_code=False)
    if len(pycode) > cfg.n_python_bucket:
        pycode = pycode.shuffle(seed=cfg.seed).select(range(cfg.n_python_bucket))

    mbpp = load_dataset("mbpp", split="train", trust_remote_code=False)
    if len(mbpp) > cfg.n_mbpp_train:
        mbpp = mbpp.shuffle(seed=cfg.seed).select(range(cfg.n_mbpp_train))

    patch_ds = load_patch_sft_jsonl(cfg.patch_jsonl)
    if patch_ds is None or len(patch_ds) == 0:
        patch_ds = None
        print("[WARN] No patch_sft.jsonl found or empty; patch bucket disabled for v3.")
    else:
        print(f"[OK] Loaded patch SFT: {cfg.patch_jsonl} ({len(patch_ds)} rows)")

    # -----------------------
    # Format each bucket -> text
    # -----------------------
    def fmt_instruct(examples):
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

            user = (
                "Write correct Python code to solve the following problem.\n"
                "Rules: return ONLY Python code, no markdown, no explanation.\n\n"
                f"Problem:\n{p}\n"
            )
            outs.append(make_chat_text(user, s_code))

            # Optional completion augmentation to protect HumanEval behavior:
            lines = s_code.splitlines()
            if len(lines) >= 16:
                cut = random.randint(6, max(7, len(lines) * 2 // 3))
                prefix = "\n".join(lines[:cut]).rstrip() + "\n"
                suffix = "\n".join(lines[cut:]).lstrip() + "\n"
                if len(suffix.strip()) > 50:
                    user2 = prompt_code_completion(prefix)
                    outs.append(make_chat_text(user2, suffix))

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
            user = (
                "Write correct Python code for the following instruction.\n"
                "Rules: return ONLY Python code, no markdown, no explanation.\n\n"
                f"Instruction:\n{instr}\n"
            )
            outs.append(make_chat_text(user, out_code))
        return {"text": outs}

    def fmt_pycode(examples):
        # Your earlier debug showed ['instruction','output','system'] sometimes.
        # For jtatman/python-code-dataset-500k, schema may vary; handle both cases.
        outs = []

        if "instruction" in examples and "output" in examples:
            sys_msgs = examples.get("system", [""] * len(examples["instruction"]))
            instrs = examples.get("instruction", [])
            outs_raw = examples.get("output", [])
            for sysm, instr, out in zip(sys_msgs, instrs, outs_raw):
                if not instr or not out:
                    continue
                out_code = strip_to_code_only(out)
                if not looks_like_python_code(out_code):
                    continue
                sysm = (sysm or "").strip()
                if sysm:
                    user = (
                        f"System:\n{sysm}\n\n"
                        "Write correct Python code for the following instruction.\n"
                        "Rules: return ONLY Python code, no markdown, no explanation.\n\n"
                        f"Instruction:\n{instr}\n"
                    )
                else:
                    user = (
                        "Write correct Python code for the following instruction.\n"
                        "Rules: return ONLY Python code, no markdown, no explanation.\n\n"
                        f"Instruction:\n{instr}\n"
                    )
                outs.append(make_chat_text(user, out_code))
            return {"text": outs}

        # Otherwise treat as raw code corpus with a 'text' or 'code' column.
        codes = None
        for k in ["code", "content", "text", "file_content", "source", "src"]:
            if k in examples:
                codes = examples[k]
                break
        if codes is None:
            return {"text": outs}

        for code in codes:
            if not code or len(code) < 200:
                continue
            code = strip_to_code_only(code)
            if not looks_like_python_code(code):
                continue
            lines = code.splitlines()
            if len(lines) < 12:
                continue
            cut = random.randint(max(5, len(lines)//4), max(6, (len(lines)*3)//5))
            prefix = "\n".join(lines[:cut]).rstrip() + "\n"
            suffix = "\n".join(lines[cut:]).lstrip() + "\n"
            if len(suffix.strip()) < 40:
                continue
            user = prompt_code_completion(prefix)
            outs.append(make_chat_text(user, suffix))
        return {"text": outs}

    def fmt_patch(examples):
        outs = []
        prompts = examples.get("prompt", [])
        patches = examples.get("patch", [])
        for p, d in zip(prompts, patches):
            if not p or not d:
                continue
            d = d.strip() + "\n"
            # accept both unified diffs and git diffs; still prefer real diffs
            if ("diff --git" not in d) and (not d.startswith("---")):
                continue
            outs.append(make_chat_text(p, d))
        return {"text": outs}

    def fmt_mbpp_row(ex):
        problem = ex.get("text", "") or ""
        tests = ex.get("test_list", []) or []
        code = ex.get("code", "") or ""
        code = strip_to_code_only(code)
        if not problem or not code:
            return None
        if not looks_like_python_code(code):
            return None
        entry = infer_entry_point_from_mbpp_tests(tests, problem)
        user = prompt_mbpp(problem, entry_point=entry)
        return make_chat_text(user, code)

    # Apply mapping
    instruct = instruct.map(fmt_instruct, batched=True, remove_columns=instruct.column_names)
    evol     = evol.map(fmt_evol, batched=True, remove_columns=evol.column_names)
    pycode   = pycode.map(fmt_pycode, batched=True, remove_columns=pycode.column_names)

    mbpp_texts = []
    for ex in mbpp:
        t = fmt_mbpp_row(ex)
        if t is not None:
            mbpp_texts.append(t)
    mbpp = Dataset.from_dict({"text": mbpp_texts})

    if patch_ds is not None:
        patch_ds = patch_ds.map(fmt_patch, batched=True, remove_columns=patch_ds.column_names)
    else:
        patch_ds = Dataset.from_dict({"text": []})

    # filter empties
    instruct = nonempty(instruct, 32)
    evol     = nonempty(evol, 32)
    pycode   = nonempty(pycode, 32)
    mbpp     = nonempty(mbpp, 32)
    patch_ds = nonempty(patch_ds, 32)

    print("Sizes after formatting:")
    print("  instruct:", len(instruct), instruct.column_names)
    print("  evol    :", len(evol), evol.column_names)
    print("  pycode  :", len(pycode), pycode.column_names)
    print("  mbpp    :", len(mbpp), mbpp.column_names)
    print("  patch   :", len(patch_ds), patch_ds.column_names)

    # -----------------------
    # Mix datasets (weighted sample) - robust
    # -----------------------
    buckets: List[Tuple[str, Dataset, float]] = [
        ("pycode", pycode, 0.45),
        ("evol", evol, 0.25),
        ("instruct", instruct, 0.20),
        ("mbpp", mbpp, 0.10),
    ]
    if len(patch_ds) > 0:
        buckets.append(("patch", patch_ds, 0.10))

    # Drop invalid/empty buckets
    cleaned = []
    for name, ds, w in buckets:
        ds2 = drop_if_bad(name, ds)
        if ds2 is not None:
            cleaned.append((name, ds2, w))
    buckets = cleaned
    if not buckets:
        raise RuntimeError("All buckets empty or missing 'text' after formatting.")

    # Renormalize weights
    total_w = sum(w for _, _, w in buckets)
    buckets = [(n, ds, w / total_w) for (n, ds, w) in buckets]

    # Choose a total size
    target_size = min(sum(len(ds) for _, ds, _ in buckets), cfg.max_mixed)
    print(f"Mixing to target_size={target_size} from buckets:", [(n, len(ds), w) for n, ds, w in buckets])

    rng = random.Random(cfg.seed)
    mixed = []
    for _ in range(target_size):
        r = rng.random()
        cum = 0.0
        chosen_ds = None
        for _, ds, w in buckets:
            cum += w
            if r <= cum:
                chosen_ds = ds
                break
        if chosen_ds is None:
            chosen_ds = buckets[-1][1]
        # safe: chosen_ds always has text
        idx = rng.randrange(len(chosen_ds))
        mixed.append(chosen_ds[idx]["text"])

    mixed_ds = Dataset.from_dict({"text": mixed}).shuffle(seed=cfg.seed)

    split = mixed_ds.train_test_split(test_size=0.02, seed=cfg.seed)
    train_ds = split["train"]
    eval_ds = split["test"]

    print(f"Mixed dataset size: {len(mixed_ds)} | Train: {len(train_ds)} | Eval: {len(eval_ds)}")

    # -----------------------
    # Train
    # -----------------------
    training_args = make_training_args(
        TrainingArguments,
        output_dir="outputs_sft_v3",
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

    print("Starting SFT v3 training...")
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

    print(f"Saving LoRA to: {cfg.output_lora}")
    model.save_pretrained(cfg.output_lora)
    tokenizer.save_pretrained(cfg.output_lora)
    print("Done.")


if __name__ == "__main__":
    main()
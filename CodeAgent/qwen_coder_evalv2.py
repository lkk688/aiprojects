#!/usr/bin/env python3
"""
coder_eval_v2.py

A more reliable evaluation script for:
- PPL on WikiText2 (raw LM perplexity via sliding window)
- PPL on a held-out code dataset (MBPP "text" or CodeSearchNet Python docstrings)
- Induction NLL (repeat-half loss, token-accurate)
- Passkey retrieval (token-accurate needle insertion, strict answer format)
- HumanEval (subset or full) with safer execution and more robust code extraction

Notes:
- For meaningful perplexity, load in bf16/fp16 (NOT 4-bit). Quantization skews loss.
- For pass@1, use deterministic decoding (temperature=0, do_sample=False).
- HumanEval code execution is dangerous. Run in a sandboxed environment.

Usage:
  python coder_eval_v2.py \
    --base unsloth/Qwen2.5-Coder-14B-Instruct \
    --fine ./qwen_coder_lora \
    --out eval_results_v2.json \
    --humaneval_n 40 \
    --passkey_ctx 4096 \
    --device cuda

If you want to run full HumanEval:
  --humaneval_n 164
"""

import os
import sys
import json
import math
import time
import random
import argparse
import subprocess
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

# Optional plotting (matplotlib only; no seaborn needed)
import matplotlib.pyplot as plt


# -----------------------------
# Dependency install (optional)
# -----------------------------
def install_dependencies():
    pkgs = [
        "unsloth",
        "transformers>=4.45.0",
        "datasets",
        "numpy",
        "tqdm",
        "matplotlib",
    ]
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade"] + pkgs)


# -----------------------------
# Utilities
# -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_bf16_supported() -> bool:
    # Prefer Unsloth's helper if available; otherwise fallback
    try:
        from unsloth import is_bfloat16_supported
        return bool(is_bfloat16_supported())
    except Exception:
        return torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8


def safe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)


def now_ts() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


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
    Loads a model using Unsloth if available, otherwise falls back to Transformers.
    For perplexity, prefer bf16/fp16 (use_4bit=False).
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
        # Inference optimizations
        try:
            FastLanguageModel.for_inference(model)
        except Exception:
            pass
        return model, tokenizer
    except Exception as e:
        print(f"[WARN] Unsloth load failed ({e}). Falling back to Transformers.")

    # Transformers fallback (no LoRA auto-merge unless your folder is already merged)
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
    """
    Computes perplexity using a standard sliding-window approach:
    - Tokenize full text
    - Evaluate loss on each window with overlap
    - Average NLL per token, then exp()

    This is much more stable than truncating to max_length once.
    """
    model.eval()
    total_nll = 0.0
    total_tokens = 0

    for text in tqdm(texts, desc="PPL"):
        enc = tokenizer(text, return_tensors="pt", add_special_tokens=True)
        input_ids = enc["input_ids"][0].to(device)

        if input_ids.numel() < 2:
            continue

        # Sliding windows
        # We predict tokens [i : i+max_length] with labels shifted; mask out context overlap properly.
        for start in range(0, input_ids.numel(), stride):
            end = min(start + max_length, input_ids.numel())
            window = input_ids[start:end]

            # Labels: same as window, but mask the overlap tokens except the new ones
            labels = window.clone()
            if start > 0:
                # Mask out the tokens that are only context
                overlap = max_length - stride
                # In the first window after start>0, the "context" part is window[0:overlap]
                labels[:overlap] = -100

            outputs = model(window.unsqueeze(0), labels=labels.unsqueeze(0))
            # outputs.loss is average over unmasked labels
            # Convert average loss to total NLL by multiplying by number of contributing tokens
            contrib = (labels != -100).sum().item()
            if contrib > 0:
                total_nll += outputs.loss.item() * contrib
                total_tokens += contrib

            if end == input_ids.numel():
                break

    if total_tokens == 0:
        return float("inf")

    avg_nll = total_nll / total_tokens
    return float(math.exp(avg_nll))


def sample_texts_from_dataset(ds, text_cols: List[str], n_samples: int, seed: int) -> List[str]:
    """
    Samples n_samples texts from dataset ds using the first non-empty column among text_cols.
    """
    rng = random.Random(seed)
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    idxs = idxs[: min(n_samples, len(idxs))]

    texts = []
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
    """
    Repeat-half loss:
    Input: [A][A]
    Labels masked for first half, compute loss on second half.
    Lower loss => better copying/repetition.

    Token-accurate and simple.
    """
    model.eval()
    nlls = []

    for _ in tqdm(range(n_samples), desc="Induction"):
        # Sample random tokens away from special tokens range; keep simple
        # If tokenizer has special ids, this still may include them; in practice it's fine for a probe.
        A = torch.randint(low=1000, high=min(vocab_size, 50000), size=(1, seq_len), device=device)
        inp = torch.cat([A, A], dim=1)
        labels = inp.clone()
        labels[:, :seq_len] = -100

        out = model(inp, labels=labels)
        nlls.append(float(out.loss.item()))

    return float(np.mean(nlls)) if nlls else float("inf")


# -----------------------------
# Passkey retrieval (token-accurate)
# -----------------------------
@torch.no_grad()
def passkey_retrieval_acc(
    model,
    tokenizer,
    context_tokens: int,
    needle_depth: float,
    n_trials: int,
    device: str,
    use_chat_template: bool = True,
    max_new_tokens: int = 64,
    min_new_tokens: int = 4,
    debug_trials: int = 2,
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
            messages = [
                {"role": "user", "content": ctx_text + question}
            ]
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
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
            # Raw prompt fallback
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

        # Extract first integer from output
        digits = "".join([ch if ch.isdigit() else " " for ch in out_txt]).split()
        pred = digits[0] if digits else ""

        if t < debug_trials:
            print(f"[DEBUG] GT={passkey} pred={pred} raw={repr(out_txt[:200])}")

        if pred == str(passkey):
            hits += 1

    return float(hits / n_trials) if n_trials > 0 else 0.0

# -----------------------------
# HumanEval (safer subset runner)
# -----------------------------
def extract_code_from_generation(gen_text: str) -> str:
    """
    More robust than the simple split:
    - Prefer fenced ```python blocks
    - Else any ``` block
    - Else return raw text
    """
    t = gen_text
    if "```python" in t:
        t = t.split("```python", 1)[1]
        t = t.split("```", 1)[0]
        return t.strip()
    if "```" in t:
        t = t.split("```", 1)[1]
        t = t.split("```", 1)[0]
        return t.strip()
    return t.strip()


def run_code_subprocess(code: str, timeout_s: float = 6.0) -> bool:
    """
    Execute code in a subprocess. Still unsafe; run in a sandbox.
    Returns True if prints "SUCCESS".
    """
    wrapper = (
        "import sys\n"
        "try:\n"
        + "\n".join("    " + line for line in code.splitlines())
        + "\n    print('SUCCESS')\n"
        "except Exception as e:\n"
        "    print('FAILURE')\n"
    )
    try:
        res = subprocess.run(
            [sys.executable, "-c", wrapper],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        return ("SUCCESS" in res.stdout) and ("FAILURE" not in res.stdout)
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False


@torch.no_grad()
def humaneval_pass_at_1(
    model,
    tokenizer,
    n_problems: int,
    device: str,
    max_new_tokens: int = 512,
) -> float:
    """
    HumanEval pass@1 on first n_problems with deterministic decoding.
    This is still not the official HumanEval harness, but is improved:
    - deterministic decoding
    - better code extraction
    - longer timeout
    """
    from datasets import load_dataset
    ds = load_dataset("openai_humaneval", split="test")
    n = min(n_problems, len(ds))
    ds = ds.select(range(n))

    passed = 0

    for ex in tqdm(ds, desc="HumanEval"):
        prompt = ex["prompt"]
        test_code = ex["test"]
        entry_point = ex["entry_point"]

        # Use chat template if available; otherwise raw prompt
        try:
            messages = [{"role": "user", "content": f"Complete the following Python function.\nReturn only code.\n\n{prompt}"}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            text = f"Complete the following Python function. Return only code.\n\n{prompt}\n"

        inputs = tokenizer(text, return_tensors="pt").to(device)

        gen = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            use_cache=True,
        )

        gen_text = tokenizer.decode(gen[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        code_pred = extract_code_from_generation(gen_text)

        # Build executable script
        # HumanEval's test uses `check(entry_point)` expecting the function to exist.
        full = (
            "import math\n"
            "from typing import *\n\n"
            f"{prompt}\n"
            f"{code_pred}\n\n"
            f"{test_code}\n\n"
            f"check({entry_point})\n"
        )

        ok = run_code_subprocess(full, timeout_s=8.0)
        if ok:
            passed += 1

    return float(100.0 * passed / n) if n > 0 else 0.0


# -----------------------------
# Plotting
# -----------------------------
def plot_comparison(base: Dict[str, float], fine: Dict[str, float], save_path: str):
    keys = [k for k in base.keys() if k in fine.keys()]
    if not keys:
        print("[WARN] Nothing to plot.")
        return

    # Sort metrics for consistent order
    keys = sorted(keys)

    fig_w = min(5 * len(keys), 22)
    fig, axes = plt.subplots(1, len(keys), figsize=(fig_w, 5))
    if len(keys) == 1:
        axes = [axes]

    for ax, k in zip(axes, keys):
        ax.bar(["Base", "Finetuned"], [base[k], fine[k]])
        ax.set_title(k.replace("_", " "))
        for i, v in enumerate([base[k], fine[k]]):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom")
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    print(f"[OK] Saved plot to: {save_path}")


# -----------------------------
# Main evaluation routine
# -----------------------------
def evaluate(
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
    model.to(device)
    model.eval()

    from datasets import load_dataset

    results: Dict[str, float] = {}

    # 1) PPL WikiText2 (raw text)
    wt2 = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    wt_texts = sample_texts_from_dataset(wt2, ["text"], ppl_samples, seed=seed + 1)
    results["PPL_WikiText2"] = perplexity_sliding_window(
        model, tokenizer, wt_texts, device=device, max_length=ppl_max_len, stride=ppl_stride
    )

    # 2) Held-out code PPL: MBPP "text" (natural language prompt + examples)
    # This is not instruction-code exactly, but it's held-out from your Evol-Instruct-Code training.
    mbpp = load_dataset("mbpp", split="test")
    mbpp_texts = sample_texts_from_dataset(mbpp, ["text"], ppl_samples, seed=seed + 2)
    results["PPL_MBPP"] = perplexity_sliding_window(
        model, tokenizer, mbpp_texts, device=device, max_length=ppl_max_len, stride=ppl_stride
    )

    # 3) Induction NLL
    vocab_size = getattr(getattr(model, "config", None), "vocab_size", 100000)
    results["Induction_Repeat_NLL"] = induction_repeat_nll(
        model=model,
        vocab_size=int(vocab_size),
        seq_len=induction_seq_len,
        n_samples=induction_samples,
        device=device,
    )

    # 4) Passkey retrieval
    results["Passkey_Acc"] = passkey_retrieval_acc(
        model=model,
        tokenizer=tokenizer,
        context_tokens=passkey_ctx,
        needle_depth=passkey_depth,
        n_trials=passkey_trials,
        device=device,
    )

    # 5) HumanEval pass@1
    results["HumanEval_Pass@1"] = humaneval_pass_at_1(
        model=model,
        tokenizer=tokenizer,
        n_problems=humaneval_n,
        device=device,
        max_new_tokens=humaneval_max_new_tokens,
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
    parser.add_argument("--out", type=str, default="eval_results_v2.json")
    parser.add_argument("--plot", type=str, default="qwen_eval_comparison_v2.png")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=3407)

    # Model / context
    parser.add_argument("--max_seq_length", type=int, default=8192)

    # PPL params
    parser.add_argument("--ppl_max_len", type=int, default=2048)
    parser.add_argument("--ppl_stride", type=int, default=512)
    parser.add_argument("--ppl_samples", type=int, default=32)

    # Induction params
    parser.add_argument("--induction_seq_len", type=int, default=256)
    parser.add_argument("--induction_samples", type=int, default=20)

    # Passkey params
    parser.add_argument("--passkey_ctx", type=int, default=4096)
    parser.add_argument("--passkey_depth", type=float, default=0.5)
    parser.add_argument("--passkey_trials", type=int, default=10)

    # HumanEval params
    parser.add_argument("--humaneval_n", type=int, default=164) #40
    parser.add_argument("--humaneval_max_new_tokens", type=int, default=512)

    # Precision for eval
    parser.add_argument(
        "--use_4bit_for_eval",
        action="store_true",
        help="If set, evaluate in 4-bit (faster, but PPL less reliable).",
    )

    args = parser.parse_args()

    # If user lacks deps, they can uncomment:
    # install_dependencies()

    print("WARNING: This script executes generated code for HumanEval. Run in a sandbox if possible.\n")

    print(f"Evaluating BASE: {args.base}")
    base_res = evaluate(
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
        use_4bit_for_eval=args.use_4bit_for_eval,
        seed=args.seed,
    )
    print(json.dumps(base_res, indent=2))

    print(f"\nEvaluating FINETUNED: {args.fine}")
    fine_res = evaluate(
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
            "ppl": {
                "max_len": args.ppl_max_len,
                "stride": args.ppl_stride,
                "samples": args.ppl_samples,
            },
            "induction": {
                "seq_len": args.induction_seq_len,
                "samples": args.induction_samples,
            },
            "passkey": {
                "ctx": args.passkey_ctx,
                "depth": args.passkey_depth,
                "trials": args.passkey_trials,
            },
            "humaneval": {
                "n": args.humaneval_n,
                "max_new_tokens": args.humaneval_max_new_tokens,
            },
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
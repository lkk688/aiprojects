#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pre-SFT Data Quality Analyzer for Qwen Code SFT

Features:
- Analyze each SFT bucket before training
- Estimate token lengths with tokenizer
- Measure dedup ratio
- Measure Python parseability
- Visualize token distributions and mix plan
- Support custom local JSONL data:
    * instruction/output
    * completion prefix/suffix
    * raw_code
    * patch

Usage example:
python analyze_sft_data.py \
  --base_model your-qwen-9b-model-id \
  --output_dir output/analysis_report \
  --custom_jsonl data/my_curated_code.jsonl \
  --custom_mix_target 8000

python CodeAgent/sft_data_analyzer.py \
  --base_model Qwen/Qwen3.5-9B \
  --output_dir output/analysis_report \
  --custom_mix_target 8000
"""

import os
import re
import ast
import csv
import json
import math
import random
import argparse
import hashlib
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

# matplotlib only, no seaborn
import matplotlib.pyplot as plt

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# ============================================================
# Helpers
# ============================================================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def json_dump(obj: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def percentile(sorted_vals: List[int], p: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    k = (len(sorted_vals) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(sorted_vals[int(k)])
    d0 = sorted_vals[f] * (c - k)
    d1 = sorted_vals[c] * (k - f)
    return float(d0 + d1)


def safe_mean(vals: List[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def trim_text(s: str, n: int = 300) -> str:
    s = s.replace("\n", "\\n")
    return s[:n] + ("..." if len(s) > n else "")


# ============================================================
# Code cleaning / prompts / formatting
# ============================================================
_FENCE = "`" * 3
_CODE_FENCE_RE = re.compile(
    rf"{_FENCE}(?:python|py)?\s*(.*?){_FENCE}",
    re.DOTALL | re.IGNORECASE,
)

def strip_to_code_only(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    t = text.strip()
    m = _CODE_FENCE_RE.search(t)
    if m:
        t = m.group(1).strip()
    t = re.sub(r"^\s*(Sure|Here(?:'|’)s|Here is|Below is).*?\n", "", t, flags=re.IGNORECASE).strip()
    return (t + "\n") if t else ""


def looks_like_python_code(text: str) -> bool:
    if not isinstance(text, str):
        return False
    t = text.strip()
    if len(t) < 20:
        return False
    hints = ("def ", "class ", "import ", "from ", "return ", "try:", "except ")
    return sum(int(h in t) for h in hints) >= 1


def safe_parse_python(text: str) -> bool:
    if not isinstance(text, str) or not text.strip():
        return False
    try:
        ast.parse(text)
        return True
    except SyntaxError:
        return False


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


def make_chat_text(tokenizer, user: str, assistant: str) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        convo = [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
        return tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
    return f"User:\n{user}\n\nAssistant:\n{assistant}"


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


# ============================================================
# Completion extraction
# ============================================================
def extract_function_completion_pairs(
    code: str,
    seed: int,
    min_func_lines: int = 12,
    min_suffix_chars: int = 60,
    max_pairs_per_code: int = 2,
) -> List[Tuple[str, str]]:
    if not isinstance(code, str) or len(code) < 120:
        return []

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    lines = code.splitlines(keepends=True)
    if not lines:
        return []

    rng = random.Random(seed ^ len(code))
    pairs = []

    candidates = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start = node.lineno - 1
            end = getattr(node, "end_lineno", node.lineno + len(node.body))
            if (end - start) >= min_func_lines and len(node.body) >= 2:
                candidates.append(node)

    rng.shuffle(candidates)

    for node in candidates:
        body = node.body
        low = max(1, int(len(body) * 0.3))
        high = max(low + 1, int(len(body) * 0.8))
        if high <= low:
            continue
        split_idx = rng.randint(low, high - 1)
        cut_lineno = body[split_idx].lineno - 1
        if cut_lineno <= 0 or cut_lineno >= len(lines):
            continue

        prefix = "".join(lines[:cut_lineno])
        suffix = "".join(lines[cut_lineno:])
        if len(suffix.strip()) < min_suffix_chars:
            continue
        pairs.append((prefix, suffix))
        if len(pairs) >= max_pairs_per_code:
            break

    if pairs:
        return pairs

    if len(lines) >= min_func_lines:
        lo = max(1, int(len(lines) * 0.2))
        hi = max(lo + 1, int(len(lines) * 0.8))
        if hi > lo:
            cut = rng.randint(lo, hi - 1)
            prefix = "".join(lines[:cut])
            suffix = "".join(lines[cut:])
            if len(suffix.strip()) >= min_suffix_chars:
                return [(prefix, suffix)]
    return []


# ============================================================
# Custom JSONL loader
# ============================================================
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[WARN] JSONL parse error at line {i}: {e}")
    return rows


# ============================================================
# Bucket builders
# ============================================================
def build_magicoder_bucket(ds, tokenizer, max_samples: int) -> List[str]:
    out = []
    for ex in ds.select(range(min(len(ds), max_samples))):
        problem = ex.get("problem", "")
        solution = ex.get("solution", "")
        lang = str(ex.get("lang", "") or "")
        if not problem or not solution:
            continue
        if lang and "python" not in lang.lower():
            continue
        code = strip_to_code_only(solution)
        if not looks_like_python_code(code):
            continue
        out.append(make_chat_text(tokenizer, prompt_instruction_to_code(problem), code))
    return out


def build_evol_bucket(ds, tokenizer, max_samples: int) -> List[str]:
    out = []
    for ex in ds.select(range(min(len(ds), max_samples))):
        instr = ex.get("instruction", "")
        ans = ex.get("output", "")
        if not instr or not ans:
            continue
        code = strip_to_code_only(ans)
        if not looks_like_python_code(code):
            continue
        out.append(make_chat_text(tokenizer, prompt_instruction_to_code(instr), code))
    return out


def build_py_bucket(ds, tokenizer, max_samples: int) -> List[str]:
    out = []
    for ex in ds.select(range(min(len(ds), max_samples))):
        instr = ex.get("instruction", "") or ex.get("prompt", "") or ex.get("problem", "")
        sysm = ex.get("system", "") or ""
        ans = ex.get("output", "") or ex.get("solution", "") or ex.get("answer", "")

        if instr and ans:
            code = strip_to_code_only(ans)
            if looks_like_python_code(code):
                out.append(make_chat_text(tokenizer, prompt_instruction_to_code(instr, sysm), code))
                continue

        raw_code = ex.get("text", "") or ex.get("code", "") or ex.get("content", "") or ex.get("body", "")
        raw_code = strip_to_code_only(raw_code)
        if looks_like_python_code(raw_code):
            pseudo_user = (
                "Write Python code.\n"
                "Rules:\n"
                "- Return ONLY valid Python code.\n"
                "- No markdown.\n"
                "- No explanation.\n"
            )
            out.append(make_chat_text(tokenizer, pseudo_user, raw_code))
    return out


def build_mbpp_bucket(ds, tokenizer, max_samples: int) -> List[str]:
    out = []
    for ex in ds.select(range(min(len(ds), max_samples))):
        problem = ex.get("text", "") or ""
        tests = ex.get("test_list", []) or []
        code = strip_to_code_only(ex.get("code", "") or "")
        if not problem or not code or not looks_like_python_code(code):
            continue
        entry = infer_entry_point_from_mbpp_tests(tests, problem)
        out.append(make_chat_text(tokenizer, prompt_mbpp(problem, entry), code))
    return out


def build_completion_bucket_from_code_list(
    code_list: List[str],
    tokenizer,
    seed: int,
    max_samples: int,
) -> List[str]:
    out = []
    n = min(len(code_list), max_samples)
    for i, code in enumerate(code_list[:n]):
        code = strip_to_code_only(code)
        if not looks_like_python_code(code):
            continue
        for prefix, suffix in extract_function_completion_pairs(code, seed + i):
            out.append(make_chat_text(tokenizer, prompt_code_completion(prefix), suffix))
    return out


def build_custom_bucket(rows: List[Dict[str, Any]], tokenizer, seed: int) -> Dict[str, List[str]]:
    """
    Return multiple buckets from custom JSONL:
    - custom_instruction
    - custom_completion
    - custom_raw_completion
    - custom_patch
    """
    custom_instruction = []
    custom_completion = []
    custom_raw_completion = []
    custom_patch = []

    for i, row in enumerate(rows):
        typ = row.get("type", "")

        if typ == "instruction":
            instr = row.get("instruction", "")
            out = row.get("output", "")
            sysm = row.get("system", "")
            code = strip_to_code_only(out)
            if instr and looks_like_python_code(code):
                custom_instruction.append(
                    make_chat_text(tokenizer, prompt_instruction_to_code(instr, sysm), code)
                )

        elif typ == "completion":
            prefix = row.get("prefix", "")
            suffix = row.get("suffix", "")
            if prefix and suffix:
                custom_completion.append(
                    make_chat_text(tokenizer, prompt_code_completion(prefix), suffix.rstrip() + "\n")
                )

        elif typ == "raw_code":
            code = strip_to_code_only(row.get("code", ""))
            if looks_like_python_code(code):
                for prefix, suffix in extract_function_completion_pairs(code, seed + i):
                    custom_raw_completion.append(
                        make_chat_text(tokenizer, prompt_code_completion(prefix), suffix)
                    )

        elif typ == "patch":
            prompt = row.get("prompt", "")
            patch = row.get("patch", "")
            if prompt and patch:
                custom_patch.append(make_chat_text(tokenizer, prompt, patch.rstrip() + "\n"))

        else:
            # auto detect
            if "instruction" in row and "output" in row:
                code = strip_to_code_only(row["output"])
                if looks_like_python_code(code):
                    custom_instruction.append(
                        make_chat_text(
                            tokenizer,
                            prompt_instruction_to_code(row["instruction"], row.get("system", "")),
                            code,
                        )
                    )
            elif "prefix" in row and "suffix" in row:
                custom_completion.append(
                    make_chat_text(tokenizer, prompt_code_completion(row["prefix"]), row["suffix"].rstrip() + "\n")
                )
            elif "code" in row:
                code = strip_to_code_only(row["code"])
                if looks_like_python_code(code):
                    for prefix, suffix in extract_function_completion_pairs(code, seed + i):
                        custom_raw_completion.append(
                            make_chat_text(tokenizer, prompt_code_completion(prefix), suffix)
                        )

    return {
        "custom_instruction": custom_instruction,
        "custom_completion": custom_completion,
        "custom_raw_completion": custom_raw_completion,
        "custom_patch": custom_patch,
    }


# ============================================================
# Analyzer
# ============================================================
def extract_assistant_code_from_chat_text(text: str) -> str:
    """
    Best-effort parser for parse-rate estimation.
    This is heuristic because templates differ.
    """
    if not isinstance(text, str):
        return ""
    # Try common chat template fragments
    markers = [
        "<|im_start|>assistant\n",
        "<|assistant|>\n",
        "Assistant:\n",
    ]
    for m in markers:
        idx = text.rfind(m)
        if idx >= 0:
            return text[idx + len(m):].strip()
    return text.strip()


def summarize_bucket(
    bucket_name: str,
    texts: List[str],
    tokenizer,
    sample_preview_count: int = 5,
) -> Dict[str, Any]:
    raw_count = len(texts)
    deduped = []
    seen = set()
    for t in texts:
        h = sha1_text(t)
        if h not in seen:
            seen.add(h)
            deduped.append(t)

    dedup_count = len(deduped)
    dedup_ratio = (1.0 - dedup_count / raw_count) if raw_count else 0.0

    char_lens = []
    token_lens = []
    parse_ok = 0

    previews = []

    for i, t in enumerate(deduped):
        cl = len(t)
        char_lens.append(cl)

        try:
            tl = len(tokenizer(t, add_special_tokens=False)["input_ids"])
        except Exception:
            tl = 0
        token_lens.append(tl)

        assistant_code = extract_assistant_code_from_chat_text(t)
        assistant_code = strip_to_code_only(assistant_code)
        if looks_like_python_code(assistant_code) and safe_parse_python(assistant_code):
            parse_ok += 1

        if len(previews) < sample_preview_count:
            previews.append({
                "index": i,
                "chars": cl,
                "tokens": tl,
                "preview": trim_text(t, 400),
            })

    token_lens_sorted = sorted(token_lens)
    char_lens_sorted = sorted(char_lens)

    return {
        "bucket": bucket_name,
        "raw_count": raw_count,
        "dedup_count": dedup_count,
        "dedup_ratio": dedup_ratio,
        "parse_rate": (parse_ok / dedup_count) if dedup_count else 0.0,
        "chars_mean": safe_mean(char_lens),
        "chars_p50": percentile(char_lens_sorted, 0.50),
        "chars_p90": percentile(char_lens_sorted, 0.90),
        "chars_p95": percentile(char_lens_sorted, 0.95),
        "chars_max": max(char_lens) if char_lens else 0,
        "tokens_mean": safe_mean(token_lens),
        "tokens_p50": percentile(token_lens_sorted, 0.50),
        "tokens_p90": percentile(token_lens_sorted, 0.90),
        "tokens_p95": percentile(token_lens_sorted, 0.95),
        "tokens_max": max(token_lens) if token_lens else 0,
        "sample_previews": previews,
        "token_lengths": token_lens,
    }


def plot_histogram(token_lengths: List[int], title: str, out_path: str) -> None:
    if not token_lengths:
        return
    plt.figure(figsize=(8, 5))
    plt.hist(token_lengths, bins=40)
    plt.xlabel("Token length")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_bucket_mean_tokens(summary_rows: List[Dict[str, Any]], out_path: str) -> None:
    names = [r["bucket"] for r in summary_rows]
    vals = [r["tokens_mean"] for r in summary_rows]
    plt.figure(figsize=(10, 5))
    plt.bar(names, vals)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Mean token length")
    plt.title("Mean Token Length by Bucket")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_mix_plan(mix_plan: Dict[str, int], out_path: str) -> None:
    names = list(mix_plan.keys())
    vals = list(mix_plan.values())
    plt.figure(figsize=(10, 5))
    plt.bar(names, vals)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Planned sample count")
    plt.title("Planned Mix Quota")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def write_csv_summary(summary_rows: List[Dict[str, Any]], out_path: str) -> None:
    fields = [
        "bucket", "raw_count", "dedup_count", "dedup_ratio", "parse_rate",
        "chars_mean", "chars_p50", "chars_p90", "chars_p95", "chars_max",
        "tokens_mean", "tokens_p50", "tokens_p90", "tokens_p95", "tokens_max",
    ]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({k: row[k] for k in fields})


# ============================================================
# Main
# ============================================================
@dataclass
class CFG:
    base_model: str
    output_dir: str
    custom_jsonl: str = ""
    custom_mix_target: int = 8000
    seed: int = 3407

    n_magicoder: int = 8000
    n_evol: int = 8000
    n_py_instr: int = 20000
    n_mbpp_train: int = 2000

    mix_completion: int = 42000
    mix_mbpp: int = 12000
    mix_evol: int = 8000
    mix_magicoder: int = 6000
    mix_py_instr: int = 6000


def parse_args() -> CFG:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="analysis_report")
    parser.add_argument("--custom_jsonl", type=str, default="")
    parser.add_argument("--custom_mix_target", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=3407)

    parser.add_argument("--n_magicoder", type=int, default=8000)
    parser.add_argument("--n_evol", type=int, default=8000)
    parser.add_argument("--n_py_instr", type=int, default=20000)
    parser.add_argument("--n_mbpp_train", type=int, default=2000)

    parser.add_argument("--mix_completion", type=int, default=42000)
    parser.add_argument("--mix_mbpp", type=int, default=12000)
    parser.add_argument("--mix_evol", type=int, default=8000)
    parser.add_argument("--mix_magicoder", type=int, default=6000)
    parser.add_argument("--mix_py_instr", type=int, default=6000)

    a = parser.parse_args()
    return CFG(**vars(a))


def main():
    cfg = parse_args()
    ensure_dir(cfg.output_dir)

    from datasets import load_dataset
    from transformers import AutoTokenizer

    print(f"[*] Loading tokenizer from {cfg.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True)

    print("[*] Loading datasets...")
    ds_magic = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train")
    if len(ds_magic) > cfg.n_magicoder:
        ds_magic = ds_magic.shuffle(seed=cfg.seed).select(range(cfg.n_magicoder))

    ds_evol = load_dataset("nickrosh/Evol-Instruct-Code-80k-v1", split="train")
    if len(ds_evol) > cfg.n_evol:
        ds_evol = ds_evol.shuffle(seed=cfg.seed).select(range(cfg.n_evol))

    ds_py = load_dataset("jtatman/python-code-dataset-500k", split="train")
    if len(ds_py) > cfg.n_py_instr:
        ds_py = ds_py.shuffle(seed=cfg.seed).select(range(cfg.n_py_instr))

    ds_mbpp = load_dataset("mbpp", split="train")
    if len(ds_mbpp) > cfg.n_mbpp_train:
        ds_mbpp = ds_mbpp.shuffle(seed=cfg.seed).select(range(cfg.n_mbpp_train))

    print("[*] Building buckets...")
    magic_bucket = build_magicoder_bucket(ds_magic, tokenizer, cfg.n_magicoder)
    evol_bucket = build_evol_bucket(ds_evol, tokenizer, cfg.n_evol)
    py_bucket = build_py_bucket(ds_py, tokenizer, cfg.n_py_instr)
    mbpp_bucket = build_mbpp_bucket(ds_mbpp, tokenizer, cfg.n_mbpp_train)

    raw_codes = []
    if "solution" in ds_magic.column_names:
        raw_codes.extend([x for x in ds_magic["solution"] if isinstance(x, str)])
    if "output" in ds_evol.column_names:
        raw_codes.extend([x for x in ds_evol["output"] if isinstance(x, str)])
    if "code" in ds_mbpp.column_names:
        raw_codes.extend([x for x in ds_mbpp["code"] if isinstance(x, str)])

    completion_bucket = build_completion_bucket_from_code_list(
        raw_codes, tokenizer, cfg.seed, max_samples=len(raw_codes)
    )

    bucket_map = {
        "completion": completion_bucket,
        "mbpp": mbpp_bucket,
        "evol": evol_bucket,
        "magicoder": magic_bucket,
        "py_instr": py_bucket,
    }

    custom_mix = 0
    if cfg.custom_jsonl and os.path.exists(cfg.custom_jsonl):
        print(f"[*] Loading custom JSONL: {cfg.custom_jsonl}")
        rows = load_jsonl(cfg.custom_jsonl)
        custom_buckets = build_custom_bucket(rows, tokenizer, cfg.seed)
        for k, v in custom_buckets.items():
            bucket_map[k] = v
        custom_mix = cfg.custom_mix_target

    print("[*] Summarizing buckets...")
    summaries = []
    for name, texts in bucket_map.items():
        summary = summarize_bucket(name, texts, tokenizer)
        summaries.append(summary)

        hist_path = os.path.join(cfg.output_dir, f"token_hist_{name}.png")
        plot_histogram(summary["token_lengths"], f"Token Length Distribution: {name}", hist_path)

    summaries_sorted = sorted(summaries, key=lambda x: x["bucket"])
    write_csv_summary(summaries_sorted, os.path.join(cfg.output_dir, "bucket_summary.csv"))

    mix_plan = {
        "completion": cfg.mix_completion,
        "mbpp": cfg.mix_mbpp,
        "evol": cfg.mix_evol,
        "magicoder": cfg.mix_magicoder,
        "py_instr": cfg.mix_py_instr,
    }

    if custom_mix > 0:
        mix_plan["custom"] = custom_mix

    plot_bucket_mean_tokens(summaries_sorted, os.path.join(cfg.output_dir, "bucket_mean_tokens.png"))
    plot_mix_plan(mix_plan, os.path.join(cfg.output_dir, "mix_plan.png"))

    report = {
        "config": asdict(cfg),
        "summaries": [
            {k: v for k, v in row.items() if k != "token_lengths"}
            for row in summaries_sorted
        ],
        "recommended_checks": [
            "Look for buckets with very low parse_rate.",
            "Look for buckets with very high tokens_p95; these may dominate packing.",
            "Look for high dedup_ratio indicating repetitive or templated samples.",
            "Ensure completion remains the dominant training bucket.",
            "Check sample_previews manually before final SFT.",
        ],
    }

    json_dump(report, os.path.join(cfg.output_dir, "report.json"))

    print(f"[+] Analysis report written to: {cfg.output_dir}")
    print("[+] Files generated:")
    print("    - report.json")
    print("    - bucket_summary.csv")
    print("    - bucket_mean_tokens.png")
    print("    - mix_plan.png")
    print("    - token_hist_*.png")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
swebench_patch_agent.py

MVP SWE-bench patch generation agent:
- Loads SWE-bench Verified dataset instances
- Clones repo + checks out base commit
- Prompts model to output a unified diff patch (PATCH ONLY)
- Applies patch
- Runs a basic test command
- Writes predictions JSONL: {"instance_id": ..., "model_patch": ...}

WARNING: Running arbitrary repos/tests can be risky. Prefer running in containers.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch
from datasets import load_dataset


# -------------------------
# Shell helpers
# -------------------------
def sh(cmd: List[str], cwd: Optional[Path] = None, timeout: int = 1200) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        timeout=timeout,
    )

def sh_ok(cmd: List[str], cwd: Optional[Path] = None, timeout: int = 1200) -> str:
    p = sh(cmd, cwd=cwd, timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\nSTDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}")
    return p.stdout


# -------------------------
# Model loading (Unsloth if available)
# -------------------------
def load_model(model_name_or_path: str, device: str = "cuda"):
    try:
        from unsloth import FastLanguageModel, is_bfloat16_supported
        dtype = torch.bfloat16 if (device.startswith("cuda") and is_bfloat16_supported()) else torch.float16
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name_or_path,
            max_seq_length=8192,
            dtype=dtype,
            load_in_4bit=False,   # prefer full precision for patch gen stability
        )
        FastLanguageModel.for_inference(model)
        model.to(device).eval()
        return model, tokenizer
    except Exception:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16 if device.startswith("cuda") else None,
            device_map="auto" if device.startswith("cuda") else None,
            trust_remote_code=True,
        )
        model.eval()
        return model, tokenizer


# -------------------------
# Patch extraction
# -------------------------
DIFF_START_RE = re.compile(r"^diff --git ", re.MULTILINE)

def extract_unified_diff(text: str) -> str:
    """
    Extract unified diff from model output. If model prints extra text,
    try to slice from the first 'diff --git'.
    """
    m = DIFF_START_RE.search(text)
    if m:
        return text[m.start():].strip()
    # Also accept plain '---' patch style
    if "\n--- " in text and "\n+++ " in text:
        i = text.find("\n--- ")
        return text[i+1:].strip()
    return text.strip()


# -------------------------
# Prompting
# -------------------------
def build_patch_prompt(instance: Dict[str, Any], repo_tree_summary: str, failing_log: str) -> str:
    """
    Strong constraints: patch-only output, no prose.
    """
    problem = instance.get("problem_statement", "")
    return f"""You are a senior software engineer. Produce a correct fix as a unified diff patch.

Rules:
- Output ONLY a unified diff patch.
- Start with lines like: diff --git a/... b/...
- Do NOT include any explanation or markdown fences.
- Keep changes minimal and focused.
- Ensure tests pass.

Issue:
{problem}

Repository snapshot summary:
{repo_tree_summary}

Failing test / error log:
{failing_log}

Now output the patch ONLY:
"""


def generate_patch(model, tokenizer, prompt: str, device: str = "cuda", max_new_tokens: int = 1200) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = prompt

    inputs = tokenizer(text, return_tensors="pt").to(device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        use_cache=True,
        pad_token_id=getattr(tokenizer, "eos_token_id", None),
    )
    gen_text = tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return extract_unified_diff(gen_text)


# -------------------------
# Repo/test utilities
# -------------------------
def clone_and_checkout(repo: str, commit: str, workdir: Path):
    sh_ok(["git", "clone", repo, str(workdir)])
    sh_ok(["git", "checkout", commit], cwd=workdir)

def repo_summary(workdir: Path, max_lines: int = 200) -> str:
    """
    Simple structure summary: top-level + python packages. Keep short for prompt.
    """
    p = sh(["bash", "-lc", "ls -la"], cwd=workdir)
    lines = (p.stdout.splitlines() + p.stderr.splitlines())[:max_lines]
    return "\n".join(lines)

def run_tests_basic(workdir: Path, timeout: int = 1200) -> str:
    """
    MVP: run pytest if present. Many SWE-bench repos use pytest.
    For production, use the repo's specified test command (SWE-bench provides info in metadata).
    """
    # Quick heuristic:
    if (workdir / "pytest.ini").exists() or (workdir / "pyproject.toml").exists() or (workdir / "setup.cfg").exists():
        p = sh(["bash", "-lc", "pytest -q"], cwd=workdir, timeout=timeout)
        return (p.stdout + "\n" + p.stderr).strip()
    # fallback: try unit tests
    p = sh(["bash", "-lc", "python -m pytest -q"], cwd=workdir, timeout=timeout)
    return (p.stdout + "\n" + p.stderr).strip()

def apply_patch(workdir: Path, patch: str) -> bool:
    """
    Apply unified diff using git apply.
    """
    # write patch to file
    patch_path = workdir / "model.patch"
    patch_path.write_text(patch, encoding="utf-8")
    p = sh(["git", "apply", "--whitespace=nowarn", str(patch_path)], cwd=workdir)
    return p.returncode == 0

def git_diff(workdir: Path) -> str:
    return sh_ok(["git", "diff"], cwd=workdir)


# -------------------------
# Main agent loop
# -------------------------
def solve_instance(model, tokenizer, inst: Dict[str, Any], device: str, rounds: int) -> Dict[str, Any]:
    """
    Returns dict containing instance_id and model_patch (diff), possibly empty if failed.
    """
    instance_id = inst["instance_id"]
    repo = inst["repo"]              # often like "https://github.com/ORG/REPO"
    base_commit = inst["base_commit"]

    with tempfile.TemporaryDirectory() as td:
        workdir = Path(td) / "repo"
        clone_and_checkout(repo, base_commit, workdir)

        # Reproduce failure (best effort)
        fail_log = ""
        try:
            fail_log = run_tests_basic(workdir, timeout=1200)
        except Exception as e:
            fail_log = f"(test run failed to execute) {e}"

        summary = repo_summary(workdir)

        best_patch = ""
        for r in range(1, rounds + 1):
            prompt = build_patch_prompt(inst, summary, fail_log)
            patch = generate_patch(model, tokenizer, prompt, device=device)

            if not patch.strip():
                fail_log = "Model produced empty patch."
                continue

            # reset repo
            sh_ok(["git", "reset", "--hard"], cwd=workdir)
            sh_ok(["git", "clean", "-fd"], cwd=workdir)

            ok_apply = apply_patch(workdir, patch)
            if not ok_apply:
                fail_log = "git apply failed. Patch was not applicable."
                continue

            # Run tests again
            try:
                out = run_tests_basic(workdir, timeout=1200)
            except Exception as e:
                out = f"(test run failed) {e}"

            # Simple success criterion:
            # pytest -q exits 0 => output often empty or has "passed"
            # We can check return code but our helper returns output only.
            # Instead, use subprocess directly to inspect return code:
            p = sh(["bash", "-lc", "pytest -q"], cwd=workdir, timeout=1200)
            if p.returncode == 0:
                best_patch = git_diff(workdir)
                break
            else:
                fail_log = (p.stdout + "\n" + p.stderr).strip()[-4000:]

        return {
            "instance_id": instance_id,
            "model_patch": best_patch,
        }



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model name or local path (base or finetuned).")
    ap.add_argument("--dataset", default="princeton-nlp/SWE-bench_Verified")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="predictions.jsonl")
    ap.add_argument("--n", type=int, default=20, help="How many instances to attempt.")
    ap.add_argument("--rounds", type=int, default=2, help="Max repair iterations per instance.")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    model, tokenizer = load_model(args.model, device=args.device)

    ds = load_dataset(args.dataset, split=args.split)
    ds = ds.select(range(min(args.n, len(ds))))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for inst in ds:
            pred = solve_instance(model, tokenizer, inst, device=args.device, rounds=args.rounds)
            f.write(json.dumps(pred) + "\n")
            f.flush()
            print(f"[{pred['instance_id']}] patch_len={len(pred['model_patch'])}")

    print(f"Saved predictions to {out_path}")


if __name__ == "__main__":
    main()
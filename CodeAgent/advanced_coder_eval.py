#!/usr/bin/env python3
"""
advanced_coder_eval.py

Runs:
  1) EvalPlus: HumanEval+ and MBPP+ (official evalplus.evaluate)
  2) SWE-bench Verified evaluation harness (official swebench.harness.run_evaluation)
  3) lm-eval-harness (official lm_eval CLI)

Outputs:
  - results/advanced_eval.json (merged summary)
  - results/raw/... (raw tool outputs)
  - results/figures/*.png (research-grade plots)

Refs:
  - EvalPlus quickstart & CLI usage: evalplus.evaluate ... --dataset [humaneval|mbpp] --backend hf --greedy
     [oai_citation:3‡PyPI](https://pypi.org/project/evalplus/)
  - SWE-bench evaluation harness: python -m swebench.harness.run_evaluation --dataset_name ... --predictions_path ...
     [oai_citation:4‡PyPI](https://pypi.org/project/swebench/)
  - SWE-bench Verified dataset: princeton-nlp/SWE-bench_Verified (500 instances)
     [oai_citation:5‡Hugging Face](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified/blob/main/README.md?code=true&utm_source=chatgpt.com)
  - lm-eval CLI (PyPI): lm_eval --model hf --model_args pretrained=...
     [oai_citation:6‡PyPI](https://pypi.org/project/lm-eval/?utm_source=chatgpt.com)

SECURITY WARNING:
  - Code execution and patch application are performed by external harnesses (Docker for SWE-bench; EvalPlus can use Docker too).
  - Run in a sandboxed environment if possible.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


# -----------------------------
# Helpers
# -----------------------------
def ts() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def which_or_raise(cmd: str, install_hint: str) -> str:
    path = shutil.which(cmd)
    if not path:
        raise RuntimeError(f"Missing command `{cmd}`.\nInstall hint:\n{install_hint}")
    return path


def run_cmd(
    cmd: List[str],
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
    tee_to: Optional[Path] = None,
    timeout: Optional[int] = None,
) -> subprocess.CompletedProcess:
    """
    Run subprocess, optionally tee stdout+stderr to file.
    """
    print("\n>>", " ".join(cmd))
    p = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
    )
    if tee_to:
        ensure_dir(tee_to.parent)
        tee_to.write_text(p.stdout, encoding="utf-8")
    if p.returncode != 0:
        print(p.stdout[-4000:])
        raise RuntimeError(f"Command failed (code={p.returncode}): {' '.join(cmd)}")
    return p


def read_json_maybe(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def write_json(path: Path, obj: Any):
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


# -----------------------------
# Parsers
# -----------------------------
def parse_evalplus_score(stdout: str) -> Dict[str, Any]:
    """
    EvalPlus prints and/or writes results files. We keep a simple stdout parser as fallback.

    We try to capture:
      - pass@1 (or pass@k)
      - dataset name
    If not found, returns empty dict.
    """
    out: Dict[str, Any] = {}

    # Common patterns seen in eval frameworks:
    # "pass@1: 0.XX" or "pass@1 = 0.XX" or "pass@1: XX.XX%"
    m = re.search(r"pass@1\s*[:=]\s*([0-9.]+)\s*%?", stdout, flags=re.IGNORECASE)
    if m:
        val = float(m.group(1))
        if val > 1.0:  # likely percent
            val = val / 100.0
        out["pass@1"] = val

    mk = re.search(r"pass@(\d+)\s*[:=]\s*([0-9.]+)\s*%?", stdout, flags=re.IGNORECASE)
    if mk and "pass@1" not in out:
        k = int(mk.group(1))
        val = float(mk.group(2))
        if val > 1.0:
            val = val / 100.0
        out[f"pass@{k}"] = val

    return out


def parse_lm_eval_json(path: Path) -> Dict[str, Any]:
    """
    lm-eval produces JSON containing "results" (metric dict per task).
    """
    data = read_json_maybe(path) or {}
    results = data.get("results", {})
    # Keep only common scalar metrics for plotting
    slim: Dict[str, Any] = {}
    for task, md in results.items():
        # choose a representative metric if present
        # common: acc, acc_norm, exact_match, f1, etc.
        if isinstance(md, dict):
            for key in ["acc", "acc_norm", "exact_match", "f1", "bleu", "pass@1"]:
                if key in md and isinstance(md[key], (int, float)):
                    slim[task] = {key: float(md[key])}
                    break
            else:
                # fall back: first numeric
                for k, v in md.items():
                    if isinstance(v, (int, float)):
                        slim[task] = {k: float(v)}
                        break
    return {"raw": data, "slim": slim}


def parse_swebench_summary(run_dir: Path) -> Dict[str, Any]:
    """
    SWE-bench writes reports under a run_id directory (varies by version).
    We make a best-effort parse by searching for JSON summary files.

    If you know the exact output file name in your environment, you can hardcode it.
    """
    out: Dict[str, Any] = {"run_dir": str(run_dir)}
    if not run_dir.exists():
        return out

    # Heuristic: look for json files that contain "resolved" counts
    json_files = list(run_dir.rglob("*.json"))
    for jf in json_files:
        try:
            j = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            continue

        # known patterns in swebench harness reports:
        # - may include aggregate stats or list of instance results
        if isinstance(j, dict):
            # If has "resolved"/"completed" keys
            if any(k in j for k in ["resolved", "completed", "success", "accuracy"]):
                out.setdefault("summaries", []).append({"path": str(jf), "data": j})
                continue

            # If includes list of results under a key
            for key in ["results", "instances", "data"]:
                if key in j and isinstance(j[key], list):
                    # compute quick aggregates if possible
                    res_list = j[key]
                    resolved = 0
                    completed = 0
                    for item in res_list:
                        if not isinstance(item, dict):
                            continue
                        if item.get("completed") is True:
                            completed += 1
                        if item.get("resolved") is True:
                            resolved += 1
                    if completed > 0:
                        out.setdefault("summaries", []).append({
                            "path": str(jf),
                            "derived": {
                                "completed": completed,
                                "resolved": resolved,
                                "resolve_rate": resolved / max(1, completed),
                            }
                        })
                    break

    # If we derived at least one resolve_rate, take the max as "best" summary
    best = None
    for s in out.get("summaries", []):
        rr = None
        if "derived" in s and "resolve_rate" in s["derived"]:
            rr = s["derived"]["resolve_rate"]
        elif "data" in s and isinstance(s["data"], dict) and "resolve_rate" in s["data"]:
            rr = s["data"]["resolve_rate"]
        if rr is not None:
            if best is None or rr > best[0]:
                best = (rr, s)
    if best:
        out["best"] = best[1]
    return out


# -----------------------------
# Plotting (matplotlib-only, research style)
# -----------------------------
def _style_axes(ax):
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_main_summary(summary: Dict[str, Any], out_png: Path):
    """
    One clean figure for the headline metrics:
      - HumanEval+ pass@1
      - MBPP+ pass@1
      - SWE-bench Verified resolve rate (if available)
    """
    base = summary.get("base", {})
    fine = summary.get("finetuned", {})

    def get_metric(section: Dict[str, Any], key_path: List[str]) -> Optional[float]:
        cur = section
        for k in key_path:
            if not isinstance(cur, dict) or k not in cur:
                return None
            cur = cur[k]
        return float(cur) if isinstance(cur, (int, float)) else None

    metrics = [
        ("HumanEval+ pass@1", ["evalplus", "humaneval", "pass@1"]),
        ("MBPP+ pass@1", ["evalplus", "mbpp", "pass@1"]),
        ("SWE-bench Verified\nresolve_rate", ["swebench", "best", "derived", "resolve_rate"]),
    ]

    labels = []
    base_vals = []
    fine_vals = []

    for title, path in metrics:
        b = get_metric(base, path)
        f = get_metric(fine, path)
        if b is None and f is None:
            continue
        labels.append(title)
        base_vals.append(b if b is not None else float("nan"))
        fine_vals.append(f if f is not None else float("nan"))

    if not labels:
        return

    fig = plt.figure(figsize=(max(8, 2.2 * len(labels)), 4.5))
    ax = fig.add_subplot(111)
    x = list(range(len(labels)))
    width = 0.38

    ax.bar([i - width / 2 for i in x], base_vals, width=width, label="Base")
    ax.bar([i + width / 2 for i in x], fine_vals, width=width, label="Finetuned")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel("Score (fraction)" if any(v <= 1.0 for v in base_vals + fine_vals if not (v != v)) else "Score")
    ax.legend(frameon=False, loc="upper left")
    _style_axes(ax)

    # annotate
    for i, (b, f) in enumerate(zip(base_vals, fine_vals)):
        if not (b != b):
            ax.text(i - width / 2, b, f"{b:.3f}", ha="center", va="bottom")
        if not (f != f):
            ax.text(i + width / 2, f, f"{f:.3f}", ha="center", va="bottom")

    fig.tight_layout()
    ensure_dir(out_png.parent)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def plot_lm_eval_tasks(summary: Dict[str, Any], out_png: Path, top_n: int = 12):
    """
    Plot top-N lm-eval tasks by base score (representative metric per task).
    """
    base = summary.get("base", {}).get("lm_eval", {}).get("slim", {})
    fine = summary.get("finetuned", {}).get("lm_eval", {}).get("slim", {})
    if not base:
        return

    def pick_metric(d: Dict[str, Any]) -> Tuple[str, float]:
        # single metric dict: {metric_name: value}
        for k, v in d.items():
            return k, float(v)
        return "metric", float("nan")

    rows = []
    for task, md in base.items():
        metric_name, bval = pick_metric(md)
        fmd = fine.get(task, {})
        _, fval = pick_metric(fmd) if fmd else (metric_name, float("nan"))
        rows.append((task, metric_name, bval, fval))

    # sort by base value descending
    rows.sort(key=lambda r: (r[2] if r[2] == r[2] else -1e9), reverse=True)
    rows = rows[:top_n]

    tasks = [r[0] for r in rows]
    metric = rows[0][1] if rows else "metric"
    bvals = [r[2] for r in rows]
    fvals = [r[3] for r in rows]

    fig = plt.figure(figsize=(10.5, 0.55 * len(tasks) + 2.2))
    ax = fig.add_subplot(111)
    y = list(range(len(tasks)))
    height = 0.36

    ax.barh([i - height / 2 for i in y], bvals, height=height, label="Base")
    ax.barh([i + height / 2 for i in y], fvals, height=height, label="Finetuned")

    ax.set_yticks(y)
    ax.set_yticklabels(tasks)
    ax.invert_yaxis()
    ax.set_xlabel(metric)
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    ensure_dir(out_png.parent)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


# -----------------------------
# Harness runners
# -----------------------------
@dataclass
class EvalConfig:
    base_model: str
    fine_model: str
    device: str
    results_dir: Path
    seed: int
    # EvalPlus
    evalplus_backend: str  # "hf" or "vllm"
    evalplus_parallel: int
    evalplus_greedy: bool
    evalplus_attn_impl: Optional[str]
    # SWE-bench
    swebench_dataset: str
    swebench_max_workers: int
    swebench_predictions_base: Optional[Path]
    swebench_predictions_fine: Optional[Path]
    swebench_run_id_base: str
    swebench_run_id_fine: str
    # lm-eval
    lm_eval_tasks: List[str]
    lm_eval_batch_size: int


def run_evalplus(model: str, dataset: str, cfg: EvalConfig, out_dir: Path, tag: str) -> Dict[str, Any]:
    """
    Runs EvalPlus for a dataset ("humaneval" or "mbpp"). EvalPlus evaluates HumanEval(+) / MBPP(+)  [oai_citation:7‡PyPI](https://pypi.org/project/evalplus/)

    We capture stdout, and also preserve evalplus_results if created.
    """
    which_or_raise(
        "evalplus.evaluate",
        "pip install --upgrade 'evalplus[hf]'  (or)  pip install --upgrade \"evalplus[vllm] @ git+https://github.com/evalplus/evalplus\"",
    )

    ensure_dir(out_dir)
    log_path = out_dir / f"evalplus_{dataset}_{tag}.log"

    cmd = [
        "evalplus.evaluate",
        "--model", model,
        "--dataset", dataset,
        "--backend", cfg.evalplus_backend,
    ]
    if cfg.evalplus_greedy:
        cmd.append("--greedy")
    if cfg.evalplus_attn_impl:
        cmd += ["--attn-implementation", cfg.evalplus_attn_impl]
    if cfg.evalplus_parallel and cfg.evalplus_parallel > 1:
        cmd += ["--parallel", str(cfg.evalplus_parallel)]

    p = run_cmd(cmd, tee_to=log_path)
    parsed = parse_evalplus_score(p.stdout)

    return {
        "dataset": dataset,
        "backend": cfg.evalplus_backend,
        "greedy": cfg.evalplus_greedy,
        "attn_impl": cfg.evalplus_attn_impl,
        "log": str(log_path),
        "parsed": parsed,
        "note": "For fully reliable numbers, prefer EvalPlus' own result files under evalplus_results/ if present.",
    }


def run_swebench_eval(
    predictions_path: Path,
    dataset_name: str,
    run_id: str,
    max_workers: int,
    out_dir: Path,
) -> Dict[str, Any]:
    """
    Runs SWE-bench harness evaluation in Docker environment.  [oai_citation:8‡PyPI](https://pypi.org/project/swebench/)
    Dataset SWE-bench Verified: princeton-nlp/SWE-bench_Verified  [oai_citation:9‡Hugging Face](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified/blob/main/README.md?code=true&utm_source=chatgpt.com)
    """
    # swebench CLI is a python module
    # Ensure import works:
    try:
        import swebench  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "Missing `swebench` package.\nInstall hint:\n  pip install swebench\n"
            f"Import error: {e}"
        )

    ensure_dir(out_dir)
    log_path = out_dir / f"swebench_eval_{run_id}.log"

    cmd = [
        sys.executable, "-m", "swebench.harness.run_evaluation",
        "--dataset_name", dataset_name,
        "--predictions_path", str(predictions_path),
        "--max_workers", str(max_workers),
        "--run_id", run_id,
    ]
    p = run_cmd(cmd, tee_to=log_path)

    # Heuristic: swebench creates reports in a run directory; try to find it.
    # Most setups create something under ./runs/<run_id> or similar; we search near cwd/out_dir.
    # You can also pass your own run output dir by customizing the swebench command if supported.
    candidates = []
    for root in [Path.cwd(), out_dir, out_dir.parent]:
        if root.exists():
            candidates += list(root.rglob(run_id))
    run_dirs = [c for c in candidates if c.is_dir()]

    parsed = {}
    if run_dirs:
        # pick the most recently modified
        run_dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
        parsed = parse_swebench_summary(run_dirs[0])

    return {
        "dataset_name": dataset_name,
        "predictions_path": str(predictions_path),
        "run_id": run_id,
        "max_workers": max_workers,
        "log": str(log_path),
        "parsed": parsed,
        "stdout_tail": p.stdout[-2000:],
    }


def run_lm_eval(model: str, tasks: List[str], batch_size: int, device: str, out_dir: Path, tag: str) -> Dict[str, Any]:
    """
    Runs lm-eval-harness via CLI.  [oai_citation:10‡PyPI](https://pypi.org/project/lm-eval/?utm_source=chatgpt.com)
    """
    which_or_raise(
        "lm_eval",
        "pip install 'lm_eval[hf]'  # installs lm-eval-harness CLI with HuggingFace backend",
    )

    ensure_dir(out_dir)
    out_json = out_dir / f"lm_eval_{tag}.json"
    log_path = out_dir / f"lm_eval_{tag}.log"

    # Note: lm_eval expects --model hf and model_args pretrained=<...>
    # device like cuda:0
    device_arg = device
    if device == "cuda":
        device_arg = "cuda:0"

    cmd = [
        "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model}",
        "--tasks", ",".join(tasks),
        "--device", device_arg,
        "--batch_size", str(batch_size),
        "--output_path", str(out_json),
    ]

    run_cmd(cmd, tee_to=log_path)
    parsed = parse_lm_eval_json(out_json)

    return {
        "tasks": tasks,
        "batch_size": batch_size,
        "device": device_arg,
        "out_json": str(out_json),
        "log": str(log_path),
        "parsed": parsed["slim"],
        "raw_path": str(out_json),
    }


# -----------------------------
# Main
# -----------------------------
def build_summary(
    meta: Dict[str, Any],
    base_block: Dict[str, Any],
    fine_block: Dict[str, Any],
) -> Dict[str, Any]:
    return {"meta": meta, "base": base_block, "finetuned": fine_block}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Base model name/path (HF or local).")
    ap.add_argument("--fine", required=True, help="Finetuned model name/path (HF or local).")
    ap.add_argument("--device", default="cuda", help="cuda | cuda:0 | cpu")
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--results_dir", default="results", help="Output directory root.")

    # EvalPlus
    ap.add_argument("--run_evalplus", action="store_true")
    ap.add_argument("--evalplus_backend", default="hf", choices=["hf", "vllm"])
    ap.add_argument("--evalplus_parallel", type=int, default=0, help="EvalPlus --parallel N (0 disables).")
    ap.add_argument("--evalplus_attn_impl", default=None, help="EvalPlus --attn-implementation flash_attention_2|sdpa")
    ap.add_argument("--evalplus_no_greedy", action="store_true", help="Disable greedy decoding in EvalPlus.")

    # SWE-bench
    ap.add_argument("--run_swebench", action="store_true")
    ap.add_argument("--swebench_dataset", default="princeton-nlp/SWE-bench_Verified")
    ap.add_argument("--swebench_max_workers", type=int, default=8)
    ap.add_argument("--swebench_predictions_base", default=None, help="Path to base predictions JSONL for SWE-bench.")
    ap.add_argument("--swebench_predictions_fine", default=None, help="Path to finetuned predictions JSONL for SWE-bench.")

    # lm-eval
    ap.add_argument("--run_lm_eval", action="store_true")
    ap.add_argument("--lm_eval_tasks", default="humaneval,mbpp", help="Comma list of lm-eval tasks/groups.")
    ap.add_argument("--lm_eval_batch_size", type=int, default=8)

    # Plotting
    ap.add_argument("--make_figures", action="store_true")

    args = ap.parse_args()

    results_root = ensure_dir(Path(args.results_dir))
    raw_dir = ensure_dir(results_root / "raw" / ts())
    fig_dir = ensure_dir(results_root / "figures")

    cfg = EvalConfig(
        base_model=args.base,
        fine_model=args.fine,
        device=args.device,
        results_dir=results_root,
        seed=args.seed,
        evalplus_backend=args.evalplus_backend,
        evalplus_parallel=args.evalplus_parallel,
        evalplus_greedy=not args.evalplus_no_greedy,
        evalplus_attn_impl=args.evalplus_attn_impl,
        swebench_dataset=args.swebench_dataset,
        swebench_max_workers=args.swebench_max_workers,
        swebench_predictions_base=Path(args.swebench_predictions_base) if args.swebench_predictions_base else None,
        swebench_predictions_fine=Path(args.swebench_predictions_fine) if args.swebench_predictions_fine else None,
        swebench_run_id_base=f"base_{ts()}",
        swebench_run_id_fine=f"fine_{ts()}",
        lm_eval_tasks=[t.strip() for t in args.lm_eval_tasks.split(",") if t.strip()],
        lm_eval_batch_size=args.lm_eval_batch_size,
    )

    meta = {
        "timestamp": ts(),
        "base": cfg.base_model,
        "fine": cfg.fine_model,
        "device": cfg.device,
        "seed": cfg.seed,
        "tools": {
            "evalplus": bool(args.run_evalplus),
            "swebench": bool(args.run_swebench),
            "lm_eval": bool(args.run_lm_eval),
        },
        "notes": [
            "EvalPlus runs HumanEval+/MBPP+ (enhanced tests).",
            "SWE-bench Verified evaluation requires a predictions file (patches).",
            "lm-eval provides broad standardized tasks; choose coding-related tasks/groups as desired.",
        ],
    }

    base_block: Dict[str, Any] = {}
    fine_block: Dict[str, Any] = {}

    # -------- EvalPlus --------
    if args.run_evalplus:
        base_block["evalplus"] = {}
        fine_block["evalplus"] = {}
        for ds in ["humaneval", "mbpp"]:
            base_block["evalplus"][ds] = run_evalplus(cfg.base_model, ds, cfg, raw_dir, tag="base")
            fine_block["evalplus"][ds] = run_evalplus(cfg.fine_model, ds, cfg, raw_dir, tag="fine")

            # Normalize to convenient keys for plotting if parse succeeded
            for blk, tag in [(base_block, "base"), (fine_block, "fine")]:
                parsed = blk["evalplus"][ds].get("parsed", {})
                if "pass@1" in parsed:
                    blk["evalplus"][ds]["pass@1"] = float(parsed["pass@1"])

    # -------- SWE-bench --------
    if args.run_swebench:
        base_block["swebench"] = {}
        fine_block["swebench"] = {}

        if cfg.swebench_predictions_base:
            base_block["swebench"] = run_swebench_eval(
                predictions_path=cfg.swebench_predictions_base,
                dataset_name=cfg.swebench_dataset,
                run_id=cfg.swebench_run_id_base,
                max_workers=cfg.swebench_max_workers,
                out_dir=raw_dir,
            )
        else:
            base_block["swebench"]["skipped"] = "No --swebench_predictions_base provided."

        if cfg.swebench_predictions_fine:
            fine_block["swebench"] = run_swebench_eval(
                predictions_path=cfg.swebench_predictions_fine,
                dataset_name=cfg.swebench_dataset,
                run_id=cfg.swebench_run_id_fine,
                max_workers=cfg.swebench_max_workers,
                out_dir=raw_dir,
            )
        else:
            fine_block["swebench"]["skipped"] = "No --swebench_predictions_fine provided."

    # -------- lm-eval --------
    if args.run_lm_eval:
        base_block["lm_eval"] = run_lm_eval(
            model=cfg.base_model,
            tasks=cfg.lm_eval_tasks,
            batch_size=cfg.lm_eval_batch_size,
            device=cfg.device,
            out_dir=raw_dir,
            tag="base",
        )
        fine_block["lm_eval"] = run_lm_eval(
            model=cfg.fine_model,
            tasks=cfg.lm_eval_tasks,
            batch_size=cfg.lm_eval_batch_size,
            device=cfg.device,
            out_dir=raw_dir,
            tag="fine",
        )

    summary = build_summary(meta, base_block, fine_block)

    out_json = results_root / "advanced_eval.json"
    write_json(out_json, summary)
    print(f"\n[OK] Wrote merged summary: {out_json}")

    # -------- Figures --------
    if args.make_figures:
        # headline summary
        plot_main_summary(summary, fig_dir / "summary_headline.png")

        # lm-eval tasks
        plot_lm_eval_tasks(summary, fig_dir / "lm_eval_top_tasks.png", top_n=12)

        print(f"[OK] Wrote figures to: {fig_dir}")


if __name__ == "__main__":
    main()

"""
#Run EvalPlus + lm-eval (no SWE-bench)
python advanced_coder_eval.py \
  --base unsloth/Qwen2.5-Coder-14B-Instruct \
  --fine ./qwen_coder_lora \
  --run_evalplus \
  --run_lm_eval \
  --lm_eval_tasks humaneval,mbpp \
  --make_figures

#Add SWE-bench Verified (needs predictions files)
#SWE-bench evaluation is patch-based; you need a predictions JSONL file and then run the harness
#SWE-bench Verified has 500 instances  ￼ and evaluation is storage + Docker heavy  ￼. Most teams treat SWE-bench as a separate pipeline:
python advanced_coder_eval.py \
  --base unsloth/Qwen2.5-Coder-14B-Instruct \
  --fine ./qwen_coder_lora \
  --run_evalplus \
  --run_lm_eval \
  --run_swebench \
  --swebench_dataset princeton-nlp/SWE-bench_Verified \
  --swebench_predictions_base /path/base_preds.jsonl \
  --swebench_predictions_fine /path/fine_preds.jsonl \
  --make_figures
"""
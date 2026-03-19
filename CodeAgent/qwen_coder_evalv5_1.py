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
import asyncio
import importlib.util
import statistics
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
    import subprocess
    import sys
    
    print(f"\n>> {' '.join(cmd)}")
    
    # 额外加一道保险，防止 HuggingFace Tokenizer 在多进程下死锁
    if env is None:
        env = os.environ.copy()
    env["TOKENIZERS_PARALLELISM"] = "false"

    # 使用 Popen 实时流式输出 stdout，彻底告别“假死”
    p = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    
    output_lines = []
    # 实时逐行打印子进程的输出
    for line in p.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        output_lines.append(line)
        
    p.wait()
    full_output = "".join(output_lines)
    
    if tee_to:
        ensure_dir(os.path.dirname(tee_to))
        with open(tee_to, "w", encoding="utf-8") as f:
            f.write(full_output)
            
    res = subprocess.CompletedProcess(cmd, p.returncode, full_output, "")
    
    if p.returncode != 0:
        print(f"\n[WARN] Command failed (code={p.returncode}): {' '.join(cmd)}")

    return res


def _resolve_command(binary_name: str, module_name: str) -> Optional[List[str]]:
    """Resolve a CLI command: first try binary, then python -m fallback."""
    cmd_path = shutil.which(binary_name)
    if cmd_path:
        return [cmd_path]
    try:
        if importlib.util.find_spec(module_name) is not None:
            return [sys.executable, "-m", module_name]
    except (ModuleNotFoundError, ImportError, ValueError):
        pass
    return None


def _percentile(values: List[float], pct: float) -> float:
    """Compute a percentile value using linear interpolation."""
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(float(v) for v in values)
    rank = (len(ordered) - 1) * (pct / 100.0)
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return ordered[low]
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _summarize_speed_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute percentile statistics for a list of speed benchmark rows."""
    total_runs = len(rows)
    ok = [r for r in rows if r.get("success")]
    if not ok:
        return {"runs": total_runs, "successful_runs": 0, "success_rate": 0.0,
                "ttft_p50_sec": 0.0, "ttft_p95_sec": 0.0, "e2e_p50_sec": 0.0,
                "decode_p50_tokens_per_sec": 0.0, "decode_p95_tokens_per_sec": 0.0,
                "throughput_tokens_per_sec": 0.0}
    ttft_vals = [float(r["ttft_sec"]) for r in ok]
    e2e_vals = [float(r["e2e_latency_sec"]) for r in ok]
    e2e_tps_vals = [float(r["e2e_tokens_per_sec"]) for r in ok]
    decode_tps_vals = [float(r["decode_tokens_per_sec"]) for r in ok]
    total_completion_tokens = sum(int(r["completion_tokens"]) for r in ok)
    total_elapsed = sum(float(r["e2e_latency_sec"]) for r in ok)
    return {
        "runs": total_runs,
        "successful_runs": len(ok),
        "success_rate": round(len(ok) / total_runs, 4) if total_runs else 0.0,
        "ttft_p50_sec": round(_percentile(ttft_vals, 50), 4),
        "ttft_p95_sec": round(_percentile(ttft_vals, 95), 4),
        "e2e_p50_sec": round(_percentile(e2e_vals, 50), 4),
        "e2e_p95_sec": round(_percentile(e2e_vals, 95), 4),
        "e2e_speed_p50_tokens_per_sec": round(_percentile(e2e_tps_vals, 50), 3),
        "decode_p50_tokens_per_sec": round(_percentile(decode_tps_vals, 50), 3),
        "decode_p95_tokens_per_sec": round(_percentile(decode_tps_vals, 95), 3),
        "throughput_tokens_per_sec": round(
            (total_completion_tokens / total_elapsed) if total_elapsed > 0 else 0.0, 3
        ),
    }


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


def create_hf_batch_generator(
    model,
    tokenizer,
    device: str,
    max_new_tokens: int,
    batch_size: int = 8,
):
    """
    Returns a function that accepts a list of prompt strings and generates
    completions for all of them in batches.  Left-padding is applied so the
    last token of every prompt aligns to the same position before generation.

    Returns: List[Dict] in the same per-item format as create_hf_generator.
    """
    # Causal-LM generation requires left-padding so positions are consistent.
    orig_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    @torch.no_grad()
    def batch_generator(prompts: List[str]) -> List[Dict[str, Any]]:
        all_results: List[Dict[str, Any]] = []
        for i in range(0, len(prompts), batch_size):
            chunk = prompts[i : i + batch_size]
            t0 = time.time()
            inputs = tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            ).to(device)
            gen = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )
            dt = time.time() - t0
            prompt_len = inputs["input_ids"].shape[1]
            for j in range(len(chunk)):
                out_ids = gen[j, prompt_len:]
                # Strip trailing pad tokens
                mask = out_ids != tokenizer.pad_token_id
                if mask.any():
                    out_ids = out_ids[: int(mask.nonzero()[-1].item()) + 1]
                out_text = tokenizer.decode(out_ids, skip_special_tokens=True)
                all_results.append({
                    "text": out_text,
                    "prompt_tokens": int(prompt_len),
                    "completion_tokens": int(out_ids.shape[0]),
                    "elapsed_seconds": round(dt / max(1, len(chunk)), 4),
                })
        tokenizer.padding_side = orig_padding_side
        return all_results

    return batch_generator


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


def run_api_speed_benchmark(
    spec: ModelSpec,
    out_dir: str,
    prefill_tokens: List[int] = None,
    runs: int = 5,
    max_output_tokens: int = 256,
    timeout_s: float = 60.0,
) -> Dict[str, Any]:
    """Measure TTFT, decode TPS, and E2E latency for an API endpoint.

    Ported from evaluator_main.py: run_llm_speed_evaluation().
    Useful for benchmarking base vs fine-tuned model served via vLLM.

    Returns a dict with raw rows + percentile statistics.
    """
    if prefill_tokens is None:
        prefill_tokens = [128, 512, 2048]

    try:
        import httpx as _httpx
        from openai import AsyncOpenAI as _AsyncOpenAI
    except ImportError:
        return {"status": "unavailable", "error": "httpx / openai package required for speed benchmark"}

    api_key = os.getenv(spec.api_key_env) or "none"

    # Filler context for different prefill lengths
    _seed_text = (
        "Autoregressive decoding benchmark. Measure prompt ingestion speed, "
        "decode speed, and latency distributions. "
        "Keep semantic coherence while expanding token count. "
    )

    def _build_context(n_tokens: int) -> str:
        chunks: List[str] = []
        while sum(len(c.split()) for c in chunks) < n_tokens:
            chunks.append(_seed_text)
        return "\n".join(chunks)

    async def _run_async() -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        async with _httpx.AsyncClient(timeout=timeout_s) as http_client:
            client = _AsyncOpenAI(
                base_url=spec.api_base,
                api_key=api_key,
                http_client=http_client,
            )
            for pt in prefill_tokens:
                context = _build_context(pt)
                for run_idx in range(runs):
                    first_tok_ts: Optional[float] = None
                    t0 = time.perf_counter()
                    completion_text = ""
                    error_text = ""
                    prompt_toks, completion_toks = 0, 0
                    try:
                        stream = await client.chat.completions.create(
                            model=spec.path,
                            messages=[
                                {"role": "system", "content": "You are a concise assistant."},
                                {"role": "user", "content": f"{context}\n\nSummarize in 3 bullets."},
                            ],
                            max_tokens=max_output_tokens,
                            temperature=0.0,
                            stream=True,
                        )
                        async for chunk in stream:
                            delta = chunk.choices[0].delta.content or "" if chunk.choices else ""
                            if delta and first_tok_ts is None:
                                first_tok_ts = time.perf_counter()
                            completion_text += delta
                            # Extract usage from final chunk if available
                            if hasattr(chunk, "usage") and chunk.usage:
                                prompt_toks = chunk.usage.prompt_tokens or 0
                                completion_toks = chunk.usage.completion_tokens or 0
                    except Exception as exc:
                        error_text = str(exc)

                    elapsed = time.perf_counter() - t0
                    ttft = (first_tok_ts - t0) if first_tok_ts else elapsed
                    if completion_toks == 0:
                        completion_toks = max(1, len(completion_text.split()))
                    if prompt_toks == 0:
                        prompt_toks = pt  # approximate

                    decode_elapsed = max(elapsed - ttft, 0.001)
                    decode_tps = completion_toks / decode_elapsed
                    e2e_tps = (prompt_toks + completion_toks) / max(elapsed, 0.001)
                    success = error_text == "" and completion_toks > 0
                    rows.append({
                        "run_index": run_idx + 1,
                        "prefill_target_tokens": pt,
                        "prompt_tokens": prompt_toks,
                        "completion_tokens": completion_toks,
                        "ttft_sec": round(ttft, 4),
                        "e2e_latency_sec": round(elapsed, 4),
                        "decode_tokens_per_sec": round(decode_tps, 2),
                        "e2e_tokens_per_sec": round(e2e_tps, 2),
                        "success": success,
                        "error": error_text,
                    })
                    status_icon = "OK" if success else "ERR"
                    print(f"  [{status_icon}] prefill={pt} run={run_idx+1} "
                          f"ttft={ttft:.3f}s decode_tps={decode_tps:.1f} e2e={elapsed:.3f}s")
        return rows

    rows = asyncio.run(_run_async())

    # Group by prefill length for summary
    by_prefill: Dict[int, List[Dict]] = {}
    for r in rows:
        by_prefill.setdefault(r["prefill_target_tokens"], []).append(r)
    summary = {str(pt): _summarize_speed_rows(vs) for pt, vs in by_prefill.items()}
    overall = _summarize_speed_rows(rows)

    # Save raw rows
    out_path = os.path.join(out_dir, f"api_speed_{sanitize_name(spec.name)}.json")
    json_dump({"rows": rows, "summary_by_prefill": summary, "overall": overall}, out_path)

    return {
        "status": "ok",
        "overall": overall,
        "summary_by_prefill": summary,
        "raw_path": out_path,
    }


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
def _run_codegen_batched(
    task_items: List[Tuple[str, str]],   # [(task_id, prompt_text), ...]
    generator_fn: Callable,
    out_file: str,
    desc: str,
) -> int:
    """
    Generate completions for a list of (task_id, prompt) pairs.
    Supports both the single-prompt generator (returns dict) and the
    batch generator (accepts list[str], returns list[dict]).
    Writes JSONL to out_file and returns sample count.
    """
    import json
    import inspect
    is_batch = (
        inspect.isfunction(generator_fn)
        and len(inspect.signature(generator_fn).parameters) == 1
        and any(
            str(ann) in ("typing.List[str]", "List[str]")
            for ann in inspect.signature(generator_fn).parameters.values()
        )
    )
    # Simpler heuristic: if calling with a list doesn't raise TypeError, it's batch.
    # We detect by checking the first param annotation or just try both.
    try:
        # probe: if the generator accepts a list it is a batch generator
        sig = inspect.signature(generator_fn)
        first_param = list(sig.parameters.values())[0]
        _is_batch_gen = getattr(first_param.annotation, "__origin__", None) is list
    except Exception:
        _is_batch_gen = False

    with open(out_file, "w", encoding="utf-8") as f_out:
        if _is_batch_gen:
            # Batch path: pass all prompts at once
            prompts = [pt for _, pt in task_items]
            gens = generator_fn(prompts)
            for (task_id, _), g in tqdm(
                zip(task_items, gens), total=len(task_items), desc=desc
            ):
                code_pred = extract_code_from_generation(g["text"])
                f_out.write(json.dumps({"task_id": task_id, "completion": code_pred}) + "\n")
        else:
            # Single-prompt fallback
            for task_id, prompt_text in tqdm(task_items, desc=desc):
                g = generator_fn(prompt_text)
                code_pred = extract_code_from_generation(g["text"])
                f_out.write(json.dumps({"task_id": task_id, "completion": code_pred}) + "\n")

    return len(task_items)


def humaneval_pass_at_1(generator_fn: Callable, tokenizer: Any, n: int, args: argparse.Namespace, out_file: str) -> Dict[str, Any]:
    try:
        from evalplus.data import get_human_eval_plus as _get_he
    except ImportError:
        from evalplus.data import get_human_eval as _get_he  # type: ignore[no-redef]

    ds = _get_he()
    task_items = []
    for task_id, ex in ds.items():
        user_prompt = (
            "Complete the following Python function.\n"
            "Return ONLY code.\n\n"
            f"{ex['prompt']}"
        )
        task_items.append((task_id, apply_chat_prompt(tokenizer, user_prompt)))

    count = _run_codegen_batched(task_items, generator_fn, out_file, "Generating HumanEval")
    return {"status": "ok_generated", "samples": count, "note": "Delegated pass@1 to EvalPlus"}


def mbpp_pass_at_1(generator_fn: Callable, tokenizer: Any, n: int, args: argparse.Namespace, out_file: str) -> Dict[str, Any]:
    try:
        from evalplus.data import get_mbpp_plus as _get_mbpp
    except ImportError:
        from evalplus.data import get_mbpp as _get_mbpp  # type: ignore[no-redef]

    ds = _get_mbpp()
    task_items = []
    for task_id, ex in ds.items():
        user_prompt = (
            "Write a Python function that solves the following problem.\n"
            "Return ONLY code.\n\n"
            f"{ex['prompt']}"
        )
        if "test_list" in ex and ex["test_list"]:
            user_prompt += "\n\nFor example:\n" + "\n".join(ex["test_list"][:3])
        task_items.append((task_id, apply_chat_prompt(tokenizer, user_prompt)))

    count = _run_codegen_batched(task_items, generator_fn, out_file, "Generating MBPP")
    return {"status": "ok_generated", "samples": count, "note": "Delegated pass@1 to EvalPlus"}


# ============================================================
# External harnesses
# ============================================================
def parse_evalplus_pass(stdout_text: str) -> Tuple[Optional[float], Optional[float]]:
    """Parse evalplus output and return (base_pass@1, plus_pass@1) both in [0-100] percentage.

    Evalplus prints two blocks:
      <dataset> (base tests)
      pass@1:  0.788
      <dataset>+ (base + extra tests)
      pass@1:  0.656
    We collect all matches in order: first = base, second = plus.
    """
    all_matches = re.findall(r"pass@1\s*:\s*([0-9.]+)", stdout_text)
    # Fallback: try underscore variant if the tab-separated form wasn't found
    if not all_matches:
        all_matches = re.findall(r"pass_at_1\s*[:=]\s*([0-9.]+)", stdout_text)

    def _to_pct(v: str) -> float:
        fv = float(v)
        return fv * 100.0 if fv <= 1.0 else fv

    base = _to_pct(all_matches[0]) if len(all_matches) >= 1 else None
    plus = _to_pct(all_matches[1]) if len(all_matches) >= 2 else None
    return base, plus


def run_evalplus(
    spec: ModelSpec, dataset: str, parallel: int, out_dir: str,
    timeout: int = 7200, backend: str = "vllm",
) -> Dict[str, Any]:
    log_path = os.path.join(out_dir, f"evalplus_{sanitize_name(spec.name)}_{dataset}.log")
    cmd = ["evalplus.evaluate", "--dataset", dataset, "--greedy"]

    if spec.type == "api":
        cmd.extend(["--model", spec.path, "--backend", "openai"])
        if spec.api_base:
            cmd.extend(["--base-url", spec.api_base])
    else:
        # Use vllm backend by default: ~10-50x faster than hf (continuous batching).
        cmd.extend(["--model", spec.path, "--backend", backend])

    if parallel > 0:
        cmd.extend(["--parallel", str(parallel)])

    env = os.environ.copy()
    if spec.is_local_api and "OPENAI_API_KEY" not in env:
        env["OPENAI_API_KEY"] = "none"

    p = run_cmd(cmd, tee_to=log_path, env=env, timeout=timeout)
    base_score, plus_score = parse_evalplus_pass(p.stdout)

    return {
        "status": "ok" if p.returncode == 0 else "failed",
        "dataset": dataset,
        "pass@1": base_score,
        "plus_pass@1": plus_score,
        "returncode": p.returncode,
        "log": log_path,
    }


# ============================================================
# vLLM server management (fast evalplus via continuous batching)
# ============================================================
def _find_free_port(start: int = 18811) -> int:
    import socket
    for port in range(start, start + 40):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", port))
                return port
            except OSError:
                continue
    return start


def _start_vllm_server(
    model_path: str,
    port: int,
    max_model_len: int = 4096,
    gpu_memory_utilization: float = 0.85,
    dtype: str = "bfloat16",
) -> Tuple[subprocess.Popen, Any]:
    """Start vLLM OpenAI-compatible server as background process."""
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--served-model-name", model_path,
        "--port", str(port),
        "--dtype", dtype,
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--trust-remote-code",
        "--disable-log-requests",
        "--disable-log-stats",
    ]
    log_path = f"/tmp/vllm_server_{port}.log"
    log_fh = open(log_path, "w")
    env = os.environ.copy()
    env["TOKENIZERS_PARALLELISM"] = "false"
    print(f"[vLLM] Starting server on port {port} (log: {log_path})")
    print(f"[vLLM] cmd: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT, env=env)
    return proc, log_fh


def _wait_for_vllm_server(port: int, timeout: int = 300) -> bool:
    """Poll /health until server responds or timeout."""
    import urllib.request
    url = f"http://localhost:{port}/health"
    deadline = time.time() + timeout
    print(f"[vLLM] Waiting for server at {url} ...", flush=True)
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                if r.getcode() == 200:
                    print("[vLLM] Server ready.")
                    return True
        except Exception:
            time.sleep(3)
    print(f"[vLLM] Server did not start within {timeout}s.")
    return False


def run_evalplus_datasets_via_vllm_server(
    spec: ModelSpec,
    datasets: List[str],
    parallel: int,
    out_dir: str,
    max_model_len: int = 4096,
    gpu_memory_utilization: float = 0.85,
    timeout: int = 7200,
) -> Dict[str, Dict]:
    """
    Start ONE vLLM server, run all requested evalplus datasets against it, then
    shut it down.  Benefits vs --backend hf:
      - Model loaded exactly once (not once per dataset)
      - vLLM continuous batching: all problems processed concurrently
      - PagedAttention: much better KV-cache throughput
    Typical speedup: 10-50x vs --backend hf for a 9B model on H100.
    """
    port = _find_free_port(18811)
    proc, log_fh = _start_vllm_server(
        spec.path, port=port,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
    )

    results: Dict[str, Dict] = {}
    try:
        if not _wait_for_vllm_server(port, timeout=300):
            for ds in datasets:
                results[ds] = {"status": "failed", "error": "vLLM server did not start",
                                "dataset": ds, "pass@1": None}
            return results

        for dataset in datasets:
            log_path = os.path.join(out_dir, f"evalplus_{sanitize_name(spec.name)}_{dataset}.log")
            # evalplus.evaluate with openai backend sends concurrent requests to
            # the vLLM server, which batches them via continuous batching.
            cmd = [
                "evalplus.evaluate",
                "--dataset", dataset,
                "--greedy",
                "--model", spec.path,
                "--backend", "openai",
                "--base-url", f"http://localhost:{port}/v1",
                "--parallel", str(max(parallel, 16)),
            ]
            env = os.environ.copy()
            env.setdefault("OPENAI_API_KEY", "dummy")
            env["TOKENIZERS_PARALLELISM"] = "false"

            t0 = time.time()
            p = run_cmd(cmd, env=env, tee_to=log_path, timeout=timeout)
            elapsed = round(time.time() - t0, 1)
            base_score, plus_score = parse_evalplus_pass(p.stdout)
            results[dataset] = {
                "status": "ok" if p.returncode == 0 else "failed",
                "dataset": dataset,
                "pass@1": base_score,
                "plus_pass@1": plus_score,
                "returncode": p.returncode,
                "elapsed_seconds": elapsed,
                "log": log_path,
                "backend": "vllm_server",
            }
            print(f"[vLLM] {dataset}: pass@1={base_score}  plus_pass@1={plus_score}  elapsed={elapsed:.0f}s")

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        try:
            log_fh.close()
        except Exception:
            pass
        print(f"[vLLM] Server on port {port} stopped.")

    return results


def parse_lm_eval_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        data = json_load(path)
        return data.get("results", {})
    except Exception:
        return {}


def run_lm_eval(
    spec: ModelSpec,
    tasks: str,
    bs: int,
    dev: str,
    out_dir: str,
    timeout: int = 7200,
    lm_eval_tokenizer: Optional[str] = None,
) -> Dict[str, Any]:
    """Run lm-evaluation-harness against a model.

    lm_eval_tokenizer: explicit tokenizer path/HF-repo for the local-completions
        backend.  Required when the served model name (e.g. 'qwen-base') is just
        an alias that doesn't exist on HuggingFace Hub.  Falls back to
        spec.tokenizer_path if not given.
    """
    out_json_path = os.path.join(out_dir, f"lm_eval_{sanitize_name(spec.name)}.json")
    log_path = out_json_path.replace(".json", ".log")

    # Use _resolve_command so we work whether lm_eval is installed as a binary
    # or only as a Python module (python -m lm_eval).
    lm_eval_cmd = _resolve_command("lm_eval", "lm_eval")
    if lm_eval_cmd is None:
        return {
            "status": "unavailable",
            "tasks": tasks,
            "results": {},
            "error": "lm_eval binary/module not found. Install with: pip install lm-eval",
        }

    cmd = lm_eval_cmd + [
        "--tasks", tasks,
        "--output_path", out_json_path,
        "--confirm_run_unsafe_code",
    ]

    if spec.type == "api":
        if spec.is_local_api:
            # ── local-completions backend (vLLM / local OpenAI-compatible server) ──
            # Root cause of the qwen-base error: lm_eval's local-completions backend
            # tries to load the HF tokenizer using the *model name* passed to the API
            # (e.g. "qwen-base"). When that name is an alias rather than a real HF
            # repo ID, the HF hub download fails with 404.
            # Fix: pass tokenizer=<real_model_path> in model_args so lm_eval loads
            # the tokenizer from the correct location instead.
            model_args_parts = [f"model={spec.path}", "num_concurrent=16"]
            if spec.api_base:
                api_base = spec.api_base.strip("/")
                completions_url = (
                    api_base + "/completions"
                    if api_base.endswith("/v1")
                    else api_base + "/v1/completions"
                )
                model_args_parts.append(f"base_url={completions_url}")
            # Tokenizer override: CLI arg > spec.tokenizer_path
            tok_path = lm_eval_tokenizer or spec.tokenizer_path
            if tok_path:
                model_args_parts.append(f"tokenizer={tok_path}")
            else:
                print(
                    "[WARN] run_lm_eval: no tokenizer path found for local-completions backend. "
                    "lm_eval will try to load tokenizer from the served model name "
                    f"'{spec.path}', which may fail if it's not a valid HF repo ID.\n"
                    "Fix: pass --lm_eval_tokenizer <actual_model_path> or --prompt_tokenizer."
                )
            cmd.extend([
                "--model", "local-completions",
                "--model_args", ",".join(model_args_parts),
                "--batch_size", str(bs),
            ])
        else:
            # Remote OpenAI API
            model_args_parts = [f"model={spec.path}"]
            if spec.api_base:
                model_args_parts.append(f"base_url={spec.api_base}")
            tok_path = lm_eval_tokenizer or spec.tokenizer_path
            if tok_path:
                model_args_parts.append(f"tokenizer={tok_path}")
            cmd.extend(["--model", "openai", "--model_args", ",".join(model_args_parts)])
    else:
        # HF / local model loaded directly
        cmd.extend([
            "--model", "hf",
            "--model_args", f"pretrained={spec.path}",
            "--device", dev,
            "--batch_size", str(bs),
        ])

    env = os.environ.copy()
    env["LM_EVAL_ALLOW_CODE_EXECUTION"] = "1"
    env["HF_ALLOW_CODE_EVAL"] = "1"
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    if spec.is_local_api:
        env.setdefault("OPENAI_API_KEY", "none")

    p = run_cmd(cmd, tee_to=log_path, env=env, timeout=timeout)

    # Parse results: extract the best representative metric per task
    # (acc / exact_match / pass), following evaluator_main.py convention.
    raw_results = parse_lm_eval_json(out_json_path)
    parsed: Dict[str, Any] = {}
    for task_name, task_result in raw_results.items():
        # First pass: find primary metric (acc*, exact_match, pass*)
        best_key, best_val = None, None
        for k, v in task_result.items():
            if not isinstance(v, (int, float)):
                continue
            if any(x in k for x in ("acc", "exact_match", "pass")):
                best_key, best_val = k, v
                break
        # Second pass: keep all numeric fields for completeness
        numeric = {k: v for k, v in task_result.items() if isinstance(v, (int, float))}
        parsed[task_name] = numeric
        if best_key is not None:
            # Promote primary metric to top-level for easy access
            parsed[task_name]["_primary_metric"] = best_key
            parsed[task_name]["_primary_value"] = round(float(best_val) * 100.0
                                                         if float(best_val) <= 1.0
                                                         else float(best_val), 4)

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
            if "plus_pass@1" in val and isinstance(val["plus_pass@1"], (int, float)):
                flat[f"{key}_plus_pass@1"] = round(float(val["plus_pass@1"]), 4)
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

    # ── Per-task resume: load existing results, skip tasks that already succeeded ──
    # Unlike the old all-or-nothing approach, this allows a run that failed
    # mid-way (e.g. lm_eval OOM) to be restarted without re-running benchmarks
    # that already completed successfully.
    existing: Dict[str, Any] = {}
    if args.resume and os.path.exists(model_json_path):
        try:
            existing = json_load(model_json_path)
            _done = [k for k, v in existing.items()
                     if isinstance(v, dict) and v.get("status") == "ok"]
            print(f"[RESUME] Loaded {model_json_path}. Already-successful tasks: {_done or 'none'}")
        except Exception as e:
            print(f"[WARN] Could not load existing results ({e}), starting fresh.")
            existing = {}

    def _skip(key: str) -> bool:
        """Return True when the benchmark already produced a successful result."""
        r = existing.get(key)
        return isinstance(r, dict) and r.get("status") == "ok"

    def _checkpoint():
        """Write intermediate results to disk so a crash is recoverable."""
        json_dump(results, model_json_path)

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

    # Seed results with any previously-successful task outputs, then add/update
    # base metadata fields.
    results: Dict[str, Any] = {
        k: v for k, v in existing.items()
        if isinstance(v, dict) and v.get("status") == "ok"
    }
    results.update({
        "model_name": spec.name,
        "model_path": spec.path,
        "model_type": spec.type,
        "tokenizer_path": spec.tokenizer_path,
        # Preserve original start timestamp if resuming
        "started_at": existing.get("started_at") or now_ts(),
    })

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
                # Use batch generator for higher GPU utilisation during local eval.
                # Falls back to single-prompt path automatically for the text tokenizer.
                gen_fn = create_hf_batch_generator(
                    model, prompt_tokenizer, args.device,
                    args.humaneval_max_new_tokens,
                    batch_size=args.local_gen_batch_size,
                )
            except Exception as e:
                results["model_loading"] = {"status": "failed", "error": str(e)}

        elif spec.type == "api":
            try:
                gen_fn = create_api_generator(spec, args.humaneval_max_new_tokens, temperature=0.0)
            except Exception as e:
                results["generator_creation"] = {"status": "failed", "error": str(e)}

        # ── API speed benchmark (TTFT / decode TPS) — runs against live endpoint ──
        if spec.type == "api" and getattr(args, "run_api_speed", False):
            if _skip("api_speed"):
                print("[SKIP] api_speed: already successful.")
            else:
                print("[INFO] Running API speed benchmark (TTFT / decode TPS)...")
                prefill_list = [
                    int(x.strip())
                    for x in str(getattr(args, "api_speed_prefill_tokens", "128,512,2048")).split(",")
                    if x.strip().isdigit()
                ]
                try:
                    results["api_speed"] = run_api_speed_benchmark(
                        spec,
                        out_dir=raw_out_dir,
                        prefill_tokens=prefill_list,
                        runs=getattr(args, "api_speed_runs", 5),
                        max_output_tokens=args.humaneval_max_new_tokens,
                    )
                except Exception as e:
                    results["api_speed"] = {"status": "failed", "error": str(e)}
                _checkpoint()

        # ── local-only benchmarks (require model in GPU memory) ──
        if model is not None and args.run_benchmark:
            if _skip("benchmark"):
                print("[SKIP] benchmark: already successful.")
            else:
                try:
                    results["benchmark"] = benchmark_generation(model, prompt_tokenizer, args)
                except Exception as e:
                    results["benchmark"] = {"status": "failed", "error": str(e)}
                _checkpoint()

        if model is not None and args.run_ppl:
            for ppl_key, ppl_src in [("ppl_wikitext", "wikitext"), ("ppl_mbpp", "mbpp")]:
                if _skip(ppl_key):
                    print(f"[SKIP] {ppl_key}: already successful.")
                else:
                    try:
                        results[ppl_key] = perplexity_sliding_window(model, prompt_tokenizer, ppl_src, args)
                    except Exception as e:
                        results[ppl_key] = {"status": "failed", "error": str(e)}
                    _checkpoint()

        if model is not None and args.run_induction:
            if _skip("induction"):
                print("[SKIP] induction: already successful.")
            else:
                try:
                    results["induction"] = induction_repeat_nll(model, prompt_tokenizer, args)
                except Exception as e:
                    results["induction"] = {"status": "failed", "error": str(e)}
                _checkpoint()

        if model is not None and args.run_passkey:
            if _skip("passkey"):
                print("[SKIP] passkey: already successful.")
            else:
                try:
                    results["passkey"] = passkey_retrieval_acc(model, prompt_tokenizer, args)
                except Exception as e:
                    results["passkey"] = {"status": "failed", "error": str(e)}
                _checkpoint()

        if gen_fn is None:
            results["codegen_status"] = "skipped_no_generator"
            _checkpoint()
            return results

        humaneval_samples_path = os.path.join(raw_out_dir, f"{sanitize_name(spec.name)}_humaneval_samples.jsonl")
        mbpp_samples_path = os.path.join(raw_out_dir, f"{sanitize_name(spec.name)}_mbpp_samples.jsonl")

        if args.run_humaneval:
            if _skip("humaneval"):
                print(f"[SKIP] humaneval: already successful (pass@1={existing.get('humaneval', {}).get('pass@1')})")
            else:
                print("Running Human Eval...")
                try:
                    results["humaneval"] = humaneval_pass_at_1(
                        gen_fn, prompt_tokenizer, args.humaneval_n, args, humaneval_samples_path
                    )
                except Exception as e:
                    results["humaneval"] = {"status": "failed", "error": str(e)}
                _checkpoint()

        if args.run_mbpp:
            if _skip("mbpp"):
                print(f"[SKIP] mbpp: already successful (pass@1={existing.get('mbpp', {}).get('pass@1')})")
            else:
                print("Running MBPP...")
                try:
                    results["mbpp"] = mbpp_pass_at_1(
                        gen_fn, prompt_tokenizer, args.mbpp_n, args, mbpp_samples_path
                    )
                except Exception as e:
                    results["mbpp"] = {"status": "failed", "error": str(e)}
                _checkpoint()

        # Clear VRAM before running external evaluators (evalplus / lm_eval
        # need the full GPU memory for their own model loading / vLLM server).
        _needs_vram_clear = (
            (args.run_evalplus or args.run_lm_eval)
            and model is not None
            # Skip VRAM clear if all remaining external benchmarks will be skipped
            and not (
                _skip("evalplus_humaneval") and _skip("evalplus_mbpp") and _skip("lm_eval")
            )
        )
        if _needs_vram_clear:
            print("\n[INFO] Local generation complete. Clearing VRAM before running external evaluators...")
            cleanup_local_model(model, tokenizer)
            model = None
            tokenizer = None
            prompt_tokenizer = None
            gen_fn = None

        if args.run_evalplus and spec.type != "api":
            # Determine which evalplus datasets still need to run
            pending_datasets = [
                ds for ds in ["humaneval", "mbpp"]
                if not _skip(f"evalplus_{ds}")
            ]
            skipped_datasets = [ds for ds in ["humaneval", "mbpp"] if _skip(f"evalplus_{ds}")]
            for ds in skipped_datasets:
                print(f"[SKIP] evalplus_{ds}: already successful "
                      f"(pass@1={existing.get(f'evalplus_{ds}', {}).get('pass@1')})")

            if pending_datasets:
                # ── Fast path: single vLLM server, all pending datasets share one load ──
                try:
                    vllm_results = run_evalplus_datasets_via_vllm_server(
                        spec,
                        pending_datasets,
                        parallel=args.evalplus_parallel,
                        out_dir=raw_out_dir,
                        max_model_len=args.max_seq_length,
                        gpu_memory_utilization=args.evalplus_vllm_gpu_util,
                        timeout=args.evalplus_timeout,
                    )
                    for ds in pending_datasets:
                        results[f"evalplus_{ds}"] = vllm_results.get(ds, {})
                    _checkpoint()
                except Exception as e:
                    print(f"[WARN] vLLM server path failed ({e}). Falling back to --backend vllm subprocess.")
                    for ds in pending_datasets:
                        key = f"evalplus_{ds}"
                        try:
                            results[key] = run_evalplus(
                                spec, ds, args.evalplus_parallel,
                                raw_out_dir, timeout=args.evalplus_timeout,
                                backend="vllm",
                            )
                        except Exception as e2:
                            results[key] = {"status": "failed", "error": str(e2)}
                        _checkpoint()

        elif args.run_evalplus and spec.type == "api":
            # API models: use openai backend directly (no vLLM server needed)
            for dataset in ["humaneval", "mbpp"]:
                key = f"evalplus_{dataset}"
                if _skip(key):
                    print(f"[SKIP] {key}: already successful "
                          f"(pass@1={existing.get(key, {}).get('pass@1')})")
                    continue
                try:
                    results[key] = run_evalplus(
                        spec, dataset, args.evalplus_parallel,
                        raw_out_dir, timeout=args.evalplus_timeout,
                    )
                except Exception as e:
                    results[key] = {"status": "failed", "error": str(e)}
                _checkpoint()

        if args.run_lm_eval:
            if _skip("lm_eval"):
                print("[SKIP] lm_eval: already successful.")
            else:
                try:
                    # lm_eval_tokenizer overrides prompt_tokenizer specifically for lm_eval.
                    # Needed when the vLLM served-model-name is an alias (e.g. "qwen-base")
                    # that is not a valid HF repo ID — see run_lm_eval() docstring.
                    _lm_tok = getattr(args, "lm_eval_tokenizer", None) or spec.tokenizer_path
                    results["lm_eval"] = run_lm_eval(
                        spec,
                        args.lm_eval_tasks,
                        args.lm_eval_batch_size,
                        args.device,
                        raw_out_dir,
                        timeout=args.lm_eval_timeout,
                        lm_eval_tokenizer=_lm_tok,
                    )
                except Exception as e:
                    results["lm_eval"] = {"status": "failed", "error": str(e)}
                _checkpoint()

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
    parser.add_argument(
        "--run_api_speed", action="store_true",
        help="Run async API speed benchmark (TTFT + decode TPS) for API models. "
             "Ported from evaluator_main.py. Useful for comparing base vs SFT model throughput.",
    )
    parser.add_argument(
        "--api_speed_prefill_tokens", type=str, default="128,512,2048",
        help="Comma-separated list of prefill token counts for API speed benchmark (default: 128,512,2048).",
    )
    parser.add_argument("--api_speed_runs", type=int, default=5,
                        help="Number of runs per prefill length for API speed benchmark (default 5).")

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

    parser.add_argument("--evalplus_parallel", type=int, default=16,
                        help="Concurrent test-execution workers AND concurrent API requests to vLLM server.")
    parser.add_argument("--evalplus_timeout", type=int, default=7200)
    parser.add_argument("--evalplus_vllm_gpu_util", type=float, default=0.85,
                        help="GPU memory fraction for the vLLM server (default 0.85).")

    # Local generation batch size: larger = fewer kernel launches, faster throughput.
    # Reduce if you hit OOM during local humaneval/mbpp generation.
    parser.add_argument("--local_gen_batch_size", type=int, default=8,
                        help="Batch size for local HF generation (humaneval/mbpp). Default 8.")

    parser.add_argument("--lm_eval_tasks", type=str, default=None)
    parser.add_argument("--lm_eval_batch_size", type=int, default=8)
    parser.add_argument("--lm_eval_timeout", type=int, default=7200)
    parser.add_argument(
        "--lm_eval_tokenizer", type=str, default=None,
        help=(
            "Tokenizer path / HF repo ID for lm_eval's local-completions backend.\n"
            "Required when the vLLM served-model-name (e.g. 'qwen-base') is an alias\n"
            "that does not exist on HuggingFace Hub.  lm_eval would otherwise try to\n"
            "download the tokenizer from HF using the served name and get a 404 error.\n"
            "Example: --lm_eval_tokenizer Qwen/Qwen2.5-Coder-7B-Instruct\n"
            "Falls back to --prompt_tokenizer if not set."
        ),
    )

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
        args.run_api_speed = True

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
# ── EXAMPLE 1: Evaluate a vLLM-served model with lm_eval (the 'qwen-base' fix) ──
# 'qwen-base' is the --served-model-name passed to vLLM.
# --lm_eval_tokenizer points to the *real* model so lm_eval can load the tokenizer.
python CodeAgent/qwen_coder_evalv5_1.py \
  --models openai:qwen-base \
  --api_base http://127.0.0.1:8000/v1 \
  --prompt_tokenizer Qwen/Qwen2.5-Coder-7B-Instruct \
  --lm_eval_tokenizer Qwen/Qwen2.5-Coder-7B-Instruct \
  --out_dir eval_api_test \
  --run_humaneval \
  --run_mbpp \
  --run_evalplus \
  --run_lm_eval \
  --lm_eval_tasks humaneval \
  --make_figures

# ── EXAMPLE 2: Resume a failed run (skips already-successful benchmarks) ──
python CodeAgent/qwen_coder_evalv5_1.py \
  --models openai:qwen-base \
  --api_base http://127.0.0.1:8000/v1 \
  --lm_eval_tokenizer Qwen/Qwen2.5-Coder-7B-Instruct \
  --out_dir eval_api_test \
  --resume \
  --run_lm_eval \
  --lm_eval_tasks humaneval

# ── EXAMPLE 3: Base vs SFT comparison (local HF models) ──
python CodeAgent/qwen_coder_evalv5_1.py \
  --models \
    Qwen/Qwen2.5-Coder-7B-Instruct \
    output/qwen_sft_merged \
  --out_dir eval_base_vs_sft \
  --run_humaneval \
  --run_mbpp \
  --run_evalplus \
  --run_lm_eval \
  --lm_eval_tasks humaneval \
  --make_figures

# ── EXAMPLE 4: API speed benchmark (TTFT / decode TPS from evaluator_main.py) ──
python CodeAgent/qwen_coder_evalv5_1.py \
  --models openai:qwen-base \
  --api_base http://127.0.0.1:8000/v1 \
  --lm_eval_tokenizer Qwen/Qwen2.5-Coder-7B-Instruct \
  --out_dir eval_speed_test \
  --run_api_speed \
  --api_speed_prefill_tokens 128,512,2048 \
  --api_speed_runs 5
  """

import os
import re
import json
import asyncio
import csv
import math
import statistics
import time
import argparse
import sys
import random
import subprocess
import shutil
import importlib.util
import select
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# from CodeAgent.llm_wrapper import complete_with_async, compute_stream_speed_metrics
# from CodeAgent.mini_batch_agent_libs import now_stamp, estimate_tokens

console = Console()

try:
    from openai import AsyncOpenAI as _AsyncOpenAI
except ImportError:
    _AsyncOpenAI = None

LLM_REQUEST_ERRORS = (
    httpx.HTTPError,
    asyncio.TimeoutError,
    RuntimeError,
    ValueError,
    TypeError,
    KeyError,
)
SUBPROCESS_ERRORS = (subprocess.TimeoutExpired, FileNotFoundError, OSError, ValueError)


# ==========================================
# Simple Single-Shot Async Wrapper (No Continuation)
# ==========================================
import sys
import json
from typing import Any, List, Dict, Tuple, Optional
from typing import List, Dict, Any, Optional, Tuple, Callable, Awaitable
async def _execute_openai_async(
    client: Any, model: str, messages: List[Dict[str, str]], 
    temperature: float, max_tokens: int, stream: bool,
    tools: Optional[List[Dict[str, Any]]] = None,
    verbose: bool = False,
    on_event: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None,
    backend: str = "openai",
    enable_thinking: bool = True,
) -> Tuple[str, str, Dict[str, int], List[Dict[str, Any]]]:
    
    kwargs = {
        "model": model, "messages": messages, "temperature": temperature,
        "max_tokens": max_tokens, "stream": stream,
    }
    
    # ── Inject specific parameters for vllm/llama.cpp ──
    if backend in ["llama.cpp", "vllm"] and enable_thinking is not None:
        kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": enable_thinking}}
        kwargs["stop"] = ["<|im_end|>", "<|im_start|>", "<|endoftext|>"]
        
    if tools: kwargs["tools"] = tools
    if stream: kwargs["stream_options"] = {"include_usage": True}
        
    resp = await client.chat.completions.create(**kwargs)
    
    content = ""
    finish_reason = "stop"
    usage_info = {}
    native_tool_calls = []
    tc_dict = {}
    
    if stream:
        chunk_counter = 0
        
        # State machine for suppressing XML tool prints & handling thinking
        in_think = False
        in_tool = False
        buffer = ""
        tool_args_buffer = ""
        tool_name_buffer = ""
        
        async for chunk in resp:
            chunk_counter += 1
            
            if not chunk.choices:
                if hasattr(chunk, 'usage') and chunk.usage:
                    usage_info = {"prompt_tokens": chunk.usage.prompt_tokens, "completion_tokens": chunk.usage.completion_tokens}
                continue
                
            delta = chunk.choices[0].delta
            
            # --- 1. Extract Llama.cpp / Custom backends reasonings ---
            reasoning = delta.model_dump().get("reasoning_content")
            if reasoning:
                if verbose:
                    sys.stdout.write(f"\033[90m{reasoning}\033[0m")
                    sys.stdout.flush()
                if on_event:
                    await on_event({"type": "think", "data": reasoning})
            
            # --- 2. Extract standard content ---
            if delta.content:
                text_chunk = delta.content
                content += text_chunk
                buffer += text_chunk
                
                # Tag detector
                if not in_think and "<think>" in buffer:
                    in_think = True
                    buffer = buffer.split("<think>")[-1]
                    
                if in_think and "</think>" in buffer:
                    in_think = False
                    buffer = buffer.split("</think>")[-1]
                    
                if not in_tool and "<tool_call>" in buffer:
                    in_tool = True
                    buffer = buffer.split("<tool_call>")[-1]
                    
                if in_tool and "</tool_call>" in buffer:
                    in_tool = False
                    buffer = ""
                    tool_args_buffer = ""
                    tool_name_buffer = ""
                
                # Strip XML control tags from the visible text before emitting
                clean_chunk = re.sub(r'</?think>|</?tool_call>', '', text_chunk)

                # Text processing logic
                if in_think:
                    if verbose:
                        sys.stdout.write(f"\033[90m{clean_chunk}\033[0m")
                        sys.stdout.flush()
                    if on_event and clean_chunk:
                        await on_event({"type": "think", "data": clean_chunk})
                elif in_tool:
                    # Accumulate tool text silently
                    tool_args_buffer += text_chunk

                    # Try to extract the tool name if we haven't yet (simple regex logic)
                    if not tool_name_buffer and ">" in tool_args_buffer:
                        # e.g., <web_search> or <write_file>
                        match = re.search(r"<([a-zA-Z0-9_]+)>", tool_args_buffer)
                        if match:
                            tool_name_buffer = match.group(1)
                            if verbose:
                                console.print(f"\n[bold magenta]🛠️ Parsing Tool: {tool_name_buffer}...[/bold magenta]")

                    if on_event:
                        await on_event({"type": "tool", "status": "streaming", "data": text_chunk})
                else:
                    # Normal message token — strip any stray tag fragments before emitting
                    if clean_chunk:
                        if verbose:
                            sys.stdout.write(clean_chunk)
                            sys.stdout.flush()
                        if on_event:
                            await on_event({"type": "message", "data": clean_chunk})
                
                        
            # (后续解析 Tool Call 的逻辑保持不变)
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tc_dict:
                        func_name = tc.function.name if tc.function and tc.function.name else "unknown_tool"
                        tc_dict[idx] = {"name": func_name, "arguments": ""}
                        if on_event:
                            await on_event({"type": "tool", "name": func_name, "status": "started"})
                        if verbose:
                            console.print(f"\n[bold magenta]🛠️ Calling Tool: {func_name}...[/bold magenta]")
                    
                    if tc.function and tc.function.arguments:
                        chunk_arg = tc.function.arguments
                        tc_dict[idx]["arguments"] += chunk_arg
                        if on_event:
                            await on_event({"type": "tool", "name": tc_dict[idx]["name"], "args_delta": chunk_arg, "status": "streaming"})
                            
                # Optionally print fully completed arguments when finish_reason hits, but let's wait until outside the loop for summary if needed.
                        
            if chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason
                
        if verbose and content and finish_reason != "repetition":
            print()
            
        native_tool_calls = list(tc_dict.values())
        
    else:
        # Non-streaming parsing
        msg = resp.choices[0].message
        content = msg.content or ""
        finish_reason = resp.choices[0].finish_reason or "stop"
        
        if hasattr(resp, 'usage') and resp.usage:
            usage_info = {
                "prompt_tokens": resp.usage.prompt_tokens, 
                "completion_tokens": resp.usage.completion_tokens
            }
            
        if msg.tool_calls:
            for tc in msg.tool_calls:
                native_tool_calls.append({
                    "name": tc.function.name, 
                    "arguments": tc.function.arguments
                })
                
    return content, finish_reason, usage_info, native_tool_calls


async def complete_with_async(
    client: Any,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
    max_output_tokens: int = 4096,
    model_max_context: int = 16384,
    provider: str = "openai",
    stream: bool = True,
    verbose: bool = False,
    on_event: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None,
    backend: str = "openai",
    enable_thinking: bool = True,
) -> Tuple[str, Dict[str, Any]]:
    """
    Simple single-shot async wrapper for LLM API calls.

    Unlike complete_with_continuation_async, this function:
    - Does NOT loop on finish_reason == 'length' (no auto-continuation)
    - Does NOT parse tool calls or agent actions
    - Returns (content: str, usage_info: dict)

    The optional `on_token` async callback receives each streamed token as it
    arrives, enabling FastAPI SSE endpoints to forward tokens to clients in
    real-time without any extra buffering.

    Args:
        client:            Async OpenAI-compatible client (or Anthropic client)
        model:             Model name
        messages:          List of chat messages (role/content dicts)
        temperature:       Sampling temperature
        max_output_tokens: Maximum completion tokens to request
        model_max_context: Total context window size (for token budget calc)
        provider:          'openai' or 'anthropic'
        stream:            Enable streaming API (token-by-token)
        verbose:           Print tokens to terminal as they arrive
        on_token:          Optional async callback called for each streamed token.
                           Signature: async def on_token(token: str) -> None

    Returns:
        (content, usage_info)  where usage_info has 'prompt_tokens',
        'completion_tokens', 'elapsed_seconds', 'tokens_per_second',
        and 'finish_reason' keys.
    """
    # --- 1. Adaptive Token Budget ---
    input_text = "\n".join(m.get("content", "") for m in messages)
    input_est = estimate_tokens(input_text)
    min_output = 256
    max_allowed_input = model_max_context - 1000 - min_output

    safe_tokens = max_output_tokens #compute_safe_max_tokens(input_est, model_max_context, max_output_tokens, min_output)

    # --- 2. Single API Call with Retries ---
    content: str = ""
    finish_reason: str = "stop"
    usage_info: Dict[str, Any] = {}
    start_time = time.time()

    for attempt in range(3):
        try:
            content, finish_reason, usage_info, _ = await _execute_openai_async(
                client, model, messages, temperature, safe_tokens,
                stream=stream, tools=None, verbose=verbose, on_event=on_event,
                backend=backend, enable_thinking=enable_thinking
            )
        except Exception as e:
            err_str = str(e)
            if "max_tokens" in err_str or "context length" in err_str:
                safe_tokens = max(1024, safe_tokens // 2)
                console.print(f"[red]Context overflow. Retrying max_tokens={safe_tokens}[/red]")
                await asyncio.sleep(1)
                continue
            console.print(f"[red]LLM Call failed (attempt {attempt + 1}): {e}[/red]")
            if attempt < 2:
                await asyncio.sleep(2 ** attempt)
                continue
            break  # Hard fail – return whatever we have

    # --- 3. Metrics ---
    elapsed = time.time() - start_time
    
    console.print(
        f"[bold blue][LLM][/bold blue] [dim]"
        f"{usage_info['prompt_tokens']}P, {usage_info['completion_tokens']}C | "
        #f"{speed_metrics['e2e_tokens_per_second']:.1f} T/s | {elapsed:.1f}s | finish={finish_reason}[/dim]"
    )

    usage_info["elapsed_seconds"] = round(elapsed, 2)
    # usage_info["tokens_per_second"] = round(speed_metrics["e2e_tokens_per_second"], 1)
    # usage_info["prefill_tokens_per_second"] = round(speed_metrics["prefill_tokens_per_second"], 3)
    # usage_info["decode_tokens_per_second"] = round(speed_metrics["decode_tokens_per_second"], 3)
    # usage_info["per_token_decode_latency_ms"] = round(speed_metrics["per_token_decode_latency_ms"], 3)
    usage_info["finish_reason"] = finish_reason

    return content, usage_info


@dataclass
class APICase:
    name: str
    base_url: str
    model: str
    api_key: str
    backend: str
    provider: str = "openai"
    enable_thinking: bool = False


def _parse_csv_list(raw: str) -> List[str]:
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def _require_async_openai() -> Any:
    if _AsyncOpenAI is None:
        raise ModuleNotFoundError("openai package is required for LLM evaluation. Install with: pip install openai")
    return _AsyncOpenAI


def _parse_csv_int_list(raw: str) -> List[int]:
    values: List[int] = []
    for item in _parse_csv_list(raw):
        try:
            value = int(item)
            if value > 0:
                values.append(value)
        except ValueError:
            continue
    return values


def _parse_case_entry(raw: str, defaults: Dict[str, str]) -> APICase:
    parts = [x.strip() for x in raw.split(",") if x.strip()]
    kv: Dict[str, str] = {}
    for part in parts:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        kv[key.strip().lower()] = value.strip()
    name = kv.get("name") or kv.get("label")
    base_url = kv.get("url") or kv.get("base_url")
    if not name or not base_url:
        raise ValueError(f"Invalid --llm-case '{raw}'. Required keys: name,url")
    return APICase(
        name=name,
        base_url=base_url,
        model=kv.get("model", defaults["model"]),
        api_key=kv.get("api_key", defaults["api_key"]),
        backend=kv.get("backend", defaults["backend"]),
        provider=kv.get("provider", defaults["provider"]),
        enable_thinking=kv.get("enable_thinking", "false").lower() in {"1", "true", "yes", "y"},
    )


def _build_llm_cases(args: argparse.Namespace) -> List[APICase]:
    defaults = {
        "model": args.model,
        "api_key": args.api_key,
        "backend": args.backend,
        "provider": args.provider,
    }
    if args.llm_case:
        return [_parse_case_entry(raw, defaults) for raw in args.llm_case]
    if args.base_urls:
        urls = _parse_csv_list(args.base_urls)
        return [
            APICase(
                name=f"case_{idx+1}",
                base_url=url,
                model=args.model,
                api_key=args.api_key,
                backend=args.backend,
                provider=args.provider,
                enable_thinking=args.enable_thinking,
            )
            for idx, url in enumerate(urls)
        ]
    return [
        APICase(
            name=args.base_url_name,
            base_url=args.base_url,
            model=args.model,
            api_key=args.api_key,
            backend=args.backend,
            provider=args.provider,
            enable_thinking=args.enable_thinking,
        ),
        APICase(
            name=args.base_url_alt_name,
            base_url=args.base_url_alt,
            model=args.model,
            api_key=args.api_key,
            backend=args.backend,
            provider=args.provider,
            enable_thinking=args.enable_thinking,
        ),
    ]


def _percentile(values: List[float], pct: float) -> float:
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


def _build_prefill_text(target_tokens: int) -> str:
    seed = (
        "Autoregressive decoding benchmark context. "
        "Measure prompt ingestion speed, decode speed, and latency distributions. "
        "Keep semantic coherence while expanding token count for prefill stress testing."
    )
    chunks: List[str] = []
    while estimate_tokens("\n".join(chunks)) < target_tokens:
        chunks.append(seed)
    return "\n".join(chunks)


def _write_rows_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _summarize_speed_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_runs = len(rows)
    ok = [r for r in rows if r.get("success")]
    if not ok:
        return {
            "runs": total_runs,
            "successful_runs": 0,
            "success_rate": 0.0,
            "ttft_p50_sec": 0.0,
            "ttft_p95_sec": 0.0,
            "ttft_p99_sec": 0.0,
            "e2e_p50_sec": 0.0,
            "e2e_p95_sec": 0.0,
            "e2e_p99_sec": 0.0,
            "e2e_speed_p50_tokens_per_sec": 0.0,
            "decode_p50_tokens_per_sec": 0.0,
            "decode_p95_tokens_per_sec": 0.0,
            "decode_p99_tokens_per_sec": 0.0,
            "prefill_mean_tokens_per_sec": 0.0,
            "per_token_decode_latency_p50_ms": 0.0,
            "per_token_decode_latency_p95_ms": 0.0,
            "per_token_decode_latency_p99_ms": 0.0,
            "throughput_tokens_per_sec": 0.0,
        }
    ttft_vals = [float(r["ttft_sec"]) for r in ok]
    e2e_vals = [float(r["e2e_latency_sec"]) for r in ok]
    e2e_tps_vals = [float(r["e2e_tokens_per_sec"]) for r in ok]
    decode_tps_vals = [float(r["decode_tokens_per_sec"]) for r in ok]
    prefill_tps_vals = [float(r["prefill_tokens_per_sec"]) for r in ok]
    per_token_ms_vals = [float(r["per_token_decode_latency_ms"]) for r in ok if r.get("per_token_decode_latency_ms") is not None]
    total_completion_tokens = sum(int(r["completion_tokens"]) for r in ok)
    total_elapsed = sum(float(r["e2e_latency_sec"]) for r in ok)
    return {
        "runs": total_runs,
        "successful_runs": len(ok),
        "success_rate": round(len(ok) / total_runs, 4) if total_runs else 0.0,
        "ttft_p50_sec": round(_percentile(ttft_vals, 50), 4),
        "ttft_p95_sec": round(_percentile(ttft_vals, 95), 4),
        "ttft_p99_sec": round(_percentile(ttft_vals, 99), 4),
        "e2e_p50_sec": round(_percentile(e2e_vals, 50), 4),
        "e2e_p95_sec": round(_percentile(e2e_vals, 95), 4),
        "e2e_p99_sec": round(_percentile(e2e_vals, 99), 4),
        "e2e_speed_p50_tokens_per_sec": round(_percentile(e2e_tps_vals, 50), 3),
        "decode_p50_tokens_per_sec": round(_percentile(decode_tps_vals, 50), 3),
        "decode_p95_tokens_per_sec": round(_percentile(decode_tps_vals, 95), 3),
        "decode_p99_tokens_per_sec": round(_percentile(decode_tps_vals, 99), 3),
        "prefill_mean_tokens_per_sec": round(statistics.fmean(prefill_tps_vals), 3),
        "per_token_decode_latency_p50_ms": round(_percentile(per_token_ms_vals, 50), 3) if per_token_ms_vals else 0.0,
        "per_token_decode_latency_p95_ms": round(_percentile(per_token_ms_vals, 95), 3) if per_token_ms_vals else 0.0,
        "per_token_decode_latency_p99_ms": round(_percentile(per_token_ms_vals, 99), 3) if per_token_ms_vals else 0.0,
        "throughput_tokens_per_sec": round((total_completion_tokens / total_elapsed) if total_elapsed > 0 else 0.0, 3),
    }


def _build_speed_report_html(summary_by_case: Dict[str, Any], summary_by_prefill: Dict[str, Any], raw_rows: List[Dict[str, Any]]) -> str:
    summary_json = json.dumps(summary_by_case)
    summary_prefill_json = json.dumps(summary_by_prefill)
    raw_rows_json = json.dumps(raw_rows)
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>LLM Speed Benchmark Report</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{
      font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 0;
      background: #0b1020;
      color: #e5e7eb;
    }}
    .container {{
      max-width: 1400px;
      margin: 0 auto;
      padding: 24px;
    }}
    h1, h2 {{
      margin: 8px 0 12px 0;
      font-weight: 700;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
    }}
    .card {{
      background: #111827;
      border: 1px solid #1f2937;
      border-radius: 12px;
      padding: 12px;
    }}
    .full {{
      grid-column: 1 / -1;
    }}
    .meta {{
      color: #9ca3af;
      font-size: 13px;
      margin-bottom: 16px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    th, td {{
      padding: 6px 8px;
      border-bottom: 1px solid #1f2937;
      text-align: right;
    }}
    th:first-child, td:first-child {{
      text-align: left;
    }}
  </style>
</head>
<body>
  <div class="container">
    <h1>LLM Inference Speed Benchmark</h1>
    <div class="meta">Metrics: TTFT, End-to-End Latency, E2E Speed, Decode Speed, Prefill Speed, Per-token Decode Latency, P95/P99, Throughput</div>
    <div class="grid">
      <div class="card"><div id="ttft_chart"></div></div>
      <div class="card"><div id="e2e_chart"></div></div>
      <div class="card"><div id="decode_chart"></div></div>
      <div class="card"><div id="prefill_chart"></div></div>
      <div class="card"><div id="throughput_chart"></div></div>
      <div class="card"><div id="e2e_speed_chart"></div></div>
      <div class="card full"><div id="prefill_curve"></div></div>
      <div class="card full">
        <h2>Case Summary</h2>
        <table id="summary_table"></table>
      </div>
    </div>
  </div>
  <script>
    const summaryByCase = {summary_json};
    const summaryByPrefill = {summary_prefill_json};
    const rawRows = {raw_rows_json};
    const labels = Object.keys(summaryByCase);
    const ttftP95 = labels.map(k => summaryByCase[k].ttft_p95_sec);
    const e2eP95 = labels.map(k => summaryByCase[k].e2e_p95_sec);
    const decodeP50 = labels.map(k => summaryByCase[k].decode_p50_tokens_per_sec);
    const prefillMean = labels.map(k => summaryByCase[k].prefill_mean_tokens_per_sec);
    const throughput = labels.map(k => summaryByCase[k].throughput_tokens_per_sec);
    const e2eSpeed = labels.map(k => summaryByCase[k].e2e_speed_p50_tokens_per_sec);
    const darkLayout = {{
      paper_bgcolor: '#111827',
      plot_bgcolor: '#111827',
      font: {{color: '#e5e7eb'}},
      margin: {{l: 60, r: 20, t: 40, b: 60}},
    }};
    Plotly.newPlot('ttft_chart', [{{ x: labels, y: ttftP95, type: 'bar', marker: {{color: '#60a5fa'}} }}], {{...darkLayout, title: 'TTFT P95', yaxis: {{title: 'Seconds'}}}});
    Plotly.newPlot('e2e_chart', [{{ x: labels, y: e2eP95, type: 'bar', marker: {{color: '#f97316'}} }}], {{...darkLayout, title: 'End-to-End Latency P95', yaxis: {{title: 'Seconds'}}}});
    Plotly.newPlot('decode_chart', [{{ x: labels, y: decodeP50, type: 'bar', marker: {{color: '#34d399'}} }}], {{...darkLayout, title: 'Decode Speed P50', yaxis: {{title: 'Tokens/s'}}}});
    Plotly.newPlot('prefill_chart', [{{ x: labels, y: prefillMean, type: 'bar', marker: {{color: '#22d3ee'}} }}], {{...darkLayout, title: 'Prefill Speed Mean', yaxis: {{title: 'Tokens/s'}}}});
    Plotly.newPlot('throughput_chart', [{{ x: labels, y: throughput, type: 'bar', marker: {{color: '#a78bfa'}} }}], {{...darkLayout, title: 'Throughput', yaxis: {{title: 'Tokens/s'}}}});
    Plotly.newPlot('e2e_speed_chart', [{{ x: labels, y: e2eSpeed, type: 'bar', marker: {{color: '#facc15'}} }}], {{...darkLayout, title: 'End-to-End Speed P50', yaxis: {{title: 'Tokens/s'}}}});
    const grouped = {{}};
    Object.entries(summaryByPrefill).forEach(([k, v]) => {{
      const sep = k.lastIndexOf('::');
      if (sep < 0) return;
      const name = k.slice(0, sep);
      const prefill = Number(k.slice(sep + 2));
      if (!grouped[name]) grouped[name] = [];
      grouped[name].push({{prefill, ttft: v.ttft_p50_sec, decode: v.decode_p50_tokens_per_sec}});
    }});
    const traces = Object.entries(grouped).map(([name, arr]) => {{
      arr.sort((a, b) => a.prefill - b.prefill);
      return {{ x: arr.map(x => x.prefill), y: arr.map(x => x.ttft), mode: 'lines+markers', name: name + ' TTFT P50' }};
    }});
    Plotly.newPlot('prefill_curve', traces, {{...darkLayout, title: 'TTFT vs Prefill Tokens', xaxis: {{title: 'Prefill Tokens'}}, yaxis: {{title: 'TTFT P50 (s)'}}}});
    const table = document.getElementById('summary_table');
    const headers = ['Case', 'Success', 'TTFT P95', 'E2E P95', 'E2E Speed P50', 'Prefill Mean', 'Decode P50', 'Per-token P95 ms', 'Throughput'];
    const thead = document.createElement('thead');
    const trh = document.createElement('tr');
    headers.forEach(h => {{ const th = document.createElement('th'); th.textContent = h; trh.appendChild(th); }});
    thead.appendChild(trh);
    table.appendChild(thead);
    const tbody = document.createElement('tbody');
    labels.forEach(k => {{
      const s = summaryByCase[k];
      const tr = document.createElement('tr');
      [k, `${{s.successful_runs}}/${{s.runs}}`, s.ttft_p95_sec, s.e2e_p95_sec, s.e2e_speed_p50_tokens_per_sec, s.prefill_mean_tokens_per_sec, s.decode_p50_tokens_per_sec, s.per_token_decode_latency_p95_ms, s.throughput_tokens_per_sec].forEach((v, idx) => {{
        const td = document.createElement('td');
        td.textContent = String(v);
        if (idx === 0) td.style.textAlign = 'left';
        tr.appendChild(td);
      }});
      tbody.appendChild(tr);
    }});
    table.appendChild(tbody);
  </script>
</body>
</html>"""


async def run_llm_speed_evaluation(args: argparse.Namespace, report_dir: Path, cases: List[APICase]) -> Dict[str, Any]:
    prefill_targets = _parse_csv_int_list(args.prefill_tokens)
    if not prefill_targets:
        raise ValueError("No prefill token targets configured.")
    rows: List[Dict[str, Any]] = []
    for case in cases:
        console.print(f"[bold cyan]Speed benchmark case:[/bold cyan] {case.name} ({case.base_url})")
        http_client = httpx.AsyncClient(timeout=float(args.timeout_seconds))
        async_openai_cls = _require_async_openai()
        client = async_openai_cls(base_url=case.base_url, api_key=case.api_key, http_client=http_client)
        try:
            for prefill_target in prefill_targets:
                prompt_context = _build_prefill_text(prefill_target)
                for run_idx in range(args.runs):
                    first_token_ts: Optional[float] = None
                    event_chunks = 0

                    async def on_event(evt: Dict[str, Any]) -> None:
                        nonlocal first_token_ts, event_chunks
                        evt_type = str(evt.get("type", ""))
                        if evt_type not in {"message", "think"}:
                            return
                        if not str(evt.get("data", "")):
                            return
                        event_chunks += 1
                        ts = time.perf_counter()
                        if first_token_ts is None:
                            first_token_ts = ts

                    messages = [
                        {"role": "system", "content": "You are a precise and concise assistant."},
                        {"role": "user", "content": f"{prompt_context}\n\nTask: Summarize the technical context in five concise bullets. Do not use markdown tables."},
                    ]
                    started = time.perf_counter()
                    content = ""
                    usage_info: Dict[str, Any] = {}
                    error_text = ""
                    finish_reason = "error"
                    try:
                        content, usage_info = await complete_with_async(
                            client=client,
                            model=case.model,
                            messages=messages,
                            temperature=0.0,
                            max_output_tokens=args.max_output_tokens,
                            model_max_context=args.model_max_context,
                            provider=case.provider,
                            stream=True,
                            verbose=False,
                            on_event=on_event,
                            backend=case.backend,
                            enable_thinking=case.enable_thinking,
                        )
                        finish_reason = str(usage_info.get("finish_reason", "stop"))
                    except LLM_REQUEST_ERRORS as e:
                        error_text = str(e)
                    elapsed = time.perf_counter() - started
                    prompt_tokens = int(usage_info.get("prompt_tokens", estimate_tokens("\n".join(m["content"] for m in messages))))
                    completion_tokens = int(usage_info.get("completion_tokens", estimate_tokens(content)))
                    ttft_sec = (first_token_ts - started) if first_token_ts is not None else elapsed
                    speed_metrics = compute_stream_speed_metrics(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        elapsed_seconds=elapsed,
                        ttft_seconds=ttft_sec,
                    )
                    success = (error_text == "") and (completion_tokens > 0 or len(content.strip()) > 0)
                    if not success and error_text == "":
                        error_text = "Empty completion"
                    row = {
                        "case_name": case.name,
                        "endpoint": case.base_url,
                        "model": case.model,
                        "run_index": run_idx + 1,
                        "prefill_target_tokens": prefill_target,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "ttft_sec": round(speed_metrics["ttft_seconds"], 6),
                        "e2e_latency_sec": round(elapsed, 6),
                        "decode_latency_sec": round(speed_metrics["decode_latency_seconds"], 6),
                        "e2e_tokens_per_sec": round(speed_metrics["e2e_tokens_per_second"], 6),
                        "prefill_tokens_per_sec": round(speed_metrics["prefill_tokens_per_second"], 6),
                        "decode_tokens_per_sec": round(speed_metrics["decode_tokens_per_second"], 6),
                        "per_token_decode_latency_ms": round(speed_metrics["per_token_decode_latency_ms"], 6),
                        "finish_reason": finish_reason,
                        "success": success,
                        "error": error_text,
                        "event_chunks": event_chunks,
                    }
                    rows.append(row)
                    icon = "✔" if success else "✖"
                    console.print(f"[dim]{icon} {case.name} run={run_idx + 1} prefill={prefill_target} ttft={row['ttft_sec']:.3f}s e2e={row['e2e_latency_sec']:.3f}s e2e_tps={row['e2e_tokens_per_sec']:.2f} decode_tps={row['decode_tokens_per_sec']:.2f}[/dim]")
        finally:
            await http_client.aclose()
    grouped_case: Dict[str, List[Dict[str, Any]]] = {}
    grouped_prefill: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped_case.setdefault(row["case_name"], []).append(row)
        grouped_prefill.setdefault(f"{row['case_name']}::{row['prefill_target_tokens']}", []).append(row)
    summary_by_case = {k: _summarize_speed_rows(v) for k, v in grouped_case.items()}
    summary_by_prefill = {k: _summarize_speed_rows(v) for k, v in grouped_prefill.items()}
    speed_dir = report_dir / "speed"
    speed_dir.mkdir(parents=True, exist_ok=True)
    raw_json_path = speed_dir / "raw_runs.json"
    raw_csv_path = speed_dir / "raw_runs.csv"
    summary_json_path = speed_dir / "summary.json"
    report_html_path = speed_dir / "report.html"
    raw_json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_rows_csv(
        raw_csv_path,
        rows,
        [
            "case_name",
            "endpoint",
            "model",
            "run_index",
            "prefill_target_tokens",
            "prompt_tokens",
            "completion_tokens",
            "ttft_sec",
            "e2e_latency_sec",
            "decode_latency_sec",
            "e2e_tokens_per_sec",
            "prefill_tokens_per_sec",
            "decode_tokens_per_sec",
            "per_token_decode_latency_ms",
            "finish_reason",
            "success",
            "error",
            "event_chunks",
        ],
    )
    summary_json_path.write_text(
        json.dumps(
            {
                "suite": "speed",
                "runs": args.runs,
                "prefill_targets": prefill_targets,
                "cases": [case.__dict__ for case in cases],
                "summary_by_case": summary_by_case,
                "summary_by_prefill": summary_by_prefill,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    report_html_path.write_text(_build_speed_report_html(summary_by_case, summary_by_prefill, rows), encoding="utf-8")
    return {
        "raw_json": str(raw_json_path),
        "raw_csv": str(raw_csv_path),
        "summary_json": str(summary_json_path),
        "report_html": str(report_html_path),
        "summary_by_case": summary_by_case,
    }


def _extract_digits(text: str) -> str:
    return "".join(ch for ch in text if ch.isdigit())


def _build_passkey_context(target_tokens: int, passkey: int, depth: float) -> str:
    filler_sentence = "The sun sets in the west. "
    filler = []
    while estimate_tokens("".join(filler)) < max(64, target_tokens):
        filler.append(filler_sentence)
    filler_text = "".join(filler)
    needle = f"\nThe secret passkey is {passkey}.\n"
    insert_at = int(len(filler_text) * max(0.0, min(depth, 1.0)))
    return filler_text[:insert_at] + needle + filler_text[insert_at:]


async def _run_passkey_for_case(args: argparse.Namespace, case: APICase) -> Dict[str, Any]:
    hits = 0
    trials: List[Dict[str, Any]] = []
    http_client = httpx.AsyncClient(timeout=float(args.timeout_seconds))
    async_openai_cls = _require_async_openai()
    client = async_openai_cls(base_url=case.base_url, api_key=case.api_key, http_client=http_client)
    try:
        for idx in range(args.passkey_trials):
            passkey = random.randint(10000, 99999)
            context_text = _build_passkey_context(args.passkey_ctx, passkey, args.passkey_depth)
            prompt = f"{context_text}\nWhat is the secret passkey? Answer with ONLY the number."
            content = ""
            error_text = ""
            try:
                content, _ = await complete_with_async(
                    client=client,
                    model=case.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_output_tokens=32,
                    model_max_context=args.model_max_context,
                    provider=case.provider,
                    stream=False,
                    verbose=False,
                    on_event=None,
                    backend=case.backend,
                    enable_thinking=case.enable_thinking,
                )
            except LLM_REQUEST_ERRORS as e:
                error_text = str(e)
            predicted_digits = _extract_digits(content)
            hit = str(passkey) in predicted_digits and error_text == ""
            if hit:
                hits += 1
            trials.append(
                {
                    "case_name": case.name,
                    "trial": idx + 1,
                    "expected_passkey": passkey,
                    "prediction": content.strip(),
                    "prediction_digits": predicted_digits,
                    "hit": hit,
                    "error": error_text,
                }
            )
    finally:
        await http_client.aclose()
    accuracy = (100.0 * hits / args.passkey_trials) if args.passkey_trials > 0 else 0.0
    return {"benchmark": "passkey", "case_name": case.name, "trials": trials, "passkey_accuracy": round(accuracy, 4)}


def _which_or_none(cmd: str) -> Optional[str]:
    return shutil.which(cmd)


def _run_subprocess(cmd: List[str], env: Dict[str, str], log_path: Path, timeout_sec: int) -> subprocess.CompletedProcess:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    output_chunks: List[str] = []
    start_time = time.monotonic()
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
    )
    if process.stdout is None:
        raise OSError("Failed to capture subprocess output")
    fd = process.stdout.fileno()
    with log_path.open("w", encoding="utf-8") as log_file:
        while True:
            elapsed = time.monotonic() - start_time
            if elapsed > timeout_sec:
                process.kill()
                process.wait()
                raise subprocess.TimeoutExpired(cmd, timeout_sec, output="".join(output_chunks))
            ready, _, _ = select.select([fd], [], [], 1.0)
            if ready:
                raw_chunk = os.read(fd, 4096)
                if raw_chunk:
                    text_chunk = raw_chunk.decode("utf-8", errors="replace")
                    output_chunks.append(text_chunk)
                    log_file.write(text_chunk)
                    log_file.flush()
                    for line in text_chunk.splitlines():
                        if line.strip():
                            console.print(f"[dim]{line}[/dim]")
                elif process.poll() is not None:
                    break
            elif process.poll() is not None:
                break
    return_code = process.wait()
    return subprocess.CompletedProcess(cmd, return_code, "".join(output_chunks), None)


def _resolve_command(binary_name: str, module_name: str) -> Optional[List[str]]:
    cmd_path = _which_or_none(binary_name)
    if cmd_path:
        return [cmd_path]
    try:
        if importlib.util.find_spec(module_name) is not None:
            return [sys.executable, "-m", module_name]
    except (ModuleNotFoundError, ImportError, ValueError):
        return None
    return None


def _parse_evalplus_scores(stdout_text: str) -> tuple[Optional[float], Optional[float]]:
    """Parse evalplus output: returns (base_pass@1, plus_pass@1) both in 0-100% scale.

    Evalplus prints two pass@1 lines in order:
      <dataset> (base tests)          → first match = base score
      <dataset>+ (base + extra tests) → second match = plus score
    """
    all_matches = re.findall(r"pass@1\s*:\s*([0-9.]+)", stdout_text)
    if not all_matches:
        all_matches = re.findall(r"pass_at_1\s*[:=]\s*([0-9.]+)", stdout_text)

    def _pct(v: str) -> float:
        fv = float(v)
        return fv * 100.0 if fv <= 1.0 else fv

    base = _pct(all_matches[0]) if len(all_matches) >= 1 else None
    plus = _pct(all_matches[1]) if len(all_matches) >= 2 else None
    return base, plus


def _run_evalplus_for_case(case: APICase, dataset: str, parallel: int, out_dir: Path, timeout_sec: int) -> Dict[str, Any]:
    cmd_prefix = _resolve_command("evalplus.evaluate", "evalplus.evaluate")
    if not cmd_prefix:
        return {"benchmark": f"evalplus_{dataset}", "case_name": case.name, "status": "unavailable", "reason": "evalplus package not found"}
    log_path = out_dir / f"evalplus_{case.name}_{dataset}.log"
    evalplus_root = out_dir / f"evalplus_results_{case.name}_{dataset}"
    cmd = cmd_prefix + [
        "--dataset",
        dataset,
        "--greedy",
        "--model",
        case.model,
        "--backend",
        "openai",
        "--base-url",
        case.base_url,
        "--root",
        str(evalplus_root),
        "--resume",
        "False",
    ]
    if parallel > 0:
        cmd.extend(["--parallel", str(parallel)])
    env = os.environ.copy()
    env["OPENAI_API_KEY"] = case.api_key or "EMPTY"
    console.print(f"{case.name} evalplus_{dataset}: running (timeout={timeout_sec}s)")
    try:
        proc = _run_subprocess(cmd, env, log_path, timeout_sec)
    except SUBPROCESS_ERRORS as e:
        return {"benchmark": f"evalplus_{dataset}", "case_name": case.name, "status": "failed", "error": str(e), "log": str(log_path)}
    if proc.returncode != 0:
        return {"benchmark": f"evalplus_{dataset}", "case_name": case.name, "status": "failed", "code": proc.returncode, "log": str(log_path)}
    base_score, plus_score = _parse_evalplus_scores(proc.stdout)
    return {
        "benchmark": f"evalplus_{dataset}",
        "case_name": case.name,
        "status": "ok",
        "score_pass_at_1": round(base_score, 4) if base_score is not None else 0.0,
        "score_plus_pass_at_1": round(plus_score, 4) if plus_score is not None else None,
        "log": str(log_path),
    }


def _run_lm_eval_for_case(case: APICase, tasks: str, batch_size: int, out_dir: Path, timeout_sec: int) -> Dict[str, Any]:
    cmd_prefix = _resolve_command("lm_eval", "lm_eval")
    if not cmd_prefix:
        return {"benchmark": "lm_eval", "case_name": case.name, "status": "unavailable", "reason": "lm_eval package not found"}
    out_json_path = out_dir / f"lm_eval_{case.name}.json"
    log_path = out_dir / f"lm_eval_{case.name}.log"
    api_base = case.base_url.rstrip("/")
    completions_url = f"{api_base}/completions" if api_base.endswith("/v1") else f"{api_base}/v1/completions"
    model_args = f"model={case.model},base_url={completions_url},num_concurrent=8"
    cmd = cmd_prefix + [
        "--tasks",
        tasks,
        "--output_path",
        str(out_json_path),
        "--confirm_run_unsafe_code",
        "--model",
        "local-completions",
        "--model_args",
        model_args,
        "--batch_size",
        str(batch_size),
    ]
    env = os.environ.copy()
    env["OPENAI_API_KEY"] = case.api_key or "EMPTY"
    env["LM_EVAL_ALLOW_CODE_EXECUTION"] = "1"
    env["HF_ALLOW_CODE_EVAL"] = "1"
    console.print(f"{case.name} lm_eval: running (timeout={timeout_sec}s)")
    try:
        proc = _run_subprocess(cmd, env, log_path, timeout_sec)
    except SUBPROCESS_ERRORS as e:
        return {"benchmark": "lm_eval", "case_name": case.name, "status": "failed", "error": str(e), "log": str(log_path)}
    if proc.returncode != 0 and not out_json_path.exists():
        return {"benchmark": "lm_eval", "case_name": case.name, "status": "failed", "code": proc.returncode, "log": str(log_path)}
    if not out_json_path.exists():
        return {"benchmark": "lm_eval", "case_name": case.name, "status": "failed", "reason": "No JSON output", "log": str(log_path)}
    try:
        payload = json.loads(out_json_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        return {"benchmark": "lm_eval", "case_name": case.name, "status": "failed", "error": str(e), "log": str(log_path)}
    metrics: Dict[str, float] = {}
    for task_name, task_result in payload.get("results", {}).items():
        best_metric = None
        for key, value in task_result.items():
            if any(x in key for x in ["acc", "exact_match", "pass"]) and isinstance(value, (int, float)):
                best_metric = float(value)
                break
        if best_metric is not None:
            metrics[task_name] = round(best_metric * 100.0, 4)
    return {"benchmark": "lm_eval", "case_name": case.name, "status": "ok", "tasks": tasks, "metrics": metrics, "output_json": str(out_json_path), "log": str(log_path)}


def _build_accuracy_report_html(records: List[Dict[str, Any]]) -> str:
    records_json = json.dumps(records)
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>LLM Accuracy Benchmark Report</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{
      font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 0;
      background: #0b1020;
      color: #e5e7eb;
    }}
    .container {{
      max-width: 1400px;
      margin: 0 auto;
      padding: 24px;
    }}
    .card {{
      background: #111827;
      border: 1px solid #1f2937;
      border-radius: 12px;
      padding: 12px;
      margin-bottom: 16px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    th, td {{
      padding: 6px 8px;
      border-bottom: 1px solid #1f2937;
      text-align: left;
    }}
  </style>
</head>
<body>
  <div class="container">
    <h1>LLM Accuracy Benchmark</h1>
    <div class="card"><div id="accuracy_chart"></div></div>
    <div class="card">
      <h2>Detailed Records</h2>
      <table id="records_table"></table>
    </div>
  </div>
  <script>
    const records = {records_json};
    const numericRows = records.filter(r => r.status === 'ok' && r.score !== '' && !Number.isNaN(Number(r.score)));
    const grouped = {{}};
    numericRows.forEach(r => {{
      if (!grouped[r.benchmark]) grouped[r.benchmark] = {{}};
      grouped[r.benchmark][r.case_name] = Number(r.score);
    }});
    const benchmarkNames = Object.keys(grouped).sort();
    const caseNames = [...new Set(numericRows.map(r => r.case_name))].sort();
    const traces = caseNames.map(caseName => {{
      return {{
        x: benchmarkNames,
        y: benchmarkNames.map(b => grouped[b][caseName] ?? null),
        type: 'bar',
        name: caseName
      }};
    }});
    Plotly.newPlot(
      'accuracy_chart',
      traces,
      {{
        barmode: 'group',
        title: 'Accuracy Scores by Benchmark',
        paper_bgcolor: '#111827',
        plot_bgcolor: '#111827',
        font: {{color: '#e5e7eb'}},
        yaxis: {{title: 'Score (%)'}}
      }}
    );
    const table = document.getElementById('records_table');
    const headers = ['Case', 'Benchmark', 'Status', 'Score', 'Meta'];
    const thead = document.createElement('thead');
    const trh = document.createElement('tr');
    headers.forEach(h => {{
      const th = document.createElement('th');
      th.textContent = h;
      trh.appendChild(th);
    }});
    thead.appendChild(trh);
    table.appendChild(thead);
    const tbody = document.createElement('tbody');
    records.forEach(r => {{
      const tr = document.createElement('tr');
      [r.case_name, r.benchmark, r.status, String(r.score), r.meta].forEach(value => {{
        const td = document.createElement('td');
        td.textContent = String(value);
        tr.appendChild(td);
      }});
      tbody.appendChild(tr);
    }});
    table.appendChild(tbody);
  </script>
</body>
</html>"""


async def _collect_accuracy_for_case(
    args: argparse.Namespace,
    case: APICase,
    benchmarks: set[str],
    raw_dir: Path,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    records: List[Dict[str, Any]] = []
    passkey_trials_rows: List[Dict[str, Any]] = []
    if "passkey" in benchmarks:
        passkey_result = await _run_passkey_for_case(args, case)
        records.append(
            {
                "case_name": case.name,
                "benchmark": "passkey",
                "status": "ok",
                "score": passkey_result["passkey_accuracy"],
                "meta": "",
            }
        )
        passkey_trials_rows.extend(passkey_result["trials"])
        console.print(f"[dim]{case.name} passkey: {passkey_result['passkey_accuracy']:.2f}%[/dim]")
    if "ppl" in benchmarks:
        records.append(
            {
                "case_name": case.name,
                "benchmark": "ppl",
                "status": "not_supported_for_api",
                "score": "",
                "meta": "Use local/HF model pipeline from qwen_coder_evalv4.py",
            }
        )
        console.print(f"[yellow]{case.name} ppl: not supported for OpenAI-compatible API mode[/yellow]")
    if "humaneval" in benchmarks or "evalplus_humaneval" in benchmarks:
        res = _run_evalplus_for_case(case, "humaneval", args.evalplus_parallel, raw_dir, args.command_timeout_seconds)
        status = res.get("status", "failed")
        records.append(
            {
                "case_name": case.name,
                "benchmark": "evalplus_humaneval",
                "status": status,
                "score": res.get("score_pass_at_1", ""),
                "score_plus": res.get("score_plus_pass_at_1", ""),
                "meta": res.get("log", res.get("reason", "")),
            }
        )
        if status == "ok":
            console.print(
                f"[dim]{case.name} evalplus_humaneval: base={res.get('score_pass_at_1')}%  "
                f"plus={res.get('score_plus_pass_at_1')}%[/dim]"
            )
        else:
            console.print(f"[yellow]{case.name} evalplus_humaneval: {status}[/yellow]")
    if "mbpp" in benchmarks or "evalplus_mbpp" in benchmarks:
        res = _run_evalplus_for_case(case, "mbpp", args.evalplus_parallel, raw_dir, args.command_timeout_seconds)
        status = res.get("status", "failed")
        records.append(
            {
                "case_name": case.name,
                "benchmark": "evalplus_mbpp",
                "status": status,
                "score": res.get("score_pass_at_1", ""),
                "score_plus": res.get("score_plus_pass_at_1", ""),
                "meta": res.get("log", res.get("reason", "")),
            }
        )
        if status == "ok":
            console.print(
                f"[dim]{case.name} evalplus_mbpp: base={res.get('score_pass_at_1')}%  "
                f"plus={res.get('score_plus_pass_at_1')}%[/dim]"
            )
        else:
            console.print(f"[yellow]{case.name} evalplus_mbpp: {status}[/yellow]")
    return records, passkey_trials_rows


async def run_llm_accuracy_evaluation(args: argparse.Namespace, report_dir: Path, cases: List[APICase]) -> Dict[str, Any]:
    benchmarks = set(_parse_csv_list(args.accuracy_benchmarks))
    console.print(f"[bold]Accuracy benchmarks:[/bold] {', '.join(sorted(benchmarks))}")
    accuracy_dir = report_dir / "accuracy"
    raw_dir = accuracy_dir / "raw_outputs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, Any]] = []
    passkey_trials_rows: List[Dict[str, Any]] = []
    for case in cases:
        console.print(f"[bold cyan]Accuracy benchmark case:[/bold cyan] {case.name} ({case.base_url})")
        case_records, case_passkey_trials = await _collect_accuracy_for_case(args, case, benchmarks, raw_dir)
        records.extend(case_records)
        passkey_trials_rows.extend(case_passkey_trials)
    summary: Dict[str, Dict[str, Any]] = {}
    for row in records:
        case_name = row["case_name"]
        summary.setdefault(case_name, {})
        summary[case_name][row["benchmark"]] = {
            "status": row["status"],
            "score": row["score"],
            "score_plus": row.get("score_plus", ""),
            "meta": row["meta"],
        }
    records_json_path = accuracy_dir / "accuracy_records.json"
    records_csv_path = accuracy_dir / "accuracy_records.csv"
    summary_json_path = accuracy_dir / "summary.json"
    passkey_trials_path = accuracy_dir / "passkey_trials.csv"
    report_html_path = accuracy_dir / "report.html"
    records_json_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_rows_csv(records_csv_path, records, ["case_name", "benchmark", "status", "score", "score_plus", "meta"])
    _write_rows_csv(
        passkey_trials_path,
        passkey_trials_rows,
        ["case_name", "trial", "expected_passkey", "prediction", "prediction_digits", "hit", "error"],
    )
    summary_json_path.write_text(
        json.dumps(
            {
                "suite": "accuracy",
                "benchmarks": sorted(list(benchmarks)),
                "cases": [case.__dict__ for case in cases],
                "summary_by_case": summary,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    report_html_path.write_text(_build_accuracy_report_html(records), encoding="utf-8")
    return {
        "records_json": str(records_json_path),
        "records_csv": str(records_csv_path),
        "passkey_trials_csv": str(passkey_trials_path),
        "summary_json": str(summary_json_path),
        "report_html": str(report_html_path),
        "summary_by_case": summary,
    }


async def run_agent_evaluation(_: argparse.Namespace, __: Path) -> Dict[str, Any]:
    raise NotImplementedError("Agent-task evaluation is reserved for future extensions.")


# ============================================================
# Multi-folder comparison report
# ============================================================

# Human-readable display names for known metric keys
_METRIC_DISPLAY: Dict[str, str] = {
    "evalplus_humaneval_pass@1": "HumanEval pass@1 (%)",
    "evalplus_humaneval_plus_pass@1": "HumanEval+ pass@1 (%)",
    "evalplus_mbpp_pass@1": "MBPP pass@1 (%)",
    "evalplus_mbpp_plus_pass@1": "MBPP+ pass@1 (%)",
    "passkey_accuracy": "Passkey Accuracy (%)",
    "api_speed_ttft_p50_sec": "TTFT P50 (s)",
    "api_speed_decode_p50_tokens_per_sec": "Decode Speed P50 (tok/s)",
    "api_speed_throughput_tokens_per_sec": "Throughput (tok/s)",
}

# Metrics where lower is better (used to annotate figures)
_LOWER_IS_BETTER: set = {"api_speed_ttft_p50_sec", "api_speed_e2e_p50_sec"}


def _flatten_summary_results(model_results: Dict[str, Any]) -> Dict[str, float]:
    """Flatten a per-model result dict (from qwen_coder_evalv5_1.py summary.json) into scalar metrics."""
    flat: Dict[str, float] = {}

    def _safe(v: Any) -> Optional[float]:
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    for key, val in model_results.items():
        if not isinstance(val, dict):
            continue
        status = val.get("status", "")
        # evalplus results
        if key.startswith("evalplus_"):
            for score_field, suffix in [("pass@1", "pass@1"), ("plus_pass@1", "plus_pass@1")]:
                v = _safe(val.get(score_field))
                if v is not None:
                    flat[f"{key}_{suffix}"] = round(v, 4)
            # handle legacy score_pass_at_1 from older evaluator_main runs
            if "score_pass_at_1" in val:
                v = _safe(val["score_pass_at_1"])
                if v is not None:
                    flat[f"{key}_pass@1"] = round(v, 4)
            if "score_plus_pass_at_1" in val:
                v = _safe(val["score_plus_pass_at_1"])
                if v is not None:
                    flat[f"{key}_plus_pass@1"] = round(v, 4)
            # Recover plus score from log file when not stored in JSON (legacy runs)
            if f"{key}_plus_pass@1" not in flat and val.get("log"):
                log_path = Path(val["log"])
                if log_path.exists():
                    try:
                        log_text = log_path.read_text(encoding="utf-8", errors="replace")
                        _, plus = _parse_evalplus_scores(log_text)
                        if plus is not None:
                            flat[f"{key}_plus_pass@1"] = round(plus, 4)
                    except OSError:
                        pass
        # passkey
        elif key == "passkey":
            v = _safe(val.get("accuracy"))
            if v is not None:
                flat["passkey_accuracy"] = round(v * 100.0 if v <= 1.0 else v, 4)
        # api_speed — flatten overall stats
        elif key == "api_speed" and status == "ok":
            overall = val.get("overall", {})
            for speed_key in [
                "ttft_p50_sec", "ttft_p95_sec", "e2e_p50_sec", "e2e_p95_sec",
                "decode_p50_tokens_per_sec", "decode_p95_tokens_per_sec",
                "throughput_tokens_per_sec", "e2e_speed_p50_tokens_per_sec",
            ]:
                v = _safe(overall.get(speed_key))
                if v is not None:
                    flat[f"api_speed_{speed_key}"] = round(v, 4)
        # induction (local models only)
        elif key == "induction" and status == "ok":
            v = _safe(val.get("mean_nll"))
            if v is not None and v != float("inf"):
                flat["induction_mean_nll"] = round(v, 6)
        # ppl (local models only)
        elif key.startswith("ppl_") and status == "ok":
            v = _safe(val.get("ppl"))
            if v is not None:
                flat[f"{key}_ppl"] = round(v, 4)

    return flat


def _load_eval_dir(directory: str) -> Optional[Dict[str, Any]]:
    """Load evaluation results from a directory (qwen_coder_evalv5_1.py summary.json format)."""
    summary_path = Path(directory) / "summary.json"
    if not summary_path.exists():
        console.print(f"[yellow]Warning: summary.json not found in {directory}[/yellow]")
        return None
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        console.print(f"[red]Error loading {summary_path}: {e}[/red]")
        return None
    return data


def _build_comparison_figures(
    model_metrics: Dict[str, Dict[str, float]],
    out_dir: Path,
    dpi: int = 150,
) -> List[str]:
    """Generate paper-ready bar chart figures for each metric. Returns list of saved PNG paths."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        console.print("[yellow]matplotlib not installed — skipping PNG figures (HTML report will still use Plotly)[/yellow]")
        return []

    if not model_metrics:
        return []

    model_names = list(model_metrics.keys())
    all_metric_keys = sorted({k for m in model_metrics.values() for k in m})
    if not all_metric_keys:
        return []

    # Color palette — distinct, publication-friendly
    palette = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860", "#DA8BC3", "#8C8C8C"]
    colors = [palette[i % len(palette)] for i in range(len(model_names))]

    saved_paths: List[str] = []
    out_dir.mkdir(parents=True, exist_ok=True)

    # Group metrics into thematic figures
    groups = {
        "code_accuracy": [k for k in all_metric_keys if "humaneval" in k or "mbpp" in k or "passkey" in k],
        "speed": [k for k in all_metric_keys if "api_speed" in k],
        "other": [k for k in all_metric_keys if not any(x in k for x in ["humaneval", "mbpp", "passkey", "api_speed"])],
    }

    for group_name, metric_keys in groups.items():
        if not metric_keys:
            continue

        n_metrics = len(metric_keys)
        ncols = min(3, n_metrics)
        nrows = math.ceil(n_metrics / ncols)
        fig_w = max(5.0 * ncols, 8.0)
        fig_h = max(4.0 * nrows, 4.0)
        fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)
        fig.patch.set_facecolor("white")

        for idx, metric in enumerate(metric_keys):
            ax = axes[idx // ncols][idx % ncols]
            values = [model_metrics[m].get(metric, float("nan")) for m in model_names]
            bars = ax.bar(model_names, values, color=colors[: len(model_names)], edgecolor="white", linewidth=0.8)
            label = _METRIC_DISPLAY.get(metric, metric.replace("_", " "))
            ax.set_title(label, fontsize=10, fontweight="bold", pad=6)
            ax.tick_params(axis="x", rotation=30, labelsize=8)
            ax.tick_params(axis="y", labelsize=8)
            ax.grid(axis="y", linestyle="--", alpha=0.4, linewidth=0.7)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            # Annotate bar values
            for bar, v in zip(bars, values):
                if not math.isnan(v):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        bar.get_height() + 0.01 * (ax.get_ylim()[1] - ax.get_ylim()[0] + 1e-9),
                        f"{v:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=7.5,
                        fontweight="bold",
                    )
            if metric in _LOWER_IS_BETTER:
                ax.set_ylabel("↓ lower is better", fontsize=7, color="#888")
            else:
                ax.set_ylabel("↑ higher is better", fontsize=7, color="#888")

        # Hide unused subplots
        for idx in range(n_metrics, nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)

        fig.suptitle(f"Model Comparison — {group_name.replace('_', ' ').title()}", fontsize=13, fontweight="bold", y=1.01)
        fig.tight_layout()
        png_path = out_dir / f"comparison_{group_name}.png"
        fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        saved_paths.append(str(png_path))
        console.print(f"[green]Figure saved:[/green] {png_path}")

    return saved_paths


def _build_comparison_html(
    model_metrics: Dict[str, Dict[str, float]],
    figure_paths: List[str],
    report_dir: Path,
    source_dirs: List[str],
    labels: List[str],
) -> str:
    """Generate a self-contained HTML comparison report with inline images + interactive Plotly charts."""
    import base64

    model_names = list(model_metrics.keys())
    all_metrics = sorted({k for m in model_metrics.values() for k in m})

    # Encode figures as base64 for offline portability
    inline_imgs: List[str] = []
    for fig_path in figure_paths:
        try:
            with open(fig_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            inline_imgs.append(f'<img src="data:image/png;base64,{b64}" style="max-width:100%; margin:8px 0;" />')
        except OSError:
            pass

    figures_html = "\n".join(f'<div class="card">{img}</div>' for img in inline_imgs)

    # Build Plotly data for interactive grouped bar chart
    plotly_traces = []
    palette = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860", "#DA8BC3", "#8C8C8C"]
    for i, model in enumerate(model_names):
        y_vals = [model_metrics[model].get(m) for m in all_metrics]
        plotly_traces.append({
            "type": "bar",
            "name": model,
            "x": all_metrics,
            "y": y_vals,
            "marker": {"color": palette[i % len(palette)]},
        })

    traces_json = json.dumps(plotly_traces)

    # Build summary table HTML
    table_header = "<tr><th>Model</th>" + "".join(f"<th>{_METRIC_DISPLAY.get(m, m)}</th>" for m in all_metrics) + "</tr>"
    table_rows = []
    for model in model_names:
        cells = "".join(
            f"<td>{model_metrics[model][m]:.4f}</td>" if m in model_metrics[model] else "<td>—</td>"
            for m in all_metrics
        )
        table_rows.append(f"<tr><td><strong>{model}</strong></td>{cells}</tr>")
    table_html = f"<table><thead>{table_header}</thead><tbody>{''.join(table_rows)}</tbody></table>"

    source_list = "".join(f"<li><code>{d}</code> → <em>{l}</em></li>" for d, l in zip(source_dirs, labels))

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Model Comparison Report</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{
      font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 0;
      background: #f8fafc;
      color: #1e293b;
    }}
    .container {{ max-width: 1400px; margin: 0 auto; padding: 28px 24px; }}
    h1 {{ font-size: 1.8rem; font-weight: 700; margin-bottom: 4px; }}
    .subtitle {{ color: #64748b; font-size: 0.95rem; margin-bottom: 24px; }}
    .card {{
      background: #fff;
      border: 1px solid #e2e8f0;
      border-radius: 12px;
      padding: 20px;
      margin-bottom: 20px;
      box-shadow: 0 1px 3px rgba(0,0,0,.06);
    }}
    h2 {{ font-size: 1.15rem; font-weight: 600; margin: 0 0 12px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ padding: 8px 10px; border-bottom: 1px solid #e2e8f0; text-align: left; white-space: nowrap; }}
    th {{ background: #f1f5f9; font-weight: 600; position: sticky; top: 0; }}
    tr:hover td {{ background: #f8fafc; }}
    ul.sources {{ color: #475569; font-size: 13px; }}
    .overflow-x {{ overflow-x: auto; }}
  </style>
</head>
<body>
  <div class="container">
    <h1>Model Comparison Report</h1>
    <p class="subtitle">Generated {time.strftime("%Y-%m-%d %H:%M:%S")} &nbsp;|&nbsp; {len(model_names)} model(s) compared</p>

    <div class="card">
      <h2>Data Sources</h2>
      <ul class="sources">{source_list}</ul>
    </div>

    <div class="card">
      <h2>Interactive Comparison</h2>
      <div id="plotly_chart"></div>
    </div>

    {figures_html}

    <div class="card">
      <h2>Summary Table</h2>
      <div class="overflow-x">{table_html}</div>
    </div>
  </div>

  <script>
    const traces = {traces_json};
    Plotly.newPlot(
      "plotly_chart",
      traces,
      {{
        barmode: "group",
        title: {{ text: "All Metrics by Model", font: {{ size: 15 }} }},
        paper_bgcolor: "#fff",
        plot_bgcolor: "#f8fafc",
        font: {{ color: "#1e293b", size: 11 }},
        xaxis: {{ tickangle: -35 }},
        yaxis: {{ title: "Score / Value" }},
        legend: {{ orientation: "h", y: -0.25 }},
        margin: {{ b: 160 }},
      }}
    );
  </script>
</body>
</html>"""


def generate_comparison_report(
    result_dirs: List[str],
    labels: List[str],
    output_dir: str,
) -> str:
    """Load eval results from multiple directories and generate a comparison report.

    Args:
        result_dirs: Paths to directories each containing a ``summary.json``
                     (produced by qwen_coder_evalv5_1.py or evaluator_main.py).
        labels: Human-readable name for each directory (used in figures/HTML).
        output_dir: Where to write the report (figures + HTML).

    Returns:
        Path to the generated HTML report.
    """
    if len(labels) != len(result_dirs):
        labels = [Path(d).name for d in result_dirs]

    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect model_name → flat metrics across all dirs
    model_metrics: Dict[str, Dict[str, float]] = {}

    for label, directory in zip(labels, result_dirs):
        data = _load_eval_dir(directory)
        if data is None:
            continue
        results = data.get("results", {})
        if not results:
            console.print(f"[yellow]No 'results' key in {directory}/summary.json[/yellow]")
            continue
        for model_name, model_data in results.items():
            # Use label as display key if multiple dirs contain different models;
            # if same model name appears in multiple dirs, suffix with label.
            display_name = label if len(results) == 1 else f"{label}/{model_name}"
            model_metrics[display_name] = _flatten_summary_results(model_data)
            console.print(
                f"[cyan]Loaded[/cyan] {display_name}: "
                f"{len(model_metrics[display_name])} metrics"
            )

    if not model_metrics:
        console.print("[red]No metrics found — nothing to report.[/red]")
        return ""

    figure_paths = _build_comparison_figures(model_metrics, out_dir / "figures")
    html_content = _build_comparison_html(model_metrics, figure_paths, out_dir, result_dirs, labels)
    html_path = out_dir / "comparison_report.html"
    html_path.write_text(html_content, encoding="utf-8")
    console.print(f"[bold green]Comparison report:[/bold green] {html_path}")
    return str(html_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BatchAgent evaluation entrypoint")
    parser.add_argument("--eval-type", default="llm", choices=["llm", "agent"])
    parser.add_argument("--llm-suites", default="accuracy", help="Comma-separated suites: speed,accuracy,all")
    parser.add_argument("--output-dir", default="./agent_workspace")
    parser.add_argument("--provider", default="openai", choices=["openai", "anthropic"])
    parser.add_argument("--model", default=os.environ.get("VLLM_MODEL", "qwen3.5-9b"))
    parser.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", "EMPTY"))
    parser.add_argument("--base-url-name", default=os.environ.get("VLLM_BASE_URL_NAME", "vllm_e0_rtx3090"))
    parser.add_argument("--base-url", default=os.environ.get("VLLM_BASE_URL", "http://100.110.236.127:8000/v1"))
    parser.add_argument("--base-url-alt-name", default=os.environ.get("VLLM_BASE_URL_ALT_NAME", "vllm_e1_rtx5090"))
    parser.add_argument("--base-url-alt", default=os.environ.get("VLLM_BASE_URL_ALT", "http://100.65.193.60:8001/v1"))
    parser.add_argument("--base-urls", default=os.environ.get("BENCHMARK_BASE_URLS", ""), help="CSV endpoint URLs for unnamed cases")
    parser.add_argument("--llm-case", action="append", default=[], help="Repeatable case: name=<name>,url=<url>,model=<optional>,backend=<optional>,api_key=<optional>")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--prefill-tokens", default="512,2048,4096")
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--backend", default="vllm", choices=["vllm", "llama.cpp", "openai"])
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--model-max-context", type=int, default=16384)
    parser.add_argument(
        "--accuracy-benchmarks",
        default="passkey,evalplus_humaneval,evalplus_mbpp",
        help="Comma-separated subset of: passkey,evalplus_humaneval,evalplus_mbpp,ppl (ppl requires local model, not API)",
    )
    parser.add_argument("--passkey-ctx", type=int, default=4096)
    parser.add_argument("--passkey-depth", type=float, default=0.5)
    parser.add_argument("--passkey-trials", type=int, default=10)
    parser.add_argument("--evalplus-parallel", type=int, default=8)
    parser.add_argument("--command-timeout-seconds", type=int, default=7200)
    # Multi-folder comparison report
    parser.add_argument(
        "--report-dirs",
        nargs="+",
        default=[],
        metavar="DIR",
        help="One or more eval result directories (each containing summary.json). "
             "When provided, generates a comparison report without running new evaluations.",
    )
    parser.add_argument(
        "--report-labels",
        nargs="+",
        default=[],
        metavar="LABEL",
        help="Display labels for each --report-dirs entry (must match count if provided).",
    )
    parser.add_argument(
        "--report-output",
        default="./comparison_report",
        help="Output directory for the comparison report (default: ./comparison_report).",
    )
    return parser


async def main_async() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # ── Report-only mode: compare existing eval result directories ──
    if args.report_dirs:
        generate_comparison_report(
            result_dirs=args.report_dirs,
            labels=args.report_labels,
            output_dir=args.report_output,
        )
        return

    workspace_dir = Path(args.output_dir).resolve()
    workspace_dir.mkdir(parents=True, exist_ok=True)
    if args.eval_type == "agent":
        await run_agent_evaluation(args, workspace_dir)
        return
    cases = _build_llm_cases(args)
    if not cases:
        raise ValueError("No LLM cases configured.")
    report_dir = (workspace_dir / "evaluations" / "llm" / now_stamp()).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    suites = set(_parse_csv_list(args.llm_suites))
    if "all" in suites:
        suites = {"speed", "accuracy"}
    outputs: Dict[str, Dict[str, Any]] = {}
    if "speed" in suites:
        outputs["speed"] = await run_llm_speed_evaluation(args, report_dir, cases)
    if "accuracy" in suites:
        outputs["accuracy"] = await run_llm_accuracy_evaluation(args, report_dir, cases)
    top_summary_path = report_dir / "evaluation_summary.json"
    top_summary_path.write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8")
    console.print("[bold green]Evaluation completed.[/bold green]")
    console.print(f"[green]Report Dir:[/green] {report_dir}")
    console.print(f"[green]Top Summary:[/green] {top_summary_path}")
    if "speed" in outputs:
        console.print(f"[green]Speed Raw JSON:[/green] {outputs['speed']['raw_json']}")
        console.print(f"[green]Speed Raw CSV:[/green] {outputs['speed']['raw_csv']}")
        console.print(f"[green]Speed Summary:[/green] {outputs['speed']['summary_json']}")
        console.print(f"[green]Speed HTML:[/green] {outputs['speed']['report_html']}")
    if "accuracy" in outputs:
        console.print(f"[green]Accuracy Records JSON:[/green] {outputs['accuracy']['records_json']}")
        console.print(f"[green]Accuracy Records CSV:[/green] {outputs['accuracy']['records_csv']}")
        console.print(f"[green]Accuracy Passkey CSV:[/green] {outputs['accuracy']['passkey_trials_csv']}")
        console.print(f"[green]Accuracy Summary:[/green] {outputs['accuracy']['summary_json']}")
        console.print(f"[green]Accuracy HTML:[/green] {outputs['accuracy']['report_html']}")

    # Auto-generate comparison report when multiple cases were evaluated
    if len(cases) > 1:
        console.print("\n[bold]Generating comparison report for evaluated cases...[/bold]")
        case_dirs = [str(report_dir / "accuracy")] * len(cases)  # placeholder; real dirs vary per run
        generate_comparison_report(
            result_dirs=[str(workspace_dir)],
            labels=[workspace_dir.name],
            output_dir=str(workspace_dir / "comparison_report"),
        )


if __name__ == "__main__":
    asyncio.run(main_async())

"""
python CodeAgent/evaluator_main.py \
  --report-dirs output/eval_api_test output/eval_api_test_sft output/qwen_eval_27b output/qwen_eval_35b\
  --report-labels "Base Qwen9b" "Qwen9b SFT" "Qwen27b" "Qwen35b"\
  --report-output ./comparison_report
  
"""

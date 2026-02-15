import os
import json
import subprocess
import re
import argparse
import sys
import time
import requests
import shutil
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime, timezone

from openai import OpenAI
from tqdm import tqdm


# ================= Utilities =================

def to_root_url(vllm_base_url: str) -> str:
    """Convert OpenAI-compatible endpoint to root vLLM endpoint for tokenize."""
    return re.sub(r"/v1/?$", "", vllm_base_url)

def vllm_count_tokens(vllm_base_url: str, model: str, text: str) -> int:
    """Query vLLM /tokenize endpoint for accurate token counting."""
    root = to_root_url(vllm_base_url)
    try:
        r = requests.post(
            f"{root}/tokenize",
            json={"model": model, "prompt": text},
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        if "count" in data and isinstance(data["count"], int):
            return data["count"]
        if "tokens" in data and isinstance(data["tokens"], list):
            return len(data["tokens"])
        return int(data.get("num_tokens", 0))
    except Exception:
        # Heuristic fallback: ~4 chars per token
        return len(text) // 4

def truncate_text_tokens(vllm_url: str, model: str, text: str, max_tokens: int) -> str:
    """Truncate text to fit within a specific token budget."""
    current_tokens = vllm_count_tokens(vllm_url, model, text)
    if current_tokens <= max_tokens:
        return text
    # Heuristic character-based truncation with safety margin
    ratio = max_tokens / current_tokens
    keep_chars = int(len(text) * ratio * 0.9)
    return text[:keep_chars] + "\n... [TRUNCATED DUE TO CONTEXT LIMIT] ..."

def compute_max_tokens(vllm_url: str, model: str, messages: List[Dict], max_model_len: int, desired: int = 4096, safety: int = 256) -> int:
    """Calculate remaining token budget for generation."""
    joined = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages])
    prompt_tokens = vllm_count_tokens(vllm_url, model, joined)
    remaining = max_model_len - prompt_tokens - safety
    return min(desired, max(64, remaining))

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json(path: str, obj: Any) -> None:
    safe_mkdir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# ================= Code Agent =================

class CodeAgent:
    def __init__(self, client: OpenAI, args: argparse.Namespace, protocols: Dict = None):
        self.client = client
        self.args = args
        self.protocols = protocols or {}

        # General-purpose system prompt focusing on agentic behavior and formatting
        self.system_prompt = (
            "You are a professional software engineer agent. Your goal is to implement requested functionality with high precision and robustness.\n\n"
            "OPERATING PRINCIPLES:\n"
            "1. OUTPUT FORMAT: You MUST output files using the 'FILE: path/to/file' marker followed by a markdown code block. Do NOT use single large JSON objects.\n"
            "   Example:\n"
            "   FILE: tasks/<task_id>/task.py\n"
            "   ```python\n"
            "   # code here...\n"
            "   ```\n"
            "2. MODULARITY: Strictly adhere to any specified project structure and interfaces provided in the task instructions.\n"
            "3. CONTINUATION: If your output is truncated (hits context limit), stop exactly where you are. You will be asked to continue. "
            "When continuing, do NOT repeat previous content; start exactly from the next character or line to maintain code integrity.\n"
            "4. ROBUSTNESS: Ensure code is logically sound, handles edge cases, and follows best practices for requested frameworks.\n"
            "5. PHASES: Focus only on the current phase requested. Do not implement everything if only a specific part is asked for."
        )

    def generate_content(self, task: Dict, phase_instruction: str, context_code: Optional[str] = None, error_log: Optional[str] = None) -> Tuple[str, Dict]:
        task_id = task.get("id", "unknown_task")
        protocol_name = task.get("interface_protocol")
        protocol = self.protocols.get(protocol_name, {})
        protocol_instr = protocol.get("prompt_instructions", "")

        user_prompt = (
            f"Task ID: {task_id}\nSeries: {task.get('series')}\nLevel: {task.get('level')}\n"
            f"Algorithm: {task.get('algorithm')}\nDescription: {task.get('description')}\n"
            f"Requirements:\n{json.dumps(task.get('requirements', {}), indent=2)}\n\n"
        )

        # Dynamically load task-specific protocol instructions
        if protocol_instr:
            user_prompt += f"--- SPECIFIC PROJECT RULES ---\n{protocol_instr}\n\n"

        if context_code:
            code_budget = self.args.max_model_len // 2
            truncated_code = truncate_text_tokens(self.args.vllm_url, self.args.model, context_code, code_budget)
            user_prompt += f"--- EXISTING CODE CONTEXT ---\n{truncated_code}\n\n"

        if error_log:
            err_budget = self.args.max_model_len // 10
            truncated_error = truncate_text_tokens(self.args.vllm_url, self.args.model, error_log, err_budget)
            user_prompt += f"--- ERROR LOG FROM PREVIOUS ATTEMPT ---\n{truncated_error}\n\n"
            user_prompt += "Fix the implementation based on the error above. "

        user_prompt += f"CURRENT PHASE INSTRUCTION: {phase_instruction}"

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        full_content = ""
        total_completion_tokens = 0
        start_time = time.time()
        
        # Continuation loop
        for turn in range(5):
            max_tokens = compute_max_tokens(
                self.args.vllm_url, self.args.model, messages, self.args.max_model_len, desired=self.args.gen_max_tokens
            )

            try:
                resp = self.client.chat.completions.create(
                    model=self.args.model,
                    messages=messages,
                    temperature=self.args.temperature,
                    max_tokens=max_tokens,
                )
            except Exception as e:
                if "400" in str(e) or "context length" in str(e).lower():
                    print(f"  [Agent] Warning: Context limit reached, retrying with tiny buffer...")
                    resp = self.client.chat.completions.create(
                        model=self.args.model, messages=messages, temperature=self.args.temperature, max_tokens=128
                    )
                else:
                    raise e

            chunk = resp.choices[0].message.content
            full_content += chunk
            if resp.usage:
                total_completion_tokens += resp.usage.completion_tokens

            if resp.choices[0].finish_reason != "length":
                break
            
            print(f"  [Agent] Turn {turn+1} truncated. Requesting continuation...")
            messages.append({"role": "assistant", "content": chunk})
            messages.append({"role": "user", "content": "The output was truncated. Continue exactly from where you stopped. Do NOT repeat previous headers or code."})

        duration = time.time() - start_time
        stats = {"duration": duration, "completion_tokens": total_completion_tokens}
        return full_content, stats

    @staticmethod
    def extract_files(raw_text: str) -> Dict[str, str]:
        """Extract files from 'FILE: path' markers with support for multi-turn continuations."""
        files = {}
        # Split by FILE: marker, but handle cases where it might be in the middle of a line due to bad continuation
        parts = re.split(r"(?mi)^FILE:\s*|FILE:\s*", raw_text)
        for part in parts:
            if not part.strip(): continue
            # Chunk starts with the path
            lines = part.strip().splitlines()
            if not lines: continue
            path = lines[0].strip()
            body = "\n".join(lines[1:])
            
            # Remove markdown fences and any dangling "FILE:" text that might have survived the split
            clean_body = re.sub(r"```(?:\w+)?\n", "", body)
            clean_body = re.sub(r"\n```", "", clean_body)
            # Remove potentially repeated headers in bad continuations
            clean_body = re.sub(r"(?mi)^FILE:\s*.*?\n", "", clean_body)
            
            if path in files:
                if clean_body.strip() not in files[path]:
                    files[path] += "\n" + clean_body
            else:
                files[path] = clean_body
        return files


# ================= Runner & Orchestrator =================

def run_subprocess(cmd: List[str], cwd: str, timeout: int) -> Tuple[bool, str]:
    """Helper to run subprocess with environment configuration."""
    try:
        env = {
            **os.environ, 
            "PYTHONUTF8": "1",
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
            "CUDA_LAUNCH_BLOCKING": "1"
        }
        res = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout,
            env=env
        )
        return res.returncode == 0, res.stdout + res.stderr
    except subprocess.TimeoutExpired:
        return False, f"Error: Execution timed out after {timeout}s"
    except Exception as e:
        return False, f"Error: {str(e)}"

def run_task_tests(task_id: str, output_dir: str) -> Tuple[bool, str]:
    """Execute tests for a task with OOM handling and isolation."""
    task_dir = os.path.join(output_dir, "tasks", task_id)
    pycache = os.path.join(task_dir, "__pycache__")
    if os.path.exists(pycache): shutil.rmtree(pycache)
    
    test_path = os.path.join("tasks", task_id, "tests")
    ok, out = run_subprocess(
        [sys.executable, "-m", "pytest", "-q", "--import-mode=importlib", test_path],
        output_dir, 300
    )
    
    # Retry on CPU if GPU is OOM
    if any(msg in out for msg in ["CUDA error: out of memory", "RuntimeError: CUDA out of memory"]):
        print("  [Runner] CUDA OOM. Retrying on CPU...")
        env_cpu = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
        res = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "--import-mode=importlib", test_path],
            cwd=output_dir, capture_output=True, text=True, timeout=300, env=env_cpu
        )
        return res.returncode == 0, res.stdout + res.stderr

    if ok or "No module named pytest" not in out:
        return ok, out
    
    return run_subprocess([sys.executable, "-m", f"tasks.{task_id}.task"], output_dir, 180)

def main():
    parser = argparse.ArgumentParser(description="General Purpose ML Code Agent")
    parser.add_argument("--model", default="Qwen/Qwen3-Coder-Next-FP8")
    parser.add_argument("--vllm_url", default="http://localhost:8000/v1")
    parser.add_argument("--api_key", default="myhpcvllmqwen")
    parser.add_argument("--max_model_len", type=int, default=16384)
    parser.add_argument("--gen_max_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--tasks_file", default="ml_tasks.json")
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--passes", type=int, default=1)
    parser.add_argument("--resume", choices=["failed", "all", "pending"], default="failed")
    parser.add_argument("--workspace", default="./workspace")
    args = parser.parse_args()

    output_dir = os.path.join(args.workspace, "code_repo")
    status_file = os.path.join(args.workspace, "task_status.json")
    raw_outputs_dir = os.path.join(args.workspace, "raw_outputs")
    for d in [args.workspace, output_dir, raw_outputs_dir]: safe_mkdir(d)

    payload = read_json(args.tasks_file)
    tasks = payload.get("tasks", []) if isinstance(payload, dict) else payload
    protocols = payload.get("interface_protocols", {}) if isinstance(payload, dict) else {}
    status_db = read_json(status_file) if os.path.exists(status_file) else {}

    client = OpenAI(base_url=args.vllm_url, api_key=args.api_key)
    agent = CodeAgent(client, args, protocols)

    print(f"🚀 Auto-Coder starting. Context={args.max_model_len}")

    for p_idx in range(args.passes):
        print(f"\n===== PASS {p_idx+1}/{args.passes} =====")
        for task in tqdm(tasks):
            tid = task.get("id")
            if not tid: continue
            
            current_status = status_db.get(tid, {}).get("status")
            if args.resume == "failed" and current_status == "success": continue
            if args.resume == "pending" and current_status in ["success", "failed"]: continue

            print(f"\nProcessing: {tid}")
            try:
                if tid not in status_db: status_db[tid] = {"status": "pending", "attempts": 0}

                error_history = []
                phase1_code = None

                for attempt in range(args.max_retries):
                    status_db[tid]["attempts"] = status_db[tid].get("attempts", 0) + 1
                    status_db[tid]["updated_at"] = utc_now_iso()
                    write_json(status_file, status_db)

                    last_err = error_history[-1] if error_history else None

                    # PHASE 1: Core Implementation
                    print(f"  [Agent] Phase 1: Algorithm & Docs...")
                    raw1, stats1 = agent.generate_content(
                        task, 
                        "PHASE 1: Implement core algorithm in tasks/<task_id>/task.py and README.md.",
                        error_log=last_err
                    )
                    files1 = agent.extract_files(raw1)
                    if not files1:
                        error_history.append("Error: Phase 1 output empty.")
                        continue
                    phase1_code = raw1

                    # PHASE 2: Tests
                    print(f"  [Agent] Phase 2: Tests...")
                    raw2, stats2 = agent.generate_content(
                        task, 
                        "PHASE 2: Now provide comprehensive pytest tests in tasks/<task_id>/tests/test_task.py.", 
                        context_code=phase1_code,
                        error_log=last_err # Pass error to tests too if relevant
                    )
                    files2 = agent.extract_files(raw2)

                    combined_files = {**files1, **files2}
                    for rel_path, content in combined_files.items():
                        abs_path = os.path.join(output_dir, rel_path)
                        safe_mkdir(os.path.dirname(abs_path))
                        with open(abs_path, "w") as f: f.write(content)

                    with open(os.path.join(raw_outputs_dir, f"{tid}_p{p_idx+1}_a{attempt+1}_raw.txt"), "w") as f:
                        f.write(f"--- PHASE 1 ---\n{raw1}\n\n--- PHASE 2 ---\n{raw2}")

                    ok, out = run_task_tests(tid, output_dir)
                    if ok:
                        status_db[tid].update({"status": "success", "last_error": None})
                        print(f"✅ Success: {tid}")
                        break
                    else:
                        err_tail = "\n".join(out.splitlines()[-80:])
                        print(f"❌ Attempt {attempt+1} failed. Tail:\n{err_tail}")
                        error_history.append(err_tail)
                        status_db[tid].update({"status": "failed", "last_error": err_tail})

                write_json(status_file, status_db)

            except Exception as loop_e:
                print(f"🔥 Critical task error {tid}: {str(loop_e)}")
                status_db[tid] = {**status_db.get(tid, {}), "status": "error", "last_error": str(loop_e)}
                write_json(status_file, status_db)
                continue

if __name__ == "__main__":
    main()

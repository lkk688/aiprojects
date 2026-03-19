
import os
import re
import json
import time
import hashlib
import subprocess
import ast
import fnmatch
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set

from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.panel import Panel
from openai import OpenAI
try:
    import tiktoken
except ImportError:
    tiktoken = None

import json
import logging
from typing import Dict, Any, Optional
#pip install json-repair
try:
    import json_repair
except ImportError:
    json_repair = None

console = Console()

# ---------------------------
# Utilities
# ---------------------------
import re
from collections import Counter
import os
from typing import Dict, Set

def build_debug_prompt(traceback_str: str, window_size: int = 15, root_dir: str = ".") -> str:
    """
    Automatically extracts file skeletons and context from errors to build high-quality LLM prompts.
    """
    tb_pattern = re.compile(r'File\s+"([^"]+)",\s+line\s+(\d+)')
    error_locations: Dict[str, Set[int]] = {}
    
    for match in tb_pattern.finditer(traceback_str):
        filepath = match.group(1)
        line_num = int(match.group(2))
        
        abs_path = os.path.abspath(filepath)
        abs_root = os.path.abspath(root_dir)
        if not abs_path.startswith(abs_root) or "site-packages" in abs_path:
            continue
            
        if os.path.exists(filepath) and os.path.isfile(filepath):
            if filepath not in error_locations:
                error_locations[filepath] = set()
            error_locations[filepath].add(line_num)

    # If no local files matched, just return the truncated traceback
    if not error_locations:
        return f"## Error Output\n```text\n{traceback_str[-2000:]}\n```\n"

    prompt_parts = []
    prompt_parts.append("## Error Traceback\n```text\n" + traceback_str[-2000:].strip() + "\n```\n")

    for filepath, lines in error_locations.items():
        rel_path = os.path.relpath(filepath, root_dir)
        prompt_parts.append(f"## Context for `{rel_path}`\n")
        
        # Make sure you have _generate_ast_map_from_file and _extract_snippets_from_string defined!
        try:
            # Assuming these functions exist in your libs
            from CodeAgent.codeagent_libs import _generate_ast_map_from_file, _extract_snippets_from_string
            
            ast_map = _generate_ast_map_from_file(filepath)
            if ast_map:
                prompt_parts.append("### File Map (Structure)\n```python\n" + ast_map + "\n```\n")
            
            with open(filepath, 'r', encoding='utf-8') as f:
                source = f.read()
            snippets = _extract_snippets_from_string(source, lines, window_size)
            prompt_parts.append("### Error Context Snippets\n```python\n" + snippets + "\n```\n")
            
        except Exception as e:
            prompt_parts.append(f"> [Warning] Could not extract detailed AST/snippets: {e}\n")

    return "\n".join(prompt_parts)

def robust_json_loads(json_str: str, tool_name: str = "") -> Optional[Dict[str, Any]]:
    """
    A highly fault-tolerant JSON parser.
    Uses standard json first, falls back to json-repair for LLM hallucinations,
    and includes a violent regex fallback specifically for 'write_file' tools.
    """
    if not json_str or not isinstance(json_str, str): 
        return None

    # 1. 原生解析 (最快)
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    # 2. json-repair 抢救 (如果安装了的话)
    if json_repair is not None:
        try:
            repaired = json_repair.repair_json(json_str, return_objects=True)
            if isinstance(repaired, dict): 
                return repaired
            if isinstance(repaired, list) and len(repaired) > 0 and isinstance(repaired[0], dict):
                return repaired[0]
        except Exception:
            pass

    # 3. 【终极防线】针对 write_file 的暴力正则提取
    # 专门处理大模型在 content 里输出了未转义换行符导致整个 JSON 崩溃的情况
    if tool_name == "write_file":
        logging.warning("JSON parsing failed. Attempting violent regex extraction for write_file.")
        try:
            def _extract_string_field(payload: str, field_name: str) -> Optional[str]:
                key = f'"{field_name}"'
                key_idx = payload.find(key)
                if key_idx < 0:
                    return None
                colon_idx = payload.find(":", key_idx + len(key))
                if colon_idx < 0:
                    return None
                start_quote_idx = payload.find('"', colon_idx + 1)
                if start_quote_idx < 0:
                    return None
                value_chars: List[str] = []
                escaped = False
                i = start_quote_idx + 1
                while i < len(payload):
                    ch = payload[i]
                    if escaped:
                        value_chars.append(ch)
                        escaped = False
                    elif ch == "\\":
                        value_chars.append(ch)
                        escaped = True
                    elif ch == '"':
                        return "".join(value_chars)
                    else:
                        value_chars.append(ch)
                    i += 1
                return "".join(value_chars) if value_chars else None

            path_raw = _extract_string_field(json_str, "path")
            content_raw = _extract_string_field(json_str, "content")
            if path_raw is not None and content_raw is not None:
                path = bytes(path_raw, "utf-8").decode("unicode_escape")
                try:
                    clean_content = bytes(content_raw, "utf-8").decode("unicode_escape")
                except Exception:
                    clean_content = content_raw.replace("\\n", "\n").replace('\\"', '"').replace("\\\\", "\\")
                return {"path": path, "content": clean_content}
        except Exception as e:
            logging.error(f"Violent regex extraction failed: {e}")

    # 4. 最后的挣扎：简单的换行符清洗
    try:
        clean_str = json_str.replace('\n', '\\n').replace('\r', '')
        return json.loads(clean_str)
    except Exception:
        return None
    
def now_stamp() -> str:
    return time.strftime("%Y-%m-%d_%H%M%S")

def sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    if tiktoken:
        try:
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:
            pass
    # Rough fallback: 1 token ~= 4 chars
    return len(text) // 4

def truncate_to_tokens(text: str, max_tokens: int) -> str:
    if estimate_tokens(text) <= max_tokens:
        return text
    target_chars = int(max_tokens * 3.5)
    return text[:target_chars] + "\n...[TRUNCATED]..."

def _handle_missing_modules(error_output: str) -> Optional[str]:
    """
    Detects ModuleNotFoundError and installs the missing package.
    Returns the installation log if an installation occurred, else None.
    """
    # Regex for "ModuleNotFoundError: No module named 'xyz'"
    # Also handle "ImportError: No module named xyz" (older python)
    match = re.search(r"ModuleNotFoundError: No module named '(.+?)'", error_output)
    if not match:
        match = re.search(r"ImportError: No module named '(.+?)'", error_output)
    
    if not match:
        return None
        
    module_name = match.group(1)
    
    # Map common imports to package names
    # This is a heuristic mapping.
    package_map = {
        "sklearn": "scikit-learn",
        "PIL": "Pillow",
        "cv2": "opencv-python",
        "yaml": "PyYAML",
        "bs4": "beautifulsoup4",
        "dotenv": "python-dotenv",
        "dateutil": "python-dateutil"
    }
    
    package_name = package_map.get(module_name, module_name)
    
    console.print(f"[yellow]Detected missing module: '{module_name}'. Attempting auto-install of '{package_name}'...[/yellow]")
    
    # Use --no-input to prevent hanging on prompts
    cmd = f"pip install --no-input {package_name}"
    code, out = run_shell(cmd)
    
    log = f"\n[Auto-Install: {cmd}]\nExit Code: {code}\nOutput:\n{out}\n"
    
    if code == 0:
        console.print(f"[green]Successfully installed '{package_name}'.[/green]")
    else:
        console.print(f"[red]Failed to install '{package_name}'.[/red]")
        
    return log


def query_model_context_length(client: OpenAI, model_name: str) -> int:
    """
    Query the vLLM /v1/models endpoint to discover the model's max context length.
    Returns 0 if the query fails (caller should use a fallback).
    """
    try:
        models = client.models.list()
        for m in models.data:
            if m.id == model_name:
                # vLLM exposes max_model_len in the model info
                ctx = getattr(m, 'max_model_len', 0)
                if ctx and ctx > 0:
                    console.print(f"[green]Auto-detected model context length: {ctx}[/green]")
                    return int(ctx)
        console.print(f"[yellow]Model '{model_name}' not found in /v1/models. Using fallback.[/yellow]")
    except Exception as e:
        console.print(f"[yellow]Could not query model context length: {e}. Using fallback.[/yellow]")
    return 0


def compute_safe_max_tokens(input_tokens: int, model_max_context: int, desired_max_output: int,
                            safety_margin: int = 1000, min_output: int = 1024) -> int:
    """
    Compute the largest safe max_tokens value that won't exceed the model's context limit.
    
    Args:
        input_tokens: Estimated token count of the input (system + user messages)
        model_max_context: Model's maximum context window
        desired_max_output: The user's requested max output tokens
        safety_margin: Extra buffer for tokenizer estimation errors
        min_output: Minimum output tokens; below this, signal an error condition
    
    Returns:
        Clamped max_tokens value without blindly asserting min_output if budget is tight.
    """
    # Qwen/other tokenizer density can be ~10% higher than cl100k_base (used for est.)
    adjusted_input = int(input_tokens * 1.1)
    available = model_max_context - adjusted_input - safety_margin
    if available < min_output:
        console.print(f"[red]Context budget very tight: {available} tokens available "
                      f"(est_input={input_tokens} -> {adjusted_input}, limit={model_max_context}). "
                      f"Returning available tokens to avoid exceeding context window.[/red]")
        return max(1, available)
    safe = min(desired_max_output, available)
    return safe

def compress_messages(messages: List[Dict[str, str]], max_allowed_tokens: int) -> List[Dict[str, str]]:
    """
    Compress the messages list to fit within max_allowed_tokens by truncating 
    the longest text blocks in 'user' or 'assistant' messages.
    """
    import copy
    msgs = copy.deepcopy(messages)
    
    while True:
        current_tokens = sum(estimate_tokens(m.get("content", "")) for m in msgs)
        if current_tokens <= max_allowed_tokens:
            break
            
        longest_idx = -1
        longest_len = 0
        for i, m in enumerate(msgs):
            if i in (0, 1):
                continue # Never truncate System Prompt (0) and Initial Task (1) to keep vLLM prefix caching intact
            content_len = len(m.get("content", ""))
            if content_len > longest_len:
                longest_len = content_len
                longest_idx = i
                
        if longest_idx == -1: 
            # fallback if everything else is tiny (extremely rare)
            for i, m in enumerate(msgs):
                if i == 0: continue # Still protect System
                content_len = len(m.get("content", ""))
                if content_len > longest_len:
                    longest_len = content_len
                    longest_idx = i
                    
        if longest_idx == -1 or longest_len < 400:
            break # Can't compress meaningfully further
            
        content = msgs[longest_idx]["content"]
        keep_chars = int(longest_len * 0.45) # trim middle 10%
        
        # Prevent infinite loops where the truncation tag makes the string longer than before
        if keep_chars * 2 + 35 >= longest_len:
            break
            
        msgs[longest_idx]["content"] = content[:keep_chars] + "\n...[TRUNCATED TO FIT CONTEXT]...\n" + content[-keep_chars:]
        
    return msgs


def ensure_dirs(base_dir: Path):
    (base_dir / "sessions").mkdir(parents=True, exist_ok=True)
    (base_dir / "skilldb").mkdir(parents=True, exist_ok=True)
    for p in [base_dir / "skilldb/successes.jsonl", base_dir / "skilldb/failures.jsonl", base_dir / "runs.jsonl"]:
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("", encoding="utf-8")

def run_shell(cmd: str, cwd: Optional[str] = None, cap: int = 20000, 
              timeout: Optional[int] = 1200, sandbox_container: Optional[str] = None) -> Tuple[int, str]:
    if sandbox_container:
        if cwd:
            cmd_to_run = f'docker exec -i -w "{cwd}" {sandbox_container} /bin/bash -c {cmd!r}'
        else:
            cmd_to_run = f'docker exec -i {sandbox_container} /bin/bash -c {cmd!r}'
    else:
        cmd_to_run = cmd

    start_time = time.time()
    try:
        p = subprocess.run(
            cmd_to_run, 
            shell=True, 
            text=True, 
            capture_output=True, 
            cwd=cwd if not sandbox_container else None, 
            timeout=timeout
        )
        code = p.returncode
        out = (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired as e:
        code = 124
        out = (e.stdout or "") + (e.stderr or "")
        out += f"\n[Error] Command timed out after {timeout} seconds!"
    except Exception as e:
        code = 1
        out = f"[Error] Failed to execute command: {e}"

    elapsed = time.time() - start_time
    out += f"\n[Execution Time: {elapsed:.2f}s]"

    if len(out) > cap:
        out = out[-cap:]
    return code, out

def is_git_repo() -> bool:
    code, _ = run_shell("git rev-parse --is-inside-work-tree")
    return code == 0

def git_status() -> str:
    code, out = run_shell("git status -sb")
    return out if code == 0 else ""

def git_diff() -> str:
    code, out = run_shell("git diff")
    return out if code == 0 else ""

def read_file(path: str, max_chars: int = 64000) -> str:
    p = Path(path)
    if not p.exists():
        return f"[MISSING FILE] {path}"
    data = p.read_text(encoding="utf-8", errors="ignore")
    if len(data) > max_chars:
        return data[:max_chars] + "\n\n[TRUNCATED]\n"
    return data

# ---------------------------
# Custom Exploration Tools
# ---------------------------

def _is_binary(file_path: Path) -> bool:
    if not file_path.exists():
        return True # Treat missing/broken as binary to skip
    try:
        with open(file_path, 'tr') as f:
            f.read(1024)
            return False
    except UnicodeDecodeError:
        return True

def search_code(query: str, root_dir: str = ".") -> str:
    """
    Search for a text query across all non-binary, non-hidden files.
    Format: <filepath>:<line_number>: <content>
    Truncates at 50 matches.
    """
    results = []
    root = Path(root_dir).resolve()
    
    for root_path_str, dirs, files in os.walk(root):
        # Clean dirs in-place to prevent walking into them
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("__pycache__", "node_modules", "site-packages", "venv", "env", ".venv")]
        
        root_path = Path(root_path_str)
        for file in files:
            if file.startswith("."):
                continue
            
            path = root_path / file
            
            # Skip massive files (e.g. > 1MB) to prevent freezing
            try:
                if path.stat().st_size > 1024 * 1024:
                    continue
            except Exception:
                pass
                
            if _is_binary(path):
                continue
                
            try:
                with path.open("r", encoding="utf-8", errors="ignore") as f:
                    for i, line in enumerate(f, start=1):
                        if query in line:
                            rel_path = path.relative_to(root)
                            results.append(f"{rel_path}:{i}: {line.rstrip()}")
                            if len(results) >= 50:
                                break
            except Exception:
                pass
                
            if len(results) >= 50:
                break
                
        if len(results) >= 50:
            results.append("Warning: Too many matches (>50). Showing first 50. Please refine your search.")
            break
            
    if not results:
        return "No matches found."
    return "\n".join(results)

def find_file(filename_pattern: str, root_dir: str = ".") -> str:
    """
    Find files by name using glob fuzzy matching.
    """
    results = []
    root = Path(root_dir).resolve()
    pattern = filename_pattern.lower()
    
    for root_path_str, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("__pycache__", "node_modules", "site-packages", "venv", "env", ".venv")]
        
        root_path = Path(root_path_str)
        for file in files:
            if file.startswith("."):
                continue
                
            if fnmatch.fnmatch(file.lower(), f"*{pattern}*"):
                path = root_path / file
                results.append(str(path.relative_to(root)))
                
    if not results:
        return "No matching files found."
    return "\n".join(results)

def read_file_chunk(filepath: str, start_line: int, end_line: int) -> str:
    """
    Read a specific range of lines from a file, adding line numbers.
    Replaces view_file_content.
    """
    try:
        raw_path = Path(filepath.strip()).expanduser()
        cwd = Path.cwd()
        candidate_paths: List[Path] = []
        if raw_path.is_absolute():
            candidate_paths.append(raw_path)
        else:
            candidate_paths.append((cwd / raw_path))
            candidate_paths.append(raw_path)

        file_path: Optional[Path] = None
        for candidate in candidate_paths:
            resolved = candidate.resolve()
            if resolved.exists() and resolved.is_file():
                file_path = resolved
                break

        if file_path is None and raw_path.name:
            suffix_parts = [part.lower() for part in raw_path.parts if part not in (".", "")]
            matches: List[Path] = []
            for match in cwd.rglob(raw_path.name):
                if not match.is_file():
                    continue
                rel_parts = [part.lower() for part in match.relative_to(cwd).parts]
                if len(rel_parts) >= len(suffix_parts) and rel_parts[-len(suffix_parts):] == suffix_parts:
                    matches.append(match.resolve())
            if not matches:
                for match in cwd.rglob("*"):
                    if not match.is_file():
                        continue
                    if match.name.lower() != raw_path.name.lower():
                        continue
                    rel_parts = [part.lower() for part in match.relative_to(cwd).parts]
                    if len(rel_parts) >= len(suffix_parts) and rel_parts[-len(suffix_parts):] == suffix_parts:
                        matches.append(match.resolve())
            unique_matches = list(dict.fromkeys(matches))
            if len(unique_matches) == 1:
                file_path = unique_matches[0]

        if file_path is None:
            return f"[Error] File not found: {filepath}"

        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
            
        if start_line < 1:
            start_line = 1
        if end_line > len(lines):
            end_line = len(lines)
            
        if start_line > end_line:
            return f"[Error] Invalid range: start_line ({start_line}) > end_line ({end_line})"
            
        result = [f"--- {file_path} (Lines {start_line}-{end_line}) ---"]
        for i in range(start_line - 1, end_line):
            result.append(f"{i+1:4d} | {lines[i].rstrip()}")
            
        return "\n".join(result)
        
    except Exception as e:
        return f"[Error] Could not read file {filepath}: {e}"

# Backward compatibility alias for older agents (e.g. mini_code_agent.py)
view_file_content = read_file_chunk
def list_directory(dir_path: str = ".") -> str:
    """
    List contents of a directory using ls -la.
    """
    try:
        target_dir = Path(dir_path).resolve()
        if not target_dir.exists() or not target_dir.is_dir():
            return f"[Error] Directory not found or not a directory: {dir_path}"
            
        result = subprocess.run(
            ["ls", "-la", str(target_dir)], 
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            return f"[Error] ls failed: {result.stderr.strip()}"
    except Exception as e:
        return f"[Error] Failed to list directory {dir_path}: {e}"

def run_bash_command(command: str) -> str:
    """
    Execute a terminal command with a standard timeout.
    Returns stdout + stderr.
    """
    try:
        # Use a short timeout of 60 seconds to prevent blocking the agent indefinitely
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=60
        )
        
        output = []
        if result.stdout:
            output.append("STDOUT:\n" + result.stdout)
        if result.stderr:
            output.append("STDERR:\n" + result.stderr)
            
        if result.returncode != 0:
            output.insert(0, f"[Error] Command exited with code {result.returncode}")
            
        if not output:
            return "[Empty Output - Success]"
            
        return "\n".join(output)
    except subprocess.TimeoutExpired:
        return f"[Error] Command timed out after 60 seconds:\n{command}"
    except Exception as e:
        return f"[Error] Failed to run command: {e}"

def top_level_tree(max_items: int = 200) -> str:
    items = []
    try:
        for p in Path(".").iterdir():
            if p.name.startswith(".agent") or p.name.startswith(".git"):
                continue
            items.append(p.name + ("/" if p.is_dir() else ""))
    except Exception:
        pass
    items = sorted(items)[:max_items]
    return "\n".join(items)

def write_jsonl(path: Path, obj: Dict[str, Any]):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# ---------------------------
# Diff Extraction & Sanitization (IMPROVED)
# ---------------------------

def sanitize_diff_text(diff_text: str) -> str:
    """
    Clean up a diff block extracted from LLM output.
    Removes common LLM artifacts that corrupt patches:
    - Stray ``` fence markers (but NOT inside diff hunk content)
    - ALL 'index' lines (never converts them to --- headers)
    - Missing ---/+++ headers (injects them from diff --git header)
    - Enforces strict --- before +++ order
    """
    lines = diff_text.split("\n")
    cleaned = []
    
    # State tracking for header repair
    current_file_a = None
    current_file_b = None
    seen_header_a = False
    seen_header_b = False
    in_hunk = False  # Track if we're inside a @@ hunk
    
    for line in lines:
        stripped = line.strip()
        
        # 1. Skip stray fence markers — BUT ONLY if they are NOT diff content.
        #    Lines like '-    #```python' are valid diff removals and must be kept.
        #    A fence is "stray" only if it's a bare fence line (not prefixed by +/-/space).
        if re.match(r'^```', stripped):
            # Check: is this a diff content line? (starts with +, -, or space)
            if not (line.startswith('+') or line.startswith('-') or line.startswith(' ')):
                continue  # Bare fence — skip it
            # Otherwise it's diff content like '-    #```python' — keep it
            
        # 2. Skip HTML tags (only bare ones, not diff content)
        if re.match(r'^</?(?:details|summary|br|hr)', stripped, re.IGNORECASE):
            if not (line.startswith('+') or line.startswith('-') or line.startswith(' ')):
                continue
        
        # 3. Handle 'diff --git' header to reset state
        if line.startswith('diff --git'):
            m = re.match(r'^diff --git a/(\S+) b/(\S+)', line)
            if m:
                current_file_a = m.group(1)
                current_file_b = m.group(2)
            else:
                current_file_a = None
                current_file_b = None
            seen_header_a = False
            seen_header_b = False
            in_hunk = False
            cleaned.append(line)
            continue
            
        # 4. Handle ALL 'index' lines — always skip.
        #    This includes 'index abc123..def456' and malformed 'index --- a/foo'.
        #    Never convert to --- headers (causes duplicate headers).
        if line.startswith('index '):
            continue
            
        # 5. Handle --- (Original file)
        if line.startswith('--- '):
            seen_header_a = True
            in_hunk = False
            cleaned.append(line)
            continue
            
        # 6. Handle +++ (New file)
        if line.startswith('+++ '):
            # CRITICAL: If we see +++ but missed ---, inject --- NOW to preserve order
            if not seen_header_a and current_file_a:
                cleaned.append(f"--- a/{current_file_a}")
                seen_header_a = True
                
            seen_header_b = True
            in_hunk = False
            cleaned.append(line)
            continue
            
        # 7. Handle Hunk Header @@
        # If we hit @@ and missed headers, inject them (respecting order)
        if line.startswith('@@ ') and current_file_a and current_file_b:
            if not seen_header_a:
                cleaned.append(f"--- a/{current_file_a}")
                seen_header_a = True
            if not seen_header_b:
                cleaned.append(f"+++ b/{current_file_b}")
                seen_header_b = True
            in_hunk = True
                
        cleaned.append(line)
    
    result = "\n".join(cleaned)
    if not result.endswith("\n"):
        result += "\n"
    return result


def extract_all_diffs(text: str) -> Optional[str]:
    """
    Extract unified diffs from model output.
    
    IMPORTANT: When the model outputs multiple diff blocks (reasoning drafts + final),
    we use only the LAST one, which is typically the final/correct version.
    
    Handles:
      - Multiple diffs inside a single fenced ```diff block
      - Multiple separate fenced blocks each containing a diff
      - Raw diffs starting with 'diff --git' (unfenced)
      - Mixed format: 'diff --git' header OUTSIDE a fenced block with hunks inside
    Returns the last (most likely correct) diff text or None.
    """
    t = text.strip()
    
    # --- Pre-processing: Merge split diffs ---
    # LLMs sometimes put 'diff --git ...' on a line BEFORE ```diff,
    # with the actual hunks inside the fenced block. Merge them.
    # Pattern: diff --git a/X b/X\n```diff\n@@ ...\n```
    t = re.sub(
        r'^(diff --git [^\n]+)\n\s*```(?:diff|python|python3)?\s*\n',
        r'\1\n',
        t,
        flags=re.MULTILINE
    )
    
    # Strategy 1: Look for fenced ```diff blocks and extract content
    fenced_diffs = []
    # Match all ```diff ... ``` blocks (or just ``` blocks containing diffs)
    # CRITICAL: Use line-anchored closing fence (^```\s*$) to avoid matching
    # backticks inside diff content lines like '-    #```python'
    fence_pattern = re.compile(r'```(?:diff)?\s*\n(.*?)^```\s*$', re.DOTALL | re.MULTILINE)
    for m in fence_pattern.finditer(t):
        block = m.group(1).strip()
        if 'diff --git' in block:
            fenced_diffs.append(block)
    
    if fenced_diffs:
        # Use the LAST diff block — model often puts reasoning diffs first,
        # then the final correct diff last ("Final Answer", "## Action", etc.)
        last_diff = fenced_diffs[-1]
        return sanitize_diff_text(last_diff)
    
    # Strategy 2: Find all raw 'diff --git' blocks (unfenced)
    # Split text at each 'diff --git' boundary and collect
    parts = re.split(r'(?=^diff --git )', t, flags=re.MULTILINE)
    raw_diffs = []
    for part in parts:
        part = part.strip()
        if part.startswith('diff --git'):
            # Clean trailing prose — stop at blank line followed by non-diff text
            # Keep everything that looks like diff content (lines starting with
            # diff, ---, +++, @@, +, -, space, or 'new file mode', backslash)
            diff_lines = []
            for line in part.split('\n'):
                # Lines that are valid diff content
                if (line.startswith('diff --git') or
                    line.startswith('---') or
                    line.startswith('+++') or
                    line.startswith('@@') or
                    line.startswith('+') or
                    line.startswith('-') or
                    line.startswith(' ') or
                    line.startswith('\\') or
                    line.startswith('index ') or
                    line.startswith('new file') or
                    line.startswith('old mode') or
                    line.startswith('new mode') or
                    line.startswith('deleted file') or
                    line.startswith('similarity') or
                    line.startswith('rename') or
                    line == ''):
                    diff_lines.append(line)
                else:
                    # Non-diff line (prose, markdown, etc.) — stop
                    break
            if diff_lines:
                raw_diffs.append('\n'.join(diff_lines))
    
    if raw_diffs:
        # For raw diffs, use the last complete diff block
        return sanitize_diff_text(raw_diffs[-1])
    
    return None

import re
from typing import List, Tuple

def extract_write_file_actions_v2(text: str) -> List[Tuple[str, str]]:
    """
    Extract WRITE_FILE actions with high-robustness regex.
    Handles:
    - Merged headers (e.g. 'code...WRITE_FILE: path')
    - Malformed closers (CONTENT>>, CONTENT]>>)
    - Truncated output (EOF)
    - Prose injection (stops at '## Reasoning')
    - Diff artifacts (ignores '-WRITE_FILE' or '-<<<CONTENT')
    - [NEW in v2] XML tag leakage from underlying LLM tool templates (e.g., </parameter>)
    """
    results = []
    
    # Regex Breakdown:
    # 1. (?:^|\n)(?!\-).*?WRITE_FILE: (Header, safe from diffs)
    # 2. \s*(\S+)                      (Capture filepath)
    # 3. .*?\n                         (Consume rest of header)
    # 4. \s*<<<CONTENT\n               (Start Tag)
    # 5. (.*?)                         (Content Capture)
    # 6. Terminator Group              (Various exit conditions including EOF)
    pattern = re.compile(
        r'(?:^|\n)(?!\-).*?WRITE_FILE:\s*(\S+).*?\n'  
        r'\s*<<<CONTENT\n'                            
        r'(.*?)'                                      
        r'(?:CONTENT>{2,3}|<<<CONTENT\s*$|(?=\n.*?WRITE_FILE:)|(?=\ndiff --git)|(?=\n\#\#\s)|(?=\n```)|$)',
        re.DOTALL
    )
    
    for m in pattern.finditer(text):
        filepath = m.group(1).strip()
        content = m.group(2)
        
        # --- POST-PROCESSING PIPELINE ---
        
        # 1. Strip accidentally captured proprietary closers
        for strip_tag in ["CONTENT>>>", "<<<CONTENT"]:
            if strip_tag in content:
                content = content.replace(strip_tag, "")
                
        # 2. [NEW in v2] Scrub trailing XML tool-call artifacts
        # This regex matches one or more closing XML tags (like </parameter>\n</function>)
        # strictly at the very END of the file content, ignoring whitespace/newlines in between.
        xml_artifact_pattern = r'(?:\s*</[a-zA-Z0-9_]+>)+\s*$'
        content = re.sub(xml_artifact_pattern, '', content)
        
        # 3. Diff Artifact check (double safety)
        # If the path looks like a diff path (a/foo.py, b/foo.py), ignore it
        if filepath.startswith("a/") or filepath.startswith("b/") or filepath == "/dev/null":
            continue
            
        # 4. Content validation
        # If content is extremely short (< 15 chars), it's likely a parsing artifact or hallucination
        if len(content.strip()) < 15:
            continue
            
        # Ensure the file ends with a clean, single trailing newline (POSIX standard)
        results.append((filepath, content.rstrip() + "\n"))
        
    return results

def extract_write_file_actions(text: str) -> List[Tuple[str, str]]:
    """
    Extract WRITE_FILE actions with high-robustness regex.
    Handles:
    - Merged headers (e.g. 'code...WRITE_FILE: path')
    - Malformed closers (CONTENT>>, CONTENT]>>)
    - Truncated output (EOF)
    - Prose injection (stops at '## Reasoning')
    - Diff artifacts (ignores '-WRITE_FILE' or '-<<<CONTENT')
    """
    results = []
    
    # Regex Breakdown:
    # 1. (?:^|\n)(?!\-).*?WRITE_FILE:
    #    - Matches start of line OR new line.
    #    - (?!\-) Negative lookahead: Ensure line does NOT start with '-' (diff removal).
    #    - .*? Consumes garbage prefix (e.g. 'model = ...').
    
    # 2. \s*(\S+)
    #    - Capture filepath (stops at whitespace).
    
    # 3. .*?\n
    #    - Consume rest of the header line.
    
    # 4. \s*<<<CONTENT\n
    #    - Match start tag. 
    #    - \s* matches spaces but NOT hyphens (diff safety).
    
    # 5. (.*?)
    #    - Capture content non-greedily.
    
    # 6. Terminator Group:
    #    - CONTENT>{2,3}        -> Normal closer (>>> or >>)
    #    - <<<CONTENT\s*$       -> Hallucinated closer
    #    - (?=\n.*?WRITE_FILE:) -> Lookahead: Next file starts
    #    - (?=\ndiff --git)     -> Lookahead: Diff starts
    #    - (?=\n\#\#\s)         -> Lookahead: Markdown header (e.g. ## Reasoning)
    #    - (?=\n```)            -> Lookahead: Code block fence
    #    - $                    -> EOF (Truncation)
    
    pattern = re.compile(
        r'(?:^|\n)(?!\-).*?WRITE_FILE:\s*(\S+).*?\n'  # Header (safe from diffs)
        r'\s*<<<CONTENT\n'                            # Start Tag
        r'(.*?)'                                      # Content Capture
        r'(?:CONTENT>{2,3}|<<<CONTENT\s*$|(?=\n.*?WRITE_FILE:)|(?=\ndiff --git)|(?=\n\#\#\s)|(?=\n```)|$)', # Robust Terminator
        re.DOTALL
    )
    
    for m in pattern.finditer(text):
        filepath = m.group(1).strip()
        content = m.group(2)
        
        # Post-processing: Strip accidentally captured closers
        for strip_tag in ["CONTENT>>>", "<<<CONTENT"]:
            if strip_tag in content:
                content = content.replace(strip_tag, "")
        
        # Post-processing checks
        
        # 1. Diff Artifact check (double safety)
        # If the path looks like a diff path (a/foo.py, b/foo.py), ignore it
        if filepath.startswith("a/") or filepath.startswith("b/") or filepath == "/dev/null":
            continue
            
        # 2. Content validation
        # If content is extremely short (< 15 chars), it's likely a parsing artifact or hallucination
        if len(content.strip()) < 15:
            continue
            
        results.append((filepath, content))
        
    return results




def apply_patch_guarded(diff_text: str, turn_dir: Path, auto_approve: bool = False) -> bool:
    """
    Apply patches robustly with multiple fallback strategies.
    1. Sanitize the diff text.
    2. Create parent directories for new files.
    3. Try git apply --check with --recount.
    4. If combined patch fails, try each diff block separately.
    5. Apply on success.
    """
    patch_path = turn_dir / "patch.diff"
    
    # Pre-sanitize
    diff_text = sanitize_diff_text(diff_text)
    patch_path.write_text(diff_text, encoding="utf-8")

    # Pre-create directories for new files mentioned in the diff
    for m in re.finditer(r'^\+\+\+ b/(.+)$', diff_text, re.MULTILINE):
        fpath = Path(m.group(1))
        fpath.parent.mkdir(parents=True, exist_ok=True)

    apply_log_parts = []

    def try_apply(patch_file: Path, label: str) -> bool:
        """Try applying a patch file with multiple strategies. Returns True on success."""
        strategies = [
            f"git apply --check --recount {patch_file.as_posix()}",
            f"git apply --check {patch_file.as_posix()}",
        ]
        for cmd_check in strategies:
            check_code, check_out = run_shell(cmd_check)
            apply_log_parts.append(f"[{cmd_check}] exit={check_code}\n{check_out}\n")
            
            if check_code == 0:
                # Check passed — apply
                cmd_apply = cmd_check.replace("--check ", "")
                app_code, app_out = run_shell(cmd_apply)
                apply_log_parts.append(f"[{cmd_apply}] exit={app_code}\n{app_out}\n")
                
                if app_code == 0:
                    console.print(f"[green]Patch applied ({label}).[/green]")
                    return True
                else:
                    console.print(f"[yellow]Apply failed after check passed ({label}): {app_out[:200]}[/yellow]")
        return False

    # Strategy 1: Try the full combined patch
    success = try_apply(patch_path, "combined")
    
    if not success:
        # Strategy 2: Split into individual file diffs and try each
        individual_diffs = re.split(r'(?=^diff --git )', diff_text, flags=re.MULTILINE)
        individual_diffs = [d for d in individual_diffs if d.strip().startswith('diff --git')]
        
        if len(individual_diffs) > 1:
            console.print(f"[yellow]Combined patch failed. Trying {len(individual_diffs)} individual patches...[/yellow]")
            all_ok = True
            for idx, single_diff in enumerate(individual_diffs):
                single_path = turn_dir / f"patch_part{idx}.diff"
                single_path.write_text(sanitize_diff_text(single_diff), encoding="utf-8")
                if not try_apply(single_path, f"part {idx+1}/{len(individual_diffs)}"):
                    all_ok = False
                    # Extract filename for error message
                    fname_m = re.search(r'diff --git a/(\S+)', single_diff)
                    fname = fname_m.group(1) if fname_m else f"part {idx+1}"
                    console.print(f"[red]Individual patch for {fname} also failed.[/red]")
            success = all_ok
    
    # Write full log
    (turn_dir / "apply.log").write_text("\n".join(apply_log_parts), encoding="utf-8")
    
    if not success:
        console.print(Panel(
            apply_log_parts[-1][:500] if apply_log_parts else "(no output)",
            title="Patch check failed", style="red"
        ))
    
    return success

def apply_fuzzy_patch(file_path: Path, diff_content: str, log_buffer: list = None) -> bool:
    """
    Applies a Unified Diff with 'fuzzy' matching logic.
    1. Ignores line numbers (@@ -12,4 +12,5 @@).
    2. Matches context by stripping whitespace (ignoring indentation changes).
    3. Falls back to anchor-based matching (first+last lines of search block).
    4. Handles 'New File' creation via diff.
    5. Preserves trailing newline state from original file.
    """
    def log(msg: str):
        if log_buffer is not None:
            log_buffer.append(msg)

    # 1. Handle New File Creation
    if "new file mode" in diff_content or "--- /dev/null" in diff_content:
         new_content = []
         for line in diff_content.splitlines():
             if line.startswith('+') and not line.startswith('+++'):
                 new_content.append(line[1:]) # Remove '+'
         
         # Sanity check: verify it's not just an empty file or metadata
         if len(new_content) > 0:
             file_path.parent.mkdir(parents=True, exist_ok=True)
             file_path.write_text("\n".join(new_content) + "\n", encoding="utf-8")
             msg = f"[green]Created new file from diff: {file_path}[/green]"
             console.print(msg)
             log(msg)
             return True
         log(f"New file creation failed: content empty for {file_path}")
         return False

    if not file_path.exists():
        msg = f"[red]Target file {file_path} not found for diff.[/red]"
        console.print(msg)
        log(msg)
        return False


    original_text = file_path.read_text(encoding="utf-8")
    had_trailing_newline = original_text.endswith("\n")
    original_lines = original_text.splitlines()
    # Work on a copy
    modified_lines = list(original_lines)
    
    # 2. Parse Hunks
    # Regex to split by @@ ... @@ header, consuming the rest of the line (e.g. function context)
    hunks = re.split(r'^@@\s.*?\s@@.*$', diff_content, flags=re.MULTILINE)
    # The first part is the header (diff --git ...), skip it
    hunks = hunks[1:]
    
    if not hunks:
        msg = "[yellow]No hunks found in diff.[/yellow]"
        console.print(msg)
        log(msg)
        return False
    
    applied_hunks = 0
    for hunk in hunks:
        # IMPORTANT: Don't filter out empty lines — they are valid context!
        # An empty line in a hunk (no leading +/-/space) is a context line.
        hunk_lines = hunk.splitlines()
        # Skip the very first element if it's empty (artifact from split)
        if hunk_lines and hunk_lines[0] == '':
            hunk_lines = hunk_lines[1:]
        if not hunk_lines:
            continue
            
        # 3. Identify the "Search Block" (Context + Removed lines)
        search_block = []
        replace_block = []
        
        for line in hunk_lines:
            if line.startswith(' '): # Context — remove leading space
                search_block.append(line[1:])
                replace_block.append(line[1:])
            elif line.startswith('-'): # Remove
                search_block.append(line[1:])
            elif line.startswith('+'): # Add
                replace_block.append(line[1:])
            elif line.startswith('\\'): # '\ No newline at end of file'
                pass
            elif line == '': # Empty context line (common in diffs)
                search_block.append('')
                replace_block.append('')
            # Any other line (shouldn't happen) — treat as context
            else:
                search_block.append(line)
                replace_block.append(line)
        
        if not search_block:
            # Pure addition hunk — no context. Insert at beginning if first hunk.
            if replace_block:
                for i, rl in enumerate(replace_block):
                    modified_lines.insert(i, rl)
                msg = f"[green]Applied pure-addition hunk ({len(replace_block)} lines)[/green]"
                console.print(msg)
                log(msg)
                applied_hunks += 1
            continue

        # Strategy 0: Already Applied?
        # If the *result* (replace_block) is already in the file, we can skip this hunk.
        # This handles cases where the file was partially updated or the model is repetitive.
        if replace_block:
            replace_stripped = [l.strip() for l in replace_block]
            n_replace = len(replace_block)
            found_already = -1
            
            # Fuzzy match (strip whitespace) for replace block
            for i in range(len(modified_lines) - n_replace + 1):
                file_subset = modified_lines[i : i+n_replace]
                if [l.strip() for l in file_subset] == replace_stripped:
                    found_already = i
                    break
            
            if found_already != -1:
                msg = f"[green]Hunk already applied (found replacement at line {found_already+1})[/green]"
                console.print(msg)
                log(msg)
                applied_hunks += 1
                continue

        # 4. Find where this block exists in the file
        match_index = -1
        n_search = len(search_block)
        
        # Strategy A: Exact Match
        for i in range(len(modified_lines) - n_search + 1):
            if modified_lines[i : i+n_search] == search_block:
                match_index = i
                log(f"Strategy A (Exact) match at line {i+1}")
                break
        
        # Strategy B: Fuzzy Match (strip whitespace)
        if match_index == -1:
            search_stripped = [l.strip() for l in search_block]
            for i in range(len(modified_lines) - n_search + 1):
                file_subset = modified_lines[i : i+n_search]
                file_stripped = [l.strip() for l in file_subset]
                if file_stripped == search_stripped:
                    match_index = i
                    msg = f"[green]Fuzzy-matched hunk at line {match_index+1} (whitespace-insensitive)[/green]"
                    console.print(msg)
                    log(msg)
                    break
        
        # Strategy C: Anchor Match (first + last non-empty lines)
        # When context has drifted or the file has extra/missing lines,
        # match using boundary lines as anchors. Search for the last
        # anchor independently within a window after the first anchor.
        n_delete = n_search  # Default: delete same number of lines as search block
        if match_index == -1 and n_search >= 2:
            # Find first and last non-empty search lines
            anchors = [(idx, l) for idx, l in enumerate(search_block) if l.strip()]
            if len(anchors) >= 2:
                first_idx, first_line = anchors[0]
                last_idx, last_line = anchors[-1]
                expected_span = last_idx - first_idx
                first_stripped = first_line.strip()
                last_stripped = last_line.strip()
                
                # Search for the first anchor
                for i in range(len(modified_lines)):
                    if modified_lines[i].strip() != first_stripped:
                        continue
                    
                    # Search for the last anchor within 2x expected span
                    max_end = min(len(modified_lines), i + expected_span * 2 + 2)
                    found_last = -1
                    # Allow search to start immediately after first anchor (handle missing/hallucinated context lines)
                    for j in range(i + 1, max_end):
                        if j < len(modified_lines) and modified_lines[j].strip() == last_stripped:
                            found_last = j
                            break
                    
                    if found_last != -1:
                        # Found both anchors!
                        match_index = i - first_idx
                        # Calculate how many lines to actually delete from the file
                        # This bridges the gap between where the first anchor matched and where the last matched.
                        # The 'last' anchor in the file is at index 'found_last'.
                        # The 'last' anchor in the search block is at index 'last_idx'.
                        # The match starts at `i - first_idx`.
                        # So the theoretical end of the match block is `i - first_idx + n_search`.
                        # But in the file, the end is at `found_last + (len(search_block) - last_idx - 1)`.
                        # Actually simpler: we are deleting from `match_index` to `found_last + <lines after last anchor>`.
                        
                        lines_after_last_anchor = len(search_block) - last_idx - 1
                        actual_end = found_last + lines_after_last_anchor + 1
                        n_delete = actual_end - match_index
                        
                        msg = f"[cyan]Anchor-matched hunk at line {match_index+1} (anchors at {match_index+1}..{actual_end}, deleting {n_delete} lines)[/cyan]"
                        console.print(msg)
                        log(msg)
                        break

        # Strategy D: Sliding Window Fuzzy Match
        if match_index == -1 and n_search >= 4:
            # Try to match a significant subset of lines (e.g. 60%) in sequence
            search_stripped = [l.strip() for l in search_block]
            best_ratio = 0
            best_pos = -1
            best_wsize = 0
            
            # Use a slightly smaller window to allow for missing lines
            min_window = max(3, int(n_search * 0.5))
            
            for wsize in range(n_search, min_window - 1, -1):
                # If we found a good match already, stop shrinking
                if best_ratio > 0.8: break
                
                window_stripped = search_stripped[:wsize]
                for i in range(len(modified_lines) - wsize + 1):
                    file_subset = [l.strip() for l in modified_lines[i : i+wsize]]
                    # Calculate similarity ratio
                    matches = sum(1 for a, b in zip(file_subset, window_stripped) if a == b)
                    ratio = matches / wsize
                    
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_pos = i
                        best_wsize = wsize
            
            if best_ratio >= 0.5 and best_pos >= 0:
                match_index = best_pos
                n_delete = best_wsize
                msg = f"[cyan]Partial-matched hunk at line {best_pos+1} ({best_ratio:.0%} match, window={best_wsize})[/cyan]"
                console.print(msg)
                log(msg)

        if match_index != -1:
            # Apply replacement
            # Remove n_delete lines starting from match_index
            # Insert replace_block
            
            # Handles index out of bounds if match is near end?
            # List slicing is forgiving.
            del modified_lines[match_index : match_index + n_delete]
            
            for i, line in enumerate(replace_block):
                modified_lines.insert(match_index + i, line)
            
            msg = f"[green]Applied hunk at line {match_index+1}[/green]"
            console.print(msg)
            log(msg)
            applied_hunks += 1
        else:
            msg = "[red]Failed to find matching context for hunk:[/red]"
            console.print(msg)
            log(msg)
            
            # Print context for debugging
            ctx_head = search_block[:5]
            ctx_box = Panel("\n".join(ctx_head) + ("\n..." if len(search_block) > 5 else ""), title="Expected Context (First 5 lines)")
            console.print(ctx_box)
            # Log expected context
            log("Expected Context snippet:")
            for l in ctx_head:
                log(f"| {l}")
    
    if applied_hunks == len(hunks):
        # Success! Write back
        new_text = "\n".join(modified_lines)
        if had_trailing_newline and not new_text.endswith("\n"):
            new_text += "\n"
        elif not had_trailing_newline and new_text.endswith("\n"):
            new_text = new_text[:-1]
            
        file_path.write_text(new_text, encoding="utf-8")
        return True
    
    return False
def extract_files_from_diff(diff_text: str) -> List[Tuple[str, str]]:
    """
    Extract file contents from diff '+' lines — ONLY FOR NEW FILES.
    
    SAFETY: This function ONLY extracts from diffs where `--- /dev/null`
    (i.e., entirely new files). For EDIT diffs (partial patches), scraping
    '+' lines would produce a tiny fragment and OVERWRITE the existing
    file, destroying it. This was the root cause of the 'task.py destroyed'
    bug in session 2026-02-16_215657.
    
    Returns list of (filepath, content) tuples.
    """
    results = []
    
    # Split into individual file diffs
    file_diffs = re.split(r'(?=^diff --git )', diff_text, flags=re.MULTILINE)
    file_diffs = [d for d in file_diffs if d.strip().startswith('diff --git')]
    
    for single_diff in file_diffs:
        # Extract target filename from diff header
        fname_match = re.search(r'diff --git a/\S+ b/(\S+)', single_diff)
        if not fname_match:
            continue
        filepath = fname_match.group(1)
        
        # CRITICAL SAFETY: Only extract from NEW FILE diffs
        is_new_file = ('new file mode' in single_diff or 
                       '--- /dev/null' in single_diff)
        
        if not is_new_file:
            console.print(f"[yellow]Skipping diff extraction for '{filepath}' "
                          f"(edit diff — would destroy existing file)[/yellow]")
            continue
        
        # Collect all '+' lines (for new files, every line is a '+' line)
        lines = single_diff.split('\n')
        content_lines = []
        in_hunk = False
        
        for line in lines:
            # Skip diff metadata lines
            if line.startswith('diff --git') or line.startswith('---') or line.startswith('+++'):
                continue
            if line.startswith('@@'):
                in_hunk = True
                continue
            if line.startswith('\\ No newline'):
                continue
            
            if in_hunk:
                if line.startswith('+'):
                    content_lines.append(line[1:])  # Remove leading '+'
                elif line.startswith(' '):
                    content_lines.append(line[1:])  # Context line
                elif line == '':
                    content_lines.append('')
        
        if not content_lines:
            continue
        
        # Join with newlines, ensure trailing newline
        content = '\n'.join(content_lines)
        if not content.endswith('\n'):
            content += '\n'
        results.append((filepath, content))
        console.print(f"[cyan]Extracted NEW file '{filepath}' from diff ({len(content)} bytes)[/cyan]")
    
    return results


def apply_write_files(
    actions: List[Tuple[str, str]], 
    allowlist: List[str], 
    turn_dir: Path
) -> bool:
    """
    Write files directly from WRITE_FILE actions.
    Validates paths against the allowlist.
    Returns True if at least one file was written.
    """
    written = 0
    log_parts = []
    
    # Normalize allowlist for comparison — convert PosixPath to str, extract basenames too
    norm_allowlist = set()
    for p in allowlist:
        s = str(p)
        norm_allowlist.add(s)                           # full path as string
        norm_allowlist.add(str(Path(s)))                 # normalized
        norm_allowlist.add(os.path.basename(s))          # just filename
        # Also add without leading dirs that might differ
        # e.g. "output/foo.py" from "./output/foo.py" or "/abs/output/foo.py"
        parts = Path(s).parts
        for i in range(len(parts)):
            norm_allowlist.add(str(Path(*parts[i:])))
    
    for filepath, content in actions:
        # Normalize the filepath
        clean_path = filepath.strip().lstrip('/')
        
        # Check if file is in allowlist (flexible matching)
        allowed = False
        for ap in norm_allowlist:
            ap_str = str(ap)
            if (clean_path == ap_str or 
                clean_path.endswith(ap_str) or 
                ap_str.endswith(clean_path) or
                os.path.basename(clean_path) == ap_str):
                allowed = True
                break
        # Also allow if no strict allowlist or auto mode
        if not norm_allowlist or not allowlist:
            allowed = True
            
        if not allowed:
            log_parts.append(f"SKIPPED (not in allowlist): {filepath} (allowlist: {[str(a) for a in allowlist]})")
            console.print(f"[yellow]Skipping {filepath} — not in allowlist ({[str(a) for a in allowlist]})[/yellow]")
            continue
        
        try:
            target = Path(clean_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            log_parts.append(f"WROTE: {filepath} ({len(content)} bytes)")
            console.print(f"[green]Wrote file: {filepath}[/green]")
            written += 1
            
            # Also git add if in a repo
            if is_git_repo():
                run_shell(f"git add {target.as_posix()}")
        except Exception as e:
            log_parts.append(f"FAILED: {filepath} — {e}")
            console.print(f"[red]Failed to write {filepath}: {e}[/red]")
    
    (turn_dir / "write_files.log").write_text("\n".join(log_parts), encoding="utf-8")
    return written > 0



def extract_json_robust(text: str) -> Optional[dict]:
    """
    Robustly extract JSON from LLM output.
    Tries multiple strategies including truncation repair.
    """
    text = text.strip()
    
    # Strip <think>...</think> tags (Qwen thinking mode)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    
    # Strategy 1: Direct parse
    try:
        return json.loads(text)
    except Exception:
        pass
    
    # Strategy 2: Extract from ```json block
    m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    
    # Strategy 3: Find first {...} in text using brace-matching
    start = text.find('{')
    if start >= 0:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i+1])
                    except Exception:
                        break
    
    # Strategy 4: Truncation repair — model hit max_tokens and JSON was cut off
    # Common pattern: {"complex": true, "steps": ["step1", "step2"  (missing ]})
    if start is not None and start >= 0:
        candidate = text[start:]
        # Try appending common missing closers
        for suffix in [']}', ']', '}', '"]}', '"]}']:
            try:
                return json.loads(candidate + suffix)
            except Exception:
                pass
        # Try finding last complete string and closing from there
        # Find the last complete quoted string
        last_quote = candidate.rfind('"')
        if last_quote > 0:
            # Try closing after last complete string
            trimmed = candidate[:last_quote+1]
            for suffix in [']}', ']}', ']}\n']:
                try:
                    return json.loads(trimmed + suffix)
                except Exception:
                    pass
    
    # Strategy 5: Try to fix common JSON issues (unquoted keys)
    m = re.search(r'\{[^{}]+\}', text)
    if m:
        candidate = m.group(0)
        fixed = re.sub(r'(\w+)\s*:', r'"\1":', candidate)
        try:
            return json.loads(fixed)
        except Exception:
            pass
    
    return None

# ---------------------------
# Sub-task Execution
# ---------------------------
def resolve_path(raw_path: str, allowlist: List[str], root_dir: Path = Path(".")) -> Optional[Path]:
    """
    Robustly resolves an LLM-generated path to a valid local file path.
    Prioritizes:
    1. Exact match in allowlist.
    2. Basename match in allowlist (e.g. '/abs/path/task.py' -> 'task.py').
    3. Relative path from root_dir.
    """
    clean = raw_path.strip().strip("'").strip('"').replace("\\", "/")
    clean = re.sub(r"^\./+", "", clean)
    clean = clean.lstrip("/")
    if not clean:
        return None

    root = root_dir.resolve()
    clean_parts = [part for part in Path(clean).parts if part not in ("", ".")]
    clean_parts_lower = [part.lower() for part in clean_parts]
    allowed_paths = [Path(p).expanduser().resolve() for p in allowlist if str(p).strip()]

    for allowed in allowed_paths:
        if allowed.as_posix() == Path(clean).as_posix() or str(allowed) == clean:
            return allowed

    suffix_matches: List[Path] = []
    for allowed in allowed_paths:
        allowed_parts_lower = [part.lower() for part in allowed.parts]
        if len(allowed_parts_lower) >= len(clean_parts_lower) and allowed_parts_lower[-len(clean_parts_lower):] == clean_parts_lower:
            suffix_matches.append(allowed)
    if len(suffix_matches) == 1:
        return suffix_matches[0]

    basename = Path(clean).name.lower()
    basename_matches = [p for p in allowed_paths if p.name.lower() == basename]
    if len(basename_matches) == 1:
        return basename_matches[0]

    candidate = (root / Path(clean)).resolve()
    if candidate.exists() or candidate.parent.exists():
        return candidate

    return None


from typing import List, Optional, Any

def _determine_verify_cmd(
    allowlist: List[str], 
    modified_files: List[str], 
    auto_verify_cmd: Optional[str], 
    config: Any
) -> str:
    """
    Determine the verification command based on context and LLM output.
    Priority:
    1. Model's explicit 'Verification:' line.
    2. Python file found in 'modified_files' (the file just generated).
    3. Python file found in 'allowlist'.
    """
    candidate = auto_verify_cmd
    
    # 2. Look for a runnable Python file in modified files
    if not candidate:
        py_files = [str(f) for f in modified_files if str(f).endswith('.py')]
        if py_files:
            candidate = f"python3 {py_files[0]}"
            
    # 3. Check allowlist as a fallback
    if not candidate:
        py_files = [str(f) for f in allowlist if str(f).endswith('.py')]
        if py_files:
            candidate = f"python3 {py_files[0]}"
    
    # ========================================================
    # [FIXED] Graceful Config Compatibility
    # ========================================================
    # If using the new UniversalAgent architecture (require_approval), 
    # the interactive prompt is handled at the Orchestrator level.
    # We just return the candidate string purely.
    if hasattr(config, 'require_approval'):
        return candidate or ""

    # Legacy Mode Compatibility (If other old scripts still call this)
    auto_approve = getattr(config, 'auto_approve', True)
    if not auto_approve:
        from rich.prompt import Prompt, Confirm
        if Confirm.ask("Run verification?", default=True):
            return Prompt.ask("Command", default=candidate or "").strip()
        return ""

    return candidate or ""

def run_linter(files: List[str]) -> Optional[str]:
    """
    Run fast static analysis (Ruff) on Python files.
    Catches syntax errors and undefined names before execution.
    Requires: pip install ruff
    """
    py_files = [str(f) for f in files if str(f).endswith('.py')]
    if not py_files:
        return None
    
    # E9: Syntax, F821: Undefined name, F823: Local var referenced before assign
    cmd = f"ruff check --select=E9,F821,F823 --output-format=text {' '.join(py_files)}"
    code, out = run_shell(cmd)
    
    if code != 0:
        return f"STATIC ANALYSIS FAILED (Ruff):\n{out}\n(Fix these syntax/name errors first!)"
    return None


# ---------------------------
# SkillDB
# ---------------------------

@dataclass
class Skill:
    category: str      # e.g., "PyTorch", "Syntax", "Logic", "API"
    pattern: str       # Trigger keywords (e.g., "conv2d", "plot")
    insight: str       # The lesson (e.g. "Do not use .cuda() on inputs...")
    evidence: str      # Short snippet or original output
    count: int = 1     # Dedup counter
    created_at: str = ""

def load_skills(skill_dir: Path) -> List[Skill]:
    skills: List[Skill] = []
    # Load separate success/failure logs or a unified DB
    # For v2, we can just load all jsonl files in skilldb
    if not skill_dir.exists():
        return []
        
    for path in skill_dir.glob("*.jsonl"):
        try:
            for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = line.strip()
                if not line: continue
                try:
                    obj = json.loads(line)
                    # Migration: Handle legacy format
                    if "insight" not in obj:
                        # Legacy skill
                        tag = obj.get("tag", "general")
                        kind = obj.get("kind", "unknown")
                        text = obj.get("text", "")
                        evidence = obj.get("evidence", "")
                        # Construct a basic migration
                        skills.append(Skill(
                            category="Legacy",
                            pattern=obj.get("pattern", "general"),
                            insight=f"Legacy {kind}: {text[:100]}...",
                            evidence=evidence[:200],  # Truncate legacy evidence
                            created_at=obj.get("created_at", "")
                        ))
                    else:
                        # New format
                        skills.append(Skill(
                            category=obj.get("category", "Uncategorized"),
                            pattern=obj.get("pattern", ""),
                            insight=obj.get("insight", ""),
                            evidence=obj.get("evidence", ""),
                            count=obj.get("count", 1),
                            created_at=obj.get("created_at", "")
                        ))
                except Exception:
                    continue
        except Exception:
            pass
    return skills

def score_skill(skill: Skill, query: str) -> int:
    """
    Scoring Strategy v2:
    - High match on Pattern
    - Medium match on Insight words
    """
    q = query.lower()
    s = 0
    
    # 1. Pattern Match (Strong signal)
    patt = (skill.pattern or "").lower().strip()
    if patt and patt in q:
        s += 5
        
    # 2. Insight Match (Contextual signal)
    # Tokenize insight
    words = re.findall(r"[a-zA-Z0-9_]{3,}", (skill.insight or "").lower())
    hits = 0
    for w in set(words):
        if w in q:
            hits += 1
    s += min(hits, 5)  # Cap at 5 points
    
    return s

def select_relevant_skills(goal_and_notes: str, skill_dir: Path, topk: int = 6) -> List[Skill]:
    skills = load_skills(skill_dir)
    scored = [(score_skill(sk, goal_and_notes), sk) for sk in skills]
    # Sort by score desc
    scored.sort(key=lambda x: x[0], reverse=True)
    # Filter only relevant ones (score > 2)
    picked = [sk for sc, sk in scored if sc >= 2][:topk]
    return picked

def format_skill_injection(skills: List[Skill]) -> str:
    if not skills:
        return ""
    
    # Group by category
    by_cat = {}
    for sk in skills:
        by_cat.setdefault(sk.category, []).append(sk)
    
    lines = ["## Teacher Guidelines (From Experience)"]
    for cat, sk_list in by_cat.items():
        if cat == "Legacy": continue # Skip legacy unless very relevant?
        lines.append(f"### {cat}")
        for sk in sk_list:
            # Format: "- [Pattern] Insight"
            lines.append(f"- [{sk.pattern}] {sk.insight}")
            
    if len(lines) == 1: # Only header
        return ""
        
    return "\n".join(lines).strip() + "\n"


def detect_tech_stack(goal: str, allowlist: List[str], skill_teacher_path: Path = None) -> str:
    """
    Heuristics to detect the tech stack (PyTorch, NumPy, etc.) 
    and return strict 'Teacher Guidelines' to prevent common runtime errors.
    Loads guidelines from SKILL_TEACHER (teacher.jsonl).
    """
    if not skill_teacher_path or not skill_teacher_path.exists():
        return ""

    goal_lower = goal.lower()
    combined_text = goal_lower + " ".join(str(x).lower() for x in allowlist)
    
    guidelines = []
    
    try:
        # Load teacher guidelines from JSONL
        # Format: {"category": "...", "triggers": [...], "header": "...", "guidelines": [...]}
        with open(skill_teacher_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                try:
                    entry = json.loads(line)
                    triggers = entry.get("triggers", [])
                    
                    # Check if any trigger matches the context
                    if any(t.lower() in combined_text for t in triggers):
                        header = entry.get("header")
                        if header:
                            guidelines.append(header)
                        
                        rules = entry.get("guidelines", [])
                        guidelines.extend(rules)
                        guidelines.append("") # Spacer
                        
                except json.JSONDecodeError:
                    continue
                    
    except Exception as e:
        console.print(f"[yellow]Failed to load teacher guidelines: {e}[/yellow]")
        return ""

    if guidelines:
        return "\n".join(guidelines).strip()
    return ""

def error_code_extraction(source_code: str, error_message: str = "") -> str:
    """
    Analyzes Python code for syntax errors using AST or extracts error context based on a provided traceback.
    Returns the numbered lines of code around the error for the LLM.
    """
    error_lines = set()

    # 1. Parse traceback for line numbers if provided
    if error_message:
        matches = re.finditer(r'line (\d+)', error_message)
        for match in matches:
            error_lines.add(int(match.group(1)))

    # 2. Try parsing AST for syntax errors
    try:
        ast.parse(source_code)
    except SyntaxError as e:
        if e.lineno:
            error_lines.add(e.lineno)
            if not error_message:
                error_message = f"SyntaxError: {e.msg} at line {e.lineno}"
    except Exception as e:
        pass # Other parsing errors ignored
    
    # 3. If no specific lines found, but there's an error
    if not error_lines:
        if error_message:
            return f"Error: {error_message}\n(No specific line numbers found in traceback)"
        return "No syntax errors detected."

    # 4. Format result
    result = []
    if error_message:
        result.append(f"Error Information:\n{error_message}\n")
    else:
        result.append("Error Information:\nSyntax Error detected.\n")
        
    result.append("Code Context:")
    # Use robust snippet extractor
    snippets = _extract_snippets_from_string(source_code, error_lines, window=5)
    result.append(snippets)
        
    return "\n".join(result)

def build_debug_prompt(traceback_str: str, window_size: int = 15, root_dir: str = ".") -> str:
    """
    Automatically extracts file skeletons and context from errors to build high-quality LLM prompts.
    """
    tb_pattern = re.compile(r'File\s+"([^"]+)",\s+line\s+(\d+)')
    error_locations: Dict[str, Set[int]] = {}
    
    for match in tb_pattern.finditer(traceback_str):
        filepath = match.group(1)
        line_num = int(match.group(2))
        
        abs_path = os.path.abspath(filepath)
        abs_root = os.path.abspath(root_dir)
        if not abs_path.startswith(abs_root) or "site-packages" in abs_path:
            continue
            
        if os.path.exists(filepath) and os.path.isfile(filepath):
            if filepath not in error_locations:
                error_locations[filepath] = set()
            error_locations[filepath].add(line_num)

    prompt_parts = []
    prompt_parts.append("# Debug Task\n")
    prompt_parts.append("## Error Traceback\n```text\n" + traceback_str.strip() + "\n```\n")

    for filepath, lines in error_locations.items():
        rel_path = os.path.relpath(filepath, root_dir)
        prompt_parts.append(f"## Context for `{rel_path}`\n")
        
        prompt_parts.append("### File Map (Structure)\n```python")
        prompt_parts.append(_generate_ast_map_from_file(filepath))
        prompt_parts.append("```\n")
        
        prompt_parts.append("### Error Context Snippets\n```python")
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                source = f.read()
            prompt_parts.append(_extract_snippets_from_string(source, lines, window_size))
        except Exception as e:
            prompt_parts.append(f"# [Warning] Could not extract snippets: {e}")
        prompt_parts.append("```\n")

    prompt_parts.append("## Instructions")
    prompt_parts.append("1. Analyze the Traceback and the provided Error Context Snippets.")
    prompt_parts.append("2. Use the File Map to understand the structure and available methods.")
    prompt_parts.append("3. Provide the corrected code using Format B (WRITE_FILE) for the entire file, or output the necessary diff.")
    
    return "\n".join(prompt_parts)


def _generate_ast_map_from_string(source: str) -> str:
    try:
        tree = ast.parse(source)
        lines = []
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                lines.append(f"class {node.name}:")
                for sub_node in node.body:
                    if isinstance(sub_node, ast.FunctionDef):
                        args = [arg.arg for arg in sub_node.args.args]
                        lines.append(f"    def {sub_node.name}({', '.join(args)}): ...")
            elif isinstance(node, ast.FunctionDef):
                args = [arg.arg for arg in node.args.args]
                lines.append(f"def {node.name}({', '.join(args)}): ...")
                
        if not lines:
            return "# No top-level classes or functions found."
        return "\n".join(lines)
    except SyntaxError as e:
        return f"# [Warning] SyntaxError in file, cannot generate AST: {e}"
    except Exception as e:
        return f"# [Warning] Could not generate file map: {e}"


def _generate_ast_map_from_file(filepath: str) -> str:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            source = f.read()
        return _generate_ast_map_from_string(source)
    except Exception as e:
        return f"# [Warning] Could not read file for AST: {e}"


def _extract_snippets_from_string(source: str, error_lines: Set[int], window: int) -> str:
    try:
        source_lines = source.splitlines()
        total_lines = len(source_lines)
        windows = []
        for eline in sorted(error_lines):
            start = max(1, eline - window)
            end = min(total_lines, eline + window)
            windows.append([start, end, {eline}])
            
        merged_windows = []
        for w in sorted(windows, key=lambda x: x[0]):
            if not merged_windows:
                merged_windows.append(w)
            else:
                prev = merged_windows[-1]
                if w[0] <= prev[1] + 1:
                    prev[1] = max(prev[1], w[1])
                    prev[2].update(w[2])
                else:
                    merged_windows.append(w)
                    
        snippets = []
        for start, end, elines in merged_windows:
            snippet_lines = [f"# --- Snippet from line {start} to {end} ---"]
            for i in range(start, end + 1):
                idx = i - 1
                if 0 <= idx < len(source_lines):
                    line_content = source_lines[idx]
                    marker = ">> " if i in elines else "   "
                    snippet_lines.append(f"{marker}{i:4d}: {line_content}")
            snippets.append("\n".join(snippet_lines))
            
        return "\n\n".join(snippets)
    except Exception as e:
        return f"# [Warning] Could not extract snippets: {e}"


# ---------------------------
# [NEW] Web Search Tool Implementation
# ---------------------------
import urllib.request # [NEW] For lightweight web search
def perform_web_search(query: str, api_key: str) -> str:
    """Lightweight Serper.dev integration for web search"""
    if not api_key:
        return "Error: SERPER_API_KEY is not set. Cannot perform web search."
    
    url = "https://google.serper.dev/search"
    req = urllib.request.Request(url, method="POST")
    req.add_header("X-API-KEY", api_key)
    req.add_header("Content-Type", "application/json")
    data = json.dumps({"q": query, "num": 5}).encode("utf-8")
    
    try:
        with urllib.request.urlopen(req, data=data, timeout=10) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            organic = res_data.get("organic", [])
            results = []
            for item in organic:
                title = item.get("title", "")
                snippet = item.get("snippet", "")
                link = item.get("link", "")
                results.append(f"Title: {title}\nSnippet: {snippet}\nSource: {link}\n")
            return "\n".join(results) if results else "No results found."
    except Exception as e:
        return f"Web search failed: {str(e)}"


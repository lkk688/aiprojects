可以，下面我给你一个增强版完整脚本，把你现有的 repo 扫描器扩展成一个 多源输入 → 自动抽取代码 → 生成 CPT / SFT 混合训练数据集 的工具。

它支持这些输入源混合：
	•	本地文件路径
	•	本地文件夹路径
	•	GitHub 单个代码文件 URL
	•	GitHub 文件夹 URL
	•	普通 raw 文件 URL

它会生成这些数据：
	•	raw_code.jsonl：原始代码语料，适合 CPT
	•	completion.jsonl：前缀/后缀补全，适合 SFT completion
	•	instruction.jsonl：自动构造 instruction/output，适合 SFT instruction
	•	mixed_sft.jsonl：混合后的 SFT 数据
	•	cpt_text.jsonl：直接可用于 CPT 的文本数据
	•	manifest.json：统计信息

同时支持：
	•	自动从 repo 代码里抽 function/class
	•	自动生成 heuristic instruction
	•	可选调用你的本地 LLM，用 complete_with_async(...) 生成更自然的 instruction
	•	去重
	•	忽略大文件 / 二进制 / 无关目录
	•	支持多种输出比例控制

⸻

设计建议

对于你这种“本地调试通过的 repo 代码”，我建议：
	•	CPT：直接吃 raw_code
	•	SFT：主要用
	•	completion
	•	instruction
	•	混合比例 推荐先这样：
	•	60% completion
	•	30% instruction
	•	10% raw_code-derived synthetic or fallback

如果你更看重 IDE 补全 / HumanEval：
	•	把 completion 提到 70%+

⸻

增强版脚本

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
prepare_mixed_code_dataset.py

Multi-source code collector + CPT/SFT dataset builder.

Supports:
- local file path
- local directory
- GitHub file URL
- GitHub folder URL
- raw file URL

Outputs:
- raw_code.jsonl        -> for CPT / completion source
- completion.jsonl      -> {"type":"completion","prefix","suffix",...}
- instruction.jsonl     -> {"type":"instruction","instruction","output",...}
- mixed_sft.jsonl       -> merged SFT data
- cpt_text.jsonl        -> {"text": "..."} ready for CPT
- manifest.json         -> summary stats

Optional:
- Call local LLM through complete_with_async(...) to create instruction variants.
"""

import os
import re
import ast
import io
import sys
import json
import time
import math
import asyncio
import random
import hashlib
import argparse
import mimetypes
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm


# =========================
# Config
# =========================
ALLOWED_EXTENSIONS = {
    ".py", ".ipynb",
    ".js", ".ts", ".jsx", ".tsx", ".vue",
    ".java", ".kt", ".scala",
    ".c", ".cpp", ".h", ".hpp", ".cs",
    ".go", ".rs",
    ".sh", ".bash", ".zsh",
    ".sql", ".json", ".yaml", ".yml", ".md"
}

IGNORE_DIRS = {
    ".git", ".svn", ".hg", ".idea", ".vscode",
    "node_modules", "dist", "build", "target",
    "__pycache__", "env", "venv", ".env",
    "site-packages", "migrations", ".next", ".nuxt",
    ".pytest_cache", ".mypy_cache", ".tox", ".cache",
    "coverage", "vendor"
}

MAX_FILE_SIZE = 1024 * 1024 * 2   # 2MB
MIN_FILE_SIZE = 10

DEFAULT_TIMEOUT = 30
DEFAULT_USER_AGENT = "CodeDatasetBuilder/1.0"

# Optional import: your wrapper
try:
    from BatchAgent.llm_wrapper import complete_with_async
except Exception:
    complete_with_async = None


# =========================
# Utilities
# =========================
def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(obj: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_text_file(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        if not text.strip():
            return None
        return text
    except Exception:
        return None


def is_probably_text(content: bytes) -> bool:
    if not content:
        return False
    if b"\x00" in content:
        return False
    try:
        content.decode("utf-8")
        return True
    except UnicodeDecodeError:
        try:
            content.decode("latin-1")
            return True
        except Exception:
            return False


def normalize_newlines(s: str) -> str:
    return s.replace("\r\n", "\n").replace("\r", "\n")


def looks_minified(text: str, ext: str) -> bool:
    if ext not in {".js", ".ts", ".css", ".json"}:
        return False
    lines = text.splitlines()
    if len(lines) <= 2 and len(text) > 2000:
        return True
    avg_line_len = (len(text) / max(1, len(lines)))
    return avg_line_len > 500


def trim_text(s: str, max_chars: int) -> str:
    if len(s) <= max_chars:
        return s
    return s[:max_chars].rstrip() + "\n"


def detect_language_from_ext(ext: str) -> str:
    ext = ext.lower()
    mapping = {
        ".py": "python",
        ".ipynb": "python",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".vue": "vue",
        ".java": "java",
        ".kt": "kotlin",
        ".scala": "scala",
        ".c": "c",
        ".cpp": "cpp",
        ".h": "c_header",
        ".hpp": "cpp_header",
        ".cs": "csharp",
        ".go": "go",
        ".rs": "rust",
        ".sh": "bash",
        ".bash": "bash",
        ".zsh": "zsh",
        ".sql": "sql",
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".md": "markdown",
    }
    return mapping.get(ext, ext.lstrip("."))


def path_depth(path: str) -> int:
    return len(path.replace("\\", "/").split("/"))


def unique_rows(rows: List[Dict[str, Any]], key_fields: List[str]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for row in rows:
        key = tuple(row.get(k, "") for k in key_fields)
        h = sha1_text(json.dumps(key, ensure_ascii=False, sort_keys=True))
        if h in seen:
            continue
        seen.add(h)
        out.append(row)
    return out


# =========================
# URL helpers
# =========================
def fetch_url_bytes(url: str, timeout: int = DEFAULT_TIMEOUT, headers: Optional[Dict[str, str]] = None) -> bytes:
    req = urllib.request.Request(
        url,
        headers=headers or {"User-Agent": DEFAULT_USER_AGENT, "Accept": "application/vnd.github+json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def fetch_url_text(url: str, timeout: int = DEFAULT_TIMEOUT, headers: Optional[Dict[str, str]] = None) -> str:
    data = fetch_url_bytes(url, timeout=timeout, headers=headers)
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1", errors="ignore")


def is_url(s: str) -> bool:
    return s.startswith("http://") or s.startswith("https://")


def is_github_url(url: str) -> bool:
    p = urllib.parse.urlparse(url)
    return p.netloc in {"github.com", "raw.githubusercontent.com", "api.github.com"}


def parse_github_blob_or_tree_url(url: str) -> Optional[Dict[str, str]]:
    """
    Supports:
    - https://github.com/owner/repo/blob/branch/path/to/file.py
    - https://github.com/owner/repo/tree/branch/path/to/folder
    Limit:
    - branch name with slashes is not fully handled.
    """
    p = urllib.parse.urlparse(url)
    parts = p.path.strip("/").split("/")
    if len(parts) < 4:
        return None
    owner, repo = parts[0], parts[1]
    mode = parts[2]
    branch = parts[3]
    rest = "/".join(parts[4:]) if len(parts) > 4 else ""
    if mode not in {"blob", "tree"}:
        return None
    return {
        "owner": owner,
        "repo": repo,
        "mode": mode,
        "branch": branch,
        "path": rest,
    }


def github_raw_url(owner: str, repo: str, branch: str, path: str) -> str:
    return f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"


def github_contents_api(owner: str, repo: str, path: str, ref: str) -> str:
    if path:
        return f"https://api.github.com/repos/{owner}/{repo}/contents/{path}?ref={ref}"
    return f"https://api.github.com/repos/{owner}/{repo}/contents?ref={ref}"


# =========================
# Source collection
# =========================
@dataclass
class CodeFile:
    source: str                   # local_path / github_file / github_folder / raw_url
    origin: str                   # original input string
    file_path: str                # relative/logical path
    ext: str
    language: str
    content: str
    size: int
    metadata: Dict[str, Any]


def collect_local_file(path: str, root_hint: Optional[str] = None) -> Optional[CodeFile]:
    if not os.path.isfile(path):
        return None

    _, ext = os.path.splitext(path)
    ext = ext.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return None

    try:
        size = os.path.getsize(path)
    except OSError:
        return None

    if size < MIN_FILE_SIZE or size > MAX_FILE_SIZE:
        return None

    content = read_text_file(path)
    if not content:
        return None
    content = normalize_newlines(content)
    if looks_minified(content, ext):
        return None

    if root_hint and os.path.exists(root_hint):
        try:
            rel_path = os.path.relpath(path, root_hint)
        except Exception:
            rel_path = os.path.basename(path)
    else:
        rel_path = os.path.basename(path)

    return CodeFile(
        source="local_file",
        origin=path,
        file_path=rel_path.replace("\\", "/"),
        ext=ext,
        language=detect_language_from_ext(ext),
        content=content,
        size=size,
        metadata={"absolute_path": os.path.abspath(path)},
    )


def collect_local_directory(root_dir: str) -> List[CodeFile]:
    rows: List[CodeFile] = []
    for current_root, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        for file in files:
            path = os.path.join(current_root, file)
            row = collect_local_file(path, root_hint=root_dir)
            if row is not None:
                rows.append(row)
    return rows


def collect_raw_url(url: str) -> Optional[CodeFile]:
    path = urllib.parse.urlparse(url).path
    _, ext = os.path.splitext(path)
    ext = ext.lower()

    if ext and ext not in ALLOWED_EXTENSIONS:
        return None

    try:
        data = fetch_url_bytes(url)
    except Exception:
        return None

    if len(data) < MIN_FILE_SIZE or len(data) > MAX_FILE_SIZE:
        return None
    if not is_probably_text(data):
        return None

    try:
        content = data.decode("utf-8")
    except UnicodeDecodeError:
        content = data.decode("latin-1", errors="ignore")

    content = normalize_newlines(content)
    if not content.strip():
        return None
    if looks_minified(content, ext):
        return None

    file_name = os.path.basename(path) or "downloaded_file"
    if not ext:
        ext = os.path.splitext(file_name)[1].lower()
    language = detect_language_from_ext(ext)

    return CodeFile(
        source="raw_url",
        origin=url,
        file_path=file_name,
        ext=ext,
        language=language,
        content=content,
        size=len(data),
        metadata={"url": url},
    )


def collect_github_file(url: str) -> Optional[CodeFile]:
    info = parse_github_blob_or_tree_url(url)
    if not info or info["mode"] != "blob":
        return None

    raw_url = github_raw_url(info["owner"], info["repo"], info["branch"], info["path"])
    row = collect_raw_url(raw_url)
    if row is None:
        return None
    row.source = "github_file"
    row.origin = url
    row.file_path = info["path"] or row.file_path
    row.metadata.update({
        "github_owner": info["owner"],
        "github_repo": info["repo"],
        "github_branch": info["branch"],
        "github_path": info["path"],
        "github_raw_url": raw_url,
    })
    return row


def collect_github_folder(url: str) -> List[CodeFile]:
    info = parse_github_blob_or_tree_url(url)
    if not info or info["mode"] != "tree":
        return []

    owner = info["owner"]
    repo = info["repo"]
    branch = info["branch"]
    folder = info["path"]

    api_url = github_contents_api(owner, repo, folder, branch)
    rows: List[CodeFile] = []

    def walk_folder(api_endpoint: str, logical_prefix: str = ""):
        try:
            text = fetch_url_text(api_endpoint, headers={
                "User-Agent": DEFAULT_USER_AGENT,
                "Accept": "application/vnd.github+json",
            })
            data = json.loads(text)
        except Exception:
            return

        if isinstance(data, dict) and data.get("type") == "file":
            download_url = data.get("download_url")
            path = data.get("path", "")
            if download_url:
                row = collect_raw_url(download_url)
                if row:
                    row.source = "github_folder"
                    row.origin = url
                    row.file_path = path
                    row.metadata.update({
                        "github_owner": owner,
                        "github_repo": repo,
                        "github_branch": branch,
                        "github_path": path,
                        "github_download_url": download_url,
                    })
                    rows.append(row)
            return

        if not isinstance(data, list):
            return

        for item in data:
            typ = item.get("type")
            path = item.get("path", "")
            if typ == "dir":
                name = os.path.basename(path)
                if name in IGNORE_DIRS:
                    continue
                sub_api = item.get("url")
                if sub_api:
                    walk_folder(sub_api, logical_prefix=path)
            elif typ == "file":
                name = os.path.basename(path)
                _, ext = os.path.splitext(name)
                ext = ext.lower()
                if ext not in ALLOWED_EXTENSIONS:
                    continue
                download_url = item.get("download_url")
                if not download_url:
                    continue
                row = collect_raw_url(download_url)
                if row:
                    row.source = "github_folder"
                    row.origin = url
                    row.file_path = path
                    row.metadata.update({
                        "github_owner": owner,
                        "github_repo": repo,
                        "github_branch": branch,
                        "github_path": path,
                        "github_download_url": download_url,
                    })
                    rows.append(row)

    walk_folder(api_url)
    return rows


def collect_input_item(item: str) -> List[CodeFile]:
    if is_url(item):
        if is_github_url(item):
            info = parse_github_blob_or_tree_url(item)
            if info and info["mode"] == "blob":
                row = collect_github_file(item)
                return [row] if row else []
            elif info and info["mode"] == "tree":
                return collect_github_folder(item)
            else:
                row = collect_raw_url(item)
                return [row] if row else []
        else:
            row = collect_raw_url(item)
            return [row] if row else []
    else:
        if os.path.isfile(item):
            row = collect_local_file(item, root_hint=os.path.dirname(item))
            return [row] if row else []
        elif os.path.isdir(item):
            return collect_local_directory(item)
        else:
            return []


# =========================
# Code analysis helpers
# =========================
PY_DOCSTRING_RE = re.compile(r'^\s*[ruRUfF]*("""|\'\'\')(.*?)(\1)', re.DOTALL)


def looks_like_python_code(text: str) -> bool:
    if not isinstance(text, str):
        return False
    t = text.strip()
    if len(t) < 20:
        return False
    hints = ("def ", "class ", "import ", "from ", "return ", "try:", "except ")
    return sum(int(h in t) for h in hints) >= 1


def safe_parse_python(text: str) -> bool:
    try:
        ast.parse(text)
        return True
    except Exception:
        return False


def extract_python_defs(content: str) -> List[Dict[str, Any]]:
    """
    Extract top-level functions/classes from Python file.
    """
    items: List[Dict[str, Any]] = []
    try:
        tree = ast.parse(content)
    except Exception:
        return items

    lines = content.splitlines(keepends=True)

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = node.lineno - 1
            end = getattr(node, "end_lineno", None)
            if end is None:
                continue
            code = "".join(lines[start:end]).strip()
            if not code:
                continue

            item = {
                "kind": "class" if isinstance(node, ast.ClassDef) else "function",
                "name": getattr(node, "name", ""),
                "lineno": node.lineno,
                "end_lineno": end,
                "code": code + "\n",
                "docstring": ast.get_docstring(node) or "",
            }

            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args = [a.arg for a in node.args.args]
                item["args"] = args
                item["signature"] = build_python_signature(node)
            else:
                item["args"] = []
                item["signature"] = f"class {node.name}"

            items.append(item)

    return items


def build_python_signature(node: ast.AST) -> str:
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return ""
    args = []
    for a in node.args.args:
        arg_name = a.arg
        if getattr(a, "annotation", None) is not None:
            try:
                ann = ast.unparse(a.annotation)
                arg_name = f"{arg_name}: {ann}"
            except Exception:
                pass
        args.append(arg_name)
    sig = f"def {node.name}(" + ", ".join(args) + ")"
    if getattr(node, "returns", None) is not None:
        try:
            ret = ast.unparse(node.returns)
            sig += f" -> {ret}"
        except Exception:
            pass
    return sig


def heuristic_instruction_for_python_def(item: Dict[str, Any], file_path: str) -> str:
    name = item.get("name", "")
    kind = item.get("kind", "function")
    signature = item.get("signature", "")
    docstring = (item.get("docstring") or "").strip()

    if kind == "class":
        base = f"Implement a Python class `{name}`"
    else:
        base = f"Implement a Python function `{name}`"

    if signature:
        base += f" with signature `{signature}`"

    if docstring:
        first_sentence = docstring.split("\n")[0].strip()
        if first_sentence:
            base += f" that {normalize_docstring_sentence(first_sentence)}"
    else:
        base += f" based on the code context from file `{file_path}`"

    base += ". Return only valid Python code."
    return base


def normalize_docstring_sentence(s: str) -> str:
    s = s.strip().rstrip(".")
    if not s:
        return "matches the intended behavior"
    if s[0].isupper():
        s = s[0].lower() + s[1:]
    if s.startswith("return "):
        return "returns " + s[len("return "):]
    return s


def extract_completion_pairs_python(
    code: str,
    seed: int = 3407,
    min_func_lines: int = 8,
    min_suffix_chars: int = 40,
    max_pairs_per_file: int = 3,
) -> List[Tuple[str, str]]:
    """
    Prefer function-body splits using AST.
    """
    if not safe_parse_python(code):
        return []

    rng = random.Random(seed ^ len(code))
    pairs: List[Tuple[str, str]] = []
    lines = code.splitlines(keepends=True)
    tree = ast.parse(code)

    candidates = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start = node.lineno - 1
            end = getattr(node, "end_lineno", node.lineno)
            if (end - start) >= min_func_lines and len(getattr(node, "body", [])) >= 2:
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
        if len(pairs) >= max_pairs_per_file:
            break

    return pairs


def extract_completion_pairs_generic(
    code: str,
    seed: int = 3407,
    min_lines: int = 8,
    min_suffix_chars: int = 40,
    max_pairs_per_file: int = 2,
) -> List[Tuple[str, str]]:
    lines = code.splitlines(keepends=True)
    if len(lines) < min_lines:
        return []

    rng = random.Random(seed ^ len(code))
    pairs = []
    for _ in range(max_pairs_per_file):
        lo = max(1, int(len(lines) * 0.2))
        hi = max(lo + 1, int(len(lines) * 0.8))
        if hi <= lo:
            continue
        cut = rng.randint(lo, hi - 1)
        prefix = "".join(lines[:cut])
        suffix = "".join(lines[cut:])
        if len(suffix.strip()) >= min_suffix_chars:
            pairs.append((prefix, suffix))
    return pairs


# =========================
# LLM instruction generation
# =========================
LLM_PROMPT_TEMPLATE = """You are creating high-quality code SFT training data.

Given a code snippet from a repository, generate {n_variants} concise programming instructions
that ask for exactly the behavior implemented by the code.

Rules:
- Do NOT add requirements not present in the code.
- Do NOT mention hidden implementation details unless obvious from code/docstring.
- Keep each instruction under 40 words.
- Focus on observable behavior.
- Return strict JSON:
{{"instructions": ["...", "..."]}}

File path:
{file_path}

Language:
{language}

Code:
```{language}
{code}

“””

async def llm_generate_instructions_for_item(
client: Any,
model: str,
provider: str,
backend: str,
file_path: str,
language: str,
code: str,
n_variants: int = 2,
) -> List[str]:
if complete_with_async is None:
return []

prompt = LLM_PROMPT_TEMPLATE.format(
    n_variants=n_variants,
    file_path=file_path,
    language=language,
    code=trim_text(code, 12000),
)

messages = [
    {"role": "system", "content": "You produce precise code dataset instructions."},
    {"role": "user", "content": prompt},
]

try:
    text, _ = await complete_with_async(
        client=client,
        model=model,
        messages=messages,
        temperature=0.2,
        max_output_tokens=1024,
        model_max_context=16384,
        provider=provider,
        backend=backend,
        stream=False,
        verbose=False,
        enable_thinking=True,
    )
except Exception:
    return []

text = text.strip()
try:
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start:end + 1]
    obj = json.loads(text)
    instrs = obj.get("instructions", [])
    instrs = [x.strip() for x in instrs if isinstance(x, str) and x.strip()]
    return instrs[:n_variants]
except Exception:
    return []

async def llm_generate_instruction_batch(
rows: List[Dict[str, Any]],
client: Any,
model: str,
provider: str,
backend: str,
concurrency: int = 4,
n_variants: int = 2,
) -> List[Dict[str, Any]]:
sem = asyncio.Semaphore(concurrency)
out: List[Dict[str, Any]] = []

async def worker(row: Dict[str, Any]):
    async with sem:
        instrs = await llm_generate_instructions_for_item(
            client=client,
            model=model,
            provider=provider,
            backend=backend,
            file_path=row["file_path"],
            language=row["language"],
            code=row["output"],
            n_variants=n_variants,
        )
        local = []
        for instr in instrs:
            local.append({
                "type": "instruction",
                "instruction": instr,
                "output": row["output"],
                "language": row["language"],
                "file_path": row["file_path"],
                "source": row["source"],
                "source_kind": row.get("source_kind", "llm_instruction"),
                "metadata": {
                    **row.get("metadata", {}),
                    "instruction_origin": "llm",
                }
            })
        return local

tasks = [worker(r) for r in rows]
for batch in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="LLM instructions"):
    try:
        items = await batch
        out.extend(items)
    except Exception:
        continue
return out

=========================

Dataset builders

=========================

def build_raw_code_dataset(files: List[CodeFile], max_chars: int) -> List[Dict[str, Any]]:
rows = []
for f in files:
rows.append({
“type”: “raw_code”,
“code”: trim_text(f.content, max_chars),
“language”: f.language,
“file_path”: f.file_path,
“source”: f.source,
“origin”: f.origin,
“size”: f.size,
“metadata”: f.metadata,
})
return rows

def build_cpt_dataset(raw_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
rows = []
for r in raw_rows:
code = r[“code”]
if not code.strip():
continue
rows.append({
“text”: code,
“language”: r[“language”],
“file_path”: r[“file_path”],
“source”: r[“source”],
“origin”: r[“origin”],
})
return rows

def build_completion_dataset(
files: List[CodeFile],
seed: int,
max_chars: int,
max_pairs_per_file: int = 3,
) -> List[Dict[str, Any]]:
rows: List[Dict[str, Any]] = []

for i, f in enumerate(files):
    code = trim_text(f.content, max_chars)
    if not code.strip():
        continue

    if f.language == "python" and safe_parse_python(code):
        pairs = extract_completion_pairs_python(
            code,
            seed=seed + i,
            max_pairs_per_file=max_pairs_per_file,
        )
    else:
        pairs = extract_completion_pairs_generic(
            code,
            seed=seed + i,
            max_pairs_per_file=max_pairs_per_file,
        )

    for j, (prefix, suffix) in enumerate(pairs):
        rows.append({
            "type": "completion",
            "prefix": prefix,
            "suffix": suffix,
            "language": f.language,
            "file_path": f.file_path,
            "source": f.source,
            "origin": f.origin,
            "metadata": {
                **f.metadata,
                "pair_index": j,
                "completion_origin": "repo_split",
            }
        })
return rows

def build_instruction_dataset_heuristic(
files: List[CodeFile],
max_chars: int,
) -> List[Dict[str, Any]]:
rows: List[Dict[str, Any]] = []

for f in files:
    code = trim_text(f.content, max_chars)

    if f.language == "python" and safe_parse_python(code):
        defs = extract_python_defs(code)
        for item in defs:
            snippet = item["code"]
            if len(snippet.strip()) < 40:
                continue
            instr = heuristic_instruction_for_python_def(item, f.file_path)
            rows.append({
                "type": "instruction",
                "instruction": instr,
                "output": snippet,
                "language": "python",
                "file_path": f.file_path,
                "source": f.source,
                "origin": f.origin,
                "source_kind": "heuristic_python_def",
                "metadata": {
                    **f.metadata,
                    "name": item.get("name", ""),
                    "kind": item.get("kind", ""),
                    "signature": item.get("signature", ""),
                    "instruction_origin": "heuristic",
                }
            })
    else:
        # fallback: whole-file weak instruction for non-Python
        if len(code.strip()) < 60:
            continue
        instr = f"Implement the code in `{f.file_path}`. Return only valid {f.language} code."
        rows.append({
            "type": "instruction",
            "instruction": instr,
            "output": code,
            "language": f.language,
            "file_path": f.file_path,
            "source": f.source,
            "origin": f.origin,
            "source_kind": "heuristic_whole_file",
            "metadata": {
                **f.metadata,
                "instruction_origin": "heuristic",
            }
        })

return rows

def mix_sft_datasets(
completion_rows: List[Dict[str, Any]],
instruction_rows: List[Dict[str, Any]],
raw_rows: List[Dict[str, Any]],
seed: int,
completion_ratio: float = 0.6,
instruction_ratio: float = 0.3,
raw_as_instruction_ratio: float = 0.1,
max_samples: int = 100000,
) -> List[Dict[str, Any]]:
rng = random.Random(seed)

raw_as_instruction: List[Dict[str, Any]] = []
for r in raw_rows:
    raw_as_instruction.append({
        "type": "instruction",
        "instruction": f"Write code equivalent to the implementation in `{r['file_path']}`. Return only valid {r['language']} code.",
        "output": r["code"],
        "language": r["language"],
        "file_path": r["file_path"],
        "source": r["source"],
        "origin": r["origin"],
        "source_kind": "raw_fallback_instruction",
        "metadata": {
            **r.get("metadata", {}),
            "instruction_origin": "fallback",
        }
    })

def sample_pool(pool: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    if not pool or n <= 0:
        return []
    if len(pool) >= n:
        idxs = list(range(len(pool)))
        rng.shuffle(idxs)
        return [pool[k] for k in idxs[:n]]
    out = []
    while len(out) < n:
        out.append(pool[rng.randrange(len(pool))])
    return out[:n]

c_n = int(max_samples * completion_ratio)
i_n = int(max_samples * instruction_ratio)
r_n = max_samples - c_n - i_n

mixed = []
mixed.extend(sample_pool(completion_rows, c_n))
mixed.extend(sample_pool(instruction_rows, i_n))
mixed.extend(sample_pool(raw_as_instruction, r_n))

rng.shuffle(mixed)
return mixed

=========================

Main pipeline

=========================

@dataclass
class CFG:
inputs: List[str]
output_dir: str
mode: str = “both”                      # cpt / sft / both
seed: int = 3407
max_chars_per_file: int = 12000
max_pairs_per_file: int = 3
completion_ratio: float = 0.6
instruction_ratio: float = 0.3
raw_ratio: float = 0.1
max_sft_samples: int = 50000
use_llm_instruction: bool = False
llm_model: str = “”
llm_provider: str = “openai”
llm_backend: str = “openai”
llm_instruction_variants: int = 2
llm_concurrency: int = 4

def collect_all_sources(inputs: List[str]) -> Tuple[List[CodeFile], Dict[str, Any]]:
all_rows: List[CodeFile] = []
stats = {
“inputs_total”: len(inputs),
“files_collected”: 0,
“by_source”: {},
“by_language”: {},
}

for item in inputs:
    rows = collect_input_item(item)
    all_rows.extend(rows)

dedup_map = {}
deduped: List[CodeFile] = []
for row in all_rows:
    h = sha1_text(row.content)
    if h in dedup_map:
        continue
    dedup_map[h] = True
    deduped.append(row)

for r in deduped:
    stats["by_source"][r.source] = stats["by_source"].get(r.source, 0) + 1
    stats["by_language"][r.language] = stats["by_language"].get(r.language, 0) + 1

stats["files_collected"] = len(deduped)
return deduped, stats

async def maybe_expand_instruction_with_llm(
heuristic_instruction_rows: List[Dict[str, Any]],
cfg: CFG,
client: Any = None,
) -> List[Dict[str, Any]]:
if not cfg.use_llm_instruction:
return []

if complete_with_async is None:
    print("[WARN] complete_with_async not available, skip LLM instruction generation.")
    return []

if client is None:
    print("[WARN] use_llm_instruction=True but no client provided, skip.")
    return []

# Prefer smaller subset of good-quality rows
candidates = []
for r in heuristic_instruction_rows:
    output = r.get("output", "")
    if 80 <= len(output) <= 6000:
        candidates.append(r)

return await llm_generate_instruction_batch(
    rows=candidates,
    client=client,
    model=cfg.llm_model,
    provider=cfg.llm_provider,
    backend=cfg.llm_backend,
    concurrency=cfg.llm_concurrency,
    n_variants=cfg.llm_instruction_variants,
)

async def build_all_datasets(cfg: CFG, client: Any = None) -> Dict[str, Any]:
ensure_dir(cfg.output_dir)

print("[*] Collecting sources...")
files, collect_stats = collect_all_sources(cfg.inputs)

print(f"[+] Collected {len(files)} unique files.")
for k, v in collect_stats["by_source"].items():
    print(f"    source[{k}] = {v}")
for k, v in sorted(collect_stats["by_language"].items(), key=lambda x: (-x[1], x[0]))[:20]:
    print(f"    lang[{k}] = {v}")

raw_rows = build_raw_code_dataset(files, cfg.max_chars_per_file)
raw_rows = unique_rows(raw_rows, ["code"])

cpt_rows = build_cpt_dataset(raw_rows) if cfg.mode in {"cpt", "both"} else []

completion_rows = []
instruction_rows = []

if cfg.mode in {"sft", "both"}:
    print("[*] Building completion dataset...")
    completion_rows = build_completion_dataset(
        files,
        seed=cfg.seed,
        max_chars=cfg.max_chars_per_file,
        max_pairs_per_file=cfg.max_pairs_per_file,
    )
    completion_rows = unique_rows(completion_rows, ["prefix", "suffix"])

    print("[*] Building heuristic instruction dataset...")
    instruction_rows = build_instruction_dataset_heuristic(
        files,
        max_chars=cfg.max_chars_per_file,
    )
    instruction_rows = unique_rows(instruction_rows, ["instruction", "output"])

    if cfg.use_llm_instruction:
        print("[*] Expanding instruction dataset with local LLM...")
        llm_rows = await maybe_expand_instruction_with_llm(
            instruction_rows,
            cfg,
            client=client,
        )
        llm_rows = unique_rows(llm_rows, ["instruction", "output"])
        instruction_rows.extend(llm_rows)
        instruction_rows = unique_rows(instruction_rows, ["instruction", "output"])

mixed_sft_rows = []
if cfg.mode in {"sft", "both"}:
    print("[*] Mixing SFT datasets...")
    mixed_sft_rows = mix_sft_datasets(
        completion_rows=completion_rows,
        instruction_rows=instruction_rows,
        raw_rows=raw_rows,
        seed=cfg.seed,
        completion_ratio=cfg.completion_ratio,
        instruction_ratio=cfg.instruction_ratio,
        raw_as_instruction_ratio=cfg.raw_ratio,
        max_samples=cfg.max_sft_samples,
    )
    mixed_sft_rows = unique_rows(mixed_sft_rows, ["type", "instruction", "output", "prefix", "suffix"])

outputs = {}

raw_path = os.path.join(cfg.output_dir, "raw_code.jsonl")
write_jsonl(raw_path, raw_rows)
outputs["raw_code"] = raw_path

if cfg.mode in {"cpt", "both"}:
    cpt_path = os.path.join(cfg.output_dir, "cpt_text.jsonl")
    write_jsonl(cpt_path, cpt_rows)
    outputs["cpt_text"] = cpt_path

if cfg.mode in {"sft", "both"}:
    completion_path = os.path.join(cfg.output_dir, "completion.jsonl")
    instruction_path = os.path.join(cfg.output_dir, "instruction.jsonl")
    mixed_path = os.path.join(cfg.output_dir, "mixed_sft.jsonl")

    write_jsonl(completion_path, completion_rows)
    write_jsonl(instruction_path, instruction_rows)
    write_jsonl(mixed_path, mixed_sft_rows)

    outputs["completion"] = completion_path
    outputs["instruction"] = instruction_path
    outputs["mixed_sft"] = mixed_path

manifest = {
    "config": asdict(cfg),
    "stats": {
        "files_collected": len(files),
        "raw_code_rows": len(raw_rows),
        "cpt_rows": len(cpt_rows),
        "completion_rows": len(completion_rows),
        "instruction_rows": len(instruction_rows),
        "mixed_sft_rows": len(mixed_sft_rows),
        "by_source": collect_stats["by_source"],
        "by_language": collect_stats["by_language"],
    },
    "outputs": outputs,
}
manifest_path = os.path.join(cfg.output_dir, "manifest.json")
save_json(manifest, manifest_path)
outputs["manifest"] = manifest_path

return {
    "files": files,
    "raw_rows": raw_rows,
    "cpt_rows": cpt_rows,
    "completion_rows": completion_rows,
    "instruction_rows": instruction_rows,
    "mixed_sft_rows": mixed_sft_rows,
    "manifest": manifest,
}

=========================

CLI

=========================

def parse_args() -> CFG:
parser = argparse.ArgumentParser(description=“Build CPT/SFT code datasets from local paths and URLs”)
parser.add_argument(
“–inputs”,
nargs=”+”,
required=True,
help=“Multiple inputs: local file / local folder / github file URL / github folder URL / raw URL”
)
parser.add_argument(”–output_dir”, type=str, default=“prepared_code_dataset”)
parser.add_argument(”–mode”, type=str, default=“both”, choices=[“cpt”, “sft”, “both”])

parser.add_argument("--seed", type=int, default=3407)
parser.add_argument("--max_chars_per_file", type=int, default=12000)
parser.add_argument("--max_pairs_per_file", type=int, default=3)

parser.add_argument("--completion_ratio", type=float, default=0.6)
parser.add_argument("--instruction_ratio", type=float, default=0.3)
parser.add_argument("--raw_ratio", type=float, default=0.1)
parser.add_argument("--max_sft_samples", type=int, default=50000)

parser.add_argument("--use_llm_instruction", action="store_true")
parser.add_argument("--llm_model", type=str, default="")
parser.add_argument("--llm_provider", type=str, default="openai")
parser.add_argument("--llm_backend", type=str, default="openai")
parser.add_argument("--llm_instruction_variants", type=int, default=2)
parser.add_argument("--llm_concurrency", type=int, default=4)

args = parser.parse_args()

total = args.completion_ratio + args.instruction_ratio + args.raw_ratio
if abs(total - 1.0) > 1e-6:
    raise ValueError("completion_ratio + instruction_ratio + raw_ratio must equal 1.0")

return CFG(
    inputs=args.inputs,
    output_dir=args.output_dir,
    mode=args.mode,
    seed=args.seed,
    max_chars_per_file=args.max_chars_per_file,
    max_pairs_per_file=args.max_pairs_per_file,
    completion_ratio=args.completion_ratio,
    instruction_ratio=args.instruction_ratio,
    raw_ratio=args.raw_ratio,
    max_sft_samples=args.max_sft_samples,
    use_llm_instruction=args.use_llm_instruction,
    llm_model=args.llm_model,
    llm_provider=args.llm_provider,
    llm_backend=args.llm_backend,
    llm_instruction_variants=args.llm_instruction_variants,
    llm_concurrency=args.llm_concurrency,
)

=========================

Client hook

=========================

def build_llm_client_from_env() -> Any:
“””
You should replace this with your own local client builder if needed.

Example for OpenAI-compatible local server:
    from openai import AsyncOpenAI
    return AsyncOpenAI(base_url=os.environ["OPENAI_BASE_URL"], api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"))
"""
try:
    from openai import AsyncOpenAI
    base_url = os.environ.get("OPENAI_BASE_URL")
    api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")
    if base_url:
        return AsyncOpenAI(base_url=base_url, api_key=api_key)
except Exception:
    pass
return None

async def main_async():
cfg = parse_args()
client = build_llm_client_from_env() if cfg.use_llm_instruction else None

result = await build_all_datasets(cfg, client=client)
manifest = result["manifest"]

print("\n[Success] Dataset build complete.")
print(json.dumps(manifest["stats"], ensure_ascii=False, indent=2))
print("\n[Outputs]")
for k, v in manifest["outputs"].items():
    print(f"  {k}: {os.path.abspath(v)}")

def main():
asyncio.run(main_async())

if name == “main”:
main()


"""


# 怎么用

## 1. 本地 repo 文件夹
```bash
python prepare_mixed_code_dataset.py \
  --inputs ../../MyRepo/DeepDataMiningLearning/ \
  --output_dir out_repo_dataset \
  --mode both

2. 多个本地目录 + 单文件

python prepare_mixed_code_dataset.py \
  --inputs ../../RepoA ../../RepoB ./some_file.py \
  --output_dir out_multi \
  --mode both

3. GitHub 文件 + GitHub 文件夹 + 本地目录混合

python prepare_mixed_code_dataset.py \
  --inputs \
    https://github.com/owner/repo/blob/main/src/model.py \
    https://github.com/owner/repo/tree/main/src \
    ../../MyRepo/DeepDataMiningLearning/ \
  --output_dir out_mixed \
  --mode both

4. 开启本地 LLM 生成 instruction

前提是你本地有 OpenAI-compatible 服务，并设置环境变量：

export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
export OPENAI_API_KEY=EMPTY

然后：

python prepare_mixed_code_dataset.py \
  --inputs ../../MyRepo/DeepDataMiningLearning/ \
  --output_dir out_llm \
  --mode both \
  --use_llm_instruction \
  --llm_model Qwen/Qwen2.5-Coder-7B-Instruct \
  --llm_provider openai \
  --llm_backend openai \
  --llm_instruction_variants 2


⸻

生成的数据分别怎么用

raw_code.jsonl

适合你后面再做：
	•	completion 再抽取
	•	CPT 语料
	•	自定义数据清洗

格式示例：

{"type":"raw_code","code":"import torch\n...","language":"python","file_path":"models/a.py", ...}

cpt_text.jsonl

适合直接做 CPT：

{"text":"import torch\n...","language":"python","file_path":"models/a.py", ...}

completion.jsonl

适合你的 SFT 训练脚本里的 custom completion bucket：

{"type":"completion","prefix":"def f(x):\n    ...","suffix":"return y\n", ...}

instruction.jsonl

适合 custom instruction bucket：

{"type":"instruction","instruction":"Implement a Python function `compute_iou` ...","output":"def compute_iou(...):\n...", ...}

mixed_sft.jsonl

直接作为你的自有混合 SFT 数据输入源。

⸻

这个版本相比你原始脚本增强了什么

你原始版本本质上只是：
	•	扫目录
	•	读文件
	•	存 content

现在这个版本多了：
	1.	支持多输入源
	•	本地文件
	•	本地目录
	•	GitHub 文件
	•	GitHub 文件夹
	•	raw URL
	2.	支持 CPT + SFT 双输出
	3.	支持结构化 completion 样本生成
	4.	支持 instruction 自动构造
	•	heuristic
	•	LLM 增强
	5.	支持混合 SFT 训练集生成
	6.	支持 manifest 统计

⸻

我对你后续训练的建议

如果这些 repo 都是你们自己调试通过、质量高的代码，我建议：

用于 SFT

优先吃：
	•	completion.jsonl
	•	instruction.jsonl
	•	mixed_sft.jsonl

用于 CPT

用：
	•	cpt_text.jsonl

第一版训练建议

先不要让 LLM instruction 占太高，建议：
	•	heuristic instruction 为主
	•	LLM instruction 只是增强
	•	completion 仍是最大头

你可以先这样：

--completion_ratio 0.65
--instruction_ratio 0.25
--raw_ratio 0.10


⸻

两个很重要的注意点

1. GitHub folder URL 解析限制

这个脚本对 github.com/.../tree/branch/path 的解析，默认假设 branch 没有复杂斜杠命名。
大部分仓库主分支 main/master/dev 没问题。

2. instruction 质量

自动 instruction 里：
	•	heuristic 最稳
	•	LLM 生成更自然，但可能会轻微漂移
所以最好先让 LLM 只做 paraphrase，不要自己发明需求。

⸻

如果你要，我下一条可以继续直接给你：

“把这个数据构造脚本和你前面那个 Qwen SFT 训练脚本无缝接起来，支持 --custom_jsonl 自动训练”。
"""
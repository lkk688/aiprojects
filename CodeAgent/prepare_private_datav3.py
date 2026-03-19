#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
prepare_private_datav3.py — Multi-source Code Dataset Builder

Enhances v2 with:
  1. GitHub repo support      — clone a list of repos and ingest all code
  2. CPT data                 — raw code formatted for continued pretraining
  3. Rich SFT task varieties  — 7 complementary task types for code navigation:
       skeleton_to_impl   : function signature + docstring → full body
       partial_nav        : full file with one function stubbed → implement it
       code_to_docstring  : strip docstring from documented function → regenerate
       completion         : prefix/suffix splits (AST-guided or line-based)
       import_resolution  : strip import section → restore imports
       generate_main      : existing __main__ block as target
       test_pair          : source module + matching test file

All SFT items are saved as  {"user": "...", "assistant": "...", "type": "..."}
and can be loaded directly by qwen_coder_sft_v5.py with make_chat_text().

Usage
─────
# From local directories
python CodeAgent/prepare_private_datav3.py \\
    --local_dirs /path/to/repo1 /path/to/repo2 \\
    --output_dir data/private_v3

# From GitHub repos (cloned automatically)
python CodeAgent/prepare_private_datav3.py \\
    --github_repos \\
        https://github.com/user/repo1 \\
        https://github.com/user/repo2 \\
    --output_dir data/private_v3

# Mixed + size controls
python CodeAgent/prepare_private_datav3.py \
    --local_dirs \
        /data/rnd-liu/MyRepo/DeepDataMiningLearning \
        /data/rnd-liu/MyRepo/edgeAI \
    --github_repos https://github.com/huggingface/transformers \
    --output_dir data/private_v1 \
    --max_files_per_repo 800 \
    --max_sft_per_file 6 \
    --sft_tasks skeleton_to_impl partial_nav code_to_docstring completion

Output files
────────────
  {output_dir}/cpt_data.jsonl     — raw code for CPT
  {output_dir}/sft_{task}.jsonl   — per-task SFT
  {output_dir}/sft_all.jsonl      — merged SFT (all tasks)
  {output_dir}/manifest.json      — statistics

Every code file wrapped as # File: path\n# Language: ...\n\n{content} — ready for continued pretraining.

7 SFT Task Varieties (sft_{task}.jsonl + merged sft_all.jsonl):

Task	What it teaches
skeleton_to_impl	Function signature + docstring → full body (imports + class context shown)
partial_nav	Full file with one function stubbed out → implement it (code navigation)
code_to_docstring	Strip docstring from well-documented function → regenerate
completion	AST-guided prefix/suffix split within a function body
import_resolution	Strip import section → restore all imports
generate_main	Module body → realistic __main__ block
test_pair	Source file + matching test_*.py → full test file
Each task uses a pool of varied prompt templates for generalisation. sft_all.jsonl output uses {"user": ..., "assistant": ...} format consumed directly by v5.
  
"""

import os
import re
import ast
import sys
import json
import random
import hashlib
import argparse
import shutil
import subprocess
import tempfile
import textwrap
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):  # simple fallback
        return it


# =====================================================================
# Constants
# =====================================================================
ALLOWED_EXTENSIONS = {
    ".py", ".ipynb",
    ".js", ".ts", ".jsx", ".tsx", ".vue",
    ".java", ".kt", ".scala",
    ".c", ".cpp", ".h", ".hpp", ".cs",
    ".go", ".rs",
    ".sh", ".bash", ".zsh",
    ".sql", ".json", ".yaml", ".yml", ".md",
}
CODE_EXTENSIONS = {".py", ".js", ".ts", ".java", ".go", ".rs", ".cpp", ".c", ".cs"}
IGNORE_DIRS = {
    ".git", ".svn", ".hg", ".idea", ".vscode",
    "node_modules", "dist", "build", "target",
    "__pycache__", "env", "venv", ".env",
    "site-packages", "migrations", ".mypy_cache",
    ".pytest_cache", "htmlcov",
}
MAX_FILE_BYTES = 512 * 1024   # 512 KB — skip giant generated files
MIN_FILE_BYTES = 100           # 100 B  — skip empty/trivial files

# SFT generation thresholds
MIN_FUNC_BODY_LINES = 5        # skip trivial 1-2 line functions
MIN_DOCSTRING_WORDS = 8        # docstring must be substantive enough to use as target
MIN_IMPORT_LINES = 2           # at least 2 imports for import_resolution task
MAX_CONTEXT_CHARS = 6000       # truncate file context shown in prompt


# =====================================================================
# Prompt template pools  (varied prompts → better generalisation)
# =====================================================================
SKELETON_PROMPTS = [
    "Given the following Python code, implement the `{name}` function.\n"
    "The signature and docstring are provided; write the complete body.",

    "Complete the implementation of `{name}` in the code below.\n"
    "Output ONLY the full function (including the `def` line).",

    "The function `{name}` is marked as unimplemented (`...`).\n"
    "Write the complete implementation consistent with its docstring and the surrounding code.",

    "Fill in the body of `{name}`. The function must satisfy its docstring.",
]

PARTIAL_NAV_PROMPTS = [
    "You are working on the following Python file.\n"
    "The function `{name}` is marked as `# TODO: implement`.\n"
    "Implement `{name}` so it is consistent with the rest of the code.\n"
    "Output ONLY the complete function (including the `def` line).",

    "In the codebase below, `{name}` is not yet implemented.\n"
    "Write the full implementation. Match the coding style used in the rest of the file.",

    "Given this partial Python file, implement the missing function `{name}`.\n"
    "The implementation should integrate naturally with the existing helper functions.",
]

DOCSTRING_PROMPTS = [
    "Write a comprehensive Google-style docstring for the following Python function.",
    "Add a detailed docstring to this Python function. Include Args, Returns, and Raises sections where applicable.",
    "Document this function with a proper Python docstring that explains its purpose, parameters, and return value.",
    "Write a NumPy-style docstring for this function including parameters, returns, and an example.",
]

COMPLETION_PROMPTS = [
    "Complete the following Python code. Return ONLY the continuation — no markdown, no explanation.",
    "Continue the Python code below. Output only raw code.",
    "Fill in the rest of this Python implementation:",
]

IMPORT_PROMPTS = [
    "The following Python code is missing its import statements. Add all necessary imports at the top.",
    "Restore the missing `import` and `from ... import ...` lines for this Python module.",
    "This code has no imports. Write the complete import section required for it to run.",
]

MAIN_PROMPTS = [
    "Write a `if __name__ == '__main__':` block that demonstrates how to use the module below.",
    "Add a runnable `__main__` block to this Python module that exercises its main functionality.",
    "Create an example `__main__` section for this module showing typical usage.",
]


# =====================================================================
# Data structures
# =====================================================================
@dataclass
class CodeFile:
    path: str        # relative path inside the repo
    language: str    # file extension without dot
    content: str
    source: str      # e.g. "github:user/repo" or "local:/path"


@dataclass
class FunctionInfo:
    name: str
    qualname: str          # "ClassName.method" or "func"
    full_source: str       # full text of function including decorators
    signature_line: str    # e.g. "def foo(x: int) -> str:"
    docstring: str         # extracted docstring, or ""
    body_lines: int        # number of lines in the body (excl. def + docstring)
    start_line: int        # 1-indexed
    end_line: int          # 1-indexed inclusive
    class_name: str        # set if it is a method
    decorators: str        # e.g. "@staticmethod"


@dataclass
class ParsedFile:
    code_file: CodeFile
    content: str
    lines: List[str]
    imports_section: str   # all import statements extracted
    functions: List[FunctionInfo]
    has_main_block: bool
    main_block_source: str  # content from "if __name__ ==" onward


# =====================================================================
# GitHub Cloning
# =====================================================================
def normalise_github_url(url: str) -> str:
    """Return a canonical https clone URL."""
    url = url.strip().rstrip("/")
    if url.startswith("git@github.com:"):
        url = "https://github.com/" + url[len("git@github.com:"):]
    if not url.endswith(".git"):
        pass  # git clone works without .git suffix
    return url


def repo_slug(url: str) -> str:
    """Extract 'owner__repo' from a GitHub URL for use as a directory name."""
    m = re.search(r"github\.com[/:](.+?)(?:\.git)?$", url)
    if m:
        return m.group(1).replace("/", "__")
    return hashlib.sha1(url.encode()).hexdigest()[:12]


def clone_repo(github_url: str, clone_base: str, token: str = "") -> Optional[str]:
    """
    Clone a GitHub repo (shallow) into {clone_base}/{slug}/.
    Returns the local path on success, None on failure.
    Tries git clone first, then GitHub zip download as fallback.
    """
    url = normalise_github_url(github_url)
    slug = repo_slug(url)
    target = os.path.join(clone_base, slug)

    if os.path.isdir(target):
        print(f"[clone] Already exists, reusing: {target}")
        return target

    os.makedirs(clone_base, exist_ok=True)

    # Inject token for private repos
    if token:
        m = re.match(r"https://github\.com/(.+)", url)
        if m:
            url = f"https://{token}@github.com/{m.group(1)}"

    # Attempt 1: git clone
    print(f"[clone] Cloning {github_url} → {target}")
    result = subprocess.run(
        ["git", "clone", "--depth", "1", "--single-branch", url, target],
        capture_output=True, text=True, timeout=300,
    )
    if result.returncode == 0:
        print(f"[clone] OK — {target}")
        return target

    print(f"[clone] git failed: {result.stderr.strip()[:200]}")

    # Attempt 2: download zip via GitHub API
    clean_url = normalise_github_url(github_url)
    m = re.match(r"https://github\.com/([^/]+)/([^/]+)", clean_url)
    if not m:
        print("[clone] Cannot parse owner/repo from URL.")
        return None

    owner, repo = m.group(1), m.group(2)
    zip_url = f"https://github.com/{owner}/{repo}/archive/refs/heads/main.zip"
    zip_path = target + ".zip"
    headers = ["-H", f"Authorization: token {token}"] if token else []

    print(f"[clone] Fallback: downloading zip from {zip_url}")
    dl = subprocess.run(
        ["curl", "-sL", *headers, "-o", zip_path, zip_url],
        capture_output=True, timeout=300,
    )
    if dl.returncode != 0 or not os.path.exists(zip_path):
        print("[clone] zip download failed.")
        return None

    extract_dir = target + "_extract"
    unzip = subprocess.run(
        ["unzip", "-q", zip_path, "-d", extract_dir],
        capture_output=True, timeout=120,
    )
    if unzip.returncode != 0:
        print("[clone] unzip failed.")
        return None

    # The zip contains a single subdirectory like repo-main/
    subdirs = [d for d in os.listdir(extract_dir)
               if os.path.isdir(os.path.join(extract_dir, d))]
    if subdirs:
        shutil.move(os.path.join(extract_dir, subdirs[0]), target)
    else:
        shutil.move(extract_dir, target)

    for p in [zip_path, extract_dir]:
        if os.path.exists(p):
            shutil.rmtree(p, ignore_errors=True)

    print(f"[clone] zip OK — {target}")
    return target


# =====================================================================
# File Scanning
# =====================================================================
def is_text_readable(path: str) -> bool:
    try:
        with open(path, "r", encoding="utf-8") as f:
            f.read(512)
        return True
    except (UnicodeDecodeError, OSError):
        return False


def scan_directory(
    root_dir: str,
    source_tag: str = "local",
    max_files: int = 0,
) -> List[CodeFile]:
    """Walk root_dir and collect code files."""
    files: List[CodeFile] = []
    for cur_root, dirs, filenames in os.walk(root_dir):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        for fname in filenames:
            fp = os.path.join(cur_root, fname)
            _, ext = os.path.splitext(fname)
            if ext.lower() not in ALLOWED_EXTENSIONS:
                continue
            try:
                size = os.path.getsize(fp)
            except OSError:
                continue
            if not (MIN_FILE_BYTES <= size <= MAX_FILE_BYTES):
                continue
            if not is_text_readable(fp):
                continue
            try:
                content = open(fp, "r", encoding="utf-8", errors="ignore").read()
            except OSError:
                continue
            if not content.strip():
                continue
            rel = os.path.relpath(fp, root_dir)
            files.append(CodeFile(
                path=rel,
                language=ext.lstrip(".").lower(),
                content=content,
                source=source_tag,
            ))
            if max_files and len(files) >= max_files:
                return files
    return files


# =====================================================================
# Python AST Parsing
# =====================================================================
def _source_lines(node: ast.AST, lines: List[str]) -> str:
    start = getattr(node, "lineno", 1) - 1
    end = getattr(node, "end_lineno", start + 1)
    return "".join(lines[start:end])


def _build_signature_line(node: ast.FunctionDef) -> str:
    """Reconstruct the 'def ...:' line (simple, avoids unparsing)."""
    args = ast.unparse(node.args) if hasattr(ast, "unparse") else "..."
    ret = ""
    if node.returns and hasattr(ast, "unparse"):
        ret = f" -> {ast.unparse(node.returns)}"
    return f"def {node.name}({args}){ret}:"


def _get_decorators(node: ast.FunctionDef, lines: List[str]) -> str:
    decs = []
    for d in node.decorator_list:
        decs.append("@" + (ast.unparse(d) if hasattr(ast, "unparse") else _source_lines(d, lines).strip()))
    return "\n".join(decs)


def _count_body_lines(node: ast.FunctionDef) -> int:
    """Lines in the body, not counting def line or docstring-only line."""
    start = node.lineno
    end = getattr(node, "end_lineno", start)
    total = end - start
    # Subtract docstring if present
    if ast.get_docstring(node):
        ds_node = node.body[0]
        ds_end = getattr(ds_node, "end_lineno", ds_node.lineno)
        total -= (ds_end - ds_node.lineno + 1)
    return max(0, total)


def parse_python_file(cf: CodeFile) -> Optional[ParsedFile]:
    """Parse a Python file into structured FunctionInfo + metadata."""
    try:
        tree = ast.parse(cf.content)
    except SyntaxError:
        return None

    lines = cf.content.splitlines(keepends=True)

    # ── Extract import section ───────────────────────────────────────
    import_lines = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            import_lines.append(_source_lines(node, lines).rstrip())
    imports_section = "\n".join(sorted(set(import_lines)))

    # ── Extract __main__ block ───────────────────────────────────────
    has_main = False
    main_source = ""
    for node in tree.body:
        if (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and any(
                isinstance(c, ast.Constant) and c.value == "__main__"
                for c in ast.walk(node.test)
            )
        ):
            has_main = True
            main_source = _source_lines(node, lines)
            break

    # ── Extract functions / methods ──────────────────────────────────
    functions: List[FunctionInfo] = []

    def collect_functions(nodes, class_name=""):
        for node in nodes:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                body_lines_count = _count_body_lines(node)
                if body_lines_count < MIN_FUNC_BODY_LINES:
                    continue
                full_src = _source_lines(node, lines)
                sig = _build_signature_line(node)
                decs = _get_decorators(node, lines)
                docstring = ast.get_docstring(node) or ""
                qualname = f"{class_name}.{node.name}" if class_name else node.name
                functions.append(FunctionInfo(
                    name=node.name,
                    qualname=qualname,
                    full_source=full_src,
                    signature_line=sig,
                    docstring=docstring,
                    body_lines=body_lines_count,
                    start_line=node.lineno,
                    end_line=getattr(node, "end_lineno", node.lineno),
                    class_name=class_name,
                    decorators=decs,
                ))
            elif isinstance(node, ast.ClassDef):
                collect_functions(node.body, class_name=node.name)

    collect_functions(tree.body)

    return ParsedFile(
        code_file=cf,
        content=cf.content,
        lines=lines,
        imports_section=imports_section,
        functions=functions,
        has_main_block=has_main,
        main_block_source=main_source,
    )


# =====================================================================
# SFT Task Generators
# =====================================================================

def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def _pick(pool: List[str], seed: int) -> str:
    return pool[seed % len(pool)]


def _truncate_context(text: str, max_chars: int = MAX_CONTEXT_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    # Keep beginning (imports, class header) + note truncation
    return text[:max_chars] + "\n# ... (truncated) ...\n"


def _stub_function(func: FunctionInfo) -> str:
    """Return the function signature + docstring + '...' as stub."""
    parts = []
    if func.decorators:
        parts.append(func.decorators)
    parts.append(func.signature_line)
    if func.docstring:
        # Re-indent docstring as first line of body
        parts.append('    """' + func.docstring.strip().replace('"""', "'''") + '"""')
    parts.append("    ...")
    return "\n".join(parts)


def _build_partial_file(parsed: ParsedFile, stub_func: FunctionInfo) -> str:
    """Build file content with stub_func replaced by a stub."""
    lines = parsed.lines
    before = "".join(lines[: stub_func.start_line - 1])
    after  = "".join(lines[stub_func.end_line :])
    stub   = _stub_function(stub_func) + "\n"
    return before + stub + after


# ── 1. skeleton_to_impl ──────────────────────────────────────────────
def generate_skeleton_tasks(
    parsed: ParsedFile, rng: random.Random, max_per_file: int
) -> List[Dict]:
    tasks = []
    funcs = [f for f in parsed.functions if f.body_lines >= MIN_FUNC_BODY_LINES]
    rng.shuffle(funcs)

    # Build import + class header context for each function
    src_header = _truncate_context(
        "\n".join(parsed.imports_section.splitlines()[:30])
        + "\n\n"
        + "# ... (rest of file omitted for brevity) ...",
        max_chars=1500,
    )

    for func in funcs[:max_per_file]:
        stub = _stub_function(func)
        prompt_template = _pick(SKELETON_PROMPTS, len(tasks))

        # Provide imports + class context if it is a method
        context_prefix = ""
        if func.class_name:
            # Find class definition lines
            context_prefix = f"# File: {parsed.code_file.path}\n\n"
            context_prefix += parsed.imports_section + "\n\n"
            context_prefix += f"class {func.class_name}:\n    ...\n\n"
        else:
            context_prefix = f"# File: {parsed.code_file.path}\n\n"
            context_prefix += parsed.imports_section + "\n\n"

        user = (
            prompt_template.format(name=func.name)
            + f"\n\n```python\n{_truncate_context(context_prefix + stub)}\n```"
        )
        assistant = f"```python\n{func.full_source.rstrip()}\n```"
        tasks.append({
            "user": user, "assistant": assistant,
            "type": "skeleton_to_impl",
            "source": parsed.code_file.source,
        })

    return tasks


# ── 2. partial_nav ───────────────────────────────────────────────────
def generate_partial_nav_tasks(
    parsed: ParsedFile, rng: random.Random, max_per_file: int
) -> List[Dict]:
    """Show the full file with target function stubbed; ask to implement it."""
    tasks = []
    funcs = [f for f in parsed.functions if f.body_lines >= MIN_FUNC_BODY_LINES]
    rng.shuffle(funcs)

    for func in funcs[:max_per_file]:
        partial = _build_partial_file(parsed, func)
        partial = _truncate_context(partial)

        prompt_template = _pick(PARTIAL_NAV_PROMPTS, len(tasks))
        user = (
            f"File: `{parsed.code_file.path}`\n\n"
            f"```python\n{partial}\n```\n\n"
            + prompt_template.format(name=func.name)
        )
        assistant = f"```python\n{func.full_source.rstrip()}\n```"
        tasks.append({
            "user": user, "assistant": assistant,
            "type": "partial_nav",
            "source": parsed.code_file.source,
        })

    return tasks


# ── 3. code_to_docstring ─────────────────────────────────────────────
def generate_docstring_tasks(
    parsed: ParsedFile, rng: random.Random, max_per_file: int
) -> List[Dict]:
    """Use existing docstrings as gold targets; strip them and ask to regenerate."""
    tasks = []
    funcs = [
        f for f in parsed.functions
        if f.docstring and len(f.docstring.split()) >= MIN_DOCSTRING_WORDS
    ]
    rng.shuffle(funcs)

    for func in funcs[:max_per_file]:
        # Build version of the function WITHOUT the docstring
        src_lines = func.full_source.splitlines(keepends=True)
        # Remove lines that are part of the docstring (first string literal in body)
        no_doc_lines = []
        in_doc = False
        doc_removed = False
        for i, line in enumerate(src_lines):
            stripped = line.strip()
            if not doc_removed and i > 0 and stripped.startswith(('"""', "'''")):
                in_doc = True
            if in_doc:
                if stripped.endswith(('"""', "'''")) and len(stripped) > 3:
                    in_doc = False
                    doc_removed = True
                elif stripped in ('"""', "'''") and i > 0:
                    in_doc = False
                    doc_removed = True
                continue
            no_doc_lines.append(line)

        code_without_doc = "".join(no_doc_lines).rstrip()
        if not code_without_doc:
            continue

        prompt_template = _pick(DOCSTRING_PROMPTS, len(tasks))
        user = prompt_template + f"\n\n```python\n{code_without_doc}\n```"
        assistant = f'"""\n{func.docstring.strip()}\n"""'
        tasks.append({
            "user": user, "assistant": assistant,
            "type": "code_to_docstring",
            "source": parsed.code_file.source,
        })

    return tasks


# ── 4. completion (prefix/suffix) ────────────────────────────────────
def generate_completion_tasks(
    parsed: ParsedFile, rng: random.Random, max_per_file: int
) -> List[Dict]:
    """Split each function at a random interior point → completion task."""
    tasks = []
    funcs = [f for f in parsed.functions if f.body_lines >= MIN_FUNC_BODY_LINES * 2]
    rng.shuffle(funcs)

    for func in funcs[:max_per_file]:
        src_lines = func.full_source.splitlines(keepends=True)
        n = len(src_lines)
        if n < 6:
            continue
        # Choose split point in the middle 40-80% of the function
        lo = max(2, int(n * 0.3))
        hi = max(lo + 1, int(n * 0.8))
        cut = rng.randint(lo, hi - 1)
        prefix = "".join(src_lines[:cut])
        suffix = "".join(src_lines[cut:])
        if len(suffix.strip()) < 40:
            continue

        prompt_template = _pick(COMPLETION_PROMPTS, len(tasks))
        user = prompt_template + f"\n\n```python\n{prefix}```"
        assistant = f"```python\n{suffix.rstrip()}\n```"
        tasks.append({
            "user": user, "assistant": assistant,
            "type": "completion",
            "source": parsed.code_file.source,
        })

    return tasks


# ── 5. import_resolution ─────────────────────────────────────────────
def generate_import_tasks(parsed: ParsedFile) -> List[Dict]:
    """Strip import section → ask model to restore it."""
    if not parsed.imports_section:
        return []
    import_lines = [l for l in parsed.imports_section.splitlines() if l.strip()]
    if len(import_lines) < MIN_IMPORT_LINES:
        return []

    # Build code without import lines
    no_import_lines = []
    import_set = set(l.strip() for l in import_lines)
    for line in parsed.content.splitlines():
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            if stripped in import_set:
                continue
        no_import_lines.append(line)
    code_no_imports = _truncate_context("\n".join(no_import_lines))

    prompt_template = _pick(IMPORT_PROMPTS, 0)
    user = prompt_template + f"\n\n```python\n{code_no_imports}\n```"
    assistant = "```python\n" + parsed.imports_section.strip() + "\n```"
    return [{
        "user": user, "assistant": assistant,
        "type": "import_resolution",
        "source": parsed.code_file.source,
    }]


# ── 6. generate_main ─────────────────────────────────────────────────
def generate_main_tasks(parsed: ParsedFile) -> List[Dict]:
    """If file has __main__ block, use the rest of the file as context."""
    if not parsed.has_main_block or not parsed.main_block_source.strip():
        return []

    # Show file WITHOUT the __main__ block
    main_start_line = None
    for i, line in enumerate(parsed.content.splitlines()):
        if 'if __name__' in line and '__main__' in line:
            main_start_line = i
            break
    if main_start_line is None:
        return []

    module_body = _truncate_context(
        "\n".join(parsed.content.splitlines()[:main_start_line])
    )
    prompt_template = _pick(MAIN_PROMPTS, 0)
    user = (
        f"File: `{parsed.code_file.path}`\n\n"
        f"```python\n{module_body}\n```\n\n"
        + prompt_template
    )
    assistant = "```python\n" + parsed.main_block_source.strip() + "\n```"
    return [{
        "user": user, "assistant": assistant,
        "type": "generate_main",
        "source": parsed.code_file.source,
    }]


# ── 7. test_pair  ────────────────────────────────────────────────────
def find_test_file_pairs(
    parsed_files: List[ParsedFile],
) -> List[Tuple[ParsedFile, ParsedFile]]:
    """Find (source, test) pairs: utils.py ↔ test_utils.py or tests/test_utils.py."""
    by_path: Dict[str, ParsedFile] = {p.code_file.path: p for p in parsed_files}
    pairs: List[Tuple[ParsedFile, ParsedFile]] = []
    for pf in parsed_files:
        name = Path(pf.code_file.path).stem
        # Possible test file locations
        candidates = [
            f"test_{name}.py",
            f"{name}_test.py",
            f"tests/test_{name}.py",
            f"tests/{name}_test.py",
            os.path.join(os.path.dirname(pf.code_file.path), f"test_{name}.py"),
            os.path.join(os.path.dirname(pf.code_file.path), f"tests", f"test_{name}.py"),
        ]
        for cand in candidates:
            # Normalise
            cand = os.path.normpath(cand)
            if cand in by_path:
                pairs.append((pf, by_path[cand]))
                break
    return pairs


def generate_test_pair_task(src: ParsedFile, test: ParsedFile) -> Optional[Dict]:
    """Source module → matching test file as target."""
    module_ctx = _truncate_context(src.content)
    test_content = test.content.strip()
    if len(test_content) < 100:
        return None

    user = (
        f"File: `{src.code_file.path}`\n\n"
        f"```python\n{module_ctx}\n```\n\n"
        "Write a comprehensive pytest test file for this module.\n"
        "Include imports, test class or functions, and meaningful assertions."
    )
    assistant = f"```python\n{test_content}\n```"
    return {
        "user": user, "assistant": assistant,
        "type": "test_pair",
        "source": src.code_file.source,
    }


# =====================================================================
# CPT Formatter
# =====================================================================
def format_for_cpt(cf: CodeFile) -> List[Dict]:
    """
    Wrap a code file as a CPT text sample with structured metadata header.
    For Python, also add extracted function stubs as a brief directory.
    """
    header = f"# File: {cf.path}\n# Language: {cf.language}\n\n"
    text = header + cf.content.rstrip() + "\n"
    return [{"text": text, "source": cf.source, "language": cf.language}]


# =====================================================================
# Deduplication
# =====================================================================
def dedup_items(items: List[Dict], key: str = "user") -> List[Dict]:
    seen: set = set()
    out: List[Dict] = []
    for item in items:
        h = _sha1(item.get(key, "") + item.get("assistant", ""))
        if h not in seen:
            seen.add(h)
            out.append(item)
    return out


# =====================================================================
# Writing Output
# =====================================================================
def write_jsonl(items: List[Dict], path: str) -> int:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return len(items)


def write_dataset(
    sft_items: List[Dict],
    cpt_items: List[Dict],
    output_dir: str,
) -> Dict[str, int]:
    os.makedirs(output_dir, exist_ok=True)
    stats: Dict[str, int] = {}

    # Per-task SFT files
    by_type: Dict[str, List[Dict]] = {}
    for item in sft_items:
        t = item.get("type", "unknown")
        by_type.setdefault(t, []).append(item)

    for task_type, items in by_type.items():
        path = os.path.join(output_dir, f"sft_{task_type}.jsonl")
        n = write_jsonl(items, path)
        stats[f"sft_{task_type}"] = n
        print(f"  [write] {task_type:22s} {n:6d} samples → {path}")

    # Merged SFT
    all_sft_path = os.path.join(output_dir, "sft_all.jsonl")
    rng = random.Random(42)
    shuffled = list(sft_items)
    rng.shuffle(shuffled)
    n_sft = write_jsonl(shuffled, all_sft_path)
    stats["sft_all"] = n_sft
    print(f"  [write] {'sft_all':22s} {n_sft:6d} samples → {all_sft_path}")

    # CPT
    cpt_path = os.path.join(output_dir, "cpt_data.jsonl")
    n_cpt = write_jsonl(cpt_items, cpt_path)
    stats["cpt_data"] = n_cpt
    print(f"  [write] {'cpt_data':22s} {n_cpt:6d} samples → {cpt_path}")

    return stats


# =====================================================================
# CLI
# =====================================================================
ALL_TASKS = [
    "skeleton_to_impl", "partial_nav", "code_to_docstring",
    "completion", "import_resolution", "generate_main", "test_pair",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-source Code Dataset Builder v3 (CPT + SFT)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Input sources ─────────────────────────────────────────────────
    parser.add_argument(
        "--local_dirs", nargs="+", default=[],
        help="One or more local repository/directory paths to scan.",
    )
    parser.add_argument(
        "--github_repos", nargs="+", default=[],
        help="GitHub repo URLs (https://github.com/owner/repo). "
             "Each will be shallow-cloned into --clone_dir.",
    )
    parser.add_argument(
        "--repos_file", type=str, default="",
        help="Path to a text file with one GitHub URL or local path per line.",
    )
    parser.add_argument(
        "--github_token", type=str, default="",
        help="GitHub personal access token for private repos.",
    )
    parser.add_argument(
        "--clone_dir", type=str, default="data/cloned_repos",
        help="Base directory where GitHub repos are cloned.",
    )

    # ── Output ────────────────────────────────────────────────────────
    parser.add_argument("--output_dir", type=str, default="data/private_v3")

    # ── Generation controls ───────────────────────────────────────────
    parser.add_argument(
        "--sft_tasks", nargs="+", default=ALL_TASKS,
        choices=ALL_TASKS, metavar="TASK",
        help="Which SFT task types to generate. Default: all.",
    )
    parser.add_argument(
        "--max_files_per_repo", type=int, default=0,
        help="Max files to ingest per repo (0 = unlimited).",
    )
    parser.add_argument(
        "--max_sft_per_file", type=int, default=4,
        help="Max SFT items generated per source file per task type.",
    )
    parser.add_argument(
        "--cpt_only", action="store_true",
        help="Skip SFT generation; only produce CPT data.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# =====================================================================
# Main
# =====================================================================
def main():
    args = parse_args()
    rng = random.Random(args.seed)

    # ── Resolve source list ──────────────────────────────────────────
    local_dirs = list(args.local_dirs)
    github_urls: List[str] = list(args.github_repos)

    if args.repos_file and os.path.exists(args.repos_file):
        with open(args.repos_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("https://") or line.startswith("git@"):
                    github_urls.append(line)
                elif os.path.isdir(line):
                    local_dirs.append(line)

    if not local_dirs and not github_urls:
        print("[ERROR] Provide at least one --local_dirs or --github_repos.")
        sys.exit(1)

    # ── Clone GitHub repos ───────────────────────────────────────────
    for url in github_urls:
        cloned = clone_repo(url, args.clone_dir, token=args.github_token)
        if cloned:
            local_dirs.append(cloned)

    # ── Scan files ───────────────────────────────────────────────────
    print("\n[*] Scanning source directories...")
    all_files: List[CodeFile] = []
    for d in local_dirs:
        tag = f"local:{d}"
        # Check if it's a cloned GitHub repo
        for url in github_urls:
            if repo_slug(url) in d:
                tag = f"github:{url}"
                break
        files = scan_directory(d, source_tag=tag, max_files=args.max_files_per_repo)
        print(f"    {d}: {len(files)} files")
        all_files.extend(files)

    print(f"[+] Total files collected: {len(all_files)}")
    if not all_files:
        print("[ERROR] No files found. Check paths and extensions.")
        sys.exit(1)

    # ── Generate CPT data ────────────────────────────────────────────
    print("\n[*] Generating CPT data...")
    cpt_items: List[Dict] = []
    for cf in tqdm(all_files, desc="CPT"):
        cpt_items.extend(format_for_cpt(cf))
    cpt_items = dedup_items(cpt_items, key="text")
    print(f"[+] CPT items: {len(cpt_items)}")

    # ── Parse Python files + generate SFT ────────────────────────────
    sft_items: List[Dict] = []

    if not args.cpt_only:
        print("\n[*] Parsing Python files and generating SFT tasks...")
        py_files = [cf for cf in all_files if cf.language == "py"]
        parsed_files: List[ParsedFile] = []

        for cf in tqdm(py_files, desc="Parse"):
            pf = parse_python_file(cf)
            if pf and pf.functions:
                parsed_files.append(pf)

        print(f"[+] Parsed files with functions: {len(parsed_files)}")

        # Find test file pairs before looping
        test_pairs = find_test_file_pairs(parsed_files)
        test_pair_set = {id(p[0]) for p in test_pairs}
        print(f"[+] Test file pairs found: {len(test_pairs)}")

        per_file_rng = random.Random(args.seed)
        for pf in tqdm(parsed_files, desc="SFT gen"):
            local_rng = random.Random(per_file_rng.randint(0, 2**31))

            if "skeleton_to_impl" in args.sft_tasks:
                sft_items.extend(generate_skeleton_tasks(pf, local_rng, args.max_sft_per_file))
            if "partial_nav" in args.sft_tasks:
                sft_items.extend(generate_partial_nav_tasks(pf, local_rng, max(1, args.max_sft_per_file // 2)))
            if "code_to_docstring" in args.sft_tasks:
                sft_items.extend(generate_docstring_tasks(pf, local_rng, args.max_sft_per_file))
            if "completion" in args.sft_tasks:
                sft_items.extend(generate_completion_tasks(pf, local_rng, args.max_sft_per_file))
            if "import_resolution" in args.sft_tasks:
                sft_items.extend(generate_import_tasks(pf))
            if "generate_main" in args.sft_tasks:
                sft_items.extend(generate_main_tasks(pf))

        if "test_pair" in args.sft_tasks:
            for src_pf, test_pf in tqdm(test_pairs, desc="test_pair"):
                item = generate_test_pair_task(src_pf, test_pf)
                if item:
                    sft_items.append(item)

        sft_items = dedup_items(sft_items)
        print(f"[+] SFT items (after dedup): {len(sft_items)}")

        # Print per-type counts
        by_type: Dict[str, int] = {}
        for item in sft_items:
            t = item.get("type", "unknown")
            by_type[t] = by_type.get(t, 0) + 1
        print("[+] Per-task breakdown:")
        for task, cnt in sorted(by_type.items(), key=lambda x: -x[1]):
            print(f"    {task:22s} {cnt:6d}")

    # ── Write output ─────────────────────────────────────────────────
    print(f"\n[*] Writing output to: {args.output_dir}")
    stats = write_dataset(sft_items, cpt_items, args.output_dir)

    # Manifest
    manifest = {
        "sources": {
            "local_dirs": local_dirs,
            "github_repos": github_urls,
        },
        "total_files": len(all_files),
        "py_files_parsed": len(parsed_files) if not args.cpt_only else 0,
        "sft_tasks_enabled": args.sft_tasks,
        "counts": stats,
    }
    manifest_path = os.path.join(args.output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[+] Done.  Manifest → {manifest_path}")


if __name__ == "__main__":
    main()

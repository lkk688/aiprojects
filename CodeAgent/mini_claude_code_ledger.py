#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
mini_claude_code_ledger.py
A minimal, human-in-the-loop "Claude Code"-like CLI with:
- Prompt ledger (editable prompt.md each turn)
- Model I/O logging (response.md)
- Patch extraction (patch.diff) + guarded apply (git apply --check)
- Run logs (run.jsonl)
- SkillDB (JSONL): successes/failures with tags, patterns, evidence
- Skill injection: auto-select Top-K relevant skills to include in prompt

Requirements:
  pip install openai rich

Env:
  VLLM_BASE_URL (default http://127.0.0.1:8000/v1)
  VLLM_API_KEY  (default EMPTY)
  VLLM_MODEL    (default Qwen/Qwen3-Coder-Next-FP8)
"""

import os
import re
import json
import time
import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.panel import Panel
from openai import OpenAI

console = Console()


# ---------------------------
# Config
# ---------------------------

AGENT_DIR = Path(".agent")
SESSIONS_DIR = AGENT_DIR / "sessions"
SKILL_DIR = AGENT_DIR / "skilldb"
SKILL_SUCCESS = SKILL_DIR / "successes.jsonl"
SKILL_FAIL = SKILL_DIR / "failures.jsonl"
RUNS_LOG = AGENT_DIR / "runs.jsonl"

DEFAULT_SYSTEM = """You are an editing agent.

HARD OUTPUT CONTRACT:
- Output MUST be a single unified diff patch that can be applied with `git apply`.
- Include diff headers: diff --git, --- a/..., +++ b/...
- Do NOT include any explanations outside the patch.
- If you need more info, ask for EXACT file path(s) and minimal snippet or command output.

SCOPE & SAFETY:
- Only modify files in the explicit ALLOWLIST (provided in the prompt).
- Make the smallest change that solves the issue. No refactors, renames, or formatting changes unless required.
- Stay within change budget: max 2 files, max 120 changed lines.
"""

# Skill injection limits (keep short to save tokens)
SKILL_INJECT_TOPK = 6
SKILL_INJECT_MAX_LINES = 40  # total lines injected into prompt


# ---------------------------
# Utilities
# ---------------------------

def now_stamp() -> str:
    return time.strftime("%Y-%m-%d_%H%M%S")

def sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

def ensure_dirs():
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    SKILL_DIR.mkdir(parents=True, exist_ok=True)
    AGENT_DIR.mkdir(parents=True, exist_ok=True)
    for p in [SKILL_SUCCESS, SKILL_FAIL, RUNS_LOG]:
        if not p.exists():
            p.write_text("", encoding="utf-8")

def run_shell(cmd: str, cwd: Optional[str] = None, cap: int = 20000) -> Tuple[int, str]:
    p = subprocess.run(cmd, shell=True, text=True, capture_output=True, cwd=cwd)
    out = (p.stdout or "") + (p.stderr or "")
    if len(out) > cap:
        out = out[-cap:]
    return p.returncode, out

def is_git_repo() -> bool:
    code, _ = run_shell("git rev-parse --is-inside-work-tree")
    return code == 0

def git_status() -> str:
    code, out = run_shell("git status -sb")
    return out if code == 0 else ""

def git_diff() -> str:
    code, out = run_shell("git diff")
    return out if code == 0 else ""

def read_file(path: str, max_chars: int = 16000) -> str:
    p = Path(path)
    if not p.exists():
        return f"[MISSING FILE] {path}"
    data = p.read_text(encoding="utf-8", errors="ignore")
    if len(data) > max_chars:
        return data[:max_chars] + "\n\n[TRUNCATED]\n"
    return data

def top_level_tree(max_items: int = 200) -> str:
    items = []
    for p in Path(".").iterdir():
        if p.name.startswith(".agent"):
            continue
        items.append(p.name + ("/" if p.is_dir() else ""))
    items = sorted(items)[:max_items]
    return "\n".join(items)

def extract_unified_diff(text: str) -> Optional[str]:
    """
    Try to extract a single unified diff from model output.
    Accepts either:
      - raw diff starting with 'diff --git'
      - diff inside fenced ``` blocks
    Returns diff text or None.
    """
    t = text.strip()

    # If starts with diff --git, take from there
    m = re.search(r"(?ms)^(diff --git .*?$.*)", t)
    if m:
        return m.group(1).strip() + "\n"

    # Try fenced code blocks
    fence = re.search(r"(?ms)```(?:diff)?\s*(diff --git .*?)```", t)
    if fence:
        return fence.group(1).strip() + "\n"

    return None

def parse_diff_paths(diff_text: str) -> List[str]:
    paths = []
    for line in diff_text.splitlines():
        if line.startswith("diff --git "):
            # diff --git a/path b/path
            parts = line.split()
            if len(parts) >= 4:
                a = parts[2]
                b = parts[3]
                # strip a/ b/
                a = a[2:] if a.startswith("a/") else a
                b = b[2:] if b.startswith("b/") else b
                # prefer b (new path)
                paths.append(b)
    return sorted(set(paths))

def count_changed_lines(diff_text: str) -> int:
    # rough: count + and - lines excluding diff headers
    n = 0
    for line in diff_text.splitlines():
        if line.startswith(("+++ ", "--- ", "diff --git", "@@")):
            continue
        if line.startswith("+") and not line.startswith("++"):
            n += 1
        elif line.startswith("-") and not line.startswith("--"):
            n += 1
    return n

def write_jsonl(path: Path, obj: Dict[str, Any]):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# ---------------------------
# SkillDB
# ---------------------------

@dataclass
class Skill:
    tag: str
    kind: str  # "success" or "failure"
    text: str  # short guidance
    pattern: str  # keyword/pattern
    evidence: str  # session/turn summary
    created_at: str

def load_skills() -> List[Skill]:
    skills: List[Skill] = []
    for kind, path in [("success", SKILL_SUCCESS), ("failure", SKILL_FAIL)]:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                skills.append(Skill(
                    tag=obj.get("tag", ""),
                    kind=kind,
                    text=obj.get("text", ""),
                    pattern=obj.get("pattern", ""),
                    evidence=obj.get("evidence", ""),
                    created_at=obj.get("created_at", ""),
                ))
            except Exception:
                continue
    return skills

def score_skill(skill: Skill, query: str) -> int:
    """
    Simple lexical scoring:
    - +2 if pattern token appears in query
    - +1 for each word from skill.text appearing in query (cap)
    """
    q = query.lower()
    s = 0
    patt = (skill.pattern or "").lower().strip()
    if patt and patt in q:
        s += 2
    # token overlap
    words = re.findall(r"[a-zA-Z0-9_]{3,}", (skill.text or "").lower())
    hits = 0
    for w in set(words):
        if w in q:
            hits += 1
    s += min(hits, 3)
    return s

def select_relevant_skills(goal_and_notes: str, topk: int = SKILL_INJECT_TOPK) -> List[Skill]:
    skills = load_skills()
    scored = [(score_skill(sk, goal_and_notes), sk) for sk in skills]
    scored.sort(key=lambda x: x[0], reverse=True)
    picked = [sk for sc, sk in scored if sc > 0][:topk]
    return picked

def format_skill_injection(skills: List[Skill]) -> str:
    if not skills:
        return ""
    lines = ["## History-based guardrails (SkillDB)"]
    for sk in skills:
        prefix = "✅" if sk.kind == "success" else "⛔"
        # keep short
        text = sk.text.strip().replace("\n", " ")
        evidence = sk.evidence.strip().replace("\n", " ")
        lines.append(f"- {prefix} [{sk.tag}] {text} (evidence: {evidence})")
    # cap lines
    if len(lines) > SKILL_INJECT_MAX_LINES:
        lines = lines[:SKILL_INJECT_MAX_LINES] + ["- (truncated)"]
    return "\n".join(lines).strip() + "\n"


# ---------------------------
# Prompt Ledger
# ---------------------------

def build_prompt_md(
    goal: str,
    allowlist: List[str],
    context_files: List[str],
    extra_notes: str,
    inject_skills: str,
) -> str:
    ctx_parts = []
    ctx_parts.append("## Repo snapshot")
    ctx_parts.append("### Top-level tree\n" + top_level_tree())

    if is_git_repo():
        ctx_parts.append("### git status\n```\n" + git_status() + "\n```")
        d = git_diff()
        if d.strip():
            ctx_parts.append("### git diff\n```diff\n" + d + "\n```")

    for f in context_files:
        ctx_parts.append(f"## File: {f}\n```python\n{read_file(f)}\n```")

    allow_txt = "\n".join(f"- {p}" for p in allowlist) if allowlist else "- (none)"

    md = f"""# Turn Prompt

## Goal
{goal}

## ALLOWLIST (only these files may be modified)
{allow_txt}

{inject_skills if inject_skills else ""}

## Notes / Constraints
{extra_notes.strip() if extra_notes.strip() else "(none)"}

## Context
{("\n\n".join(ctx_parts)).strip()}

## Output Contract (repeat)
Return ONLY a unified diff patch that applies with `git apply`.
"""
    return md.strip() + "\n"


# ---------------------------
# Core Loop
# ---------------------------

def apply_patch_guarded(diff_text: str, turn_dir: Path) -> bool:
    patch_path = turn_dir / "patch.diff"
    patch_path.write_text(diff_text, encoding="utf-8")

    check_code, check_out = run_shell(f"git apply --check {patch_path.as_posix()}")
    (turn_dir / "apply.log").write_text(
        f"[git apply --check]\nexit={check_code}\n\n{check_out}\n", encoding="utf-8"
    )
    if check_code != 0:
        console.print(Panel(check_out or "(no output)", title="Patch check failed", style="red"))
        return False

    if not Confirm.ask("Patch check passed. Apply patch with git apply?"):
        return False

    app_code, app_out = run_shell(f"git apply {patch_path.as_posix()}")
    with (turn_dir / "apply.log").open("a", encoding="utf-8") as f:
        f.write(f"\n[git apply]\nexit={app_code}\n\n{app_out}\n")
    if app_code != 0:
        console.print(Panel(app_out or "(no output)", title="git apply failed", style="red"))
        return False

    console.print("[green]Patch applied.[/green]")
    return True

def quick_gates(turn_dir: Path) -> Dict[str, Any]:
    """
    Lightweight always-on gates: compileall, optional pytest if present.
    """
    gates = []
    # compileall
    code, out = run_shell("python -m compileall -q .")
    gates.append({"name": "compileall", "exit": code, "output_tail": out[-4000:]})
    # pytest if tests exist
    if Path("pytest.ini").exists() or Path("pyproject.toml").exists() or Path("tests").exists():
        # don't assume it's fast; user can disable by declining later if desired
        code2, out2 = run_shell("pytest -q", cap=40000)
        gates.append({"name": "pytest", "exit": code2, "output_tail": out2[-8000:]})
    (turn_dir / "gates.json").write_text(json.dumps(gates, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"gates": gates}

def run_and_log_command(turn_dir: Path) -> Optional[Dict[str, Any]]:
    if not Confirm.ask("Run a command and feed output back to model?"):
        return None
    cmd = Prompt.ask("Command to run")
    code, out = run_shell(cmd, cap=120000)
    rec = {
        "ts": now_stamp(),
        "cmd": cmd,
        "exit": code,
        "output_tail": out[-20000:],
    }
    # write per-turn run log
    runlog_path = turn_dir / "run.jsonl"
    write_jsonl(runlog_path, rec)
    console.print(Panel(rec["output_tail"] or "(no output)", title=f"Command output (exit={code})"))
    return rec

def skill_entry_from_turn(
    kind: str,
    tag: str,
    pattern: str,
    text: str,
    evidence: str,
) -> Dict[str, Any]:
    return {
        "created_at": now_stamp(),
        "tag": tag.strip(),
        "pattern": pattern.strip(),
        "text": text.strip(),
        "evidence": evidence.strip(),
    }

def main():
    ensure_dirs()

    base_url = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
    api_key = os.environ.get("VLLM_API_KEY", "EMPTY")
    model = os.environ.get("VLLM_MODEL", "Qwen/Qwen3-Coder-Next-FP8")

    client = OpenAI(base_url=base_url, api_key=api_key)

    session_id = now_stamp()
    session_dir = SESSIONS_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    console.print(Panel(
        f"Session: {session_id}\nbase_url={base_url}\nmodel={model}\nlogs: {session_dir}",
        title="mini-claude-code (Ledger + SkillDB)",
        style="cyan"
    ))

    # Gather allowlist and context files
    allowlist: List[str] = []
    console.print("\n[bold]ALLOWLIST[/bold] (only these files may be modified)")
    while True:
        p = Prompt.ask("Add allowlisted file path (empty to stop)", default="").strip()
        if not p:
            break
        allowlist.append(p)

    context_files = list(dict.fromkeys(allowlist))  # start with allowlist
    console.print("\n[bold]Extra context files[/bold] (read-only context)")
    while True:
        p = Prompt.ask("Add context file path (empty to stop)", default="").strip()
        if not p:
            break
        if p not in context_files:
            context_files.append(p)

    goal = Prompt.ask("\nGoal / task description")
    extra_notes = Prompt.ask("Constraints / notes (optional)", default="").strip()

    # Conversation messages (we keep system + last user prompt; store everything on disk anyway)
    messages: List[Dict[str, str]] = [{"role": "system", "content": DEFAULT_SYSTEM}]

    turn = 1
    while True:
        turn_dir = session_dir / f"{turn:04d}"
        turn_dir.mkdir(parents=True, exist_ok=True)

        # Inject relevant skills based on goal + notes + last run errors (we'll add later too)
        query_for_skills = goal + "\n" + extra_notes
        inject = format_skill_injection(select_relevant_skills(query_for_skills))

        # Build prompt.md, let user edit it before sending
        prompt_md = build_prompt_md(goal, allowlist, context_files, extra_notes, inject)
        prompt_path = turn_dir / "prompt.md"
        prompt_path.write_text(prompt_md, encoding="utf-8")

        console.print(Panel(
            f"Wrote prompt: {prompt_path}\nEdit it now if you want to trim tokens.\n"
            f"Tip: remove irrelevant file sections / diff blocks.",
            title=f"Turn {turn:04d}",
            style="yellow"
        ))

        if not Confirm.ask("Open/edit prompt.md now? (you do it in your editor)"):
            pass
        else:
            console.print("Edit the file, save, then come back here.")

        if not Confirm.ask("Send the current prompt.md to the model?"):
            console.print("[yellow]Turn canceled by user.[/yellow]")
            break

        prompt_final = prompt_path.read_text(encoding="utf-8", errors="ignore")
        prompt_hash = sha1_text(prompt_final)

        # Add user message for this turn
        messages_turn = messages + [{"role": "user", "content": prompt_final}]

        console.print("[cyan]Calling model...[/cyan]")
        resp = client.chat.completions.create(
            model=model,
            messages=messages_turn,
            temperature=0.2
        )
        content = resp.choices[0].message.content or ""
        (turn_dir / "response.md").write_text(content, encoding="utf-8")

        console.rule(f"MODEL OUTPUT (turn {turn:04d})")
        console.print(content)

        diff = extract_unified_diff(content)
        meta: Dict[str, Any] = {
            "session": session_id,
            "turn": turn,
            "goal": goal,
            "prompt_path": str(prompt_path),
            "response_path": str(turn_dir / "response.md"),
            "prompt_sha1": prompt_hash,
            "model": model,
            "base_url": base_url,
            "allowlist": allowlist,
            "context_files": context_files,
            "ts": now_stamp(),
        }

        patch_applied = False
        patch_ok = False
        patch_paths: List[str] = []
        changed_lines = 0

        if diff:
            patch_paths = parse_diff_paths(diff)
            changed_lines = count_changed_lines(diff)

            (turn_dir / "patch.diff").write_text(diff, encoding="utf-8")
            console.print(Panel(
                f"Files in patch: {patch_paths}\nChanged lines (rough): {changed_lines}",
                title="Patch summary",
                style="blue"
            ))

            # Enforce allowlist
            outside = [p for p in patch_paths if p not in allowlist]
            if outside:
                msg = "Patch modifies files outside ALLOWLIST:\n" + "\n".join(outside)
                (turn_dir / "apply.log").write_text(msg + "\n", encoding="utf-8")
                console.print(Panel(msg, title="ALLOWLIST violation", style="red"))
                patch_ok = False
            else:
                patch_ok = True

            # Enforce budget (soft guard; you can still override by editing patch.diff)
            if patch_ok and (len(patch_paths) > 2 or changed_lines > 120):
                warn = f"Change budget exceeded (files={len(patch_paths)}, lines~={changed_lines})."
                console.print(Panel(warn, title="Budget warning", style="yellow"))
                if Confirm.ask("Edit patch.diff to reduce scope before applying?"):
                    console.print(f"Edit: {turn_dir / 'patch.diff'} then continue.")
                    if not Confirm.ask("Continue to apply after editing patch.diff?"):
                        patch_ok = False
                    else:
                        diff = (turn_dir / "patch.diff").read_text(encoding="utf-8", errors="ignore")
                        patch_paths = parse_diff_paths(diff)
                        outside = [p for p in patch_paths if p not in allowlist]
                        if outside:
                            console.print(Panel("Still outside allowlist:\n" + "\n".join(outside), style="red"))
                            patch_ok = False
                        else:
                            patch_ok = True

            if patch_ok:
                if not is_git_repo():
                    console.print(Panel("Not a git repo: cannot apply patch via git apply.", style="red"))
                else:
                    patch_applied = apply_patch_guarded(diff, turn_dir)
        else:
            console.print(Panel(
                "No valid unified diff found in model output.\n"
                "Tip: model must output ONLY a patch starting with 'diff --git'.",
                title="No patch",
                style="red"
            ))

        # Always-on quick gates (optional but recommended)
        gates_info = {}
        if patch_applied and Confirm.ask("Run quick gates (compileall / pytest if present)?"):
            gates_info = quick_gates(turn_dir)
            # show gate summary
            gate_lines = []
            for g in gates_info["gates"]:
                gate_lines.append(f"{g['name']}: exit={g['exit']}")
            console.print(Panel("\n".join(gate_lines), title="Gate summary"))

        # Run command (human-confirmed)
        run_rec = None
        if patch_applied:
            run_rec = run_and_log_command(turn_dir)

        # Turn verdict and skill logging
        verdict = Prompt.ask("Verdict for this turn", choices=["success", "fail", "partial"], default="partial")
        note = Prompt.ask("Short note (what worked / what failed)", default="").strip()

        # Optionally write a skill entry
        if Confirm.ask("Write a SkillDB entry from this turn?"):
            kind = Prompt.ask("Skill kind", choices=["success", "failure"], default="success" if verdict=="success" else "failure")
            tag = Prompt.ask("Skill tag (short, unique-ish)")
            pattern = Prompt.ask("Pattern / keywords to match later (e.g., 'NaN loss', 'apply failed')", default=goal.split()[0] if goal else "")
            text = Prompt.ask("Skill text (what to do / avoid)", default=note or "")
            evidence = f"session={session_id} turn={turn:04d} verdict={verdict}"
            entry = skill_entry_from_turn(kind, tag, pattern, text, evidence)
            if kind == "success":
                write_jsonl(SKILL_SUCCESS, entry)
            else:
                write_jsonl(SKILL_FAIL, entry)
            console.print("[green]SkillDB updated.[/green]")

        # Write global runs.jsonl entry
        run_obj = {
            **meta,
            "patch_found": bool(diff),
            "patch_files": patch_paths,
            "changed_lines_est": changed_lines,
            "patch_ok": patch_ok,
            "patch_applied": patch_applied,
            "gates": gates_info.get("gates", []),
            "run": run_rec or {},
            "verdict": verdict,
            "note": note,
        }
        write_jsonl(RUNS_LOG, run_obj)

        # Feed model with next step info (keep conversation light; you already have on disk)
        # We'll include only: whether patch applied, plus run output tail if any.
        feedback = []
        if not diff:
            feedback.append("Your previous output did not contain a valid unified diff. Output ONLY a patch.")
        elif diff and not patch_ok:
            feedback.append("Your patch violated ALLOWLIST or budget. Output a patch modifying ONLY allowed files.")
        elif diff and patch_ok and not patch_applied:
            applylog = (turn_dir / "apply.log").read_text(encoding="utf-8", errors="ignore")
            feedback.append("Patch did not apply. Error:\n" + applylog[-4000:])
        else:
            feedback.append("Patch applied.")
            if gates_info.get("gates"):
                feedback.append("Quick gates results:\n" + "\n".join([f"{g['name']}: exit={g['exit']}" for g in gates_info["gates"]]))
                # include tail for failures
                for g in gates_info["gates"]:
                    if g["exit"] != 0:
                        feedback.append(f"{g['name']} output tail:\n{g.get('output_tail','')}")
            if run_rec:
                feedback.append(f"Command: {run_rec['cmd']}\nExit: {run_rec['exit']}\nOutput tail:\n{run_rec['output_tail']}")

        # Update rolling "notes" for skill relevance in next turn
        extra_notes = (extra_notes + "\n\n" + "\n\n".join(feedback)).strip()

        if not Confirm.ask("Continue to next turn?"):
            break

        turn += 1

    console.print(Panel(
        f"Session complete.\nLogs: {session_dir}\nSkillDB: {SKILL_DIR}\nGlobal runs: {RUNS_LOG}",
        title="Done",
        style="green"
    ))


if __name__ == "__main__":
    main()
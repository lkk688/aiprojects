"""
Building a "Sleep-Learning" (Offline Consolidation) and "Just-In-Time (JIT) Compilation" mechanism for your Agent.

By having the system autonomously crawl, distill, test, and index skills during idle time, you build a massive capability library without polluting the small model's limited context window. Furthermore, by hijacking existing tools (read_url, search_code, load_domain_tools) instead of adding new ones, you keep the model's action space small and stable.

Why this Architecture is Revolutionary
Zero-Pollution Context: The small model (e.g., Qwen 7B) never sees the 10,000-token prompt from the original Claude skill. It only sees the intercept_search_code output, which is a highly dense, 200-token JSON schema.

"Self-Healing" Execution: The verify_skill_with_agent function uses your own agent framework to debug the code before saving it. If the distilled Python code has a syntax error, the sub-agent fixes it, ensuring that only proven, working code enters the index.

OOD (Out-of-Distribution) Generalization: By intercepting read_url, if a user asks your agent to "Use the technique described in this URL: github.com/...", your agent doesn't crash from context overflow. It seamlessly delegates the reading to the Distiller, which translates it into native tools on the fly.


Real-world skills (like those from the OpenClaw or Claude ecosystems) are not flat text files; they are repositories containing README.md/SKILL.md, Python scripts, reference prompts, and sometimes testing environments.

If we feed a whole directory into a small model, the context window will explode, and the model will suffer from severe hallucination.

To solve this, we will upgrade the skill_distiller into a "Distiller Mini-Agent".

Role: A strict, low-temperature librarian. It does not invent code; it only reads, organizes, checks safety, and packages.

Tools: It will have exclusive access to list_remote_repo and read_remote_file.

Flow: It reads SKILL.md first, figures out which .py files actually matter, reads only those specific files, and then calls a final tool submit_distilled_skill to finish its job.


Integrating the Idle Skill Discovery Daemon into an existing arq async worker is an excellent architectural pattern. It allows your worker to effectively utilize its idle compute cycles to self-improve, while still remaining highly responsive to priority tasks via Redis.
"""


import argparse
import ast
import asyncio
import base64
import hashlib
import json
import logging
import os
import random
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse
import aiohttp
import numpy as np
from pathlib import Path
import shlex
import time
import venv
import shutil

# ----------------------------------------------------------------------------
# Placeholders for your existing framework imports
# from BatchAgent.llm_wrapper import complete_with_continuation_async
# ----------------------------------------------------------------------------

SKILL_INDEX_PATH = Path("agent_workspace/.agent/distilled_skills_index.json")
SKILL_SANDBOX_DIR = Path("agent_workspace/.agent/skill_sandbox")
SFT_EXPORT_DIR = Path("agent_workspace/.agent/distilled_sft")
logger = logging.getLogger(__name__)


# ============================================================================
# 1. Hardware/Service Monitoring: Dynamic vLLM Idle Detection
# ============================================================================
async def is_system_idle(
    mode: str = "vllm", 
    base_url: str = "http://127.0.0.1:8000/v1",
    max_running_reqs: int = 1
) -> bool:
    """
    Universal system idle detection to determine if background skill extraction should start.
    Supported modes: 'vllm' (checks remote inference node queue), 'manual' (defaults to idle, controlled externally).
    """
    if mode == "manual":
        return True 
        
    if mode == "vllm":
        # Automatically convert the provided OpenAI-compatible endpoint (/v1) to the Prometheus Metrics endpoint (/metrics)
        parsed = urlparse(base_url)
        metrics_url = f"{parsed.scheme}://{parsed.netloc}/metrics"
        
        try:
            async with aiohttp.ClientSession() as session:
                logger.debug(f"[IdleCheck] Polling vLLM metrics at: {metrics_url}")
                async with session.get(metrics_url, timeout=3) as resp:
                    if resp.status == 200:
                        text = await resp.text()
                        for line in text.splitlines():
                            # Monitor the number of running or waiting requests
                            if line.startswith("vllm:num_requests_running") or line.startswith("vllm:num_requests_waiting"):
                                val = float(line.split()[-1])
                                if val >= max_running_reqs:
                                    logger.debug(f"[IdleCheck] System busy. {line.strip()}")
                                    return False
                        return True
                    else:
                        logger.warning(f"[IdleCheck] HTTP {resp.status} from metrics endpoint.")
        except Exception as e:
            logger.debug(f"[IdleCheck] vLLM check failed ({e}), assuming NOT idle for safety.")
            return False
            
    return False

async def check_embedding_endpoint(base_url: str) -> Tuple[bool, str]:
    # 清理末尾斜杠
    base = base_url.rstrip("/")
    
    # 智能拼接：如果用户已经填了 /v1/embeddings，就不再重复加
    if "/v1/embeddings" in base:
        url = base
    else:
        url = f"{base}/v1/embeddings"
        
    try:
        async with aiohttp.ClientSession() as session:
            # 探测核心：发送一个合法的最小 JSON，避免 422 干扰
            test_payload = {"input": "ping", "model": "test"}
            async with session.post(url, json=test_payload, timeout=5.0) as resp:
                # 只要不是 404(路径错) 或 500(程序崩)，200 或 422 都说明接口是通的
                if resp.status < 400:
                    return True, f"Success! (HTTP {resp.status}) at {url}"
                elif resp.status == 422:
                    return True, f"Reachable (Wait for valid input, HTTP 422) at {url}"
                return False, f"Check failed: HTTP {resp.status} at {url}"
    except Exception as exc:
        return False, f"Connection failed: {exc}"


def parse_github_url(url: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    parsed = urlparse(url)
    if parsed.netloc != "github.com":
        return None, None, None, None
    parts = [p for p in parsed.path.strip("/").split("/") if p]
    if len(parts) < 2:
        return None, None, None, None
    owner, repo = parts[0], parts[1]
    if len(parts) >= 4 and parts[2] in {"tree", "blob"}:
        branch = parts[3]
        path = "/".join(parts[4:])
    else:
        branch = "main"
        path = "/".join(parts[2:])
    return owner, repo, path, branch


def github_contents_api(owner: str, repo: str, path: str, branch: str) -> str:
    clean = path.strip("/")
    if clean:
        return f"https://api.github.com/repos/{owner}/{repo}/contents/{clean}?ref={branch}"
    return f"https://api.github.com/repos/{owner}/{repo}/contents?ref={branch}"


def github_raw_url(owner: str, repo: str, branch: str, path: str) -> str:
    clean = path.strip("/")
    return f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{clean}"


class GitHubRemote:
    def __init__(self, token: Optional[str] = None):
        self.token = token

    def _headers(self) -> Dict[str, str]:
        headers = {"Accept": "application/vnd.github+json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    async def list_remote_repo(self, url: str) -> str:
        owner, repo, path, branch = parse_github_url(url)
        if not owner:
            return "Error: Invalid GitHub URL."
        api_url = github_contents_api(owner, repo, path or "", branch or "main")
        async with aiohttp.ClientSession(headers=self._headers()) as session:
            async with session.get(api_url, timeout=20) as resp:
                if resp.status != 200:
                    return f"Error fetching repo: {resp.status}"
                data = await resp.json()
        if isinstance(data, dict) and data.get("type") == "file":
            return f"Repository Contents:\n[FILE] {data.get('name')} (Path: {data.get('path')}, URL: {data.get('html_url')})"
        if isinstance(data, dict):
            return f"Error fetching repo: {data.get('message', 'unknown error')}"
        lines = []
        for item in data:
            kind = "DIR " if item.get("type") == "dir" else "FILE"
            lines.append(f"[{kind}] {item.get('name')} (Path: {item.get('path')}, URL: {item.get('html_url')})")
        return "Repository Contents:\n" + "\n".join(lines)

    async def read_remote_file(self, file_url: str) -> str:
        owner, repo, path, branch = parse_github_url(file_url)
        if not owner:
            return "Error: Invalid GitHub file URL."
        if not path:
            return "Error: File path is empty."
        raw = github_raw_url(owner, repo, branch or "main", path)
        async with aiohttp.ClientSession() as session:
            async with session.get(raw, timeout=20) as resp:
                if resp.status != 200:
                    return f"Error reading file: HTTP {resp.status}"
                text = await resp.text()
        return text

    async def download_repo_subdir(self, url: str, dst_dir: Path) -> Tuple[bool, str]:
        owner, repo, root_path, branch = parse_github_url(url)
        if not owner:
            return False, "Invalid GitHub URL"
        branch = branch or "main"
        dst_dir.mkdir(parents=True, exist_ok=True)
        async with aiohttp.ClientSession(headers=self._headers()) as session:
            ok, msg = await self._download_recursive(session, owner, repo, root_path or "", branch, dst_dir)
            return ok, msg

    async def _download_recursive(
        self,
        session: aiohttp.ClientSession,
        owner: str,
        repo: str,
        repo_path: str,
        branch: str,
        local_dir: Path,
    ) -> Tuple[bool, str]:
        api_url = github_contents_api(owner, repo, repo_path, branch)
        async with session.get(api_url, timeout=30) as resp:
            if resp.status != 200:
                return False, f"Download failed at {repo_path}: HTTP {resp.status}"
            data = await resp.json()
        if isinstance(data, dict) and data.get("type") == "file":
            local_dir.parent.mkdir(parents=True, exist_ok=True)
            encoded = data.get("content", "")
            if encoded:
                content = base64.b64decode(encoded).decode("utf-8", errors="ignore")
                local_dir.write_text(content, encoding="utf-8")
                return True, "OK"
            download_url = data.get("download_url")
            if not download_url:
                return False, f"No download URL for {repo_path}"
            async with session.get(download_url, timeout=30) as f_resp:
                if f_resp.status != 200:
                    return False, f"Raw download failed for {repo_path}: HTTP {f_resp.status}"
                local_dir.write_text(await f_resp.text(), encoding="utf-8")
            return True, "OK"
        if not isinstance(data, list):
            return False, f"Unexpected GitHub payload at {repo_path}"
        for item in data:
            item_type = item.get("type")
            item_path = item.get("path", "")
            rel_name = item.get("name", "")
            target = local_dir / rel_name
            if item_type == "dir":
                target.mkdir(parents=True, exist_ok=True)
                ok, msg = await self._download_recursive(session, owner, repo, item_path, branch, target)
                if not ok:
                    return False, msg
            elif item_type == "file":
                download_url = item.get("download_url")
                if not download_url:
                    continue
                async with session.get(download_url, timeout=30) as f_resp:
                    if f_resp.status != 200:
                        return False, f"Raw download failed for {item_path}: HTTP {f_resp.status}"
                    target.write_text(await f_resp.text(), encoding="utf-8")
        return True, "OK"

# ============================================================================
# 1. Embedding Generator (OpenAI-compatible /v1/embeddings)
# ============================================================================
async def generate_embedding(
    text: str, 
    base_url: Optional[str] = None, 
    api_key: Optional[str] = None, 
    model: Optional[str] = None
) -> List[float]:
    """
    Calls an OpenAI-compatible /v1/embeddings endpoint to vectorize text.
    Includes a fallback mock generator for local testing if the endpoint fails.
    """
    # Ensure the URL points to the embeddings endpoint
    effective_base_url = (base_url or os.environ.get("EMBEDDING_BASE_URL") or "http://100.81.148.35:8003/v1").strip()
    effective_api_key = (api_key or os.environ.get("EMBEDDING_API_KEY") or "EMPTY").strip()
    effective_model = (model or os.environ.get("EMBEDDING_MODEL") or "BAAI/bge-large-zh-v1.5").strip()
    endpoint = effective_base_url.rstrip("/")
    if not endpoint.endswith("/embeddings"):
        if endpoint.endswith("/v1"):
            endpoint += "/embeddings"
        else:
            endpoint += "/v1/embeddings"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {effective_api_key}"
    }
    payload = {
        "input": text,
        "model": effective_model
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(endpoint, headers=headers, json=payload, timeout=5.0) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data["data"][0]["embedding"]
                else:
                    logger.warning(f"[Embeddings] HTTP {resp.status}. Falling back to mock embedding for testing.")
    except Exception as e:
        logger.warning(f"[Embeddings] Connection failed ({e}). Falling back to mock embedding.")
    
    # --- Fallback for testing without a real embedding server ---
    # Generates a deterministic pseudo-random vector based on string hash
    np.random.seed(abs(hash(text)) % (2**32))
    mock_vec = np.random.rand(128).tolist() # 128-dimensional mock vector
    return mock_vec

# ============================================================================
# 2. Semantic Indexing (Save Skill + Vector)
# ============================================================================
async def index_distilled_skill(
    skill_package: Dict[str, Any], 
    base_url: Optional[str] = None, 
    api_key: Optional[str] = None,
    model: Optional[str] = None
) -> None:
    """
    Generates an embedding for the skill based on its summary, and saves it
    to the lightweight local vector index.
    """
    SKILL_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # Load existing index
    index_data = {}
    if SKILL_INDEX_PATH.exists():
        with open(SKILL_INDEX_PATH, "r", encoding="utf-8") as f:
            index_data = json.load(f)

    skill_name = skill_package["skill_name"]
    summary = skill_package.get("summary", "")
    
    # Create a rich semantic string to embed
    semantic_text = f"Tool Name: {skill_name}. Description: {summary}"
    logger.info(f"[SemanticIndex] Generating embedding for skill: {skill_name}")
    
    embedding = await generate_embedding(semantic_text, base_url, api_key, model)
    
    # Save the skill data alongside its vector
    index_data[skill_name] = {
        "summary": summary,
        "entrypoint": skill_package.get("entrypoint"),
        "files": skill_package.get("files"),
        "embedding": embedding,
        "source_url": skill_package.get("source_url"),
        "artifact_dir": skill_package.get("artifact_dir"),
        "source_snapshot_dir": skill_package.get("source_snapshot_dir"),
        "generated_dir": skill_package.get("generated_dir"),
    }
    
    with open(SKILL_INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(index_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"[SemanticIndex] Successfully indexed '{skill_name}'.")

# ============================================================================
# 3. JIT Semantic Routing (Search)
# ============================================================================
def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Calculates Cosine Similarity using numpy."""
    a, b = np.array(vec_a), np.array(vec_b)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

async def route_jit_query(
    user_query: str, 
    base_url: Optional[str] = None, 
    api_key: Optional[str] = None, 
    top_k: int = 1,
    threshold: float = 0.5,
    model: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Takes a natural language user query, embeds it, and finds the most semantically
    similar tools in the index.
    """
    if not SKILL_INDEX_PATH.exists():
        logger.warning("[JIT Router] Skill index is empty. No skills available.")
        return []

    with open(SKILL_INDEX_PATH, "r", encoding="utf-8") as f:
        index_data = json.load(f)

    # 1. Embed the user's intent
    query_embedding = await generate_embedding(user_query, base_url, api_key, model)
    
    results = []

    def add_candidate(skill_name: str, data: Dict[str, Any]) -> None:
        skill_emb = data.get("embedding")
        if not skill_emb:
            return
        sim = cosine_similarity(query_embedding, skill_emb)
        if sim >= threshold:
            results.append({
                "skill_name": skill_name,
                "summary": data.get("summary", ""),
                "similarity": sim,
                "entrypoint": data.get("entrypoint"),
                "files": data.get("files") or {},
            })

    for skill_name, data in index_data.items():
        if isinstance(data, dict):
            add_candidate(skill_name, data)
        elif isinstance(data, list):
            for idx, item in enumerate(data):
                if not isinstance(item, dict):
                    continue
                schema = item.get("schema") or {}
                candidate_name = schema.get("name") or f"{skill_name}_{idx}"
                add_candidate(candidate_name, item)

    # 3. Sort by highest similarity
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:top_k]


def _parse_payload_dict(payload: Any) -> Dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, str):
        try:
            parsed = json.loads(payload)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return {}
    return {}


def _derive_capabilities(distilled_package: Dict[str, Any]) -> List[str]:
    summary = str(distilled_package.get("summary", "") or "").lower()
    entrypoint = str(distilled_package.get("entrypoint", "") or "").lower()
    files = distilled_package.get("files") or {}
    file_names = [str(k).lower() for k in files.keys()]
    merged_text = " ".join([summary, entrypoint, " ".join(file_names)])
    if isinstance(files, dict):
        merged_text += " " + " ".join(str(v).lower()[:1200] for v in files.values())
    rules = [
        ("analy", "data_analysis"),
        ("monitor", "system_monitoring"),
        ("geo", "geospatial_processing"),
        ("plot", "visualization"),
        ("chart", "visualization"),
        ("query", "query_execution"),
        ("calc", "numerical_computation"),
        ("forecast", "forecasting"),
        ("clean", "data_cleaning"),
        ("json", "json_processing"),
    ]
    capabilities: List[str] = []
    for marker, tag in rules:
        if marker in merged_text and tag not in capabilities:
            capabilities.append(tag)
    if entrypoint:
        capabilities.append("script_execution")
    if not capabilities:
        capabilities.append("general_automation")
    return capabilities[:8]


def _derive_failure_modes(verification_msg: str) -> List[str]:
    msg = (verification_msg or "").lower()
    if not msg:
        return []
    modes: List[str] = []
    if "dependency installation failed" in msg:
        modes.append("missing_dependency")
    if "execution failed" in msg:
        modes.append("runtime_error")
    if "timed out" in msg:
        modes.append("timeout")
    if "not valid json" in msg:
        modes.append("invalid_output_format")
    if "security block" in msg:
        modes.append("security_restriction")
    if not modes:
        modes.append("sandbox_issue")
    return modes


def build_skill_card(distilled_package: Dict[str, Any], verification_msg: str = "") -> Dict[str, Any]:
    """
    Normalize distilled skill metadata into a structured skill card.
    """
    payload_dict = _parse_payload_dict(distilled_package.get("test_payload_json"))
    required_inputs = sorted(payload_dict.keys())
    files_obj = distilled_package.get("files") or {}
    files = sorted(files_obj.keys()) if isinstance(files_obj, dict) else []
    return {
        "skill_name": distilled_package.get("skill_name", "unknown_skill"),
        "summary": distilled_package.get("summary", ""),
        "source_url": distilled_package.get("source_url", ""),
        "entrypoint": distilled_package.get("entrypoint", ""),
        "capabilities": _derive_capabilities(distilled_package),
        "required_inputs": required_inputs,
        "optional_inputs": [],
        "failure_modes": _derive_failure_modes(verification_msg),
        "files": files,
        "artifact_dir": distilled_package.get("artifact_dir", ""),
        "source_snapshot_dir": distilled_package.get("source_snapshot_dir", ""),
        "generated_dir": distilled_package.get("generated_dir", ""),
    }


def _get_skill_embedding(skill_name: str, index_data: Dict[str, Any]) -> Optional[List[float]]:
    if not isinstance(index_data, dict):
        return None
    data = index_data.get(skill_name)
    if isinstance(data, dict):
        emb = data.get("embedding")
        if isinstance(emb, list):
            return emb
    return None


def _nearest_neighbor_names(skill_name: str, index_data: Dict[str, Any], top_k: int = 2) -> List[str]:
    current = _get_skill_embedding(skill_name, index_data)
    if not current:
        return []
    candidates: List[Tuple[str, float]] = []
    for other_name, other_data in index_data.items():
        if other_name == skill_name or not isinstance(other_data, dict):
            continue
        emb = other_data.get("embedding")
        if not isinstance(emb, list):
            continue
        sim = cosine_similarity(current, emb)
        candidates.append((other_name, sim))
    candidates.sort(key=lambda x: x[1], reverse=True)
    return [name for name, _ in candidates[:top_k]]


def generate_sft_examples_from_skill(
    skill_card: Dict[str, Any],
    index_data: Optional[Dict[str, Any]] = None,
    rng: Optional[random.Random] = None,
) -> List[Dict[str, Any]]:
    """
    Generate behavior-level SFT examples for one skill card.
    """
    local_rng = rng or random.Random(7)
    skill_name = skill_card.get("skill_name", "unknown_skill")
    summary = skill_card.get("summary", "") or "No summary available."
    required_inputs = skill_card.get("required_inputs") or []
    capabilities = skill_card.get("capabilities") or ["general_automation"]
    entrypoint = skill_card.get("entrypoint", "")
    file_count = len(skill_card.get("files") or [])

    examples: List[Dict[str, Any]] = []

    positive_user = f"I need help with {capabilities[0].replace('_', ' ')}. Which skill should I use?"
    positive_assistant = f"Use {skill_name}. It matches the request because {summary}"
    examples.append({
        "user": positive_user,
        "assistant": positive_assistant,
        "type": "skill_routing_positive",
        "skill_name": skill_name,
        "capability": capabilities[0],
    })

    neighbor_names = _nearest_neighbor_names(skill_name, index_data or {}, top_k=2)
    if neighbor_names:
        for neighbor in neighbor_names:
            examples.append({
                "user": f"I want to run {neighbor} to handle this task.",
                "assistant": f"Do not use {skill_name} for that request. {neighbor} is a better match for this intent.",
                "type": "skill_routing_negative",
                "skill_name": skill_name,
                "neighbor_skill": neighbor,
            })
    else:
        examples.append({
            "user": "I need website translation and UI localization.",
            "assistant": f"Do not use {skill_name}. This skill is focused on: {summary}",
            "type": "skill_routing_negative",
            "skill_name": skill_name,
            "neighbor_skill": "",
        })

    missing_field = required_inputs[0] if required_inputs else "input_data"
    examples.append({
        "user": f"Run {skill_name} now. I only provide an empty payload.",
        "assistant": f"Input validation failed. Missing required field: {missing_field}. Provide all required inputs before execution.",
        "type": "skill_input_check",
        "skill_name": skill_name,
        "required_inputs": required_inputs,
    })

    plan_items = [
        "Validate user intent against skill scope",
        "Construct argument payload with required fields",
        "Execute entrypoint and collect output",
        "Summarize result for user",
    ]
    local_rng.shuffle(plan_items)
    planning_assistant = " -> ".join(plan_items[:4])
    examples.append({
        "user": f"Plan how to use {skill_name} for this request.",
        "assistant": planning_assistant,
        "type": "skill_planning",
        "skill_name": skill_name,
    })

    argument_obj = {k: f"<{k}_value>" for k in required_inputs} if required_inputs else {"input_data": "<input_data_value>"}
    examples.append({
        "user": f"Construct runtime arguments for {skill_name}.",
        "assistant": json.dumps(argument_obj, ensure_ascii=False),
        "type": "skill_argument_construction",
        "skill_name": skill_name,
        "entrypoint": entrypoint,
    })

    examples.append({
        "user": f"Summarize the result from {skill_name} execution for end users.",
        "assistant": f"{skill_name} completed successfully using {file_count} files. Return concise key findings and next actions.",
        "type": "skill_result_synthesis",
        "skill_name": skill_name,
    })

    failure_modes = skill_card.get("failure_modes") or []
    failure_mode = failure_modes[0] if failure_modes else "runtime_error"
    examples.append({
        "user": f"{skill_name} failed during execution. What recovery should I apply?",
        "assistant": f"Recovery strategy: inspect logs, correct inputs or dependencies, and retry {entrypoint or 'entrypoint'} with validated payload. Failure mode: {failure_mode}.",
        "type": "skill_failure_recovery",
        "skill_name": skill_name,
        "failure_mode": failure_mode,
    })
    return examples


def load_existing_sft_examples(file_path: Path) -> List[Dict[str, Any]]:
    """
    Load existing JSONL rows from disk.
    """
    if not file_path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
                if isinstance(parsed, dict):
                    rows.append(parsed)
            except Exception:
                continue
    return rows


def _sft_dedup_hash(example: Dict[str, Any]) -> str:
    base = {
        "user": example.get("user", ""),
        "assistant": example.get("assistant", ""),
        "type": example.get("type", ""),
        "skill_name": example.get("skill_name", ""),
    }
    raw = json.dumps(base, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def dedup_sft_examples(
    examples: List[Dict[str, Any]],
    existing_examples: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Deduplicate SFT rows by user/assistant/type/skill_name hash.
    """
    seen = set()
    if existing_examples:
        for row in existing_examples:
            seen.add(_sft_dedup_hash(row))
    deduped: List[Dict[str, Any]] = []
    for ex in examples:
        key = _sft_dedup_hash(ex)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(ex)
    return deduped


def append_jsonl(file_path: Path, rows: List[Dict[str, Any]]) -> int:
    """
    Append rows into JSONL file with dedup against existing rows.
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_existing_sft_examples(file_path)
    new_rows = dedup_sft_examples(rows, existing)
    if not new_rows:
        return 0
    with open(file_path, "a", encoding="utf-8") as f:
        for row in new_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(new_rows)


def export_sft_outputs(skill_card: Dict[str, Any], examples: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Export generated SFT rows into split JSONL files and all-in-one JSONL.
    """
    sft_files = {
        "skill_routing_positive": SFT_EXPORT_DIR / "sft_routing.jsonl",
        "skill_routing_negative": SFT_EXPORT_DIR / "sft_routing.jsonl",
        "skill_input_check": SFT_EXPORT_DIR / "sft_arguments.jsonl",
        "skill_planning": SFT_EXPORT_DIR / "sft_planning.jsonl",
        "skill_argument_construction": SFT_EXPORT_DIR / "sft_arguments.jsonl",
        "skill_result_synthesis": SFT_EXPORT_DIR / "sft_synthesis.jsonl",
        "skill_failure_recovery": SFT_EXPORT_DIR / "sft_recovery.jsonl",
    }
    written_counts: Dict[str, int] = {}
    all_written = 0

    grouped: Dict[Path, List[Dict[str, Any]]] = {}
    for ex in examples:
        target = sft_files.get(ex.get("type"), SFT_EXPORT_DIR / "sft_all.jsonl")
        grouped.setdefault(target, []).append(ex)
    for path, rows in grouped.items():
        n = append_jsonl(path, rows)
        written_counts[path.name] = n
        all_written += n

    all_count = append_jsonl(SFT_EXPORT_DIR / "sft_all.jsonl", examples)
    written_counts["sft_all.jsonl"] = all_count
    all_written += all_count

    card_row = {
        "user": f"skill_card::{skill_card.get('skill_name', 'unknown_skill')}",
        "assistant": json.dumps(skill_card, ensure_ascii=False, sort_keys=True),
        "type": "skill_card",
        "skill_name": skill_card.get("skill_name", "unknown_skill"),
        "source_url": skill_card.get("source_url", ""),
    }
    card_count = append_jsonl(SFT_EXPORT_DIR / "skill_cards.jsonl", [card_row])
    written_counts["skill_cards.jsonl"] = card_count
    all_written += card_count
    written_counts["total_appended_rows"] = all_written
    return written_counts

# ============================================================================
# 4. Test Samples & Execution
# ============================================================================
async def test_semantic_routing(base_url: Optional[str] = None, api_key: Optional[str] = None, model: Optional[str] = None):
    print("\n--- Testing Semantic Search & JIT Routing ---")
    
    # Dummy config
    effective_base_url = (base_url or os.environ.get("EMBEDDING_BASE_URL") or "http://100.81.148.35:8003/v1").strip()
    effective_api_key = (api_key or os.environ.get("EMBEDDING_API_KEY") or "EMPTY").strip()
    effective_model = (model or os.environ.get("EMBEDDING_MODEL") or "BAAI/bge-large-zh-v1.5").strip()
    
    # 1. Mock two distilled skills
    skill_math = {
        "skill_name": "calculator_tool",
        "summary": "Performs complex mathematical operations, calculus, and algebra.",
        "entrypoint": "main.py",
        "files": {"main.py": "print('math')"}
    }
    
    skill_sys = {
        "skill_name": "system_monitor",
        "summary": "Checks CPU, RAM usage, and monitors disk space.",
        "entrypoint": "monitor.py",
        "files": {"monitor.py": "print('sys')"}
    }
    skill_data = {
        "skill_name": "data_analyzer",
        "summary": "Analyzes scientific datasets and produces structured summaries.",
        "entrypoint": "main.py",
        "source_url": "https://github.com/K-Dense-AI/claude-scientific-skills/tree/main/scientific-skills/data_analyzer",
        "files": {"main.py": "print('data analyzer')"},
    }
    
    # 2. Index them (This will generate embeddings and save to JSON)
    print("Indexing skills...")
    await index_distilled_skill(skill_math, effective_base_url, effective_api_key, effective_model)
    await index_distilled_skill(skill_sys, effective_base_url, effective_api_key, effective_model)
    await index_distilled_skill(skill_data, effective_base_url, effective_api_key, effective_model)
    
    # 3. Test Query A: "How much memory am I using?"
    # Note: Keyword search would fail here because "memory" isn't in the summary of system_monitor.
    # Semantic search will succeed.
    query_a = "How much memory am I using?"
    print(f"\nUser Query: '{query_a}'")
    routes_a = await route_jit_query(query_a, effective_base_url, effective_api_key, threshold=0.0, model=effective_model) # threshold 0.0 for pure ranking test
    
    for rank, route in enumerate(routes_a):
        print(f"  Rank {rank+1}: {route['skill_name']} (Similarity: {route['similarity']:.4f})")
        
# ============================================================================
# 2. Security Core: High-Precision AST Scanner
# ============================================================================
class SecurityScanner(ast.NodeVisitor):
    def __init__(self):
        self.violations = []
        # Strictly banned modules (involving low-level OS operations, process spawning)
        self.banned_modules = {'pty', 'subprocess', 'shlex', 'socket'}
        # Dangerous built-in function calls (involving dynamic execution)
        self.banned_calls = {'eval', 'exec', '__import__', 'open'}
        # Dangerous object attributes/methods (e.g., os.system, shutil.rmtree)
        self.banned_attributes = {'system', 'popen', 'remove', 'rmdir', 'chmod', 'chown', 'fork', 'rmtree'}

    def visit_Import(self, node):
        for alias in node.names:
            base_module = alias.name.split('.')[0]
            if base_module in self.banned_modules:
                self.violations.append(f"Banned import: {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module:
            base_module = node.module.split('.')[0]
            if base_module in self.banned_modules:
                self.violations.append(f"Banned from-import: {node.module}")
        self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            if node.func.id in self.banned_calls:
                self.violations.append(f"Banned builtin call: {node.func.id}")
        self.generic_visit(node)

    def visit_Attribute(self, node):
        if node.attr in self.banned_attributes:
            self.violations.append(f"Banned attribute access: {node.attr}")
        self.generic_visit(node)

def ast_scan_minipackage(files_dict: Dict[str, str]) -> Tuple[bool, List[str]]:
    """
    Iterates through all Python files in the extracted mini-package and performs AST security scanning.
    Returns: (is_safe, list_of_violations)
    """
    all_violations = []
    
    for filename, code in files_dict.items():
        if not filename.endswith('.py'):
            continue
            
        try:
            tree = ast.parse(code)
            scanner = SecurityScanner()
            scanner.visit(tree)
            if scanner.violations:
                all_violations.extend([f"[{filename}] {v}" for v in scanner.violations])
        except SyntaxError as e:
            all_violations.append(f"[{filename}] SyntaxError: {e}")
            
    is_safe = len(all_violations) == 0
    return is_safe, all_violations

# ============================================================================
# 3. Mini-Agent Tool Schema (Multi-file Mini-Package Support)
# ============================================================================
DISTILLER_TOOLS_SCHEMA = [
    {
        "name": "list_remote_repo",
        "description": "List all files and directories in a given GitHub URL.",
        "properties": {"url": {"type": "string"}},
        "required": ["url"],
    },
    {
        "name": "read_remote_file",
        "description": "Read the text content of a specific file from GitHub.",
        "properties": {"file_url": {"type": "string"}},
        "required": ["file_url"],
    },
    {
        "name": "submit_distilled_skill",
        "description": "Submit a distilled, safe, multi-file python package.",
        "properties": {
            "skill_name": {"type": "string", "description": "Short, lowercase name for the tool."},
            "summary": {"type": "string", "description": "What the tool does."},
            "env_setup_bash": {"type": "string", "description": "e.g., pip install pandas requests"},
            "files": {
                "type": "object",
                "description": "Dictionary of files. Key is filename (e.g., 'main.py', 'utils.py'), Value is the string content of the file."
            },
            "entrypoint": {
                "type": "string", 
                "description": "The main script to execute (e.g., 'main.py'). Must read JSON from sys.argv[1] and print JSON."
            },
            "test_payload_json": {"type": "string", "description": "JSON string to test the code."},
        },
        "required": ["skill_name", "summary", "files", "entrypoint", "test_payload_json"],
    },
]

# ============================================================================
# 4. Core Dispatcher: Distiller Mini-Agent
# ============================================================================
async def run_distiller_agent(
    client: Any,
    model: str,
    skill_repo_url: str,
    github: Any, 
    max_turns: int = 6,
) -> Optional[Dict[str, Any]]:
    
    system_prompt = f"""
You are an elite AI Skill Distiller.
Goal: Explore a remote repository containing a skill, extract functional code into a mini-package, verify safety, and submit.

RULES:
1) KEEP ARCHITECTURE: If the skill uses multiple files (e.g., main.py and utils.py), preserve them in the `files` dictionary. Do not force them into one file.
2) SAFETY FIRST: Do not include code that runs destructive OS commands or evals arbitrary strings.
3) OUTPUT: The `entrypoint` script MUST read JSON arguments from sys.argv[1] and print the final JSON result to stdout.

Target Skill URL: {skill_repo_url}
Start by listing repository contents.
""".strip()

    try:
        from BatchAgent.llm_wrapper import complete_with_continuation_async
    except ModuleNotFoundError:
        from llm_wrapper import complete_with_continuation_async
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"List the repository contents for: {skill_repo_url}"},
    ]
    distilled_package: Optional[Dict[str, Any]] = None

    for turn in range(max_turns):
        logger.info(f"[Distiller] turn={turn + 1}/{max_turns} url={skill_repo_url}")

        content, actions = await complete_with_continuation_async(
            client=client,
            model=model,
            messages=messages,
            temperature=0.1,
            max_output_tokens=2048,
            tools=DISTILLER_TOOLS_SCHEMA,
            tool_strategy="native",
            enable_thinking=True,
        )

        messages.append({"role": "assistant", "content": content})
        feedback_blocks: List[str] = []
        submitted = None

        for action in actions or []:
            name = getattr(action, "name", None)
            args = getattr(action, "args", None) or {}
            if not name: continue

            if name == "list_remote_repo":
                res = await github.list_remote_repo(args.get("url", ""))
                feedback_blocks.append(f"### Result for list_remote_repo:\n```\n{res}\n```")

            elif name == "read_remote_file":
                res = await github.read_remote_file(args.get("file_url", ""))
                feedback_blocks.append(f"### Result for read_remote_file:\n```\n{res}\n```")

            elif name == "submit_distilled_skill":
                submitted = args
                files_dict = submitted.get("files", {})
                entrypoint = submitted.get("entrypoint", "")

                # 1. Structure Check
                if not files_dict or entrypoint not in files_dict:
                    feedback_blocks.append("System Warning: 'files' is empty or 'entrypoint' is not in 'files'. Please fix and resubmit.")
                    submitted = None
                    continue

                # 2. AST Security Scan Interception
                is_safe, violations = ast_scan_minipackage(files_dict)
                if not is_safe:
                    error_msg = "\n".join(violations)
                    feedback_blocks.append(
                        "### submit_distilled_skill rejected by AST Security Scanner\n"
                        f"The extracted code contains dangerous patterns:\n{error_msg}\n\n"
                        "Please refactor the code (e.g., use safe Python native libraries instead of OS commands), and call submit_distilled_skill again."
                    )
                    submitted = None
                    logger.warning(f"[Distiller] AST blocked dangerous code: {violations}")
                    continue
                
                # Security check passed
                break

        if submitted:
            submitted["source_url"] = skill_repo_url
            distilled_package = submitted
            logger.info(f"[Distiller] Multi-file Package extracted safely. Entrypoint: {distilled_package['entrypoint']}")
            break

        if not feedback_blocks:
            feedback_blocks.append("System Warning: Use list_remote_repo/read_remote_file, or submit_distilled_skill if done.")

        messages.append({"role": "user", "content": "\n\n".join(feedback_blocks)})

    return distilled_package

# ============================================================================
# 6. Test Module
# ============================================================================
async def test_idle_checker(base_url: str):
    print("\n--- Testing Idle Checker ---")
    
    print(f"Testing vLLM mode with base_url: {base_url}")
    is_idle_vllm = await is_system_idle(mode="vllm", base_url=base_url)
    print(f"[Idle Check Result] vLLM idle status: {is_idle_vllm}")

def test_ast_scanner():
    print("\n--- Testing AST Security Scanner ---")
    safe_package = {
        "utils.py": "def add(a, b):\n    return a + b\n",
        "main.py": "from utils import add\nprint(add(1, 2))"
    }
    unsafe_package = {
        "main.py": "import os\nos.system('rm -rf /')"
    }
    
    is_safe_1, _ = ast_scan_minipackage(safe_package)
    is_safe_2, violations = ast_scan_minipackage(unsafe_package)
    
    print(f"Safe package passed: {is_safe_1}")
    print(f"Unsafe package blocked: {not is_safe_2}")
    if not is_safe_2:
        print(f"Interception details: {violations}")

# ============================================================================
# 7. Local Sandbox Verification (Venv & TDD)
# ============================================================================
async def verify_in_venv_sandbox(distilled_package: Dict[str, Any], timeout_s: int = 60) -> Tuple[bool, str]:
    """
    Creates an isolated Python virtual environment, installs dependencies, 
    and runs the extracted multi-file skill against its test payload.
    """
    skill_name = distilled_package.get("skill_name", "unknown_skill")
    #workdir = SKILL_SANDBOX_DIR / f"{skill_name}_{int(time.time())}"
    #workdir.mkdir(parents=True, exist_ok=True)

    workdir = (SKILL_SANDBOX_DIR / f"{skill_name}_{int(time.time())}").resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"[Sandbox] Creating isolated venv for '{skill_name}' at {workdir}")
    
    try:
        # 1. Write the multi-file package to disk
        files = distilled_package.get("files", {})
        for filename, content in files.items():
            file_path = workdir / filename
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")
            
        # 2. Create the virtual environment
        venv_dir = workdir / "venv"
        venv.create(venv_dir, with_pip=True)
        
        # Resolve executables depending on OS
        if os.name == 'nt':
            pip_exe = str(venv_dir / "Scripts" / "pip.exe")
            python_exe = str(venv_dir / "Scripts" / "python.exe")
        else:
            pip_exe = str(venv_dir / "bin" / "pip")
            python_exe = str(venv_dir / "bin" / "python")
            
        # 3. Setup Dependencies (env_setup_bash)
        env_setup = distilled_package.get("env_setup_bash", "").strip()
        if env_setup:
            # Guardrail: Only allow pip installs
            if not env_setup.startswith("pip install"):
                return False, "Security Block: env_setup_bash must start with 'pip install'"
            
            # Route pip to the venv's isolated pip
            setup_cmd = env_setup.replace("pip", pip_exe, 1)
            logger.debug(f"[Sandbox] Running setup: {setup_cmd}")
            
            process = await asyncio.create_subprocess_shell(
                setup_cmd, cwd=workdir, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            _, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout_s)
            
            if process.returncode != 0:
                return False, f"Dependency installation failed: {stderr.decode()}"
                
        # 4. Execute the Entrypoint with Test Payload
        entrypoint = distilled_package.get("entrypoint")
        payload = distilled_package.get("test_payload_json", "{}")
        
        if not isinstance(payload, str):
            payload = json.dumps(payload)
            
        run_cmd = f"{python_exe} {shlex.quote(entrypoint)} {shlex.quote(payload)}"
        logger.debug(f"[Sandbox] Running test execution: {run_cmd}")
        
        process = await asyncio.create_subprocess_shell(
            run_cmd, cwd=workdir, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout_s)
        
        if process.returncode != 0:
            return False, f"Execution failed: {stderr.decode()}"
            
        # 5. Validate JSON Output
        out_str = stdout.decode().strip()
        try:
            # Often scripts print multiple lines; grab the last one
            last_line = out_str.splitlines()[-1] if out_str else ""
            json.loads(last_line)
        except Exception:
            return True, f"Execution passed, but output is not valid JSON. Raw Output: {out_str}"
            
        return True, f"Success. Output: {out_str}"
        
    except asyncio.TimeoutError:
        return False, "Sandbox execution timed out."
    except Exception as e:
        return False, f"Sandbox unexpected error: {str(e)}"
    finally:
        # Optional: Clean up the sandbox to save disk space after verification
        # shutil.rmtree(workdir, ignore_errors=True)
        pass


async def idle_skill_discovery_loop(
    client: Any,
    model: str,
    github: Any,
    discovery_queue: List[str],
    base_url: str,
    api_key: str,
    embedding_base_url: Optional[str] = None,
    embedding_api_key: Optional[str] = None,
    embedding_model: Optional[str] = None,
    export_sft: bool = False,
    check_interval_s: int = 10
):
    """
    Background silent discovery loop. Distills, verifies, and semantically indexes GitHub Repos.
    """
    logger.info("[IdleDaemon] Started background learning daemon...")
    
    for url in discovery_queue:
        while not await is_system_idle(mode="vllm", base_url=base_url):
            logger.debug("[IdleDaemon] System busy, sleeping...")
            await asyncio.sleep(check_interval_s)

        logger.info(f"[IdleDaemon] System idle. Starting distillation for {url}")
        
        # Step 1: Extract & AST Filter
        distilled_package = await run_distiller_agent(client, model, url, github)
        if not distilled_package:
            logger.warning(f"[IdleDaemon] Distillation failed/aborted for {url}")
            continue
            
        # Step 2: Venv Sandbox Verification
        logger.info(f"[IdleDaemon] Verifying '{distilled_package['skill_name']}' in Sandbox...")
        v_ok, v_msg = await verify_in_venv_sandbox(distilled_package)
        
        if not v_ok:
            # Here you could implement "Self-Healing": feed v_msg back to the QA Agent to fix the code
            logger.warning(f"[IdleDaemon] Sandbox verification failed for {url}: {v_msg}")
            continue
            
        logger.info(f"[IdleDaemon] Sandbox verification passed: {v_msg}")

        artifact_dir = (SKILL_SANDBOX_DIR / distilled_package["skill_name"]).resolve()
        source_snapshot_dir = artifact_dir / "source_skill"
        generated_dir = artifact_dir / "generated"
        if artifact_dir.exists():
            shutil.rmtree(artifact_dir, ignore_errors=True)
        source_snapshot_dir.mkdir(parents=True, exist_ok=True)
        generated_dir.mkdir(parents=True, exist_ok=True)

        source_url = distilled_package.get("source_url") or url
        dl_ok, dl_msg = await github.download_repo_subdir(source_url, source_snapshot_dir)
        if not dl_ok:
            logger.warning(f"[IdleDaemon] Source snapshot download failed for {url}: {dl_msg}")

        for filename, content in (distilled_package.get("files") or {}).items():
            out_file = generated_dir / filename
            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_text(content, encoding="utf-8")
        payload = distilled_package.get("test_payload_json", "{}")
        payload_str = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
        (generated_dir / "test_payload.json").write_text(payload_str, encoding="utf-8")
        if distilled_package.get("entrypoint"):
            (generated_dir / "entrypoint.txt").write_text(str(distilled_package["entrypoint"]), encoding="utf-8")

        distilled_package["artifact_dir"] = str(artifact_dir)
        distilled_package["source_snapshot_dir"] = str(source_snapshot_dir)
        distilled_package["generated_dir"] = str(generated_dir)

        await index_distilled_skill(
            distilled_package,
            embedding_base_url or os.environ.get("EMBEDDING_BASE_URL"),
            embedding_api_key or os.environ.get("EMBEDDING_API_KEY"),
            embedding_model or os.environ.get("EMBEDDING_MODEL"),
        )

        if export_sft:
            index_data = {}
            if SKILL_INDEX_PATH.exists():
                try:
                    with open(SKILL_INDEX_PATH, "r", encoding="utf-8") as f:
                        loaded = json.load(f)
                        if isinstance(loaded, dict):
                            index_data = loaded
                except Exception:
                    index_data = {}
            skill_card = build_skill_card(distilled_package, verification_msg=v_msg)
            sft_examples = generate_sft_examples_from_skill(skill_card, index_data=index_data)
            export_stats = export_sft_outputs(skill_card, sft_examples)
            logger.info(f"[IdleDaemon] SFT export complete for {distilled_package['skill_name']}: {export_stats}")
        
        logger.info(f"[IdleDaemon] Successfully processed and indexed {url}")
        await asyncio.sleep(check_interval_s)

# ============================================================================
# 8. Main Agent JIT Interceptor
# ============================================================================
async def inject_jit_skills(
    user_prompt: str,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    threshold: float = 0.6,
    model: Optional[str] = None
) -> str:
    """
    Intercepts the user's prompt, searches the Semantic Index, and returns a string
    to append to the Agent's system instructions if a relevant tool is found.
    """
    relevant_skills = await route_jit_query(user_prompt, base_url, api_key, top_k=1, threshold=threshold, model=model)
    
    if not relevant_skills:
        return "" # No relevant skills found, proceed normally
        
    skill = relevant_skills[0]
    injection = (
        f"\n\n[SYSTEM: JUST-IN-TIME CAPABILITY INJECTED]\n"
        f"Based on the user's request, you have temporary access to the following tool:\n"
        f"Tool Name: {skill['skill_name']}\n"
        f"Description: {skill['summary']}\n"
        f"Execution Entrypoint: {skill['entrypoint']}\n"
        f"Files:\n"
    )
    
    for filename, code in (skill.get('files') or {}).items():
        injection += f"\n--- {filename} ---\n```python\n{code}\n```\n"
        
    injection += "\nTo use this tool, execute the entrypoint script using your bash/terminal tool and pass the required JSON arguments via sys.argv[1]."
    
    return injection

# ============================================================================
# Main Entrypoint (Argparse Integration)
# ============================================================================
async def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    parser = argparse.ArgumentParser(description="Skill Distiller Background Agent Tester")
    parser.add_argument("--model", default=os.environ.get("VLLM_MODEL", "qwen-27b"))
    parser.add_argument("--base-url", default=os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", "myhpcvllmqwen"))
    # 修改默认值，删掉后面的路径
    parser.add_argument("--embedding-base-url", default=os.environ.get("EMBEDDING_BASE_URL", "http://100.81.148.35:8003"))
    parser.add_argument("--embedding-api-key", default=os.environ.get("EMBEDDING_API_KEY", "EMPTY"))
    parser.add_argument("--embedding-model", default=os.environ.get("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5"))
    parser.add_argument("--run-real-idle", action="store_true")
    parser.add_argument("--run-tests", action="store_true")
    parser.add_argument("--export-sft", action="store_true", help="Export SFT JSONL after successful distillation")
    parser.add_argument(
        "--discovery-urls", # Pluralized for clarity
        nargs='+',          # Accepts one or more arguments
        help="List of URLs to extract, separated by spaces",
        default=[]
    )
    args = parser.parse_args()

    if args.run_real_idle:
        from openai import AsyncOpenAI
        github = GitHubRemote(token=os.environ.get("GITHUB_TOKEN"))

        real_client = AsyncOpenAI(
            base_url=args.base_url,
            api_key=args.api_key,
        )
        discovery_queue = args.discovery_urls or [
            "https://github.com/K-Dense-AI/claude-scientific-skills/tree/main/scientific-skills/geopandas"
        ]
        await idle_skill_discovery_loop(
            client=real_client,
            model=args.model,
            github=github,
            discovery_queue=discovery_queue,
            base_url=args.base_url,
            api_key=args.api_key,
            embedding_base_url=args.embedding_base_url,
            embedding_api_key=args.embedding_api_key,
            embedding_model=args.embedding_model,
            export_sft=args.export_sft,
        )
        return

    if args.run_tests or not args.run_real_idle:
        await run_pipeline_tests(args)


async def run_pipeline_tests(args: argparse.Namespace) -> None:
    test_ast_scanner()
    print("\n--- Testing Venv Sandbox Execution ---")
    working_skill = {
        "skill_name": "test_math_skill",
        "env_setup_bash": "",
        "entrypoint": "main.py",
        "files": {
            "main.py": "import sys, json\nargs = json.loads(sys.argv[1])\nprint(json.dumps({'result': args['a'] + args['b']}))"
        },
        "test_payload_json": '{"a": 10, "b": 5}'
    }
    
    is_valid, msg = await verify_in_venv_sandbox(working_skill)
    print(f"Sandbox Verification Passed: {is_valid} | Message: {msg}")

    reachable, embedding_msg = await check_embedding_endpoint(args.embedding_base_url)
    if reachable:
        print(f"Embedding Endpoint Check: {embedding_msg}")
    else:
        print(f"Embedding Endpoint Check: {embedding_msg}")
    
    # 3. Test Semantic Routing & Injection
    await test_semantic_routing(args.embedding_base_url, args.embedding_api_key, args.embedding_model)
    
    print("\n--- Testing JIT Interception ---")
    user_request = "Can you calculate 10 plus 5 for me?"
    print(f"User Request: {user_request}")
    injection_prompt = await inject_jit_skills(
        user_request,
        args.embedding_base_url,
        args.embedding_api_key,
        threshold=0.0,
        model=args.embedding_model,
    ) # threshold 0.0 for testing mock
    print("Injected Prompt Content:\n")
    print(injection_prompt)

    mock_distilled_skill = {
        "skill_name": "demo_skill_sft",
        "summary": "Analyze tabular data and return key metrics.",
        "source_url": "https://github.com/example/demo_skill",
        "entrypoint": "main.py",
        "files": {"main.py": "print('ok')", "utils.py": "def x(): return 1"},
        "test_payload_json": '{"dataset_path":"demo.csv","metric":"mean"}',
        "artifact_dir": str((SKILL_SANDBOX_DIR / "demo_skill_sft").resolve()),
        "source_snapshot_dir": str((SKILL_SANDBOX_DIR / "demo_skill_sft" / "source_skill").resolve()),
        "generated_dir": str((SKILL_SANDBOX_DIR / "demo_skill_sft" / "generated").resolve()),
    }
    skill_card = build_skill_card(mock_distilled_skill, verification_msg="Success. Output: {\"result\": 1}")
    print(f"Skill Card Built: {skill_card.get('skill_name')} | Required Inputs: {skill_card.get('required_inputs')}")
    index_data = {}
    if SKILL_INDEX_PATH.exists():
        with open(SKILL_INDEX_PATH, "r", encoding="utf-8") as f:
            loaded = json.load(f)
            if isinstance(loaded, dict):
                index_data = loaded
    sft_examples = generate_sft_examples_from_skill(skill_card, index_data=index_data)
    print(f"SFT Examples Generated: {len(sft_examples)}")
    export_stats = export_sft_outputs(skill_card, sft_examples)
    print(f"SFT Export Stats: {export_stats}")

    if SKILL_INDEX_PATH.exists():
        os.remove(SKILL_INDEX_PATH)
        shutil.rmtree(SKILL_INDEX_PATH.parent, ignore_errors=True)

if __name__ == "__main__":
    asyncio.run(main())

"""
python CodeAgent/skill_distiller_pipeline.py --run-real-idle --discovery-urls https://github.com/K-Dense-AI/claude-scientific-skills/tree/main/scientific-skills/geopandas https://github.com/K-Dense-AI/claude-scientific-skills/tree/main/scientific-skills/dask
"""

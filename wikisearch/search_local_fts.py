#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
search_local_fts.py (enhanced)

A local, data-driven "search layer" for Wikimedia dumps indexed into SQLite + FTS5.

You already built (or will build) these local databases:
- zhwiki_fts.sqlite        : Chinese Wikipedia (explanations / encyclopedic entries)
- enwiki_fts.sqlite        : English Wikipedia (explanations / encyclopedic entries)
- zhwikisource_fts.sqlite  : Chinese Wikisource (primary texts / books / classics)
- enwikisource_fts.sqlite  : English Wikisource (primary texts / books / classics)

This script provides:
1) Schema inspection utilities (prints the SQLite schema and validates tables/FTS triggers)
2) A unified_search(...) function with rich configuration:
   - choose sources: explain vs primary_text vs images
   - choose language: zh, en, both
   - choose providers explicitly
   - book/work discovery helpers (title-first search) to support "download/book view" UX later
   - consistent return structure designed for later AI calling

IMPORTANT NOTES ABOUT THE LOCAL DB SCHEMA
-----------------------------------------
These DBs are created by build_wikimedia_fts_multi*.py scripts (v2/v3). They all share the same schema:

Tables:
  pages
    - id           INTEGER PRIMARY KEY AUTOINCREMENT
    - project      TEXT   NOT NULL    (e.g. "zhwiki", "enwiki", "zhwikisource", "enwikisource")
    - ns           INTEGER NOT NULL   (namespace; 0 is main/article namespace)
    - title        TEXT   NOT NULL    (page title)
    - url          TEXT   NOT NULL    (full URL like https://zh.wikipedia.org/wiki/...)
    - updated_at   TEXT            (revision timestamp as ISO string, e.g. 2024-01-01T00:00:00Z)
    - raw_wikitext TEXT            (original wikitext; large)
    - clean_text   TEXT            (cleaned plain text for search/RAG)
    - snippet      TEXT            (short preview; derived from clean_text)

  pages_fts  (FTS5 virtual table)
    - title        (indexed)
    - clean_text   (indexed)
    - uses external content='pages', content_rowid='id'
    - query with: WHERE pages_fts MATCH '...'

Indices:
  - uq_pages_project_ns_title unique index for idempotent reruns
  - idx_pages_project_ns, idx_pages_project_title

FTS triggers:
  - pages_ai: after insert into pages -> insert into pages_fts
  - pages_au/ad: keep in sync on update/delete

Typical search query pattern:
  SELECT p.title, p.url, snippet(pages_fts, 1, '[', ']', '…', 12), bm25(pages_fts)
  FROM pages_fts JOIN pages p ON p.id = pages_fts.rowid
  WHERE pages_fts MATCH ?
  ORDER BY bm25(pages_fts)
  LIMIT ?;

FTS5 + Chinese:
  - With tokenize='unicode61', Chinese isn't segmented like Jieba.
  - Best practice: use short keywords (2-4 chars), use AND, or use quoted phrases "..."
  - For very short single-character queries, results may be noisy.

RETURN FORMAT
-------------
Every search result item is a dict:

{
  "provider": "zhwiki" | "enwiki" | "zhwikisource" | "enwikisource" | "commons",
  "kind": "explain" | "primary_text" | "image" | "book",
  "title": "...",
  "url": "...",
  "snippet": "...",     # snippet from FTS, with highlights [like this]
  "rank": float,        # bm25 score (lower is better for sqlite bm25)
  "source_db": "/path/to/db.sqlite",
  "project": "...",     # equals provider for local db
  "ns": 0,              # namespace
  "meta": { ... }       # extra info (optional)
}

CONFIGURATION OVERVIEW
----------------------
Use UnifiedSearchConfig to control:
- languages: "zh", "en", "both"
- include_kinds: set of {"explain", "primary_text", "image", "book"}
- providers: explicit list (overrides language/kinds mapping)
- limits per kind/provider
- query_mode: "fts" (full-text) or "title" (title-only) or "hybrid"
- book_mode: "discover" (find likely work pages) and/or "chapters" (find passages)
- commons_images: placeholder structure (API not implemented here)

"""

from __future__ import annotations

import os
import sqlite3
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Literal, Set, Any, Tuple
from dataclasses import asdict

# DEFAULT_DB_PATHS: Dict[str, str] = {
#     "zhwiki": os.environ.get("ZHWIKI_DB", "zhwiki_fts.sqlite"),
#     "enwiki": os.environ.get("ENWIKI_DB", "enwiki_fts.sqlite"),
#     "zhwikisource": os.environ.get("ZHWIKISOURCE_DB", "zhwikisource_fts.sqlite"),
#     "enwikisource": os.environ.get("ENWIKISOURCE_DB", "enwikisource_fts.sqlite"),
# }
WIKI_DB_DIR = os.environ.get("WIKI_DB_DIR", "/data/rnd-liu/Datasets/wikidata/")
DEFAULT_DB_PATHS: Dict[str, str] = {
    "zhwiki": os.path.join(WIKI_DB_DIR, "zhwiki_fts.sqlite"),
    "enwiki": os.path.join(WIKI_DB_DIR, "enwiki_fts.sqlite"),
    "zhwikisource": os.path.join(WIKI_DB_DIR, "zhwikisource_fts.sqlite"),
    "enwikisource": os.path.join(WIKI_DB_DIR, "enwikisource_fts.sqlite"),
}


Provider = Literal["zhwiki", "enwiki", "zhwikisource", "enwikisource"]
Kind = Literal["explain", "primary_text", "image", "book"]
Language = Literal["zh", "en", "both"]
QueryMode = Literal["fts", "title", "hybrid"]


@dataclass
class ProviderConfig:
    """Per-provider knobs."""
    provider: Provider
    db_path: str
    enabled: bool = True
    limit: int = 10
    namespaces: Optional[Set[int]] = None  # None => all


@dataclass
class UnifiedSearchConfig:
    """
    Main config for unified_search.

    language:
      - "zh": only zhwiki + zhwikisource by default
      - "en": only enwiki + enwikisource by default
      - "both": all four

    include_kinds:
      - "explain": Wikipedia-style explanation pages
      - "primary_text": Wikisource passages/pages
      - "book": title-first "work discovery" (e.g. find the main "史记" / "资治通鉴" page)
      - "image": placeholder; integrate Commons API later

    query_mode:
      - "fts": use pages_fts MATCH for text search
      - "title": title-only via title:... or LIKE
      - "hybrid": title discovery + fts passages (recommended for books)
    """
    language: Language = "zh"
    include_kinds: Set[Kind] = field(default_factory=lambda: {"explain", "primary_text"})
    query_mode: QueryMode = "fts"

    providers: Optional[List[Provider]] = None

    limit_explain: int = 10
    limit_primary: int = 10
    limit_book: int = 8
    limit_image: int = 0

    book_discover: bool = True
    book_discover_title_only: bool = True
    book_discover_phrase: bool = True

    primary_prefer_phrase: bool = False

    provider_overrides: Dict[Provider, ProviderConfig] = field(default_factory=dict)

    include_clean_text: bool = False
    max_clean_text_chars: int = 4000

    include_images: bool = False
    commons_query: Optional[str] = None
    commons_limit: int = 6

    dedupe_by_url: bool = True
    
    def to_dict(self):
        # asdict() recursively handles nested dataclasses, lists, dicts, and sets
        d = asdict(self)
        # JSON doesn't support sets, so convert them to lists
        d['include_kinds'] = list(self.include_kinds)
        return d


def _connect(db_path: str) -> sqlite3.Connection:
    return sqlite3.connect(db_path)


def inspect_schema(db_path: str) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "db_path": db_path,
        "tables": {},
        "indices": [],
        "triggers": [],
        "ok": True,
        "errors": [],
    }
    if not os.path.exists(db_path):
        report["ok"] = False
        report["errors"].append(f"DB not found: {db_path}")
        return report

    conn = _connect(db_path)
    cur = conn.cursor()

    def q(sql: str, params: Tuple = ()):
        cur.execute(sql, params)
        return cur.fetchall()

    tables = q("SELECT name, type, sql FROM sqlite_master WHERE type IN ('table','view') ORDER BY name;")
    for name, typ, sql in tables:
        if name.startswith("sqlite_"):
            continue
        report["tables"].setdefault(name, {"type": typ, "sql": sql, "columns": []})
        try:
            cols = q(f"PRAGMA table_info({name});")
            report["tables"][name]["columns"] = [{"cid": c[0], "name": c[1], "type": c[2], "notnull": c[3], "dflt": c[4], "pk": c[5]} for c in cols]
        except Exception as e:
            report["errors"].append(f"Failed PRAGMA table_info({name}): {e}")
            report["ok"] = False

    idx = q("SELECT name, tbl_name, sql FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%' ORDER BY name;")
    report["indices"] = [{"name": n, "table": t, "sql": s} for n, t, s in idx]

    trg = q("SELECT name, tbl_name, sql FROM sqlite_master WHERE type='trigger' ORDER BY name;")
    report["triggers"] = [{"name": n, "table": t, "sql": s} for n, t, s in trg]

    expected_tables = {"pages", "pages_fts"}
    missing = [t for t in expected_tables if t not in report["tables"]]
    if missing:
        report["ok"] = False
        report["errors"].append(f"Missing expected tables: {missing}")

    conn.close()
    return report


def print_schema_report(report: Dict[str, Any], max_sql_chars: int = 400) -> None:
    print("=" * 80)
    print(f"DB: {report['db_path']}")
    print(f"OK: {report['ok']}")
    if report["errors"]:
        print("Errors:")
        for e in report["errors"]:
            print(f"  - {e}")

    print("\nTables:")
    for tname, tinfo in report["tables"].items():
        print(f"  - {tname} ({tinfo.get('type')})")
        cols = tinfo.get("columns") or []
        if cols:
            for c in cols:
                print(f"      {c['name']:<12} {c['type']:<10} pk={c['pk']} notnull={c['notnull']}")
        sql = (tinfo.get("sql") or "").strip().replace("\n", " ")
        if sql:
            print(f"      sql: {sql[:max_sql_chars]}{'...' if len(sql) > max_sql_chars else ''}")

    print("\nIndices:")
    for idx in report["indices"]:
        print(f"  - {idx['name']} on {idx['table']}")

    print("\nTriggers:")
    for trg in report["triggers"]:
        print(f"  - {trg['name']} on {trg['table']}")
    print("=" * 80)


def _fts_query(
    db_path: str,
    match_query: str,
    limit: int,
    namespaces: Optional[Set[int]] = None,
    include_clean_text: bool = False,
    max_clean_text_chars: int = 4000,
) -> List[Dict[str, Any]]:
    conn = _connect(db_path)
    cur = conn.cursor()

    base_sql = """
    SELECT
      p.project,
      p.ns,
      p.title,
      p.url,
      snippet(pages_fts, 1, '[', ']', '…', 12) AS snip,
      bm25(pages_fts) AS rank,
      p.clean_text
    FROM pages_fts
    JOIN pages p ON p.id = pages_fts.rowid
    WHERE pages_fts MATCH ?
    """
    params: List[Any] = [match_query]

    if namespaces is not None:
        placeholders = ",".join(["?"] * len(namespaces))
        base_sql += f" AND p.ns IN ({placeholders})"
        params.extend(sorted(list(namespaces)))

    base_sql += " ORDER BY rank LIMIT ?;"
    params.append(limit)

    cur.execute(base_sql, params)
    rows = cur.fetchall()
    conn.close()

    out: List[Dict[str, Any]] = []
    for project, ns, title, url, snip, rank, clean_text in rows:
        item = {
            "provider": project,
            "project": project,
            "ns": ns,
            "title": title,
            "url": url,
            "snippet": snip or "",
            "rank": float(rank) if rank is not None else None,
            "source_db": db_path,
            "meta": {},
        }
        if include_clean_text:
            item["meta"]["clean_text"] = (clean_text or "")[:max_clean_text_chars]
        out.append(item)
    return out


def _title_discover(
    db_path: str,
    title_phrase: str,
    limit: int,
    namespaces: Optional[Set[int]] = None,
) -> List[Dict[str, Any]]:
    match_q = f'title:"{title_phrase}"' if (" " in title_phrase or len(title_phrase) >= 2) else f"title:{title_phrase}"
    try:
        return _fts_query(db_path, match_q, limit, namespaces=namespaces, include_clean_text=False)
    except Exception:
        pass

    conn = _connect(db_path)
    cur = conn.cursor()
    sql = "SELECT project, ns, title, url, snippet, 0.0 as rank, clean_text FROM pages WHERE title LIKE ?"
    params: List[Any] = [f"%{title_phrase}%"]
    if namespaces is not None:
        placeholders = ",".join(["?"] * len(namespaces))
        sql += f" AND ns IN ({placeholders})"
        params.extend(sorted(list(namespaces)))
    sql += " LIMIT ?"
    params.append(limit)
    cur.execute(sql, params)
    rows = cur.fetchall()
    conn.close()

    out = []
    for project, ns, title, url, snippet, rank, clean_text in rows:
        out.append({
            "provider": project,
            "project": project,
            "ns": ns,
            "title": title,
            "url": url,
            "snippet": snippet or "",
            "rank": float(rank),
            "source_db": db_path,
            "meta": {},
        })
    return out


def _default_providers_for_language(language: Language) -> List[Provider]:
    if language == "zh":
        return ["zhwiki", "zhwikisource"]
    if language == "en":
        return ["enwiki", "enwikisource"]
    return ["zhwiki", "zhwikisource", "enwiki", "enwikisource"]


def unified_search(query: str, config: Optional[UnifiedSearchConfig] = None) -> Dict[str, Any]:
    """
    Unified search across multiple local SQLite FTS DBs.

    Returns:
      {
        "query": str,
        "config": dict,
        "groups": { "explain": [...], "primary_text": [...], "book": [...], "image": [...] },
        "all": [...]
      }

    How to read results:
      - rank: SQLite FTS5 bm25 score; smaller = better.
      - snippet: contains [highlighted] terms (good for UI previews).
      - provider/project: which dataset produced the result.
      - kind: explain vs primary_text vs book vs image.
    """
    
    cfg = config or UnifiedSearchConfig()
    providers: List[Provider] = cfg.providers if cfg.providers else _default_providers_for_language(cfg.language)

    provider_configs: Dict[Provider, ProviderConfig] = {}
    for p in providers:
        provider_configs[p] = cfg.provider_overrides.get(p, ProviderConfig(provider=p, db_path=DEFAULT_DB_PATHS[p]))

    groups: Dict[str, List[Dict[str, Any]]] = {"explain": [], "primary_text": [], "book": [], "image": []}

    explain_providers: List[Provider] = [p for p in providers if p in ("zhwiki", "enwiki")]
    primary_providers: List[Provider] = [p for p in providers if p in ("zhwikisource", "enwikisource")]

    def fts_query_for_primary(q: str) -> str:
        if cfg.primary_prefer_phrase and len(q) >= 2 and '"' not in q:
            return f'"{q}"'
        return q

    if "explain" in cfg.include_kinds:
        for p in explain_providers:
            pc = provider_configs[p]
            if not pc.enabled:
                continue
            q = query if cfg.query_mode in ("fts", "hybrid") else f'title:"{query}"'
            res = _fts_query(
                pc.db_path, q, limit=cfg.limit_explain,
                namespaces=pc.namespaces, include_clean_text=cfg.include_clean_text,
                max_clean_text_chars=cfg.max_clean_text_chars
            )
            for it in res:
                it["kind"] = "explain"
            groups["explain"].extend(res)

    if "primary_text" in cfg.include_kinds:
        for p in primary_providers:
            pc = provider_configs[p]
            if not pc.enabled:
                continue
            q = fts_query_for_primary(query) if cfg.query_mode in ("fts", "hybrid") else f'title:"{query}"'
            res = _fts_query(
                pc.db_path, q, limit=cfg.limit_primary,
                namespaces=pc.namespaces, include_clean_text=cfg.include_clean_text,
                max_clean_text_chars=cfg.max_clean_text_chars
            )
            for it in res:
                it["kind"] = "primary_text"
            groups["primary_text"].extend(res)

    if "book" in cfg.include_kinds and cfg.book_discover:
        for p in primary_providers:
            pc = provider_configs[p]
            if not pc.enabled:
                continue
            res = _title_discover(pc.db_path, title_phrase=query, limit=cfg.limit_book, namespaces=pc.namespaces)
            for it in res:
                it["kind"] = "book"
                it["meta"]["hint"] = "Title-first discovery; treat as work/chapter entry."
            groups["book"].extend(res)

    if "image" in cfg.include_kinds or cfg.include_images:
        q = cfg.commons_query or query
        groups["image"] = [{
            "provider": "commons",
            "kind": "image",
            "title": f"(placeholder) Commons image search not implemented yet for query: {q}",
            "url": "",
            "snippet": "",
            "rank": None,
            "source_db": None,
            "project": "commons",
            "ns": None,
            "meta": {"todo": "Integrate Wikimedia Commons API and store license metadata."}
        }][: max(cfg.limit_image, 1)]

    all_items: List[Dict[str, Any]] = groups["explain"] + groups["book"] + groups["primary_text"] + groups["image"]
    if cfg.dedupe_by_url:
        seen = set()
        deduped = []
        for it in all_items:
            u = it.get("url") or (it.get("provider"), it.get("title"))
            if u in seen:
                continue
            seen.add(u)
            deduped.append(it)
        all_items = deduped

    return {
        "query": query,
        #"config": json.loads(json.dumps(cfg, default=lambda o: list(o) if isinstance(o, set) else o.__dict__, ensure_ascii=False)),
        "config": cfg.to_dict(),
        "groups": groups,
        "all": all_items,
    }


def _safe_db_exists(db_path: str) -> bool:
    return os.path.exists(db_path) and os.path.getsize(db_path) > 0


def run_examples():
    """
    Examples / tests.

    NOTE:
    - If you only built zhwiki_fts.sqlite so far, the Wikisource queries will return empty.
    - The examples still demonstrate config usage and return format.
    """
    print("\n\n### SCHEMA INSPECTION ###")
    for prov, db in DEFAULT_DB_PATHS.items():
        if _safe_db_exists(db):
            rep = inspect_schema(db)
            print_schema_report(rep)
        else:
            print(f"[SKIP] DB missing or empty: {prov} -> {db}")

    print("\n\n### EXAMPLE 1: default unified_search (zh, explain+primary) ###")
    cfg1 = UnifiedSearchConfig(language="zh", include_kinds={"explain", "primary_text"}, query_mode="fts",
                               limit_explain=5, limit_primary=5)
    r1 = unified_search('唐 AND 太宗', cfg1)
    print(json.dumps(r1["groups"], ensure_ascii=False, indent=2)[:4000] + "\n...")

    print("\n\n### EXAMPLE 2: explain-only (zhwiki only) ###")
    cfg2 = UnifiedSearchConfig(language="zh", include_kinds={"explain"}, query_mode="fts", limit_explain=8)
    r2 = unified_search('资治通鉴', cfg2)
    for it in r2["groups"]["explain"][:8]:
        print(f"- {it['title']}  ({it['url']})")

    print("\n\n### EXAMPLE 3: primary_text-only (zhwikisource) ###")
    cfg3 = UnifiedSearchConfig(language="zh", include_kinds={"primary_text"}, query_mode="fts", limit_primary=8,
                               primary_prefer_phrase=True)
    r3 = unified_search('太史公曰', cfg3)
    for it in r3["groups"]["primary_text"][:8]:
        print(f"- {it['title']}  rank={it['rank']}\n  {it['snippet']}\n  {it['url']}")

    print("\n\n### EXAMPLE 4: book discovery (title-first) ###")
    cfg4 = UnifiedSearchConfig(language="zh", include_kinds={"book"}, query_mode="hybrid", limit_book=10)
    r4 = unified_search('史记', cfg4)
    for it in r4["groups"]["book"][:10]:
        print(f"- {it['title']}  ({it['url']})  hint={it['meta'].get('hint')}")

    print("\n\n### EXAMPLE 5: English explain + primary_text ###")
    cfg5 = UnifiedSearchConfig(language="en", include_kinds={"explain", "primary_text"}, query_mode="fts",
                               limit_explain=5, limit_primary=5)
    r5 = unified_search('"Zizhi Tongjian"', cfg5)
    print({k: len(v) for k, v in r5["groups"].items()})

    print("\n\n### EXAMPLE 6: both languages explain-only ###")
    cfg6 = UnifiedSearchConfig(language="both", include_kinds={"explain"}, query_mode="fts", limit_explain=6)
    r6 = unified_search('Silk Road OR 丝绸之路', cfg6)
    for it in r6["groups"]["explain"][:12]:
        print(f"[{it['provider']}] {it['title']} -> {it['url']}")


def main():
    import argparse as _argparse
    ap = _argparse.ArgumentParser(description="Local FTS5 search layer for Wikimedia dump indexes.")
    ap.add_argument("--inspect", action="store_true", help="Inspect schema of known DBs and exit.")
    ap.add_argument("--query", type=str, default="", help="Run unified_search with a simple default config.")
    ap.add_argument("--language", choices=["zh", "en", "both"], default="zh")
    ap.add_argument("--kinds", type=str, default="explain,primary_text", help="Comma-separated kinds, e.g. explain,book")
    ap.add_argument("--mode", choices=["fts", "title", "hybrid"], default="fts")
    ap.add_argument("--examples", action="store_true", help="Run built-in examples/tests.")
    args = ap.parse_args()

    if args.inspect:
        for prov, db in DEFAULT_DB_PATHS.items():
            rep = inspect_schema(db)
            print_schema_report(rep)
        return

    if args.examples:
        run_examples()
        return

    if args.query.strip():
        kinds = {k.strip() for k in args.kinds.split(",") if k.strip()}
        cfg = UnifiedSearchConfig(language=args.language, include_kinds=kinds, query_mode=args.mode,
                                  limit_explain=10, limit_primary=10, limit_book=10)
        r = unified_search(args.query.strip(), cfg)
        print(json.dumps(r, ensure_ascii=False, indent=2)[:8000])
        return

    print("Nothing to do. Try --examples or --inspect or --query '...'.")


if __name__ == "__main__":
    main()

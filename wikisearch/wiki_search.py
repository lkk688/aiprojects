#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
enwiki_search_api.py

A FastAPI wrapper around an SQLite FTS (FTS5/FTS4) database built from Wikipedia.

Features:
- /healthz: process liveness
- /readyz: DB connectivity + lightweight query readiness
- /search?q=...&k=...: FTS search with optional snippet/highlight

Usage:
  pip install fastapi uvicorn pydantic
  python enwiki_search_api.py --db /path/to/enwiki.db --host 0.0.0.0 --port 8009

Then:
  curl 'http://localhost:8009/search?q=Qwen3%20MoE&k=5'
"""

from __future__ import annotations

import argparse
import os
import re
import sqlite3
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field


# =========================
# Models
# =========================

class SearchHit(BaseModel):
    rank: int = Field(..., description="1-based rank in results")
    score: Optional[float] = Field(None, description="FTS score if available (bm25/rank), otherwise null")
    doc_id: Optional[str] = Field(None, description="Document/page identifier if available")
    title: Optional[str] = Field(None, description="Title if available")
    snippet: Optional[str] = Field(None, description="Highlighted snippet if available")
    text: Optional[str] = Field(None, description="Raw text (optional, may be large)")
    extra: Dict[str, Any] = Field(default_factory=dict, description="Other columns if returned")


class SearchResponse(BaseModel):
    query: str
    k: int
    took_ms: float
    fts_table: str
    hits: List[SearchHit]


# =========================
# SQLite helpers
# =========================

@dataclass
class FtsInfo:
    fts_table: str
    # core columns we try to read
    text_col: str
    title_col: Optional[str]
    id_col: Optional[str]
    # whether bm25() exists (FTS5)
    has_bm25: bool


def _connect_sqlite(db_path: str) -> sqlite3.Connection:
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"SQLite DB not found: {db_path}")

    # per-request connection: safe for multi-threaded ASGI
    con = sqlite3.connect(db_path, check_same_thread=False)
    con.row_factory = sqlite3.Row

    # safer concurrent reads
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    con.execute("PRAGMA temp_store=MEMORY;")
    con.execute("PRAGMA cache_size=-20000;")  # ~20MB
    con.execute("PRAGMA busy_timeout=5000;")
    return con


def _list_tables(con: sqlite3.Connection) -> List[str]:
    rows = con.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table','view') ORDER BY name"
    ).fetchall()
    return [r["name"] for r in rows]


def _is_fts_table(con: sqlite3.Connection, name: str) -> bool:
    # FTS5 tables appear as normal tables with special shadow tables: name_data/name_idx/name_docsize/name_config
    # FTS4 has name_content/name_segments/name_segdir
    # We'll detect by presence of shadow tables.
    tables = set(_list_tables(con))
    fts5_shadows = {f"{name}_data", f"{name}_idx", f"{name}_docsize", f"{name}_config"}
    fts4_shadows = {f"{name}_content", f"{name}_segments", f"{name}_segdir"}
    return (len(fts5_shadows & tables) >= 2) or (len(fts4_shadows & tables) >= 2)


def _table_columns(con: sqlite3.Connection, table: str) -> List[str]:
    rows = con.execute(f"PRAGMA table_info({table});").fetchall()
    return [r["name"] for r in rows]


def _detect_bm25(con: sqlite3.Connection) -> bool:
    # bm25() exists in FTS5. We can try a harmless query.
    try:
        con.execute("SELECT bm25('x')").fetchone()
        return True
    except Exception:
        return False


def _guess_core_columns(cols: List[str]) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Heuristics to guess: text_col, title_col, id_col
    text: prefer 'text'/'body'/'content'/'paragraph'/'wiki_text'
    title: prefer 'title'/'page_title'/'doc_title'
    id: prefer 'id'/'doc_id'/'page_id'/'rowid'
    """
    lower = {c.lower(): c for c in cols}

    # text
    for key in ["text", "body", "content", "paragraph", "wiki_text", "passage", "chunk", "chunk_text"]:
        if key in lower:
            text_col = lower[key]
            break
    else:
        # fallback: first column (common in FTS5: first is usually the content)
        text_col = cols[0]

    # title
    title_col = None
    for key in ["title", "page_title", "doc_title", "name"]:
        if key in lower:
            title_col = lower[key]
            break

    # id
    id_col = None
    for key in ["doc_id", "page_id", "id", "article_id"]:
        if key in lower:
            id_col = lower[key]
            break

    # rowid is implicit; we can use it even if id_col missing
    return text_col, title_col, id_col


def detect_fts(con: sqlite3.Connection, prefer_name: Optional[str] = None) -> FtsInfo:
    tables = _list_tables(con)

    candidates = [t for t in tables if _is_fts_table(con, t)]
    if not candidates:
        raise RuntimeError(
            "No FTS tables detected. Please provide --fts-table explicitly, or verify your enwiki DB has FTS."
        )

    if prefer_name:
        if prefer_name not in candidates:
            raise RuntimeError(
                f"FTS table '{prefer_name}' not found among detected candidates: {candidates}"
            )
        fts_table = prefer_name
    else:
        # prefer a table name that looks like main index
        priority = ["enwiki", "wiki", "documents", "docs", "pages", "passages", "chunks", "fts"]
        fts_table = None
        for p in priority:
            for c in candidates:
                if p in c.lower():
                    fts_table = c
                    break
            if fts_table:
                break
        if not fts_table:
            fts_table = candidates[0]

    cols = _table_columns(con, fts_table)
    if not cols:
        raise RuntimeError(f"FTS table '{fts_table}' has no columns?")

    text_col, title_col, id_col = _guess_core_columns(cols)
    return FtsInfo(
        fts_table=fts_table,
        text_col=text_col,
        title_col=title_col,
        id_col=id_col,
        has_bm25=_detect_bm25(con),
    )


def _sanitize_fts_query(q: str) -> str:
    """
    Conservative sanitation for FTS MATCH:
    - strip control characters
    - collapse whitespace
    We do NOT attempt to escape all FTS operators; users may want quotes/NEAR/etc.
    """
    q = re.sub(r"[\x00-\x1f\x7f]", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q


def run_search(
    con: sqlite3.Connection,
    fts: FtsInfo,
    query: str,
    k: int,
    with_text: bool = False,
) -> List[sqlite3.Row]:
    q = _sanitize_fts_query(query)
    if not q:
        return []

    # snippet() exists for FTS5/FTS4 tables
    # Syntax differs slightly; this is broadly compatible:
    # snippet(table, -1, start, end, ellipsis, tokens)
    # If it errors, we will fallback to substr(text,...)
    snippet_expr = f"snippet({fts.fts_table}, -1, '[', ']', ' … ', 24) AS snippet"

    # bm25() only for FTS5; rank() for FTS4 often differs; we keep optional
    if fts.has_bm25:
        score_expr = f"bm25({fts.fts_table}) AS score"
        order_by = "score ASC"  # bm25 lower is better
    else:
        score_expr = "NULL AS score"
        # FTS4 often supports "ORDER BY rank" depending on custom rank; fallback to rowid
        order_by = "rowid DESC"

    select_cols = []
    if fts.id_col:
        select_cols.append(f"{fts.id_col} AS doc_id")
    else:
        select_cols.append("CAST(rowid AS TEXT) AS doc_id")

    if fts.title_col:
        select_cols.append(f"{fts.title_col} AS title")
    else:
        select_cols.append("NULL AS title")

    select_cols.append(snippet_expr)
    if with_text:
        select_cols.append(f"{fts.text_col} AS text")
    else:
        select_cols.append("NULL AS text")

    select_cols.append(score_expr)

    sql = f"""
    SELECT
      {", ".join(select_cols)}
    FROM {fts.fts_table}
    WHERE {fts.fts_table} MATCH ?
    ORDER BY {order_by}
    LIMIT ?
    """

    try:
        rows = con.execute(sql, (q, k)).fetchall()
        return rows
    except sqlite3.OperationalError as e:
        # Fallback snippet if snippet() not supported
        if "snippet" in str(e).lower():
            sql2 = f"""
            SELECT
              {fts.id_col + " AS doc_id" if fts.id_col else "CAST(rowid AS TEXT) AS doc_id"},
              {fts.title_col + " AS title" if fts.title_col else "NULL AS title"},
              substr({fts.text_col}, 1, 320) AS snippet,
              {"(" + fts.text_col + ") AS text" if with_text else "NULL AS text"},
              {score_expr}
            FROM {fts.fts_table}
            WHERE {fts.fts_table} MATCH ?
            ORDER BY {order_by}
            LIMIT ?
            """
            return con.execute(sql2, (q, k)).fetchall()
        raise


# =========================
# FastAPI app
# =========================

def create_app(db_path: str, fts_table: Optional[str] = None) -> FastAPI:
    app = FastAPI(title="enwiki-search-api", version="0.1.0")

    # Cache detected FTS info at startup (for fast requests)
    state: Dict[str, Any] = {
        "db_path": db_path,
        "fts_table": fts_table,
        "fts_info": None,
        "startup_error": None,
    }

    @app.on_event("startup")
    def _startup() -> None:
        try:
            con = _connect_sqlite(db_path)
            try:
                state["fts_info"] = detect_fts(con, prefer_name=fts_table)
            finally:
                con.close()
        except Exception as e:
            state["startup_error"] = str(e)

    @app.get("/healthz")
    def healthz() -> Dict[str, Any]:
        return {"ok": True, "service": "enwiki-search-api"}

    @app.get("/readyz")
    def readyz() -> Dict[str, Any]:
        if state["startup_error"]:
            return {"ready": False, "error": state["startup_error"]}
        if not state["fts_info"]:
            return {"ready": False, "error": "FTS info not initialized"}
        # Run a tiny query to verify DB reads are OK
        try:
            con = _connect_sqlite(state["db_path"])
            try:
                # some FTS implementations error on empty match; do a simple term
                _ = run_search(con, state["fts_info"], "the", 1, with_text=False)
            finally:
                con.close()
            return {"ready": True, "fts_table": state["fts_info"].fts_table}
        except Exception as e:
            return {"ready": False, "error": str(e)}

    @app.get("/search", response_model=SearchResponse)
    def search(
        q: str = Query(..., min_length=1, description="FTS query string"),
        k: int = Query(10, ge=1, le=50, description="Number of results to return"),
        with_text: bool = Query(False, description="Include full text field in response (may be large)"),
    ) -> SearchResponse:
        if state["startup_error"]:
            raise HTTPException(status_code=500, detail=state["startup_error"])
        fts_info: FtsInfo = state["fts_info"]
        if not fts_info:
            raise HTTPException(status_code=500, detail="FTS info not initialized")

        t0 = time.time()
        con = _connect_sqlite(state["db_path"])
        try:
            rows = run_search(con, fts_info, q, k, with_text=with_text)
        except sqlite3.OperationalError as e:
            raise HTTPException(status_code=400, detail=f"FTS query failed: {e}")
        finally:
            con.close()

        hits: List[SearchHit] = []
        for i, r in enumerate(rows, start=1):
            hits.append(
                SearchHit(
                    rank=i,
                    score=r["score"] if "score" in r.keys() else None,
                    doc_id=r["doc_id"] if "doc_id" in r.keys() else None,
                    title=r["title"] if "title" in r.keys() else None,
                    snippet=r["snippet"] if "snippet" in r.keys() else None,
                    text=r["text"] if "text" in r.keys() else None,
                    extra={k2: r[k2] for k2 in r.keys() if k2 not in {"doc_id", "title", "snippet", "text", "score"}},
                )
            )

        return SearchResponse(
            query=q,
            k=k,
            took_ms=(time.time() - t0) * 1000.0,
            fts_table=fts_info.fts_table,
            hits=hits,
        )

    return app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="/data/rnd-liu/Datasets/wikidata/enwiki_fts.sqlite", help="Path to enwiki SQLite DB")
    ap.add_argument("--fts-table", default=None, help="Explicit FTS table name (optional)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", default=8009, type=int)
    args = ap.parse_args()

    import uvicorn

    app = create_app(args.db, fts_table=args.fts_table)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
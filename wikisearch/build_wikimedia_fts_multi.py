#!/usr/bin/env python3
"""
Wikimedia Dump -> SQLite FTS5 Index Builder (multi-project)

Supports building indexes for 4 dumps in one run:
- zhwiki (Chinese Wikipedia)         : explanations
- zhwikisource (Chinese Wikisource)  : original texts
- enwiki (English Wikipedia)         : explanations
- enwikisource (English Wikisource)  : original texts

What it does (per project):
- Streams XML dump (.xml or .xml.bz2) without loading into RAM
- Extracts: title, namespace, revision timestamp, revision text (wikitext)
- Cleans wikitext to plain text (basic regex cleaner or optional mwparserfromhell)
- Writes:
    pages(id, ns, title, url, updated_at, raw_wikitext, clean_text, snippet, project)
    pages_fts (FTS5 virtual table over title + clean_text) with BM25 ranking

You can run:
- single project:  --project zhwiki
- multi projects:  --projects zhwiki zhwikisource enwiki enwikisource
- config file:     --config projects.json

USAGE QUICK START (recommended):
1) pip install mwparserfromhell  (strongly recommended for Wikipedia)
2) Prepare dumps (pages-articles.xml.bz2) for each project
3) Create a config JSON (example below) and run:

    python build_wikimedia_fts_multi.py --config projects.json

Then query with sqlite:
    sqlite3 ./data/indexes/zhwiki_fts.sqlite
    SELECT title, url, snippet(pages_fts, 1, '[', ']', '…', 12)
    FROM pages_fts JOIN pages p ON p.id = pages_fts.rowid
    WHERE pages_fts MATCH '唐 AND 壁画'
    ORDER BY bm25(pages_fts)
    LIMIT 10;

NOTE:
- enwiki is enormous. Start with --max-pages to test, then run on a strong machine + SSD.
"""

import argparse
import bz2
import contextlib
import json
import os
import re
import sqlite3
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


# -----------------------------
# Project config
# -----------------------------
@dataclass
class ProjectSpec:
    project: str                  # "zhwiki" / "enwiki" / "zhwikisource" / "enwikisource"
    dump_path: str                # path to pages-articles.xml(.bz2)
    db_path: str                  # output sqlite path
    base_url: str                 # like "https://zh.wikipedia.org/wiki/"
    namespaces: Optional[Set[int]]  # None => all
    mode: str = "wiki"            # "wiki" or "wikisource" (affects default cleaning heuristics)


DEFAULT_BASE_URL = {
    "zhwiki": "https://zh.wikipedia.org/wiki/",
    "enwiki": "https://en.wikipedia.org/wiki/",
    "zhwikisource": "https://zh.wikisource.org/wiki/",
    "enwikisource": "https://en.wikisource.org/wiki/",
}


# -----------------------------
# Wikitext cleaning
# -----------------------------
RE_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
RE_REF_TAG = re.compile(r"<ref\b[^>]*>.*?</ref\s*>", re.IGNORECASE | re.DOTALL)
RE_REF_SELF = re.compile(r"<ref\b[^>]*/\s*>", re.IGNORECASE)
RE_TABLE = re.compile(r"\{\|.*?\|\}", re.DOTALL)
RE_FILE_LINK = re.compile(r"\[\[(?:File|Image|文件|圖像|图像):[^\]]+\]\]", re.IGNORECASE)
RE_CATEGORY = re.compile(r"\[\[(?:Category|分类|分類):[^\]]+\]\]", re.IGNORECASE)
RE_TEMPLATE = re.compile(r"\{\{[^{}]*\}\}")  # shallow templates; repeated loop
RE_TAGS = re.compile(
    r"</?(?:br|p|div|span|small|big|center|blockquote|poem|nowiki|code|pre|sup|sub)\b[^>]*>",
    re.IGNORECASE,
)
RE_HTML = re.compile(r"<[^>]+>")  # last resort strip
RE_EXT_LINK = re.compile(r"\[(https?://[^\s\]]+)\s*([^\]]*)\]")
RE_WIKI_LINK = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]+)?(?:\|([^\]]+))?\]\]")
RE_BOLD_ITALIC = re.compile(r"'''''|'''|''")
RE_HEADINGS = re.compile(r"^={2,}\s*(.*?)\s*={2,}\s*$", re.MULTILINE)
RE_LISTMARK = re.compile(r"^\s*[*#:;]+\s*", re.MULTILINE)
RE_MULTISPACE = re.compile(r"[ \t]+")
RE_MULTINEWLINE = re.compile(r"\n{3,}")


# Wikipedia: drop common trailing sections (best-effort)
WIKI_DROP_SECTION_TITLES = {
    "references", "external links", "see also", "further reading", "notes", "bibliography",
    "参考文献", "外部链接", "参见", "延伸阅读", "注释", "参考資料", "參考文獻", "外部連結"
}


def make_page_url(base_url: str, title: str) -> str:
    safe = title.replace(" ", "_")
    return base_url + safe


def clean_wikitext_basic(text: str, mode: str, max_len: int = 800_000) -> str:
    if not text:
        return ""
    if len(text) > max_len:
        text = text[:max_len]

    s = text
    s = RE_COMMENT.sub(" ", s)
    s = RE_REF_TAG.sub(" ", s)
    s = RE_REF_SELF.sub(" ", s)
    s = RE_TABLE.sub(" ", s)

    # For both modes: remove file/category clutter
    s = RE_FILE_LINK.sub(" ", s)
    s = RE_CATEGORY.sub(" ", s)

    # Remove templates iteratively (shallow)
    for _ in range(8):
        new = RE_TEMPLATE.sub(" ", s)
        if new == s:
            break
        s = new

    # Headings => plain line
    s = RE_HEADINGS.sub(r"\n\1\n", s)

    s = RE_TAGS.sub(" ", s)
    s = RE_HTML.sub(" ", s)

    # [url label] -> label
    def _ext_link(m):
        url = m.group(1) or ""
        label = (m.group(2) or "").strip()
        return label if label else url

    s = RE_EXT_LINK.sub(_ext_link, s)

    # [[target|label]] -> label, [[target]] -> target
    def _wiki_link(m):
        target = (m.group(1) or "").strip()
        label = (m.group(2) or "").strip()
        return label if label else target

    s = RE_WIKI_LINK.sub(_wiki_link, s)

    s = RE_BOLD_ITALIC.sub("", s)
    s = RE_LISTMARK.sub("", s)

    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = RE_MULTISPACE.sub(" ", s)
    s = RE_MULTINEWLINE.sub("\n\n", s)
    s = s.strip()

    # Best-effort Wikipedia-only: remove trailing reference-ish sections by heading titles
    if mode == "wiki" and s:
        s = drop_wikipedia_trailing_sections(s)

    return s.strip()


def drop_wikipedia_trailing_sections(clean_text: str) -> str:
    """
    After headings have been converted to plain lines, we can try to remove
    everything after a line that equals a known section title.
    This is heuristic and language-dependent.
    """
    lines = clean_text.splitlines()
    out: List[str] = []
    for line in lines:
        key = line.strip().lower()
        if key in WIKI_DROP_SECTION_TITLES:
            break
        out.append(line)
    return "\n".join(out).strip()


def clean_wikitext_mwparser(text: str, mode: str, max_len: int = 800_000) -> str:
    if not text:
        return ""
    if len(text) > max_len:
        text = text[:max_len]

    import mwparserfromhell  # type: ignore

    code = mwparserfromhell.parse(text)

    # Remove templates (better than regex)
    for tpl in code.filter_templates(recursive=True):
        try:
            code.remove(tpl)
        except Exception:
            pass

    s = code.strip_code(normalize=True, collapse=True)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = RE_MULTISPACE.sub(" ", s)
    s = RE_MULTINEWLINE.sub("\n\n", s)
    s = s.strip()

    if mode == "wiki" and s:
        s = drop_wikipedia_trailing_sections(s)

    return s.strip()


# -----------------------------
# Dump streaming
# -----------------------------
def open_dump(path: str):
    if path.endswith(".bz2"):
        return bz2.open(path, "rb")
    return open(path, "rb")


def strip_ns(tag: str) -> str:
    if tag.startswith("{"):
        return tag.split("}", 1)[1]
    return tag


def iter_pages(dump_path: str) -> Iterable[Tuple[str, int, str, str]]:
    with open_dump(dump_path) as f:
        context = ET.iterparse(f, events=("end",))
        for _, elem in context:
            if strip_ns(elem.tag) != "page":
                continue

            title = ""
            ns = -1
            timestamp = ""
            wikitext = ""

            for child in elem:
                ctag = strip_ns(child.tag)
                if ctag == "title":
                    title = child.text or ""
                elif ctag == "ns":
                    try:
                        ns = int((child.text or "").strip())
                    except Exception:
                        ns = -1
                elif ctag == "revision":
                    for rchild in child:
                        rtag = strip_ns(rchild.tag)
                        if rtag == "timestamp":
                            timestamp = rchild.text or ""
                        elif rtag == "text":
                            wikitext = rchild.text or ""

            yield (title, ns, timestamp, wikitext)
            elem.clear()


# -----------------------------
# SQLite schema
# -----------------------------
SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA temp_store=MEMORY;

CREATE TABLE IF NOT EXISTS pages (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  project TEXT NOT NULL,
  ns INTEGER NOT NULL,
  title TEXT NOT NULL,
  url TEXT NOT NULL,
  updated_at TEXT,
  raw_wikitext TEXT,
  clean_text TEXT,
  snippet TEXT
);

CREATE INDEX IF NOT EXISTS idx_pages_project_ns ON pages(project, ns);
CREATE INDEX IF NOT EXISTS idx_pages_project_title ON pages(project, title);

CREATE VIRTUAL TABLE IF NOT EXISTS pages_fts USING fts5(
  title,
  clean_text,
  content='pages',
  content_rowid='id',
  tokenize = 'unicode61'
);

CREATE TRIGGER IF NOT EXISTS pages_ai AFTER INSERT ON pages BEGIN
  INSERT INTO pages_fts(rowid, title, clean_text) VALUES (new.id, new.title, new.clean_text);
END;

CREATE TRIGGER IF NOT EXISTS pages_ad AFTER DELETE ON pages BEGIN
  INSERT INTO pages_fts(pages_fts, rowid, title, clean_text) VALUES('delete', old.id, old.title, old.clean_text);
END;

CREATE TRIGGER IF NOT EXISTS pages_au AFTER UPDATE ON pages BEGIN
  INSERT INTO pages_fts(pages_fts, rowid, title, clean_text) VALUES('delete', old.id, old.title, old.clean_text);
  INSERT INTO pages_fts(rowid, title, clean_text) VALUES (new.id, new.title, new.clean_text);
END;
"""


def connect_db(db_path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


# -----------------------------
# Builders
# -----------------------------
def build_one(
    spec: ProjectSpec,
    batch_size: int,
    max_pages: int,
    min_clean_len: int,
    use_mwparser: bool,
    verbose_every: int,
):
    log = lambda m: print(m, flush=True)

    if not os.path.exists(spec.dump_path):
        raise FileNotFoundError(f"Dump not found: {spec.dump_path}")

    log("============================================================")
    log(f"[Project] {spec.project} mode={spec.mode}")
    log(f"[Dump]    {spec.dump_path}")
    log(f"[DB]      {spec.db_path}")
    log(f"[BaseURL]  {spec.base_url}")
    log(f"[NS]      {sorted(list(spec.namespaces)) if spec.namespaces else 'ALL'}")
    log(f"[Params]  batch={batch_size} max_pages={max_pages if max_pages>0 else 'ALL'} min_len={min_clean_len} use_mwparser={use_mwparser}")
    log("============================================================")

    if use_mwparser:
        try:
            import mwparserfromhell  # noqa: F401
        except Exception:
            log("[ERROR] --use-mwparser requested but mwparserfromhell not installed.")
            log("        Install: pip install mwparserfromhell")
            raise

    cleaner = clean_wikitext_mwparser if use_mwparser else clean_wikitext_basic

    conn = connect_db(spec.db_path)
    cur = conn.cursor()
    cur.executescript(SCHEMA_SQL)
    conn.commit()

    insert_sql = """
      INSERT INTO pages(project, ns, title, url, updated_at, raw_wikitext, clean_text, snippet)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """

    total_seen = 0
    total_kept = 0
    batch: List[Tuple] = []

    t0 = time.time()
    t_last = t0

    for title, ns, ts, wikitext in iter_pages(spec.dump_path):
        total_seen += 1

        if spec.namespaces is not None and ns not in spec.namespaces:
            continue
        if not title:
            continue

        clean = cleaner(wikitext or "", spec.mode)
        if len(clean) < min_clean_len:
            continue

        url = make_page_url(spec.base_url, title)
        snippet = clean[:260].replace("\n", " ").strip()

        batch.append((spec.project, ns, title, url, ts or None, wikitext or "", clean, snippet))
        total_kept += 1

        if len(batch) >= batch_size:
            cur.executemany(insert_sql, batch)
            conn.commit()
            batch.clear()

        if verbose_every > 0 and total_seen % verbose_every == 0:
            now = time.time()
            dt = now - t_last
            elapsed = now - t0
            rate = (verbose_every / dt) if dt > 0 else 0.0
            log(f"[Progress {spec.project}] seen={total_seen} kept={total_kept} rate={rate:.1f}/s elapsed={elapsed/60:.1f}m")
            t_last = now

        if max_pages > 0 and total_kept >= max_pages:
            log(f"[Stop {spec.project}] Reached max_pages={max_pages}")
            break

    if batch:
        cur.executemany(insert_sql, batch)
        conn.commit()
        batch.clear()

    log(f"[DB {spec.project}] Optimize FTS...")
    with contextlib.suppress(Exception):
        cur.execute("INSERT INTO pages_fts(pages_fts) VALUES ('optimize');")
        conn.commit()

    # Vacuum can take time for huge DB; keep it optional-ish:
    log(f"[DB {spec.project}] VACUUM (may take time)...")
    with contextlib.suppress(Exception):
        cur.execute("VACUUM;")
        conn.commit()

    conn.close()

    log(f"[Done {spec.project}] total_seen={total_seen} total_indexed={total_kept}")
    log(f"[Done {spec.project}] db={spec.db_path}")


def parse_config(path: str) -> List[ProjectSpec]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    projects = []
    for item in cfg.get("projects", []):
        project = item["project"]
        dump_path = item["dump_path"]
        db_path = item["db_path"]
        base_url = item.get("base_url") or DEFAULT_BASE_URL.get(project)
        if not base_url:
            raise ValueError(f"Missing base_url for project {project}")

        # namespaces: null/absent => default, [] => ALL, [0] => only main
        ns = item.get("namespaces", None)
        if ns is None:
            # defaults: ns=0 for wiki and wikisource
            namespaces = {0}
        elif isinstance(ns, list) and len(ns) == 0:
            namespaces = None
        else:
            namespaces = set(int(x) for x in ns)

        mode = item.get("mode", "wiki" if project.endswith("wiki") else "wikisource")
        projects.append(ProjectSpec(project=project, dump_path=dump_path, db_path=db_path, base_url=base_url, namespaces=namespaces, mode=mode))
    return projects


def build_specs_from_args(args) -> List[ProjectSpec]:
    # If --config provided, use it
    if args.config:
        return parse_config(args.config)

    # Otherwise, build specs from --dump-dir / --out-dir and --projects
    dump_dir = args.dump_dir
    out_dir = args.out_dir
    if not dump_dir or not out_dir:
        raise ValueError("Either provide --config, or provide both --dump-dir and --out-dir.")

    specs: List[ProjectSpec] = []
    for project in args.projects:
        base_url = DEFAULT_BASE_URL.get(project)
        if not base_url:
            raise ValueError(f"Unknown project: {project}. Add to DEFAULT_BASE_URL or use --config with base_url.")
        # expected filename pattern:
        #   <project>-latest-pages-articles.xml.bz2
        # You can override by config if your filename differs.
        dump_path = os.path.join(dump_dir, project, f"{project}-latest-pages-articles.xml.bz2")
        db_path = os.path.join(out_dir, f"{project}_fts.sqlite")

        mode = "wiki" if project.endswith("wiki") else "wikisource"
        namespaces = {0} if args.ns_default_main_only else None
        specs.append(ProjectSpec(project=project, dump_path=dump_path, db_path=db_path, base_url=base_url, namespaces=namespaces, mode=mode))
    return specs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="Path to projects.json config (recommended)")

    ap.add_argument("--dump-dir", help="Root folder containing per-project dump subfolders (used if no --config)")
    ap.add_argument("--out-dir", help="Output folder for sqlite dbs (used if no --config)")
    ap.add_argument("--projects", nargs="*", default=["zhwiki", "zhwikisource", "enwiki", "enwikisource"],
                    help="Projects to build when using --dump-dir/--out-dir")

    ap.add_argument("--use-mwparser", action="store_true", help="Use mwparserfromhell for better cleanup (recommended for Wikipedia)")
    ap.add_argument("--batch", type=int, default=300, help="Insert batch size")
    ap.add_argument("--max-pages", type=int, default=0, help="Max indexed pages per project (0=ALL)")
    ap.add_argument("--min-clean-len", type=int, default=120, help="Min cleaned length to keep")
    ap.add_argument("--verbose-every", type=int, default=5000, help="Print progress every N pages seen (0 disables)")
    ap.add_argument("--ns-default-main-only", action="store_true",
                    help="When using --dump-dir mode, default namespaces to {0} only; otherwise ALL namespaces")
    args = ap.parse_args()

    specs = build_specs_from_args(args)

    for spec in specs:
        build_one(
            spec=spec,
            batch_size=args.batch,
            max_pages=args.max_pages,
            min_clean_len=args.min_clean_len,
            use_mwparser=args.use_mwparser,
            verbose_every=args.verbose_every,
        )


if __name__ == "__main__":
    main()

"""
pip install mwparserfromhell

python build_wikimedia_fts_multi.py \
  --dump-dir /data/rnd-liu/Datasets/wikidata/ \
  --out-dir /data/rnd-liu/Datasets/wikidata/ \
  --projects zhwikisource \
  --ns-default-main-only \
  --use-mwparser

python build_wikimedia_fts_multi.py \
  --dump-dir /data/rnd-liu/Datasets/wikidata/ \
  --out-dir /data/rnd-liu/Datasets/wikidata/ \
  --projects zhwiki \
  --ns-default-main-only \
  --use-mwparser

python build_wikimedia_fts_multi.py \
  --dump-dir /data/rnd-liu/Datasets/wikidata/ \
  --out-dir /data/rnd-liu/Datasets/wikidata/ \
  --projects enwikisource \
  --ns-default-main-only \
  --use-mwparser

python build_wikimedia_fts_multi.py \
  --dump-dir /data/rnd-liu/Datasets/wikidata/ \
  --out-dir /data/rnd-liu/Datasets/wikidata/ \
  --projects enwiki \
  --ns-default-main-only \
  --use-mwparser
# enwiki enwikisource

sqlite3 ./data/indexes/zhwiki_fts.sqlite

python build_wikimedia_fts_multi.py --config projects.json --use-mwparser
"""
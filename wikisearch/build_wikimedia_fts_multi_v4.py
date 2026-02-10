#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_wikimedia_fts_multi_v3.py

Adds:
- checkpoint resume by "seen pages" count (fast skip: no cleaning/building batches before checkpoint)
- real counters: selected vs inserted vs ignored
- stores checkpoint per project DB: <db>.checkpoint.json
- keeps sanitize fix + batch fallback
- keeps idempotency: UNIQUE(project, ns, title) + INSERT OR IGNORE

Resume behavior:
- First run writes checkpoints every N pages (default 50k).
- If interrupted, re-run with same args; it loads checkpoint and skips work
  until it reaches that "seen" count (still decompresses, but much faster).

"""

import argparse
import bz2
import contextlib
import json
import os
import re
import sqlite3
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Set, Tuple
from collections import deque
import concurrent.futures


def sanitize_text(s: str) -> str:
    if not s:
        return ""
    return s.encode("utf-8", "replace").decode("utf-8")


@dataclass
class ProjectSpec:
    project: str
    dump_path: str
    db_path: str
    base_url: str
    namespaces: Optional[Set[int]]
    mode: str = "wiki"


DEFAULT_BASE_URL = {
    "zhwiki": "https://zh.wikipedia.org/wiki/",
    "enwiki": "https://en.wikipedia.org/wiki/",
    "zhwikisource": "https://zh.wikisource.org/wiki/",
    "enwikisource": "https://en.wikisource.org/wiki/",
}


# -------- cleaning ----------
RE_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
RE_REF_TAG = re.compile(r"<ref\b[^>]*>.*?</ref\s*>", re.IGNORECASE | re.DOTALL)
RE_REF_SELF = re.compile(r"<ref\b[^>]*/\s*>", re.IGNORECASE)
RE_TABLE = re.compile(r"\{\|.*?\|\}", re.DOTALL)
RE_FILE_LINK = re.compile(r"\[\[(?:File|Image|文件|圖像|图像):[^\]]+\]\]", re.IGNORECASE)
RE_CATEGORY = re.compile(r"\[\[(?:Category|分类|分類):[^\]]+\]\]", re.IGNORECASE)
RE_TEMPLATE = re.compile(r"\{\{[^{}]*\}\}")
RE_TAGS = re.compile(
    r"</?(?:br|p|div|span|small|big|center|blockquote|poem|nowiki|code|pre|sup|sub)\b[^>]*>",
    re.IGNORECASE,
)
RE_HTML = re.compile(r"<[^>]+>")
RE_EXT_LINK = re.compile(r"\[(https?://[^\s\]]+)\s*([^\]]*)\]")
RE_WIKI_LINK = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]+)?(?:\|([^\]]+))?\]\]")
RE_BOLD_ITALIC = re.compile(r"'''''|'''|''")
RE_HEADINGS = re.compile(r"^={2,}\s*(.*?)\s*={2,}\s*$", re.MULTILINE)
RE_LISTMARK = re.compile(r"^\s*[*#:;]+\s*", re.MULTILINE)
RE_MULTISPACE = re.compile(r"[ \t]+")
RE_MULTINEWLINE = re.compile(r"\n{3,}")

WIKI_DROP_SECTION_TITLES = {
    "references", "external links", "see also", "further reading", "notes", "bibliography",
    "参考文献", "外部链接", "参见", "延伸阅读", "注释", "参考資料", "參考文獻", "外部連結"
}


def make_page_url(base_url: str, title: str) -> str:
    return base_url + title.replace(" ", "_")


def drop_wikipedia_trailing_sections(clean_text: str) -> str:
    lines = clean_text.splitlines()
    out: List[str] = []
    for line in lines:
        if line.strip().lower() in WIKI_DROP_SECTION_TITLES:
            break
        out.append(line)
    return "\n".join(out).strip()


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
    s = RE_FILE_LINK.sub(" ", s)
    s = RE_CATEGORY.sub(" ", s)

    for _ in range(8):
        new = RE_TEMPLATE.sub(" ", s)
        if new == s:
            break
        s = new

    s = RE_HEADINGS.sub(r"\n\1\n", s)
    s = RE_TAGS.sub(" ", s)
    s = RE_HTML.sub(" ", s)

    def _ext_link(m):
        url = m.group(1) or ""
        label = (m.group(2) or "").strip()
        return label if label else url

    s = RE_EXT_LINK.sub(_ext_link, s)

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

    if mode == "wiki" and s:
        s = drop_wikipedia_trailing_sections(s)
    return s.strip()


def clean_wikitext_mwparser(text: str, mode: str, max_len: int = 800_000) -> str:
    if not text:
        return ""
    if len(text) > max_len:
        text = text[:max_len]

    import mwparserfromhell  # type: ignore
    code = mwparserfromhell.parse(text)
    for tpl in code.filter_templates(recursive=True):
        with contextlib.suppress(Exception):
            code.remove(tpl)

    s = code.strip_code(normalize=True, collapse=True)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = RE_MULTISPACE.sub(" ", s)
    s = RE_MULTINEWLINE.sub("\n\n", s)
    s = s.strip()

    if mode == "wiki" and s:
        s = drop_wikipedia_trailing_sections(s)
    return s.strip()


# -------- dump streaming ----------
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


# -------- SQLite schema ----------
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

CREATE UNIQUE INDEX IF NOT EXISTS uq_pages_project_ns_title
ON pages(project, ns, title);

CREATE INDEX IF NOT EXISTS idx_pages_project_ns ON pages(project, ns);
CREATE INDEX IF NOT EXISTS idx_pages_project_title ON pages(project, title);

CREATE VIRTUAL TABLE IF NOT EXISTS pages_fts USING fts5(
  title,
  clean_text,
  content='pages',
  content_rowid='id',
  tokenize='unicode61'
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


def checkpoint_path(db_path: str) -> str:
    return db_path + ".checkpoint.json"


def load_checkpoint(db_path: str) -> int:
    cp = checkpoint_path(db_path)
    if not os.path.exists(cp):
        return 0
    try:
        with open(cp, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return int(obj.get("seen", 0))
    except Exception:
        return 0


def save_checkpoint(db_path: str, seen: int):
    cp = checkpoint_path(db_path)
    tmp = cp + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"seen": int(seen), "ts": time.time()}, f)
    os.replace(tmp, cp)


def process_page_worker(
    project: str,
    ns: int,
    title: str,
    url: str,
    ts: str,
    wikitext: str,
    mode: str,
    min_clean_len: int,
    cleaner_func: callable,
) -> Optional[Tuple]:
    """
    Worker function to process a single page.
    Returns the tuple for DB insertion, or None if filtered out.
    """
    if not title:
        return None

    title_s = sanitize_text(title)
    ts_s = sanitize_text(ts or "")
    wikitext_s = sanitize_text(wikitext or "")

    clean = cleaner_func(wikitext_s, mode)
    clean_s = sanitize_text(clean)

    if len(clean_s) < min_clean_len:
        return None

    snippet = sanitize_text(clean_s[:260].replace("\n", " ").strip())

    return (project, ns, title_s, url, ts_s or None, wikitext_s, clean_s, snippet)


def batch_insert_with_fallback(cur, conn, insert_sql, rows, project) -> int:
    """
    Returns number of rows actually inserted.
    Since we use INSERT OR IGNORE, sqlite changes() tells actual inserts per statement,
    but for executemany that's tricky. We'll use total_changes diff.
    """
    if not rows:
        return 0
    before = conn.total_changes
    try:
        cur.executemany(insert_sql, rows)
        conn.commit()
        return conn.total_changes - before
    except Exception as e:
        print(f"[WARN {project}] Batch insert failed -> fallback row-by-row. err={type(e).__name__}: {e}", flush=True)
        conn.rollback()

    inserted = 0
    for r in rows:
        try:
            b = conn.total_changes
            cur.execute(insert_sql, r)
            inserted += (conn.total_changes - b)
        except Exception as e2:
            title = r[2] if len(r) > 2 else ""
            print(f"[SKIP {project}] title={title!r} err={type(e2).__name__}: {e2}", flush=True)
    conn.commit()
    return inserted


def build_one(
    spec: ProjectSpec,
    batch_size: int,
    max_pages: int,
    min_clean_len: int,
    use_mwparser: bool,
    verbose_every: int,
    checkpoint_every: int,
    resume: bool,
):
    log = lambda m: print(m, flush=True)

    if not os.path.exists(spec.dump_path):
        raise FileNotFoundError(f"Dump not found: {spec.dump_path}")

    if use_mwparser:
        import mwparserfromhell  # noqa: F401
        cleaner = clean_wikitext_mwparser
    else:
        cleaner = clean_wikitext_basic

    conn = connect_db(spec.db_path)
    cur = conn.cursor()
    cur.executescript(SCHEMA_SQL)
    conn.commit()

    insert_sql = """
      INSERT OR IGNORE INTO pages(project, ns, title, url, updated_at, raw_wikitext, clean_text, snippet)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """

    resume_seen = load_checkpoint(spec.db_path) if resume else 0

    log("============================================================")
    log(f"[Project] {spec.project} mode={spec.mode}")
    log(f"[Dump]    {spec.dump_path}")
    log(f"[DB]      {spec.db_path}")
    log(f"[BaseURL]  {spec.base_url}")
    log(f"[NS]      {sorted(list(spec.namespaces)) if spec.namespaces else 'ALL'}")
    log(f"[Params]  batch={batch_size} max_pages={max_pages if max_pages>0 else 'ALL'} min_len={min_clean_len} use_mwparser={use_mwparser}")
    log(f"[Resume]  enabled={resume} resume_seen={resume_seen}")
    log("============================================================")

    total_seen = 0
    selected = 0      # passed ns + len filter
    inserted = 0      # actually inserted (not ignored)
    batch: List[Tuple] = []

    # Using a thread pool to process pages in parallel.
    # We must maintain order to ensure resume checkpoints are valid.
    # Approach:
    # 1. Main thread iterates XML, submits tasks to executor.
    # 2. Main thread keeps a deque of futures in submission order.
    # 3. If deque length reaches LIMIT, pop the oldest future, wait for it, add result to batch.
    # This ensures we yield results in the exact same order as the XML.

    MAX_PENDING_FUTURES = batch_size * 2
    pending_futures = deque()
    
    t0 = time.time()
    t_last = t0

    # We need a function reference for pickleability/callability (though passing function is fine in threadpool)
    # The worker logic is now in `process_page_worker`.

    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
        
        # We attach (index, future) to deque.
        # When we pop valid results and commit, we update `last_committed_index`.
        
        last_committed_index = resume_seen # Initialize with where we started
        
        for title, ns, ts, wikitext in iter_pages(spec.dump_path):
            total_seen += 1

            # FAST SKIP logic
            # If we haven't reached the checkpoint yet, valid
            if total_seen <= resume_seen:
                if verbose_every > 0 and total_seen % verbose_every == 0:
                    now = time.time()
                    dt = now - t_last
                    elapsed = now - t0
                    rate = (verbose_every / dt) if dt > 0 else 0.0
                    log(f"[Skipping {spec.project}] seen={total_seen}/{resume_seen} rate={rate:.1f}/s elapsed={elapsed/60:.1f}m")
                    t_last = now
                continue

            # Prepare task
            # Filter NS here? Or inside worker?
            # Filtering here saves thread overhead.
            should_skip = False
            if not title:
                should_skip = True
            elif spec.namespaces is not None and ns not in spec.namespaces:
                should_skip = True
            
            if not should_skip:
                url = make_page_url(spec.base_url, title)
                # Submit
                fut = executor.submit(
                    process_page_worker, 
                    spec.project, ns, title, url, ts, wikitext, 
                    spec.mode, min_clean_len, cleaner
                )
                pending_futures.append((total_seen, fut))
            else:
                # It's skipped, but we need to track that `total_seen` is processed.
                # We can append (total_seen, None) to preserve order for checkpointing?
                # Or just rely on the fact that if pending_futures is empty, we are at total_seen.
                # Let's use (total_seen, None) for skipped items if we want perfect checkpointing resolution,
                # but that might fill memory if we have millions of skipped items?
                # No, we drain constantly.
                pending_futures.append((total_seen, None))

            # Drain if too many pending
            while len(pending_futures) >= MAX_PENDING_FUTURES:
                idx, fut = pending_futures.popleft()
                if fut:
                    try:
                        res = fut.result()
                        if res:
                            batch.append(res)
                            selected += 1
                    except Exception as e:
                        log(f"[WARN] Future failed: {e}")
                
                # Checkpoint logic: We just processed `idx`. 
                # So we are safe to say we "saw" up to `idx`.
                last_committed_index = idx

                # Batch write
                if len(batch) >= batch_size:
                    inserted += batch_insert_with_fallback(cur, conn, insert_sql, batch, spec.project)
                    batch.clear()
                    
                    if checkpoint_every > 0:
                        # We only save occasionally
                         if last_committed_index % checkpoint_every == 0: # Approximate optimization
                             pass
                
                # We can't use `idx % checkpoint` easily because idx might skip numbers if we popped multiple? 
                # No, we pop one by one. 
                # Let's just use time or a separate counter for checkpoint frequency?
                # Or just `if last_committed_index % checkpoint_every == 0`? 
                # But `checkpoint_every` is e.g. 50000. `last_committed_index` will hit it exactly eventually.
                if checkpoint_every > 0 and last_committed_index % checkpoint_every == 0:
                    save_checkpoint(spec.db_path, last_committed_index)

            # Verbose log
            if verbose_every > 0 and total_seen % verbose_every == 0:
                 now = time.time()
                 dt = now - t_last
                 if dt > 0:
                    # Rate based on input read, not output committed
                    rate = verbose_every / dt
                    elapsed = now - t0
                    log(f"[Progress {spec.project}] seen={total_seen} (committed~{last_committed_index}) selected={selected} inserted={inserted} rate={rate:.1f}/s elapsed={elapsed/60:.1f}m")
                    t_last = now

            if max_pages > 0 and inserted >= max_pages:
                log(f"[Stop {spec.project}] Reached max_pages (inserted)={max_pages}")
                break

        # End of loop, drain remaining
        while pending_futures:
            idx, fut = pending_futures.popleft()
            if fut:
                try:
                    res = fut.result()
                    if res:
                        batch.append(res)
                        selected += 1
                except Exception:
                    pass
            last_committed_index = idx
            
            if len(batch) >= batch_size:
                inserted += batch_insert_with_fallback(cur, conn, insert_sql, batch, spec.project)
                batch.clear()
        
        # Final flush
        if batch:
            inserted += batch_insert_with_fallback(cur, conn, insert_sql, batch, spec.project)
            batch.clear()
        
        # Final checkpoint
        save_checkpoint(spec.db_path, last_committed_index)

    log(f"[DB {spec.project}] Optimize FTS...")
    with contextlib.suppress(Exception):
        cur.execute("INSERT INTO pages_fts(pages_fts) VALUES ('optimize');")
        conn.commit()

    log(f"[Done {spec.project}] seen={last_committed_index} selected={selected} inserted={inserted}")
    conn.close()


def parse_config(path: str) -> List[ProjectSpec]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    specs: List[ProjectSpec] = []
    for item in cfg.get("projects", []):
        project = item["project"]
        dump_path = item["dump_path"]
        db_path = item["db_path"]
        base_url = item.get("base_url") or DEFAULT_BASE_URL.get(project)
        if not base_url:
            raise ValueError(f"Missing base_url for project={project}")

        ns = item.get("namespaces", None)
        if ns is None:
            namespaces = {0}
        elif isinstance(ns, list) and len(ns) == 0:
            namespaces = None
        else:
            namespaces = set(int(x) for x in ns)

        mode = item.get("mode", "wiki" if project.endswith("wiki") else "wikisource")
        specs.append(ProjectSpec(project, dump_path, db_path, base_url, namespaces, mode))
    return specs


def build_specs_from_args(args) -> List[ProjectSpec]:
    if args.config:
        return parse_config(args.config)

    if not args.dump_dir or not args.out_dir:
        raise ValueError("Either provide --config, or provide both --dump-dir and --out-dir.")

    specs: List[ProjectSpec] = []
    for project in args.projects:
        base_url = DEFAULT_BASE_URL.get(project)
        if not base_url:
            raise ValueError(f"Unknown project: {project}")

        dump_path = os.path.join(args.dump_dir, project, f"{project}-latest-pages-articles.xml.bz2")
        db_path = os.path.join(args.out_dir, f"{project}_fts.sqlite")

        mode = "wiki" if project.endswith("wiki") else "wikisource"
        namespaces = {0} if args.ns_default_main_only else None
        specs.append(ProjectSpec(project, dump_path, db_path, base_url, namespaces, mode))
    return specs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="Path to projects.json config (recommended)")
    ap.add_argument("--dump-dir", help="Root folder containing per-project dump subfolders (used if no --config)")
    ap.add_argument("--out-dir", help="Output folder for sqlite dbs (used if no --config)")
    ap.add_argument("--projects", nargs="*", default=["zhwiki", "zhwikisource", "enwiki", "enwikisource"])

    ap.add_argument("--use-mwparser", action="store_true")
    ap.add_argument("--batch", type=int, default=1000)
    ap.add_argument("--max-pages", type=int, default=0)
    ap.add_argument("--min-clean-len", type=int, default=120)
    ap.add_argument("--verbose-every", type=int, default=5000)
    ap.add_argument("--ns-default-main-only", action="store_true")

    ap.add_argument("--resume", action="store_true", help="Resume from checkpoint (<db>.checkpoint.json)")
    ap.add_argument("--checkpoint-every", type=int, default=50000, help="Write checkpoint every N seen pages")
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
            checkpoint_every=args.checkpoint_every,
            resume=args.resume,
        )


if __name__ == "__main__":
    main()

"""
python build_wikimedia_fts_multi_v4.py \
  --dump-dir /data/rnd-liu/Datasets/wikidata/ \
  --out-dir  /data/rnd-liu/Datasets/wikidata/ \
  --projects enwiki \
  --ns-default-main-only \
  --use-mwparser \
  --resume
"""
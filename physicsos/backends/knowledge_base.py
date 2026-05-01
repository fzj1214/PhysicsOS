from __future__ import annotations

import re
import sqlite3
from pathlib import Path

from physicsos.schemas.knowledge import KnowledgeChunk, KnowledgeSource

DEFAULT_KB_PATH = Path("data/knowledge/physicsos_knowledge.sqlite")


def _connect(path: str | Path = DEFAULT_KB_PATH) -> sqlite3.Connection:
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    _init(conn)
    return conn


def _init(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS sources (id TEXT PRIMARY KEY, kind TEXT NOT NULL, title TEXT NOT NULL, uri TEXT, authors TEXT, published TEXT, summary TEXT)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS chunks (id TEXT PRIMARY KEY, source_id TEXT NOT NULL, text TEXT NOT NULL, metadata TEXT DEFAULT '{}', FOREIGN KEY(source_id) REFERENCES sources(id))"
    )
    try:
        conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(id UNINDEXED, text)")
    except sqlite3.OperationalError:
        pass
    conn.commit()


def _chunk_text(text: str, chunk_size: int = 1800, overlap: int = 250) -> list[str]:
    clean = re.sub(r"\s+", " ", text).strip()
    chunks: list[str] = []
    start = 0
    while clean and start < len(clean):
        end = min(start + chunk_size, len(clean))
        chunks.append(clean[start:end])
        if end == len(clean):
            break
        start = max(0, end - overlap)
    return chunks


def upsert_document(source: KnowledgeSource, text: str, db_path: str | Path = DEFAULT_KB_PATH) -> int:
    chunks = _chunk_text(text)
    with _connect(db_path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO sources(id, kind, title, uri, authors, published, summary) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (source.id, source.kind, source.title, source.uri, "\n".join(source.authors), source.published, source.summary),
        )
        conn.execute("DELETE FROM chunks WHERE source_id = ?", (source.id,))
        try:
            for row in conn.execute("SELECT id FROM chunks_fts WHERE id LIKE ?", (f"{source.id}:%",)).fetchall():
                conn.execute("DELETE FROM chunks_fts WHERE id = ?", (row["id"],))
        except sqlite3.OperationalError:
            pass
        for index, chunk in enumerate(chunks):
            chunk_id = f"{source.id}:{index:04d}"
            conn.execute("INSERT OR REPLACE INTO chunks(id, source_id, text) VALUES (?, ?, ?)", (chunk_id, source.id, chunk))
            try:
                conn.execute("INSERT INTO chunks_fts(id, text) VALUES (?, ?)", (chunk_id, chunk))
            except sqlite3.OperationalError:
                pass
        conn.commit()
    return len(chunks)


def search_knowledge(query: str, top_k: int = 8, db_path: str | Path = DEFAULT_KB_PATH) -> list[KnowledgeChunk]:
    with _connect(db_path) as conn:
        rows = _search_fts(conn, query, top_k)
        if not rows:
            fallback_query = _or_query(query)
            if fallback_query and fallback_query != query:
                rows = _search_fts(conn, fallback_query, top_k)
        if not rows:
            rows = conn.execute(
                """
                SELECT c.id, c.text, s.id AS source_id, s.kind, s.title, s.uri, s.authors, s.published, s.summary, 1.0 AS score
                FROM chunks c
                JOIN sources s ON c.source_id = s.id
                WHERE c.text LIKE ?
                LIMIT ?
                """,
                (f"%{query}%", top_k),
            ).fetchall()
    output: list[KnowledgeChunk] = []
    for row in rows:
        source = KnowledgeSource(
            id=row["source_id"],
            kind=row["kind"],
            title=row["title"],
            uri=row["uri"],
            authors=[item for item in (row["authors"] or "").split("\n") if item],
            published=row["published"],
            summary=row["summary"],
        )
        output.append(KnowledgeChunk(id=row["id"], source=source, text=row["text"], score=float(row["score"])))
    return output


def _search_fts(conn: sqlite3.Connection, query: str, top_k: int):
    try:
        return conn.execute(
            """
            SELECT c.id, c.text, s.id AS source_id, s.kind, s.title, s.uri, s.authors, s.published, s.summary, bm25(chunks_fts) AS score
            FROM chunks_fts
            JOIN chunks c ON chunks_fts.id = c.id
            JOIN sources s ON c.source_id = s.id
            WHERE chunks_fts MATCH ?
            ORDER BY score
            LIMIT ?
            """,
            (query, top_k),
        ).fetchall()
    except sqlite3.OperationalError:
        return []


def _or_query(query: str) -> str:
    tokens = [token for token in re.findall(r"[A-Za-z0-9_]+", query) if len(token) > 2]
    return " OR ".join(tokens[:12])

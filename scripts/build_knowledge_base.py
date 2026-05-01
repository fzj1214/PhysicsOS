from __future__ import annotations

import argparse
from pathlib import Path

from physicsos.backends.knowledge_base import DEFAULT_KB_PATH, upsert_document
from physicsos.schemas.knowledge import KnowledgeSource

DEFAULT_DOCS = ["ARCHITECTURE.md", "taps.md", "physicsOS.md", "QUICKSTART.md", "scratch/taps_paper_text.txt"]


def read_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        import fitz

        doc = fitz.open(path)
        return "\n".join(page.get_text() for page in doc)
    return path.read_text(encoding="utf-8", errors="ignore")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(DEFAULT_KB_PATH))
    parser.add_argument("paths", nargs="*", default=DEFAULT_DOCS)
    args = parser.parse_args()
    total = 0
    for item in args.paths:
        path = Path(item)
        if not path.exists():
            print(f"skip missing {path}")
            continue
        source = KnowledgeSource(id=f"local:{path.as_posix()}", kind="local_doc", title=path.name, uri=path.as_posix())
        chunks = upsert_document(source, read_text(path), db_path=args.db)
        total += chunks
        print(f"ingested {path} -> {chunks} chunks")
    print(f"knowledge base: {args.db}; chunks={total}")


if __name__ == "__main__":
    main()


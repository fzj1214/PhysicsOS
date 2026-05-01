from __future__ import annotations

import argparse
import time
import urllib.request
from pathlib import Path

import fitz

from physicsos.backends.knowledge_base import DEFAULT_KB_PATH, upsert_document
from physicsos.schemas.knowledge import KnowledgeSource
from physicsos.tools.knowledge_tools import ArxivSearchInput, DeepSearchInput, run_deepsearch, search_arxiv

ARXIV_QUERIES = [
    "all:finite element method PDE Galerkin scientific computing",
    "all:finite volume method computational fluid dynamics conservation law",
    "all:multigrid PETSc scalable solvers PDE",
    "all:reduced basis method parametric PDE model order reduction",
    "all:proper generalized decomposition parametric PDE tensor decomposition",
    "all:Fourier neural operator PDE surrogate",
    "all:DeepONet operator learning PDE",
    "all:physics informed neural networks PDE",
    "all:MeshGraphNets simulation",
    "all:geometry informed neural operator complex geometry",
    "all:mesh generation computational geometry simulation",
    "all:signed distance function immersed boundary method PDE",
    "all:verification validation uncertainty quantification computational physics",
    "all:Kohn Sham density functional theory plane wave",
    "all:molecular dynamics LAMMPS force field",
    "all:additive manufacturing thermal simulation laser powder bed fusion",
    "all:Tensor-decomposition-based A Priori Surrogate TAPS",
]

CURATED_PDF_URLS = {
    "arxiv:2503.13933": "https://arxiv.org/pdf/2503.13933",
    "arxiv:2409.18032": "https://arxiv.org/pdf/2409.18032",
    "arxiv:2010.08895": "https://arxiv.org/pdf/2010.08895",
    "arxiv:1910.03193": "https://arxiv.org/pdf/1910.03193",
    "arxiv:1711.10561": "https://arxiv.org/pdf/1711.10561",
    "arxiv:2010.03409": "https://arxiv.org/pdf/2010.03409",
    "arxiv:2310.00120": "https://arxiv.org/pdf/2310.00120",
}

LOCAL_DOCS = [
    "ARCHITECTURE.md",
    "taps.md",
    "physicsOS.md",
    "QUICKSTART.md",
    "docs/knowledge_seed/core_formulas.md",
    "scratch/taps_paper_text.txt",
]

DEEPSEARCH_TOPICS = [
    "Computational physics and scientific computing knowledge map: PDE solvers, FEM, FVM, spectral methods, multigrid, PETSc, verification and validation. Include key formulas and canonical papers.",
    "TAPS Tensor-decomposition-based A Priori Surrogate: explain C-HiDeNN, tensor decomposition, space-parameter-time Galerkin weak form, subspace iteration, applicability, limitations, and implementation roadmap.",
    "Neural operators for scientific computing: FNO, TFNO, DeepONet, MeshGraphNets, PINNs, geometry-informed neural operators. Compare assumptions, input representations, and verification risks.",
    "Geometry and mesh generation for AI-native simulation: CAD repair, SDF, occupancy grids, immersed boundary, unstructured mesh graphs, boundary labeling, adaptive refinement.",
]


def read_local_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        doc = fitz.open(path)
        return "\n".join(page.get_text() for page in doc)
    return path.read_text(encoding="utf-8", errors="ignore")


def ingest_local_docs(db_path: str) -> int:
    total = 0
    for item in LOCAL_DOCS:
        path = Path(item)
        if not path.exists():
            print(f"skip missing local doc {path}")
            continue
        text = read_local_text(path)
        source = KnowledgeSource(id=f"local:{path.as_posix()}", kind="local_doc", title=path.name, uri=path.as_posix())
        chunks = upsert_document(source, text, db_path=db_path)
        total += chunks
        print(f"local {path} -> {chunks} chunks")
    return total


def ingest_arxiv_searches(db_path: str, max_results: int) -> int:
    seen: set[str] = set()
    total = 0
    for query in ARXIV_QUERIES:
        print(f"arxiv query: {query}")
        papers = search_arxiv(ArxivSearchInput(query=query, max_results=max_results)).papers
        time.sleep(3.1)
        for paper in papers:
            if paper.id in seen:
                continue
            seen.add(paper.id)
            text = f"Title: {paper.title}\nAuthors: {', '.join(paper.authors)}\nPublished: {paper.published}\nCategories: {', '.join(paper.categories)}\nAbstract: {paper.summary}\nPDF: {paper.pdf_url}\n"
            chunks = upsert_document(paper, text, db_path=db_path)
            total += chunks
        print(f"  collected {len(papers)} papers; unique={len(seen)}")
    return total


def download_pdf(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 1024:
        return
    req = urllib.request.Request(url, headers={"User-Agent": "PhysicsOS knowledge seeding"})
    with urllib.request.urlopen(req, timeout=60) as response:
        output_path.write_bytes(response.read())


def ingest_curated_pdfs(db_path: str) -> int:
    total = 0
    for source_id, url in CURATED_PDF_URLS.items():
        arxiv_id = source_id.split(":", 1)[1]
        pdf_path = Path("data/knowledge/papers") / f"{arxiv_id}.pdf"
        print(f"pdf {arxiv_id}")
        try:
            download_pdf(url, pdf_path)
            text = read_local_text(pdf_path)
        except Exception as exc:
            print(f"  failed {arxiv_id}: {exc}")
            continue
        source = KnowledgeSource(id=source_id, kind="arxiv", title=f"arXiv {arxiv_id}", uri=url)
        chunks = upsert_document(source, text, db_path=db_path)
        total += chunks
        print(f"  -> {chunks} chunks")
        time.sleep(3.1)
    return total


def ingest_deepsearch_reports(db_path: str) -> int:
    total = 0
    for index, topic in enumerate(DEEPSEARCH_TOPICS, 1):
        print(f"deepsearch {index}/{len(DEEPSEARCH_TOPICS)}")
        report = run_deepsearch(DeepSearchInput(query=topic, temperature=0.2)).report
        if report.error:
            print(f"  skipped failed DeepSearch report: {report.error[:160]}")
            continue
        else:
            text = report.content
        source = KnowledgeSource(id=f"deepsearch:{index:02d}", kind="deepsearch", title=f"DeepSearch report {index}", summary=topic)
        chunks = upsert_document(source, text, db_path=db_path)
        total += chunks
        print(f"  -> {chunks} chunks; error={bool(report.error)}")
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(DEFAULT_KB_PATH))
    parser.add_argument("--max-results", type=int, default=6)
    parser.add_argument("--skip-deepsearch", action="store_true")
    parser.add_argument("--skip-pdfs", action="store_true")
    args = parser.parse_args()

    total = 0
    total += ingest_local_docs(args.db)
    total += ingest_arxiv_searches(args.db, max_results=args.max_results)
    if not args.skip_pdfs:
        total += ingest_curated_pdfs(args.db)
    if not args.skip_deepsearch:
        total += ingest_deepsearch_reports(args.db)
    print(f"seeded knowledge base {args.db}; new/updated chunks={total}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import os
from urllib.error import URLError
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

from pydantic import Field

from physicsos.backends.knowledge_base import DEFAULT_KB_PATH, search_knowledge, upsert_document
from physicsos.config import load_env_file
from physicsos.schemas.common import StrictBaseModel
from physicsos.schemas.knowledge import ArxivPaper, DeepSearchReport, KnowledgeChunk, KnowledgeContext, KnowledgeSource


class ArxivSearchInput(StrictBaseModel):
    query: str
    max_results: int = 5
    sort_by: str = "relevance"
    sort_order: str = "descending"


class ArxivSearchOutput(StrictBaseModel):
    papers: list[ArxivPaper] = Field(default_factory=list)

    def __str__(self) -> str:
        """Return compact JSON instead of pydantic repr to save context."""
        import json
        return json.dumps([p.model_dump(exclude_none=True) for p in self.papers], ensure_ascii=False)


def search_arxiv(input: ArxivSearchInput) -> ArxivSearchOutput:
    """Search arXiv through the official Atom API."""
    if input.max_results <= 0:
        return ArxivSearchOutput()
    # Validate sort parameters against arXiv API requirements
    valid_sort_by = {"relevance", "lastUpdatedDate", "submittedDate"}
    valid_sort_order = {"ascending", "descending"}
    if input.sort_by not in valid_sort_by:
        raise ValueError(f"sort_by must be one of {valid_sort_by}, got: {input.sort_by}")
    if input.sort_order not in valid_sort_order:
        raise ValueError(f"sort_order must be one of {valid_sort_order}, got: {input.sort_order}")
    params = {
        "search_query": input.query,
        "start": "0",
        "max_results": str(input.max_results),
        "sortBy": input.sort_by,
        "sortOrder": input.sort_order,
    }
    url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            root = ET.fromstring(response.read())
    except (URLError, TimeoutError, ET.ParseError) as exc:
        raise RuntimeError(f"arXiv API request failed: {exc}") from exc
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    papers: list[ArxivPaper] = []
    for entry in root.findall("atom:entry", ns):
        entry_id = entry.findtext("atom:id", default="", namespaces=ns)
        arxiv_id = entry_id.rstrip("/").split("/")[-1]
        pdf_url = None
        for link in entry.findall("atom:link", ns):
            if link.attrib.get("title") == "pdf" or link.attrib.get("type") == "application/pdf":
                pdf_url = link.attrib.get("href")
        papers.append(
            ArxivPaper(
                id=f"arxiv:{arxiv_id}",
                title=" ".join(entry.findtext("atom:title", default="", namespaces=ns).split()),
                uri=entry_id,
                arxiv_id=arxiv_id,
                authors=[
                    author.findtext("atom:name", default="", namespaces=ns)
                    for author in entry.findall("atom:author", ns)
                    if author.findtext("atom:name", default="", namespaces=ns)
                ],
                published=entry.findtext("atom:published", default=None, namespaces=ns),
                categories=[cat.attrib.get("term", "") for cat in entry.findall("atom:category", ns) if cat.attrib.get("term")],
                pdf_url=pdf_url,
                abstract_url=entry_id,
                summary=" ".join(entry.findtext("atom:summary", default="", namespaces=ns).split()),
            )
        )
    return ArxivSearchOutput(papers=papers)


class DeepSearchInput(StrictBaseModel):
    query: str
    model: str | None = None
    temperature: float = 0.3


class DeepSearchOutput(StrictBaseModel):
    report: DeepSearchReport


def run_deepsearch(input: DeepSearchInput) -> DeepSearchOutput:
    """Run OpenAI-compatible DeepSearch model through the configured provider."""
    load_env_file()
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("openai package is required for run_deepsearch.") from exc

    api_key = os.getenv("PHYSICSOS_OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set PHYSICSOS_OPENAI_API_KEY before using run_deepsearch.")
    base_url = os.getenv("PHYSICSOS_OPENAI_BASE_URL") or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model = input.model or os.getenv("PHYSICSOS_DEEPSEARCH_MODEL", "gemini-2.5-pro-deepsearch")
    client = OpenAI(api_key=api_key, base_url=base_url)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": input.query}],
            temperature=input.temperature,
            stream=False,
        )
    except Exception as exc:
        return DeepSearchOutput(
            report=DeepSearchReport(
                query=input.query,
                model=model,
                content="",
                error=f"{type(exc).__name__}: {exc}",
            )
        )
    content = response.choices[0].message.content or ""
    # Detect provider-level refusal/error disguised as success
    error = None
    if any(phrase in content.lower() for phrase in (
        "i encountered an error",
        "i'm having a hard time",
        "can i help you with something else",
        "could you try again",
    )):
        error = f"Provider returned refusal or error message: {content[:200]}"
        content = ""
    return DeepSearchOutput(report=DeepSearchReport(query=input.query, model=model, content=content, error=error))


class IngestKnowledgeDocumentInput(StrictBaseModel):
    path: str
    title: str | None = None
    kind: str = "local_doc"
    db_path: str = str(DEFAULT_KB_PATH)


class IngestKnowledgeDocumentOutput(StrictBaseModel):
    source: KnowledgeSource
    chunks: int


def ingest_knowledge_document(input: IngestKnowledgeDocumentInput) -> IngestKnowledgeDocumentOutput:
    """Ingest a local text/markdown document into the PhysicsOS knowledge base."""
    path = Path(input.path)
    source = KnowledgeSource(id=f"local:{path.as_posix()}", kind=input.kind, title=input.title or path.name, uri=path.as_posix())  # type: ignore[arg-type]
    chunks = upsert_document(source, path.read_text(encoding="utf-8"), db_path=input.db_path)
    return IngestKnowledgeDocumentOutput(source=source, chunks=chunks)


class SearchKnowledgeBaseInput(StrictBaseModel):
    query: str
    top_k: int = 8
    db_path: str = str(DEFAULT_KB_PATH)


class SearchKnowledgeBaseOutput(StrictBaseModel):
    chunks: list[KnowledgeChunk] = Field(default_factory=list)


def search_knowledge_base(input: SearchKnowledgeBaseInput) -> SearchKnowledgeBaseOutput:
    """Search the local PhysicsOS knowledge base."""
    return SearchKnowledgeBaseOutput(chunks=search_knowledge(input.query, top_k=input.top_k, db_path=input.db_path))


class BuildKnowledgeContextInput(StrictBaseModel):
    query: str
    local_top_k: int = 5
    arxiv_max_results: int = 5
    use_deepsearch: bool = False


class BuildKnowledgeContextOutput(StrictBaseModel):
    context: KnowledgeContext


def build_knowledge_context(input: BuildKnowledgeContextInput) -> BuildKnowledgeContextOutput:
    """Build combined context from local KB, arXiv, and optional DeepSearch."""
    chunks = search_knowledge(input.query, top_k=input.local_top_k)
    papers = []
    if input.arxiv_max_results > 0:
        try:
            papers = search_arxiv(ArxivSearchInput(query=input.query, max_results=input.arxiv_max_results)).papers
        except (OSError, URLError, TimeoutError, ET.ParseError):
            papers = []
    deepsearch = run_deepsearch(DeepSearchInput(query=input.query)).report if input.use_deepsearch else None
    return BuildKnowledgeContextOutput(
        context=KnowledgeContext(
            query=input.query,
            chunks=chunks,
            papers=papers,
            deepsearch=deepsearch,
            recommended_next_action="Use local KB first, arXiv for paper discovery, DeepSearch for broad synthesis.",
        )
    )


for _tool, _input, _output, _side_effects in [
    (search_arxiv, ArxivSearchInput, ArxivSearchOutput, "network: arxiv"),
    (run_deepsearch, DeepSearchInput, DeepSearchOutput, "network: openai-compatible deepsearch"),
    (ingest_knowledge_document, IngestKnowledgeDocumentInput, IngestKnowledgeDocumentOutput, "writes local knowledge base"),
    (search_knowledge_base, SearchKnowledgeBaseInput, SearchKnowledgeBaseOutput, "reads local knowledge base"),
    (build_knowledge_context, BuildKnowledgeContextInput, BuildKnowledgeContextOutput, "network optional plus local knowledge base"),
]:
    _tool.input_model = _input
    _tool.output_model = _output
    _tool.side_effects = _side_effects
    _tool.requires_approval = False

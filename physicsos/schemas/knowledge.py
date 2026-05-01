from __future__ import annotations

from typing import Literal

from pydantic import Field

from physicsos.schemas.common import StrictBaseModel


class KnowledgeSource(StrictBaseModel):
    id: str
    kind: Literal["local_doc", "arxiv", "deepsearch", "case_memory", "web", "manual"]
    title: str
    uri: str | None = None
    authors: list[str] = Field(default_factory=list)
    published: str | None = None
    summary: str | None = None


class KnowledgeChunk(StrictBaseModel):
    id: str
    source: KnowledgeSource
    text: str
    score: float | None = None
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)


class ArxivPaper(KnowledgeSource):
    kind: Literal["arxiv"] = "arxiv"
    arxiv_id: str
    categories: list[str] = Field(default_factory=list)
    pdf_url: str | None = None
    abstract_url: str | None = None


class DeepSearchReport(StrictBaseModel):
    query: str
    model: str
    content: str
    error: str | None = None
    sources: list[KnowledgeSource] = Field(default_factory=list)


class KnowledgeContext(StrictBaseModel):
    query: str
    chunks: list[KnowledgeChunk] = Field(default_factory=list)
    papers: list[ArxivPaper] = Field(default_factory=list)
    deepsearch: DeepSearchReport | None = None
    recommended_next_action: str | None = None

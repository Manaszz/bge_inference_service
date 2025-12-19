from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Sequence, Union

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    service: str
    embedder_loaded: bool
    reranker_loaded: bool
    device: str


# -----------------------------------------------------------------------------
# OpenAI-compatible embeddings
# -----------------------------------------------------------------------------


class OpenAIEmbeddingsRequest(BaseModel):
    model: Optional[str] = None
    input: Union[str, List[str]]
    encoding_format: Optional[Literal["float"]] = "float"


class OpenAIEmbeddingData(BaseModel):
    object: Literal["embedding"] = "embedding"
    index: int
    embedding: List[float]


class OpenAIEmbeddingsUsage(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0


class OpenAIEmbeddingsResponse(BaseModel):
    object: Literal["list"] = "list"
    model: str
    data: List[OpenAIEmbeddingData]
    usage: OpenAIEmbeddingsUsage = Field(default_factory=OpenAIEmbeddingsUsage)


# -----------------------------------------------------------------------------
# Sparse / Hybrid (custom)
# -----------------------------------------------------------------------------


class SparseEmbeddingsRequest(BaseModel):
    model: Optional[str] = None
    input: Union[str, List[str]]


class SparseVector(BaseModel):
    indices: List[int]
    values: List[float]
    mapping: str
    index_space: int


class SparseEmbeddingsResponseItem(BaseModel):
    index: int
    sparse: SparseVector


class SparseEmbeddingsResponse(BaseModel):
    model: str
    data: List[SparseEmbeddingsResponseItem]


class HybridEmbeddingsRequest(BaseModel):
    model: Optional[str] = None
    input: Union[str, List[str]]


class HybridEmbeddingsResponseItem(BaseModel):
    index: int
    dense: List[float]
    sparse: SparseVector


class HybridEmbeddingsResponse(BaseModel):
    model: str
    data: List[HybridEmbeddingsResponseItem]


# -----------------------------------------------------------------------------
# Rerank (Cohere-style)
# -----------------------------------------------------------------------------


class RerankDocument(BaseModel):
    text: str
    id: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None


class RerankRequest(BaseModel):
    model: Optional[str] = None
    query: str
    documents: List[Union[str, RerankDocument]]
    top_n: Optional[int] = Field(default=None, ge=1)
    return_documents: bool = Field(default=False)


class RerankResult(BaseModel):
    index: int
    relevance_score: float
    document: Optional[RerankDocument] = None


class RerankResponse(BaseModel):
    model: str
    results: List[RerankResult]

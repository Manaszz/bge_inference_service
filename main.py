from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException

from bge_inference_service.batcher import EmbeddingMicroBatcher
from bge_inference_service.config import settings
from bge_inference_service.engine import BGEEngine
from bge_inference_service.schemas import (
    HealthResponse,
    HybridEmbeddingsRequest,
    HybridEmbeddingsResponse,
    HybridEmbeddingsResponseItem,
    OpenAIEmbeddingsRequest,
    OpenAIEmbeddingsResponse,
    OpenAIEmbeddingData,
    RerankDocument,
    RerankRequest,
    RerankResponse,
    RerankResult,
    SparseEmbeddingsRequest,
    SparseEmbeddingsResponse,
    SparseEmbeddingsResponseItem,
    SparseVector,
)


def _as_list(inp) -> List[str]:
    if isinstance(inp, list):
        return [str(x) for x in inp]
    return [str(inp)]


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))
    engine = BGEEngine(settings)
    try:
        engine.load()
    except Exception as e:
        logger.exception("Failed to load models: %s", e)
        # Keep app running but mark degraded
    app.state.engine = engine
    app.state.microbatcher = EmbeddingMicroBatcher(
        engine=engine,
        enabled=settings.embedding_microbatch_enabled,
        max_wait_ms=settings.embedding_microbatch_max_wait_ms,
        max_batch_texts=settings.embedding_microbatch_max_batch_texts,
        queue_maxsize=settings.embedding_microbatch_queue_maxsize,
    )
    await app.state.microbatcher.start()
    yield
    try:
        await app.state.microbatcher.stop()
    except Exception:
        logger.exception("Failed to stop microbatcher")


app = FastAPI(
    title="BGE Inference Service",
    description="Standalone embeddings (dense/sparse/hybrid) and rerank API backed by FlagEmbedding.",
    version="0.3.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    engine: BGEEngine = app.state.engine
    return HealthResponse(
        status="ok" if engine.is_loaded else "degraded",
        service=settings.service_name,
        embedder_loaded=engine.embedder_loaded,
        reranker_loaded=engine.reranker_loaded,
        device=settings.device,
    )


@app.post("/v1/embeddings", response_model=OpenAIEmbeddingsResponse)
async def openai_embeddings(req: OpenAIEmbeddingsRequest) -> OpenAIEmbeddingsResponse:
    engine: BGEEngine = app.state.engine
    if not engine.is_loaded:
        raise HTTPException(status_code=503, detail="Models are not loaded")

    texts = _as_list(req.input)
    microbatcher: EmbeddingMicroBatcher = app.state.microbatcher
    dense_vecs = await microbatcher.embed_dense(texts)

    model = req.model or settings.openai_default_model_alias
    data = [OpenAIEmbeddingData(index=i, embedding=emb) for i, emb in enumerate(dense_vecs)]
    return OpenAIEmbeddingsResponse(model=model, data=data)


@app.post("/v1/sparse-embeddings", response_model=SparseEmbeddingsResponse)
async def sparse_embeddings(req: SparseEmbeddingsRequest) -> SparseEmbeddingsResponse:
    engine: BGEEngine = app.state.engine
    if not engine.is_loaded:
        raise HTTPException(status_code=503, detail="Models are not loaded")

    texts = _as_list(req.input)
    microbatcher: EmbeddingMicroBatcher = app.state.microbatcher
    sparse_vecs = await microbatcher.embed_sparse(texts)

    model = req.model or settings.embedding_model_name
    data = [
        SparseEmbeddingsResponseItem(
            index=i,
            sparse=SparseVector(
                indices=v.indices,
                values=v.values,
                mapping=v.mapping,
                index_space=v.index_space,
            ),
        )
        for i, v in enumerate(sparse_vecs)
    ]
    return SparseEmbeddingsResponse(model=model, data=data)


@app.post("/v1/hybrid-embeddings", response_model=HybridEmbeddingsResponse)
async def hybrid_embeddings(req: HybridEmbeddingsRequest) -> HybridEmbeddingsResponse:
    engine: BGEEngine = app.state.engine
    if not engine.is_loaded:
        raise HTTPException(status_code=503, detail="Models are not loaded")

    texts = _as_list(req.input)
    microbatcher: EmbeddingMicroBatcher = app.state.microbatcher
    dense_vecs, sparse_vecs = await microbatcher.embed_hybrid(texts)

    model = req.model or settings.embedding_model_name
    data = []
    for i, (dense, sparse) in enumerate(zip(dense_vecs, sparse_vecs)):
        data.append(
            HybridEmbeddingsResponseItem(
                index=i,
                dense=dense,
                sparse=SparseVector(
                    indices=sparse.indices,
                    values=sparse.values,
                    mapping=sparse.mapping,
                    index_space=sparse.index_space,
                ),
            )
        )

    return HybridEmbeddingsResponse(model=model, data=data)


@app.post("/v1/rerank", response_model=RerankResponse)
def rerank(req: RerankRequest) -> RerankResponse:
    engine: BGEEngine = app.state.engine
    if not engine.is_loaded:
        raise HTTPException(status_code=503, detail="Models are not loaded")

    docs_text: List[str] = []
    docs_norm: List[RerankDocument] = []
    for d in req.documents:
        if isinstance(d, str):
            rd = RerankDocument(text=d)
        else:
            rd = d
        docs_norm.append(rd)
        docs_text.append(rd.text)

    scores = engine.rerank(req.query, docs_text)

    results = [
        RerankResult(
            index=i,
            relevance_score=float(score),
            document=(docs_norm[i] if req.return_documents else None),
        )
        for i, score in enumerate(scores)
    ]
    results.sort(key=lambda r: r.relevance_score, reverse=True)

    if req.top_n is not None:
        results = results[: req.top_n]

    model = req.model or settings.reranker_model_name
    return RerankResponse(model=model, results=results)

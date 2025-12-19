from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from bge_inference_service.config import Settings


@dataclass(frozen=True)
class SparseVector:
    indices: List[int]
    values: List[float]
    mapping: str
    index_space: int


class BGEEngine:
    """In-process inference for BGE-M3 (dense/sparse/hybrid) and BGE reranker.

    This intentionally mirrors the current project's algorithms:
    - sparse token mapping: SHA256 hashing into a fixed index space OR tokenizer IDs
    - collision handling: sum weights for colliding indices
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.logger = logging.getLogger(__name__)

        self._embedder: Any = None
        self._reranker: Any = None
        self._tokenizer: Any = None
        self._sparse_mapping_effective: str = settings.sparse_token_mapping
        self._sparse_index_space: int = int(settings.sparse_index_space)

    def load(self) -> None:
        # Import lazily so service can start even if deps are missing (but will fail on load)
        from FlagEmbedding import BGEM3FlagModel, FlagReranker

        device = self.settings.device
        use_fp16 = bool(self.settings.use_fp16)

        # For safety: fp16 only makes sense on cuda
        if device != "cuda" and use_fp16:
            self.logger.info("USE_FP16 requested but DEVICE!=cuda; forcing fp16 off")
            use_fp16 = False

        self.logger.info(
            "Loading embedder model=%s device=%s fp16=%s",
            self.settings.embedding_model_name,
            device,
            use_fp16,
        )
        self._embedder = BGEM3FlagModel(
            self.settings.embedding_model_name,
            use_fp16=use_fp16,
            device=device,
        )

        self.logger.info(
            "Loading reranker model=%s device=%s fp16=%s",
            self.settings.reranker_model_name,
            device,
            use_fp16,
        )
        self._reranker = FlagReranker(
            self.settings.reranker_model_name,
            use_fp16=use_fp16,
            device=device,
        )

        self._sparse_mapping_effective = self.settings.sparse_token_mapping
        self._sparse_index_space = int(self.settings.sparse_index_space)

        if self._sparse_mapping_effective == "tokenizer":
            try:
                from transformers import AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.settings.embedding_model_name,
                    trust_remote_code=self.settings.tokenizer_trust_remote_code,
                )
                self.logger.info("Sparse token mapping: tokenizer (%s)", self.settings.embedding_model_name)
            except Exception as e:
                self.logger.warning(
                    "Failed to load tokenizer for sparse mapping; falling back to hash. error=%s",
                    e,
                )
                self._tokenizer = None
                self._sparse_mapping_effective = "hash"
        else:
            self._tokenizer = None
            self.logger.info("Sparse token mapping: hash (SHA256)")

    @property
    def is_loaded(self) -> bool:
        return self.embedder_loaded and self.reranker_loaded

    @property
    def embedder_loaded(self) -> bool:
        return self._embedder is not None

    @property
    def reranker_loaded(self) -> bool:
        return self._reranker is not None

    def _truncate_text(self, text: str) -> str:
        if text is None:
            raise ValueError("text is required")
        text = str(text)
        if not text.strip():
            raise ValueError("Cannot process empty text")
        if len(text) > self.settings.max_text_chars:
            return text[: self.settings.max_text_chars]
        return text

    def _prepare_texts(self, texts: Sequence[str]) -> List[str]:
        if not isinstance(texts, (list, tuple)):
            raise ValueError("texts must be a list")
        if len(texts) == 0:
            raise ValueError("texts must be non-empty")
        if len(texts) > self.settings.max_batch_size:
            raise ValueError(f"batch too large: {len(texts)} > {self.settings.max_batch_size}")
        return [self._truncate_text(t) for t in texts]

    def dense(self, texts: Sequence[str]) -> List[List[float]]:
        if not self.is_loaded:
            raise RuntimeError("Engine not loaded")
        prepared = self._prepare_texts(texts)
        out = self._embedder.encode(prepared, return_sparse=False, return_dense=True)
        dense_vecs = [list(map(float, v)) for v in out["dense_vecs"]]
        expected = int(self.settings.embedding_size)
        for i, v in enumerate(dense_vecs):
            if len(v) != expected:
                raise RuntimeError(
                    f"Unexpected dense embedding size for item {i}: got {len(v)}, expected {expected}. "
                    f"Model={self.settings.embedding_model_name}"
                )
        return dense_vecs

    def sparse_lexical_weights(self, texts: Sequence[str]) -> List[Dict[str, float]]:
        if not self.is_loaded:
            raise RuntimeError("Engine not loaded")
        prepared = self._prepare_texts(texts)
        out = self._embedder.encode(prepared, return_sparse=True, return_dense=False)
        # lexical_weights: List[Dict[token, weight]]
        return out["lexical_weights"]

    def sparse(self, texts: Sequence[str]) -> List[SparseVector]:
        weights_list = self.sparse_lexical_weights(texts)
        return [self._lexical_weights_to_sparse_vector(w) for w in weights_list]

    def hybrid(self, texts: Sequence[str]) -> Tuple[List[List[float]], List[SparseVector]]:
        if not self.is_loaded:
            raise RuntimeError("Engine not loaded")
        prepared = self._prepare_texts(texts)
        out = self._embedder.encode(prepared, return_sparse=True, return_dense=True)
        dense_vecs = [list(map(float, v)) for v in out["dense_vecs"]]
        expected = int(self.settings.embedding_size)
        for i, v in enumerate(dense_vecs):
            if len(v) != expected:
                raise RuntimeError(
                    f"Unexpected dense embedding size for item {i}: got {len(v)}, expected {expected}. "
                    f"Model={self.settings.embedding_model_name}"
                )
        sparse_vecs = [self._lexical_weights_to_sparse_vector(w) for w in out["lexical_weights"]]
        return dense_vecs, sparse_vecs

    def rerank(self, query: str, documents: Sequence[str]) -> List[float]:
        if not self.is_loaded:
            raise RuntimeError("Engine not loaded")
        q = self._truncate_text(query)
        docs = self._prepare_texts(list(documents))
        pairs = [[q, d] for d in docs]
        scores = self._reranker.compute_score(pairs)
        if isinstance(scores, (float, int)):
            return [float(scores)]
        return [float(s) for s in scores]

    def _lexical_weights_to_sparse_vector(self, weights: Dict[Any, Any]) -> SparseVector:
        indices: List[int] = []
        values: List[float] = []

        mapping = self._sparse_mapping_effective
        index_space = self._sparse_index_space

        if mapping == "tokenizer" and self._tokenizer is not None:
            unk_id = getattr(self._tokenizer, "unk_token_id", None)
            for token, weight in weights.items():
                w = float(weight)
                if isinstance(token, str):
                    token_id = int(self._tokenizer.convert_tokens_to_ids(token))
                    if unk_id is not None and token_id == int(unk_id):
                        continue
                    indices.append(token_id)
                    values.append(w)
                else:
                    indices.append(int(token))
                    values.append(w)
        else:
            # Deterministic SHA256 hashing (matches app/services/rag/embedding_service.py)
            for token, weight in weights.items():
                w = float(weight)
                if isinstance(token, str):
                    token_hash = int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:8], 16) % index_space
                    indices.append(int(token_hash))
                    values.append(w)
                else:
                    indices.append(int(token) % index_space)
                    values.append(w)

        # Ensure indices are unique; sum colliding weights
        if len(set(indices)) != len(indices):
            agg: Dict[int, float] = {}
            for i, v in zip(indices, values):
                agg[i] = agg.get(i, 0.0) + float(v)
            indices = list(agg.keys())
            values = list(agg.values())

        return SparseVector(indices=indices, values=values, mapping=mapping, index_space=index_space)

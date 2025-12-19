from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import anyio

from bge_inference_service.engine import BGEEngine, SparseVector

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _BatchItem:
    texts: List[str]  # already prepared (validated per-request)
    fut: asyncio.Future


class EmbeddingMicroBatcher:
    """Microbatch controller for embeddings endpoints.

    Goals:
    - accept concurrent HTTP requests
    - combine many small requests into fewer `encode()` calls
    - keep GPU utilization high while preserving request boundaries

    Important:
    - per-request validation uses engine.prepare_texts(..., max_batch_size=settings.max_batch_size)
    - combined batch size is controlled by microbatch settings (may exceed max_batch_size)
    """

    def __init__(
        self,
        *,
        engine: BGEEngine,
        enabled: bool,
        max_wait_ms: int,
        max_batch_texts: int,
        queue_maxsize: int,
    ) -> None:
        self._engine = engine
        self._enabled = bool(enabled)
        self._max_wait_s = max(0.0, float(max_wait_ms) / 1000.0)
        self._max_batch_texts = int(max_batch_texts)
        self._queue_maxsize = int(queue_maxsize)

        self._dense_q: asyncio.Queue[_BatchItem] = asyncio.Queue(maxsize=self._queue_maxsize)
        self._sparse_q: asyncio.Queue[_BatchItem] = asyncio.Queue(maxsize=self._queue_maxsize)
        self._hybrid_q: asyncio.Queue[_BatchItem] = asyncio.Queue(maxsize=self._queue_maxsize)

        self._tasks: List[asyncio.Task] = []
        self._closing = asyncio.Event()

    @property
    def enabled(self) -> bool:
        return self._enabled

    async def start(self) -> None:
        if not self._enabled:
            return
        if self._tasks:
            return
        loop = asyncio.get_running_loop()
        self._tasks = [
            loop.create_task(self._run_dense_loop(), name="microbatch-dense"),
            loop.create_task(self._run_sparse_loop(), name="microbatch-sparse"),
            loop.create_task(self._run_hybrid_loop(), name="microbatch-hybrid"),
        ]
        logger.info(
            "Embedding microbatcher started enabled=%s max_wait_ms=%s max_batch_texts=%s queue_maxsize=%s",
            self._enabled,
            int(self._max_wait_s * 1000),
            self._max_batch_texts,
            self._queue_maxsize,
        )

    async def stop(self) -> None:
        if not self._tasks:
            return
        self._closing.set()
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks = []

    async def embed_dense(self, texts: Sequence[str]) -> List[List[float]]:
        if not self._enabled:
            return self._engine.dense(texts)

        prepared = self._engine.prepare_texts(texts, max_batch_size=self._engine.settings.max_batch_size)
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        await self._dense_q.put(_BatchItem(texts=prepared, fut=fut))
        return await fut

    async def embed_sparse(self, texts: Sequence[str]) -> List[SparseVector]:
        if not self._enabled:
            return self._engine.sparse(texts)

        prepared = self._engine.prepare_texts(texts, max_batch_size=self._engine.settings.max_batch_size)
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        await self._sparse_q.put(_BatchItem(texts=prepared, fut=fut))
        return await fut

    async def embed_hybrid(self, texts: Sequence[str]) -> Tuple[List[List[float]], List[SparseVector]]:
        if not self._enabled:
            return self._engine.hybrid(texts)

        prepared = self._engine.prepare_texts(texts, max_batch_size=self._engine.settings.max_batch_size)
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        await self._hybrid_q.put(_BatchItem(texts=prepared, fut=fut))
        return await fut

    async def _drain_until_deadline(
        self,
        q: asyncio.Queue[_BatchItem],
    ) -> Tuple[List[_BatchItem], List[str]]:
        """Collect items up to max_wait or max_batch_texts."""
        items: List[_BatchItem] = []
        texts: List[str] = []

        if self._max_batch_texts <= 0:
            self._max_batch_texts = 1

        loop = asyncio.get_running_loop()
        start = loop.time()

        # Always wait for at least 1 item
        timeout = None if self._max_wait_s <= 0 else self._max_wait_s
        try:
            first = await asyncio.wait_for(q.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return items, texts

        items.append(first)
        texts.extend(first.texts)

        # Then opportunistically fill until deadline or capacity
        while len(texts) < self._max_batch_texts:
            remaining = self._max_wait_s - (loop.time() - start)
            if remaining <= 0:
                break
            try:
                nxt = await asyncio.wait_for(q.get(), timeout=remaining)
            except asyncio.TimeoutError:
                break
            items.append(nxt)
            texts.extend(nxt.texts)

        # Hard cap: if we overfilled due to last item, keep it (simpler) but
        # never exceed too much: split if necessary.
        if len(texts) > self._max_batch_texts and len(items) > 1:
            # Move last item back to queue for next batch.
            last = items.pop()
            # Remove last item's texts from combined
            if last.texts:
                texts = texts[: -len(last.texts)]
            try:
                q.put_nowait(last)
            except asyncio.QueueFull:
                # If queue is full, fail that last item fast
                if not last.fut.done():
                    last.fut.set_exception(RuntimeError("Microbatch queue overflow"))

        return items, texts

    async def _run_dense_loop(self) -> None:
        while not self._closing.is_set():
            items, all_texts = await self._drain_until_deadline(self._dense_q)
            if not items:
                continue

            try:
                out = await anyio.to_thread.run_sync(
                    self._engine.encode_embedder,
                    all_texts,
                    return_dense=True,
                    return_sparse=False,
                )
                dense_vecs = [list(map(float, v)) for v in out["dense_vecs"]]
                expected = int(self._engine.settings.embedding_size)
                for i, v in enumerate(dense_vecs):
                    if len(v) != expected:
                        raise RuntimeError(
                            f"Unexpected dense embedding size for item {i}: got {len(v)}, expected {expected}. "
                            f"Model={self._engine.settings.embedding_model_name}"
                        )
            except Exception as e:
                for it in items:
                    if not it.fut.done():
                        it.fut.set_exception(e)
                continue

            # Split by request boundaries
            idx = 0
            for it in items:
                n = len(it.texts)
                part = dense_vecs[idx : idx + n]
                idx += n
                if not it.fut.done():
                    it.fut.set_result(part)

    async def _run_sparse_loop(self) -> None:
        while not self._closing.is_set():
            items, all_texts = await self._drain_until_deadline(self._sparse_q)
            if not items:
                continue

            try:
                out = await anyio.to_thread.run_sync(
                    self._engine.encode_embedder,
                    all_texts,
                    return_dense=False,
                    return_sparse=True,
                )
                sparse_vecs = [self._engine.lexical_weights_to_sparse_vector(w) for w in out["lexical_weights"]]
            except Exception as e:
                for it in items:
                    if not it.fut.done():
                        it.fut.set_exception(e)
                continue

            idx = 0
            for it in items:
                n = len(it.texts)
                part = sparse_vecs[idx : idx + n]
                idx += n
                if not it.fut.done():
                    it.fut.set_result(part)

    async def _run_hybrid_loop(self) -> None:
        while not self._closing.is_set():
            items, all_texts = await self._drain_until_deadline(self._hybrid_q)
            if not items:
                continue

            try:
                out = await anyio.to_thread.run_sync(
                    self._engine.encode_embedder,
                    all_texts,
                    return_dense=True,
                    return_sparse=True,
                )
                dense_vecs = [list(map(float, v)) for v in out["dense_vecs"]]
                expected = int(self._engine.settings.embedding_size)
                for i, v in enumerate(dense_vecs):
                    if len(v) != expected:
                        raise RuntimeError(
                            f"Unexpected dense embedding size for item {i}: got {len(v)}, expected {expected}. "
                            f"Model={self._engine.settings.embedding_model_name}"
                        )
                sparse_vecs = [self._engine.lexical_weights_to_sparse_vector(w) for w in out["lexical_weights"]]
            except Exception as e:
                for it in items:
                    if not it.fut.done():
                        it.fut.set_exception(e)
                continue

            idx = 0
            for it in items:
                n = len(it.texts)
                d_part = dense_vecs[idx : idx + n]
                s_part = sparse_vecs[idx : idx + n]
                idx += n
                if not it.fut.done():
                    it.fut.set_result((d_part, s_part))



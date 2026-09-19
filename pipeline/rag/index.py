"""RAG Index: embedding and FAISS index build/load."""
from __future__ import annotations

import logging
from typing import Any, Callable

import faiss
import numpy as np
import openai

from pipeline.schema import DocumentChunk
from pipeline.config import EMBEDDING_BASE_URL, EMBEDDING_MODEL

logger = logging.getLogger(__name__)


class RAGIndex:
    """Retrieval-Augmented Generation Index over a single annual report.

    Embeddings are produced by a local CPU-only llama-server.
    The FAISS index (``IndexFlatIP`` with L2-normalised vectors for cosine
    similarity) is built once per report and cached to disk.
    """

    def __init__(
        self,
        embedding_base_url: str = EMBEDDING_BASE_URL,
        cache: Any | None = None,
    ) -> None:
        self.embed_client = openai.AsyncOpenAI(
            base_url=f"{embedding_base_url}/v1", api_key="not-needed"
        )
        self.cache = cache
        self.index: faiss.Index | None = None
        self.chunks: list[DocumentChunk] = []

    # ------------------------------------------------------------------
    # Index build / load
    # ------------------------------------------------------------------

    async def build_index(
        self, chunks: list[DocumentChunk], company_id: str, progress_callback: Callable[[int, int], None] | None = None
    ) -> None:
        """Embed all chunks and build a FAISS inner-product index.

        Args:
            chunks: Document chunks to embed.
            company_id: Slugified identifier for cache persistence.
            progress_callback: Optional callable for reporting progress.
        """
        logger.info(
            "Building FAISS index for '%s' with %d chunks", company_id, len(chunks)
        )

        texts = [c.text for c in chunks]
        embeddings = await self.embed_texts(texts, progress_callback=progress_callback)

        if not embeddings:
            logger.error("No embeddings returned – cannot build index")
            return

        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb

        self.chunks = chunks
        dim = len(embeddings[0])
        self.index = faiss.IndexFlatIP(dim)

        vectors = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(vectors)
        self.index.add(vectors)

        logger.info(
            "FAISS index ready: %d vectors, dim=%d", self.index.ntotal, dim
        )

        # Persist via cache
        if self.cache is not None:
            try:
                self.cache.save_faiss_index(company_id, self.index)
                self.cache.save_chunks(company_id, chunks)
            except Exception:
                logger.exception("Failed to persist index/chunks to cache")

    def load_index(self, company_id: str) -> bool:
        """Try to load a previously built index from cache.

        Returns:
            ``True`` if the index was loaded successfully.
        """
        if self.cache is None:
            return False

        try:
            index = self.cache.load_faiss_index(company_id)
            chunks = self.cache.load_chunks(company_id)
            if index is not None and chunks is not None:
                self.index = index
                self.chunks = chunks
                logger.info(
                    "Loaded cached index for '%s' (%d vectors, %d chunks)",
                    company_id,
                    self.index.ntotal,
                    len(self.chunks),
                )
                return True
        except Exception:
            logger.exception("Failed to load cached index for '%s'", company_id)

        return False

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    async def embed_texts(
        self, texts: list[str], batch_size: int = 32, progress_callback: Callable[[int, int], None] | None = None
    ) -> list[list[float]]:
        """Embed a list of texts via the local embedding server.

        Args:
            texts: Strings to embed.
            batch_size: Number of texts per API call.
            progress_callback: Optional callable for reporting progress.

        Returns:
            List of embedding vectors (same order as *texts*).
        """
        logger.info("Embedding %d texts in batches of %d", len(texts), batch_size)
        all_embeddings: list[list[float]] = []
        total_texts = len(texts)

        for i in range(0, total_texts, batch_size):
            batch = texts[i : i + batch_size]
            try:
                response = await self.embed_client.embeddings.create(
                    model=EMBEDDING_MODEL,
                    input=batch,
                )
                batch_embs = [d.embedding for d in response.data]
                all_embeddings.extend(batch_embs)
                logger.debug("Embedded batch %d/%d", i // batch_size + 1,
                             (total_texts + batch_size - 1) // batch_size)
            except Exception:
                logger.exception("Error embedding batch starting at index %d", i)
                # Pad with zeros so indices stay aligned.
                dummy_dim = len(all_embeddings[0]) if all_embeddings else 1024
                all_embeddings.extend([[0.0] * dummy_dim] * len(batch))

            if progress_callback:
                progress_callback(min(i + batch_size, total_texts), total_texts)

        return all_embeddings

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    async def search(self, query: str, top_k: int = 10, threshold: float = 0.3) -> list[tuple[DocumentChunk, float]]:
        """Embed *query* and return the top-k chunks that meet the threshold along with their confidence scores."""
        from pipeline.model_manager import model_manager
        
        if self.index is None or self.index.ntotal == 0:
            logger.warning("Search called on empty or uninitialized index")
            return []

        logger.info("Searching for: '%s'", query)
        
        model_manager.require_embedding()
        query_emb = await self.embed_texts([query])
        if not query_emb:
            return []

        vec = np.array(query_emb, dtype=np.float32)
        faiss.normalize_L2(vec)

        k = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(vec, k)

        results = [
            (self.chunks[idx], float(dist))
            for idx, dist in zip(indices[0], distances[0])
            if idx != -1 and dist >= threshold
        ]
        logger.info("Retrieved %d chunks (above threshold %.2f) for query", len(results), threshold)
        return results

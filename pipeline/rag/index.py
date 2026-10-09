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
        embeddings = await self.embed_texts(texts, progress_callback=progress_callback, company_id=company_id)

        if not embeddings or len(embeddings) != len(chunks):
            logger.error("Embeddings incomplete – cannot build index")
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
                
                # Cleanup checkpoint since we are done
                from pipeline.config import DATA_DIR
                from pathlib import Path
                ckpt = Path(DATA_DIR) / "cache" / f"{company_id}_embeddings_ckpt.json"
                if ckpt.exists():
                    ckpt.unlink()
            except Exception:
                logger.exception("Failed to save FAISS cache")

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
        self, texts: list[str], batch_size: int = 30, progress_callback: Callable[[int, int], None] | None = None,
        company_id: str | None = None
    ) -> list[list[float]]:
        """Embed a list of texts via the local embedding server.

        Args:
            texts: Strings to embed.
            batch_size: Number of texts per API call.
            progress_callback: Optional callable for reporting progress.
            company_id: Optional ID to enable checkpoint saving.

        Returns:
            List of embedding vectors (same order as *texts*).
        """
        import json
        from pathlib import Path
        from pipeline.config import DATA_DIR
        
        checkpoint_file = None
        if company_id:
            checkpoint_dir = Path(DATA_DIR) / "cache"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_file = checkpoint_dir / f"{company_id}_embeddings_ckpt.json"
            
        all_embeddings: list[list[float]] = []
        start_idx = 0
        total_texts = len(texts)
        
        if checkpoint_file and checkpoint_file.exists():
            try:
                with open(checkpoint_file, "r") as f:
                    all_embeddings = json.load(f)
                start_idx = len(all_embeddings)
                logger.info("Loaded %d embeddings from checkpoint. Resuming...", start_idx)
                if progress_callback and start_idx > 0:
                    progress_callback(min(start_idx, total_texts), total_texts)
            except Exception as e:
                logger.warning("Failed to load embedding checkpoint: %s", e)
                all_embeddings = []
                start_idx = 0

        if start_idx >= total_texts:
            return all_embeddings[:total_texts]

        logger.info("Embedding %d remaining texts in batches of %d", total_texts - start_idx, batch_size)

        for i in range(start_idx, total_texts, batch_size):
            batch = texts[i : i + batch_size]
            
            # TRUNCATE: Since we confirmed the computer runs out of memory on huge chunks,
            # we must slice anomalous chunks down to ~4,000 tokens so it survives.
            safe_batch = [text[:16000] for text in batch]
            
            try:
                response = await self.embed_client.embeddings.create(
                    model=EMBEDDING_MODEL,
                    input=safe_batch,
                )
                batch_embs = [d.embedding for d in response.data]
                all_embeddings.extend(batch_embs)
                
                # Save checkpoint immediately
                if checkpoint_file:
                    with open(checkpoint_file, "w") as f:
                        json.dump(all_embeddings, f)
                        
                logger.debug("Embedded batch %d/%d", i // batch_size + 1,
                             (total_texts + batch_size - 1) // batch_size)
            except Exception as e:
                logger.exception("CRITICAL: Error embedding batch starting at index %d. Stopping to preserve progress.", i)
                raise e # Stop execution so we don't pad zeros, allowing a safe resume later!

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

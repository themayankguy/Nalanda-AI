"""Vector index and local embedding management for Nalanda AI.

CRITICAL REQUIREMENTS:
- Uses local sentence-transformers/all-MiniLM-L6-v2 model.
- Strictly partitioned by user_id and folder_id.
- Persisted on disk across application restarts.
- Extracted text remains internal and is never returned to frontend callers.
"""
import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

logger = logging.getLogger("nalanda.vector_index")

# Model identifier
DEFAULT_EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# Global lazy-loaded model instance
_model_instance = None


def get_embedding_model(offline_only: bool = False):
    """Retrieve or lazily initialize the SentenceTransformer model instance.

    If offline_only is True, prevents external network requests.
    """
    global _model_instance
    if _model_instance is not None:
        return _model_instance

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "The 'sentence-transformers' library is not installed. "
            "Please run: pip install sentence-transformers"
        ) from exc

    try:
        logger.info("Initializing embedding model: %s", DEFAULT_EMBEDDING_MODEL_NAME)
        # If offline mode requested, force local files only
        kwargs = {}
        if offline_only or os.environ.get("NALANDA_OFFLINE_MODE", "0") == "1":
            kwargs["local_files_only"] = True

        _model_instance = SentenceTransformer(DEFAULT_EMBEDDING_MODEL_NAME, **kwargs)
        return _model_instance
    except Exception as exc:
        logger.error("Failed to load embedding model %s: %s", DEFAULT_EMBEDDING_MODEL_NAME, exc)
        raise RuntimeError(
            f"Embedding model '{DEFAULT_EMBEDDING_MODEL_NAME}' could not be loaded. "
            f"Error details: {exc}"
        ) from exc


def get_folder_index_dir(base_storage_dir: Path, user_id: str, folder_id: str) -> Path:
    """Generate isolated directory path for a folder's vector index."""
    index_dir = base_storage_dir / "users" / user_id / "folders" / folder_id / "vector_index"
    index_dir.mkdir(parents=True, exist_ok=True)
    return index_dir


class FolderVectorIndex:
    """Manages an isolated vector index strictly bound to a single user and folder."""

    def __init__(self, index_dir: Path):
        self.index_dir = index_dir
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = index_dir / "metadata.json"
        self.embeddings_file = index_dir / "embeddings.npy"
        self.chunks: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        self._load()

    def _load(self) -> None:
        """Load index and chunk metadata from disk if available."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r", encoding="utf-8") as f:
                    self.chunks = json.load(f)
            except Exception as exc:
                logger.warning("Error loading metadata from %s: %s", self.metadata_file, exc)
                self.chunks = []

        if self.embeddings_file.exists():
            try:
                self.embeddings = np.load(str(self.embeddings_file))
                if len(self.embeddings) != len(self.chunks):
                    logger.warning("Mismatch between embeddings count and metadata chunks. Rebuilding needed.")
            except Exception as exc:
                logger.warning("Error loading embeddings from %s: %s", self.embeddings_file, exc)
                self.embeddings = None

    def _save(self) -> None:
        """Persist chunk metadata and embeddings atomically to disk."""
        tmp_meta = self.metadata_file.with_suffix(".tmp")
        with open(tmp_meta, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)
        tmp_meta.replace(self.metadata_file)

        if self.embeddings is not None and len(self.embeddings) > 0:
            tmp_emb = self.embeddings_file.with_suffix(".tmp.npy")
            np.save(str(tmp_emb), self.embeddings)
            tmp_emb.replace(self.embeddings_file)
        elif self.embeddings_file.exists():
            self.embeddings_file.unlink()

    def add_document_chunks(self, new_chunks: List[Dict[str, Any]], model=None) -> int:
        """Add or update chunks for a document in this folder's vector index.

        Avoids recalculating embeddings for unchanged chunks (verified via text_hash).
        """
        if not new_chunks:
            return 0

        if model is None:
            model = get_embedding_model()

        doc_id = new_chunks[0]["document_id"]

        # Filter out existing chunks for this document
        existing_other_chunks = []
        existing_other_embeddings = []
        existing_doc_hashes = {}

        if self.embeddings is not None and len(self.embeddings) == len(self.chunks):
            for idx, c in enumerate(self.chunks):
                if c.get("document_id") == doc_id:
                    existing_doc_hashes[c.get("text_hash")] = self.embeddings[idx]
                else:
                    existing_other_chunks.append(c)
                    existing_other_embeddings.append(self.embeddings[idx])

        # Prepare new embeddings (reusing cached vectors if unchanged)
        chunks_to_encode_texts = []
        chunks_to_encode_indices = []
        final_doc_embeddings = [None] * len(new_chunks)

        for i, chunk in enumerate(new_chunks):
            thash = chunk.get("text_hash")
            if thash in existing_doc_hashes:
                final_doc_embeddings[i] = existing_doc_hashes[thash]
            else:
                chunks_to_encode_texts.append(chunk["text"])
                chunks_to_encode_indices.append(i)

        if chunks_to_encode_texts:
            encoded_vectors = model.encode(
                chunks_to_encode_texts,
                batch_size=32,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            for orig_idx, vec in zip(chunks_to_encode_indices, encoded_vectors):
                final_doc_embeddings[orig_idx] = vec

        # Combine existing other chunks with newly added chunks
        combined_chunks = existing_other_chunks + new_chunks
        if existing_other_embeddings:
            combined_embeddings = np.vstack([
                np.array(existing_other_embeddings, dtype=np.float32),
                np.array(final_doc_embeddings, dtype=np.float32),
            ])
        else:
            combined_embeddings = np.array(final_doc_embeddings, dtype=np.float32)

        self.chunks = combined_chunks
        self.embeddings = combined_embeddings
        self._save()
        return len(new_chunks)

    def remove_document(self, document_id: str) -> None:
        """Remove all vectors and metadata associated with a document."""
        if not self.chunks:
            return

        remaining_chunks = []
        remaining_indices = []
        for idx, c in enumerate(self.chunks):
            if c.get("document_id") != document_id:
                remaining_chunks.append(c)
                remaining_indices.append(idx)

        if len(remaining_chunks) == len(self.chunks):
            return  # Document was not in index

        self.chunks = remaining_chunks
        if self.embeddings is not None and remaining_indices:
            self.embeddings = self.embeddings[remaining_indices]
        else:
            self.embeddings = None
        self._save()

    def search(self, query: str, top_k: int = 5, model=None) -> List[Tuple[Dict[str, Any], float]]:
        """Perform vectorized cosine similarity search against this folder's vectors.

        Returns list of (chunk_dict, relevance_score) sorted by score descending.
        """
        if self.embeddings is None or len(self.chunks) == 0:
            return []

        if model is None:
            model = get_embedding_model()

        query_vector = model.encode([query], normalize_embeddings=True)[0]

        # Fast vectorized dot product (equivalent to cosine similarity for normalized vectors)
        scores = np.dot(self.embeddings, query_vector)

        # Get top-k indices
        top_k = min(top_k, len(self.chunks))
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            raw_score = float(scores[idx])
            # Clamp cosine similarity to clean [0.0, 1.0] relevance range
            relevance = max(0.0, min(1.0, (raw_score + 1.0) / 2.0 if raw_score < 0 else raw_score))
            results.append((self.chunks[idx], round(relevance, 4)))

        return results

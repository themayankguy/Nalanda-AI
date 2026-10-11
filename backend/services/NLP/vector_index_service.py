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
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

logger = logging.getLogger("nalanda.vector_index")


def infer_question_type(query: str) -> str:
    """Infer the likely retrieval intent for a conceptual question."""
    clean_q = (query or "").lower().strip()
    if not clean_q:
        return "conceptual"

    if re.search(r"\b(where|when|used|applied|application|applications|example|examples|implemented|deployment)\b", clean_q):
        return "application"
    if re.search(r"\b(how|works|working|mechanism|process|algorithm|steps|operation|function)\b", clean_q):
        return "working"
    if re.search(r"\b(what is|what are|what does|define|definition|overview|introduction|describe|explain|meaning|purpose)\b", clean_q):
        return "definition"
    return "conceptual"


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


INDEX_SCHEMA_VERSION = 13


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
                    data = json.load(f)
                    if isinstance(data, dict) and "chunks" in data:
                        if data.get("version") == INDEX_SCHEMA_VERSION:
                            self.chunks = data["chunks"]
                        else:
                            logger.info("Index schema mismatch in %s. Upgrade needed.", self.metadata_file)
                            self.chunks = []
                    elif isinstance(data, list):
                        # Legacy schema list -> upgrade needed
                        logger.info("Legacy index detected in %s. Upgrading to schema v%d.", self.metadata_file, INDEX_SCHEMA_VERSION)
                        self.chunks = []
            except Exception as exc:
                logger.warning("Error loading metadata from %s: %s", self.metadata_file, exc)
                self.chunks = []

        if self.embeddings_file.exists() and len(self.chunks) > 0:
            try:
                self.embeddings = np.load(str(self.embeddings_file))
                if len(self.embeddings) != len(self.chunks):
                    logger.warning("Mismatch between embeddings count and metadata chunks. Rebuilding needed.")
                    self.embeddings = None
                    self.chunks = []
            except Exception as exc:
                logger.warning("Error loading embeddings from %s: %s", self.embeddings_file, exc)
                self.embeddings = None
        else:
            self.embeddings = None

    def _save(self) -> None:
        """Persist chunk metadata and embeddings atomically to disk."""
        tmp_meta = self.metadata_file.with_suffix(".tmp")
        payload = {
            "version": INDEX_SCHEMA_VERSION,
            "chunks": self.chunks,
        }
        with open(tmp_meta, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
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
        """Perform vectorized cosine similarity search combined with lexical and intent alignment.

        Returns list of (chunk_dict, relevance_score) sorted by score descending.
        """
        if self.embeddings is None or len(self.chunks) == 0:
            return []

        if model is None:
            model = get_embedding_model()

        query_vector = model.encode([query], normalize_embeddings=True)[0]

        # Fast vectorized dot product (cosine similarity for normalized vectors)
        dot_scores = np.dot(self.embeddings, query_vector)
        # Map cosine similarity [-1.0, 1.0] smoothly into [0.0, 1.0]
        dense_scores = np.clip((dot_scores + 1.0) / 2.0, 0.0, 1.0)

        # Lexical and intent analysis
        clean_q = query.lower().strip()
        stop_words = {
            "what", "which", "show", "tell", "explain", "about", "from", "with",
            "this", "that", "does", "have", "been", "the", "are", "and", "is",
            "why", "how", "who", "when", "where", "can", "you", "for", "the",
            "meaning", "define", "give", "all", "its", "also", "into", "onto"
        }
        q_tokens = set(re.findall(r"\b\w{3,}\b", clean_q)) - stop_words
        question_type = infer_question_type(clean_q)
        target_phrase = re.sub(r"^(what is|what are|what does|define|definition of|meaning of|overview of|explain|describe)\s+", "", clean_q, flags=re.I).rstrip("?").strip()

        # Differentiate substantive subject entities from generic structural words
        generic_structure_words = {
            "step", "steps", "mode", "modes", "flow", "flows", "process", "procedure",
            "stages", "phases", "working", "mechanism", "mechanisms", "architecture",
            "diagram", "diagrams", "figure", "figures", "table", "tables",
            "difference", "differences", "comparison", "versus", "overview", "introduction",
            "type", "types", "feature", "features", "component", "components",
            "method", "methods", "technique", "techniques"
        }
        subject_tokens = q_tokens - generic_structure_words

        final_scores = np.zeros(len(self.chunks), dtype=np.float32)

        for idx, c in enumerate(self.chunks):
            dense_s = float(dense_scores[idx])
            sec_title = (c.get("section_title") or "")
            sec_title_lower = sec_title.lower()
            chunk_text = (c.get("text") or "")
            chunk_text_lower = chunk_text.lower()
            chunk_head = chunk_text[:250].lower()

            title_tokens = set(re.findall(r"\b\w{3,}\b", sec_title_lower)) if sec_title else set()
            head_tokens = set(re.findall(r"\b\w{3,}\b", chunk_head))
            text_tokens = set(re.findall(r"\b\w{3,}\b", chunk_text_lower))

            # 1. Lexical match score
            lexical_score = 0.0
            phrase_bonus = 0.0
            if q_tokens:
                title_match = len(q_tokens & title_tokens) / len(q_tokens) if title_tokens else 0.0
                head_match = len(q_tokens & head_tokens) / len(q_tokens) if head_tokens else 0.0
                body_match = len(q_tokens & text_tokens) / len(q_tokens) if text_tokens else 0.0

                lexical_score = 0.40 * title_match + 0.35 * head_match + 0.25 * body_match

                # Verbatim multi-word phrase matching
                if target_phrase and len(target_phrase) >= 4:
                    if target_phrase in sec_title_lower:
                        phrase_bonus += 0.35
                    if target_phrase in chunk_text_lower:
                        phrase_bonus += 0.25

            # 2. Subject entity alignment
            matched_subj_body = (subject_tokens & text_tokens) if subject_tokens else set()
            matched_subj_title = (subject_tokens & title_tokens) if subject_tokens else set()

            subject_bonus = 0.0
            has_subject_conflict = False
            if subject_tokens:
                if not matched_subj_body and not matched_subj_title:
                    # Query specified concrete subjects (e.g. 'hive'), but chunk does not mention any of them!
                    has_subject_conflict = True
                    subject_bonus = -0.30
                else:
                    if matched_subj_title:
                        if question_type == "definition" and re.search(r"\b(why|reasons? for)\b", sec_title_lower):
                            pass
                        else:
                            subject_bonus += 0.25 * (len(matched_subj_title) / len(subject_tokens))
                    if matched_subj_body:
                        subject_bonus += 0.15 * (len(matched_subj_body) / len(subject_tokens))

            # 3. Intent-specific alignment bonus (guarded by subject relevance)
            intent_bonus = 0.0
            if not has_subject_conflict:
                if question_type == "definition":
                    target_words = set(re.findall(r"\b\w+\b", target_phrase)) - stop_words
                    is_intro_sec = any(m in sec_title_lower for m in ["introduction", "overview", "definition", "concept", "meaning"]) or re.match(r"^(what is|what are|define)\b", sec_title_lower)
                    if is_intro_sec:
                        if not target_words or any(tw in sec_title_lower for tw in target_words) or any(tw in chunk_head for tw in target_words):
                            intent_bonus += 0.25

                    op_terms = ["reusing", "running", "stopping", "commands for", "lifecycle", "deleting", "updating", "installing"]
                    if any(op in sec_title_lower for op in op_terms) and not any(op in clean_q for op in op_terms):
                        intent_bonus -= 0.15

                    has_def_in_chunk = False
                    if target_words:
                        head_words = set(re.findall(r"\b\w+\b", chunk_head[:100]))
                        if target_words & head_words:
                            def_markers = ["is a ", "is an ", "is the ", "refers to", "stands for", "means", "is defined", "is the process of", "is a platform"]
                            if any(marker in chunk_head for marker in def_markers):
                                intent_bonus += 0.20
                                has_def_in_chunk = True

                    if re.search(r"\b(why|reasons? for)\b", sec_title_lower):
                        intent_bonus -= 0.25

                    if target_phrase and target_phrase in sec_title_lower and not re.search(r"\b(why|reasons? for)\b", sec_title_lower):
                        intent_bonus += 0.15

                elif question_type == "working":
                    if any(m in sec_title_lower for m in ["steps", "procedure", "phases", "stages", "working", "mechanism", "process", "architecture", "algorithm", "how", "modes"]):
                        intent_bonus += 0.22
                    elif any(m in chunk_head for m in ["steps", "works by", "process of", "mechanism", "architecture", "algorithm"]):
                        intent_bonus += 0.12
                    if re.search(r"\b(steps?|procedure)\b", clean_q) and not re.search(r"\b(steps?|procedure)\b", sec_title_lower):
                        intent_bonus -= 0.10

                elif question_type == "application":
                    if any(m in sec_title_lower for m in ["application", "applications", "usage", "use case", "example", "where"]):
                        intent_bonus += 0.18
                    elif any(m in chunk_head for m in ["used in", "applied in", "application", "use case"]):
                        intent_bonus += 0.10

                # General topic match bonus: only if subject is not conflicted
                if re.search(r"\b(modes?|execution)\b", clean_q) and re.search(r"\b(modes?|execution)\b", sec_title_lower):
                    intent_bonus += 0.20

            # Penalize question banks or exam question lists masquerading as content
            exam_markers = len(re.findall(r"\bQ\d+[\(\.]|\b(?:Question|Ans|\[\d+M\])\b", chunk_text))
            if exam_markers >= 2:
                intent_bonus -= 0.35

            # Blended hybrid score
            comb_score = 0.50 * dense_s + 0.35 * lexical_score + phrase_bonus + intent_bonus + subject_bonus
            final_scores[idx] = comb_score

        # Get top-k indices based on true unclipped scores (avoids saturation ties)
        top_k = min(top_k, len(self.chunks))
        top_indices = np.argsort(final_scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(final_scores[idx])
            scaled_score = min(1.0, max(0.0, score))
            results.append((self.chunks[idx], round(scaled_score, 4)))

        return results

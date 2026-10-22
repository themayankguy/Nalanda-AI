"""Incremental Knowledge Integration Service for Nalanda AI.

Implements multi-stage matching, duplicate detection, concept enrichment,
subtopic extension, conflict tracking, and strict provenance preservation.
Uses 100% local NLP, MiniLM embeddings, and SQLite storage.
"""
import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set

import numpy as np
from sqlalchemy.orm import Session

from backend.models import (
    Document,
    Folder,
    KnowledgeConcept,
    KnowledgeItem,
    DocumentIngestionLog,
    DocumentExtraction,
)
from backend.services.NLP.vector_index_service import (
    get_embedding_model,
    get_folder_index_dir,
    FolderVectorIndex,
)
from backend.services.NLP.document_chunking_service import chunk_document_data
from backend.services.storage_service import BASE_STORAGE_DIR

logger = logging.getLogger("nalanda.knowledge_integration")


def utcnow():
    return datetime.now(timezone.utc)


def normalize_text_for_comparison(text: str) -> str:
    """Normalize text by lowercasing, stripping punctuation/markdown, and collapsing whitespace."""
    if not text:
        return ""
    # Strip markdown headers, bold, bullets
    t = re.sub(r"^[#\*\-\s\d\.\)]+", "", text, flags=re.MULTILINE)
    t = re.sub(r"[\*\_`~]", "", t)
    # Normalize common OCR character confusions in word bodies
    t = re.sub(r"(?<=[a-zA-Z])1(?=[a-zA-Z])", "l", t)
    t = re.sub(r"(?<=[a-zA-Z])0(?=[a-zA-Z])", "o", t)
    # Collapse whitespace and convert to lowercase
    t = re.sub(r"\s+", " ", t).strip().lower()
    return t


def compute_normalized_hash(text: str) -> str:
    """Deterministic SHA-256 hash of normalized text for exact duplicate detection."""
    norm = normalize_text_for_comparison(text)
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def compute_token_jaccard(text1: str, text2: str) -> float:
    """Token-level Jaccard similarity over words of length >= 3."""
    t1 = set(re.findall(r"\b\w{3,}\b", normalize_text_for_comparison(text1)))
    t2 = set(re.findall(r"\b\w{3,}\b", normalize_text_for_comparison(text2)))
    if not t1 or not t2:
        return 0.0
    return len(t1 & t2) / len(t1 | t2)


def extract_canonical_concept_name(title: Optional[str], fallback_text: str = "") -> str:
    """Extract a canonical concept name by stripping slide numbers, chapter indices, and filler words."""
    raw = (title or "").strip()
    if not raw and fallback_text:
        # Use first line or heading if available
        first_line = fallback_text.strip().split("\n")[0]
        raw = first_line[:60].strip()

    if not raw:
        return "Core Concept"

    # Remove slide/page references e.g. "Slide 3", "Page 12"
    clean = re.sub(r"^(?:slide|page|unit|chapter)\s+\d+[:\-\s]*", "", raw, flags=re.I)
    # Remove numbering like "1.2 ", "3.1.4 "
    clean = re.sub(r"^\d+(?:\.\d+)*\s*[:\-\.]*\s*", "", clean)
    # Remove markdown headers
    clean = re.sub(r"^#+\s*", "", clean)
    # Remove common filler prefixes
    clean = re.sub(
        r"^(?:introduction to|overview of|understanding|functions of the|responsibilities of the|functions of|responsibilities of|fundamentals of|basics of)\s+",
        "",
        clean,
        flags=re.I,
    )
    clean = clean.strip(" :-\t\n")
    return clean if len(clean) >= 2 else (raw if len(raw) >= 2 else "Core Concept")


def normalize_concept_name(name: str) -> str:
    """Normalized concept name for canonical index matching."""
    norm = normalize_text_for_comparison(name)
    # Strip common suffixes when matching concept identity
    norm = re.sub(r"\b(concept|overview|introduction|fundamentals|notes?)\b", "", norm).strip()
    return norm


def are_same_concept_entities(name1: str, name2: str) -> bool:
    """Check if two concept names refer to the exact same substantive entity or topic."""
    norm1 = normalize_concept_name(name1)
    norm2 = normalize_concept_name(name2)
    if not norm1 or not norm2:
        return False
    if norm1 == norm2:
        return True

    words1 = set(re.findall(r"\b[a-zA-Z]{3,}\b", norm1))
    words2 = set(re.findall(r"\b[a-zA-Z]{3,}\b", norm2))

    stop_words = {
        "overview", "architecture", "functions", "responsibilities", "system", "systems",
        "model", "models", "algorithm", "algorithms", "classifier", "classifiers",
        "deep", "dive", "intro", "introduction"
    }
    core1 = words1 - stop_words
    core2 = words2 - stop_words

    if not core1 or not core2:
        return len(words1 & words2) >= 1

    if core1 == core2:
        return True
    if len(core1) > 1 and len(core2) > 1 and (core1.issubset(core2) or core2.issubset(core1)):
        return True
    return False



def detect_contradiction(text1: str, text2: str) -> Optional[str]:
    """Detect potential factual contradictions between two statements on the same subtopic.
    
    Checks for:
    1. Direct polar antonyms / negation flags (can vs cannot, synchronous vs asynchronous, etc.)
    2. Contradicting numerical metrics on the same attribute (e.g. block size 64MB vs 128MB).
    """
    t1_lower = text1.lower()
    t2_lower = text2.lower()

    # Negation pairs
    polar_pairs = [
        (r"\bcan\b", r"\bcannot\b|\bcan not\b"),
        (r"\bis synchronous\b|\bsynchronously\b", r"\bis asynchronous\b|\basynchronously\b"),
        (r"\bcentralized\b", r"\bdecentralized\b|\bdistributed\b"),
        (r"\bsingle-threaded\b", r"\bmulti-threaded\b"),
        (r"\bread-only\b", r"\bread-write\b"),
        (r"\blossless\b", r"\blossy\b"),
        (r"\bmutable\b", r"\bimmutable\b"),
        (r"\ballows?\b", r"\bprohibits?\b|\bdisallows?\b|\bprevents?\b"),
    ]

    for p1, p2 in polar_pairs:
        if (re.search(p1, t1_lower) and re.search(p2, t2_lower)) or (re.search(p2, t1_lower) and re.search(p1, t2_lower)):
            return f"Polar assertion conflict detected between '{p1}' and '{p2}'."

    # Conflicting numerical values on identical metric keywords
    metric_pattern = r"\b(default|size|limit|port|maximum|minimum|timeout|factor|threshold|capacity|replicas?)\b[^\.\n]{0,35}?(\d+(?:\.\d+)?\s*(?:mb|gb|tb|ms|s|sec|seconds?|minutes?|hrs?|hours?|nodes?|blocks?|threads?|%|percent)?)"
    m1 = re.findall(metric_pattern, t1_lower)
    m2 = re.findall(metric_pattern, t2_lower)

    if m1 and m2:
        for kw1, val1 in m1:
            for kw2, val2 in m2:
                if kw1 == kw2 and val1.strip() != val2.strip():
                    return f"Conflicting metric values for '{kw1}': '{val1.strip()}' vs '{val2.strip()}'."

    return None


class KnowledgeIntegrationService:
    """Coordinates incremental knowledge merging, multi-stage matching, and provenance tracking."""

    @classmethod
    def integrate_document_knowledge(
        cls,
        db: Session,
        document: Document,
        chunks: Optional[List[Dict[str, Any]]] = None,
        model=None,
    ) -> Dict[str, Any]:
        """Integrate an extracted document's chunks into the folder's existing knowledge structure.
        
        Guarantees:
        - Exact duplicates are linked to existing items with provenance and skipped from duplicate creation.
        - Additions to existing concepts are attached as subtopics/knowledge items under the existing concept.
        - Genuinely new concepts create new concept nodes.
        - Potential factual conflicts are preserved and flagged.
        - Full idempotency: safe to retry without creating duplicates.
        """
        user_id = document.user_id
        folder_id = document.folder_id
        doc_id = document.id

        if model is None:
            model = get_embedding_model()

        # 1. Fetch chunks if not provided
        if not chunks:
            extraction = db.query(DocumentExtraction).filter(DocumentExtraction.document_id == doc_id).first()
            json_path = Path(extraction.json_storage_path) if extraction and extraction.json_storage_path else None
            fallback_text = extraction.full_text if extraction else None
            chunks = chunk_document_data(
                document_id=doc_id,
                user_id=user_id,
                folder_id=folder_id,
                source_filename=document.original_filename,
                json_path=json_path,
                fallback_text=fallback_text,
            )

        if not chunks:
            logger.info("No chunks to integrate for document %s", doc_id)
            summary = {
                "document_id": doc_id,
                "folder_id": folder_id,
                "items_matched": 0,
                "concepts_enriched": 0,
                "concepts_created": 0,
                "duplicates_skipped": 0,
                "conflicts_flagged": 0,
                "summary_message": "No extracted text found in document.",
            }
            cls._save_ingestion_log(db, document, summary)
            return summary

        # 2. Idempotency cleanup: remove any knowledge items previously contributed exclusively by this document
        cls._cleanup_document_prior_items(db, doc_id, folder_id, user_id)

        # 3. Load all existing concepts and items in the same folder for this user
        existing_concepts = (
            db.query(KnowledgeConcept)
            .filter(
                KnowledgeConcept.folder_id == folder_id,
                KnowledgeConcept.user_id == user_id,
            )
            .all()
        )

        existing_items = (
            db.query(KnowledgeItem)
            .filter(
                KnowledgeItem.folder_id == folder_id,
                KnowledgeItem.user_id == user_id,
            )
            .all()
        )

        # Map existing normalized hashes for exact duplicate checks
        existing_hash_map: Dict[str, KnowledgeItem] = {
            item.normalized_hash: item for item in existing_items
        }

        # Embeddings for existing concepts (computed from concept name + summary + items)
        concept_embeddings_map: Dict[str, np.ndarray] = {}
        for c in existing_concepts:
            concept_texts = [c.name]
            if c.summary:
                concept_texts.append(c.summary)
            for itm in c.items[:4]:
                concept_texts.append(itm.content[:200])
            c_text = " ".join(concept_texts)
            c_vec = model.encode([c_text], normalize_embeddings=True)[0]
            concept_embeddings_map[c.id] = c_vec

        # Counters for integration summary
        items_matched_count = 0
        concepts_enriched_count = 0
        concepts_created_count = 0
        duplicates_skipped_count = 0
        conflicts_flagged_count = 0

        # Track concepts created or enriched in this run to avoid internal duplicates
        session_concepts: Dict[str, KnowledgeConcept] = {c.normalized_name: c for c in existing_concepts}

        # Process each chunk as a knowledge item
        for chunk in chunks:
            chunk_text = (chunk.get("text") or "").strip()
            if len(chunk_text) < 15:
                continue

            sec_title = chunk.get("section_title")
            page_num = chunk.get("page_number")
            c_hash = compute_normalized_hash(chunk_text)

            # --- STAGE A: Exact & Near-Exact Duplicate Detection ---
            matched_dup_item: Optional[KnowledgeItem] = None

            if c_hash in existing_hash_map:
                matched_dup_item = existing_hash_map[c_hash]
            else:
                # Fast near-duplicate check via token similarity with existing items
                for ext_item in existing_items:
                    if compute_token_jaccard(chunk_text, ext_item.content) >= 0.94:
                        matched_dup_item = ext_item
                        break

            if matched_dup_item:
                # Case A: Exact or near-exact duplicate found!
                # Update provenance: add source reference to existing item without creating a duplicate node
                sources = []
                if matched_dup_item.provenance_sources_json:
                    try:
                        sources = json.loads(matched_dup_item.provenance_sources_json)
                    except Exception:
                        sources = []
                
                new_source = {
                    "document_id": doc_id,
                    "filename": document.original_filename,
                    "page": page_num,
                    "chunk_id": chunk.get("chunk_id"),
                }
                # Check if this document is already in provenance
                if not any(s.get("document_id") == doc_id and s.get("page") == page_num for s in sources):
                    sources.append(new_source)
                    matched_dup_item.provenance_sources_json = json.dumps(sources)
                    matched_dup_item.updated_at = utcnow()

                duplicates_skipped_count += 1
                items_matched_count += 1
                continue

            # --- STAGE B & C: Concept Identity and Semantic Matching ---
            chunk_vec = model.encode([chunk_text], normalize_embeddings=True)[0]
            raw_concept_name = extract_canonical_concept_name(sec_title, chunk_text)
            norm_concept_name = normalize_concept_name(raw_concept_name)

            best_concept: Optional[KnowledgeConcept] = None
            best_sim = 0.0
            has_explicit_title = bool(sec_title and len(sec_title.strip()) > 3 and not re.match(r"^(?:slide|page|unit)\s*\d+$", sec_title.strip(), re.I))

            # 1. Check entity identity against existing/session concepts
            for c_obj in session_concepts.values():
                if are_same_concept_entities(raw_concept_name, c_obj.name):
                    best_concept = c_obj
                    best_sim = 1.0
                    break

            # 2. If no direct entity match, only check semantic similarity if chunk does NOT have a distinct explicit title
            if not best_concept and concept_embeddings_map and not has_explicit_title:
                for c_id, c_vec in concept_embeddings_map.items():
                    sim = float(np.dot(chunk_vec, c_vec))
                    if sim > best_sim:
                        best_sim = sim
                        for c_obj in session_concepts.values():
                            if c_obj.id == c_id:
                                best_concept = c_obj
                                break

            # Determine decision based on calibrated thresholds
            target_concept: Optional[KnowledgeConcept] = None
            is_new_concept = False

            if best_concept and best_sim >= 0.72:
                # Case B or C: Matches an existing concept!
                target_concept = best_concept
                concepts_enriched_count += 1
                items_matched_count += 1
            else:
                # Case D: Genuinely new concept!
                target_concept = KnowledgeConcept(
                    id=str(Path(doc_id).stem) + f"_c_{len(session_concepts)}",
                    folder_id=folder_id,
                    user_id=user_id,
                    name=raw_concept_name,
                    normalized_name=norm_concept_name if norm_concept_name else f"concept_{len(session_concepts)}",
                    summary=chunk_text[:300],
                    created_at=utcnow(),
                    updated_at=utcnow(),
                )
                db.add(target_concept)
                db.flush()
                session_concepts[target_concept.normalized_name] = target_concept
                concept_embeddings_map[target_concept.id] = chunk_vec
                concepts_created_count += 1
                is_new_concept = True

            # Determine appropriate subtopic
            subtopic_name = "General"
            m_sub = re.match(r"^\s*([A-Za-z0-9\s\-]{3,35})\s*:\s*", chunk_text)
            if m_sub and m_sub.group(1).strip().lower() != target_concept.name.lower():
                subtopic_name = m_sub.group(1).strip()
            elif any(w in chunk_text.lower() for w in ["limitations", "drawbacks", "cons", "challenges"]):
                subtopic_name = "Limitations"
            elif any(w in chunk_text.lower() for w in ["advantages", "benefits", "pros"]):
                subtopic_name = "Advantages"
            elif any(w in chunk_text.lower() for w in ["responsibilities", "functions", "role"]):
                subtopic_name = "Responsibilities"
            elif any(w in chunk_text.lower() for w in ["architecture", "components"]):
                subtopic_name = "Architecture"
            elif sec_title and len(sec_title.strip()) > 3:
                clean_sec = re.sub(r"^[#\*\-\s\d\.\)]+", "", sec_title).strip()
                if clean_sec.lower() != target_concept.name.lower():
                    subtopic_name = clean_sec


            # Check for conflicting statements against existing items in this concept
            is_conflict = False
            conflict_details_str: Optional[str] = None
            for ext_item in target_concept.items:
                conflict_reason = detect_contradiction(chunk_text, ext_item.content)
                if conflict_reason:
                    is_conflict = True
                    conflicts_flagged_count += 1
                    ext_item.is_conflict = True
                    conflict_meta = {
                        "conflicting_document_id": ext_item.document_id,
                        "conflicting_filename": ext_item.source_filename,
                        "conflicting_page": ext_item.page_number,
                        "conflicting_content": ext_item.content[:250],
                        "current_content": chunk_text[:250],
                        "reason": conflict_reason,
                    }
                    conflict_details_str = json.dumps(conflict_meta)
                    break

            # Create the KnowledgeItem with complete provenance
            initial_sources = [{
                "document_id": doc_id,
                "filename": document.original_filename,
                "page": page_num,
                "chunk_id": chunk.get("chunk_id"),
            }]

            new_item = KnowledgeItem(
                concept_id=target_concept.id,
                folder_id=folder_id,
                user_id=user_id,
                document_id=doc_id,
                chunk_id=chunk.get("chunk_id"),
                subtopic=subtopic_name,
                content=chunk_text,
                normalized_hash=c_hash,
                page_number=page_num,
                source_filename=document.original_filename,
                provenance_sources_json=json.dumps(initial_sources),
                is_conflict=is_conflict,
                conflict_details=conflict_details_str,
                created_at=utcnow(),
                updated_at=utcnow(),
            )
            db.add(new_item)
            existing_items.append(new_item)
            existing_hash_map[c_hash] = new_item

            # Update concept summary if new item provides richer context
            if not is_new_concept and target_concept.summary and len(target_concept.summary) < 250:
                target_concept.summary = (target_concept.summary + " " + chunk_text)[:450]
                target_concept.updated_at = utcnow()

        db.commit()

        # Build concise user-facing integration summary message
        summary_msg = (
            f"Upload processed successfully: {items_matched_count} existing knowledge items matched, "
            f"{concepts_enriched_count} existing concepts enriched, {concepts_created_count} new concepts created, "
            f"{duplicates_skipped_count} duplicate items skipped, {conflicts_flagged_count} potentially conflicting statements flagged."
        )

        summary_data = {
            "document_id": doc_id,
            "folder_id": folder_id,
            "items_matched": items_matched_count,
            "concepts_enriched": concepts_enriched_count,
            "concepts_created": concepts_created_count,
            "duplicates_skipped": duplicates_skipped_count,
            "conflicts_flagged": conflicts_flagged_count,
            "summary_message": summary_msg,
        }

        cls._save_ingestion_log(db, document, summary_data)
        logger.info("Knowledge integration completed for document %s: %s", doc_id, summary_msg)
        return summary_data

    @classmethod
    def _save_ingestion_log(cls, db: Session, document: Document, summary: Dict[str, Any]) -> None:
        """Persist or update DocumentIngestionLog for this document."""
        log = (
            db.query(DocumentIngestionLog)
            .filter(DocumentIngestionLog.document_id == document.id)
            .first()
        )
        if not log:
            log = DocumentIngestionLog(
                document_id=document.id,
                folder_id=document.folder_id,
                user_id=document.user_id,
                items_matched=summary.get("items_matched", 0),
                concepts_enriched=summary.get("concepts_enriched", 0),
                concepts_created=summary.get("concepts_created", 0),
                duplicates_skipped=summary.get("duplicates_skipped", 0),
                conflicts_flagged=summary.get("conflicts_flagged", 0),
                summary_message=summary.get("summary_message", ""),
                created_at=utcnow(),
            )
            db.add(log)
        else:
            log.items_matched = summary.get("items_matched", 0)
            log.concepts_enriched = summary.get("concepts_enriched", 0)
            log.concepts_created = summary.get("concepts_created", 0)
            log.duplicates_skipped = summary.get("duplicates_skipped", 0)
            log.conflicts_flagged = summary.get("conflicts_flagged", 0)
            log.summary_message = summary.get("summary_message", "")
        db.commit()

    @classmethod
    def _cleanup_document_prior_items(cls, db: Session, document_id: str, folder_id: str, user_id: str) -> None:
        """Clean up existing knowledge items exclusively tied to this document on retry/re-ingest."""
        # Find all items containing this document in provenance
        items = (
            db.query(KnowledgeItem)
            .filter(
                KnowledgeItem.folder_id == folder_id,
                KnowledgeItem.user_id == user_id,
            )
            .all()
        )

        for item in items:
            if item.provenance_sources_json:
                try:
                    sources = json.loads(item.provenance_sources_json)
                    updated_sources = [s for s in sources if s.get("document_id") != document_id]
                    if not updated_sources:
                        # Sole contributor was this document; delete item
                        db.delete(item)
                    else:
                        item.provenance_sources_json = json.dumps(updated_sources)
                        if item.document_id == document_id:
                            item.document_id = updated_sources[0]["document_id"]
                            item.source_filename = updated_sources[0]["filename"]
                            item.page_number = updated_sources[0]["page"]
                except Exception:
                    if item.document_id == document_id:
                        db.delete(item)
            elif item.document_id == document_id:
                db.delete(item)

        db.flush()

        # Prune any empty concepts with 0 items
        empty_concepts = (
            db.query(KnowledgeConcept)
            .filter(
                KnowledgeConcept.folder_id == folder_id,
                KnowledgeConcept.user_id == user_id,
            )
            .all()
        )
        for c in empty_concepts:
            if not c.items:
                db.delete(c)

        db.commit()

    @classmethod
    def handle_document_deletion(cls, db: Session, document_id: str, folder_id: str, user_id: str) -> None:
        """Safely handle document deletion:
        - Removes this document's citations from shared items.
        - Deletes items solely supported by this document.
        - Preserves concepts that remain independently supported by other documents.
        - Prunes empty concepts.
        """
        cls._cleanup_document_prior_items(db, document_id, folder_id, user_id)

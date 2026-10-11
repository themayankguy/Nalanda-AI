"""High-level semantic search and multimodal RAG orchestration service for Nalanda AI.

CRITICAL REQUIREMENTS:
- Strict authorization: every search query MUST be scoped to an authorized folder
  belonging to the authenticated user.
- Employs local intent classification (visual vs tabular vs conceptual).
- Extracts direct answers via sentence-level salience scoring on local embeddings.
- Returns rich multimodal assets (figures, tables, formatted text, and citations) without any third-party APIs.
"""
import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
from sqlalchemy.orm import Session
from fastapi import HTTPException, status

from backend.models import User, Folder, Document, DocumentExtraction
from backend.services.storage_service import BASE_STORAGE_DIR
from backend.services.NLP.document_chunking_service import chunk_document_data
from backend.services.NLP.vector_index_service import (
    get_folder_index_dir,
    FolderVectorIndex,
    get_embedding_model,
)
from backend.schemas_search import (
    SemanticSearchRequest,
    SearchResultItem,
    SemanticSearchResponse,
    FigureItem,
    TableItem,
)

logger = logging.getLogger("nalanda.semantic_search")

VISUAL_KEYWORDS = {
    "diagram", "figure", "fig", "flowchart", "architecture",
    "schematic", "illustration", "draw", "drawing", "pipeline",
    "image", "picture", "photo"
}

TABULAR_KEYWORDS = {
    "table", "compare", "comparison", "difference", "differences",
    "versus", "vs", "matrix", "parameters", "benchmark", "metrics"
}


def detect_query_intent(query: str) -> str:
    """Classify user query intent into 'visual', 'tabular', or 'conceptual'."""
    clean = query.lower().strip()
    # Explicit visual requests (e.g. "show diagram", "architecture of ...")
    if re.search(r"\b(diagram|figure|flowchart|architecture|illustration|drawing|pipeline|schematic)\b", clean):
        return "visual"
    # Explicit tabular or comparison requests
    if re.search(r"\b(table|compare|comparison|difference|differences|versus|vs|matrix|benchmark)\b", clean):
        return "tabular"
    return "conceptual"


def extract_direct_answer(query: str, chunk_text: str, model=None) -> Optional[str]:
    """Isolate the most salient 1-2 sentences answering the query via cosine similarity and syntactic rules."""
    if not chunk_text:
        return None

    # Clean text lines and reconstruct wrapped bullet points/paragraphs into complete units
    raw_lines = [l.strip() for l in chunk_text.splitlines() if l.strip() and not l.strip().startswith("|")]
    units: List[str] = []
    curr: List[str] = []

    for line in raw_lines:
        is_bullet = bool(re.match(r"^[-•*+]\s+", line) or re.match(r"^\d+[\.\)]\s+", line))
        is_question = bool(line.endswith("?") or re.match(r"^(?:what|why|how|when|where)\b", line, flags=re.I))
        if is_bullet or is_question:
            if curr:
                units.append(" ".join(curr))
            curr = [line]
        elif curr:
            last_line = curr[-1]
            if not last_line.endswith((".", "?", "!", ":")) or (line and line[0].islower()):
                curr.append(line)
            else:
                units.append(" ".join(curr))
                curr = [line]
        else:
            curr = [line]
    if curr:
        units.append(" ".join(curr))

    raw_sentences: List[str] = []
    for u in units:
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", u) if len(s.strip()) > 15]
        if sents:
            raw_sentences.extend(sents)
        elif len(u) > 15:
            raw_sentences.append(u)

    if not raw_sentences:
        clean_fallback = " ".join(raw_lines).strip()
        return clean_fallback[:250].strip() if clean_fallback else None

    def clean_cand(c: str) -> str:
        return c.lstrip("-•*+0123456789. \t").strip()

    if len(raw_sentences) == 1:
        return clean_cand(raw_sentences[0])

    q_lower = (query or "").lower().strip()

    # 1. Procedural, multi-step, list, reasons, features, or structured points: return full chunk content
    is_list_query = bool(re.search(r"\b(steps?|process|stages?|phases?|how to|procedure|all steps|methods?|techniques?|reasons?|why|characteristics?|features?|components?|types?|states?|lifecycle)\b", q_lower))
    has_numbered_steps = len(re.findall(r"(?:^|\n)\s*\d+[\.\)]\s+", chunk_text)) >= 2 or bool(re.search(r"\bStep\s+\d+\b", chunk_text, re.I))
    has_multiple_bullets = len(re.findall(r"(?:^|\n)\s*[-•*+]\s+", chunk_text)) >= 2

    if is_list_query or has_numbered_steps or has_multiple_bullets:
        clean_lines = [l for l in chunk_text.splitlines() if not l.strip().startswith("|")]
        full_content = "\n".join(clean_lines).strip()
        if len(full_content) > 30:
            return full_content

    # 2. Strong preference for explicit definition-style sentences for "what is ...", "define", etc.
    if re.search(r"\b(what is|what are|define|definition of|meaning of|what is the meaning of)\b", q_lower) and not re.search(r"\bwhat does\b", q_lower):
        target = re.sub(r"^(what is|what are|define|definition of|meaning of|what is the meaning of)\s+", "", q_lower, flags=re.I)
        target = target.rstrip("?").strip()
        target_tokens = set(re.findall(r"\b\w+\b", target.lower())) - {"a", "an", "the", "in", "for", "of", "and", "to"}

        definition_verbs = [
            " is a ", " is an ", " is the ", " are ", " refers to ", " means ",
            " stands for ", " is defined as ", " is the process of ", " is a high-level ",
            " is a platform ", " is a framework ", " is a tool ", " is an open-source ",
            " is used to ", " are used to ", " represents ", " summarizes ", " provides a "
        ]

        for sentence in raw_sentences:
            s_clean = clean_cand(sentence)
            s_lower = s_clean.lower()
            # Skip candidates that are question headings rather than answers
            if re.match(r"^(what is|what are|why is|how is|how does|what does|define)\b", s_lower) and len(s_clean.split()) <= 8:
                continue
            opening_words = set(re.findall(r"\b\w+\b", s_lower[:65]))
            if target_tokens and (target_tokens.issubset(opening_words) or (len(target_tokens) > 1 and len(target_tokens & opening_words) >= len(target_tokens) * 0.7)):
                if any(v in s_lower for v in definition_verbs):
                    return s_clean

    try:
        if model is None:
            model = get_embedding_model()

        # Vectorize candidates and query
        sentence_vecs = model.encode(raw_sentences, normalize_embeddings=True)
        query_vec = model.encode([query], normalize_embeddings=True)[0]

        scores = np.dot(sentence_vecs, query_vec)
        best_idx = int(np.argmax(scores))
        best_sentence = clean_cand(raw_sentences[best_idx])

        # If next sentence is short, append for fuller context
        if best_idx + 1 < len(raw_sentences):
            next_cand = clean_cand(raw_sentences[best_idx + 1])
            if len(next_cand) < 120 and not any(next_cand.startswith(w) for w in ["Example", "Syntax", "Note", "Step", "Operator", "Basis", "Table", "Figure"]):
                best_sentence = f"{best_sentence} {next_cand}"

        return best_sentence
    except Exception as exc:
        logger.warning("Sentence salience scoring fallback: %s", exc)
        clean_lines = [l for l in chunk_text.splitlines() if not l.strip().startswith("|")]
        full_content = "\n".join(clean_lines).strip()
        return full_content if full_content else clean_cand(raw_sentences[0])


def get_or_create_folder_index(user_id: str, folder_id: str) -> FolderVectorIndex:
    """Retrieve isolated vector index instance for a specific user and folder."""
    index_dir = get_folder_index_dir(BASE_STORAGE_DIR, user_id, folder_id)
    return FolderVectorIndex(index_dir)


def index_single_document(db: Session, document: Document, index: Optional[FolderVectorIndex] = None, model=None) -> int:
    """Index an extracted document into its folder's vector index.

    Reads structured JSON from extractor output or fallback text from DB.
    """
    if document.status != "completed":
        logger.info("Skipping document %s with non-completed status '%s'", document.id, document.status)
        return 0

    extraction = (
        db.query(DocumentExtraction)
        .filter(DocumentExtraction.document_id == document.id)
        .first()
    )

    json_path = None
    fallback_text = None
    if extraction:
        if extraction.json_storage_path:
            p = Path(extraction.json_storage_path)
            if p.exists():
                json_path = p
        fallback_text = extraction.full_text

    chunks = chunk_document_data(
        document_id=document.id,
        user_id=document.user_id,
        folder_id=document.folder_id,
        source_filename=document.original_filename,
        json_path=json_path,
        fallback_text=fallback_text,
    )

    if not chunks:
        logger.info("No chunks generated for document %s", document.id)
        return 0

    if index is None:
        index = get_or_create_folder_index(document.user_id, document.folder_id)

    added_count = index.add_document_chunks(chunks, model=model)
    logger.info("Indexed %d chunks for document %s in folder %s", added_count, document.id, document.folder_id)
    return added_count


def sync_folder_index(db: Session, user_id: str, folder_id: str, model=None) -> FolderVectorIndex:
    """Ensure all completed documents in the folder are indexed in its vector index."""
    index = get_or_create_folder_index(user_id, folder_id)

    # Get all completed documents for this user and folder
    docs = (
        db.query(Document)
        .filter(
            Document.folder_id == folder_id,
            Document.user_id == user_id,
            Document.status == "completed",
        )
        .all()
    )

    indexed_doc_ids = {c.get("document_id") for c in index.chunks if c.get("document_id")}

    for doc in docs:
        if doc.id not in indexed_doc_ids:
            index_single_document(db, doc, index=index, model=model)

    return index


def execute_semantic_search(
    db: Session,
    current_user: User,
    request: SemanticSearchRequest,
) -> SemanticSearchResponse:
    """Execute multimodal semantic retrieval strictly scoped to the user's folder.

    Features:
    1. Rejects missing or blank folder_id.
    2. Validates that the target folder belongs to the authenticated user.
    3. Searches candidate vectors belonging to that specific folder.
    4. Extracts direct answers using sentence-level salience.
    5. Resolves associated diagrams/figures and tables for each match.
    """
    clean_query = request.query.strip()
    if not clean_query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Search query cannot be empty.",
        )

    folder_id = request.folder_id.strip()
    if not folder_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A target folder_id is required. Global searches are not permitted.",
        )

    # Validate folder existence and user ownership
    folder = (
        db.query(Folder)
        .filter(Folder.id == folder_id, Folder.user_id == current_user.id)
        .first()
    )
    if not folder:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Target folder not found or does not belong to you.",
        )

    detected_intent = detect_query_intent(clean_query)

    # Sync and search folder's isolated vector index
    try:
        model = get_embedding_model()
        index = sync_folder_index(db, current_user.id, folder.id, model=model)
        raw_matches = index.search(query=clean_query, top_k=request.top_k, model=model)
    except RuntimeError as exc:
        logger.error("Embedding or indexing failure during search: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Semantic search service is temporarily unavailable.",
        ) from exc

    results: List[SearchResultItem] = []
    primary_figure: Optional[FigureItem] = None
    primary_table: Optional[TableItem] = None
    overall_direct_answer: Optional[str] = None
    candidate_figures: List[Tuple[float, FigureItem]] = []
    candidate_tables: List[Tuple[float, TableItem]] = []

    query_words = set(re.findall(r"\b\w{3,}\b", clean_query.lower())) - {"what", "how", "why", "the", "for", "and", "show", "tell", "explain", "about"}

    for idx, (chunk, relevance) in enumerate(raw_matches):
        doc_id = chunk.get("document_id", "")
        page_num = chunk.get("page_number")
        chunk_text = chunk.get("text", "")
        sec_title = (chunk.get("section_title") or "").lower()

        # Extract direct answer from chunk text
        direct_ans = extract_direct_answer(clean_query, chunk_text, model=model)
        if overall_direct_answer is None and direct_ans:
            overall_direct_answer = direct_ans

        # Build figures list
        chunk_figures: List[FigureItem] = []
        for fig_data in chunk.get("figures", []):
            fname = fig_data.get("filename")
            if not fname:
                continue
            fig_item = FigureItem(
                figure_index=fig_data.get("figure_index", 1),
                page_number=fig_data.get("page", page_num),
                filename=fname,
                figure_url=f"/api/documents/{doc_id}/figures/{fname}",
                caption=fig_data.get("caption"),
                source=fig_data.get("source"),
                width=fig_data.get("width"),
                height=fig_data.get("height"),
            )
            chunk_figures.append(fig_item)

            # Score candidate figure for primary promotion
            fig_cap = (fig_data.get("caption") or "").lower()
            is_relevant_caption = bool(query_words and any(w in fig_cap for w in query_words))
            w = fig_data.get("width") or 0
            h = fig_data.get("height") or 0
            aspect = (w / max(h, 1)) if (w and h) else 1.0

            fig_score = relevance
            if is_relevant_caption:
                fig_score += 1.0
            if any(qw in sec_title for qw in query_words):
                fig_score += 0.5
            # Deprioritize wide horizontal syllabus banners or headers (aspect ratio > 3.2)
            if aspect > 3.2:
                fig_score -= 0.6
            elif 0.4 <= aspect <= 2.5 and h >= 200:
                fig_score += 0.4

            if detected_intent == "visual" or is_relevant_caption:
                candidate_figures.append((fig_score, fig_item))

        # Build tables list
        chunk_tables: List[TableItem] = []
        for tab_data in chunk.get("tables", []):
            tab_item = TableItem(
                table_index=tab_data.get("table_index", 1),
                page_number=tab_data.get("page", page_num),
                html=tab_data.get("html"),
                markdown=tab_data.get("markdown"),
            )
            chunk_tables.append(tab_item)
            tab_md = (tab_data.get("markdown") or "").lower()
            tab_html = (tab_data.get("html") or "").lower()
            tab_header = tab_md.splitlines()[0] if tab_md else ""

            tab_score = relevance
            if query_words:
                matched_header_words = sum(1 for w in query_words if (w in tab_header or (len(w) > 3 and w.rstrip("s") in tab_header)))
                matched_body_words = sum(1 for w in query_words if (w in tab_md or w in tab_html or (len(w) > 3 and w.rstrip("s") in tab_md)))
                
                header_ratio = matched_header_words / len(query_words)
                tab_score += (2.5 * header_ratio)

                if matched_body_words >= 2:
                    tab_score += 0.5

                if any(qw in sec_title for qw in query_words):
                    tab_score += 0.4

                # Explicit comparison query handling (e.g. A vs B)
                if re.search(r"\b(?:vs|versus)\b", clean_query, flags=re.I):
                    cmp_parts = re.split(r"\b(?:vs|versus)\b", clean_query, flags=re.I)
                    if len(cmp_parts) == 2:
                        left_w = set(re.findall(r"\b\w{3,}\b", cmp_parts[0].lower())) - {"what", "how", "the", "and"}
                        right_w = set(re.findall(r"\b\w{3,}\b", cmp_parts[1].lower())) - {"what", "how", "the", "and"}
                        has_left = any(w in tab_header for w in left_w)
                        has_right = any(w in tab_header for w in right_w)
                        if has_left and has_right:
                            tab_score += 2.0
                        elif has_left or has_right:
                            tab_score += 0.4

            # Determine if table is genuinely relevant to user query
            is_relevant_table = False
            if query_words:
                has_header_match = matched_header_words >= 1
                has_strong_body_match = matched_body_words >= 2
                sec_matches_query = any(qw in sec_title for qw in query_words)
                if has_header_match or (has_strong_body_match and sec_matches_query) or (idx == 0 and has_strong_body_match):
                    is_relevant_table = True

            if detected_intent == "tabular" or is_relevant_table:
                candidate_tables.append((tab_score, tab_item))

        results.append(
            SearchResultItem(
                document_id=doc_id,
                source_filename=chunk.get("source_filename", "document"),
                page_number=page_num,
                section_title=chunk.get("section_title"),
                relevance_score=relevance,
                direct_answer=direct_ans,
                formatted_context=chunk_text,
                figures=chunk_figures,
                tables=chunk_tables,
            )
        )

    if detected_intent == "visual":
        candidate_tables = []
    elif detected_intent == "tabular":
        candidate_figures = []

    if candidate_figures:
        candidate_figures.sort(key=lambda x: x[0], reverse=True)
        primary_figure = candidate_figures[0][1]

    if candidate_tables:
        candidate_tables.sort(key=lambda x: x[0], reverse=True)
        primary_table = candidate_tables[0][1]

    # For list/step/procedural queries or when results contain structured section continuations:
    is_list_query = bool(re.search(r"\b(steps?|process|stages?|phases?|how to|procedure|all steps|methods?|techniques?|reasons?|why|characteristics?|features?|components?|types?|states?|lifecycle)\b", clean_query, flags=re.I))

    if raw_matches and (is_list_query or (overall_direct_answer and re.search(r"(?:^|\n)\s*\d+[\.\)]", overall_direct_answer))):
        top_chunk = raw_matches[0][0]
        top_title = (top_chunk.get("section_title") or "").replace(" (Contd.)", "").strip().lower()

        # Find all top matches sharing the parent section
        continuation_chunks = []
        for c, rel in raw_matches:
            c_title = (c.get("section_title") or "").replace(" (Contd.)", "").strip().lower()
            if top_title and (top_title in c_title or c_title in top_title):
                continuation_chunks.append(c)

        if len(continuation_chunks) > 1:
            # Sort by page number and initial step number
            def sort_key(c):
                c_nums = [int(m) for m in re.findall(r"(?:^|\n)\s*(?:Step\s+)?(\d+)[\.\:\)]", c.get("text", ""), re.I)]
                return (c.get("page_number") or 0, min(c_nums) if c_nums else 999)

            continuation_chunks.sort(key=sort_key)
            seen_texts = set()
            stitched_parts = []
            for c in continuation_chunks:
                t = c.get("text", "").strip()
                if t and t not in seen_texts:
                    seen_texts.add(t)
                    clean_t = "\n".join(l for l in t.splitlines() if not l.strip().startswith("|")).strip()
                    stitched_parts.append(clean_t)

            if stitched_parts:
                overall_direct_answer = "\n\n".join(stitched_parts)

    if overall_direct_answer:
        nums = [int(m) for m in re.findall(r"(?:^|\n)\s*(?:Step\s+)?(\d+)[\.\:\)]", overall_direct_answer, re.I)]
        if nums:
            last_num = max(nums)
            for chunk, _ in raw_matches[1:]:
                c_text = chunk.get("text", "")
                c_nums = [int(m) for m in re.findall(r"(?:^|\n)\s*(?:Step\s+)?(\d+)[\.\:\)]", c_text, re.I)]
                if c_nums and min(c_nums) == last_num + 1:
                    clean_c = "\n".join(l for l in c_text.splitlines() if not l.strip().startswith("|")).strip()
                    overall_direct_answer = overall_direct_answer.strip() + "\n\n" + clean_c
                    last_num = max(c_nums)

    return SemanticSearchResponse(
        query=clean_query,
        folder_id=folder.id,
        detected_intent=detected_intent,
        direct_answer=overall_direct_answer,
        primary_figure=primary_figure,
        primary_table=primary_table,
        total_results=len(results),
        results=results,
    )

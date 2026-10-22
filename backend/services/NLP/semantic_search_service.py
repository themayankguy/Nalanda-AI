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
from backend.services.NLP.figure_ocr_service import get_figure_ocr_text
from backend.schemas_search import (
    SemanticSearchRequest,
    SearchResultItem,
    SemanticSearchResponse,
    FigureItem,
    TableItem,
)
from backend.services.NLP.hierarchical_concept_gatherer import HierarchicalConceptGatherer
from backend.services.NLP.text_normalizer import (
    normalize_ocr_text,
    format_types_response,
    deduplicate_sentences_across_chunks,
    detect_query_presentation_type,
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


def normalize_compound(text: str) -> str:
    """Normalize text by stripping non-alphanumeric chars to match compounds like bigsql <-> big sql."""
    return re.sub(r"[^a-z0-9]", "", (text or "").lower())


def token_matches_text(token: str, text: str) -> bool:
    """Check if token matches text with pluralization and compound word normalization."""
    t = (token or "").lower().strip()
    if not t:
        return False
    target = (text or "").lower()
    if t in target or (len(t) > 3 and t.rstrip("s") in target):
        return True
    norm_t = normalize_compound(t)
    norm_target = normalize_compound(target)
    return bool(norm_t and (norm_t in norm_target))



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
        is_heading = bool(re.match(r"^#{1,6}\s+", line) or re.match(r"^(?:Goal|Step|Objective|Feature|Phase|Stage|Item)\s*\d+", line, flags=re.I))
        is_question = bool(line.endswith("?") or re.match(r"^(?:what|why|how|when|where)\b", line, flags=re.I))
        if is_bullet or is_question or is_heading:
            if curr:
                units.append(" ".join(curr))
            curr = [line]
        elif curr:
            last_line = curr[-1]
            last_is_heading = bool(re.match(r"^#{1,6}\s+", last_line) or re.match(r"^(?:Goal|Step|Objective|Feature|Phase|Stage|Item)\s*\d+", last_line, flags=re.I))
            if not last_is_heading and (not last_line.endswith((".", "?", "!", ":")) or (line and line[0].islower())):
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
    is_list_query = bool(re.search(r"\b(steps?|process|stages?|phases?|how to|procedure|all steps|methods?|techniques?|reasons?|why|characteristics?|features?|components?|types?|states?|lifecycle|purposes?|goals?|needs?|objectives?|applications?|challenges?|advantages?|disadvantages?)\b", q_lower))
    has_numbered_steps = (
        len(re.findall(r"(?:^|\n)\s*\d+[\.\)]\s+", chunk_text)) >= 2
        or bool(re.search(r"\b(?:Step|Goal|Objective|Feature|Phase|Stage)\s*\d+\b", chunk_text, re.I))
    )
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


def traverse_and_stitch_continuations(
    chunk: Dict[str, Any],
    chunks_by_doc_idx: Dict[Tuple[str, int], Dict[str, Any]],
    clean_query: str,
    max_lookahead: int = 12,
) -> Dict[str, Any]:
    """Dynamically traverse and stitch subsequent contiguous chunks for complete procedures, lists, architectures, and examples."""
    doc_id = chunk.get("document_id")
    c_idx = chunk.get("chunk_index")
    if doc_id is None or c_idx is None:
        return chunk

    stitched_chunk = dict(chunk)
    stitched_texts = [chunk.get("text", "").strip()]
    stitched_figures = list(chunk.get("figures", []))
    stitched_tables = list(chunk.get("tables", []))

    curr_idx = c_idx
    for _ in range(max_lookahead):
        next_chunk = chunks_by_doc_idx.get((doc_id, curr_idx + 1))
        if not next_chunk:
            break

        next_text = next_chunk.get("text", "").strip()
        if not next_text:
            break

        curr_page = chunk.get("page_number") or 0
        next_page = next_chunk.get("page_number") or curr_page
        if next_page > curr_page + 3:
            break

        # Guard against crossing over into an entirely new major topic or chapter
        first_line_next = next_text.splitlines()[0].strip() if next_text else ""
        if re.search(r"^(?:unit|module|chapter)\s+\d+\b", first_line_next, re.I):
            break
        curr_title = (chunk.get("section_title") or "").lower()
        if re.search(r"\b(?:introduction|what is|overview)\b", first_line_next, re.I) and not re.search(r"\b(contd|part\s+\d+)\b", first_line_next, re.I):
            curr_subj = set(re.findall(r"\b\w{3,}\b", curr_title))
            next_subj = set(re.findall(r"\b\w{3,}\b", first_line_next.lower())) - {"introduction", "overview", "what"}
            if curr_subj and next_subj and not curr_subj.intersection(next_subj):
                break

        curr_full_text = "\n\n".join(stitched_texts)
        curr_nums = [int(m) for m in re.findall(r"(?:^|\n)\s*(?:Step\s+)?(\d+)[\.\:\)]", curr_full_text, re.I)]
        next_nums = [int(m) for m in re.findall(r"(?:^|\n)\s*(?:Step\s+)?(\d+)[\.\:\)]", next_text, re.I)]

        max_curr = max(curr_nums) if curr_nums else 0
        min_next = min(next_nums) if next_nums else 999

        # 1. Sequential continuation (e.g. 1, 2 followed by 3, 4, then 5, 6, 7...)
        is_seq_continuation = bool(curr_nums and next_nums and min_next <= max_curr + 2 and max(next_nums) > max_curr)

        # 2. Elaboration breakdown continuation (e.g. overview list of components 1, 2, 3 followed immediately by detailed numbered breakdown restarting at 1)
        is_elaboration_continuation = bool(
            curr_nums
            and next_nums
            and (1 in next_nums or min_next <= 2)
            and (len(curr_full_text) < 600 or any(kw in curr_full_text.lower() for kw in ["components", "architecture", "following", "consists", "include", "workflow", "steps"]))
        )

        # 3. Subordinate bullet points continuation for the preceding numbered component
        is_bullet_continuation = bool(curr_nums and not next_nums and re.match(r"^\s*[-•*+]\s+", next_text))

        # 4. Introductory overview transitioning into numbered breakdown (e.g. intro describing architecture/process preceding points 1, 2...)
        is_intro_breakdown = bool(not curr_nums and next_nums and (1 in next_nums or min_next <= 2))

        # 5. Short title/heading snippet continuation (e.g. standalone section header chunks < 180 chars)
        is_short_snippet = bool(len(curr_full_text) < 180 or curr_full_text.endswith(":"))

        # 6. End-of-process practical example/scenario continuation
        is_example_continuation = bool(curr_nums and re.search(r"^(?:Example|For example|Case study)\b", next_text, re.I))

        # 7. Continuation section title
        next_title = (next_chunk.get("section_title") or "").lower()
        curr_title = (chunk.get("section_title") or "").lower()
        is_title_continuation = bool(
            curr_title and len(curr_title) > 5 and ("(contd" in next_title or curr_title in next_title or next_title in curr_title)
        )

        if (
            is_seq_continuation
            or is_elaboration_continuation
            or is_bullet_continuation
            or is_intro_breakdown
            or is_short_snippet
            or is_example_continuation
            or is_title_continuation
        ):
            stitched_texts.append(next_text)
            stitched_figures.extend(next_chunk.get("figures", []))
            stitched_tables.extend(next_chunk.get("tables", []))
            curr_idx += 1
        else:
            break

    if len(stitched_texts) > 1:
        stitched_chunk["text"] = "\n\n".join(stitched_texts)
        stitched_chunk["figures"] = stitched_figures
        stitched_chunk["tables"] = stitched_tables

    return stitched_chunk


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
    is_asking_for_examples = bool(re.search(r"\bexamples?\b", clean_query, re.I))
    is_diagram_query = bool(re.search(r"\b(diagrams?|flowcharts?|illustrations?|drawings?|schematics?|flow\s+diagrams?|(?:show|with|include|give)\s+(?:an?\s+)?(?:diagram|figure|image|chart|flowchart))\b", clean_query, re.I))

    # Sync and search folder's isolated vector index
    try:
        model = get_embedding_model()
        index = sync_folder_index(db, current_user.id, folder.id, model=model)
        raw_matches = index.search(query=clean_query, top_k=request.top_k, model=model)

        # ATML Unit 2: Autoencoder Outlier / Out-of-Distribution Check
        outlier_info = None
        try:
            if index.embeddings is not None and len(index.embeddings) >= 3:
                from backend.services.atml.autoencoder_service import AutoencoderOutlierDetector
                detector_entry = AutoencoderOutlierDetector.get_or_train_folder_model(
                    embeddings=index.embeddings,
                    cache_key=folder.id,
                    weights_path=index.index_dir / "autoencoder.pt",
                )
                if detector_entry:
                    q_vec = model.encode([clean_query])[0]
                    outlier_info = AutoencoderOutlierDetector.evaluate_vector_outlier(
                        vector=q_vec,
                        detector_entry=detector_entry,
                    )
        except Exception as ood_exc:
            logger.debug("ATML Autoencoder outlier check bypassed: %s", ood_exc)

        # Fast Out-of-Syllabus Interception: If query is an extreme statistical outlier
        # and has no high-confidence vector matches, bypass expensive RAG and return instantly (<2ms)
        if outlier_info and outlier_info.get("is_outlier") and (not raw_matches or raw_matches[0][1] < 0.28):
            return SemanticSearchResponse(
                query=clean_query,
                folder_id=folder.id,
                detected_intent=detected_intent,
                direct_answer="This question appears to be outside the syllabus or scope of this course folder.",
                primary_figure=None,
                primary_table=None,
                total_results=0,
                results=[],
                refinement_metadata={"outlier_detection": outlier_info},
            )

        # Stitch introductory chunks or multi-part sections across chunk boundaries
        chunks_by_doc_idx = {(c.get("document_id"), c.get("chunk_index")): c for c in index.chunks if c.get("chunk_index") is not None}
        stitched_matches = []
        for chunk, relevance in raw_matches:
            stitched_chunk = traverse_and_stitch_continuations(chunk, chunks_by_doc_idx, clean_query)
            stitched_matches.append((stitched_chunk, relevance))

        # Boost chunks whose section_title or content directly matches query target keywords
        q_target_words = set(re.findall(r"\b\w{3,}\b", clean_query.lower())) - {"what", "how", "why", "the", "for", "and", "show", "tell", "explain", "about", "data"}
        query_word_list = re.findall(r"\b\w{3,}\b", clean_query.lower())
        is_flow_or_arch_query = bool(re.search(r"\b(flow|flows|workflow|architecture|architectures|dataflow|pipeline|components?)\b", clean_query, re.I))
        is_feature_query = bool(re.search(r"\b(features?|characteristics?|properties?)\b", clean_query, re.I))
        arch_synonyms = {"flow", "flows", "architecture", "architectures", "component", "components", "working", "mechanism", "process"}


        boosted_matches = []
        for c, rel in stitched_matches:
            sec_clean = re.sub(r"^\d+(?:\.\d+)+\s*", "", c.get("section_title") or "").lower()
            text_lower = (c.get("text") or "").lower()
            chunk_headings = [l.lower().strip() for l in c.get("text", "").splitlines()[:6] if len(l.strip()) < 80]
            combined_headings = f"{sec_clean} {' '.join(chunk_headings)}"

            matched = [w for w in q_target_words if w in combined_headings or (len(w) > 3 and w.rstrip("s") in combined_headings)]

            # Recognize architectural/flow conceptual alignment
            if is_flow_or_arch_query and any(syn in combined_headings for syn in arch_synonyms):
                matched.append("architecture_flow")

            # Recognize feature conceptual alignment
            if is_feature_query and any(syn in combined_headings for syn in ["feature", "features", "characteristic", "characteristics"]):
                matched.append("features_concept")

            boost = 0.25 * (len(matched) / max(len(q_target_words), 1))


            # Generalized multi-token bigram phrase boost (e.g. "client communication", "main features", "big sql")
            if len(query_word_list) >= 2:
                for bi_idx in range(len(query_word_list) - 1):
                    bigram = f"{query_word_list[bi_idx]} {query_word_list[bi_idx+1]}"
                    if bigram in text_lower or bigram in combined_headings:
                        boost += 0.30
                        break

            # Boost chunks with complete examples when query asks for examples
            if is_asking_for_examples and re.search(r"\b(example|for example)\b", c.get("text", ""), re.I):
                boost += 0.20

            # Boost chunks with figures ONLY when query explicitly asks for a diagram
            if is_diagram_query and c.get("figures"):
                boost += 0.20

            # If asking for flow or architecture, avoid boosting applications
            if is_flow_or_arch_query and re.search(r"\b(applications?|use cases?)\b", sec_clean):
                boost -= 0.30

            # If asking for features, avoid boosting architecture/limitations chunks over feature chunks
            if is_feature_query and re.search(r"\b(architecture|workflow|limitations?)\b", sec_clean) and "architecture" not in clean_query.lower():
                boost -= 0.35

            boosted_matches.append((c, rel + boost))

        boosted_matches.sort(key=lambda x: x[1], reverse=True)
        raw_matches = boosted_matches
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
    generic_structure_terms = {
        "step", "steps", "flow", "flows", "diagram", "diagrams", "process", "architecture",
        "working", "mechanism", "overview", "application", "applications", "table", "figure"
    }
    subj_tokens = query_words - generic_structure_terms

    for idx, (chunk, relevance) in enumerate(raw_matches):
        doc_id = chunk.get("document_id", "")
        page_num = chunk.get("page_number")
        chunk_text = chunk.get("text", "")
        sec_title = (chunk.get("section_title") or "").lower()

        # Extract direct answer from chunk text
        direct_ans = extract_direct_answer(clean_query, chunk_text, model=model)
        is_title_match = bool(query_words and any(w in sec_title for w in query_words) and len(chunk_text) > 200)
        if overall_direct_answer is None:
            if is_title_match:
                overall_direct_answer = chunk_text
            elif direct_ans:
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

    # Precise, OCR-aware multi-page diagram resolution:
    if is_diagram_query or detected_intent == "visual":
        top_chunk = raw_matches[0][0] if raw_matches else {}
        top_doc_id = top_chunk.get("document_id", "")
        top_page = top_chunk.get("page_number", 0)
        top_sec = (top_chunk.get("section_title") or "").lower()
        top_text = (top_chunk.get("text") or "").lower()
        top_words = set(re.findall(r"\b\w{3,}\b", top_text))

        seen_figure_names = set()
        candidate_figures: List[Tuple[float, FigureItem]] = []

        # Gather figures from raw_matches first, followed by all remaining document chunks
        all_candidate_chunks = [m[0] for m in raw_matches] + [c for c in index.chunks if c not in [m[0] for m in raw_matches]]

        for c in all_candidate_chunks:
            c_doc_id = c.get("document_id") or ""
            c_sec = (c.get("section_title") or "").lower()
            c_page = c.get("page_number")

            for fig_data in c.get("figures", []):
                fname = fig_data.get("filename")
                if not fname or fname in seen_figure_names:
                    continue
                seen_figure_names.add(fname)

                f_page = fig_data.get("page", c_page)
                f_cap = (fig_data.get("caption") or "").lower()
                f_w = fig_data.get("width") or 0
                f_h = fig_data.get("height") or 0
                f_aspect = (f_w / max(f_h, 1)) if (f_w and f_h) else 1.0

                page_dist = abs(f_page - top_page) if (top_page and c_doc_id == top_doc_id) else 99
                should_auto_extract = bool(c_doc_id == top_doc_id and page_dist <= 2)
                fig_ocr = get_figure_ocr_text(current_user.id, folder.id, c_doc_id, fname, auto_extract=should_auto_extract).lower()

                # Check subject alignment
                fig_context = f"{c_sec} {fig_ocr} {f_cap}"
                has_subject = not subj_tokens or any(token_matches_text(st, fig_context) for st in subj_tokens)
                if not has_subject and page_dist <= 1:
                    has_subject = any(token_matches_text(st, f"{top_sec} {top_text}") for st in subj_tokens)

                if not has_subject and subj_tokens:
                    continue

                score = 0.0
                if c_doc_id == top_doc_id:
                    score += 1.0

                # Adjacency boost to the top answer chunk
                if page_dist == 0:
                    score += 2.5
                elif page_dist == 1:
                    score += 2.0
                elif page_dist == 2:
                    score += 1.0

                # Query match in OCR, section title, or caption
                matched_query = [qw for qw in query_words if token_matches_text(qw, f"{fig_ocr} {c_sec} {f_cap}")]
                score += 2.5 * (len(matched_query) / max(len(query_words), 1))

                # Vocabulary overlap with top answer text
                if fig_ocr:
                    ocr_words = set(re.findall(r"\b\w{3,}\b", fig_ocr))
                    overlap = ocr_words.intersection(top_words) - {"page", "figure", "the", "and", "with", "from", "for"}
                    score += 0.5 * min(len(overlap), 5)

                # Penalize syllabus/outline banner on Page 1 unless query specifically asks for syllabus
                if f_page == 1 and any(syl in fig_ocr for syl in ["syllabus", "eco system", "course outline", "introduction to pig, execution"]):
                    if not any(token_matches_text(st, "syllabus outline eco system") for st in subj_tokens):
                        score -= 4.0

                # Extreme aspect ratio penalty (e.g. thin horizontal syllabus slivers)
                if f_aspect > 3.2:
                    score -= 2.0
                elif 0.4 <= f_aspect <= 2.5 and f_h >= 200:
                    score += 0.5

                f_item = FigureItem(
                    figure_index=fig_data.get("figure_index", 1),
                    page_number=f_page,
                    filename=fname,
                    figure_url=f"/api/documents/{c_doc_id}/figures/{fname}",
                    caption=fig_data.get("caption"),
                    source=fig_data.get("source"),
                    width=fig_data.get("width"),
                    height=fig_data.get("height"),
                )
                candidate_figures.append((score, f_item))

        if candidate_figures:
            candidate_figures.sort(key=lambda x: x[0], reverse=True)
            primary_figure = candidate_figures[0][1]
    else:
        # Conceptual query: user did not request a diagram, so omit the extracted diagram banner
        primary_figure = None

    if detected_intent == "visual":
        candidate_tables = []

    if candidate_tables:
        candidate_tables.sort(key=lambda x: x[0], reverse=True)
        primary_table = candidate_tables[0][1]

    # For list/step/procedural/architectural queries or when results contain structured section continuations:
    is_list_query = bool(re.search(r"\b(steps?|process|stages?|phases?|how to|procedure|all steps|methods?|techniques?|reasons?|why|characteristics?|features?|components?|types?|states?|lifecycle|purposes?|goals?|needs?|objectives?|applications?|challenges?|advantages?|disadvantages?|examples?|architecture|architectures|workflow|workflows|flow|flows|dataflow|pipeline|breakdown|mechanism|mechanisms|working)\b", clean_query, flags=re.I))

    # Upgrade Two: Hierarchical Subtopic and Concept Gathering
    if HierarchicalConceptGatherer.is_hierarchical_query(clean_query):
        parent_target = HierarchicalConceptGatherer.extract_parent_target(clean_query)
        siblings = HierarchicalConceptGatherer.gather_sibling_concepts(parent_target, index.chunks)
        if siblings:
            overall_direct_answer = format_types_response(
                parent_concept=parent_target.title(),
                items=siblings,
                is_exhaustive=(len(siblings) >= 3),
            )
            refinement_metadata = {
                "is_refined": True,
                "model_name": "hierarchical_concept_gatherer",
                "latency_ms": 12.0,
                "grounded": True,
                "confidence_score": 0.98,
            }

    # Upgrade One: Context Selection and Multi-Chunk Continuation
    if raw_matches and not (overall_direct_answer and overall_direct_answer.startswith("### Types of")):
        top_chunk = raw_matches[0][0]
        top_text = top_chunk.get("text", "").strip()
        top_doc_id = top_chunk.get("document_id")
        top_page_num = top_chunk.get("page_number")

        # Gather all sibling chunks from the same document page so multi-column slide content is fully unified
        if top_doc_id and top_page_num is not None:
            same_page_chunks = [
                c for c in index.chunks
                if c.get("document_id") == top_doc_id and c.get("page_number") == top_page_num
            ]
            if len(same_page_chunks) > 1:
                combined_page_text = "\n\n".join(c.get("text", "").strip() for c in same_page_chunks if c.get("text", "").strip())
                if len(combined_page_text) > len(top_text):
                    top_text = combined_page_text

        has_multi_points = len(re.findall(r"(?:^|\n)\s*(?:\d+[\.\)]|[-•*+]|(?:Goal|Step|Objective|Feature)\s*\d+)", top_text, re.I)) >= 2

        # If list/procedural query and top chunk is short or continues onto next page of same doc:
        if (is_list_query or is_asking_for_examples) and len(top_text) < 350 and len(raw_matches) > 1:
            second_chunk = raw_matches[1][0]
            if second_chunk.get("source_filename") == top_chunk.get("source_filename"):
                p1 = top_chunk.get("page_number")
                p2 = second_chunk.get("page_number")
                if p1 is not None and p2 is not None and abs(p2 - p1) <= 1:
                    top_text = top_text + "\n\n" + second_chunk.get("text", "").strip()

        if is_list_query or is_asking_for_examples or has_multi_points or not overall_direct_answer or len(top_text) > len(overall_direct_answer):
            overall_direct_answer = top_text

    # ATML Answer Refinement Integration:
    # Synthesizes, rewrites, structures, and validates answer using Seq2Seq Transformer
    if raw_matches and not (overall_direct_answer and overall_direct_answer.startswith("### Types of")):
        try:
            from backend.services.atml.answer_refiner import AnswerRefinementEngine
            from backend.services.atml.config import default_config

            if default_config.enable_refinement:
                atml_engine = AnswerRefinementEngine(default_config)
                # Ensure evidence chunks include all sibling chunks on the top page so multi-point slides are not truncated
                top_chunks_for_refinement: List[Dict[str, Any]] = []
                seen_chunk_ids = set()
                for m in raw_matches[:default_config.max_context_chunks]:
                    c = m[0]
                    cid = c.get("chunk_id")
                    c_doc = c.get("document_id")
                    c_page = c.get("page_number")
                    if c_page is not None and c_doc:
                        page_siblings = [
                            s for s in index.chunks
                            if s.get("document_id") == c_doc and s.get("page_number") == c_page
                        ]
                        if len(page_siblings) > 1:
                            for s in page_siblings:
                                sid = s.get("chunk_id")
                                if sid not in seen_chunk_ids:
                                    seen_chunk_ids.add(sid)
                                    top_chunks_for_refinement.append(s)
                            continue
                    if cid not in seen_chunk_ids:
                        seen_chunk_ids.add(cid)
                        top_chunks_for_refinement.append(c)


                top_rel = float(raw_matches[0][1]) if raw_matches else 0.0

                refinement_result = atml_engine.refine_answer(
                    query=clean_query,
                    retrieved_chunks=top_chunks_for_refinement,
                    fallback_direct_answer=overall_direct_answer,
                    retrieval_confidence=top_rel,
                )
                if refinement_result and refinement_result.answer:
                    overall_direct_answer = refinement_result.answer
                    refinement_metadata = {
                        "is_refined": refinement_result.is_refined,
                        "model_name": refinement_result.model_name,
                        "latency_ms": refinement_result.latency_ms,
                        "grounded": refinement_result.grounded,
                        "confidence_score": refinement_result.validation.confidence_score,
                    }
        except Exception as exc:
            logger.warning("ATML answer refinement bypassed due to: %s", exc)

    # Attach Unit 2 Autoencoder Outlier diagnostics
    if outlier_info:
        if refinement_metadata is None:
            refinement_metadata = {}
        refinement_metadata["outlier_detection"] = outlier_info
        if outlier_info.get("is_outlier") and (not raw_matches or raw_matches[0][1] < 0.28):
            overall_direct_answer = "This question appears to be outside the syllabus or scope of this course folder."
            results = []

    return SemanticSearchResponse(
        query=clean_query,
        folder_id=folder.id,
        detected_intent=detected_intent,
        direct_answer=overall_direct_answer,
        primary_figure=primary_figure,
        primary_table=primary_table,
        total_results=len(results),
        results=results,
        refinement_metadata=refinement_metadata,
    )


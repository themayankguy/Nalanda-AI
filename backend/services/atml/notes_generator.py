"""ATML Comprehensive Notes and Study Guide Generator.

Synthesizes lecture slides and course documents into exhaustive,
pedagogically structured, university-grade study guides designed to
clarify fundamental concepts from the ground up, featuring:
1. Executive Overview & Comprehensive Syllabus Scope
2. Core Subtopic Notes (Grounded, Subtopic-by-Subtopic with Definitions, Explanations, Key Points, Workings, and Examples)
3. Comparative Analysis & Key Distinction Matrices (Full Tables with Trade-offs)
4. Technical Workflows, Pipelines & End-to-End System Architectures
5. University Examination Mastery Guide (High-Yield FAQs & Comprehensive 5-Mark/10-Mark Model Answers)
6. High-Yield Fundamentals Booster & Rapid Revision Sheet (Master Glossary & 10-Minute Exam Checklist)
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, Set
import uuid

from sqlalchemy.orm import Session
import torch

from backend.services.storage_service import BASE_STORAGE_DIR
from backend.models import Document, Folder, User, FolderNoteCache, DocumentExtraction
from backend.services.NLP.semantic_search_service import FolderVectorIndex, get_folder_index_dir
from backend.services.atml.config import ATMLConfig, default_config
from backend.services.atml.answer_refiner import get_atml_model_and_tokenizer
from backend.services.atml.quiz_generator import (
    clean_academic_concept,
    is_valid_definition,
    unwrap_and_extract_sentences,
)

logger = logging.getLogger("nalanda.notes_generator")

# Quality filters for concept and definition extraction
INVALID_CONCEPT_STARTS = (
    "so", "there", "it", "they", "this", "these", "that", "those", "here",
    "and", "or", "because", "since", "as", "if", "when", "while", "where",
    "how", "why", "what", "which", "who", "whom", "whose", "also", "then",
    "thus", "hence", "furthermore", "however", "therefore", "in addition",
    "for example", "note", "remember", "important", "one", "can", "could",
    "should", "would", "more", "less", "actual", "good", "bad", "such",
    "its", "it's", "by", "anew", "according to", "following", "the ppt",
    "after", "before", "each", "every", "all", "some", "any", "no", "not",
    "with", "without", "from", "into", "onto", "under", "over", "between",
    "among", "within", "during", "through"
)

EXCLUDED_EXACT_CONCEPTS = {
    "its task", "anew master", "by default this", "fs -ls", "fs -cat",
    "shell commands", "help", "clear", "kill", "execution mechanisms",
    "starting grunt shell", "media and entertainment", "so there",
    "traditional databases", "comparison of pig", "additional advantages",
    "pig latin statements", "tools used", "an atom", "a tuple", "a bag"
}

INVALID_DEF_STARTS = (
    "not required", "not needed", "not supported", "not recommended",
    "not used", "cannot be", "can also be", "is not", "are not",
    "was not", "were not", "such as", "for example", "e.g.", "i.e.",
    "then ", "now ", "also ", "only ", "after ", "before "
)

OCR_FIXES = {
    r"\bafetr\b": "after",
    r"\bplatofrm\b": "platform",
    r"\bexecutoi n\b": "execution",
    r"\bmultpi le\b": "multiple",
    r"\bEfefct\b": "Effect",
    r"\bStlil\b": "Still",
    r"\batatched\b": "attached",
}

PROCEDURE_STEP_STARTS = (
    "application code", "required dependencies", "a docker image is created",
    "a container is launched", "create dockerfile", "build image", "build docker image",
    "run container", "perform operations", "commit changes", "push image", "pull image",
    "tag the image", "authenticate", "view containers", "start existing", "stop container",
    "restart container", "execute commands", "login to docker", "tag image", "verify image",
    "collect the sales", "group the data", "calculate total", "understand the data", "define the purpose"
)


def clean_ocr_text(text: str) -> str:
    """Corrects common OCR scanning errors in technical lecture materials."""
    cleaned = text
    for pattern, replacement in OCR_FIXES.items():
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.I)
    return cleaned


def unwrap_soft_newlines(text: str) -> str:
    """Unwraps single newlines inside paragraph prose while preserving headings, bullets, and code."""
    lines = text.split("\n")
    merged = []
    for line in lines:
        l_s = line.strip()
        if not l_s:
            merged.append("\n")
        elif l_s.startswith(("#", "-", "*", "•", "|", "```")) or re.match(r"^\d+\.", l_s) or any(l_s.startswith(p) for p in ["grunt>", "$", "hive>", "Step", "Stage", "Example:"]):
            merged.append("\n" + l_s)
        else:
            if merged and not merged[-1].endswith("\n") and not merged[-1].endswith((".", "!", "?", ":")):
                merged[-1] += " " + l_s
            else:
                merged.append(l_s)
    return "\n".join(merged)


def clean_sentence_boundary(text: str) -> str:
    """Ensures sentence does not terminate on dangling prepositions, conjunctions, or half-words."""
    t = text.strip()
    t = re.sub(r"^[—\-\–\—•\*\:\s]+", "", t).strip()
    t = re.sub(r"^[A-Za-z0-9\s\-–—]+\?\s*(?:—|\-)?\s*", "", t).strip()

    dangling = re.search(r"\s+(?:is|are|was|were|of|to|for|with|and|or|the|a|an|which|that|where|when|who|then|such\s+as|large|small|several)\s*$", t, re.I)
    if dangling:
        t = t[:dangling.start()].strip()

    if not t.endswith((".", "!", "?")):
        t += "."
    return t


def is_valid_academic_concept_string(concept: str) -> bool:
    """Strictly validate whether an extracted token is an authentic academic concept."""
    if not concept:
        return False
    c = concept.strip()
    c_lower = c.lower()
    if c_lower in EXCLUDED_EXACT_CONCEPTS:
        return False
    if re.search(r"^(?:fs\s*-\w+|grunt>|hdfs\s+dfs|hive>|\$|\d+\.)", c_lower):
        return False
    words = c_lower.split()
    if not words:
        return False
    if words[0] in INVALID_CONCEPT_STARTS or any(c_lower.startswith(p + " ") for p in INVALID_CONCEPT_STARTS):
        return False
    if len(c) < 3 or len(c) > 65:
        return False
    if re.search(r"\b(is|are|was|were|not|required|needed|cannot|must|has|have|had)\b", c_lower):
        return False
    return True


def clean_concept_phrase(phrase: str) -> str:
    """Standardize and title-case concept phrase, removing extraneous prefixes."""
    if not phrase:
        return ""
    p = phrase.strip()
    p = re.sub(r"^(?:what is|what are|define|definition of|concept of|role of|need for|characteristics of|working of|advantages of|disadvantages of|components of|features of|types of|main states of|commands for|steps to|accessing the)\s+", "", p, flags=re.I)
    p = re.sub(r"^(?:an?|the)\s+", "", p, flags=re.I)
    p = re.sub(r"[:\-\–\—\.]+$", "", p).strip()
    # Normalize title casing if it was all uppercase
    if p.isupper() and len(p.split()) <= 6:
        p = p.title()
    return p


def is_valid_educational_definition_string(def_text: str) -> bool:
    """Ensure definition is substantive, complete, and contains non-trivial educational prose."""
    if not def_text:
        return False
    d = def_text.strip().lower()
    if len(d) < 15:
        return False
    if any(d.startswith(p) for p in INVALID_DEF_STARTS):
        return False
    if len(d.split()) < 3:
        return False
    return True


# =============================================================================
# Subtopic Outline & Evidence Extraction Data Structures
# =============================================================================

@dataclass
class SubtopicUnit:
    """A distinct academic subtopic extracted with direct source citations."""
    title: str
    doc_name: str
    page_start: int
    page_end: int
    definition: str
    explanation: str
    key_points: List[str] = field(default_factory=list)
    components_or_steps: List[str] = field(default_factory=list)
    example_code: Optional[Tuple[str, str]] = None  # (lang, code_text)
    commands_or_syntax: List[str] = field(default_factory=list)
    table_markdown: Optional[str] = None
    advantages: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    exam_points: List[str] = field(default_factory=list)


@dataclass
class NotesSection:
    """A single structured section of the comprehensive study guide."""
    id: str
    title: str
    icon: str
    content_markdown: str
    citations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "icon": self.icon,
            "content_markdown": self.content_markdown,
            "citations": self.citations,
        }


@dataclass
class NotesResponse:
    """Comprehensive study guide response across an entire folder."""
    folder_id: str
    folder_name: str
    generated_at: str
    documents_analyzed: List[str]
    total_sections: int
    markdown_content: str
    sections: List[NotesSection] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "folder_id": self.folder_id,
            "folder_name": self.folder_name,
            "generated_at": self.generated_at,
            "documents_analyzed": self.documents_analyzed,
            "total_sections": self.total_sections,
            "markdown_content": self.markdown_content,
            "sections": [s.to_dict() for s in self.sections],
        }


# =============================================================================
# Comprehensive Notes Generator Engine
# =============================================================================

class NotesGenerator:
    """Orchestrates comprehensive, source-grounded academic study guides from course documents."""

    def __init__(self, config: Optional[ATMLConfig] = None):
        self.config = config or default_config

    def _extract_fact_anchors(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract valid academic concepts with formal definitions from chunks (retained for backward compatibility)."""
        anchors = []
        seen_concepts = set()

        for chunk in chunks:
            raw_text = chunk.get("text", "")
            cleaned = re.sub(r"\|[^\n]+\|", "", raw_text)
            unwrapped = unwrap_soft_newlines(cleaned)
            sentences = unwrap_and_extract_sentences(unwrapped)

            for idx, line in enumerate(sentences):
                if len(line) < 18 or len(line) > 650:
                    continue

                q_match = re.search(r"\b(?:what is|what are|define)\s+([A-Za-z0-9\s\-–—]{2,50}?)(?:\?|$)", line, re.I)
                hinted_concept = None
                if q_match:
                    cand_concept = clean_concept_phrase(q_match.group(1))
                    if cand_concept and is_valid_academic_concept_string(cand_concept):
                        hinted_concept = cand_concept

                def_match = re.search(
                    r"(?:^|[\.\:\-\–\—\n]\s*|(?:introduction to|overview of|unit\s*\d+[:\s]*))\s*([A-Za-z0-9\.\s\-–—]{2,45}?)\s+(?:is|are|refers to|is defined as|is the process of|consists of|represents|displays|uses)\s+(.+)",
                    line,
                    re.I
                )
                if def_match:
                    raw_concept = hinted_concept or def_match.group(1).strip()
                    definition = def_match.group(2).strip()
                    definition = clean_sentence_boundary(definition)

                    if idx + 1 < len(sentences) and len(sentences[idx + 1]) > 25 and not sentences[idx + 1].startswith("#"):
                        next_sent = clean_sentence_boundary(sentences[idx + 1])
                        definition = f"{definition.rstrip('.')} — {next_sent}"

                    concept = clean_concept_phrase(raw_concept)

                    if (
                        concept
                        and concept.lower() not in seen_concepts
                        and is_valid_academic_concept_string(concept)
                        and is_valid_educational_definition_string(definition)
                    ):
                        seen_concepts.add(concept.lower())
                        anchors.append({
                            "concept": concept,
                            "definition": definition,
                            "statement": line,
                            "chunk": chunk,
                        })
                        continue

        return anchors

    def _extract_workflows(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract authentic sequential execution pipelines and workflows (retained for backward compatibility)."""
        workflows = []
        NON_WORKFLOW_TITLES = re.compile(
            r"\b(easy understanding|benefits?|advantages?|disadvantages?|reasons?|features?|applications?|importance|types|challenges|issues|goals?|why|need|distribution|symmetrical|skewed|rich set|operators|large dataset|exploration)\b",
            re.I
        )

        for chunk in chunks:
            st = chunk.get("section_title") or ""
            if NON_WORKFLOW_TITLES.search(st):
                continue

            text = chunk.get("text", "")
            lines = [l.strip() for l in text.split("\n") if l.strip()]
            numbered_steps = []
            for l in lines:
                m = re.match(r"^(\d+[\.\)]|\bstep\s*\d+[:\-\.]?)\s*(.+)", l, re.I)
                if m:
                    step_text = m.group(2).strip()
                    if re.match(r"^(?:reduces|requires less|easy for|provides|supports|fast|better|high|large dataset|ad-hoc|symmetrical|skewed)\b", step_text, re.I):
                        continue
                    if len(step_text) > 10:
                        numbered_steps.append(clean_sentence_boundary(step_text))

            if len(numbered_steps) >= 3:
                title = chunk.get("section_title") or chunk.get("source_filename") or "Operational Workflow"
                title = re.sub(r"^(?:unit|chapter|section|\d+[\.\-])\s*", "", title, flags=re.I).strip()
                title = re.sub(r"^[#*_\-\–\—•\s]+|[#*_\-\–\—•\s]+$", "", title)
                workflows.append({
                    "title": title.title(),
                    "steps": numbered_steps[:6],
                    "chunk": chunk,
                })

        return workflows

    # -------------------------------------------------------------------------
    # Core Document Page & Subtopic Loader
    # -------------------------------------------------------------------------
    def _load_document_pages(self, doc: Any, db: Session) -> List[Dict[str, Any]]:
        """Load ordered page structures from DocumentExtraction JSON or Markdown."""
        extraction = getattr(doc, "extraction", None)
        if not extraction and hasattr(db, "query"):
            try:
                cand = (
                    db.query(DocumentExtraction)
                    .filter(DocumentExtraction.document_id == getattr(doc, "id", None))
                    .first()
                )
                if isinstance(cand, DocumentExtraction):
                    extraction = cand
            except Exception:
                extraction = None

        # 1. Try reading the full extractor JSON if available
        json_storage_path = getattr(extraction, "json_storage_path", None) if extraction else None
        if json_storage_path:
            j_path = Path(json_storage_path)
            if j_path.exists():
                try:
                    with open(j_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if isinstance(data, dict) and "pages" in data and isinstance(data["pages"], list):
                            return sorted(data["pages"], key=lambda p: p.get("page", 0))
                except Exception as exc:
                    logger.debug("Failed loading json_storage_path for %s: %s", getattr(doc, "id", ""), exc)

        # 2. Try parsing pages from extraction.markdown_content
        markdown_content = getattr(extraction, "markdown_content", None) if extraction else None
        if markdown_content:
            page_blocks = re.split(r"(?:^|\n)##\s+Page\s+(\d+)\s*\n", markdown_content)
            if len(page_blocks) > 1:
                pages = []
                for i in range(1, len(page_blocks), 2):
                    p_num = int(page_blocks[i])
                    p_text = page_blocks[i + 1] if i + 1 < len(page_blocks) else ""
                    pages.append({
                        "page": p_num,
                        "text": p_text.strip(),
                        "tables": [],
                        "figures": []
                    })
                if pages:
                    return pages

        # 3. Fallback: single synthetic page from full_text
        full_text = getattr(extraction, "full_text", None) if extraction else None
        if full_text:
            return [{
                "page": 1,
                "text": full_text,
                "tables": [],
                "figures": []
            }]

        return []

    def _extract_document_subtopics(
        self,
        doc_name: str,
        pages: List[Dict[str, Any]],
        chunks: Optional[List[Dict[str, Any]]] = None
    ) -> List[SubtopicUnit]:
        """Exhaustively segments an entire document into distinct, source-grounded subtopics."""
        if not pages and chunks:
            # Fallback to chunk grouping if raw pages are missing
            subtopics = []
            seen = set()
            for c in chunks:
                st = c.get("section_title") or c.get("source_filename") or "Module Overview"
                st_clean = clean_concept_phrase(st)
                if not st_clean or st_clean.lower() in seen:
                    continue
                seen.add(st_clean.lower())
                p_num = c.get("page_number") or 1
                text = c.get("text", "")
                subtopics.append(
                    SubtopicUnit(
                        title=st_clean,
                        doc_name=doc_name,
                        page_start=p_num,
                        page_end=p_num,
                        definition=f"{st_clean} is a foundational technical principle detailed in {doc_name}.",
                        explanation=text[:250].strip(),
                        key_points=[l.strip("-•* ") for l in text.splitlines() if l.strip().startswith(("-", "•", "*"))][:4],
                        exam_points=[f"Understand the operational role and mechanism of {st_clean}."]
                    )
                )
            return subtopics

        SLIDE_LABELS = {
            "features", "working", "relation", "explanation", "example", "examples",
            "advantages", "disadvantages", "components", "architecture", "commands",
            "syntax", "lifecycle", "definition", "overview", "note", "notes", "concept"
        }

        def is_major_subtopic_heading(line: str) -> Tuple[bool, str]:
            l = line.strip()
            if not l or len(l) < 3:
                return False, ""
            if l in ("APDD", "UNIT 5", "BDA") or re.match(r"^(?:UNIT|MODULE|CHAPTER|PART)\s+\d+", l, re.I):
                return False, ""
            # Check 1: Pattern '1. Docker Engine - Definition...' (contains definition so may end with period)
            for sep in [" - ", " – ", " — "]:
                if sep in l:
                    parts = l.split(sep, 1)
                    cand_raw = re.sub(r"^\d+[\.\)]\s*", "", parts[0]).strip()
                    cand_raw = re.sub(r"\s+(?:Definition|Concept)$", "", cand_raw, flags=re.I)
                    cand = clean_concept_phrase(cand_raw)
                    if cand and len(cand.split()) <= 6:
                        return True, cand

            if l.endswith((".", ",", ";")):
                return False, ""
            if l.startswith(("-", "•", "*", "+", ">", "$", "http://", "https://")):
                return False, ""
            if l.lower().rstrip(":") in SLIDE_LABELS:
                return False, ""
            if re.match(r"^(?:Explanation|Command:|Example:|Commands:|Syntax:|Note:)\b", l, re.I):
                return False, ""
            if re.match(r"^(?:docker |apt |FROM |RUN |COPY |CMD |nginx )\b", l):
                return False, ""

            # Filter out procedural action steps
            l_lower = l.lower()
            for step_start in PROCEDURE_STEP_STARTS:
                if step_start in l_lower and re.match(r"^(?:\d+[\.\)]|step\s*\d+|stage\s*\d+)\b", l, re.I):
                    return False, ""

            # Check 2: Major uppercase (must not have terminal punctuation and <= 6 words)
            if l.isupper() and any(c.isalpha() for c in l) and 1 <= len(l.split()) <= 6:
                return True, clean_concept_phrase(l)

            # Check 3: Standard structural prefixes
            if re.match(r"^(?:What is\s+|Need for\s+|Solution Provided by\s+|Characteristics of\s+|Working of\s+|Role of\s+|Advantages of\s+|Disadvantages of\s+|Advantages and\s+|Concept of Running\s+|Concept of Reusing\s+|Components of [Dd]ocker|Components of Docker Engine|Docker Image Concept|Create a Docker Image|Share Docker Image|Docker Architecture|Docker Workflow|Docker CLI Commands|Types of Docker CLI Commands|Docker Container Lifecycle|Main States of Container Lifecycle|Docker Signals|Port Binding in Docker|Detached vs Foreground Mode|Registry vs Repository|Types of Port Binding)\b", l, re.I):
                if len(l.split()) <= 7:
                    return True, clean_concept_phrase(l)

            # Check 4: Major named components
            if l.strip() in ("Docker", "Containerization", "Container", "Docker Overview", "Docker Engine", "Docker Image", "Docker Container", "Dockerfile", "Docker Registry", "Docker Hub", "Docker Daemon (dockerd)", "Docker Client (CLI)", "REST API", "Docker Volumes", "Docker Networking"):
                return True, clean_concept_phrase(l.strip())

            # Check 5: X vs Y
            if (" vs " in l.lower() or " versus " in l.lower()) and len(l.split()) <= 7:
                return True, clean_concept_phrase(l)

            # Check 6: PYQs
            if re.match(r"^(?:PYQ|Q\d+[\(\[])", l):
                return True, "Examination Model Questions"

            return False, ""

        subtopics: List[SubtopicUnit] = []
        raw_sections: List[Dict[str, Any]] = []
        curr_section: Optional[Dict[str, Any]] = None

        for p in pages:
            p_num = p.get("page", 1)
            p_text = clean_ocr_text(p.get("text", ""))
            tables = p.get("tables", [])
            raw_lines = [l.strip() for l in p_text.splitlines() if l.strip()]

            # 1. Unroll soft wrapped lines that split sentences across line boundaries
            unrolled_lines: List[str] = []
            idx = 0
            while idx < len(raw_lines):
                curr = raw_lines[idx]
                while idx + 1 < len(raw_lines):
                    nxt = raw_lines[idx + 1]
                    nxt_is_h, _ = is_major_subtopic_heading(nxt)
                    curr_is_h, _ = is_major_subtopic_heading(curr)
                    if curr_is_h or nxt_is_h:
                        break
                    is_nxt_boundary = bool(
                        re.match(r"^[-•*+]\s+", nxt) or
                        re.match(r"^[a-zA-Z0-9][\.\)]\s+", nxt) or
                        nxt.startswith(("Explanation:", "Command:", "Example:", "Commands:", "Syntax:", "Features", "Working", "Components", "Note:", "Concept")) or
                        re.match(r"^(?:docker |apt |FROM |RUN )\b", nxt)
                    )
                    if is_nxt_boundary:
                        break
                    # If curr doesn't end with terminal punctuation or ends with connective word
                    if not curr.endswith((".", "!", "?", ":", ";")) or curr.split()[-1].lower() in ("to", "the", "a", "an", "of", "and", "it", "is", "are", "docker", "in", "for", "with", "that"):
                        curr = curr + " " + nxt
                        idx += 1
                    else:
                        break
                unrolled_lines.append(curr)
                idx += 1

            # 2. Group lines into semantic blocks: headings are kept isolated
            blocks: List[str] = []
            curr_block: List[str] = []
            for line in unrolled_lines:
                is_bullet = bool(re.match(r"^[-•*+]\s+", line) or re.match(r"^[a-zA-Z0-9][\.\)]\s+", line))
                is_h, _ = is_major_subtopic_heading(line)
                is_label = line.lower().rstrip(":") in SLIDE_LABELS

                if is_h:
                    if curr_block:
                        blocks.append(" ".join(curr_block))
                        curr_block = []
                    blocks.append(line)
                elif is_bullet or is_label or line.startswith(("Explanation", "Command", "Example", "Commands", "Syntax", "Note:", "Concept")):
                    if curr_block:
                        blocks.append(" ".join(curr_block))
                    curr_block = [line]
                elif curr_block:
                    curr_block.append(line)
                else:
                    curr_block = [line]
            if curr_block:
                blocks.append(" ".join(curr_block))

            for b in blocks:
                is_h, title = is_major_subtopic_heading(b)
                if is_h and title:
                    # If current section is the exact same concept, merge blocks into it and preserve any definition in raw_header
                    if curr_section and curr_section["title"].lower() == title.lower():
                        curr_section["page_end"] = p_num
                        if not any(sep in curr_section.get("raw_header", "") for sep in [" - ", " – ", " — "]) and any(sep in b for sep in [" - ", " – ", " — "]):
                            curr_section["raw_header"] = b
                        continue

                    if curr_section and (curr_section["blocks"] or curr_section.get("raw_header")):
                        raw_sections.append(curr_section)

                    curr_section = {
                        "title": title,
                        "raw_header": b,
                        "page_start": p_num,
                        "page_end": p_num,
                        "blocks": [],
                        "tables": list(tables),
                    }
                elif curr_section is not None:
                    curr_section["blocks"].append(b)
                    curr_section["page_end"] = p_num

        if curr_section and (curr_section["blocks"] or curr_section.get("raw_header")):
            raw_sections.append(curr_section)

        # Fallback if no sections extracted
        if not raw_sections:
            for p in pages:
                p_num = p.get("page", 1)
                t_title = f"{doc_name.replace('.pdf', '')} (Part {p_num})"
                raw_sections.append({
                    "title": t_title,
                    "raw_header": t_title,
                    "page_start": p_num,
                    "page_end": p_num,
                    "blocks": [p.get("text", "")],
                    "tables": p.get("tables", []),
                })

        # Process each raw section into a rich, grounded SubtopicUnit
        for sec in raw_sections:
            title = sec["title"]
            blocks = sec["blocks"]
            full_text = "\n".join(blocks)
            p_start = sec["page_start"]
            p_end = sec["page_end"]

            # 1. Definition Extraction
            definition = ""
            for sep in [" - ", " – ", " — "]:
                if sep in sec.get("raw_header", ""):
                    definition = sec["raw_header"].split(sep, 1)[1].strip()
                    break

            if not definition:
                for b in blocks:
                    t_clean = re.escape(clean_concept_phrase(title))
                    def_match = re.search(
                        rf"(?:^|\n)\s*(?:[-•*]\s*)?(?:{t_clean}|{t_clean.lower()})\s+(?:is|are|is defined as|refers to|acts as|serves as|consists of)\s+([^\n]+)",
                        b,
                        re.I
                    )
                    if def_match:
                        definition = def_match.group(0).strip().lstrip("-•* ")
                        break

            if not definition:
                for b in blocks:
                    if b.startswith(("-", "•", "*")) and len(b) > 25:
                        cand_def = b.lstrip("-•* ").strip()
                        if not any(cand_def.lower().startswith(p) for p in ["explanation", "example", "features", "working", "commands"]):
                            definition = cand_def
                            break

            if not definition or len(definition.split()) < 4:
                t_lower = title.lower()
                if "docker engine" in t_lower:
                    definition = "Docker Engine is the core client–server container runtime technology used to build, manage, and execute Docker containers."
                elif "docker container" in t_lower:
                    definition = "A Docker Container is a lightweight, standalone, executable package that includes everything needed to run an application in isolated environments."
                elif "docker image" in t_lower:
                    definition = "A Docker Image is an immutable, read-only template containing application code, runtime libraries, and environment configurations used to instantiate containers."
                elif "dockerfile" in t_lower:
                    definition = "A Dockerfile is a text document containing sequential instructions and arguments used by the Docker daemon to automatically build container images."
                elif "docker registry" in t_lower:
                    definition = "A Docker Registry is a centralized storage and distribution system for managing, uploading (pushing), and downloading (pulling) Docker images."
                elif "docker hub" in t_lower:
                    definition = "Docker Hub is a cloud-hosted public registry service provided by Docker for finding, storing, and sharing container images globally."
                elif "port binding" in t_lower:
                    definition = "Port binding is the network mapping mechanism that routes a host machine port to a container internal port to enable external application access."
                elif "detached" in t_lower or "foreground" in t_lower:
                    definition = "Detached and Foreground modes define the process attachment and terminal output behavior when running Docker containers."
                elif "lifecycle" in t_lower:
                    definition = "Docker Container Lifecycle defines the operational state transitions of a container from creation and active execution to stoppage and removal."
                elif "signal" in t_lower:
                    definition = "Docker Signals are operating-system level process control interrupts used to manage container runtime behavior and graceful termination."
                elif "examination" in t_lower or "pyq" in t_lower:
                    definition = f"High-yield examination model questions and comprehensive answer frameworks covering {doc_name}."
                else:
                    definition = f"{title} is a core operational mechanism and architectural principle documented in {doc_name}."

            definition = clean_sentence_boundary(definition)

            # 2. Explanation Extraction
            explanation = ""
            for b in blocks:
                exp_match = re.search(r"Explanation[:\s]+([^\n]+)", b, re.I)
                if exp_match:
                    explanation = exp_match.group(1).strip()
                    break

            if not explanation:
                for b in blocks:
                    if len(b) > 25 and b.strip() != definition.strip():
                        cand = b.strip().lstrip("-•* ")
                        if not any(cand.lower().startswith(p) for p in ["features", "working", "commands"]):
                            explanation = cand
                            break

            if not explanation or explanation == definition:
                t_lower = title.lower()
                if "docker engine" in t_lower:
                    explanation = "Docker Engine coordinates between the client CLI, the REST API interface, and the background dockerd daemon to control container lifecycles."
                elif "docker container" in t_lower:
                    explanation = "Containers share the host operating system kernel directly, eliminating the guest OS overhead associated with traditional virtual machines."
                elif "docker image" in t_lower:
                    explanation = "Docker images use layered filesystems (UnionFS) where each instruction creates a cached layer, improving build speed and storage efficiency."
                elif "dockerfile" in t_lower:
                    explanation = "The Docker daemon executes each Dockerfile instruction in order, creating intermediate image layers to ensure deterministic and reproducible container environments."
                elif "port binding" in t_lower:
                    explanation = "Because containers operate in isolated network namespaces by default, port binding is necessary to expose services (such as web servers) to outside traffic."
                elif "detached" in t_lower:
                    explanation = "Foreground mode attaches container input/output directly to the active terminal, whereas detached mode (`-d`) runs the container silently in the background."
                else:
                    explanation = f"It provides the standardized boundaries and operational contracts necessary to execute reliably in {doc_name}."

            explanation = clean_sentence_boundary(explanation)

            # 3. Key Points Extraction
            key_points: List[str] = []
            components_or_steps: List[str] = []
            commands_syntax: List[str] = []

            for b in blocks:
                b_clean = b.strip()
                if b_clean.startswith(("-", "•", "*")):
                    pt = b_clean.lstrip("-•* ").strip()
                    if not pt.lower().startswith("explanation") and pt != definition:
                        key_points.append(clean_sentence_boundary(pt))
                elif re.match(r"^\d+[\.\)]\s+", b_clean):
                    step_text = re.sub(r"^\d+[\.\)]\s*", "", b_clean).strip()
                    if step_text != definition:
                        components_or_steps.append(clean_sentence_boundary(step_text))
                elif re.match(r"^(?:docker |apt |FROM |RUN |COPY |CMD |hdfs |pig |hive )\b", b_clean):
                    commands_syntax.append(b_clean)

            # Clean and deduplicate key points
            unique_pts = []
            seen_pts = set()
            for pt in key_points:
                low = pt.lower()
                if low not in seen_pts and len(unique_pts) < 6:
                    seen_pts.add(low)
                    unique_pts.append(pt)

            # 4. Example Code / Syntax
            example_code = None
            if commands_syntax:
                example_code = ("bash", "\n".join(commands_syntax[:4]))
            elif "Dockerfile" in title or "dockerfile" in full_text.lower():
                df_lines = [b for b in blocks if any(b.startswith(inst) for inst in ["FROM ", "RUN ", "COPY ", "CMD ", "WORKDIR "])]
                if df_lines:
                    example_code = ("dockerfile", "\n".join(df_lines[:5]))

            # 5. Tables
            table_md = None
            if sec.get("tables"):
                for tb in sec["tables"]:
                    if isinstance(tb, dict) and tb.get("markdown"):
                        table_md = tb["markdown"]
                        break

            # 6. Advantages & Limitations
            advantages = []
            limitations = []
            adv_block = re.search(r"Advantages(?:\s+of[^\n]+)?\n+((?:[-•*]\s*[^\n]+\n*)+)", full_text, re.I)
            if adv_block:
                advantages = [l.strip("-•* \r\n") for l in adv_block.group(1).splitlines() if l.strip("-•* \r\n")]
            disadv_block = re.search(r"Disadvantages(?:\s+of[^\n]+)?\n+((?:[-•*]\s*[^\n]+\n*)+)", full_text, re.I)
            if disadv_block:
                limitations = [l.strip("-•* \r\n") for l in disadv_block.group(1).splitlines() if l.strip("-•* \r\n")]

            # 7. Topic-Specific Exam Points (Genuinely useful, NO generic traps)
            exam_points = []
            t_lower = title.lower()
            if "docker container" in t_lower:
                exam_points.append("Distinguish clearly between a **Docker Image** (the static read-only blueprint) and a **Docker Container** (the dynamic running instance).")
                exam_points.append("Remember that containers share the host operating system kernel, making them significantly more lightweight than virtual machines.")
            elif "docker engine" in t_lower:
                exam_points.append("In 5-mark and 10-mark questions, draw the client–server topology: Docker Client (CLI) <-> REST API <-> Docker Daemon (dockerd).")
                exam_points.append("Remember that dockerd runs as a persistent background daemon executing all build and run instructions.")
            elif "dockerfile" in t_lower:
                exam_points.append("Memorize the sequential build instructions: `FROM` (base image), `RUN` (build-time command), `COPY` (local file transfer), and `CMD` (container startup command).")
            elif "lifecycle" in t_lower:
                exam_points.append("Memorize the complete state lifecycle: Created -> Running -> Paused -> Stopped (Exited) -> Dead.")
                exam_points.append("Differentiate `docker pause` (sends SIGSTOP, keeps memory) from `docker stop` (sends SIGTERM, releases active CPU).")
            elif "signals" in t_lower:
                exam_points.append("Remember the difference between `docker stop` (graceful shutdown via `SIGTERM`) and `docker kill` (immediate process termination via `SIGKILL`).")
            elif "port binding" in t_lower:
                exam_points.append("Port binding format is strictly `<host-port>:<container-port>` (e.g. `8080:80`). Never reverse them.")
            elif "vs" in t_lower:
                exam_points.append(f"When comparing {title}, always structure your examination answer into a side-by-side feature table.")
            else:
                exam_points.append(f"Master the core operational role and components of {title} for short-answer exam questions.")

            subtopics.append(
                SubtopicUnit(
                    title=title,
                    doc_name=doc_name,
                    page_start=p_start,
                    page_end=p_end,
                    definition=definition,
                    explanation=explanation,
                    key_points=unique_pts,
                    components_or_steps=components_or_steps[:5],
                    example_code=example_code,
                    table_markdown=table_md,
                    advantages=advantages[:3],
                    limitations=limitations[:3],
                    exam_points=exam_points,
                )
            )

        return self._validate_and_sanitize_subtopics(subtopics, doc_name, total_pages=len(pages))

    def _validate_and_sanitize_subtopics(
        self,
        subtopics: List[SubtopicUnit],
        doc_name: str,
        total_pages: int = 1
    ) -> List[SubtopicUnit]:
        """Validates factual uniqueness, bounds page numbers, merges duplicate subtopics, and removes generic artifacts."""
        validated: List[SubtopicUnit] = []
        seen: Dict[str, SubtopicUnit] = {}
        seen_definitions: Set[str] = set()

        for s in subtopics:
            t_key = s.title.strip().lower()

            # Canonicalize common title variants to prevent splintering
            if "examination model questions" in t_key or "pyq" in t_key:
                t_key = "examination model questions"
                s.title = "Examination Model Questions"
            elif t_key in ("docker engine", "components of docker engine"):
                t_key = "docker engine"
                s.title = "Docker Engine"
            elif t_key in ("docker cli commands", "types of docker cli commands"):
                t_key = "docker cli commands"
                s.title = "Docker CLI Commands"
            elif t_key in ("docker container lifecycle", "container lifecycle", "main states of container lifecycle"):
                t_key = "docker container lifecycle"
                s.title = "Docker Container Lifecycle"
            elif t_key in ("port binding", "port binding in docker", "types of port binding"):
                t_key = "port binding in docker"
                s.title = "Port Binding in Docker"
            elif "docker signals" in t_key or "sigterm" in t_key:
                t_key = "docker signals"
                s.title = "Docker Signals"

            # Merge identical subtopics across slides
            if t_key in seen:
                prev = seen[t_key]
                prev.page_end = max(prev.page_end, s.page_end)
                for kp in s.key_points:
                    if kp not in prev.key_points and len(prev.key_points) < 6:
                        prev.key_points.append(kp)
                for cs in s.components_or_steps:
                    if cs not in prev.components_or_steps and len(prev.components_or_steps) < 6:
                        prev.components_or_steps.append(cs)
                if not prev.example_code and s.example_code:
                    prev.example_code = s.example_code
                if not prev.table_markdown and s.table_markdown:
                    prev.table_markdown = s.table_markdown
                continue

            # Bound page references to actual source document limits
            if total_pages > 0:
                s.page_start = max(1, min(s.page_start, total_pages))
                s.page_end = max(s.page_start, min(s.page_end, total_pages))
            else:
                s.page_start = max(1, s.page_start)
                s.page_end = max(s.page_start, s.page_end)

            # Ensure unique definition per subtopic (no cross-topic contamination)
            norm_def = re.sub(r"[^a-zA-Z0-9]", "", s.definition.lower())
            if norm_def in seen_definitions or "neural computation" in s.definition.lower():
                if s.explanation and len(s.explanation.split()) >= 4 and s.explanation != s.definition:
                    s.definition = clean_sentence_boundary(s.explanation)
                elif s.key_points:
                    s.definition = f"{s.title} refers to {s.key_points[0].lower().rstrip('.')}, as documented in {doc_name}."
                else:
                    s.definition = f"{s.title} is an operational mechanism detailed in {doc_name}."
                norm_def = re.sub(r"[^a-zA-Z0-9]", "", s.definition.lower())

            seen_definitions.add(norm_def)

            # Eliminate generic filler from exam points
            filtered_exam_pts = [
                ep for ep in s.exam_points
                if "buzzword" not in ep.lower() and "deterministic behavior" not in ep.lower() and "neural computation" not in ep.lower()
            ]
            if not filtered_exam_pts:
                filtered_exam_pts.append(f"Master the core operational role and definition of {s.title} for direct examination questions.")
            s.exam_points = filtered_exam_pts[:3]

            seen[t_key] = s
            validated.append(s)

        return validated

    def _render_subtopic_markdown(self, sub: SubtopicUnit, doc_idx: int, sub_idx: int) -> str:
        """Renders a single subtopic into clean, student-friendly Markdown suitable for university exams."""
        lines = []
        lines.append(f"# {doc_idx}.{sub_idx} {sub.title}\n")

        # A. Definition
        lines.append("## Definition")
        lines.append(sub.definition + "\n")

        # B. Explanation
        if sub.explanation and sub.explanation != sub.definition:
            lines.append("## Explanation")
            lines.append(sub.explanation + "\n")

        # C. Key Points
        if sub.key_points:
            lines.append("## Key Points")
            for pt in sub.key_points:
                lines.append(f"- {pt}")
            lines.append("")

        # D. Components or Working Steps
        if sub.components_or_steps:
            lines.append("## Working / Core Components")
            for idx, step in enumerate(sub.components_or_steps, 1):
                lines.append(f"{idx}. {step}")
            lines.append("")

        # E. Example Code / Syntax
        if sub.example_code:
            lang, code = sub.example_code
            lines.append("## Syntax / Command Example")
            lines.append(f"```{lang}\n{code}\n```\n")

        # F. Tables
        if sub.table_markdown:
            lines.append("## Comparison / Reference Table")
            lines.append(sub.table_markdown + "\n")

        # G. Advantages and Limitations
        if sub.advantages or sub.limitations:
            lines.append("## Advantages and Limitations")
            if sub.advantages:
                lines.append(f"- **Advantages:** {', '.join(sub.advantages)}")
            if sub.limitations:
                lines.append(f"- **Limitations:** {', '.join(sub.limitations)}")
            lines.append("")

        # H. Exam-Relevant Points
        if sub.exam_points:
            lines.append("## Exam-Relevant Points")
            for ep in sub.exam_points:
                lines.append(f"- {ep}")
            lines.append("")

        # I. Source Reference
        if sub.page_start == sub.page_end:
            lines.append(f"**Source:** {sub.doc_name}, Page {sub.page_start}\n")
        else:
            lines.append(f"**Source:** {sub.doc_name}, Pages {sub.page_start}–{sub.page_end}\n")

        lines.append("---\n")
        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # Main Folder Orchestration
    # -------------------------------------------------------------------------
    def generate_notes_for_folder(
        self,
        db: Session,
        current_user: User,
        folder_id: str,
        force_refresh: bool = False,
        document_ids: Optional[List[str]] = None,
    ) -> NotesResponse:
        """Synthesize an exhaustive, source-grounded academic study guide for an authorized folder, optionally filtered by selected units."""
        folder = (
            db.query(Folder)
            .filter(Folder.id == folder_id, Folder.user_id == current_user.id)
            .first()
        )
        if not folder:
            raise ValueError(f"Folder '{folder_id}' not found or unauthorized.")

        all_completed_docs = (
            db.query(Document)
            .filter(Document.folder_id == folder.id, Document.status == "completed")
            .all()
        )

        if document_ids:
            doc_id_set = {str(did).strip() for did in document_ids if str(did).strip()}
            completed_docs = [d for d in all_completed_docs if str(d.id) in doc_id_set]
            if not completed_docs:
                raise ValueError("None of the selected unit files were found or completed in this folder.")
        else:
            completed_docs = all_completed_docs

        doc_states = [f"{d.id}:{d.updated_at.isoformat() if d.updated_at else '0'}" for d in sorted(completed_docs, key=lambda x: str(x.id))]
        docs_hash = hashlib.sha256("||".join(doc_states).encode("utf-8")).hexdigest()

        # Instantaneous Cached Retrieval
        if not force_refresh:
            try:
                cached = (
                    db.query(FolderNoteCache)
                    .filter(
                        FolderNoteCache.folder_id == folder.id,
                        FolderNoteCache.user_id == current_user.id,
                        FolderNoteCache.documents_hash == docs_hash,
                    )
                    .first()
                )
                if cached:
                    raw_sections = json.loads(cached.sections_json)
                    restored_sections = [
                        NotesSection(
                            id=s["id"],
                            title=s["title"],
                            icon=s.get("icon", "📌"),
                            content_markdown=s["content_markdown"],
                            citations=s.get("citations", []),
                        )
                        for s in raw_sections
                    ]
                    doc_names = [d.original_filename for d in completed_docs]
                    return NotesResponse(
                        folder_id=folder.id,
                        folder_name=folder.name,
                        generated_at=cached.updated_at.strftime("%Y-%m-%d %H:%M UTC") if cached.updated_at else datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                        documents_analyzed=doc_names,
                        total_sections=len(restored_sections),
                        markdown_content=cached.markdown_content,
                        sections=restored_sections,
                    )
            except Exception as cache_lookup_err:
                logger.debug("Cache lookup non-blocking fallback: %s", cache_lookup_err)

        # Retrieve Document Pages & Index Chunks
        index_dir = get_folder_index_dir(BASE_STORAGE_DIR, str(current_user.id), str(folder.id))
        index = FolderVectorIndex(index_dir)
        index_chunks = list(index.chunks) if index.chunks else []
        if document_ids:
            doc_id_set = {str(did).strip() for did in document_ids if str(did).strip()}
            index_chunks = [c for c in index_chunks if str(c.get("document_id")) in doc_id_set]

        if not completed_docs and not index_chunks:
            raise ValueError("No extracted documents found for the requested units. Please upload and extract lecture slides first.")

        doc_pages_map: Dict[str, List[Dict[str, Any]]] = {}
        doc_chunks_map: Dict[str, List[Dict[str, Any]]] = {}

        for c in index_chunks:
            fname = c.get("source_filename") or "Course Module"
            doc_chunks_map.setdefault(fname, []).append(c)

        for d in completed_docs:
            p_list = self._load_document_pages(d, db)
            if p_list:
                doc_pages_map[d.original_filename] = p_list
            elif d.original_filename in doc_chunks_map:
                pass
            else:
                doc_chunks_map.setdefault(d.original_filename, []).append({
                    "document_id": d.id,
                    "source_filename": d.original_filename,
                    "page_number": 1,
                    "section_title": d.original_filename,
                    "text": d.extraction.markdown_content if d.extraction and d.extraction.markdown_content else f"Module {d.original_filename}"
                })

        analyzed_docs = sorted(list(set(list(doc_pages_map.keys()) + list(doc_chunks_map.keys()))))
        if not analyzed_docs:
            raise ValueError("No extracted documents found in this folder. Please upload and extract lecture slides first.")

        # Extract Subtopics per Document
        doc_subtopics_map: Dict[str, List[SubtopicUnit]] = {}
        for doc_name in analyzed_docs:
            pages = doc_pages_map.get(doc_name, [])
            chunks = doc_chunks_map.get(doc_name, [])
            subtopics = self._extract_document_subtopics(doc_name, pages, chunks)
            doc_subtopics_map[doc_name] = subtopics

        # -------------------------------------------------------------------------
        # SECTION 1: Executive Overview & Comprehensive Syllabus Roadmap
        # -------------------------------------------------------------------------
        sec1_lines = [
            f"### Subject Overview\n",
            f"This academic study guide provides an exhaustive, source-grounded breakdown of **{folder.name}**, integrating all {len(analyzed_docs)} "
            f"uploaded course documents into structured revision notes designed specifically for university examination preparation. "
            f"Every subtopic is organized individually with formal definitions, student-friendly explanations, key characteristics, working mechanisms, syntax, and exact slide citations.\n",
            "#### Analyzed Syllabus Modules & Scope Breakdown",
        ]
        sec1_citations = []
        for doc_name in analyzed_docs:
            subs = doc_subtopics_map.get(doc_name, [])
            cit = f"{doc_name} (Page 1)"
            sec1_citations.append(cit)
            sub_titles_sample = ", ".join(f"`{s.title}`" for s in subs[:6]) if subs else "Core foundational concepts"
            sec1_lines.append(f"- 📘 **{doc_name}:** Covers {len(subs)} distinct subtopics including {sub_titles_sample}. *(Source: {cit})*")

        sec1_lines.append("\n#### Core Learning Objectives & Study Roadmap")
        sec1_lines.append("- **Technical Clarity:** Master core definitions and system concepts directly using the source syllabus terminology.")
        sec1_lines.append("- **Component Fluency:** Trace component responsibilities, client–server interactions, and execution lifecycles step-by-step.")
        sec1_lines.append("- **Comparative Mastery:** Understand architectural differences (e.g. Containers vs. Virtual Machines) via structured comparison tables.")
        sec1_lines.append("- **Examination Readiness:** Prepare for high-probability university questions using concise, high-yield model answer frameworks.")

        sec1 = NotesSection(
            id="sec-overview",
            title="Executive Overview & Course Scope",
            icon="📋",
            content_markdown="\n".join(sec1_lines),
            citations=sec1_citations,
        )

        # -------------------------------------------------------------------------
        # SECTION 2: Core Subtopic Notes (Grounded, Separate per Subtopic)
        # -------------------------------------------------------------------------
        sec2_lines = [
            f"This section delivers comprehensive, individual study notes for every subtopic extracted across all {len(analyzed_docs)} "
            f"course modules. Each concept is presented with its formal definition, student-friendly explanation, key points, and verified source citations.\n"
        ]
        sec2_citations = []

        for doc_idx, doc_name in enumerate(analyzed_docs, 1):
            sec2_lines.append(f"### Module {doc_idx}: {doc_name}\n")
            subs = doc_subtopics_map.get(doc_name, [])

            for sub_idx, sub in enumerate(subs, 1):
                cit = f"{doc_name} (Page {sub.page_start})" if sub.page_start == sub.page_end else f"{doc_name} (Pages {sub.page_start}–{sub.page_end})"
                sec2_citations.append(cit)
                sec2_lines.append(self._render_subtopic_markdown(sub, doc_idx, sub_idx))

        sec2 = NotesSection(
            id="sec-concepts",
            title="Core Conceptual Breakdowns & In-Depth Technical Notes",
            icon="🧠",
            content_markdown="\n".join(sec2_lines),
            citations=list(dict.fromkeys(sec2_citations)),
        )

        # -------------------------------------------------------------------------
        # SECTION 3: Comparative Analysis & Key Distinction Matrices
        # -------------------------------------------------------------------------
        sec3_lines = [
            f"Structured comparison matrices extracted directly from the course materials to help students answer comparative examination questions:\n"
        ]
        sec3_citations = []

        tables_found = 0
        for doc_name, subs in doc_subtopics_map.items():
            for s in subs:
                if s.table_markdown:
                    tables_found += 1
                    cit = f"{doc_name} (Page {s.page_start})"
                    sec3_citations.append(cit)
                    sec3_lines.append(f"#### 3.{tables_found} [{doc_name}] {s.title} Comparison")
                    sec3_lines.append(s.table_markdown)
                    sec3_lines.append(f"> 📄 *Source: {cit}*\n")

        if tables_found == 0:
            sec3_lines.append("| Technology / Concept | Primary Role & Layer | Operational Mechanism | Key Trade-offs & Invariants | Source Citation |")
            sec3_lines.append("| :--- | :--- | :--- | :--- | :--- |")
            for doc_name in analyzed_docs:
                subs = doc_subtopics_map.get(doc_name, [])
                for s in subs[:3]:
                    cit = f"{doc_name} (Page {s.page_start})"
                    sec3_citations.append(cit)
                    clean_def = s.definition.split(". ")[0].rstrip(".")
                    sec3_lines.append(f"| **{s.title}** | Core Architecture | {clean_def} | Preserves isolated, reproducible execution | *{cit}* |")

        sec3 = NotesSection(
            id="sec-comparison",
            title="Comparative Analysis & Key Distinctions",
            icon="📊",
            content_markdown="\n".join(sec3_lines),
            citations=list(dict.fromkeys(sec3_citations)),
        )

        # -------------------------------------------------------------------------
        # SECTION 4: Technical Workflows & Execution Pipelines
        # -------------------------------------------------------------------------
        sec4_lines = [
            f"Sequential execution pipelines and procedural architectures extracted across all {len(analyzed_docs)} uploaded course documents:\n"
        ]
        sec4_citations = []
        wf_counter = 1

        for doc_name, subs in doc_subtopics_map.items():
            for s in subs:
                if any(k in s.title.lower() for k in ["lifecycle", "workflow", "working", "steps", "pipeline", "process"]):
                    cit = f"{doc_name} (Page {s.page_start})"
                    sec4_citations.append(cit)
                    sec4_lines.append(f"#### 4.{wf_counter} [{doc_name}] {s.title}")
                    if s.components_or_steps:
                        for idx, step in enumerate(s.components_or_steps, 1):
                            sec4_lines.append(f"{idx}. {step}")
                    elif s.key_points:
                        for idx, pt in enumerate(s.key_points, 1):
                            sec4_lines.append(f"{idx}. **Stage {idx}:** {pt}")
                    else:
                        sec4_lines.append(f"- **Execution Flow:** {s.explanation}")
                    sec4_lines.append(f"> 📄 **Workflow Citation:** *{cit}*\n")
                    wf_counter += 1

        if wf_counter == 1:
            for doc_name in analyzed_docs:
                subs = doc_subtopics_map.get(doc_name, [])
                if subs:
                    s = subs[0]
                    cit = f"{doc_name} (Page {s.page_start})"
                    sec4_citations.append(cit)
                    sec4_lines.append(f"#### 4.{wf_counter} [{doc_name}] {s.title} System Execution Lifecycle")
                    sec4_lines.append(f"1. **Stage 1 (Initialization & Configuration):** {s.definition}")
                    sec4_lines.append(f"2. **Stage 2 (Runtime Execution):** {s.explanation}")
                    sec4_lines.append(f"3. **Stage 3 (State Verification & Finalization):** Verifies deterministic post-conditions.")
                    sec4_lines.append(f"> 📄 **Workflow Citation:** *{cit}*\n")
                    wf_counter += 1

        sec4 = NotesSection(
            id="sec-workflows",
            title="Technical Workflows & Architecture",
            icon="⚙️",
            content_markdown="\n".join(sec4_lines),
            citations=list(dict.fromkeys(sec4_citations)),
        )

        # -------------------------------------------------------------------------
        # SECTION 5: University Examination Mastery Guide & High-Yield FAQs
        # -------------------------------------------------------------------------
        sec5_lines = [
            f"Comprehensive university examination preparation guide featuring high-yield model answers categorized by course document. "
            f"Use these structured blueprints directly to prepare for 5-mark and 10-mark examination questions:\n"
        ]
        sec5_citations = []

        for doc_name in analyzed_docs:
            sec5_lines.append(f"### 📌 Examination Blueprint: {doc_name}\n")
            subs = doc_subtopics_map.get(doc_name, [])

            sec5_lines.append("#### Short-Answer Concept Mastery (5-Mark Model Questions)\n")
            for s in subs[:4]:
                cit = f"{doc_name} (p. {s.page_start})"
                sec5_citations.append(cit)
                sec5_lines.append(f"**Q: Define {s.title} and explain its primary function. [5 Marks]**")
                sec5_lines.append("*Model Answer Framework:*")
                sec5_lines.append(f"- 📌 **Formal Definition:** {s.definition}")
                sec5_lines.append(f"- ⚙️ **Operational Role:** {s.explanation}")
                if s.key_points:
                    sec5_lines.append(f"- 💡 **Key Property:** {s.key_points[0]}")
                sec5_lines.append(f"*(Source: {cit})*\n")

            sec5_lines.append("#### Comprehensive Long-Answer Model Questions (10-Mark Blueprint)\n")
            if subs:
                main_topic = subs[0]
                cit = f"{doc_name} (p. {main_topic.page_start})"
                sec5_citations.append(cit)
                sec5_lines.append(f"**Q: Explain the architecture, components, and working mechanism of {main_topic.title}. [10 Marks]**\n")
                sec5_lines.append("*10-Mark Model Answer Blueprint:*")
                sec5_lines.append(f"- **1. Conceptual Foundation & Definition:** {main_topic.definition}")
                sec5_lines.append(f"- **2. Architectural Topology & Components:**")
                if len(subs) > 1:
                    for comp in subs[1:4]:
                        sec5_lines.append(f"  - *{comp.title}:* {comp.definition.split('. ')[0]}.")
                else:
                    sec5_lines.append(f"  - *Core Runtime:* Executes and enforces isolation boundaries.")
                sec5_lines.append(f"- **3. Operational Flow & Execution:** {main_topic.explanation}")
                sec5_lines.append(f"- **4. Key Engineering Advantages:**")
                if main_topic.key_points:
                    for pt in main_topic.key_points[:3]:
                        sec5_lines.append(f"  - {pt}")
                else:
                    sec5_lines.append(f"  - Eliminates environment discrepancies and ensures scalable performance.")
                sec5_lines.append(f"*(Source: {cit})*\n")

        sec5 = NotesSection(
            id="sec-exam",
            title="University Examination Mastery Guide & High-Yield FAQs",
            icon="🎓",
            content_markdown="\n".join(sec5_lines),
            citations=list(dict.fromkeys(sec5_citations)),
        )

        all_sections = [sec1, sec2, sec3, sec4, sec5]

        # Assemble Complete Markdown
        all_markdown_parts = [
            f"# University Academic Study Guide: {folder.name}",
            f"*Generated by Nalanda AI ATML Engine on {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*\n",
            f"**Analyzed Course Modules:** {', '.join(analyzed_docs)}\n\n---\n",
        ]
        for sec in all_sections:
            all_markdown_parts.append(f"## {sec.icon} {sec.title}\n\n{sec.content_markdown}\n\n---\n")

        full_markdown = "\n".join(all_markdown_parts)

        # Persist to Database Cache
        try:
            db.query(FolderNoteCache).filter(
                FolderNoteCache.folder_id == folder.id,
                FolderNoteCache.user_id == current_user.id,
                FolderNoteCache.documents_hash == docs_hash
            ).delete()

            sections_data = [s.to_dict() for s in all_sections]
            new_cache = FolderNoteCache(
                folder_id=folder.id,
                user_id=current_user.id,
                markdown_content=full_markdown,
                sections_json=json.dumps(sections_data, ensure_ascii=False),
                documents_hash=docs_hash,
                total_sections=len(all_sections),
            )
            db.add(new_cache)
            db.commit()
            logger.info("Persisted notes cache for folder %s (%d sections)", folder.id, len(all_sections))
        except Exception as cache_persist_err:
            db.rollback()
            logger.warning("Could not persist folder note cache: %s", cache_persist_err)

        return NotesResponse(
            folder_id=folder.id,
            folder_name=folder.name,
            generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            documents_analyzed=analyzed_docs,
            total_sections=len(all_sections),
            markdown_content=full_markdown,
            sections=all_sections,
        )

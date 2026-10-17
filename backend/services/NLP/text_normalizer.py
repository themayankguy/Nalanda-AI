"""Structure-aware OCR text normalizer and student-friendly Markdown formatter for Nalanda AI.

Handles:
- Mid-sentence line breaks introduced by PDF extraction / column margins.
- Hyphenation across lines (e.g. 'infor-\nmation' -> 'information').
- Removal of repeated headers, footers, and slide banners (e.g. 'UNIT 5 BDA', 'Page X of Y').
- Preservation of structural elements: Markdown headers, bullet/numbered lists, Markdown tables,
  fenced code blocks, and blockquotes.
- Question-specific structured formatting (Definitions, Types/Categories, Procedures, Comparisons).
"""
import re
from typing import List, Dict, Any, Optional, Tuple, Set


# Common slide and document artifact patterns
BANNER_PATTERNS = [
    r"^(?:UNIT|MODULE|CHAPTER|PART)\s+\d+[^\n]*\n*",
    r"^(?:Big Data Analytics|Course Notes|Lecture \d+|Unit \d+)[^\n]*\n*",
    r"^Page\s+\d+\s+of\s+\d+\s*",
]

# Patterns for lines that indicate new structural items (should NOT be merged with previous line)
STRUCTURAL_LINE_RE = re.compile(
    r"^(?:"
    r"#{1,6}\s+"                          # Markdown header (#, ##, ###)
    r"|(?:\d+[\.\)]|\d+(?:\.\d+)+)\s+"    # Numbered step or section (1., 1), 3.1)
    r"|[-•*+]\s+"                         # Bullet list item
    r"|\|"                                # Table row start
    r"|```"                               # Code fence start/end
    r"|>+\s+"                             # Blockquote
    r"|(?:Example|Note|Definition|Key Point|Main Purpose|Applications?|Summary):"
    r"|Step\s+\d+"
    r")",
    re.IGNORECASE
)


def normalize_ocr_text(text: str) -> str:
    """Normalize raw OCR/PDF text into clean, continuous, structure-preserving Markdown.

    Unwraps artificial line breaks while strictly protecting headers, bullets, tables, and code.
    """
    if not text:
        return ""

    # 1. Normalize line endings and whitespace
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 2. Fix hyphenation split across lines (e.g. 'visual-\nization' -> 'visualization')
    text = re.sub(r"(\b[a-zA-Z]{2,})-\s*\n\s*([a-zA-Z]{2,}\b)", r"\1\2", text)

    # 3. Strip standalone slide banner headers
    for pat in BANNER_PATTERNS:
        text = re.sub(pat, "", text, flags=re.IGNORECASE | re.MULTILINE)

    # 4. Remove stray non-printable / control characters (keep standard punctuation and math symbols)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # 5. Clean excessive repeated periods or dots (e.g. '..........' in tables of contents)
    text = re.sub(r"\.{4,}", "...", text)

    # 6. Structure-aware line unwrapping
    lines = text.split("\n")
    unwrapped: List[str] = []
    in_code_block = False
    in_table = False

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("```"):
            in_code_block = not in_code_block
            unwrapped.append(line)
            continue

        if in_code_block:
            unwrapped.append(line)
            continue

        if stripped.startswith("|"):
            in_table = True
            unwrapped.append(stripped)
            continue
        elif in_table:
            in_table = False

        if not stripped:
            # Preserve blank line boundaries between paragraphs
            if unwrapped and unwrapped[-1] != "":
                unwrapped.append("")
            continue

        # Check if this line starts a new structural element
        is_structural = bool(STRUCTURAL_LINE_RE.match(stripped))

        if unwrapped and not is_structural and unwrapped[-1] != "":
            last_line = unwrapped[-1]
            last_stripped = last_line.strip()

            # Conditions where we unwrap and join with the previous line:
            # - Previous line does not end with sentence-ending punctuation (., ?, !, :, ;)
            # - Or current line starts with a lowercase letter
            # - And previous line is not a standalone Markdown header or table
            prev_is_heading = bool(re.match(r"^#{1,6}\s+", last_stripped))
            prev_is_table = last_stripped.startswith("|")
            prev_ends_sentence = last_stripped.endswith((".", "?", "!", ":", ";", "---"))

            if not prev_is_heading and not prev_is_table and (not prev_ends_sentence or (stripped and stripped[0].islower())):
                unwrapped[-1] = last_line + " " + stripped
                continue

        unwrapped.append(stripped)

    # 7. Normalize multiple consecutive blank lines to at most two
    cleaned = "\n".join(unwrapped)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def detect_query_presentation_type(query: str) -> str:
    """Classify user query into appropriate academic response structure."""
    q = (query or "").lower().strip()

    if re.search(r"\b(compare|comparison|difference|differences|versus|\bvs\b)\b", q):
        return "comparison"
    if re.search(r"\b(types?|categories|kinds?|classifications?|list all|all types|every method|techniques?|approaches?|components?)\b", q):
        return "types"
    if re.search(r"\b(steps?|how to|process|procedure|algorithm|phases?|stages?|lifecycle|workflow)\b", q):
        return "procedure"
    if re.search(r"\b(what is|what are|define|definition of|meaning of)\b", q) and not re.search(r"\b(types|steps|how to|compare|difference)\b", q):
        return "definition"
    if re.search(r"\b(command|code|syntax|script|dockerfile|yaml|json|query)\b", q):
        return "code"

    return "broad"


def format_definition_response(
    concept: str,
    definition: str,
    explanation: Optional[str] = None,
    citations: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Format a crisp, university-style definition answer."""
    lines = [f"### Definition: {concept.strip()}", ""]
    lines.append(f"**{concept.strip()}** {definition.strip()}")
    lines.append("")

    if explanation and len(explanation.strip()) > 30 and explanation.strip() != definition.strip():
        lines.append("### Explanation & Key Concepts")
        lines.append(explanation.strip())
        lines.append("")

    return "\n".join(lines).strip()


def format_types_response(
    parent_concept: str,
    items: List[Dict[str, str]],
    intro: Optional[str] = None,
    is_exhaustive: bool = True,
) -> str:
    """Format an enumerative response of all types, categories, or methods."""
    lines = [f"### Types of {parent_concept.strip()}", ""]

    if intro:
        lines.append(intro.strip())
        lines.append("")
    else:
        lines.append(f"Based on the course materials, the following types of **{parent_concept.strip()}** are covered:")
        lines.append("")

    for idx, item in enumerate(items, 1):
        name = item.get("name", "").strip()
        desc = item.get("description", "").strip()
        source = item.get("source", "").strip()

        if source:
            if "page" in source.lower():
                source_tag = f" *({source})*"
            else:
                source_tag = f" *(Page {source})*"
        else:
            source_tag = ""
        if desc:
            lines.append(f"{idx}. **{name}**{source_tag} — {desc}")
        else:
            lines.append(f"{idx}. **{name}**{source_tag}")

    lines.append("")

    if not is_exhaustive:
        lines.append("> **Note on Source Material:** The above categories represent the complete set documented in your uploaded notes. Additional external classifications are not detailed in the available files.")
        lines.append("")

    return "\n".join(lines).strip()


def format_comparison_table(
    concept_a: str,
    concept_b: str,
    dimensions: List[Dict[str, str]],
    summary: Optional[str] = None,
) -> str:
    """Format a clean Markdown comparison table with standardized criteria."""
    lines = [f"### Comparison: {concept_a.strip()} vs. {concept_b.strip()}", ""]
    lines.append(f"| Dimension / Feature | {concept_a.strip()} | {concept_b.strip()} |")
    lines.append("| :--- | :--- | :--- |")

    for dim in dimensions:
        criteria = dim.get("criteria", "").strip()
        val_a = dim.get("val_a", "").strip().replace("|", "\\|")
        val_b = dim.get("val_b", "").strip().replace("|", "\\|")
        lines.append(f"| **{criteria}** | {val_a} | {val_b} |")

    lines.append("")
    if summary:
        lines.append(f"**Key Takeaway:** {summary.strip()}")
        lines.append("")

    return "\n".join(lines).strip()


def deduplicate_sentences_across_chunks(chunks: List[str]) -> List[str]:
    """Remove verbatim or near-verbatim duplicate sentences across multiple chunks."""
    seen_sent_signatures = set()
    cleaned_chunks = []

    for chunk in chunks:
        sentences = re.split(r"(?<=[.!?])\s+", chunk.strip())
        unique_sents = []
        for s in sentences:
            s_clean = s.strip()
            if not s_clean:
                continue
            # Generate normalized signature (lowercase alphanumeric words)
            sig = " ".join(re.findall(r"\b\w{3,}\b", s_clean.lower()))
            if not sig or sig in seen_sent_signatures:
                continue
            seen_sent_signatures.add(sig)
            unique_sents.append(s_clean)

        if unique_sents:
            cleaned_chunks.append(" ".join(unique_sents))

    return cleaned_chunks

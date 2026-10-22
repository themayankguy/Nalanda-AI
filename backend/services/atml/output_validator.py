"""Output Validator and Source-Grounding Guard for ATML Answer Refinement.

Verifies factual consistency, detects hallucinations, sanitizes OCR noise
and conversational preambles, and preserves technical/mathematical syntax.
"""
import re
from dataclasses import dataclass, field
from typing import List, Set, Tuple, Optional
from .config import ATMLConfig, default_config


@dataclass
class ValidationResult:
    """Detailed validation outcome for a generated answer."""
    is_valid: bool
    confidence_score: float
    issues: List[str] = field(default_factory=list)
    sanitized_answer: str = ""
    grounding_overlap: float = 0.0
    unsupported_entities: List[str] = field(default_factory=list)


# Conversational preambles and meta-talk to eliminate
PREAMBLE_PATTERNS = [
    r"^(?:certainly|sure|of course)[!.,]?\s*",
    r"^here (?:is|are) (?:the|an|your) (?:answer|steps|summary|information):?\s*",
    r"^(?:based on|according to) the (?:provided|retrieved|given) (?:context|documents?|text|information),?\s*",
    r"^(?:in summary|to summarize|in conclusion),?\s*",
    r"^as an ai (?:model|assistant),?\s*",
    r"^the answer is:?\s*",
    r"^(?:[^\n-]*-\s*)?(?:the\s+relevant\s+sentence\s+(?:in\s+the\s+passage\s+)?is:?|the\s+passage\s+states\s+that:?|according\s+to\s+(?:section\s+)?\d+(?:\.\d+)*:?)\s*",
    r"^the\s+relevant\s+sentence\s+(?:in\s+the\s+passage\s+)?is:?\s*",
]

TRAILING_PATTERNS = [
    r"\s*(?:hope this helps|let me know if you (?:have|need) (?:any|more) (?:questions|help))\.?\s*$",
    r"\s*(?:please let me know if you need further clarification)\.?\s*$",
]

# Common English stop words for entity extraction
STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "with", "by", "of", "from", "as", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "can",
    "could", "should", "would", "may", "might", "must", "it", "its",
    "this", "that", "these", "those", "such", "than", "then", "there",
    "their", "they", "we", "you", "which", "who", "whom", "whose", "what",
    "when", "where", "why", "how", "all", "each", "every", "both", "either",
    "neither", "more", "most", "other", "some", "such", "no", "nor", "not",
    "only", "own", "same", "so", "than", "too", "very", "also", "into"
}


def sanitize_text(text: str) -> str:
    """Clean text by removing conversational filler, PDF pointer numbers, and OCR noise while preserving math/code."""
    if not text:
        return ""

    cleaned = text.strip()

    # 1. Remove introductory preambles and meta-sentences (case-insensitive)
    for pat in PREAMBLE_PATTERNS:
        cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE).strip()

    # 2. Remove trailing boilerplate
    for pat in TRAILING_PATTERNS:
        cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE).strip()

    # 3. Strip PDF decimal section pointer numbers (e.g. 7.2, 7.1.1, 6.7)
    cleaned = re.sub(r"(?:^|\n)\s*#{0,6}\s*\d+(?:\.\d+)+\s*", "\n", cleaned)
    cleaned = re.sub(r"(?<=[:\-\—])\s*\d+(?:\.\d+)+\s+(?=[A-Z])", " ", cleaned)
    cleaned = re.sub(r"\b(?:section|sec\.)\s+\d+(?:\.\d+)+\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.strip()

    # 4. Strip standalone course slide banner headers (e.g. 'UNIT 5 BDA' or 'Unit 5 - ...')
    cleaned = re.sub(r"^(?:UNIT\s+\d+[^\n]*\n+)", "", cleaned, flags=re.IGNORECASE).strip()

    # 5. Clean OCR line-break hyphenation (e.g. 'infor- mation' -> 'information')
    cleaned = re.sub(r"(\b[a-zA-Z]{2,})-\s+([a-zA-Z]{2,}\b)", r"\1\2", cleaned)

    # 6. Clean rogue OCR artifacts: lone repeated symbols that are not equations
    cleaned = re.sub(r"(?<![a-zA-Z0-9_\)\]])\^+(?![a-zA-Z0-9_\(\[])", "", cleaned)
    cleaned = re.sub(r"\.{4,}", "...", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)

    # 7. Strip raw HTML image and layout tags
    cleaned = re.sub(r"""(?:<div[^>]*>\s*)?<img[^>]*src=["\'](?:imgs/)?img_in_image_box_[^"\']+["\'][^>]*>(?:\s*</div>)?""", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"""<div[^>]*style=["\'][^"\']*text-align:\s*center[^"\']*["\'][^>]*>[^<]*</div>""", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"""<div[^>]*>|</div>""", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"""<img[^>]*>""", "", cleaned, flags=re.IGNORECASE)

    # 8. Strip lines that consist solely of a lone punctuation symbol (e.g. '.' or '_')
    cleaned_lines = [l for l in cleaned.splitlines() if not re.match(r"^\s*[\.\,\;\:\-\_\*\•]\s*$", l)]
    cleaned = "\n".join(cleaned_lines)

    # 9. Normalize newline spacing
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    return cleaned.strip()



def extract_content_tokens(text: str) -> Set[str]:
    """Extract significant content tokens (alphanumeric, numbers, technical terms)."""
    if not text:
        return set()
    # Find all word/numeric tokens with length >= 2
    tokens = re.findall(r"\b[a-zA-Z0-9_\-\+\*\/\.]{2,}\b", text.lower())
    # Exclude pure stopwords
    return {t for t in tokens if t not in STOP_WORDS and not t.isdigit()}


def extract_named_or_technical_entities(text: str) -> List[str]:
    """Identify capitalized technical terms, acronyms, and compound technical expressions."""
    if not text:
        return []
    # Acronyms (e.g. HDFS, RDD, API, CPU, RAM) or TitleCase multi-word phrases
    entities = re.findall(r"\b[A-Z]{2,}(?:-[A-Z0-9]+)?\b|\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", text)
    return list(dict.fromkeys(entities))


class OutputValidator:
    """Validates answer fidelity against retrieved evidence passages."""

    def __init__(self, config: Optional[ATMLConfig] = None):
        self.config = config or default_config

    def validate(
        self,
        query: str,
        generated_answer: str,
        retrieved_contexts: List[str],
        retrieval_confidence: float = 1.0,
    ) -> ValidationResult:
        """Verify generated answer for source grounding, hallucination, and formatting."""
        issues: List[str] = []
        sanitized = sanitize_text(generated_answer)

        if not sanitized:
            return ValidationResult(
                is_valid=False,
                confidence_score=0.0,
                issues=["Generated answer is empty or completely stripped."],
                sanitized_answer="The retrieved documents do not provide sufficient information to answer this question.",
                grounding_overlap=0.0,
            )

        # Merge all retrieved contexts into a single reference text
        combined_evidence = " ".join(retrieved_contexts).strip()
        if not combined_evidence:
            return ValidationResult(
                is_valid=False,
                confidence_score=0.0,
                issues=["Retrieved evidence is empty."],
                sanitized_answer="The retrieved documents do not provide sufficient information to answer this question.",
                grounding_overlap=0.0,
            )

        # Check query-evidence alignment (detect irrelevant retrieval)
        query_tokens = extract_content_tokens(query)
        evidence_tokens = extract_content_tokens(combined_evidence)
        query_overlap = (
            len(query_tokens.intersection(evidence_tokens)) / len(query_tokens)
            if query_tokens else 1.0
        )

        if retrieval_confidence < 0.30 or query_overlap < 0.15:
            issues.append(f"Insufficient or irrelevant evidence for query (query-evidence overlap: {query_overlap:.2f}, retrieval confidence: {retrieval_confidence:.2f}).")
            return ValidationResult(
                is_valid=False,
                confidence_score=round(retrieval_confidence, 3),
                issues=issues,
                sanitized_answer="The retrieved documents do not provide sufficient information to answer this question.",
                grounding_overlap=round(query_overlap, 3),
            )

        answer_tokens = extract_content_tokens(sanitized)
        if not answer_tokens:
            overlap = 1.0
        else:
            shared = answer_tokens.intersection(evidence_tokens)
            overlap = len(shared) / len(answer_tokens)

        # Check for ungrounded technical entities or acronyms
        answer_entities = extract_named_or_technical_entities(sanitized)
        evidence_lower = combined_evidence.lower()
        unsupported: List[str] = []
        for ent in answer_entities:
            if ent.lower() not in evidence_lower:
                unsupported.append(ent)

        # Entity hallucination ratio
        unsupported_ratio = (len(unsupported) / len(answer_entities)) if answer_entities else 0.0

        # Assess grounding issues
        if overlap < self.config.min_faithfulness_overlap:
            issues.append(f"Low content grounding overlap with source evidence ({overlap:.2f} < {self.config.min_faithfulness_overlap:.2f}).")

        if unsupported_ratio > self.config.max_unsupported_entity_ratio and len(unsupported) >= 2:
            issues.append(f"High unsupported entity ratio ({unsupported_ratio:.2f}) with terms: {unsupported[:3]}.")

        # Repetition loop detection
        sentences = [s.strip() for s in re.split(r"[.!?\n]+", sanitized) if len(s.strip()) > 10]
        if len(sentences) >= 3 and len(set(sentences)) < len(sentences) * 0.6:
            issues.append("Repetitive phrasing loop detected in generation.")

        # Compute composite confidence score
        # Balances evidence overlap, retrieval confidence, and entity grounding
        grounding_score = max(0.0, min(1.0, (overlap * 0.6) + ((1.0 - unsupported_ratio) * 0.4)))
        composite_confidence = round(grounding_score * min(1.0, retrieval_confidence + 0.2), 3)

        is_valid = (
            len(issues) == 0
            and composite_confidence >= self.config.min_confidence_score
        )

        return ValidationResult(
            is_valid=is_valid,
            confidence_score=composite_confidence,
            issues=issues,
            sanitized_answer=sanitized,
            grounding_overlap=round(overlap, 3),
            unsupported_entities=unsupported,
        )

"""Reusable Academic Content Normalization and Section Merging Engine for Nalanda AI.

Provides deterministic, domain-agnostic text processing:
1. Normalizes OCR scanning artifacts, broken line-wraps, and redundant punctuation.
2. Removes multi-page running headers and footers based on structural repetition evidence.
3. Classifies structural line types (headings, bullets, numbered lists, prose, tables, code).
4. Detects heading similarity and merges redundant concept sections (e.g. 'Containerization',
   'What is a Container', 'Explanation of a Container').
5. Guards against over-merging distinct technical qualifiers (e.g. 'Docker Engine' vs 'Docker Container').
6. Formats bullet fragments into standardized Markdown hierarchies with verified provenance.
"""
from collections import Counter
import re
from typing import Any, Dict, List, Optional, Set, Tuple


class ContentNormalizer:
    """Domain-agnostic text normalization and section deduplication engine."""

    # Common OCR typo corrections for standard Latin and technical characters
    OCR_COMMON_REPLACEMENTS = {
        r"\bafetr\b": "after",
        r"\bplatofrm\b": "platform",
        r"\bexecutoi\s*n\b": "execution",
        r"\bmultpi\s*le\b": "multiple",
        r"\bEfefct\b": "Effect",
        r"\bStlil\b": "Still",
        r"\batatched\b": "attached",
        r"\btechnolgy\b": "technology",
        r"\bdepedenc(?:y|ies)\b": "dependencies",
        r"\barchitechture\b": "architecture",
        r"\benvrionment\b": "environment",
    }

    # Interrogative and introductory meta prefixes to strip when comparing headings
    HEADING_PREFIXES = (
        r"^(?:what\s+(?:is|are|was|were)(?:\s+(?:an?|the))?|"
        r"define(?:\s+(?:an?|the))?|"
        r"definition\s+of(?:\s+(?:an?|the))?|"
        r"explanation\s+of(?:\s+(?:an?|the))?|"
        r"explain(?:\s+(?:an?|the))?|"
        r"concept\s+of(?:\s+(?:an?|the))?|"
        r"overview\s+of(?:\s+(?:an?|the))?|"
        r"introduction\s+to(?:\s+(?:an?|the))?|"
        r"meaning\s+of(?:\s+(?:an?|the))?|"
        r"understanding(?:\s+(?:an?|the))?|"
        r"role\s+of(?:\s+(?:an?|the))?|"
        r"need\s+for(?:\s+(?:an?|the))?|"
        r"solution\s+provided\s+by(?:\s+(?:an?|the))?|"
        r"working\s+of(?:\s+(?:an?|the))?|"
        r"components\s+of(?:\s+(?:an?|the))?|"
        r"characteristics\s+of(?:\s+(?:an?|the))?|"
        r"features\s+of(?:\s+(?:an?|the))?|"
        r"advantages\s+of(?:\s+(?:an?|the))?|"
        r"disadvantages\s+of(?:\s+(?:an?|the))?|"
        r"types\s+of(?:\s+(?:an?|the))?|"
        r"notes\s+on(?:\s+(?:an?|the))?)\s+"
    )

    # Stemming / root-reduction rules for academic nominalizations
    STEM_SUFFIXES = [
        ("ization", ""), ("isation", ""), ("izations", ""), ("isations", ""),
        ("ing", ""), ("ed", ""), ("es", ""), ("s", ""),
        ("tion", "t"), ("tions", "t"), ("ment", ""), ("ments", ""),
        ("ive", ""), ("able", ""), ("ible", ""),
    ]

    @classmethod
    def clean_ocr_artifacts(cls, text: str) -> str:
        """Removes repeated OCR artifacts, fixes hyphenated linebreaks, and normalizes punctuation."""
        if not text:
            return ""

        # Normalize unicode whitespace and smart quotes/dashes
        t = text.replace("\u200b", "").replace("\ufeff", "").replace("\u00a0", " ")
        t = t.replace("\u2018", "'").replace("\u2019", "'")
        t = t.replace("\u201c", '"').replace("\u201d", '"')
        t = t.replace("\u2013", "-").replace("\u2014", " - ")

        # Fix hyphenated words split across line breaks (e.g. infor-\nmation -> information)
        t = re.sub(r"(\b[A-Za-z]{2,})-\s*\n\s*([a-z]{2,}\b)", r"\1\2", t)

        # Fix intra-word repeated hyphens (e.g. pro--vides -> provides)
        t = re.sub(r"(\b[A-Za-z]{2,})--+([a-z]{2,}\b)", r"\1\2", t)

        # Apply common OCR typo corrections
        for pat, rep in cls.OCR_COMMON_REPLACEMENTS.items():
            t = re.sub(pat, rep, t, flags=re.IGNORECASE)

        # Strip raw HTML image or layout tags produced by OCR engines
        t = re.sub(r"""(?:<div[^>]*>\s*)?<img[^>]*src=["\'](?:imgs/)?img_in_image_box_[^"\']+["\'][^>]*>(?:\s*</div>)?""", "", t, flags=re.IGNORECASE)
        t = re.sub(r"""<div[^>]*style=["\'][^"\']*text-align:\s*center[^"\']*["\'][^>]*>[^<]*</div>""", "", t, flags=re.IGNORECASE)
        t = re.sub(r"""<div[^>]*>\s*</div>""", "", t, flags=re.IGNORECASE)

        # Collapse excessive repeated punctuation artifacts (preserving markdown tables and hr)
        t = re.sub(r"\^{2,}", "", t)
        t = re.sub(r"(?<!\.)\.{4,}(?!\.)", "...", t)
        t = re.sub(r":{2,}", ":", t)
        t = re.sub(r";{2,}", ";", t)

        # Strip lone punctuation lines
        lines = []
        for line in t.splitlines():
            ls = line.strip()
            if ls and re.match(r"^[\.\,\;\:\_\~\^]+$", ls):
                continue
            lines.append(line)

        return "\n".join(lines)

    @classmethod
    def remove_running_headers_and_footers(
        cls,
        pages: List[Dict[str, Any]],
        min_page_threshold: int = 3,
        ratio_threshold: float = 0.40,
    ) -> List[Dict[str, Any]]:
        """Identifies and strips running slide headers and footers that repeat across pages."""
        if len(pages) < min_page_threshold:
            return pages

        top_candidates: List[str] = []
        bottom_candidates: List[str] = []

        for p in pages:
            raw_t = p.get("text", "")
            lines = [l.strip() for l in raw_t.splitlines() if l.strip()]
            if not lines:
                continue

            # First 2 non-empty lines
            for l in lines[:2]:
                if len(l) <= 70 and not re.match(r"^[-•*+]\s+", l):
                    top_candidates.append(l.lower())

            # Last 2 non-empty lines
            for l in lines[-2:]:
                if len(l) <= 70 and not re.match(r"^[-•*+]\s+", l):
                    bottom_candidates.append(l.lower())

        total_pages = len(pages)
        min_occurrences = max(2, int(total_pages * ratio_threshold))

        top_counts = Counter(top_candidates)
        bottom_counts = Counter(bottom_candidates)

        running_top = {l for l, c in top_counts.items() if c >= min_occurrences}
        running_bottom = {l for l, c in bottom_counts.items() if c >= min_occurrences}

        # Filter out false positives (e.g. substantive headings that appear in body)
        running_headers = running_top.union(running_bottom)

        cleaned_pages = []
        for p in pages:
            p_copy = dict(p)
            raw_t = p.get("text", "")
            lines = raw_t.splitlines()
            kept_lines = []

            for idx, line in enumerate(lines):
                ls = line.strip()
                ls_lower = ls.lower()

                # Standalone page/slide counter lines (e.g. 'Page 3', 'Slide 14 of 40', '12')
                if re.match(r"^(?:page|slide)?\s*\d+\s*(?:of\s*\d+)?$", ls_lower):
                    continue

                # Top slot repetition
                if idx <= 2 and ls_lower in running_headers:
                    continue

                # Bottom slot repetition
                if idx >= len(lines) - 3 and ls_lower in running_headers:
                    continue

                kept_lines.append(line)

            p_copy["text"] = "\n".join(kept_lines)
            cleaned_pages.append(p_copy)

        return cleaned_pages

    @classmethod
    def stem_academic_word(cls, word: str) -> str:
        """Heuristic academic stemmer to match nominalizations (e.g. containerization -> container)."""
        w = word.lower().strip(".,;:?!'\"()")
        for suffix, replacement in cls.STEM_SUFFIXES:
            if w.endswith(suffix) and len(w) - len(suffix) >= 4:
                return w[: len(w) - len(suffix)] + replacement
        return w

    @classmethod
    def extract_heading_tokens(cls, heading: str) -> Tuple[Set[str], Set[str], str]:
        """Extracts content tokens, stemmed roots, and core topic phrase from a heading."""
        clean = (heading or "").strip()
        # Remove markdown heading markers
        clean = re.sub(r"^#{1,6}\s*", "", clean).strip()
        # Remove leading numbers/letters (e.g. "1. ", "3.2 ", "A. ")
        clean = re.sub(r"^(?:\d+(?:\.\d+)*|[A-Za-z])[\.\)]\s*", "", clean).strip()

        # Strip interrogative and structural prefixes
        core = re.sub(cls.HEADING_PREFIXES, "", clean, flags=re.IGNORECASE).strip()
        # Strip trailing meta labels
        core = re.sub(r"\s+(?:concept|definition|explanation|overview|notes?)$", "", core, flags=re.IGNORECASE).strip()
        core = re.sub(r"[:\-\–\—\.]+$", "", core).strip()

        # Tokenize
        raw_tokens = {w.lower() for w in re.findall(r"\b[A-Za-z0-9]+\b", core)}
        stop_words = {"the", "a", "an", "and", "of", "in", "on", "to", "for", "with", "from", "into", "that", "this", "each", "both", "main"}
        meaningful_tokens = {t for t in raw_tokens if t not in stop_words and (len(t) >= 2 or t.isdigit())}

        # Compute stemmed roots
        stemmed_roots = {cls.stem_academic_word(t) for t in meaningful_tokens}

        return meaningful_tokens, stemmed_roots, core or clean

    @classmethod
    def should_merge_headings(
        cls,
        heading_a: str,
        heading_b: str,
        page_distance: int = 1,
    ) -> Tuple[bool, str]:
        """Determines if two headings refer to the same concept and should be merged.
        
        Examples that MUST merge:
          - 'Containerization' and 'What is a Container' -> True, canonical: 'Containerization'
          - 'Explanation of a Container' and 'Container' -> True, canonical: 'Container'
          - 'HDFS Architecture' and 'Architecture of HDFS' -> True, canonical: 'HDFS Architecture'
          
        Examples that MUST NOT merge (distinct concepts):
          - 'Docker Engine' vs 'Docker Container' -> False (distinct qualifiers: 'engine' vs 'container')
          - 'Cell Membrane' vs 'Cell Division' -> False (distinct qualifiers: 'membrane' vs 'division')
          - 'Fiscal Policy' vs 'Monetary Policy' -> False (distinct qualifiers: 'fiscal' vs 'monetary')
          - 'Subtopic 1' vs 'Subtopic 2' -> False (distinct numeric qualifiers: 1 vs 2)
        """
        # Protect distinct numbered or ordinal sections (e.g. Unit 1 vs Unit 2, Phase I vs Phase II)
        nums_a = set(re.findall(r"\b(?:\d+|i|ii|iii|iv|v|vi|vii|viii|ix|x|first|second|third|fourth|fifth)\b", heading_a.lower()))
        nums_b = set(re.findall(r"\b(?:\d+|i|ii|iii|iv|v|vi|vii|viii|ix|x|first|second|third|fourth|fifth)\b", heading_b.lower()))
        if nums_a and nums_b and nums_a != nums_b:
            return False, heading_a

        tokens_a, stems_a, core_a = cls.extract_heading_tokens(heading_a)
        tokens_b, stems_b, core_b = cls.extract_heading_tokens(heading_b)

        if not stems_a or not stems_b:
            return False, heading_a

        # 1. Exact core string identity
        if core_a.lower() == core_b.lower():
            # Prefer the cleaner title (e.g. 'Containerization' over 'What is Containerization')
            canonical = core_a if len(core_a) <= len(core_b) else core_b
            return True, canonical.title()

        # 2. Check for exact stem set equality (e.g. 'HDFS Architecture' vs 'Architecture of HDFS')
        if stems_a == stems_b:
            canonical = core_a if len(core_a) <= len(core_b) else core_b
            return True, canonical.title()

        # 3. Check for single-concept nominalization variants:
        # e.g. stems_a = {'container'}, stems_b = {'container'}
        common_stems = stems_a.intersection(stems_b)
        diff_a = stems_a - common_stems
        diff_b = stems_b - common_stems

        # Modifiers that are merely rhetorical or elaborative (not distinct concepts)
        ELABORATIVE_MODIFIERS = {
            "overview", "introduct", "concept", "definit", "explanat",
            "mean", "work", "role", "need", "solut", "provid", "use",
            "basic", "fundament", "principl", "detail", "approach"
        }

        diff_a_substantive = {d for d in diff_a if d not in ELABORATIVE_MODIFIERS}
        diff_b_substantive = {d for d in diff_b if d not in ELABORATIVE_MODIFIERS}

        # If after filtering elaborative modifiers, both share the primary topic and have no substantive differences:
        if common_stems and not diff_a_substantive and not diff_b_substantive and page_distance <= 2:
            # Select canonical title: prefer concise noun form without question words
            canonical = core_a if not re.match(r"^(?:what|how|why|explain)\b", core_a, re.I) else core_b
            return True, canonical.title()

        # If one is a single core word and the other is just that word plus elaborative prefix:
        if len(stems_a) == 1 and common_stems == stems_a and not diff_b_substantive and page_distance <= 2:
            return True, core_a.title()
        if len(stems_b) == 1 and common_stems == stems_b and not diff_a_substantive and page_distance <= 2:
            return True, core_b.title()

        return False, heading_a

    @classmethod
    def format_fragment_to_bullet(
        cls,
        fragment: str,
        context_concept: Optional[str] = None,
    ) -> str:
        """Formats a raw fragment into clean Markdown with bold anchor and proper grammar.
        
        Retains original wording for ambiguous/incomplete fragments rather than hallucinating predicates.
        """
        raw = fragment.strip().lstrip("-•*+ ").strip()
        if not raw:
            return ""

        # Pattern: <Term>[:–—\-]\s*(.+)
        kv_match = re.match(r"^(\*{0,2})([A-Z0-9][\w\s\-\/\(\)]{1,45})\1\s*[:–—\-]\s*(.+)$", raw)
        if kv_match:
            term = kv_match.group(2).strip().rstrip(":–—- ")
            body = kv_match.group(3).strip()

            if not body:
                return f"- **{term}**"

            # Check if body already has a linking verb or complete sentence structure
            has_verb = bool(re.search(r"\b(is|are|was|were|refers to|serves as|acts as|consists of|involves|functions by|provides|enables)\b", body, re.I))

            if has_verb:
                clean_body = body[0].lower() + body[1:] if len(body) > 1 and not body[:2].isupper() else body
                # If body already starts with 'is/are/refers':
                if re.match(r"^(?:is|are|refers to|serves as|acts as|consists of)\b", clean_body, re.I):
                    formatted = f"- **{term}:** It {clean_body}"
                else:
                    formatted = f"- **{term}:** {body[0].upper() + body[1:]}"
            else:
                # Add natural linking connector supported by grammatical form
                b_low = body.lower()
                if b_low.startswith(("a ", "an ", "the ")):
                    formatted = f"- **{term}:** Refers to {body[0].lower() + body[1:]}"
                elif re.match(r"^[a-z]+ing\b", b_low):
                    formatted = f"- **{term}:** Involves {body[0].lower() + body[1:]}"
                else:
                    # Incomplete fragment: safely retain original text with term anchor
                    formatted = f"- **{term}:** {body[0].upper() + body[1:]}"

            if not formatted.endswith((".", "!", "?")):
                formatted += "."
            return formatted

        # Standard non-KV bullet
        clean_text = raw[0].upper() + raw[1:] if len(raw) > 1 else raw.upper()
        if not clean_text.endswith((".", "!", "?", ":", ";")):
            clean_text += "."
        return f"- {clean_text}"

    @classmethod
    def is_assessment_or_quiz_content(cls, title: str, text: str) -> bool:
        """Determines if a section, slide, or chunk represents an assessment, quiz, or exercise question list."""
        t_clean = (title or "").strip().lower()
        combined = f"{t_clean}\n{text or ''}".strip().lower()
        if not combined:
            return False

        # Explicit quiz / assessment header indicator
        has_quiz_header = bool(
            re.search(
                r"\b(quiz\s+time|pop\s+quiz|quizzes|quiz\b|review\s+questions?|self[- ]assessment|"
                r"exercises?|practice\s+problems?|discussion\s+questions?|test\s+questions?|"
                r"check\s+your\s+understanding|knowledge\s+check)\b",
                t_clean,
                re.IGNORECASE,
            )
            or re.search(
                r"(?:^|\n)\s*#{0,6}\s*(?:quiz\s+time|pop\s+quiz|quizzes|quiz\b|review\s+questions?|self[- ]assessment|"
                r"exercises?|practice\s+problems?|discussion\s+questions?|test\s+questions?|"
                r"check\s+your\s+understanding|knowledge\s+check)\b",
                combined,
                re.IGNORECASE,
            )
        )

        lines = [l.strip() for l in (text or "").splitlines() if l.strip()]
        # Check density of question sentences (ending with ? or starting with question words)
        question_lines = [
            l for l in lines
            if l.endswith("?")
            or re.match(r"^\s*(?:\d+[\.\)]|[-•*+])\s*(?:what|why|how|which|explain|describe|define|is|are|can|name|where|when)\b", l, re.IGNORECASE)
        ]

        if has_quiz_header:
            return True

        if len(lines) >= 2 and len(question_lines) >= 2 and (len(question_lines) / len(lines)) >= 0.45:
            return True

        return False

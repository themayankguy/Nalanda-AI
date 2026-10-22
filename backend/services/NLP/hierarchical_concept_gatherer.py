"""Hierarchical Subtopic and Concept Gathering Service for Nalanda AI.

Collects comprehensive sets of sibling concepts, categories, types, methods, and
components under a parent concept across all document chunks in a folder.
Ensures the chatbot does not return only the first 1-2 examples from introductory paragraphs.
"""
import logging
import re
from typing import List, Dict, Any, Optional, Tuple, Set

logger = logging.getLogger("nalanda.hierarchical_gatherer")

# Generic non-concept headings and chapter titles to ignore
IGNORE_SUBHEADINGS = {
    "introduction", "overview", "conclusion", "summary", "references", "case study",
    "case studies", "example", "examples", "prerequisites", "notes", "questions",
    "step 1", "step 2", "step 3", "step 4", "step 5",
    "table of contents", "index", "syllabus", "course outline", "agenda", "appendix",
}


class HierarchicalConceptGatherer:
    """Gathers complete sibling concepts and subheadings under parent topics across all folder chunks."""

    @staticmethod
    def is_hierarchical_query(query: str) -> bool:
        """Detect whether the user is asking for an enumeration of categories, types, or methods."""
        q = (query or "").lower().strip()
        patterns = [
            r"\b(?:all\s+)?types\s+of\b",
            r"\b(?:all\s+)?kinds\s+of\b",
            r"\b(?:all\s+)?categories\s+of\b",
            r"\b(?:all\s+)?forms\s+of\b",
            r"\b(?:all\s+)?classes\s+of\b",
            r"\blist\s+(?:all\s+)?(?:the\s+)?(?:types|categories|charts|methods|techniques|components|modes|tools|classes|stages)\b",
            r"\bexplain\s+(?:all\s+)?(?:the\s+)?(?:types|categories|charts|methods|techniques|components|modes|classes|stages)\b",
            r"\bwhat\s+are\s+(?:all\s+)?(?:the\s+)?(?:types|categories|charts|plots|techniques|methods|components|modes|classes|stages)\b",
            r"\bwhat\s+(?:\w+)\s+(?:types|categories|forms|classes)\b",
            r"\bwhich\s+(?:types|categories|methods|techniques|components|classes)\b",
        ]
        return any(re.search(pat, q) for pat in patterns)

    @classmethod
    def extract_parent_target(cls, query: str) -> str:
        """Extract the core target topic from the enumerative query."""
        q = (query or "").lower().strip()
        clean = re.sub(
            r"^(?:what are (?:all )?(?:the )?|list (?:all )?(?:the )?|explain (?:all )?(?:the )?|show (?:all )?(?:the )?|tell me (?:about )?(?:all )?(?:the )?)",
            "",
            q
        )
        clean = re.sub(
            r"^(?:types|kinds|categories|forms|classes|methods|techniques|approaches|components|modes|tools|features)\s+(?:of\s+)?",
            "",
            clean
        )
        clean = clean.replace("?", "").replace(".", "").strip()
        # Remove trailing words like 'mentioned in notes', 'in my notes', 'available'
        clean = re.sub(r"\s+(?:mentioned\s+in\s+.*|in\s+my\s+notes|in\s+the\s+pdf|available|covered)\s*$", "", clean)
        return clean.strip() or q

    @classmethod
    def gather_sibling_concepts(
        cls,
        parent_topic: str,
        all_chunks: List[Dict[str, Any]],
        max_items: int = 15,
    ) -> List[Dict[str, Any]]:
        """Scan all chunks in the folder and collect all sibling subtopics under the parent concept."""
        parent_clean = parent_topic.lower().strip()
        # Extract meaningful root keywords (>= 3 chars)
        keywords = set(re.findall(r"\b\w{3,}\b", parent_clean))
        # Add singular/plural forms to keywords
        for k in list(keywords):
            if k.endswith("s"):
                keywords.add(k[:-1])
            else:
                keywords.add(k + "s")

        items_by_name: Dict[str, Dict[str, Any]] = {}

        def is_valid_sibling_candidate(name: str) -> bool:
            nl = name.lower().strip()
            if not nl or nl in IGNORE_SUBHEADINGS:
                return False
            # Filter generic procedural and rhetorical meta-phrases
            if any(p in nl for p in [
                "steps for", "steps in", "need for", "importance of", "challenges",
                "meaning of", "principles of", "techniques for", "overview of",
                "select an appropriate", "prepare and process"
            ]):
                return False
            # Filter explicit comparison headers
            if " vs " in f" {nl} " or " versus " in f" {nl} ":
                return False
            # Valid candidates should be short descriptive entities (typically <= 6 words)
            if len(nl.split()) > 6:
                return False
            return True

        # Pass 1: Scan section titles of all chunks
        for chunk in all_chunks:
            sec_title = (chunk.get("section_title") or "").strip()
            page_num = chunk.get("page_number")
            source_file = chunk.get("source_filename") or ""
            text = chunk.get("text", "")

            # Check for numbered or dedicated sub-headings (e.g. '3.1 Line Chart', '3.2 Histogram')
            subhead_match = re.match(r"^(?:\d+[\.\)]|\d+(?:\.\d+)+)\s+([A-Za-z][\w\s\-/:,\?]{2,60})$", sec_title)
            if subhead_match:
                candidate_name = subhead_match.group(1).strip()
                if is_valid_sibling_candidate(candidate_name):
                    # Check relevance to parent topic: section title or text mentions keywords,
                    # or parent topic is general in folder
                    sec_lower = sec_title.lower()
                    text_lower = text.lower()
                    is_relevant = any(k in sec_lower or k in text_lower for k in keywords) or not keywords
                    if is_relevant:
                        first_para = text.split("\n\n")[0].strip() if "\n\n" in text else text.strip()
                        first_sent = re.split(r"(?<=[.!?])\s+", first_para)[0].strip() if first_para else ""
                        desc = re.sub(r"^\d+(?:\.\d+)*\s*", "", first_sent).strip()
                        desc = re.sub(r"^(?:" + re.escape(candidate_name) + r")\s*[-–—:]*\s*", "", desc, flags=re.IGNORECASE).strip()
                        if len(desc) < 15:
                            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", first_para) if len(s.strip()) > 15]
                            if len(sentences) > 1:
                                desc = re.sub(r"^\d+(?:\.\d+)*\s*", "", sentences[1]).strip()
                                desc = re.sub(r"^(?:" + re.escape(candidate_name) + r")\s*[-–—:]*\s*", "", desc, flags=re.IGNORECASE).strip()

                    clean_key = candidate_name.lower()
                    if clean_key not in items_by_name:
                        items_by_name[clean_key] = {
                            "name": candidate_name,
                            "description": desc,
                            "source": f"{source_file} (Page {page_num})" if source_file and page_num else str(page_num or ""),
                            "page_number": page_num,
                            "source_filename": source_file,
                        }

        # Pass 2: Scan structured lists and bullet items within chunks discussing the parent topic
        for chunk in all_chunks:
            text = chunk.get("text", "")
            page_num = chunk.get("page_number")
            source_file = chunk.get("source_filename") or ""

            # Check if this chunk is discussing the parent topic
            text_lower = text.lower()
            if any(k in text_lower for k in keywords) or (parent_clean and parent_clean in text_lower):
                # Look for bold lead-ins: **Name** - Description or Name – Description
                list_matches = re.findall(
                    r"(?:^|\n)\s*(?:[-•*+]|\d+[\.\)])\s*(?:\*\*(.+?)\*\*|([A-Z][\w\s\-]{2,30}))\s*[-–—:]\s*([^\n]{15,250})",
                    text
                )
                for m in list_matches:
                    raw_name = (m[0] or m[1]).strip().splitlines()[0].strip()
                    name = re.sub(r"^(?:\d+[\.\)]|\d+(?:\.\d+)+)\s*", "", raw_name).strip()
                    name = re.sub(r"\s+Example$", "", name, flags=re.IGNORECASE).strip()
                    desc = m[2].strip().splitlines()[0].strip()
                    desc = re.sub(r"^Example:\s*", "", desc, flags=re.IGNORECASE).strip()
                    if name and desc and is_valid_sibling_candidate(name) and len(name.split()) <= 4:
                        name_lower = name.lower().rstrip("s")  # normalize singular/plural
                        if not any(k in items_by_name for k in [name_lower, name_lower + "s", name.lower()]):
                            items_by_name[name_lower] = {
                                "name": name.title(),
                                "description": desc,
                                "source": f"{source_file} (Page {page_num})" if source_file and page_num else str(page_num or ""),
                                "page_number": page_num,
                                "source_filename": source_file,
                            }

                # Look for subheadings inside text: e.g. "3.1 Line Chart\nA line chart represents..."
                header_matches = re.findall(
                    r"(?:^|\n)(?:###?\s+|\d+(?:\.\d+)+\s+)([A-Z][A-Za-z\s\-]{2,35})\n+([^\n]{20,250})",
                    text
                )
                for h_name, h_desc in header_matches:
                    h_clean = h_name.strip()
                    if is_valid_sibling_candidate(h_clean) and len(h_clean.split()) <= 4:
                        first_sent = re.split(r"(?<=[.!?])\s+", h_desc.strip())[0].strip()
                        h_key = h_clean.lower()
                        if h_key not in items_by_name:
                            items_by_name[h_key] = {
                                "name": h_clean,
                                "description": first_sent,
                                "source": f"{source_file} (Page {page_num})" if source_file and page_num else str(page_num or ""),
                                "page_number": page_num,
                                "source_filename": source_file,
                            }

        gathered = list(items_by_name.values())[:max_items]
        return gathered

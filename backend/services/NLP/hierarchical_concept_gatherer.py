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
    "need for visualization", "creating visualization", "select an appropriate visualization technique",
    "create the visualization", "prepare and process the data", "big data visualization challenges",
    "custom visualization solutions", "tools used for creating visualizations", "existing visualization techniques",
    "meaning of visual analysis", "principles of visual analysis", "steps for creating a visualization",
    "advanced graphing techniques", "advanced chart types", "geographic and spatial visualization",
    "techniques for visualizing big data", "hierarchical visualization", "symmetrical distribution",
    "custom visualizations", "data visualization types used in visual analysis",
    "heat maps and geographic maps", "relationships between variables",
}

# Domain mapping for parent concepts to typical category keywords
PARENT_CONCEPT_MAP = {
    "chart": ["chart", "plot", "histogram", "map", "graph", "diagram", "scatterplot", "boxplot", "treemap", "heatmap", "bubble"],
    "charts": ["chart", "plot", "histogram", "map", "graph", "diagram", "scatterplot", "boxplot", "treemap", "heatmap", "bubble"],
    "visualization": ["chart", "plot", "histogram", "visualization", "map", "graph", "diagram", "treemap", "heatmap"],
    "machine learning": ["supervised", "unsupervised", "reinforcement", "semi-supervised"],
    "learning": ["supervised", "unsupervised", "reinforcement", "semi-supervised"],
    "hadoop": ["hdfs", "mapreduce", "yarn", "namenode", "datanode", "resource manager", "node manager"],
    "pig": ["grunt", "local mode", "mapreduce mode", "tez mode", "bag", "tuple", "field"],
    "hive": ["metastore", "driver", "compiler", "executor", "hql", "beeline", "managed table", "external table"],
    "docker": ["dockerfile", "image", "container", "volume", "network", "compose", "registry", "hub"],
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
            r"\blist\s+(?:all\s+)?(?:the\s+)?(?:types|categories|charts|methods|techniques|components|modes|tools)\b",
            r"\bexplain\s+(?:all\s+)?(?:the\s+)?(?:types|categories|charts|methods|techniques|components|modes)\b",
            r"\bwhat\s+are\s+(?:all\s+)?(?:the\s+)?(?:types|categories|charts|plots|techniques|methods|components|modes)\b",
            r"\bwhat\s+(?:chart|plot|visualization|learning)\s+types\b",
            r"\bwhich\s+(?:types|categories|methods|techniques|components)\b",
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
            r"^(?:types|kinds|categories|methods|techniques|approaches|components|modes|tools|features)\s+(?:of\s+)?",
            "",
            clean
        )
        clean = clean.replace("?", "").replace(".", "").strip()
        # Remove trailing words like 'mentioned in notes', 'in big data', 'available'
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
        keywords = set(re.findall(r"\b\w{3,}\b", parent_clean))
        is_chart_query = bool(re.search(r"\b(chart|charts|plot|plots|graphs?)\b", parent_clean))

        # Expand domain-related keywords if known
        expanded_keywords = set(keywords)
        for domain, domain_kws in PARENT_CONCEPT_MAP.items():
            if domain in parent_clean or any(k in domain for k in keywords):
                expanded_keywords.update(domain_kws)

        items_by_name: Dict[str, Dict[str, Any]] = {}

        def is_valid_sibling_candidate(name: str) -> bool:
            nl = name.lower().strip()
            if not nl or nl in IGNORE_SUBHEADINGS:
                return False
            if any(p in nl for p in ["steps for", "steps in", "need for", "importance of", "challenges", "creating visualization", "select an appropriate", "advanced", "meaning of", "techniques for", "hierarchical visualization"]):
                return False
            if "mapreduce" in nl or "vs" in nl.split():
                return False
            if is_chart_query:
                # Must be a recognized chart/plot/map/graph concept (not domain applications like Airlines or IoT)
                chart_terms = ["chart", "plot", "histogram", "treemap", "radar", "heatmap", "dendrogram"]
                if any(k in nl for k in chart_terms):
                    return True
                if re.search(r"\b(heat\s*maps?|geographic\s*maps?|spatial\s*maps?|maps?)\b", nl):
                    return True
                return False
            return True

        # Pass 1: Scan section titles of all chunks
        for chunk in all_chunks:
            sec_title = (chunk.get("section_title") or "").strip()
            page_num = chunk.get("page_number")
            source_file = chunk.get("source_filename") or ""
            text = chunk.get("text", "")

            # Check for numbered or dedicated sub-headings (e.g. '3.1 Line Chart', '3.2 Histogram', 'Heat Map')
            subhead_match = re.match(r"^(?:\d+[\.\)]|\d+(?:\.\d+)+)\s+([A-Za-z][\w\s\-/:,\?]{2,60})$", sec_title)
            if subhead_match:
                candidate_name = subhead_match.group(1).strip()
                if is_valid_sibling_candidate(candidate_name):
                    # Extract definition / explanation from first sentence of chunk
                    first_para = text.split("\n\n")[0].strip() if "\n\n" in text else text.strip()
                    first_sent = re.split(r"(?<=[.!?])\s+", first_para)[0].strip() if first_para else ""
                    # Clean section numbers and repeated candidate name
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

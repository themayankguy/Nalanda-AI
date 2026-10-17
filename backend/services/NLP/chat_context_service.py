"""Multi-turn context resolution service for Nalanda AI chatbot.

Resolves anaphoric references (pronouns such as 'it', 'its', 'this', 'that'),
ordinal follow-ups ('explain the second point'), and elaboration requests
('give an example', 'compare it with X') using bounded conversation history.
"""
import logging
import re
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger("nalanda.chat_context")

PRONOUN_PATTERNS = [
    r"\b(it|its|this|that|these|those|the\s+same)\b",
]

FOLLOW_UP_TRIGGERS = [
    r"^(?:what\s+are\s+)?its\s+(?:advantages|disadvantages|features|benefits|limitations|uses|applications|components)",
    r"^(?:explain|describe|tell\s+me\s+about)\s+the\s+(?:first|second|third|fourth|fifth|1st|2nd|3rd|4th|5th|last)\s+(?:point|step|type|item|concept|method)",
    r"^(?:give|show|provide)\s+(?:an?\s+)?example(?:\s+of\s+(?:it|this|that))?",
    r"^(?:compare\s+(?:it|this)\s+with|how\s+does\s+(?:it|this)\s+differ\s+from)",
    r"^(?:why\s+is\s+(?:it|this|that)\s+used|how\s+does\s+(?:it|this|that)\s+work)",
    r"^(?:make\s+(?:it|the\s+answer)\s+shorter|summarize\s+(?:it|this|that))",
    r"^(?:explain\s+(?:it|that)\s+in\s+simpler\s+words|simplify)",
    r"^continue\b",
]


class ChatContextService:
    """Manages multi-turn conversation context and query resolution."""

    @classmethod
    def extract_key_topic_from_text(cls, text: str) -> Optional[str]:
        """Extract the central technical topic from a previous question or answer."""
        if not text:
            return None

        # Look for explicit Markdown header e.g. '### Definition: Hadoop Distributed File System'
        hdr_match = re.search(r"###\s+(?:Definition:\s+|Types of\s+|Comparison:\s+)?([^\r\n]+)", text)
        if hdr_match:
            candidate = hdr_match.group(1).strip()
            candidate = re.sub(r"[\*\#\_]", "", candidate).strip()
            if candidate and len(candidate.split()) <= 6:
                return candidate

        # Look for bold primary subject at the beginning e.g. '**Hadoop Distributed File System (HDFS)** is...'
        bold_match = re.search(r"\*\*([A-Za-z0-9_\s\-\(\)]+?)\*\*", text[:180])
        if bold_match:
            candidate = bold_match.group(1).strip().splitlines()[0].strip()
            if candidate and len(candidate.split()) <= 6 and not candidate.lower().startswith("note"):
                return candidate

        # Extract uppercase acronyms or TitleCase phrases
        entities = re.findall(r"\b[A-Z]{2,}\b|\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", text[:200])
        if entities:
            # Pick the most prominent entity
            for ent in entities:
                ent_clean = ent.strip().splitlines()[0].strip()
                if ent_clean.lower() not in {"what is", "based on", "the following", "key points", "page", "section"}:
                    return ent_clean

        return None

    @classmethod
    def extract_numbered_point(cls, text: str, ordinal: str) -> Optional[str]:
        """Extract a specific numbered point from a previous response."""
        ordinal_map = {
            "first": 1, "1st": 1,
            "second": 2, "2nd": 2,
            "third": 3, "3rd": 3,
            "fourth": 4, "4th": 4,
            "fifth": 5, "5th": 5,
        }
        target_idx = ordinal_map.get(ordinal.lower())
        if not target_idx or not text:
            return None

        # Look for numbered items: e.g. '1. NameNode', '2. DataNode'
        matches = re.findall(r"(?:^|\n)\s*(\d+)[\.\)]\s*(?:\*\*(.+?)\*\*|([^\n–—:-]+?))(?:\s*[-–—:]\s*([^\n]+))?", text)
        for num_str, bold_name, raw_name, desc in matches:
            if int(num_str) == target_idx:
                point_name = (bold_name or raw_name).strip()
                full_desc = (desc or "").strip()
                if point_name:
                    return f"{point_name}: {full_desc}" if full_desc else point_name

        return None

    @classmethod
    def resolve_followup_query(
        cls,
        current_query: str,
        conversation_history: List[Dict[str, Any]],
    ) -> Tuple[str, Optional[str]]:
        """Resolve pronouns and relative references in the user's query against recent turns.

        Returns:
            Tuple of (augmented_search_query, detected_antecedent_topic)
        """
        q = current_query.strip()
        q_lower = q.lower()

        if not conversation_history:
            return q, None

        # Get the last user message and last assistant message
        last_user_turn = None
        last_assistant_turn = None
        for msg in reversed(conversation_history):
            role = msg.get("role")
            if role == "assistant" and last_assistant_turn is None:
                last_assistant_turn = msg
            elif role == "user" and last_user_turn is None:
                last_user_turn = msg
            if last_user_turn and last_assistant_turn:
                break

        prev_topic = None
        if last_assistant_turn:
            prev_topic = cls.extract_key_topic_from_text(last_assistant_turn.get("content", ""))
        if not prev_topic and last_user_turn:
            prev_topic = cls.extract_key_topic_from_text(last_user_turn.get("content", ""))

        # Check for ordinal point request: e.g. 'Explain the second point'
        ord_match = re.search(r"\b(first|second|third|fourth|fifth|1st|2nd|3rd|4th|5th)\s+(?:point|step|item|concept|method|type)", q_lower)
        if ord_match and last_assistant_turn:
            ordinal = ord_match.group(1)
            point_content = cls.extract_numbered_point(last_assistant_turn.get("content", ""), ordinal)
            if point_content:
                augmented = f"Explain in detail: {point_content}"
                if prev_topic and prev_topic.lower() not in augmented.lower():
                    augmented += f" in {prev_topic}"
                return augmented, prev_topic

        # Check for pronoun replacement or relative follow-up
        if prev_topic:
            # Do NOT augment if current query already contains the antecedent topic
            if prev_topic.lower() in q_lower or any(w in q_lower for w in prev_topic.lower().split() if len(w) >= 4):
                return q, prev_topic

            has_pronoun = bool(re.search(r"\b(its|it|this|that|these|those)\b", q_lower))
            is_followup_trigger = any(re.match(p, q_lower) for p in FOLLOW_UP_TRIGGERS)
            is_subjectless_short = len(q.split()) <= 3 and not re.search(r"\b(what is|what are|types of|kinds of|define|list)\b", q_lower)

            if has_pronoun or is_followup_trigger or is_subjectless_short:
                # Replace pronouns or append antecedent topic
                if re.search(r"\bits\b", q_lower):
                    augmented = re.sub(r"\bits\b", f"{prev_topic}'s", q, flags=re.IGNORECASE)
                elif re.search(r"\b(it|this|that)\b", q_lower):
                    augmented = re.sub(r"\b(it|this|that)\b", prev_topic, q, flags=re.IGNORECASE)
                else:
                    augmented = f"{q} for {prev_topic}"

                logger.info("Resolved follow-up query '%s' -> '%s' (antecedent: %s)", q, augmented, prev_topic)
                return augmented, prev_topic

        return q, prev_topic

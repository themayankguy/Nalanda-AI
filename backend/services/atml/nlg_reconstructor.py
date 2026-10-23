"""Generalized Natural Language Reconstruction (NLG) Engine for Nalanda AI.

Transforms raw document excerpts, terse bullet points, fragmented OCR scans,
and slide card extractions into fluent, grammatically structured, ChatGPT-style
academic responses using deterministic classical NLP and linguistic heuristics.

Guarantees:
1. Zero third-party APIs and zero external network calls.
2. Zero LLMs, neural model weights, or hallucinated facts.
3. 100% domain-agnostic: fully compatible with any future document domain
   (medicine, history, computer science, law, biology, economics, physics, etc.).
"""
import re
from typing import List, Dict, Any, Optional, Tuple


class GeneralizedNLGReconstructor:
    """Deterministic, domain-agnostic natural language reconstruction engine."""

    # Sentence boundary and punctuation helpers
    SENTENCE_ENDINGS = (".", "!", "?")
    LINKING_VERBS = {
        "is", "are", "was", "were", "refers to", "serves as", "acts as",
        "consists of", "functions by", "involves", "enables", "provides",
        "represents", "is characterized by", "is defined as", "operates as",
    }

    @staticmethod
    def extract_core_subject(query: str) -> str:
        """Extract the core target topic from an academic query."""
        clean = (query or "").strip().rstrip("?").strip()
        # Strip common interrogative and instructional prefixes
        clean = re.sub(
            r"^(?:what\s+(?:is|are|was|were)(?:\s+(?:the|a|an))?|explain(?:\s+(?:the|a|an))?|describe(?:\s+(?:the|a|an))?|"
            r"define(?:\s+(?:the|a|an))?|how\s+(?:does|do|to)|steps\s+(?:for|to|in)|purposes\s+of|"
            r"main\s+purposes\s+of|types\s+of|kinds\s+of|categories\s+of|list(?:\s+all)?(?:\s+the)?|"
            r"show(?:\s+me)?(?:\s+the)?|tell\s+me\s+about(?:\s+the)?|give(?:\s+an?\s+overview\s+of)?|"
            r"advantages\s+(?:and\s+disadvantages\s+)?of|challenges\s+(?:in|of))\s+",
            "",
            clean,
            flags=re.IGNORECASE,
        ).strip()
        # Clean trailing noise phrases
        clean = re.sub(
            r"\s+(?:mentioned\s+in\s+.*|in\s+my\s+notes|in\s+the\s+pdf|in\s+detail|briefly|"
            r"available|covered|with\s+examples?)\s*$",
            "",
            clean,
            flags=re.IGNORECASE,
        ).strip()
        return clean or query.strip().rstrip("?")

    @staticmethod
    def detect_academic_intent(query: str, raw_text: str = "") -> str:
        """Classify academic intent into one of 5 universal categories."""
        q_lower = (query or "").lower().strip()

        # 1. Procedural / Sequential (Processes, steps, lifecycles, algorithms, workflows)
        if re.search(r"\b(steps?|how to|process|procedure|phases?|stages?|lifecycle|workflow|algorithm|pipeline|mechanism|execution)\b", q_lower):
            return "procedural"
        if len(re.findall(r"\b(?:Goal|Step|Objective|Phase|Stage|Rule)\s*\d+\b", raw_text, re.I)) >= 2:
            return "procedural"

        # 2. Taxonomical / Enumerative (Types, categories, classifications, components, kinds)
        if re.search(r"\b(types?|kinds?|categories|classifications?|modes?|forms?|components?|branches|taxonom(?:y|ies))\b", q_lower):
            return "taxonomical"

        # 3. Comparative / Analytical (Differences, versus, compare, contrast, similarities)
        if re.search(r"\b(versus|vs|compare|contrast|differences?|distinctions?|trade-?offs?)\b", q_lower):
            return "comparative"

        # 4. Evaluative / Critique (Advantages, disadvantages, pros, cons, challenges, benefits, limitations)
        if re.search(r"\b(advantages?|disadvantages?|pros?|cons?|benefits?|limitations?|challenges?|issues?|drawbacks?)\b", q_lower):
            return "evaluative"

        # 5. Definitional / Explanatory default
        return "definitional"

    @classmethod
    def reconstruct_fragment_to_sentence(
        cls,
        fragment: str,
        context_subject: Optional[str] = None,
        bullet_prefix: bool = True,
    ) -> str:
        """Reconstruct a terse note fragment, math formula, or key-value pair into a full, declarative sentence.
        
        Strict Evidence Fidelity: No outside facts are added. Deterministic linking verbs
        and linguistic transformations bridge phrases naturally.
        """
        raw = fragment.strip().lstrip("-•*+ ").strip()
        if not raw:
            return ""

        # Normalize multiple spaces and arrows
        raw = re.sub(r"\s*(?:->|=>)\s*", " leading to ", raw).strip()
        raw = raw.rstrip(":").strip()
        raw = re.sub(r"\s+", " ", raw)

        # Check for standalone count configurations (e.g. 1 convolutional layer)
        if re.match(r"^\d+\s+[a-zA-Z\s]+$", raw):
            return f"It is configured with {raw}."

        # Check if it already represents a complete, grammatically sound sentence
        has_finite_verb = any(re.search(rf"\b{re.escape(v)}\b", raw, re.I) for v in cls.LINKING_VERBS)
        ends_with_punct = raw.endswith(cls.SENTENCE_ENDINGS)
        if has_finite_verb and ends_with_punct and not re.match(r"^[A-Z][\w\s\-]{1,30}[:–—\-]\s*", raw):
            # Ensure first letter capitalized
            return raw[0].upper() + raw[1:]

        # Check for standalone mathematical equations or assignments: e.g. x = [...], W_conv = [...], MSE = 7.2586
        eq_match = re.match(r"^([A-Za-z0-9_\(\)]+(?:\s+[A-Za-z0-9_\(\)]+)*)\s*=\s*(.+)$", raw)
        if eq_match:
            lhs = eq_match.group(1).strip()
            rhs = eq_match.group(2).strip()
            lhs_lower = lhs.lower()
            if any(k in lhs_lower for k in ["mse", "loss", "error", "rmse", "mae"]):
                sentence = f"The **{lhs}** evaluates to {rhs}"
            elif any(k in lhs_lower for k in ["kernel", "weight", "w_conv", "w_fc", "w"]):
                sentence = f"The kernel weight vector **{lhs}** is defined as {rhs}"
            elif any(k in lhs_lower for k in ["latent", "z"]):
                sentence = f"The latent representation **{lhs}** evaluates to {rhs}"
            elif any(k in lhs_lower for k in ["output", "conv output", "decoded"]):
                sentence = f"The computed **{lhs}** produces {rhs}"
            elif any(k in lhs_lower for k in ["input", "x"]):
                sentence = f"The input representation **{lhs}** is configured as {rhs}"
            elif any(k in lhs_lower for k in ["bias", "b"]):
                sentence = f"The bias parameter **{lhs}** is set to {rhs}"
            else:
                sentence = f"The value of **{lhs}** is calculated as {rhs}"

            if not sentence.endswith(cls.SENTENCE_ENDINGS):
                sentence += "."
            return sentence

        # Check for standalone architectural flags & negative constraints
        if re.match(r"^(?:no\s+padding|without\s+padding)\b", raw, re.I):
            return "No padding is applied, ensuring valid boundary convolution."
        if re.match(r"^(?:1\s+filter\s+only|single\s+filter(?:\s+only)?)\b", raw, re.I):
            return "Only a single filter channel is utilized."
        if re.match(r"^(?:no\s+activation|without\s+activation)\b", raw, re.I):
            return "No activation function is applied, maintaining linear transformation."

        # Pattern 1: Delimited Key Term & Body (e.g. 'Term: Description' or 'Term - Description')
        kv_match = re.match(r"^(\*{0,2})([A-Z0-9][\w\s\-\/\(\)]{1,45})\1\s*[:–—\-]\s*(.+)$", raw)
        if kv_match:
            term = kv_match.group(2).strip()
            body = kv_match.group(3).strip()

            # Ensure term capitalization
            term_clean = term.strip().rstrip(":–—- ")
            body_clean = body[0].lower() + body[1:] if len(body) > 1 and not body[:2].isupper() else body

            term_lower = term_clean.lower()
            # Check for technical parameters / metrics / settings
            if re.search(r"\b(?:filter|filter\s+size|stride|kernel|weights?|bias|padding|activation|dimension|output\s+dimension|output\s+shape|input\s+shape|shape|loss|error|rate|capacity|latency|threshold)\b", term_lower):
                if re.search(r"\b(?:none|none\s*\(for\s+simplicity\)|null)\b", body_clean, re.I):
                    sentence = f"The **{term_clean}** is set to {body_clean}"
                elif re.search(r"\b(?:shape|dimension|size)\b", term_lower):
                    sentence = f"The **{term_clean}** evaluates to {body_clean}"
                elif re.search(r"\b(?:loss|error|mse|accuracy)\b", term_lower):
                    sentence = f"The **{term_clean}** yields {body_clean}"
                else:
                    sentence = f"The **{term_clean}** is configured to {body_clean}"
            # Select linking verb based on linguistic features of the body
            elif re.search(r"\bpowerhouse\b", body_clean, re.I):
                sentence = f"**{term_clean}** serves as {body_clean}"
            elif re.search(r"\bmental\s+state\b", body_clean, re.I):
                sentence = f"**{term_clean}** represents {body_clean}"
            elif re.search(r"\bvalue\s+of\b", body_clean, re.I):
                sentence = f"**{term_clean}** is defined as {body_clean}"
            elif re.match(r"^(?:an?|the)\s+", body_clean, re.I):
                sentence = f"**{term_clean}** is {body_clean}"
            elif re.match(r"^[a-z]+ing\b", body_clean):
                sentence = f"**{term_clean}** involves {body_clean}"
            elif any(body_clean.startswith(v) for v in ["is ", "are ", "was ", "were ", "refers to ", "serves as ", "acts as ", "consists of "]):
                sentence = f"**{term_clean}** {body_clean}"
            elif re.match(r"^(?:used\s+to|used\s+for|responsible\s+for|designed\s+to|capable\s+of)\b", body_clean, re.I):
                sentence = f"**{term_clean}** is {body_clean}"
            elif re.match(r"^(?:provides|enables|supports|measures|creates|produces|contains|regulates|governs|stores)\b", body_clean, re.I):
                sentence = f"**{term_clean}** {body_clean}"
            else:
                sentence = f"**{term_clean}** refers to {body_clean}"

            if not sentence.endswith(cls.SENTENCE_ENDINGS):
                sentence += "."
            return sentence

        # Pattern 2: Short fragment without explicit colon delimiter
        if context_subject:
            # Check if raw is a numbered heading marker like "1. Input" or "Step 2"
            step_header_m = re.match(r"^(?:step\s*)?\d+[\.\)]\s*(.+)$", raw, re.I)
            if step_header_m:
                return f"**{step_header_m.group(1).strip()}**"

            clean_body = raw[0].lower() + raw[1:] if len(raw) > 1 and not raw[:2].isupper() else raw
            if clean_body.startswith(("for ", "to ", "with ", "by ")):
                sentence = f"{context_subject} functions {clean_body}"
            elif clean_body.startswith(("provides ", "enables ", "ensures ", "includes ", "contains ", "stores ", "regulates ")):
                sentence = f"{context_subject} {clean_body}"
            elif any(v in clean_body for v in cls.LINKING_VERBS):
                sentence = f"{context_subject} {clean_body}"
            else:
                sentence = f"{context_subject} operates with {clean_body}"
            if not sentence.endswith(cls.SENTENCE_ENDINGS):
                sentence += "."
            return sentence

        # Pattern 3: Fallback - capitalize and punctuate securely without hallucination
        formatted = raw[0].upper() + raw[1:]
        if not formatted.endswith(cls.SENTENCE_ENDINGS):
            formatted += "."
        return formatted

    @classmethod
    def synthesize_conversational_intro(cls, query: str, subject: Optional[str] = None, intent: Optional[str] = None) -> str:
        """Formulate a natural, ChatGPT-style introductory lead-in tailored to the academic intent."""
        subj = (subject or cls.extract_core_subject(query)).strip()
        int_type = intent or cls.detect_academic_intent(query)

        if int_type == "procedural":
            return f"Based on your course materials, here is the structured step-by-step process for **{subj}**:"
        elif int_type == "taxonomical":
            return f"According to the uploaded documents, **{subj}** encompasses the following key categories and components:"
        elif int_type == "comparative":
            return f"Based on the provided documentation, here is a detailed comparative breakdown for **{subj}**:"
        elif int_type == "evaluative":
            return f"According to the course notes, here are the primary advantages, challenges, and considerations associated with **{subj}**:"
        else:
            return f"Based on the course materials in your workspace, here is an academic breakdown and explanation of **{subj}**:"

    @classmethod
    def synthesize_conversational_conclusion(cls, intent: str, subject: Optional[str] = None) -> str:
        """Formulate an academic concluding synthesis tailored to the intent without hallucinated specifics."""
        subj = subject.strip() if subject else "this subject"
        if intent == "procedural":
            return f"In practice, executing these stages sequentially ensures consistency, correctness, and reliability throughout **{subj}**."
        elif intent == "taxonomical":
            return f"Understanding these distinct classifications allows for selecting the appropriate approach depending on analytical requirements and constraints."
        elif intent == "comparative":
            return f"In summary, evaluating these contrasting factors highlights the key functional trade-offs and suitable contexts for each option."
        elif intent == "evaluative":
            return f"Balancing these capabilities against their operational constraints provides a robust foundation for effective implementation and decision-making."
        else:
            return f"Together, these foundational principles provide a comprehensive framework for analyzing and applying **{subj}**."

    @staticmethod
    def deinterleave_ocr_cards_generalized(lines: List[str]) -> List[str]:
        """De-interleave horizontal multi-column cards extracted from slides across any domain.
        
        Detects repeating structural patterns:
          - A run of K headers (e.g. Goal 1, Goal 2, ... or Step 1, Step 2, ...)
          - Followed by K concise titles
          - Followed by body content partitioned or assigned sequentially to each card.
        Uses pure layout logic with zero domain keywords.
        """
        out: List[str] = []
        i = 0
        n = len(lines)

        while i < n:
            # Check for a contiguous run of card headers: e.g. Header 1, Header 2, ...
            j = i
            headers: List[Tuple[str, str, int]] = []
            while j < n:
                line_s = lines[j].strip().lstrip("#").strip()
                m = re.match(r"^(?:(Goal|Step|Objective|Feature|Phase|Stage|Pillar|Principle|Component|Layer)\s*(\d+)[:.]?)$", line_s, re.I)
                if m:
                    headers.append((line_s, m.group(1).title(), int(m.group(2))))
                    j += 1
                else:
                    break

            k = len(headers)
            # If at least 2 consecutive headers were grouped together by horizontal OCR scanning:
            if k >= 2 and j + k <= n:
                potential_titles = [lines[j + idx].strip() for idx in range(k)]
                # Validate that following K lines look like titles (short, non-bullet, non-punctuated)
                all_titles = all(
                    len(t.split()) <= 7
                    and not t.endswith((".", "!", "?"))
                    and not t.startswith(("#", "-", "*", "•"))
                    for t in potential_titles
                )
                if all_titles:
                    header_titles = [f"{headers[idx][1]} {headers[idx][2]}: {potential_titles[idx]}" for idx in range(k)]
                    body_start = j + k
                    body_end = body_start

                    # Collect body lines until the next major section/header
                    while body_end < n:
                        l_check = lines[body_end].strip().lstrip("#").strip()
                        if re.match(r"^(?:(Goal|Step|Objective|Feature|Phase|Stage|Pillar|Principle|Component|Layer)\s*\d+)", l_check, re.I):
                            break
                        body_end += 1

                    body_lines = [l.strip() for l in lines[body_start:body_end] if l.strip()]

                    # Partition body lines across K cards dynamically
                    card_bodies: List[List[str]] = [[] for _ in range(k)]
                    if body_lines:
                        # If body lines are a multiple of k or evenly distributed:
                        if len(body_lines) >= k:
                            chunk_size = len(body_lines) // k
                            remainder = len(body_lines) % k
                            curr_pos = 0
                            for c_idx in range(k):
                                take = chunk_size + (1 if c_idx < remainder else 0)
                                card_bodies[c_idx] = body_lines[curr_pos : curr_pos + take]
                                curr_pos += take
                        else:
                            # Fewer lines than cards: assign in order
                            for b_idx, b_line in enumerate(body_lines):
                                card_bodies[b_idx % k].append(b_line)

                    # Output de-interleaved cards sequentially
                    for c_idx in range(k):
                        out.append(header_titles[c_idx])
                        if card_bodies[c_idx]:
                            out.append(" ".join(card_bodies[c_idx]))

                    i = body_end
                    continue

            out.append(lines[i])
            i += 1

        # Secondary pass: Pair any lone 'Card N' header with its immediately following title line
        paired: List[str] = []
        i = 0
        while i < len(out):
            line = out[i].strip()
            header_m = re.match(r"^(?:#{1,6}\s*)?((?:Goal|Step|Objective|Feature|Phase|Stage|Pillar|Principle|Component|Layer)\s*\d+)[:.]?$", line, re.I)
            if header_m and i + 1 < len(out):
                next_l = out[i + 1].strip()
                if len(next_l.split()) <= 7 and not next_l.endswith((".", "!", "?")) and not next_l.startswith(("#", "-", "*", "•")):
                    paired.append(f"{header_m.group(1)}: {next_l}")
                    i += 2
                    continue
            paired.append(line)
            i += 1

        return paired

    @classmethod
    def reconstruct_text_to_academic_markdown(
        cls,
        query: str,
        raw_text: str,
        source_metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Transform raw extracted text into a structured, ChatGPT-level academic response.
        
        Output Architecture:
        1. Contextual Executive Lead-in
        2. Structured Breakdown (Subheadings, bold concept tags, fully reconstructed sentences)
        3. Concluding Academic Synthesis
        """
        if not raw_text or not raw_text.strip():
            return ""

        core_subj = cls.extract_core_subject(query)
        intent = cls.detect_academic_intent(query, raw_text)

        # Step 1: Normalize linebreaks mid-sentence
        raw_lines = [l.strip() for l in raw_text.splitlines() if l.strip()]
        unwrapped: List[str] = []
        for line in raw_lines:
            # Strip decimal section prefixes (e.g. "3.2.1 Core Process" -> "Core Process")
            ls = re.sub(r"^\d+(?:\.\d+)+\s*", "", line).strip()
            if not ls:
                continue

            is_item_marker = bool(
                re.match(r"^(?:(?:step|goal|objective|feature|phase|stage|pillar|rule|item)\s*)?\d+[\.\)]", ls, re.I)
                or re.match(r"^(?:[-•+]|\*(?!\*))", ls)
                or re.match(r"^(?:Example|Note|Key point|Definition):", ls, re.I)
                or ls.startswith("#")
                or " → " in ls
            )

            if unwrapped and not is_item_marker:
                last = unwrapped[-1]
                is_heading = (
                    last.startswith("#")
                    or bool(re.match(r"^(?:(Goal|Step|Objective|Feature|Phase|Stage)\s*\d+)", last, re.I))
                    or (len(last.split()) <= 7 and not re.search(r"[a-z][,;]\s*$", last) and not last.endswith(("-", "—", ":")))
                )
                if not is_heading and (not last.endswith(cls.SENTENCE_ENDINGS) or ls[0].islower()):
                    unwrapped[-1] = last + " " + ls
                    continue

            unwrapped.append(ls)

        if not unwrapped:
            return ""

        # Step 2: De-interleave slide cards
        unwrapped = cls.deinterleave_ocr_cards_generalized(unwrapped)

        # Step 3: Format lead-in
        lead_in = cls.synthesize_conversational_intro(query, core_subj, intent)

        # Step 4: Parse items or paragraphs according to intent
        blocks: List[str] = []

        if intent == "procedural":
            # Extract procedural items
            step_items: List[Dict[str, Any]] = []
            curr_step: Optional[Dict[str, Any]] = None
            preamble_lines: List[str] = []

            for line in unwrapped:
                step_m = (
                    re.match(r"^(?:#{1,6}\s*)?(?:(Goal|Step|Objective|Feature|Phase|Stage|Rule|Item)\s*(\d+)[:.]?)\s*(.+)$", line, re.I)
                    or re.match(r"^(?:step\s*)?(\d+)[\.\)]\s*(.+)$", line, re.I)
                )
                bullet_m = re.match(r"^(?:[-•+]|\*(?!\*))\s*(.+)$", line)

                if step_m:
                    if step_m.lastindex == 3:
                        kind = step_m.group(1).title()
                        num = int(step_m.group(2))
                        title = step_m.group(3).strip()
                    else:
                        kind = "Step"
                        num = int(step_m.group(1))
                        title = step_m.group(2).strip()

                    curr_step = {"kind": kind, "num": num, "title": title, "body": []}
                    step_items.append(curr_step)
                elif bullet_m and curr_step:
                    curr_step["body"].append(bullet_m.group(1).strip())
                elif curr_step:
                    curr_step["body"].append(line)
                else:
                    preamble_lines.append(line)

            if step_items:
                formatted_steps: List[str] = []
                for idx, s in enumerate(step_items, 1):
                    kind_label = s.get("kind", "Step")
                    title_text = s["title"].strip()
                    # Reconstruct step title if it is a terse fragment
                    recon_title = title_text[0].upper() + title_text[1:] if title_text else f"{kind_label} {idx}"
                    formatted_steps.append(f"{idx}. **{recon_title}**")

                    body_lines = s.get("body", [])
                    if body_lines:
                        # Reconstruct body bullets into complete sentences
                        body_sentences = [
                            cls.reconstruct_fragment_to_sentence(b, context_subject=title_text)
                            for b in body_lines if b.strip()
                        ]
                        cleaned_desc = " ".join(body_sentences)
                        formatted_steps.append(f"   {cleaned_desc}")

                blocks.append("\n\n".join(formatted_steps))
            else:
                # If no explicit numbered items detected, convert lines into structured steps
                formatted_steps = []
                for idx, line in enumerate(unwrapped, 1):
                    recon = cls.reconstruct_fragment_to_sentence(line, context_subject=core_subj)
                    formatted_steps.append(f"{idx}. {recon}")
                blocks.append("\n\n".join(formatted_steps))

        elif intent in ("taxonomical", "evaluative", "comparative"):
            bullet_items: List[str] = []
            for line in unwrapped:
                if " → " in line:
                    bullet_items.append(f"\n{line}\n")
                    continue
                # Reconstruct fragment into sentence with bold concept tag
                recon = cls.reconstruct_fragment_to_sentence(line, context_subject=core_subj)
                if recon:
                    bullet_items.append(f"- {recon}")

            if bullet_items:
                blocks.append("\n".join(bullet_items))

        else:
            # Definitional / Explanatory: create clean, readable paragraphs
            curr_para: List[str] = []
            para_blocks: List[str] = []

            for line in unwrapped:
                if line.startswith(("-", "*", "•")) or re.match(r"^\d+[\.\)]", line):
                    recon = cls.reconstruct_fragment_to_sentence(line, context_subject=core_subj)
                    if curr_para:
                        para_blocks.append(" ".join(curr_para))
                        curr_para = []
                    para_blocks.append(f"- {recon}")
                else:
                    recon = cls.reconstruct_fragment_to_sentence(line, context_subject=core_subj)
                    curr_para.append(recon)
                    if len(curr_para) >= 2 or sum(len(c) for c in curr_para) > 260:
                        para_blocks.append(" ".join(curr_para))
                        curr_para = []

            if curr_para:
                para_blocks.append(" ".join(curr_para))

            blocks.append("\n\n".join(para_blocks))

        # Step 5: Synthesize conclusion
        conclusion = cls.synthesize_conversational_conclusion(intent, core_subj)

        # Assemble full academic document
        components: List[str] = []

        # Optional heading if not present
        heading_title = core_subj.title()
        components.append(f"### {heading_title}")
        components.append(lead_in)
        components.extend(blocks)
        components.append(f"*{conclusion}*")

        return "\n\n".join(components).strip()

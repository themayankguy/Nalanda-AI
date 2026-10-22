"""Answer Structurer for ATML Engine.

Formats raw generation into readable paragraphs, bullet points, numbered steps,
sensible headings, and code/math blocks based on query intent and content semantics.
Preserves complete, detailed academic explanations without shortcutting.
"""
import re
from typing import List, Dict, Any, Optional
from .output_validator import sanitize_text


class AnswerStructurer:
    """Structures synthesized answers into human-friendly, readable Markdown formats."""

    @staticmethod
    def detect_formatting_intent(query: str, raw_text: str) -> str:
        """Determine whether the answer is best represented as steps, bullets, or paragraphs."""
        q_lower = query.lower().strip()

        # Check if the generated or source text naturally possesses numbered steps or goals
        if (
            len(re.findall(r"\b(?:Goal|Step|Objective|Feature|Phase|Stage|Rule)\s*\d+\b", raw_text, re.I)) >= 2
            or len(re.findall(r"(?m)^\s*\d+[\.\)]\s+", raw_text)) >= 2
        ):
            return "steps"

        # Procedural / Sequential intent
        if re.search(r"\b(steps?|how to|procedure|algorithm|process|phases?|stages?|lifecycle|workflow)\b", q_lower):
            return "steps"

        # Comparison / List / Properties intent
        if re.search(r"\b(types?|kinds?|categories|characteristics?|features?|purposes?|goals?|needs?|advantages?|disadvantages?|pros?|cons?|reasons?|benefits?|components?|properties?|differences?|versus|vs)\b", q_lower):
            return "bullets"

        # Check if text has multiple bullet points
        if len(re.findall(r"(?:^|\n)\s*[-•*+]\s+", raw_text)) >= 2:
            return "bullets"

        return "paragraphs"

    @staticmethod
    def _format_arrow_flows(content: str) -> str:
        """Collapse sequential pipeline flows connected by arrows (↓, →, =>, ->) into formatted flowlines."""
        lines = [l.strip() for l in content.splitlines()]
        output: List[str] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            # Check if an arrow flow sequence starts here: e.g. line followed by ↓, then another line, then ↓
            if i + 2 < len(lines) and re.match(r"^[↓→⇒]|->|-->$", lines[i+1]):
                flow_steps = [line]
                curr = i + 1
                while curr < len(lines) and re.match(r"^[↓→⇒]|->|-->$", lines[curr]):
                    if curr + 1 < len(lines) and lines[curr+1] and not re.match(r"^[↓→⇒]|->|-->$", lines[curr+1]):
                        flow_steps.append(lines[curr+1])
                        curr += 2
                    else:
                        curr += 1
                if len(flow_steps) >= 2:
                    flow_md = " → ".join(f"**{s}**" for s in flow_steps)
                    output.append(flow_md)
                    i = curr
                    continue
            output.append(line)
            i += 1
        return "\n".join(output)

    @staticmethod
    def _coalesce_bullets_into_explanation(bullets: List[str]) -> str:
        """Merge fragmented bullet lines into clean, cohesive explanatory sentences using deterministic NLG."""
        from .nlg_reconstructor import GeneralizedNLGReconstructor
        filtered = [b.strip() for b in bullets if b.strip() and not re.match(r"^[\W_]+$", b.strip())]
        if not filtered:
            return ""
        reconstructed = [
            GeneralizedNLGReconstructor.reconstruct_fragment_to_sentence(b)
            for b in filtered
        ]
        clean_text = " ".join(reconstructed)
        clean_text = re.sub(r"([.?!])([A-Z])", r"\1 \2", clean_text)
        clean_text = re.sub(r"([,;])([a-zA-Z])", r"\1 \2", clean_text)
        clean_text = re.sub(r"\s+", " ", clean_text).strip()
        return clean_text

    @staticmethod
    def _deinterleave_slide_cards(lines: List[str]) -> List[str]:
        """De-interleave horizontal multi-column OCR scans where cards have headers grouped across columns."""
        from .nlg_reconstructor import GeneralizedNLGReconstructor
        return GeneralizedNLGReconstructor.deinterleave_ocr_cards_generalized(lines)

    def structure(
        self,
        query: str,
        content: str,
        source_metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Format the content according to presentation guidelines, preserving full detail and removing PDF pointers."""
        cleaned = sanitize_text(content)
        if not cleaned:
            return ""

        # Collapse sequential pipeline flows with arrows
        cleaned = self._format_arrow_flows(cleaned)

        # Step 0: Unwrap lines that were split mid-sentence by PDF column margins
        raw_lines = cleaned.splitlines()
        unwrapped_lines: List[str] = []
        for l in raw_lines:
            ls = l.strip()
            if not ls:
                continue

            # Strip decimal section pointers at start of line (e.g. '7.2 Main Purposes' -> 'Main Purposes')
            ls = re.sub(r"^\d+(?:\.\d+)+\s*", "", ls).strip()

            is_new_item = bool(
                re.match(r"^(?:(?:step|goal|objective|feature|phase|stage|rule|item)\s*)?\d+[\.\)]", ls, re.I)
                or re.match(r"^(?:(Goal|Step|Objective|Feature|Phase|Stage)\s*\d+)", ls, re.I)
                or re.match(r"^(?:[-•+]|\*(?!\*))", ls)
                or re.match(r"^(?:Example|Main purpose|Note|Key point|Definition):", ls, re.I)
                or ls.startswith("#")
                or " → " in ls
            )

            if unwrapped_lines and not is_new_item:
                last = unwrapped_lines[-1]
                is_heading = (
                    last.startswith("#")
                    or bool(re.match(r"^(?:(Goal|Step|Objective|Feature|Phase|Stage)\s*\d+)", last, re.I))
                    or (len(last.split()) <= 8 and not re.search(r"[a-z][,;]\s*$", last) and not last.endswith(("-", "—", ":")))
                )
                # If last line does not end with sentence-ending punctuation or this line begins lowercase
                if not is_heading and (not last.endswith((".", ":", "!", "?", ";")) or ls[0].islower()) and not last.startswith("#"):
                    unwrapped_lines[-1] = last + " " + ls
                    continue

            unwrapped_lines.append(ls)

        if not unwrapped_lines:
            return ""

        # De-interleave slide cards if multi-column OCR extracted headers together
        unwrapped_lines = self._deinterleave_slide_cards(unwrapped_lines)

        intent = self.detect_formatting_intent(query, cleaned)
        has_numbered_items = (
            len(re.findall(r"\b(?:Goal|Step|Objective|Feature|Phase|Stage|Rule)\s*\d+\b", cleaned, re.I)) >= 2
            or len(re.findall(r"(?m)^\s*\d+[\.\)]\s+", cleaned)) >= 2
            or intent == "steps"
        )

        # Step 1: If content contains numbered points (e.g. 1. Exploratory Analysis ... 2. Explanatory ...):
        if has_numbered_items:
            items: List[Dict[str, Any]] = []
            curr_item: Optional[Dict[str, Any]] = None
            preamble_lines: List[str] = []
            example_section_lines: List[str] = []
            in_example_section = False

            for line in unwrapped_lines:
                line_clean = line.strip()
                if not line_clean or re.match(r"^[.,;:_\-•]$", line_clean):
                    continue

                step_m = (
                    re.match(r"^(?:#{1,6}\s*)?(?:(Goal|Step|Objective|Feature|Phase|Stage|Rule|Item)\s*(\d+)[:.]?)\s*(.+)$", line_clean, re.I)
                    or re.match(r"^(?:step\s*)?(\d+)[\.\)]\s*(.+)$", line_clean, re.I)
                )
                bullet_m = re.match(r"^(?:[-•+]|\*(?!\*))\s*(.+)$", line_clean)
                example_header_m = re.match(r"^(?:Example|For\s+example)\s*:\s*(.+)$", line_clean, re.I)
                example_inline_m = re.match(r"^(Main\s+purpose|Note|Key\s+point|Example):\s*(.+)$", line_clean, re.I)

                # Check if practical walkthrough section starts
                if example_header_m and len(items) >= 2:
                    in_example_section = True
                    example_section_lines.append(f"**Scenario:** {example_header_m.group(1).strip()}")
                    continue

                if in_example_section:
                    if step_m:
                        item_num = step_m.group(2) if step_m.lastindex == 3 else step_m.group(1)
                        item_text = step_m.group(3) if step_m.lastindex == 3 else step_m.group(2)
                        example_section_lines.append(f"- **Step {item_num}:** {item_text.strip()}")
                    elif bullet_m:
                        example_section_lines.append(f"- {bullet_m.group(1).strip()}")
                    elif line_clean.lower().startswith(("thus", "outcome", "in conclusion")):
                        example_section_lines.append(f"\n*{line_clean}*")
                    else:
                        example_section_lines.append(f"- {line_clean}")
                    continue

                if step_m:
                    if step_m.lastindex == 3:
                        kind = step_m.group(1).title()
                        num = int(step_m.group(2))
                        title = step_m.group(3).strip()
                    else:
                        kind = "Step"
                        num = int(step_m.group(1))
                        title = step_m.group(2).strip()
                    title = title.strip()
                    curr_item = {"kind": kind, "num": num, "title": title, "bullets": []}
                    items.append(curr_item)
                elif bullet_m and curr_item:
                    b_text = bullet_m.group(1).strip()
                    if example_inline_m:
                        lbl, val = example_inline_m.group(1), example_inline_m.group(2).strip()
                        curr_item["bullets"].append(f"*{lbl}:* {val}")
                    else:
                        curr_item["bullets"].append(b_text)
                elif curr_item:
                    curr_item["bullets"].append(line_clean)
                else:
                    preamble_lines.append(line_clean)

            # Check for outline vs detailed duplicate list (e.g. outline list 1, 2, 3 followed by detailed 1, 2, 3)
            # or section header preceding a numbered list (e.g. 8. Applications of HBase followed by 1. Write-Heavy...)
            restart_indices = [i for i, it in enumerate(items) if i > 0 and it["num"] == 1]
            if restart_indices:
                r_idx = restart_indices[0]
                first_group = items[:r_idx]
                second_group = items[r_idx:]
                # Case 1: First item is a document section header (e.g., "8. Applications of HBase" or single header item before list)
                if first_group[0]["num"] > 1 or len(first_group) == 1:
                    header_title = first_group[0]["title"]
                    preamble_lines.insert(0, header_title)
                    for b in first_group[0]["bullets"]:
                        preamble_lines.append(b)
                    items = second_group
                # Case 2: Outline agenda list followed by detailed breakdown (e.g. 1. Client 2. Master Server 3. Region Servers)
                elif all(len(it["bullets"]) == 0 for it in first_group) and any(len(it["bullets"]) > 0 for it in second_group):
                    items = second_group
                elif len(first_group) <= len(second_group):
                    items = second_group

            is_quiz_query = bool(re.search(r"\b(quiz|quizzes|questions?|test|practice|exercise|exercises|sample\s+questions)\b", query, re.I))
            if not is_quiz_query:
                # Strip quiz header lines from preamble (e.g. 'Quiz Time!')
                preamble_lines = [
                    p for p in preamble_lines
                    if not re.search(r"^(?:#{0,6}\s*)?(?:quiz\s+time|pop\s+quiz|review\s+questions?|self[- ]assessment|exercises?)\b", p, re.I)
                ]
                # Check if items are actually quiz questions rather than informative steps
                question_count = sum(
                    1 for it in items
                    if it.get("title", "").strip().endswith("?")
                    or re.match(r"^(?:what|why|how|which|explain|describe|define|is|are|can|name|where|when)\b", it.get("title", ""), re.I)
                )
                if len(items) >= 2 and (question_count / len(items)) >= 0.4:
                    items = []

            blocks: List[str] = []
            if preamble_lines:
                clean_p0 = preamble_lines[0].lstrip("# ").strip()
                if len(preamble_lines) > 1 and len(clean_p0.split()) <= 8:
                    blocks.append(f"### {clean_p0}\n\n" + "\n\n".join(preamble_lines[1:]))
                elif len(preamble_lines) == 1 and len(clean_p0.split()) <= 8:
                    blocks.append(f"### {clean_p0}")
                else:
                    blocks.append("\n\n".join(preamble_lines))

            if items:
                if len(items) == len(set(it["num"] for it in items)):
                    items.sort(key=lambda x: x["num"])
                formatted_steps: List[str] = []
                for idx, it in enumerate(items, 1):
                    item_kind = it.get("kind", "Step")
                    if item_kind.lower() != "step":
                        formatted_steps.append(f"{idx}. **{item_kind} {it['num']}: {it['title']}**")
                    else:
                        formatted_steps.append(f"{idx}. {it['title']}")
                    bullets = it.get("bullets", [])
                    if bullets:
                        desc = self._coalesce_bullets_into_explanation(bullets)
                        if desc:
                            formatted_steps.append(f"   {desc}")
                blocks.append("\n\n".join(formatted_steps))

            if example_section_lines:
                blocks.append("### Practical Example Walkthrough\n\n" + "\n".join(example_section_lines))

            final_output = "\n\n".join(blocks).strip()

        elif intent == "bullets":
            bullet_items: List[str] = []
            intro_lines: List[str] = []
            from .nlg_reconstructor import GeneralizedNLGReconstructor

            for idx, line in enumerate(unwrapped_lines):
                bullet_m = re.match(r"^(?:[-•+]|\*(?!\*))\s*(.+)$", line)
                step_m = re.match(r"^(?:step\s*)?\d+[\.\)]\s*(.+)$", line, re.I)

                raw_item = bullet_m.group(1).strip() if bullet_m else (step_m.group(1).strip() if step_m else None)

                if raw_item is not None:
                    recon = GeneralizedNLGReconstructor.reconstruct_fragment_to_sentence(raw_item)
                    bullet_items.append(f"- {recon}")
                elif " → " in line:
                    bullet_items.append(f"\n{line}\n")
                elif line.endswith("Flow") or line.endswith("Architecture") or line.endswith("Working") or (len(line.split()) <= 4 and line.endswith(":")):
                    bullet_items.append(f"\n**{line.rstrip(':')}:**")
                elif idx == 0 and len(unwrapped_lines) > 1 and not line.startswith(("-", "*", "•")):
                    intro_lines.append(line)
                else:
                    recon = GeneralizedNLGReconstructor.reconstruct_fragment_to_sentence(line.lstrip("-•*+ "))
                    bullet_items.append(f"- {recon}")

            blocks = []
            if intro_lines:
                blocks.append("\n\n".join(intro_lines))
            if bullet_items:
                blocks.append("\n".join(bullet_items))

            final_output = "\n\n".join(blocks).strip()

        else:
            # Paragraphs intent
            curr_para: List[str] = []
            formatted_blocks: List[str] = []
            from .nlg_reconstructor import GeneralizedNLGReconstructor

            for line in unwrapped_lines:
                if line.startswith(("-", "*", "•")) or re.match(r"^\d+[\.\)]", line):
                    if curr_para:
                        formatted_blocks.append(" ".join(curr_para))
                        curr_para = []
                    recon = GeneralizedNLGReconstructor.reconstruct_fragment_to_sentence(line.lstrip("-•*+ "))
                    formatted_blocks.append(f"- {recon}")
                else:
                    recon = GeneralizedNLGReconstructor.reconstruct_fragment_to_sentence(line)
                    curr_para.append(recon)
                    if len(curr_para) >= 2 or sum(len(c) for c in curr_para) > 240:
                        formatted_blocks.append(" ".join(curr_para))
                        curr_para = []

            if curr_para:
                formatted_blocks.append(" ".join(curr_para))

            final_output = "\n\n".join(formatted_blocks).strip()

        # Add sensible heading if not already present and query warrants it
        if not final_output.startswith("#") and (has_numbered_items or len(unwrapped_lines) >= 2):
            from .nlg_reconstructor import GeneralizedNLGReconstructor
            core_subj = GeneralizedNLGReconstructor.extract_core_subject(query)
            if core_subj and len(core_subj.split()) <= 6:
                heading_title = core_subj.title()
                final_output = f"### {heading_title}\n\n{final_output}"
            else:
                clean_q = query.strip().rstrip("?")
                if len(clean_q.split()) <= 8 and not re.match(r"^(?:why|how)\b", clean_q, flags=re.I):
                    heading_title = clean_q.title()
                    final_output = f"### {heading_title}\n\n{final_output}"

        return final_output

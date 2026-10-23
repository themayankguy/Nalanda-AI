"""Core Transformer-based Answer Refinement Engine for ATML.

Employs pretrained Sequence-to-Sequence (Encoder-Decoder) Transformer (FLAN-T5)
with cross-attention to synthesize, rewrite, and structure retrieved passages
into accurate, faithful, human-readable answers.
"""
import logging
import re
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from .config import ATMLConfig, default_config
from .answer_structurer import AnswerStructurer
from .output_validator import OutputValidator, ValidationResult, sanitize_text

logger = logging.getLogger("nalanda.atml.refiner")

# Singleton model and tokenizer holders
_model_instance = None
_tokenizer_instance = None


def get_atml_model_and_tokenizer(
    config: Optional[ATMLConfig] = None,
) -> Tuple[Optional[AutoModelForSeq2SeqLM], Optional[AutoTokenizer]]:
    """Lazily load and cache the Sequence-to-Sequence Transformer model and tokenizer."""
    global _model_instance, _tokenizer_instance
    cfg = config or default_config

    if _model_instance is not None and _tokenizer_instance is not None:
        return _model_instance, _tokenizer_instance

    try:
        logger.info("Initializing ATML Seq2Seq model: %s on device: %s", cfg.model_name, cfg.device)
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_name)

        # Move to designated target device (e.g., Apple Silicon 'mps' or 'cpu')
        device = torch.device(cfg.device)
        model = model.to(device)
        model.eval()

        _tokenizer_instance = tokenizer
        _model_instance = model
        logger.info("ATML Seq2Seq model loaded successfully on %s", cfg.device)
        return _model_instance, _tokenizer_instance

    except Exception as exc:
        logger.error("Failed to load ATML Seq2Seq model '%s': %s", cfg.model_name, exc)
        return None, None


@dataclass
class RefinedAnswerResult:
    """Encapsulates the final synthesized answer and diagnostic metadata."""
    answer: str
    model_name: str
    is_refined: bool
    validation: ValidationResult
    latency_ms: float
    grounded: bool
    source_citations: List[Dict[str, Any]] = field(default_factory=list)


def formulate_conversational_intro(query: str) -> str:
    """Generate a natural, helpful ChatGPT-style introductory lead-in matching query intent."""
    from .nlg_reconstructor import GeneralizedNLGReconstructor
    return GeneralizedNLGReconstructor.synthesize_conversational_intro(query)



class AnswerRefinementEngine:
    """Orchestrates document evidence consolidation, Transformer generation, structuring, and validation."""

    def __init__(self, config: Optional[ATMLConfig] = None):
        self.config = config or default_config
        self.structurer = AnswerStructurer()
        self.validator = OutputValidator(self.config)

    def consolidate_evidence(self, chunks: List[Dict[str, Any]], query: str = "") -> Tuple[str, List[Dict[str, Any]]]:
        """Aggregate and sanitize top retrieved passages into clean evidence context."""
        cleaned_passages: List[str] = []
        citations: List[Dict[str, Any]] = []
        seen_texts = set()

        is_quiz_query = bool(re.search(r"\b(quiz|quizzes|questions?|test|practice|exercise|exercises|sample\s+questions)\b", query, re.I))

        for chunk in chunks[: self.config.max_context_chunks]:
            raw_text = chunk.get("text", "") or chunk.get("formatted_context", "")
            if not raw_text:
                continue

            sec_title = chunk.get("section_title") or ""
            if not is_quiz_query:
                from .content_normalizer import ContentNormalizer
                if ContentNormalizer.is_assessment_or_quiz_content(sec_title, raw_text):
                    continue

            # Strip raw Markdown table formatting from text stream (tables are rendered separately)
            clean_lines = [l for l in raw_text.splitlines() if not l.strip().startswith("|")]
            text_block = "\n".join(clean_lines).strip()
            text_block = sanitize_text(text_block)

            if text_block and text_block not in seen_texts:
                seen_texts.add(text_block)
                # Limit length per chunk to prevent prompt saturation
                if len(text_block) > self.config.max_chunk_chars:
                    text_block = text_block[: self.config.max_chunk_chars].rstrip() + "..."
                cleaned_passages.append(text_block)

                citations.append({
                    "document_id": chunk.get("document_id"),
                    "source_filename": chunk.get("source_filename"),
                    "page_number": chunk.get("page_number"),
                    "section_title": chunk.get("section_title"),
                    "relevance_score": chunk.get("relevance_score", 0.0),
                })

        evidence = "\n\n".join(cleaned_passages).strip()
        return evidence, citations

    def generate_raw_synthesis(
        self,
        query: str,
        evidence: str,
        model: AutoModelForSeq2SeqLM,
        tokenizer: AutoTokenizer,
    ) -> str:
        """Execute sequence-to-sequence autoregressive generation conditioning on evidence."""
        clean_q = query.strip()
        prompt = (
            f"Context:\n{evidence}\n\n"
            f"Question: {clean_q}\n\n"
            f"Provide a comprehensive, detailed, and complete answer for the question based strictly on the context above. Include all relevant points and examples:\n\n"
            f"Detailed Answer:"
        )

        device = torch.device(self.config.device)
        inputs = tokenizer(
            prompt,
            max_length=self.config.max_input_length,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            output_tokens = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                max_new_tokens=self.config.max_output_length,
                min_length=self.config.min_output_length,
                num_beams=self.config.num_beams,
                no_repeat_ngram_size=self.config.no_repeat_ngram_size,
                length_penalty=self.config.length_penalty,
                early_stopping=self.config.early_stopping,
            )

        generated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        return generated_text.strip()

    def refine_answer(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        fallback_direct_answer: Optional[str] = None,
        retrieval_confidence: float = 1.0,
    ) -> RefinedAnswerResult:
        """Execute end-to-end answer refinement: Evidence Consolidation -> Transformer Generation -> Structuring -> Validation."""
        start_time = time.time()

        # Check for empty retrieval
        if not retrieved_chunks:
            val_res = ValidationResult(
                is_valid=False,
                confidence_score=0.0,
                issues=["No retrieved chunks available."],
                sanitized_answer="The retrieved documents do not provide sufficient information to answer this question.",
            )
            return RefinedAnswerResult(
                answer=val_res.sanitized_answer,
                model_name="none",
                is_refined=False,
                validation=val_res,
                latency_ms=0.0,
                grounded=False,
            )

        evidence, citations = self.consolidate_evidence(retrieved_chunks, query=query)
        if not evidence:
            val_res = ValidationResult(
                is_valid=False,
                confidence_score=0.0,
                issues=["Retrieved evidence is empty after cleaning."],
                sanitized_answer="The retrieved documents do not provide sufficient information to answer this question.",
            )
            return RefinedAnswerResult(
                answer=val_res.sanitized_answer,
                model_name="none",
                is_refined=False,
                validation=val_res,
                latency_ms=0.0,
                grounded=False,
                source_citations=citations,
            )

        # Attempt to load model
        model, tokenizer = get_atml_model_and_tokenizer(self.config)

        # Build comprehensive structured fallback from retrieved evidence
        is_quiz_query = bool(re.search(r"\b(quiz|quizzes|questions?|test|practice|exercise|exercises|sample\s+questions)\b", query, re.I))
        fallback_source = fallback_direct_answer or evidence
        if fallback_direct_answer and not is_quiz_query:
            from .content_normalizer import ContentNormalizer
            if ContentNormalizer.is_assessment_or_quiz_content("", fallback_direct_answer):
                fallback_source = evidence

        structured_fallback = self.structurer.structure(query, fallback_source, source_metadata=citations)

        # If model is unavailable or refinement disabled, fall back safely to structured text
        if model is None or tokenizer is None or not self.config.enable_refinement:
            logger.info("Transformer refinement bypassed; applying structured formatting to direct answer.")
            val = self.validator.validate(query, structured_fallback, [evidence], retrieval_confidence=retrieval_confidence)
            latency = (time.time() - start_time) * 1000.0

            return RefinedAnswerResult(
                answer=structured_fallback,
                model_name="extractive_fallback",
                is_refined=False,
                validation=val,
                latency_ms=round(latency, 2),
                grounded=val.is_valid,
                source_citations=citations,
            )

        # Generate rewritten candidate with Transformer
        try:
            raw_generation = self.generate_raw_synthesis(query, evidence, model, tokenizer)
        except Exception as exc:
            logger.warning("Seq2Seq generation encountered an exception: %s. Using extractive fallback.", exc)
            raw_generation = fallback_source

        # Intent-driven structuring
        structured_answer = self.structurer.structure(query, raw_generation, source_metadata=citations)

        # Detect if model generation shortcutted or emitted an extractive meta-sentence
        # (e.g. 'The relevant sentence in the passage is...', or omitted key multi-point educational sections)
        is_meta_phrase = bool(re.search(r"\b(the\s+relevant\s+sentence|the\s+passage\s+states|purpose\s+of\s+[^-\n]+-|passage\s+is:?|relevant\s+sentence)\b", raw_generation, re.I))
        has_rich_fallback = len(re.findall(r"(?:^|\n)\s*\d+[\.\)]", structured_fallback)) >= 2 or len(structured_fallback) > 300
        generation_missed_points = has_rich_fallback and len(re.findall(r"(?:^|\n)\s*\d+[\.\)]", structured_answer)) < 2
        is_too_brief = len(structured_answer) < 260 and len(structured_fallback) > 300
        is_significant_shortcut = len(structured_answer) < len(structured_fallback) * 0.65 and len(structured_fallback) > 280

        # Keep answers as long and complete as they need to be (no shortcuts)
        if is_meta_phrase or is_too_brief or generation_missed_points or is_significant_shortcut:
            logger.info("Preserving comprehensive multi-point evidence over shortened summary.")
            if "Based on" in structured_fallback or "According to" in structured_fallback:
                structured_answer = structured_fallback
            elif structured_fallback.startswith("#"):
                lines_fb = structured_fallback.splitlines()
                first_heading = lines_fb[0]
                rest_fb = "\n".join(lines_fb[1:]).strip()
                intro_lead = formulate_conversational_intro(query)
                structured_answer = f"{first_heading}\n\n{intro_lead}\n\n{rest_fb}"
            else:
                intro_lead = formulate_conversational_intro(query)
                structured_answer = f"{intro_lead}\n\n{structured_fallback}"

        # Attach pedagogical synthesis conclusion if not already present
        from .nlg_reconstructor import GeneralizedNLGReconstructor
        core_subj = GeneralizedNLGReconstructor.extract_core_subject(query)
        intent = GeneralizedNLGReconstructor.detect_academic_intent(query, evidence)
        conclusion = GeneralizedNLGReconstructor.synthesize_conversational_conclusion(intent, core_subj)
        if conclusion and not structured_answer.endswith("*") and conclusion not in structured_answer:
            structured_answer = f"{structured_answer}\n\n*{conclusion}*"

        # Validate factual fidelity and hallucination guard
        validation = self.validator.validate(
            query=query,
            generated_answer=structured_answer,
            retrieved_contexts=[evidence],
            retrieval_confidence=retrieval_confidence,
        )

        # If validation fails due to low grounding, fall back safely to structured evidence
        final_answer = structured_answer
        is_refined = True

        if not validation.is_valid:
            logger.warning("Validation flagged issues: %s. Checking fallback.", validation.issues)
            val_fallback = self.validator.validate(query, structured_fallback, [evidence], retrieval_confidence=retrieval_confidence)
            if val_fallback.is_valid and val_fallback.confidence_score > validation.confidence_score:
                final_answer = structured_fallback
                validation = val_fallback
                is_refined = False
            else:
                final_answer = validation.sanitized_answer
                is_refined = False

        latency = (time.time() - start_time) * 1000.0

        return RefinedAnswerResult(
            answer=final_answer,
            model_name=self.config.model_name,
            is_refined=is_refined,
            validation=validation,
            latency_ms=round(latency, 2),
            grounded=validation.is_valid,
            source_citations=citations,
        )

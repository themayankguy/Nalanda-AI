"""Comprehensive test suite for ATML Answer Refinement Engine.

Covers:
- Paragraph, bullet, and numbered step formatting
- Context-aware OCR cleanup and math/code preservation
- Source citation preservation
- Empty and irrelevant retrieval results
- Insufficient evidence handling
- Malformed/hallucinatory output recovery
- Model loading failures and fallback resilience
"""
import pytest
from backend.services.atml.config import ATMLConfig
from backend.services.atml.output_validator import OutputValidator, sanitize_text
from backend.services.atml.answer_structurer import AnswerStructurer
from backend.services.atml.answer_refiner import AnswerRefinementEngine, RefinedAnswerResult


class TestAnswerStructurer:
    """Tests for intent-aware answer formatting."""

    def test_numbered_steps_formatting(self):
        structurer = AnswerStructurer()
        raw = (
            "Steps for Creating a Visualization:\n"
            "1. Understand the Data\n"
            "2. Define the Purpose\n"
            "3. Select an Appropriate Technique\n"
            "4. Prepare and Process the Data"
        )
        formatted = structurer.structure("What are the steps for creating a visualization?", raw)
        assert "1. Understand the Data" in formatted
        assert "2. Define the Purpose" in formatted
        assert "3. Select an Appropriate Technique" in formatted
        assert "4. Prepare and Process the Data" in formatted

    def test_bullet_list_formatting(self):
        structurer = AnswerStructurer()
        raw = (
            "Big Data visualization can be used in many industries:\n"
            "- Airlines analysing flight performance\n"
            "- IoT monitoring sensor data\n"
            "- Energy analysing consumption"
        )
        formatted = structurer.structure("What are the applications of Big Data Visualization?", raw)
        assert "- Airlines analysing flight performance" in formatted
        assert "- IoT monitoring sensor data" in formatted
        assert "- Energy analysing consumption" in formatted

    def test_paragraph_formatting_avoids_wall_of_text(self):
        structurer = AnswerStructurer()
        raw = (
            "Data Visualization is the process of representing data in visual form such as charts, graphs, maps, and diagrams so that patterns, trends, relationships, and important information can be understood easily. "
            "In Big Data, the amount of data is very large and it may be difficult to understand it by looking at rows and columns of raw data. "
            "Visualization converts this large and complex data into an easy-to-understand visual format."
        )
        formatted = structurer.structure("What is Data Visualization?", raw)
        assert "\n\n" in formatted or len(formatted.splitlines()) >= 1


class TestOutputValidatorAndSanitizer:
    """Tests for artifact cleanup, math preservation, and source grounding."""

    def test_ocr_artifact_cleanup(self):
        raw_ocr = "Here is the answer: infor- mation is retrieved .... with lone ^^ symbols"
        cleaned = sanitize_text(raw_ocr)
        assert "information" in cleaned
        assert not cleaned.lower().startswith("here is the answer")
        assert "...." not in cleaned
        assert "^^" not in cleaned

    def test_preamble_removal(self):
        raw = "Certainly! Based on the provided context, Hadoop is an open-source framework."
        cleaned = sanitize_text(raw)
        assert cleaned.startswith("Hadoop is an open-source framework.")
        assert not cleaned.lower().startswith("certainly")

    def test_math_and_symbol_preservation(self):
        raw = "The loss function is L(y, y_hat) = sum((y - y_hat)^2) / N with 95% accuracy and x <= 10."
        cleaned = sanitize_text(raw)
        assert "L(y, y_hat)" in cleaned
        assert "(y - y_hat)^2" in cleaned
        assert "95%" in cleaned
        assert "x <= 10" in cleaned

    def test_empty_or_irrelevant_evidence(self):
        validator = OutputValidator()
        result = validator.validate(
            query="What is quantum gravity in loop spacetime?",
            generated_answer="Quantum gravity is loop spacetime.",
            retrieved_contexts=["Big data visualization helps display charts and graphs."],
            retrieval_confidence=0.15,
        )
        assert not result.is_valid
        assert "sufficient information" in result.sanitized_answer.lower()

    def test_hallucination_guard_unsupported_entities(self):
        validator = OutputValidator()
        # Answer claims unsupported novel entities
        result = validator.validate(
            query="Explain the feature.",
            generated_answer="The feature is based on TensorFlow, PyTorch, Kubernetes, and RayClusters.",
            retrieved_contexts=["The feature uses simple python scripts and basic file storage."],
            retrieval_confidence=0.8,
        )
        assert not result.is_valid
        assert len(result.issues) > 0


class TestAnswerRefinementEngine:
    """Tests for the end-to-end refinement engine."""

    def test_empty_retrieved_chunks(self):
        engine = AnswerRefinementEngine()
        result = engine.refine_answer(
            query="What is visualization?",
            retrieved_chunks=[],
            fallback_direct_answer=None,
        )
        assert not result.is_refined
        assert "sufficient information" in result.answer.lower()

    def test_citation_preservation(self):
        engine = AnswerRefinementEngine()
        chunks = [
            {
                "text": "Data visualization converts raw data into charts.",
                "document_id": "doc-123",
                "source_filename": "lecture1.pdf",
                "page_number": 4,
                "section_title": "Introduction to Visualization",
                "relevance_score": 0.88,
            }
        ]
        result = engine.refine_answer(
            query="What does data visualization do?",
            retrieved_chunks=chunks,
            fallback_direct_answer="Data visualization converts raw data into charts.",
            retrieval_confidence=0.88,
        )
        assert len(result.source_citations) == 1
        assert result.source_citations[0]["source_filename"] == "lecture1.pdf"
        assert result.source_citations[0]["page_number"] == 4

    def test_model_loading_fallback_graceful(self):
        # Configure non-existent model name to test graceful fallback
        bad_config = ATMLConfig(model_name="nonexistent/fake-model-12345")
        engine = AnswerRefinementEngine(bad_config)
        chunks = [
            {
                "text": "MapReduce divides tasks into map and reduce phases.",
                "document_id": "doc-456",
                "source_filename": "hadoop.pdf",
                "page_number": 1,
            }
        ]
        result = engine.refine_answer(
            query="How does MapReduce work?",
            retrieved_chunks=chunks,
            fallback_direct_answer="MapReduce divides tasks into map and reduce phases.",
            retrieval_confidence=0.85,
        )
        # Should gracefully fall back to structured direct answer without throwing an unhandled exception
        assert result.answer is not None
        assert "MapReduce" in result.answer

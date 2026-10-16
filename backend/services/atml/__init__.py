"""ATML package: Transformer-based Answer Refinement Engine for Nalanda AI.

Exposes:
- AnswerRefinementEngine: Main coordinator for Seq2Seq rewriting, structuring, and validation.
- AnswerStructurer: Intent-aware Markdown formatting (steps, bullets, paragraphs).
- OutputValidator: Factual consistency, hallucination detection, and source-grounding checks.
- ATMLConfig: Configuration and hardware parameters.
"""
from .config import ATMLConfig, default_config
from .output_validator import OutputValidator, ValidationResult, sanitize_text
from .answer_structurer import AnswerStructurer
from .answer_refiner import (
    AnswerRefinementEngine,
    RefinedAnswerResult,
    get_atml_model_and_tokenizer,
)
from .quiz_generator import QuizGenerator, QuizQuestion, QuizResponse

__all__ = [
    "ATMLConfig",
    "default_config",
    "OutputValidator",
    "ValidationResult",
    "sanitize_text",
    "AnswerStructurer",
    "AnswerRefinementEngine",
    "RefinedAnswerResult",
    "get_atml_model_and_tokenizer",
    "QuizGenerator",
    "QuizQuestion",
    "QuizResponse",
]

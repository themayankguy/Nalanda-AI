"""Configuration module for ATML Answer Refinement Engine.

Defines model hyperparameters, generation parameters, device routing,
and validation thresholds for sequence-to-sequence answer refinement.
"""
import os
from dataclasses import dataclass, field
from typing import Optional
import torch


@dataclass
class ATMLConfig:
    """Hyperparameters and operational configuration for ATML Answer Refinement."""

    # Model identifiers: google/flan-t5-small (77M params, ~300MB) or google/flan-t5-base (250M params)
    model_name: str = os.getenv("ATML_MODEL_NAME", "google/flan-t5-small")

    # Hardware target: prioritize MPS on Apple Silicon, fallback to CPU
    device: str = os.getenv(
        "ATML_DEVICE",
        "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"),
    )

    # Feature toggle for answer refinement pipeline
    enable_refinement: bool = os.getenv("ATML_ENABLE_REFINEMENT", "1").lower() in ("1", "true", "yes")

    # Sequence generation constraints
    max_input_length: int = int(os.getenv("ATML_MAX_INPUT_LENGTH", "2048"))
    max_output_length: int = int(os.getenv("ATML_MAX_OUTPUT_LENGTH", "1536"))
    min_output_length: int = int(os.getenv("ATML_MIN_OUTPUT_LENGTH", "32"))
    num_beams: int = int(os.getenv("ATML_NUM_BEAMS", "2"))
    no_repeat_ngram_size: int = int(os.getenv("ATML_NO_REPEAT_NGRAM", "3"))
    length_penalty: float = float(os.getenv("ATML_LENGTH_PENALTY", "1.0"))
    early_stopping: bool = True

    # Grounding & Validation thresholds
    min_faithfulness_overlap: float = float(os.getenv("ATML_MIN_FAITHFULNESS", "0.35"))
    min_confidence_score: float = float(os.getenv("ATML_MIN_CONFIDENCE", "0.40"))
    max_unsupported_entity_ratio: float = float(os.getenv("ATML_MAX_UNSUPPORTED_ENTITIES", "0.45"))

    # Context formatting
    max_context_chunks: int = int(os.getenv("ATML_MAX_CONTEXT_CHUNKS", "6"))
    max_chunk_chars: int = int(os.getenv("ATML_MAX_CHUNK_CHARS", "8000"))


# Default singleton config instance
default_config = ATMLConfig()

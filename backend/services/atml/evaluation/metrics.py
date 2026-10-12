"""Evaluation metrics for ATML Answer Refinement.

Includes factual consistency, token F1, pure-Python ROUGE-L,
structural quality scoring, and latency profiling.
"""
import re
import time
from typing import List, Set, Dict, Any
try:
    from backend.services.atml.output_validator import extract_content_tokens, extract_named_or_technical_entities, PREAMBLE_PATTERNS
except ImportError:
    from ..output_validator import extract_content_tokens, extract_named_or_technical_entities, PREAMBLE_PATTERNS


def compute_token_f1(prediction: str, reference: str) -> float:
    """Compute token-level F1 score between prediction and reference."""
    pred_tokens = re.findall(r"\b\w+\b", (prediction or "").lower())
    ref_tokens = re.findall(r"\b\w+\b", (reference or "").lower())

    if not pred_tokens or not ref_tokens:
        return 0.0

    common = {}
    ref_counts = {}
    for t in ref_tokens:
        ref_counts[t] = ref_counts.get(t, 0) + 1

    for t in pred_tokens:
        if ref_counts.get(t, 0) > 0:
            common[t] = common.get(t, 0) + 1
            ref_counts[t] -= 1

    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return round(f1, 4)


def compute_rouge_l(prediction: str, reference: str) -> float:
    """Compute LCS-based ROUGE-L F1 score without external binary dependencies."""
    pred_tokens = re.findall(r"\b\w+\b", (prediction or "").lower())
    ref_tokens = re.findall(r"\b\w+\b", (reference or "").lower())

    m, n = len(pred_tokens), len(ref_tokens)
    if m == 0 or n == 0:
        return 0.0

    # Dynamic programming for LCS length
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if pred_tokens[i] == ref_tokens[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1])

    lcs_len = dp[m][n]
    if lcs_len == 0:
        return 0.0

    prec = lcs_len / m
    rec = lcs_len / n
    rouge_l = (2 * prec * rec) / (prec + rec)
    return round(rouge_l, 4)


def compute_factual_consistency(prediction: str, evidence: str) -> float:
    """Measure token precision of prediction content against source evidence."""
    pred_tokens = extract_content_tokens(prediction)
    if not pred_tokens:
        return 0.0
    evidence_tokens = extract_content_tokens(evidence)
    shared = pred_tokens.intersection(evidence_tokens)
    return round(len(shared) / len(pred_tokens), 4)


def compute_structural_score(query: str, text: str, expected_format: str) -> float:
    """Score structural quality (0.0 to 1.0) based on formatting requirements."""
    if not text:
        return 0.0

    score = 1.0

    # 1. Penalize conversational preamble
    for pat in PREAMBLE_PATTERNS:
        if re.search(pat, text, flags=re.I):
            score -= 0.25
            break

    # 2. Check format alignment
    has_numbered_steps = bool(re.search(r"(?:^|\n)\s*\d+[\.\)]\s+", text))
    has_bullets = bool(re.search(r"(?:^|\n)\s*[-•*+]\s+", text))

    if expected_format == "steps":
        if has_numbered_steps:
            score += 0.0  # target format met
        else:
            score -= 0.35
    elif expected_format == "bullets":
        if has_bullets or has_numbered_steps:
            score += 0.0
        else:
            score -= 0.30
    elif expected_format == "paragraphs":
        # Check that it is not an unreadable single wall of text if long
        lines = [l.strip() for l in text.split("\n\n") if l.strip()]
        if len(text) > 300 and len(lines) == 1:
            score -= 0.20

    # 3. Penalize raw OCR artifact leftovers (e.g. repeated lone punctuation or broken word hyphens)
    if re.search(r"\b[a-zA-Z]{2,}-\s+[a-zA-Z]{2,}\b", text):
        score -= 0.15
    if re.search(r"(?<!\w)\^+(?!\w)", text):
        score -= 0.15

    return round(max(0.0, min(1.0, score)), 3)


def compute_unsupported_entity_rate(prediction: str, evidence: str) -> float:
    """Measure proportion of named/technical entities in prediction absent from evidence."""
    entities = extract_named_or_technical_entities(prediction)
    if not entities:
        return 0.0
    ev_lower = evidence.lower()
    unsupported = [e for e in entities if e.lower() not in ev_lower]
    return round(len(unsupported) / len(entities), 4)

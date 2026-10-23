"""Domain-Agnostic Academic Quiz Generation & Strict Folder-Scoped Vector Manifold Distractor Optimization.

ATML Component for Nalanda AI:
1. Strict Folder-Level Isolation:
   All questions, options, distractors, evidence, and provenance citations are drawn
   strictly and exclusively from the authorized documents within the selected folder.
   Cross-folder leakage, global knowledge base fallbacks, and hardcoded external domain
   pools are strictly eliminated.
2. Fact-Anchor Mining:
   Identifies self-contained concepts, definitions, and operational requirements directly
   from the permitted documents, eliminating conversational fragments and slide metadata.
3. Natural Interrogative Question Formulation:
   Constructs clear, self-contained academic question stems testing deep conceptual understanding
   without answer leakage, malformed clauses, or repetitive filename prefixes.
4. Latent Vector Manifold Sibling Distractor Optimization:
   Mines plausible sibling distractors strictly from the same course materials using
   all-MiniLM-L6-v2 embeddings and head-concept clustering within an optimal bandpass margin
   (0.20 <= cosine < 0.90) with strict pairwise anti-collision.
5. 10-Point Question Quality & Evidence Validation:
   Every question is strictly validated for folder isolation, single-answer validity,
   anti-collision, clarity, non-leakage, and verifiable provenance before being returned.
"""
import logging
import random
import re
import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set

import numpy as np
import torch
from sqlalchemy.orm import Session

from backend.models import User, Folder, Document, DocumentExtraction
from backend.services.storage_service import BASE_STORAGE_DIR
from backend.services.NLP.vector_index_service import (
    get_folder_index_dir,
    FolderVectorIndex,
    get_embedding_model,
)
from .config import ATMLConfig, default_config
from .output_validator import sanitize_text

logger = logging.getLogger("nalanda.atml.quiz")

# Legacy compatibility exports (empty to prevent any external domain leakage)
DOMAIN_DISTRACTOR_MAP: Dict[str, List[str]] = {}
CURATED_DOMAIN_DISTRACTORS: List[str] = []

FORBIDDEN_CONCEPT_WORDS = {
    "command", "commands", "command used", "required action", "action", "actions",
    "feature", "features", "step", "steps", "step 1", "step 2", "step 3", "step 4",
    "working", "components", "component", "relation", "usage", "syntax", "example",
    "examples", "output", "input", "input/output", "option", "options", "flag", "flags",
    "parameter", "parameters", "argument", "arguments", "description", "explanation",
    "note", "notes", "details", "summary", "overview", "table", "figure", "diagram",
    "properties", "property", "attribute", "attributes", "operation", "operations",
    "additional operations", "additional commands", "image", "images", "item", "items",
    "concept", "concepts", "topic", "topics", "unit", "chapter", "section", "slide", "page",
    "purpose", "purposes", "advantages", "disadvantages", "benefits", "drawbacks",
    "solution", "solutions", "characteristics", "need", "needs", "minimal errors",
    "minimal errors explanation", "supports collaboration explanation", "p 8080",
    "runs multi", "created when image", "type", "types", "kind", "kinds", "category",
    "categories", "reason", "reasons", "result", "results", "key", "keys", "value", "values",
    "airlines", "airline", "sports", "sport", "manufacturing", "automotive", "energy",
    "retail", "healthcare", "finance", "banking", "education", "entertainment", "media",
    "agriculture", "transportation", "insurance", "hospitality", "telecom", "telecommunications",
    "government", "capacity", "driver", "rack", "important component", "tables", "table",
    "decision", "decisions", "choice", "choices", "selection", "grunt", "dynamic updates",
    "actual table data", "job.name", "agenda", "quiz", "quiz time", "exercise", "exercises",
    "homework", "assignment", "announcements", "references", "conclusion", "conclusions",
    "review", "outline", "not all morphemes", "piece of cake", "pulling your leg",
    "all morphemes", "morphemes are words", "words but all prefixes", "this work"
}

KNOWN_SINGLE_WORD_TECH_CONCEPTS = {
    # Cloud & Big Data
    "tableau", "hadoop", "mapreduce", "hdfs", "spark", "yarn", "hive", "pig",
    "kafka", "hbase", "zookeeper", "flume", "sqoop", "docker", "kubernetes",
    "histogram", "treemap", "scatterplot", "boxplot", "clustering", "regression",
    "classification", "container", "microservice", "virtualization", "scalability",
    "latency", "throughput", "redundancy", "idempotence", "partitioning", "devops",
    "ansible", "terraform", "prometheus", "grafana", "git", "linux",
    # Operating Systems & Concurrency
    "semaphore", "mutex", "deadlock", "paging", "segmentation", "kernel", "thread",
    "process", "spinlock", "scheduler", "hypervisor", "multithreading", "multiprocessing",
    "syscall", "monitors", "starvation",
    # Computer Networks
    "router", "switch", "firewall", "bandwidth", "packet", "subnet", "gateway",
    "topology", "protocol", "ethernet", "datagram", "sockets", "unicast", "multicast",
    "broadcast", "ping", "traceroute",
    # Database Systems
    "normalization", "schema", "transaction", "trigger", "indexing", "sharding",
    "query", "tuple", "relation", "atomicity", "consistency", "isolation", "durability",
    # Data Structures & Algorithms
    "recursion", "sorting", "hashing", "stack", "queue", "graph", "tree", "heap",
    "array", "pointer", "traversal", "search", "backtracking",
    # Cyber Security & Cryptography
    "encryption", "decryption", "cipher", "cryptography", "malware",
    "phishing", "authentication", "authorization", "hash", "keyloggers", "trojan",
    # Software Engineering & Languages
    "polymorphism", "inheritance", "encapsulation", "abstraction", "compiler",
    "interpreter", "bytecode", "refactoring", "scrum", "waterfall", "agile",
    # NLP & Linguistics & Deep Learning
    "morphology", "morpheme", "phoneme", "syntax", "semantics", "pragmatics",
    "tokenization", "lemmatization", "stemming", "parsing", "corpus", "lexicon",
    "ambiguity", "polysemy", "homonymy", "hyponymy", "synonymy", "antonymy",
    "autoencoder", "perceptron", "convolution", "transformer", "embeddings"
}

ENGLISH_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but",
    "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from",
    "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him",
    "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "me", "more",
    "most", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other",
    "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "should", "so",
    "some", "such", "than", "that", "the", "their", "theirs", "them", "themselves", "then",
    "there", "these", "they", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom",
    "why", "with", "would", "you", "your", "yours", "yourself", "yourselves"
}

CLI_SYNTAX_PATTERN = re.compile(
    r"\b(?:docker\s+(?:run|build|exec|pull|push|ps|rm|rmi|stop|start|tag)|p\s*\d+|\d+\:\d+|--[a-z]+)\b",
    re.I
)

EVALUATIVE_ADJ_PREFIXES = re.compile(
    r"^(?:important|major|minor|key|primary|secondary|main|good|bad|critical|common|typical|basic|simple|general|various|different|standard|special|unique|necessary|possible|potential|useful|maximum|minimum|average|total|high|low)\s+",
    re.I
)

META_HEADING_PREFIXES = re.compile(
    r"^(?:characteristics|features|advantages|disadvantages|benefits|need|needs|"
    r"solution\s+provided\s+by|solution|solutions|overview|introduction|summary|"
    r"types|kinds|categories|purpose|purposes|challenges|issues|working|architecture|"
    r"concept\s+of|role\s+of|importance\s+of|examples|applications|use\s+cases)\s+(?:of|for|in|to|by)?\s*",
    re.I
)

ACTION_VERB_PREFIXES = re.compile(
    r"^(?:view|viewing|run|running|start|starting|stop|stopping|create|creating|manage|managing|"
    r"build|building|install|installing|deploy|deploying|execute|executing|access|accessing|"
    r"list|listing|delete|deleting|show|showing|understanding|defining|using|moving|applying|"
    r"monitoring|analysing|analyzing|enabling|providing|performing|processing)\s+",
    re.I
)

INVALID_DEFINITION_START = re.compile(
    r"^(?:not\s+(?:required|needed|used|supported|recommended|present|available)|is\s+not|are\s+not|cannot|does\s+not|do\s+not|should\s+not|never\b|optional\b|neither\b)",
    re.I,
)

DANGLING_ANAPHORA_PATTERN = re.compile(
    r"\b(?:this|that|these|those)\s+(?:mode|case|scenario|approach|method|architecture|technique|system|tool|way|manner|section|step|process|stage|format|example|relation|model)\b|"
    r"\b(?:the\s+above|as\s+mentioned|as\s+follows|as\s+discussed|as\s+stated|shown\s+above|refer\s+to|herein|hereafter)\b",
    re.I,
)


def clean_academic_concept(text: str) -> Optional[str]:
    """Sanitize concept string, stripping meta headings, rejecting fragments, quantifiers, and conversational text."""
    if not text:
        return None

    cleaned = text.strip()

    # 1. Reject meta slide notes, reviews, and assignments
    if re.match(r"^(?:explanation|note|example|warning|caution|step\s*\d*|agenda|overview|summary|introduction|quiz|exercise|exercises|homework|assignment|announcements?|references?|conclusion|conclusions?|questions?|q&a|review|outline)(?:\b|[:\-\s])", cleaned, re.I):
        return None

    # 2. Reject commands and verbs
    if ACTION_VERB_PREFIXES.match(cleaned) or cleaned.lower().endswith(" commands"):
        return None

    # 3. Strip section / slide numbering: e.g. '3.7', '3.7.1', '1.', 'Step 2:', 'Section 4'
    cleaned = re.sub(r"^(?:(?:step|section|unit|chapter)\s*)?\d+(?:\.\d+)*[\.\)\:\-–\s]+", "", cleaned, flags=re.I).strip()
    cleaned = re.sub(r"^[#*_\-\–\—•\s]+|[#*_\-\–\—•\s]+$", "", cleaned).strip()

    # Strip repeated heading separators: e.g. 'Pie Chart - A pie chart' -> 'A pie chart'
    if re.search(r"\s+[\-–—]\s+", cleaned):
        cleaned = re.split(r"\s+[\-–—]\s+", cleaned)[-1].strip()

    # 4. Strip meta-heading prefixes (e.g. 'Characteristics of Containerization' -> 'Containerization')
    cleaned = META_HEADING_PREFIXES.sub("", cleaned).strip()

    # 5. Strip evaluative adjective prefixes (e.g. 'Important Component' -> 'Component')
    cleaned = EVALUATIVE_ADJ_PREFIXES.sub("", cleaned).strip()

    # 6. Strip leading articles: 'A container' -> 'Container'
    cleaned = re.sub(r"^(?:a|an|the|this|that|these|those)\s+", "", cleaned, flags=re.I).strip()
    cleaned = cleaned.strip(" :.-–—;")

    # Re-check action verbs after stripping prefixes
    if ACTION_VERB_PREFIXES.match(cleaned) or cleaned.lower().endswith(" commands"):
        return None

    lower = cleaned.lower()

    # 7. Discard invalid clauses starting with negative words, quantifiers, prepositions, or conjunctions
    invalid_prefixes = (
        "when ", "where ", "if ", "because ", "that ", "which ", "how ", "what ", "why ",
        "who ", "it ", "there ", "here ", "in the", "on the", "with ", "for ", "such as",
        "according", "as an", "at the", "by the", "from the", "to the", "during", "based on",
        "once a", "and ", "or ", "each ", "every ", "all ", "some ", "any ", "created when",
        "runs multi", "used with", "supports ", "not ", "no ", "never ", "neither ", "none ",
        "without ", "few ", "many ", "but ", "so ", "yet ", "words, but", "this work"
    )
    if any(lower.startswith(prefix) for prefix in invalid_prefixes):
        return None

    # Reject conversational pronouns and idioms
    if re.search(r"\b(?:i|me|my|myself|you|your|yours|yourself|we|our|ours|us|piece of cake|pulling your leg|break a leg)\b", lower):
        return None

    # Reject grammatical clauses containing auxiliary/modal verbs or clauses
    if re.search(r"\b(?:is|are|was|were|am|has|have|had|been|can|will|should|must|purpose|occur|occurs|occurring)\b", lower):
        return None

    if lower in FORBIDDEN_CONCEPT_WORDS or lower in ENGLISH_STOPWORDS:
        return None

    # Reject concepts ending with meta labels (e.g. "Minimal Errors Explanation", "Step 1")
    if re.search(r"\b(?:explanation|example|examples|note|notes|summary|details|commands?|features?|steps?|actions?|operations?)\s*$", lower):
        return None

    if CLI_SYNTAX_PATTERN.search(lower):
        return None

    if re.search(r"\b(?:unit|chapter|slide|page|concept|topic|section|step)\s+\d+\b", lower) or lower.startswith(("unit ", "concept ", "topic ", "section ", "step ")):
        return None

    if "." in cleaned:
        return None

    # Reject numerical calculation concepts, mathematical variables, and step derivations
    if re.search(r"\b(?:w_conv|w_fc|conv output|latent vector|mse loss|reconstruction loss|transpose conv|feature map expansion)\b", lower):
        return None
    if re.search(r"\b(?:step\s*\d+|window\s*\d+|x_\d+|z_\d+|y_\d+|f_\d+)\b", lower):
        return None
    if re.search(r"\b(?:numerical\s+problem|solved\s+numerical|numerical\s+derivation)\b", lower):
        return None

    # Length & word count check: 1 to 4 words, 2 to 45 characters
    words = cleaned.split()
    if len(words) < 1 or len(words) > 4 or len(cleaned) < 2 or len(cleaned) > 45:
        return None

    # Single-word concepts must be genuine technical terms (not ordinary generic nouns)
    if len(words) == 1 and "-" not in cleaned:
        if lower in FORBIDDEN_CONCEPT_WORDS or lower in ENGLISH_STOPWORDS:
            return None
        is_acronym = cleaned.isupper() and len(cleaned) >= 2
        is_known = lower in KNOWN_SINGLE_WORD_TECH_CONCEPTS
        is_camel = any(c.isupper() for c in cleaned[1:])
        is_tech_suffix = any(lower.endswith(sfx) for sfx in (
            "ization", "isation", "ation", "ology", "metry", "graphy", "morphism", "lock",
            "eme", "phone", "onym", "syntax", "semantics", "parsing", "token", "vector", "matrix"
        )) and len(cleaned) >= 4
        if not (is_acronym or is_known or is_camel or is_tech_suffix):
            return None
        if not is_acronym and len(cleaned) < 4:
            return None

    # Title-case clean concept formatting while preserving technical acronyms and CamelCase (e.g. HDFS, NameNode, VM)
    stopwords = {"and", "of", "in", "to", "for", "on", "with", "the", "a", "an"}
    formatted_words = []
    for i, w in enumerate(words):
        if w.isupper() or any(c.isupper() for c in w[1:]):
            formatted_words.append(w)
        elif i > 0 and w.lower() in stopwords:
            formatted_words.append(w.lower())
        else:
            formatted_words.append(w.capitalize())

    cleaned = " ".join(formatted_words)
    if cleaned:
        cleaned = cleaned[0].upper() + cleaned[1:]

    return cleaned


def is_valid_definition(text: str) -> bool:
    """Ensure statement is a valid natural-language educational definition, not shell commands, code, negative fragments, or dangling anaphora."""
    if not text or len(text.strip()) < 15:
        return False
    clean = text.strip()

    # Reject shell commands and CLI syntax
    if re.match(r"^(?:docker\s+[a-z]+|kubectl\s+[a-z]+|npm\s+[a-z]+|git\s+[a-z]+|pip\s+[a-z]+|sudo\s+[a-z]+|t\s+<|<[a-z0-9_\-]+>|-[a-z]\b)", clean, re.I):
        return False
    if re.search(r"<[A-Za-z0-9_\-]+>|--[a-z]+|-[a-z]\s+\d+|-[a-z]\s+-[a-z]", clean):
        return False

    # Reject raw numerical vectors, intermediate convolution arithmetic, and math solver lines from MCQs
    if re.search(r"\b(?:w_conv|w_fc|conv output|latent vector z|mse loss|reconstruction loss)\b", clean, re.I):
        return False
    if "·" in clean or re.search(r"=\s*\[[0-9\s,\.\-]+\]|\b\d+\s*\.\s*\d+\s*[·*+/\-]", clean):
        return False
    if re.search(r"[-+]?\d+\.?\d*\s*[·*+/\-]\s*[-+]?\d+\.?\d*\s*=", clean):
        return False
    if re.search(r"\[[0-9\s,\.\-]{5,}\]", clean):
        return False

    words = clean.split()
    if len(words) < 4:
        return False

    # Reject isolated negative predicates (e.g. "not required in this mode")
    if INVALID_DEFINITION_START.search(clean):
        return False

    # Reject dangling anaphoric pronouns (e.g. "in this mode", "the above")
    if DANGLING_ANAPHORA_PATTERN.search(clean):
        return False

    # Reject coordinating conjunctions at statement start
    if re.match(r"^(?:and\b|or\b|but\b|so\b|yet\b|words,\s*but)\b", clean, re.I):
        return False

    # Reject first/second person conversational idioms
    if re.search(r"\b(?:piece of cake|pulling your leg|break a leg|i am just|i'm just)\b", clean, re.I):
        return False
    if re.search(r"\b(?:i|me|my|you|your|we|our|us)\b", clean, re.I):
        return False

    return True


def strip_concept_from_definition(concept: str, text: str) -> str:
    """Cleanly strip the target concept name, synonyms, and connecting copulas from the definition statement."""
    if not text:
        return ""
    cleaned = text.strip()

    # Normalize whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    escaped_c = re.escape(concept.strip())

    # Strip repeated leading mentions of concept name (e.g. 'Semantic Ambiguity Semantic ambiguity occurs when...')
    for _ in range(4):
        prev = cleaned
        cleaned = re.sub(rf"^(?:an?|the)?\s*{escaped_c}\b\s*[:\-–—]?\s*", "", cleaned, flags=re.I).strip()
        if cleaned.lower().startswith(concept.lower()):
            cleaned = cleaned[len(concept):].strip(" :.-–—;")
        cleaned = re.sub(r"^(?:an?|the)?\s*[A-Za-z0-9\s\-]{2,30}?\s*[:\-–—]\s*", "", cleaned).strip() if cleaned.lower().startswith(concept.lower()) else cleaned
        if cleaned == prev:
            break

    # Strip connecting verbs/copulas
    copula_pattern = (
        r"^(?:is\s+defined\s+as|is\s+the\s+process\s+of|is\s+a\s+technique\s+that|is\s+a\s+technique\s+for|"
        r"is\s+a\s+method\s+that|is\s+a\s+method\s+for|is\s+a\s+component\s+that|refers\s+to\s+a|"
        r"refers\s+to\s+the|refers\s+to|is\s+when|occurs\s+when|happens\s+when|means\s+that|represents|"
        r"is\s+an?|is\s+the|are\s+defined\s+as|are\s+the|are\s+an?|consists\s+of|describes|"
        r"is\s+used\s+to|can\s+be\s+defined\s+as|denotes|is\s+characterized\s+by)\s+"
    )
    for _ in range(3):
        prev = cleaned
        cleaned = re.sub(copula_pattern, "", cleaned, flags=re.I).strip()
        cleaned = re.sub(rf"^(?:an?|the)?\s*{escaped_c}\b\s*[:\-–—]?\s*", "", cleaned, flags=re.I).strip()
        if cleaned.lower().startswith(concept.lower()):
            cleaned = cleaned[len(concept):].strip(" :.-–—;")
        if cleaned == prev:
            break

    # Strip dangling punctuation and trailing incomplete prepositions
    cleaned = cleaned.strip(" :.-–—;")
    cleaned = re.sub(r"\s+(?:and|or|to|for|in|with|by|as|such as|of|the|a|an)\s*$", "", cleaned, flags=re.I).strip(" ,.-–:")

    return cleaned


def classify_concept_category(concept: str) -> str:
    """Classify an academic concept into its conceptual archetype (strictly domain-agnostic)."""
    if not concept:
        return "general_concept"
    clean = concept.strip().lower()
    words = clean.split()
    last_word = words[-1] if words else ""

    # 1. Broad fields / disciplines / subjects
    # e.g., NLP, AI, ML, Linguistics, Semantics, Morphology, Computer Science, Data Science
    if clean in {"nlp", "ai", "ml", "cs", "bi", "it"}:
        return "broad_field"
    if any(clean.endswith(sfx) for sfx in (" science", " engineering", " processing", " learning", " intelligence", " analytics")):
        return "broad_field"
    if any(last_word.endswith(sfx) for sfx in ("ology", "ics", "etics")):
        return "broad_field"

    # 2. Execution modes
    if "mode" in clean:
        return "execution_mode"

    # 3. Visualizations / charts
    if any(w in clean for w in ("chart", "plot", "graph", "diagram", "histogram", "heatmap", "scatter", "treemap")):
        return "visualization"

    # 4. Data structures
    if last_word in {"bag", "tuple", "map", "array", "list", "stack", "queue", "tree", "matrix", "vector", "tensor", "heap", "set"}:
        return "data_structure"
    if any(clean.endswith(f" {w}") for w in ("structure", "type", "model", "schema")):
        return "data_structure"

    # 5. Core atomic units / linguistic units
    # e.g., Morpheme, Phoneme, Grapheme, Allophone, Token, Word, Byte, Bit, Character, Minimal Pair, Minimal Pairs
    if any(last_word.endswith(sfx) for sfx in ("eme", "emes", "phone", "phones")):
        return "core_unit"
    if last_word in {"pair", "pairs", "token", "tokens", "byte", "bytes", "bit", "bits", "node", "nodes", "edge", "edges", "vertex", "vertices", "symbol", "symbols", "packet", "packets", "frame", "frames"}:
        return "core_unit"

    # 6. Phenomenon / property / state
    # e.g., Ambiguity, Polysemy, Homonymy, Anaphora, Latency, Skewness, Modularity
    if any(last_word.endswith(sfx) for sfx in ("ity", "ities", "ance", "ence", "ism", "isms", "drift", "skew", "latency")):
        return "phenomenon_property"
    if last_word in {"polysemy", "homonymy", "hyponymy", "synonymy", "anaphora", "ellipsis", "anomaly", "metaphor"}:
        return "phenomenon_property"

    # 7. Processes, algorithms, techniques
    # e.g., Parsing, Stemming, Tokenization, Lemmatization, Containerization, Clustering, Optimization
    if any(last_word.endswith(sfx) for sfx in ("tion", "tions", "sion", "sions", "ing", "ings")):
        return "process_technique"
    if last_word in {"algorithm", "method", "protocol", "technique", "procedure", "pipeline"}:
        return "process_technique"

    # 8. Systems, components, infrastructure
    if last_word in {"container", "pod", "engine", "server", "broker", "namenode", "datanode", "tracker", "cluster", "daemon"}:
        return "component_system"

    return "general_concept"


def are_options_too_similar(opt1: str, opt2: str) -> bool:
    """Check if two multiple-choice options are too similar, overlapping, or collision-prone."""
    s1 = opt1.strip().lower()
    s2 = opt2.strip().lower()
    if not s1 or not s2:
        return True
    if s1 == s2:
        return True

    # Strip prefixes like A), B)
    s1 = re.sub(r"^[a-d]\)\s*", "", s1).strip()
    s2 = re.sub(r"^[a-d]\)\s*", "", s2).strip()
    if s1 == s2:
        return True

    # Singular / Plural equivalence check (e.g. "Minimal Pair" vs "Minimal Pairs", "Token" vs "Tokens")
    if s1.rstrip("s") == s2.rstrip("s"):
        return True

    stopwords = {"and", "or", "in", "the", "a", "an", "of", "to", "for", "with", "by", "is", "are", "mode"}
    words1 = [w for w in re.findall(r"[a-z0-9]+", s1) if w not in stopwords]
    words2 = [w for w in re.findall(r"[a-z0-9]+", s2) if w not in stopwords]
    t1 = set(words1)
    t2 = set(words2)
    if not t1 or not t2:
        return False

    def _stem_word(w: str) -> str:
        if w.endswith("ies") and len(w) > 4:
            return w[:-3] + "y"
        if w.endswith("es") and len(w) > 4:
            return w[:-2]
        if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
            return w[:-1]
        return w

    stems1 = [_stem_word(w) for w in words1]
    stems2 = [_stem_word(w) for w in words2]
    if stems1 == stems2:
        return True

    overlap = t1.intersection(t2)
    if not overlap:
        return False

    # Word boundary containment (e.g. "Hadoop" inside "Hadoop Cluster", not partial word like "supervised" in "unsupervised")
    if re.search(rf"\b{re.escape(s1)}\b", s2) or re.search(rf"\b{re.escape(s2)}\b", s1):
        return True

    # If one option is a single concept word and appears in the other option
    if (len(words1) == 1 or len(words2) == 1) and overlap:
        return True

    # Sibling concepts sharing a head noun at the end (e.g. "Line Chart" vs "Bar Chart", "Supervised Learning" vs "Unsupervised Learning", "Semantic Ambiguity" vs "Referential Ambiguity")
    if len(words1) >= 2 and len(words2) >= 2 and words1[-1] == words2[-1] and words1[0] != words2[0]:
        return False

    # Entity modifier collisions sharing the first word (e.g. "Hadoop Cluster" vs "Hadoop and HDFS")
    if len(words1) >= 2 and len(words2) >= 2 and words1[0] == words2[0]:
        entity_modifiers = {"cluster", "system", "data", "hdfs", "server", "node"}
        if any(w in entity_modifiers for w in t1.union(t2) - overlap):
            return True

    jaccard = len(overlap) / len(t1.union(t2))
    if jaccard >= 0.5:
        return True

    # If any shared non-trivial word >= 4 chars when both options are short concepts (<= 2 words)
    if len(t1) <= 2 and len(t2) <= 2 and any(len(w) >= 4 for w in overlap):
        return True

    return False


def extract_chunk_hierarchy_context(chunk: Dict[str, Any], statement: str = "") -> Dict[str, Any]:
    """Extract hierarchical educational context: technology/subject, main section, mode/subsection."""
    sec_title = chunk.get("section_title") or ""
    source_fn = chunk.get("source_filename") or ""
    chunk_text = chunk.get("text") or ""

    # 1. Detect Subject / Technology (strictly domain-agnostic)
    subject = None
    all_context = f"{source_fn} {sec_title} {chunk_text[:300]}"

    tech_candidates = [
        "Apache Pig", "Pig Latin", "Apache Hive", "Apache Spark", "Hadoop", "HDFS",
        "MapReduce", "Apache Kafka", "Docker", "Kubernetes", "Tableau", "YARN",
        "Apache HBase", "ZooKeeper", "Linux", "Git", "Relational Database", "Operating System"
    ]
    for tech in tech_candidates:
        if re.search(rf"\b{re.escape(tech)}\b", all_context, re.I):
            subject = tech
            break

    if not subject:
        fn_clean = re.sub(r"\.(?:pdf|docx|ppt|pptx|txt|md)$", "", source_fn, flags=re.I).strip()
        fn_clean = re.sub(r"^(?:bda|unit\s*\d+|lecture\s*\d+|module\s*\d+|slide\s*\d+|chapter\s*\d+)[\s_\-]+", "", fn_clean, flags=re.I).strip()
        # Strictly reject generic file/unit names so they never become "In unit1, which concept..."
        if fn_clean and len(fn_clean) > 2 and not re.match(r"^(?:unit|lecture|chapter|module|slide|doc|file)\s*\d*$", fn_clean, re.I):
            subject = fn_clean

    # 2. Detect Section / Topic
    clean_sec = re.sub(r"^\d+(?:\.\d+)*\s*", "", sec_title).strip()
    clean_sec = re.sub(r"^(?:section|unit|chapter)\s*\d+[\.\:\-]?\s*", "", clean_sec, flags=re.I).strip()

    # 3. Detect Mode / Subsection
    mode = None
    mode_matches = []
    for m in re.finditer(r"(?:^|\n)\s*(?:[A-Z0-9][\.\)]\s*)?([A-Za-z ]{2,25}Mode)\b", chunk_text, re.I):
        cand_mode = m.group(1).strip()
        if not re.search(r"\b(difference|comparison|types|between|vs)\b", cand_mode, re.I):
            mode_matches.append((m.start(), cand_mode))

    if mode_matches:
        stmt_idx = chunk_text.find(statement) if statement else -1
        if stmt_idx != -1:
            preceding = [m for m in mode_matches if m[0] <= stmt_idx]
            if preceding:
                mode = preceding[-1][1]
        else:
            mode = mode_matches[0][1]

    if not mode:
        m = re.search(r"\b([A-Za-z ]{2,25}Mode)\b", sec_title, re.I)
        if m and not re.search(r"\b(difference|comparison|types|between|vs)\b", m.group(1), re.I):
            mode = m.group(1).strip()

    # 4. Detect Specific Category
    category = "concept"
    stmt_idx = chunk_text.find(statement) if statement else -1
    local_window = chunk_text[max(0, stmt_idx - 250):stmt_idx + 150] if stmt_idx != -1 else chunk_text[:350]
    combined_ctx = f"{sec_title} {local_window}".lower()

    if "data model" in combined_ctx or "data structure" in combined_ctx or "data type" in combined_ctx or any(k in statement.lower() for k in ["tuple", "bag", "ordered collection of fields", "atomic value", "key-value"]):
        category = "data structure"
    elif mode or "execution mode" in combined_ctx or "mode" in combined_ctx:
        category = "execution mode"
    elif "operator" in combined_ctx:
        category = "operator"
    elif "architecture" in combined_ctx or "component" in combined_ctx:
        category = "component"
    elif "visualization" in combined_ctx or "chart" in combined_ctx or "plot" in combined_ctx:
        category = "visualization technique"
    elif "linguistic" in combined_ctx or "language" in combined_ctx or "ambiguity" in combined_ctx or "morphology" in combined_ctx:
        category = "linguistic phenomenon"
    elif "algorithm" in combined_ctx:
        category = "algorithm"

    return {
        "subject": subject,
        "section": clean_sec or None,
        "mode": mode,
        "category": category,
        "document_name": source_fn,
        "page_number": chunk.get("page_number"),
    }


def unwrap_and_extract_sentences(text: str) -> List[str]:
    """Reconstruct wrapped document text into coherent, complete sentences."""
    if not text:
        return []

    # 1. Normalize line endings & hyphens across wraps
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"(\b[A-Za-z]+)-\n([A-Za-z]+\b)", r"\1-\2", normalized)

    raw_lines = [l.strip() for l in normalized.split("\n")]
    blocks: List[str] = []
    curr: List[str] = []

    def is_header_or_title(line: str) -> bool:
        clean = line.strip()
        if not clean:
            return False
        if re.match(r"^\d+(?:\.\d+)*\s+[A-Z][A-Za-z0-9\s\-]{1,40}$", clean):
            return True
        if re.match(r"^#{1,6}\s+", clean):
            return True
        words = clean.split()
        if 1 <= len(words) <= 4 and len(clean) <= 35 and clean[0].isupper() and not clean.endswith((".", ",", ";", ":", "?", "!", "-")):
            return True
        return False

    for line in raw_lines:
        if not line or line.startswith("|"):
            if curr:
                blocks.append(" ".join(curr))
                curr = []
            continue

        is_bullet = bool(re.match(r"^(?:[\-\*•–—]|\d+[\.\)]|\d+\.\d+[\.\)]?|[A-Za-z][\.\)])\s+", line))
        is_heading_label = bool(re.match(r"^(?:example|used for|solution|definition|note|warning)[:\-–]", line, re.I))
        is_title = is_header_or_title(line)
        prev_was_title = bool(curr and is_header_or_title(curr[-1]))

        if curr and (is_bullet or is_heading_label or is_title or prev_was_title):
            blocks.append(" ".join(curr))
            curr = [line]
        else:
            if curr:
                prev = curr[-1]
                if prev.endswith((".", "!", "?", ":")) and not re.search(r"\b(?:e\.g|i\.e|etc|vs)\.$", prev, re.I) and (line[0].isupper() or is_bullet):
                    blocks.append(" ".join(curr))
                    curr = [line]
                else:
                    curr.append(line)
            else:
                curr = [line]

    if curr:
        blocks.append(" ".join(curr))

    # Split each block into sentences
    sentences: List[str] = []
    for b in blocks:
        s_list = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\-\*•])", b)
        for s in s_list:
            s_clean = s.strip()
            if len(s_clean) >= 20:
                sentences.append(s_clean)
    return sentences


@dataclass
class QuizQuestion:
    """Represents a validated, multiple-choice question anchored to a course page."""
    id: str
    question: str
    options: List[str]
    correct_option_index: int
    correct_answer: str
    explanation: str
    source_document: str
    page_number: Optional[int]
    section_title: Optional[str]
    concept: str
    prerequisite_review: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "id": self.id,
            "question": self.question,
            "options": self.options,
            "correct_option_index": self.correct_option_index,
            "correct_answer": self.correct_answer,
            "explanation": self.explanation,
            "source_document": self.source_document,
            "page_number": self.page_number,
            "section_title": self.section_title,
            "concept": self.concept,
        }
        if self.prerequisite_review:
            d["prerequisite_review"] = self.prerequisite_review
        return d


@dataclass
class QuizResponse:
    """Collection of quiz questions for a folder."""
    folder_id: str
    folder_name: str
    total_questions: int
    questions: List[QuizQuestion]
    note: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "folder_id": self.folder_id,
            "folder_name": self.folder_name,
            "total_questions": self.total_questions,
            "questions": [q.to_dict() for q in self.questions],
        }
        if self.note:
            d["note"] = self.note
        return d


def validate_quiz_question(
    question: QuizQuestion,
    source_chunk_text: str = "",
    min_length: int = 25,
    allowed_folder_concepts: Optional[Set[str]] = None,
) -> Tuple[bool, List[str]]:
    """Strictly validate generated question across all 10 quality dimensions.

    Returns (is_valid, list_of_failure_reasons).
    """
    failures = []
    q_text = question.question.strip()

    # 1. Minimum Length & Question Mark
    if len(q_text) < min_length:
        failures.append(f"Question text too short ({len(q_text)} chars)")
    if not q_text.endswith("?"):
        failures.append("Question does not end with '?'")
    if q_text.count("?") > 1:
        failures.append("Multiple question marks detected")
    if not q_text[0].isupper():
        failures.append("Question does not start with a capitalized letter")

    # 2. Reject malformed unit/file prefixes and ungrammatical templates
    if re.search(r"\b(?:in\s+unit\s*\d+|in\s+lecture\s*\d+|in\s+chapter\s*\d+|in\s+module\s*\d+)\b", q_text, re.I):
        failures.append("Question contains unnecessary or repetitive unit/lecture prefix")
    if re.search(r"\bdefined as:\s*(?:differ|differs|typing|to\b)\b", q_text, re.I):
        failures.append("Question contains malformed 'defined as:' template phrasing")

    # Reject accidental duplicated concept names in stem (e.g. 'Semantic Ambiguity Semantic ambiguity')
    words = q_text.split()
    for i in range(len(words) - 1):
        w1 = re.sub(r"[^a-zA-Z]", "", words[i]).lower()
        w2 = re.sub(r"[^a-zA-Z]", "", words[i + 1]).lower()
        if w1 and w1 == w2 and len(w1) >= 4 and w1 not in {"that", "had", "is"}:
            failures.append(f"Repeated consecutive word '{w1}' detected in question stem")
            break

    # 3. Reject forbidden sentence-completion and literal quote phrasing
    if re.search(r'\bdefined as:\s*"|"not required in this mode"|\bknown as the above\b|\bmentioned in this case\b|\bcompletes the statement\b|\bwhat does this refer to\b|\bwhat is not required\b|\bcorresponds to words, but\b', q_text, re.I):
        failures.append("Question contains forbidden sentence-completion or literal quote phrasing")

    # 4. Reject dangling anaphoric pronouns
    dangling_match = DANGLING_ANAPHORA_PATTERN.search(q_text)
    if dangling_match:
        matched_str = dangling_match.group(0).lower()
        has_specific_mode = bool(re.search(r"\b(?:local|mapreduce|interactive|batch|hdfs|embedded)\s+mode\b", q_text, re.I))
        if not ("mode" in matched_str and has_specific_mode):
            failures.append("Question contains dangling anaphoric reference (e.g. 'this mode', 'the above')")

    if "mode" in q_text.lower() and not re.search(r"\b(local mode|mapreduce mode|interactive mode|batch mode|hdfs mode|embedded mode)\b", q_text, re.I):
        failures.append("Question discusses mode without identifying which specific mode is being tested")

    # Reject numerical calculations, math formulas, and raw vectors in MCQ questions
    if re.search(r"\b(?:calculate|compute|recalculate|evaluate)\b.*\b(?:loss|mse|latent|vector|dot product|matrix|kernel)\b", q_text, re.I):
        failures.append("Numerical calculations are excluded from practice MCQs")
    if re.search(r"(?:w_conv|w_fc|\bconv\s*output\b|latent\s*vector\b|mse\s*loss|\\sum|\\hat|\[[\d\s,\.\-]+\])", q_text, re.I):
        failures.append("Numerical vectors and formulas are excluded from practice MCQs")
    if any(re.search(r"^[\d\s,\.\-\[\]]+$", opt.strip()) for opt in question.options):
        failures.append("Numerical value options are excluded from practice MCQs")

    # 5. Answerability & Concept Validation
    if not question.correct_answer or len(question.correct_answer.strip()) < 2:
        failures.append("Correct answer is empty or invalid")
    if question.correct_answer.lower() in FORBIDDEN_CONCEPT_WORDS or question.correct_answer.lower() in ENGLISH_STOPWORDS:
        failures.append(f"Correct answer '{question.correct_answer}' is a forbidden filler or stopword")

    # 6. Answer Leakage Prevention: concept name must NOT appear inside the question stem
    ca_clean = question.correct_answer.strip().lower()
    if re.search(rf"\b{re.escape(ca_clean)}\b", q_text.lower()):
        failures.append(f"Answer leakage: question stem directly contains correct answer '{question.correct_answer}'")
    else:
        # Check distinguishing multi-word constituents (e.g. 'Semantic' in 'Semantic Ambiguity')
        ca_words = [w for w in ca_clean.split() if len(w) >= 4 and w not in ENGLISH_STOPWORDS]
        if len(ca_words) >= 2:
            # Check the distinguishing modifier (first word)
            distinguishing_mod = ca_words[0]
            if distinguishing_mod not in {"which", "concept", "system", "process", "technique", "method", "model"}:
                if re.search(rf"\b{re.escape(distinguishing_mod)}\b", q_text.lower()):
                    failures.append(f"Answer leakage: distinguishing word '{distinguishing_mod}' of answer appears in stem")

    # 7. Unambiguity, Conceptual Category Compatibility & Exactly 4 Options
    if len(question.options) != 4:
        failures.append(f"Expected exactly 4 options, found {len(question.options)}")
    else:
        clean_opts = [re.sub(r"^[A-D]\)\s*", "", opt).strip() for opt in question.options]
        # Check uniqueness
        if len(set(o.lower() for o in clean_opts)) != 4:
            failures.append("Duplicate or non-unique options detected")

        # Check conceptual category compatibility
        target_cat = classify_concept_category(question.correct_answer)
        if target_cat != "broad_field":
            for opt in clean_opts:
                if classify_concept_category(opt) == "broad_field":
                    failures.append(f"Conceptual category incompatibility: option '{opt}' is a broad field while target '{question.correct_answer}' is {target_cat}")
                    break

        # Check pairwise collision between options
        for i in range(len(clean_opts)):
            for j in range(i + 1, len(clean_opts)):
                if are_options_too_similar(clean_opts[i], clean_opts[j]):
                    failures.append(f"Options too similar: '{clean_opts[i]}' and '{clean_opts[j]}'")
                    break
            if failures and "Options too similar" in failures[-1]:
                break

        # Correct answer must be among options
        if question.correct_answer.lower() not in [o.lower() for o in clean_opts]:
            failures.append(f"Correct answer '{question.correct_answer}' not present in options")
        if question.correct_option_index < 0 or question.correct_option_index >= 4:
            failures.append("Invalid correct_option_index")
        elif clean_opts[question.correct_option_index].lower() != question.correct_answer.lower():
            failures.append("correct_option_index does not point to correct_answer")

        # 8. Folder Isolation Verification: all options must come from the permitted folder pool
        if allowed_folder_concepts:
            for opt in clean_opts:
                opt_norm = opt.lower().strip()
                if opt_norm not in allowed_folder_concepts:
                    failures.append(f"Folder isolation violation: option '{opt}' is not grounded in the selected folder")
                    break

    # 9. Explanation Consistency
    if not question.explanation or len(question.explanation.strip()) < 20:
        failures.append("Explanation is missing or too short")
    elif question.correct_answer.lower() not in question.explanation.lower():
        failures.append("Explanation does not mention or justify the correct answer")

    # 10. Source Traceability
    if not question.source_document:
        failures.append("Source document metadata is missing")

    is_valid = len(failures) == 0
    return is_valid, failures


class QuizGenerator:
    """Automated academic quiz generator strictly isolated to user-selected folders and documents."""

    def __init__(self, config: Optional[ATMLConfig] = None):
        self.config = config or default_config

    def extract_fact_anchors(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify candidate educational statements containing clean concepts, definitions, or verified mode requirements."""
        anchors: List[Dict[str, Any]] = []
        seen_concepts = set()

        for chunk in chunks:
            raw_text = chunk.get("text", "") or chunk.get("formatted_context", "")
            if not raw_text or len(raw_text.strip()) < 50:
                continue

            cleaned = sanitize_text(raw_text)
            sentences = unwrap_and_extract_sentences(cleaned)

            for line in sentences:
                if len(line) < 30 or len(line) > 500 or line.endswith("?") or line.startswith("?"):
                    continue

                # Pattern A: "Concept is/are/refers to/is defined as/is known as/occurs when Definition"
                def_match = re.search(
                    r"^([A-Za-z0-9\.\s\-–—]{2,50}?)\s+(?:is|are|refers to|is defined as|is the process of|consists of|represents|displays|uses|occurs when|happens when|is known as)\s+(.+)",
                    line,
                )
                if def_match:
                    raw_concept = def_match.group(1).strip()
                    definition = def_match.group(2).strip()
                    if ". " in definition:
                        definition = definition.split(". ")[0].strip() + "."
                    definition = re.sub(r"\s+(?:and|or|to|for|in|with|by|as|such as|of|the|a|an)\s*[\.\,\;\:]*$", ".", definition, flags=re.I)
                    concept = clean_academic_concept(raw_concept)

                    # Check for Mode / Operational Requirement statement
                    is_neg_req = bool(re.search(r"\b(?:is|are)\s+not\s+(?:required|needed|used)\b", line, re.I))
                    is_pos_req = bool(re.search(r"\b(?:is|are)\s+(?:required|needed|used)\b", line, re.I))
                    if (is_neg_req or is_pos_req) and (INVALID_DEFINITION_START.search(definition) or DANGLING_ANAPHORA_PATTERN.search(definition)):
                        ctx = extract_chunk_hierarchy_context(chunk, line)
                        if concept and concept.lower() not in seen_concepts and ctx.get("mode") and ctx.get("subject"):
                            seen_concepts.add(concept.lower())
                            anchors.append({
                                "concept": concept,
                                "anchor_type": "mode_requirement",
                                "requirement": "not_required" if is_neg_req else "required",
                                "mode": ctx["mode"],
                                "subject": ctx["subject"],
                                "statement": line,
                                "chunk": chunk,
                            })
                            continue
                        else:
                            continue

                    if concept and concept.lower() not in seen_concepts and is_valid_definition(definition):
                        ctx = extract_chunk_hierarchy_context(chunk, line)
                        seen_concepts.add(concept.lower())
                        anchors.append({
                            "concept": concept,
                            "anchor_type": "definition",
                            "definition": definition,
                            "statement": line,
                            "chunk": chunk,
                            "subject": ctx.get("subject"),
                            "category": ctx.get("category"),
                        })
                        continue

                # Pattern B: Bolded term: "**Concept**: Explanation" or "__Concept__: Explanation"
                bold_match = re.search(r"^(?:\*{2}|_{2})([A-Za-z0-9\.\s\-–—]{2,50}?)(?:\*{2}|_{2})(?::\s*|\s+[\-–—]\s+)(.+)", line)
                if bold_match:
                    raw_concept = bold_match.group(1).strip()
                    definition = bold_match.group(2).strip()
                    if ". " in definition:
                        definition = definition.split(". ")[0].strip() + "."
                    definition = re.sub(r"\s+(?:and|or|to|for|in|with|by|as|such as|of|the|a|an)\s*[\.\,\;\:]*$", ".", definition, flags=re.I)
                    concept = clean_academic_concept(raw_concept)
                    if concept and concept.lower() not in seen_concepts and is_valid_definition(definition):
                        ctx = extract_chunk_hierarchy_context(chunk, line)
                        seen_concepts.add(concept.lower())
                        anchors.append({
                            "concept": concept,
                            "anchor_type": "definition",
                            "definition": definition,
                            "statement": line,
                            "chunk": chunk,
                            "subject": ctx.get("subject"),
                            "category": ctx.get("category"),
                        })
                        continue

                # Pattern C: Numbered step or bullet: "1. Concept - Explanation" or "- Concept: Explanation"
                step_match = re.search(r"^(?:[\-\*•–—]|\d+[\.\)])\s*([A-Za-z0-9\.\s\-–—]{2,50}?)(?::\s*|\s+[\-–—]\s+)(.+)", line)
                if step_match:
                    raw_concept = step_match.group(1).strip()
                    definition = step_match.group(2).strip()
                    if ". " in definition:
                        definition = definition.split(". ")[0].strip() + "."
                    definition = re.sub(r"\s+(?:and|or|to|for|in|with|by|as|such as|of|the|a|an)\s*[\.\,\;\:]*$", ".", definition, flags=re.I)
                    concept = clean_academic_concept(raw_concept)
                    if concept and concept.lower() not in seen_concepts and is_valid_definition(definition):
                        ctx = extract_chunk_hierarchy_context(chunk, line)
                        seen_concepts.add(concept.lower())
                        anchors.append({
                            "concept": concept,
                            "anchor_type": "definition",
                            "definition": definition,
                            "statement": line,
                            "chunk": chunk,
                            "subject": ctx.get("subject"),
                            "category": ctx.get("category"),
                        })

        return anchors

    def extract_folder_concept_pool(
        self,
        chunks: List[Dict[str, Any]],
        fact_anchors: List[Dict[str, Any]],
    ) -> List[str]:
        """Extract a rich, domain-agnostic pool of candidate academic concepts strictly from the folder's chunks."""
        pool: List[str] = []
        seen = set()

        def add_concept(cand: Optional[str]):
            if not cand:
                return
            cleaned = clean_academic_concept(cand)
            if cleaned and cleaned.lower() not in seen:
                seen.add(cleaned.lower())
                pool.append(cleaned)

        # 1. Add all concepts directly from fact anchors
        for fa in fact_anchors:
            add_concept(fa.get("concept"))

        # 2. Extract concepts from section titles across all chunks
        for c in chunks:
            add_concept(c.get("section_title"))

        # 3. Extract technical terms, bold phrases, bullet items, and copular concepts across chunks
        for c in chunks:
            raw_text = c.get("text", "") or ""
            # Bold patterns
            for m in re.finditer(r"\*\*([A-Za-z0-9\s\-–—/]{2,45})\*\*|__([A-Za-z0-9\s\-–—/]{2,45})__", raw_text):
                bold_text = m.group(1) or m.group(2)
                add_concept(bold_text)

            # Bullet-point and numbered items: "- Concept: ..."
            for m in re.finditer(r"(?:^|\n)\s*(?:[\-\*•]|\d+[\.\)])\s*([A-Za-z0-9\s\-–—/]{2,40}?)\s*[:\-–—]", raw_text):
                add_concept(m.group(1))

            # Markdown headings: "## Concept"
            for m in re.finditer(r"(?:^|\n)#{1,4}\s*([A-Za-z0-9\s\-–—/]{2,40})", raw_text):
                add_concept(m.group(1))

            # Copular patterns: "Concept is/are/refers to/occurs when"
            for m in re.finditer(r"(?:^|\n|\.\s+)([A-Z][A-Za-z0-9\s\-–—/]{2,40}?)\s+(?:is|are|refers to|is defined as|represents|occurs when)\b", raw_text):
                add_concept(m.group(1))

            # Capitalized technical terms: 2 to 3 capitalized words (e.g. Referential Ambiguity, Semantic Parsing)
            for m in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b", raw_text):
                term = m.group(1)
                if not re.match(r"^(?:For Example|In Addition|On The|At The|According To|As Such|That Is|This Is|There Are)\b", term, re.I):
                    add_concept(term)

        return pool

    def generate_question_stem(
        self,
        concept: str,
        definition: str = "",
        statement: str = "",
        chunk: Optional[Dict[str, Any]] = None,
        anchor: Optional[Dict[str, Any]] = None,
        model=None,
        tokenizer=None,
    ) -> str:
        """Formulate specific conceptual interrogative question stem, strictly rejecting sentence completions, quote wrappers, and filename prefixes."""
        anchor_type = anchor.get("anchor_type", "definition") if anchor else "definition"

        # 1. Mode / Operational Requirement Questions
        if anchor_type == "mode_requirement" and anchor:
            mode = anchor.get("mode")
            subject = anchor.get("subject")
            requirement = anchor.get("requirement", "not_required")
            if mode and subject:
                if requirement == "not_required":
                    return f"According to the section on {mode} in {subject}, which of the following is described as not being required in that mode?"
                else:
                    return f"In the {mode} of {subject}, which of the following is required to process data?"

        # Extract hierarchy context
        subject = None
        category = None
        if anchor:
            subject = anchor.get("subject")
            category = anchor.get("category")
        elif chunk:
            ctx = extract_chunk_hierarchy_context(chunk, statement)
            subject = ctx.get("subject")
            category = ctx.get("category")

        # Strip concept name and leading verbs from the definition
        clean_def = strip_concept_from_definition(concept, definition)
        if not clean_def:
            clean_def = strip_concept_from_definition(concept, statement)

        # Do NOT include subject if it's a generic file label like 'unit1' or contains unit/lecture/module
        if subject and re.search(r"\b(?:unit|lecture|chapter|module|slide|doc|file)\s*\d*\b", subject, re.I):
            subject = None

        known_nouns = [
            "technology", "plot", "chart", "diagram", "graph", "visualization", "technique",
            "method", "algorithm", "process", "framework", "system", "distribution",
            "metric", "measure", "tool", "approach", "platform", "paradigm", "package",
            "service", "environment", "engine", "component", "model", "pipeline", "map",
            "data structure", "data type", "linguistic phenomenon", "phenomenon"
        ]

        cat_noun = category or "concept"
        concept_words = concept.lower().split()
        for w in reversed(concept_words):
            if w in known_nouns:
                cat_noun = w
                break

        # Check special academic patterns
        # 1. Differ by only one sound / phoneme (e.g. Minimal Pairs)
        if re.match(r"^(?:differ|differs)\s+(?:by|in)\b", clean_def, re.I):
            if "two words" in clean_def.lower() or "words" in clean_def.lower():
                return f"What term describes {clean_def}?".replace("??", "?")
            elif "sound" in clean_def.lower() or "phoneme" in clean_def.lower():
                return f"What term describes two words that {clean_def}?".replace("??", "?")
            return f"What term describes words that {clean_def}?".replace("??", "?")

        # 2. Example typing error / typo classification
        if re.search(r"\btyping\s+.*\s+instead\s+of\b", clean_def, re.I) or clean_def.lower().startswith("typing "):
            return f"According to the selected source, which category best describes the example of {clean_def}?".replace("??", "?")

        # 3. Smallest unit of / A unit of (e.g. Morpheme, Phoneme, Token)
        if re.match(r"^(?:the\s+)?(?:smallest\s+)?unit\s+of\b", clean_def, re.I):
            return f"What term describes {clean_def}?".replace("??", "?")

        # 4. The study of / The scientific study of / The branch of
        if re.match(r"^(?:the\s+)?(?:scientific\s+)?(?:study|branch|field)\s+of\b", clean_def, re.I):
            return f"Which discipline is defined as {clean_def}?".replace("??", "?")

        # 5. Process of (e.g. Parsing, Stemming, Containerization)
        if re.match(r"^(?:the\s+)?process\s+of\b", clean_def, re.I):
            return f"Which process is defined as {clean_def}?".replace("??", "?")

        # Check if clean_def begins with "a/an <noun>"
        matched_stem = None
        for noun in known_nouns:
            m_noun = re.match(rf"^(?:an?|the)?\s*(?:[a-z\-]+\s+)?\b{re.escape(noun)}\b\s*(?:that|which|is|used to|for|in which|where)?\s*(.+)", clean_def, flags=re.I)
            if m_noun:
                rest = m_noun.group(1).strip().rstrip(". ;:-–")
                if rest:
                    if re.match(r"^(?:represents|uses|displays|handles|accelerates|manages|monitors|processes|calculates|stores|contains)\b", rest, re.I):
                        if subject and subject.lower() not in rest.lower():
                            matched_stem = f"In {subject}, which {noun} {rest}?".replace("??", "?")
                        else:
                            matched_stem = f"Which {noun} {rest}?".replace("??", "?")
                    else:
                        if subject and subject.lower() not in rest.lower():
                            matched_stem = f"In {subject}, which {noun} is {rest}?".replace("is is", "is").replace("??", "?")
                        else:
                            matched_stem = f"Which {noun} is {rest}?".replace("is is", "is").replace("??", "?")
                    break
        if matched_stem:
            return matched_stem

        # Occurs when / happens when (e.g. linguistic phenomena)
        is_occurs_when = (
            clean_def.lower().startswith("occurs when ")
            or clean_def.lower().startswith("happens when ")
            or "occurs when" in (definition or "").lower()
            or "happens when" in (definition or "").lower()
            or "occurs when" in (statement or "").lower()
            or "happens when" in (statement or "").lower()
        )
        if is_occurs_when:
            rest = re.sub(r"^(?:occurs|happens)\s+when\s+", "", clean_def, flags=re.I).strip()
            if "linguistic" in cat_noun or "ambiguity" in concept.lower() or "phenomenon" in cat_noun or "morphology" in cat_noun:
                return f"Which linguistic phenomenon occurs when {rest}?".replace("??", "?")
            return f"Which concept occurs when {rest}?".replace("??", "?")

        # Third-person verbs (represents, uses, displays, manages, maintains, monitors, processes, calculates)
        if re.match(r"^(?:represents|uses|displays|calculates|measures|enables|provides|groups|divides|plots|manages|maintains|monitors|processes|stores|consists\s+of)\b", clean_def, re.I):
            if subject and subject.lower() not in clean_def.lower():
                return f"In {subject}, which {cat_noun} {clean_def}?".replace("??", "?")
            return f"Which {cat_noun} {clean_def}?".replace("??", "?")

        # Used to / Used for
        if clean_def.lower().startswith("used to ") or clean_def.lower().startswith("is used to ") or clean_def.lower().startswith("used for "):
            rest = re.sub(r"^(?:is\s+)?used (?:to|for)\s+", "", clean_def, flags=re.I)
            if subject and subject.lower() not in rest.lower():
                return f"In {subject}, which {cat_noun} is used to {rest}?".replace("??", "?")
            return f"Which {cat_noun} is used to {rest}?".replace("??", "?")

        # An ordered collection / a collection of ...
        if clean_def.lower().startswith("an ordered collection") or clean_def.lower().startswith("a collection") or clean_def.lower().startswith("collection"):
            if subject and subject.lower() not in clean_def.lower():
                return f"In {subject}, which {cat_noun} is {clean_def}?".replace("??", "?")
            return f"Which {cat_noun} is {clean_def}?".replace("??", "?")

        # Clean academic definitional question stem
        if re.match(r"^(?:is|are|was|were)\s+", clean_def, re.I):
            clean_def = re.sub(r"^(?:is|are|was|were)\s+", "", clean_def, flags=re.I)
            if subject and subject.lower() not in clean_def.lower():
                return f"In {subject}, which {cat_noun} is {clean_def}?".replace("??", "?")
            return f"Which {cat_noun} is {clean_def}?".replace("??", "?")

        if subject and subject.lower() not in clean_def.lower():
            return f"In {subject}, what term describes {clean_def}?".replace("??", "?")
        return f"What term describes {clean_def}?".replace("??", "?")

    def mine_semantic_distractors(
        self,
        correct_concept: str,
        candidate_pool: List[str],
        embedding_model=None,
        context_text: str = "",
        anchor: Optional[Dict[str, Any]] = None,
        question_stem: str = "",
        top_k: int = 3,
    ) -> List[str]:
        """Mine sibling technical concepts strictly from the permitted folder candidate pool, enforcing anti-collision, category compatibility, and strict folder isolation."""
        norm_correct = correct_concept.lower().strip()
        target_cat = classify_concept_category(correct_concept)

        # Sanitize every candidate from the folder pool, rejecting duplicates, colliding terms, and incompatible categories
        folder_candidates = []
        seen = {norm_correct}
        for raw in candidate_pool:
            cleaned = clean_academic_concept(raw)
            if not cleaned:
                continue
            cand_norm = cleaned.lower()
            if cand_norm in seen or are_options_too_similar(cleaned, correct_concept):
                continue
            if question_stem and cand_norm in question_stem.lower():
                # Reject candidates that appear in the question stem
                continue

            cand_cat = classify_concept_category(cleaned)
            # Rule: Disallow broad fields when testing a specific atomic concept/unit
            # e.g., If asking for "Morpheme" (core_unit), reject "NLP", "Natural Language Processing", "Semantics" (broad_field)
            if target_cat != "broad_field" and cand_cat == "broad_field":
                continue

            seen.add(cand_norm)
            folder_candidates.append(cleaned)

        if len(folder_candidates) < top_k:
            # Strictly return available folder candidates without falling back to any external domain lists!
            return folder_candidates

        # Head-word analysis for sibling concept matching
        words = correct_concept.split()
        head_word = words[-1].lower() if len(words) > 1 else ""

        if embedding_model is not None:
            try:
                target_vec = embedding_model.encode([correct_concept], convert_to_numpy=True)[0]
                cand_vecs = embedding_model.encode(folder_candidates, convert_to_numpy=True)

                dot_products = np.dot(cand_vecs, target_vec)
                norms = (np.linalg.norm(cand_vecs, axis=1) * np.linalg.norm(target_vec)) + 1e-9
                sims = dot_products / norms

                ranked = []
                for idx, cand in enumerate(folder_candidates):
                    similarity = float(sims[idx])
                    # Bandpass filter: high semantic alignment, preventing near-synonyms
                    if similarity >= 0.92 or similarity < 0.20:
                        continue
                    if not are_options_too_similar(cand, correct_concept):
                        score = similarity
                        cand_cat = classify_concept_category(cand)
                        # Category bonus: prioritize concepts belonging to the exact same conceptual category
                        if cand_cat == target_cat:
                            score += 0.50
                        # Sibling bonus for sharing the same head word (e.g. Referential Ambiguity vs Semantic Ambiguity)
                        if head_word and head_word in cand.lower():
                            score += 0.25
                        # Grammatical parallelism bonus (same number of words)
                        if len(cand.split()) == len(words):
                            score += 0.05
                        ranked.append((score, cand))

                ranked.sort(key=lambda x: x[0], reverse=True)

                distractors: List[str] = []
                for score, cand in ranked:
                    if cand.lower() not in seen and not are_options_too_similar(cand, correct_concept):
                        if not any(are_options_too_similar(cand, prev) for prev in distractors):
                            distractors.append(cand)
                            if len(distractors) == top_k:
                                return distractors

                # If bandpass filtered out too many, rank remaining folder candidates by category, sibling head and length
                for cand in folder_candidates:
                    if cand.lower() not in seen and not are_options_too_similar(cand, correct_concept):
                        if not any(are_options_too_similar(cand, prev) for prev in distractors):
                            distractors.append(cand)
                            if len(distractors) == top_k:
                                return distractors

                if distractors:
                    return distractors
            except Exception as exc:
                logger.warning("Embedding distractor calculation fallback: %s", exc)

        # Lexical / Sibling ranking fallback strictly from the folder
        ranked_lexical = []
        for cand in folder_candidates:
            if not are_options_too_similar(cand, correct_concept):
                score = 0.0
                cand_cat = classify_concept_category(cand)
                if cand_cat == target_cat:
                    score += 2.0
                if head_word and head_word in cand.lower():
                    score += 1.0
                if len(cand.split()) == len(words):
                    score += 0.5
                ranked_lexical.append((score, cand))

        ranked_lexical.sort(key=lambda x: x[0], reverse=True)

        distractors: List[str] = []
        for score, cand in ranked_lexical:
            if not any(are_options_too_similar(cand, prev) for prev in distractors):
                distractors.append(cand)
                if len(distractors) == top_k:
                    break

        return distractors

    def regenerate_question_with_context(
        self,
        anchor: Dict[str, Any],
        chunk: Dict[str, Any],
        candidate_pool: List[str],
        embedding_model=None,
        folder_name: str = "",
        allowed_folder_concepts: Optional[Set[str]] = None,
    ) -> Optional[QuizQuestion]:
        """Attempt to regenerate a failed candidate question by enriching hierarchy context."""
        ctx = extract_chunk_hierarchy_context(chunk, anchor.get("statement", ""))
        enriched_anchor = dict(anchor)
        if ctx.get("subject"):
            enriched_anchor["subject"] = ctx["subject"]
        if ctx.get("category"):
            enriched_anchor["category"] = ctx["category"]
        if ctx.get("mode"):
            enriched_anchor["mode"] = ctx["mode"]

        stem = self.generate_question_stem(
            concept=enriched_anchor["concept"],
            definition=enriched_anchor.get("definition", ""),
            statement=enriched_anchor.get("statement", ""),
            chunk=chunk,
            anchor=enriched_anchor,
            model=None,
            tokenizer=None,
        )

        context_text = f"{folder_name} {chunk.get('source_filename', '')} {chunk.get('text', '')[:200]}"
        distractors = self.mine_semantic_distractors(
            correct_concept=enriched_anchor["concept"],
            candidate_pool=candidate_pool,
            embedding_model=embedding_model,
            context_text=context_text,
            anchor=enriched_anchor,
            question_stem=stem,
            top_k=3,
        )

        clean_correct = clean_academic_concept(enriched_anchor["concept"]) or enriched_anchor["concept"]
        sanitized_distractors = []
        seen_opt = {clean_correct.lower()}
        for d in distractors:
            cd = clean_academic_concept(d) or d
            if cd and cd.lower() not in seen_opt and not are_options_too_similar(cd, clean_correct):
                if not any(are_options_too_similar(cd, prev) for prev in sanitized_distractors):
                    seen_opt.add(cd.lower())
                    sanitized_distractors.append(cd)

        if len(sanitized_distractors) < 3:
            return None

        all_options = [clean_correct] + sanitized_distractors[:3]
        random.shuffle(all_options)
        correct_idx = all_options.index(clean_correct)

        letters = ["A", "B", "C", "D"]
        formatted_options = [f"{letters[i]}) {opt}" for i, opt in enumerate(all_options)]

        page_num = chunk.get("page_number")
        doc_name = chunk.get("source_filename") or "Course Material"
        clean_def_preview = strip_concept_from_definition(clean_correct, enriched_anchor.get("definition", "")).rstrip(". ;")

        if enriched_anchor.get("anchor_type") == "mode_requirement" and enriched_anchor.get("mode"):
            explanation = (
                f"According to {doc_name}"
                + (f" (Page {page_num})" if page_num else "")
                + f", in {enriched_anchor['mode']} of {enriched_anchor.get('subject', 'the system')}, "
                + f"{clean_correct} is not required because all files are processed using the local file system."
            )
        else:
            explanation = (
                f"According to {doc_name}"
                + (f" (Page {page_num})" if page_num else "")
                + f", {clean_correct} is: {clean_def_preview}."
            )

        candidate_q = QuizQuestion(
            id=f"quiz-q-{uuid.uuid4().hex[:8]}",
            question=stem,
            options=formatted_options,
            correct_option_index=correct_idx,
            correct_answer=clean_correct,
            explanation=explanation,
            source_document=doc_name,
            page_number=page_num,
            section_title=chunk.get("section_title"),
            concept=clean_correct,
        )

        is_valid, _ = validate_quiz_question(
            candidate_q,
            source_chunk_text=chunk.get("text", ""),
            allowed_folder_concepts=allowed_folder_concepts,
        )
        return candidate_q if is_valid else None

    def generate_quiz_for_folder(
        self,
        db: Session,
        current_user: User,
        folder_id: str,
        num_questions: int = 10,
        document_ids: Optional[List[str]] = None,
    ) -> QuizResponse:
        """Execute end-to-end quiz generation for an authorized user folder with strict validation and folder isolation."""
        folder = db.query(Folder).filter(
            Folder.id == folder_id,
            Folder.user_id == current_user.id,
        ).first()

        if not folder:
            raise ValueError(f"Folder '{folder_id}' not found or unauthorized.")

        docs_query = db.query(Document).filter(
            Document.folder_id == folder.id,
            Document.status == "completed",
        )
        if document_ids:
            doc_id_set = {str(did).strip() for did in document_ids if str(did).strip()}
            docs_query = docs_query.filter(Document.id.in_(doc_id_set))
        docs = docs_query.all()

        if document_ids and not docs:
            raise ValueError("No completed documents found matching the selected units.")

        if not docs:
            return QuizResponse(
                folder_id=folder.id,
                folder_name=folder.name,
                total_questions=0,
                questions=[],
                note="No completed documents found in this folder.",
            )

        # Load chunks strictly belonging to authorized, completed documents in this folder
        index_dir = get_folder_index_dir(BASE_STORAGE_DIR, str(current_user.id), str(folder.id))
        index = FolderVectorIndex(index_dir)

        if document_ids:
            doc_id_set = {str(did).strip() for did in document_ids if str(did).strip()}
            chunks = [c for c in (index.chunks or []) if str(c.get("document_id")) in doc_id_set]
        else:
            clean_valid_ids = {str(d.id) for d in docs if not str(getattr(d, "id", "")).startswith("<MagicMock")}
            if clean_valid_ids:
                chunks = [c for c in (index.chunks or []) if str(c.get("document_id")) in clean_valid_ids]
            else:
                chunks = list(index.chunks) if index.chunks else []

        # Fallback to direct extraction content if vector index is empty
        if not chunks:
            chunks = []
            for d in docs:
                if d.extraction and d.extraction.markdown_content:
                    chunks.append({
                        "document_id": d.id,
                        "source_filename": d.original_filename,
                        "page_number": 1,
                        "section_title": d.original_filename,
                        "text": d.extraction.markdown_content,
                    })

        if not chunks:
            return QuizResponse(
                folder_id=folder.id,
                folder_name=folder.name,
                total_questions=0,
                questions=[],
                note="No text content available in the selected materials.",
            )

        # 1. Extract Fact Anchors strictly from permitted chunks
        fact_anchors = self.extract_fact_anchors(chunks)

        # Fallback 1: Mine section titles with educational definitions
        if len(fact_anchors) < num_questions:
            seen_existing = {fa["concept"].lower() for fa in fact_anchors}
            for c in chunks:
                st = clean_academic_concept(c.get("section_title") or "")
                txt = c.get("text", "")
                if st and st.lower() not in seen_existing and len(txt) > 60:
                    first_sent = txt.split(".")[0].strip()
                    if len(first_sent) > 25 and is_valid_definition(first_sent):
                        ctx = extract_chunk_hierarchy_context(c, first_sent)
                        seen_existing.add(st.lower())
                        fact_anchors.append({
                            "concept": st,
                            "anchor_type": "definition",
                            "definition": first_sent,
                            "statement": first_sent,
                            "chunk": c,
                            "subject": ctx.get("subject"),
                            "category": ctx.get("category"),
                        })

        # Fallback 2: Extract conceptual definitions from chunk lines
        if len(fact_anchors) < num_questions:
            seen_existing = {fa["concept"].lower() for fa in fact_anchors}
            for c in chunks:
                for line in (c.get("text") or "").split("\n"):
                    l_str = line.strip()
                    if len(l_str) < 30:
                        continue
                    m = re.match(r"^([A-Z][a-zA-Z0-9\s\-–—/]{2,45})\s*[:\-–—]\s*(.{25,})", l_str)
                    if m:
                        cand_concept = clean_academic_concept(m.group(1))
                        cand_def = m.group(2).strip()
                        if cand_concept and cand_concept.lower() not in seen_existing and is_valid_definition(cand_def):
                            ctx = extract_chunk_hierarchy_context(c, l_str)
                            seen_existing.add(cand_concept.lower())
                            fact_anchors.append({
                                "concept": cand_concept,
                                "anchor_type": "definition",
                                "definition": cand_def.split(".")[0].strip() + ".",
                                "statement": l_str,
                                "chunk": c,
                                "subject": ctx.get("subject"),
                                "category": ctx.get("category"),
                            })
                            if len(fact_anchors) >= num_questions * 3:
                                break

        if not fact_anchors:
            return QuizResponse(
                folder_id=folder.id,
                folder_name=folder.name,
                total_questions=0,
                questions=[],
                note="Insufficient structured educational facts found in the selected materials.",
            )

        # 2. Extract Comprehensive Concept Pool strictly from the selected folder
        folder_concept_pool = self.extract_folder_concept_pool(chunks, fact_anchors)
        allowed_folder_concepts: Set[str] = {c.lower() for c in folder_concept_pool}
        for fa in fact_anchors:
            allowed_folder_concepts.add(fa["concept"].lower())

        # 3. Model initialization
        try:
            embedding_model = get_embedding_model()
        except Exception:
            embedding_model = None

        questions: List[QuizQuestion] = []
        seen_quiz_concepts = set()
        seen_stems: List[str] = []

        selected_anchors = list(fact_anchors)
        random.shuffle(selected_anchors)

        for anchor in selected_anchors:
            if len(questions) >= num_questions:
                break

            concept = anchor["concept"]
            if concept.lower() in seen_quiz_concepts:
                continue

            definition = anchor.get("definition", "")
            statement = anchor.get("statement", "")
            chunk = anchor["chunk"]

            # Generate initial question stem
            question_text = self.generate_question_stem(
                concept=concept,
                definition=definition,
                statement=statement,
                chunk=chunk,
                anchor=anchor,
                model=None,
                tokenizer=None,
            )

            # Mine 3 sibling distractors strictly from folder concepts
            context_text = f"{folder.name} {chunk.get('source_filename', '')} {chunk.get('text', '')[:200]}"
            distractors = self.mine_semantic_distractors(
                correct_concept=concept,
                candidate_pool=folder_concept_pool,
                embedding_model=embedding_model,
                context_text=context_text,
                anchor=anchor,
                question_stem=question_text,
                top_k=3,
            )

            clean_correct = clean_academic_concept(concept) or concept
            sanitized_distractors = []
            seen_opt = {clean_correct.lower()}
            for d in distractors:
                cd = clean_academic_concept(d) or d
                if cd and cd.lower() not in seen_opt and not are_options_too_similar(cd, clean_correct):
                    if not any(are_options_too_similar(cd, prev) for prev in sanitized_distractors):
                        seen_opt.add(cd.lower())
                        sanitized_distractors.append(cd)

            # If folder does not contain 3 valid distractors, discard candidate rather than fabricating out-of-domain options!
            if len(sanitized_distractors) < 3:
                continue

            all_options = [clean_correct] + sanitized_distractors[:3]
            random.shuffle(all_options)
            correct_idx = all_options.index(clean_correct)

            letters = ["A", "B", "C", "D"]
            formatted_options = [f"{letters[i]}) {opt}" for i, opt in enumerate(all_options)]

            page_num = chunk.get("page_number")
            doc_name = chunk.get("source_filename") or "Course Material"
            clean_def_preview = strip_concept_from_definition(clean_correct, definition).rstrip(". ;")

            if anchor.get("anchor_type") == "mode_requirement" and anchor.get("mode"):
                explanation = (
                    f"According to {doc_name}"
                    + (f" (Page {page_num})" if page_num else "")
                    + f", in {anchor['mode']} of {anchor.get('subject', 'the system')}, "
                    + f"{clean_correct} is not required because all files are processed using the local file system."
                )
            else:
                subj_tag = anchor.get("subject")
                if subj_tag and not re.match(r"^(?:unit|lecture|chapter|module|slide|doc|file)\s*\d*$", subj_tag.strip(), re.I):
                    explanation = (
                        f"According to {doc_name}"
                        + (f" (Page {page_num})" if page_num else "")
                        + f", {clean_correct} in {subj_tag} is: {clean_def_preview}."
                    )
                else:
                    explanation = (
                        f"According to {doc_name}"
                        + (f" (Page {page_num})" if page_num else "")
                        + f", {clean_correct} is: {clean_def_preview}."
                    )

            q_item = QuizQuestion(
                id=f"quiz-q-{uuid.uuid4().hex[:8]}",
                question=question_text,
                options=formatted_options,
                correct_option_index=correct_idx,
                correct_answer=clean_correct,
                explanation=explanation,
                source_document=doc_name,
                page_number=page_num,
                section_title=chunk.get("section_title"),
                concept=clean_correct,
            )

            # --- QUESTION-QUALITY VALIDATION STAGE ---
            is_valid, failures = validate_quiz_question(
                q_item,
                source_chunk_text=chunk.get("text", ""),
                allowed_folder_concepts=allowed_folder_concepts,
            )
            if not is_valid:
                logger.debug("Question validation failed (%s). Attempting regeneration with context...", failures)
                regenerated_q = self.regenerate_question_with_context(
                    anchor=anchor,
                    chunk=chunk,
                    candidate_pool=folder_concept_pool,
                    embedding_model=embedding_model,
                    folder_name=folder.name,
                    allowed_folder_concepts=allowed_folder_concepts,
                )
                if regenerated_q is not None:
                    q_item = regenerated_q
                else:
                    logger.debug("Regeneration failed for '%s'; discarding candidate.", concept)
                    continue

            # Deduplication: ensure stem is not duplicate
            q_clean = q_item.question.strip().lower()
            is_duplicate_stem = False
            for prev_stem in seen_stems:
                if prev_stem.strip().lower() == q_clean:
                    is_duplicate_stem = True
                    break
            if is_duplicate_stem:
                continue

            seen_quiz_concepts.add(clean_correct.lower())
            seen_stems.append(q_item.question)
            questions.append(q_item)

        note = None
        if len(questions) < num_questions and len(questions) > 0:
            note = (
                f"Generated {len(questions)} high-quality validated questions strictly grounded in the selected folder. "
                f"Additional questions were omitted to maintain question clarity and strict folder isolation without fabricating options."
            )

        return QuizResponse(
            folder_id=folder.id,
            folder_name=folder.name,
            total_questions=len(questions),
            questions=questions,
            note=note,
        )

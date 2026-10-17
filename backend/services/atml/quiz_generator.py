"""Inverted Sequence-to-Sequence Question Generation & Vector-Manifold Distractor Optimization.

ATML Component for Nalanda AI:
1. Educational Fact-Anchor Mining:
   Identifies self-contained concepts, stripping slide numbers (e.g., '3.7 Box Plot' -> 'Box Plot')
   and rejecting incomplete grammatical fragments (e.g. 'When the data').
2. Inverted Seq2Seq Question Formulation: P(Question | Context, Target Concept)
   using pretrained FLAN-T5, strictly rejecting meta/title-matching prompts.
3. Latent Vector Manifold Distractor Optimization:
   Mines plausible sibling distractors from dense embeddings (all-MiniLM-L6-v2)
   within an optimal semantic plausibility margin (0.40 <= cosine <= 0.85).
4. Grounding & Page Citation Verification:
   Ensures the correct answer is factually anchored to the source text and exact page.
"""
import logging
import random
import re
import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

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
from .answer_refiner import get_atml_model_and_tokenizer
from .output_validator import sanitize_text

logger = logging.getLogger("nalanda.atml.quiz")

# Multi-domain academic distractor libraries mapped by course topic
DOMAIN_DISTRACTOR_MAP = {
    "cloud": [
        "Virtual Machine", "Docker Container", "Docker Image", "Hypervisor", "Bare-Metal Server",
        "Kubernetes Pod", "Container Image", "Host Operating System", "Guest OS",
        "Namespace Isolation", "Docker Engine", "Microservices Architecture", "Container Runtime",
        "Dockerfile", "Docker Registry", "Docker Hub", "Port Binding", "Docker Compose",
        "Control Groups (cgroups)", "Container Orchestration", "Serverless Computing"
    ],
    "bigdata": [
        "MapReduce", "HDFS NameNode", "Spark RDD", "YARN Resource Manager",
        "Apache Kafka", "Hadoop DataNode", "HBase", "Apache Hive", "Zookeeper", "Distributed File System",
        "Apache Spark", "Block Storage", "Data Replication", "Apache Flink", "Columnar Storage"
    ],
    "visualization": [
        "Scatter Plot", "Box Plot", "Histogram", "Line Chart", "Bar Chart",
        "Heat Map", "Radar Chart", "Pie Chart", "Violin Plot", "Normal Distribution",
        "Skewed Distribution", "Data Visualization", "Data Profiling", "Treemap", "Density Plot"
    ],
    "ml": [
        "Supervised Learning", "Unsupervised Learning", "Reinforcement Learning",
        "Decision Tree", "Random Forest", "Support Vector Machine", "Gradient Descent",
        "Neural Network", "Feature Engineering", "Loss Function", "Logistic Regression",
        "Principal Component Analysis", "Cross Validation"
    ],
    "networks": [
        "Packet Switching", "Circuit Switching", "Routing Protocol", "TCP/IP Protocol",
        "OSI Model", "Subnet Mask", "DNS Resolution", "DHCP Server", "Network Gateway",
        "Firewall Rules", "Network Topology", "MAC Address", "Transport Layer", "Application Layer"
    ],
    "os": [
        "Virtual Memory", "CPU Scheduling", "Deadlock Prevention", "Mutual Exclusion",
        "Process Synchronization", "Page Replacement", "Interrupt Handling", "Context Switching",
        "Kernel Space", "File System Architecture", "Thread Pool", "System Call"
    ],
    "dbms": [
        "Relational Model", "Entity Relationship", "B-Tree Indexing", "ACID Properties",
        "Normalization Form", "Foreign Key Constraint", "Database Transaction", "Query Optimization",
        "Concurrency Control", "Stored Procedure", "Schema Definition"
    ],
    "security": [
        "Public Key Cryptography", "Symmetric Encryption", "Digital Signature", "Access Control List",
        "Transport Layer Security", "Hash Function", "Authentication Protocol", "Privilege Escalation",
        "Intrusion Detection", "Zero Trust Architecture"
    ],
    "bigdata_runtime": [
        "Local File System", "Java Virtual Machine", "Grunt Shell", "Standalone Client",
        "Pig Engine", "Embedded Script", "Single Machine Storage", "HDFS NameNode",
        "YARN NodeManager", "MapReduce JobTracker"
    ],
    "software": [
        "Agile Methodology", "Scrum Framework", "Continuous Integration", "Design Pattern",
        "Object-Oriented Programming", "Unit Testing", "Test-Driven Development", "Code Refactoring",
        "Microservices Architecture", "API Integration"
    ],
}

CURATED_DOMAIN_DISTRACTORS = (
    DOMAIN_DISTRACTOR_MAP["cloud"]
    + DOMAIN_DISTRACTOR_MAP["bigdata"]
    + DOMAIN_DISTRACTOR_MAP["bigdata_runtime"]
    + DOMAIN_DISTRACTOR_MAP["visualization"]
    + DOMAIN_DISTRACTOR_MAP["os"]
    + DOMAIN_DISTRACTOR_MAP["networks"]
    + DOMAIN_DISTRACTOR_MAP["dbms"]
)

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
    "actual table data", "job.name"
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
    "query", "tuple", "relation", "deadlock", "atomicity", "consistency", "isolation", "durability",
    # Data Structures & Algorithms
    "recursion", "sorting", "hashing", "stack", "queue", "graph", "tree", "heap",
    "array", "pointer", "traversal", "search", "backtracking",
    # Cyber Security & Cryptography
    "encryption", "decryption", "cipher", "cryptography", "malware",
    "phishing", "authentication", "authorization", "hash", "keyloggers", "trojan",
    # Software Engineering & Languages
    "polymorphism", "inheritance", "encapsulation", "abstraction", "compiler",
    "interpreter", "bytecode", "refactoring", "scrum", "waterfall", "agile"
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
    words = clean.split()
    if len(words) < 4:
        return False
    # Reject isolated negative predicates (e.g. "not required in this mode")
    if INVALID_DEFINITION_START.search(clean):
        return False
    # Reject dangling anaphoric pronouns (e.g. "in this mode", "the above")
    if DANGLING_ANAPHORA_PATTERN.search(clean):
        return False
    return True


def are_options_too_similar(opt1: str, opt2: str) -> bool:
    """Check if two multiple-choice options are too similar, overlapping, or collision-prone.

    Prevents degenerate option sets such as:
    - 'Hadoop' vs 'Hadoop Cluster' vs 'Data in Hadoop' vs 'Hadoop and HDFS'
    """
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

    # Substring / Superstring check for meaningful words
    if s1 in s2 or s2 in s1:
        return True

    stopwords = {"and", "or", "in", "the", "a", "an", "of", "to", "for", "with", "by", "is", "are", "mode"}
    t1 = {w for w in re.findall(r"[a-z0-9]+", s1) if w not in stopwords}
    t2 = {w for w in re.findall(r"[a-z0-9]+", s2) if w not in stopwords}
    if not t1 or not t2:
        return False

    overlap = t1.intersection(t2)
    jaccard = len(overlap) / len(t1.union(t2))
    if jaccard >= 0.4:
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

    # 1. Detect Subject / Technology
    subject = None
    all_context = f"{source_fn} {sec_title} {chunk_text[:300]}"

    tech_candidates = [
        "Apache Pig", "Pig Latin", "Apache Hive", "Apache Spark", "Hadoop", "HDFS",
        "MapReduce", "Apache Kafka", "Docker", "Kubernetes", "Tableau", "YARN",
        "Apache HBase", "ZooKeeper", "Linux", "Git"
    ]
    for tech in tech_candidates:
        if re.search(rf"\b{re.escape(tech)}\b", all_context, re.I):
            subject = tech
            break

    if not subject:
        fn_clean = re.sub(r"\.(?:pdf|docx|ppt|pptx|txt|md)$", "", source_fn, flags=re.I).strip()
        fn_clean = re.sub(r"^(?:bda|unit\s*\d+|lecture\s*\d+|module\s*\d+)[\s_\-]+", "", fn_clean, flags=re.I).strip()
        if fn_clean and len(fn_clean) > 2:
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

    return {
        "subject": subject,
        "section": clean_sec or None,
        "mode": mode,
        "category": category,
        "document_name": source_fn,
        "page_number": chunk.get("page_number"),
    }


def get_domain_distractors_for_context(text: str) -> List[str]:
    """Detect subject domain from document context to supply aligned fallback concepts."""
    lower = (text or "").lower()
    if any(k in lower for k in ["local mode", "execution mode", "runtime", "grunt"]):
        return DOMAIN_DISTRACTOR_MAP["bigdata_runtime"] + DOMAIN_DISTRACTOR_MAP["bigdata"]
    if any(k in lower for k in ["chart", "plot", "histogram", "visualization", "scatter", "graph", "treemap"]):
        return DOMAIN_DISTRACTOR_MAP["visualization"]
    if any(k in lower for k in ["docker", "container", "kubernetes", "apdd", "devops", "microservice", "virtual machine"]):
        return DOMAIN_DISTRACTOR_MAP["cloud"]
    if any(k in lower for k in ["neural", "classification", "regression", "learning", "gradient", "feature engineering"]):
        return DOMAIN_DISTRACTOR_MAP["ml"]
    if any(k in lower for k in ["bda", "hadoop", "mapreduce", "hdfs", "spark", "yarn", "hive", "pig", "kafka", "namenode"]):
        return DOMAIN_DISTRACTOR_MAP["bigdata"]
    if any(k in lower for k in ["network", "router", "packet", "protocol", "osi", "tcp", "udp", "ip", "dns", "subnet"]):
        return DOMAIN_DISTRACTOR_MAP["networks"]
    if any(k in lower for k in ["operating system", "process", "thread", "kernel", "scheduling", "semaphore", "deadlock", "paging", "mutex"]):
        return DOMAIN_DISTRACTOR_MAP["os"]
    if any(k in lower for k in ["database", "dbms", "sql", "normalization", "relational", "acid", "transaction", "query"]):
        return DOMAIN_DISTRACTOR_MAP["dbms"]
    if any(k in lower for k in ["security", "cipher", "encryption", "decryption", "cryptography", "malware", "vulnerability"]):
        return DOMAIN_DISTRACTOR_MAP["security"]
    if any(k in lower for k in ["agile", "scrum", "software engineering", "refactoring", "waterfall", "testing"]):
        return DOMAIN_DISTRACTOR_MAP["software"]
    return CURATED_DOMAIN_DISTRACTORS


def clean_academic_concept(text: str) -> Optional[str]:
    """Sanitize concept string, stripping meta headings and strictly filtering stopwords and fillers."""
    if not text:
        return None

    cleaned = text.strip()

    # 1. Reject meta slide notes
    if re.match(r"^(?:explanation|note|example|warning|caution|step\s*\d*[:\-])\b", cleaned, re.I):
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

    # 5. Strip evaluative adjective prefixes (e.g. 'Important Component' -> 'Component', 'Common Techniques' -> 'Techniques')
    cleaned = EVALUATIVE_ADJ_PREFIXES.sub("", cleaned).strip()

    # 6. Strip leading articles: 'A container' -> 'Container'
    cleaned = re.sub(r"^(?:a|an|the|this|that|these|those)\s+", "", cleaned, flags=re.I).strip()
    cleaned = cleaned.strip(" :.-–—;")

    # Re-check action verbs and commands after stripping prefixes
    if ACTION_VERB_PREFIXES.match(cleaned) or cleaned.lower().endswith(" commands"):
        return None

    # 6. Check for invalid grammatical fragments or clauses
    lower = cleaned.lower()
    invalid_prefixes = (
        "when ", "where ", "if ", "because ", "that ", "which ", "how ", "what ", "why ",
        "who ", "it ", "there ", "here ", "in the", "on the", "with ", "for ", "such as",
        "according", "as an", "at the", "by the", "from the", "to the", "during", "based on",
        "once a", "and ", "or ", "each ", "every ", "all ", "some ", "any ", "created when",
        "runs multi", "used with", "supports "
    )
    if any(lower.startswith(prefix) for prefix in invalid_prefixes):
        return None

    # Reject grammatical clauses containing articles, auxiliary verbs, or purpose markers
    if re.search(r"\b(?:the|a|an|is|are|was|were|has|have|had|been|can|will|should|must|purpose)\b", lower):
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

    # Length & word count check: 1 to 3 words, 2 to 35 characters
    words = cleaned.split()
    if len(words) < 1 or len(words) > 3 or len(cleaned) < 2 or len(cleaned) > 35:
        return None

    # Single-word concepts must be genuine technical terms (not ordinary generic nouns)
    if len(words) == 1 and "-" not in cleaned:
        if lower in FORBIDDEN_CONCEPT_WORDS or lower in ENGLISH_STOPWORDS:
            return None
        is_acronym = cleaned.isupper() and len(cleaned) >= 2
        is_known = lower in KNOWN_SINGLE_WORD_TECH_CONCEPTS
        is_camel = any(c.isupper() for c in cleaned[1:])
        is_tech_suffix = any(lower.endswith(sfx) for sfx in (
            "ization", "isation", "ation", "ology", "metry", "graphy", "morphism", "lock"
        )) and len(cleaned) >= 6
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

    def to_dict(self) -> Dict[str, Any]:
        return {
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
) -> Tuple[bool, List[str]]:
    """Strictly validate generated question across all 8 quality dimensions.

    Returns (is_valid, list_of_failure_reasons).
    """
    failures = []
    q_text = question.question.strip()

    # A. Completeness & Minimum Length
    if len(q_text) < min_length:
        failures.append(f"Question text too short ({len(q_text)} chars)")
    if not q_text.endswith("?"):
        failures.append("Question does not end with '?'")

    # Reject literal quote completions and forbidden phrases
    if re.search(r'\bdefined as:\s*"|"not required in this mode"|\bknown as the above\b|\bmentioned in this case\b|\bcompletes the statement\b|\bwhat does this refer to\b|\bwhat is not required\b', q_text, re.I):
        failures.append("Question contains forbidden sentence-completion or literal quote phrasing")

    # Reject dangling anaphoric pronouns without explicit antecedent in the question
    dangling_match = DANGLING_ANAPHORA_PATTERN.search(q_text)
    if dangling_match:
        matched_str = dangling_match.group(0).lower()
        has_specific_mode = bool(re.search(r"\b(?:local|mapreduce|interactive|batch|hdfs|embedded)\s+mode\b", q_text, re.I))
        if not ("mode" in matched_str and has_specific_mode):
            failures.append("Question contains dangling anaphoric reference (e.g. 'this mode', 'the above')")

    # B. Context Sufficiency
    # If the question asks about a mode or requirement, ensure specific mode is explicitly named
    if "mode" in q_text.lower() and not re.search(r"\b(local mode|mapreduce mode|interactive mode|batch mode|hdfs mode|embedded mode)\b", q_text, re.I):
        failures.append("Question discusses mode without identifying which specific mode is being tested")

    # C. Answerability
    if not question.correct_answer or len(question.correct_answer.strip()) < 2:
        failures.append("Correct answer is empty or invalid")

    # D. Unambiguity & Distractor Anti-Collision
    if len(question.options) != 4:
        failures.append(f"Expected exactly 4 options, found {len(question.options)}")
    else:
        clean_opts = [re.sub(r"^[A-D]\)\s*", "", opt).strip() for opt in question.options]
        # Check uniqueness
        if len(set(o.lower() for o in clean_opts)) != 4:
            failures.append("Duplicate or non-unique options detected")
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

    # E. Grammatical Correctness
    if not q_text[0].isupper():
        failures.append("Question does not start with a capitalized letter")
    if q_text.count("?") > 1:
        failures.append("Multiple question marks detected")

    # F. Conceptual Relevance
    if question.correct_answer.lower() in FORBIDDEN_CONCEPT_WORDS or question.correct_answer.lower() in ENGLISH_STOPWORDS:
        failures.append(f"Correct answer '{question.correct_answer}' is a forbidden filler or stopword")

    # G. Explanation Consistency
    if not question.explanation or len(question.explanation.strip()) < 20:
        failures.append("Explanation is missing or too short")
    elif question.correct_answer.lower() not in question.explanation.lower():
        failures.append("Explanation does not mention or justify the correct answer")

    # H. Source Traceability
    if not question.source_document:
        failures.append("Source document metadata is missing")

    is_valid = len(failures) == 0
    return is_valid, failures

def unwrap_and_extract_sentences(text: str) -> List[str]:
    """Reconstruct wrapped document text into coherent, complete sentences.

    Solves PDF hard line wrapping where definitions break across lines
    (e.g., 'A line chart represents data points using points connected by lines. It is mainly used to show changes or\\ntrends over time.')
    """
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
        # Numbered heading: e.g. "3.1 Line Chart", "3. Existing Visualization Techniques"
        if re.match(r"^\d+(?:\.\d+)*\s+[A-Z][A-Za-z0-9\s\-]{1,40}$", clean):
            return True
        # Markdown heading: e.g. "## Line Chart", "### Histogram"
        if re.match(r"^#{1,6}\s+", clean):
            return True
        # Short standalone title line: e.g. "Line Chart", "Bar Chart", "Histogram", "Heat Map"
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


class QuizGenerator:
    """Automated academic quiz generator utilizing Inverted Seq2Seq and Vector Manifold sampling."""

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

                # Pattern A: "Concept is/are/refers to/is defined as/represents/displays/uses Definition"
                def_match = re.search(
                    r"^([A-Za-z0-9\.\s\-–—]{2,50}?)\s+(?:is|are|refers to|is defined as|is the process of|consists of|represents|displays|uses)\s+(.+)",
                    line,
                )
                if def_match:
                    raw_concept = def_match.group(1).strip()
                    definition = def_match.group(2).strip()
                    if ". " in definition:
                        definition = definition.split(". ")[0].strip() + "."
                    definition = re.sub(r"\s+(?:and|or|to|for|in|with|by|as|such as|of|the|a|an)\s*[\.\,\;\:]*$", ".", definition, flags=re.I)
                    concept = clean_academic_concept(raw_concept)

                    # Check for Mode / Operational Requirement statement (e.g. "Hadoop and HDFS are not required in this mode.")
                    is_neg_req = bool(re.search(r"\b(?:is|are)\s+not\s+(?:required|needed|used)\b", line, re.I))
                    is_pos_req = bool(re.search(r"\b(?:is|are)\s+(?:required|needed|used)\b", line, re.I))
                    if (is_neg_req or is_pos_req) and (INVALID_DEFINITION_START.search(definition) or DANGLING_ANAPHORA_PATTERN.search(definition)):
                        ctx = extract_chunk_hierarchy_context(chunk, line)
                        # ONLY accept if mode and subject can be verified with 100% certainty from the document!
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
                            # Context cannot be verified without guessing: discard candidate
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

                # Pattern B: Bolded term: "**Concept**: Explanation"
                bold_match = re.search(r"^\*{2}([A-Za-z0-9\.\s\-–—]{2,50}?)\*{2}(?::\s*|\s+[\-–—]\s+)(.+)", line)
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

                # Pattern C: Numbered Step or bullet list: "1. Concept - Explanation" or "- Concept: Explanation"
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
        """Formulate specific conceptual interrogative question stem, strictly rejecting sentence completions and quote wrappers."""
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

        # Extract hierarchical context if available
        subject = None
        category = None
        if anchor:
            subject = anchor.get("subject")
            category = anchor.get("category")
        elif chunk:
            ctx = extract_chunk_hierarchy_context(chunk, statement)
            subject = ctx.get("subject")
            category = ctx.get("category")

        # 2. Seq2Seq generation attempt with quality filters
        if model is not None and tokenizer is not None and self.config.enable_refinement:
            try:
                clean_def_snippet = definition.rstrip(". ;")[:160]
                prompt = (
                    f"Given the explanation: \"{clean_def_snippet}\", write a single academic question asking which concept is being described. "
                    f"Do not mention the word \"{concept}\". Do not ask which statement is true.\n\n"
                    f"Question: Which "
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
                        max_new_tokens=64,
                        num_beams=2,
                        early_stopping=True,
                    )
                decoded = tokenizer.decode(output_tokens[0], skip_special_tokens=True).strip()
                decoded = re.sub(r"^(?:Question|Q):?\s*", "", decoded, flags=re.I).strip()
                if not decoded.endswith("?"):
                    decoded += "?"

                is_invalid = bool(re.search(
                    r"\b(best\s+title|title\s+of|title\s+for|appropriate\s+title|suitable\s+title|name\s+of\s+the|"
                    r"headline|main\s+idea|main\s+point|theme|topic\s+of|which\s+(?:of\s+the\s+following\s+)?statements?\s+is\s+true|"
                    r"which\s+is\s+true|which\s+of\s+the\s+following\s+is\s+true|is\s+true\?|"
                    r"what\s+is\s+the\s+definition\s+of|what\s+is\s+the\s+main\s+concept\s+of|what\s+is\s+the\s+meaning\s+of|"
                    r"what\s+is\s+the\s+purpose\s+of|what\s+is\s+a\b|what\s+is\s+an\b|what\s+does\s+the\s+author|"
                    r"what\s+concept\s+is\s+being\s+described|what\s+is\s+the\s+concept\s+of|what\s+is\s+being\s+described|"
                    r"defined\s+as|known\s+as\s+the\s+above|completes\s+the\s+statement)\b",
                    decoded,
                    re.I,
                ))
                contains_concept = bool(re.search(rf"\b{re.escape(concept)}\b", decoded, re.I))

                if not is_invalid and not contains_concept and len(decoded) > 25:
                    return decoded
            except Exception as exc:
                logger.warning("Seq2Seq question generation fallback: %s", exc)

        # 3. Direct Conceptual Formulation (NO quotes, NO sentence completion)
        clean_def = definition.strip(" .;-–:")
        clean_def = re.sub(rf"^(?:an?|the)?\s*{re.escape(concept)}\s+(?:is|are|refers to|represents)\s+", "", clean_def, flags=re.I).strip()
        clean_def = re.sub(r"^(?:defined as|is defined as|refers to|is the process of|is|represents)\s+", "", clean_def, flags=re.I).strip()
        clean_def = re.sub(r"\s+(?:and|or|to|for|in|with|by|as|such as|of|the|a|an)\s*$", "", clean_def, flags=re.I).strip(" ,.-–:")
        clean_def_lower = clean_def.lower()

        known_nouns = [
            "technology", "plot", "chart", "diagram", "graph", "visualization", "technique",
            "method", "algorithm", "process", "framework", "system", "distribution",
            "metric", "measure", "tool", "approach", "platform", "paradigm", "package",
            "service", "environment", "engine", "component", "model", "pipeline", "map",
            "data structure", "data type"
        ]

        cat_noun = category or "concept"
        concept_words = concept.lower().split()
        for w in reversed(concept_words):
            if w in known_nouns:
                cat_noun = w
                break

        # Check if clean_def begins with "a/an <noun>" or "a/an <adjective(s)> <noun>"
        matched_stem = None
        for noun in known_nouns:
            m_noun = re.match(rf"^(?:an?|the)?\s*(?:[a-z\-]+\s+)?\b{re.escape(noun)}\b\s*(?:that|which|is|used to|for|in which)?\s*(.+)", clean_def, flags=re.I)
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

        # Third-person verbs (represents, uses, displays, manages, maintains, monitors, processes, calculates)
        if re.match(r"^(?:represents|uses|displays|calculates|measures|enables|provides|groups|divides|plots|manages|maintains|monitors|processes|stores)\b", clean_def, re.I):
            if subject and subject.lower() not in clean_def_lower:
                return f"In {subject}, which {cat_noun} {clean_def}?".replace("??", "?")
            return f"Which {cat_noun} {clean_def}?".replace("??", "?")

        # Used to / Used for
        if clean_def_lower.startswith("used to ") or clean_def_lower.startswith("is used to ") or clean_def_lower.startswith("used for "):
            rest = re.sub(r"^(?:is\s+)?used (?:to|for)\s+", "", clean_def, flags=re.I)
            if subject and subject.lower() not in rest.lower():
                return f"In {subject}, which {cat_noun} is used to {rest}?".replace("??", "?")
            return f"Which {cat_noun} is used to {rest}?".replace("??", "?")

        # An ordered collection / a collection of ...
        if clean_def_lower.startswith("an ordered collection") or clean_def_lower.startswith("a collection") or clean_def_lower.startswith("collection"):
            if subject and subject.lower() not in clean_def_lower:
                return f"In {subject}, which {cat_noun} is {clean_def}?".replace("??", "?")
            return f"Which {cat_noun} is {clean_def}?".replace("??", "?")

        # Natural question stem fallback WITHOUT ANY QUOTES
        if subject and subject.lower() not in clean_def_lower:
            return f"In {subject}, which {cat_noun} corresponds to {clean_def}?".replace("??", "?")
        return f"Which {cat_noun} corresponds to {clean_def}?".replace("??", "?")

    def mine_semantic_distractors(
        self,
        correct_concept: str,
        candidate_pool: List[str],
        embedding_model=None,
        context_text: str = "",
        anchor: Optional[Dict[str, Any]] = None,
        model=None,
        tokenizer=None,
        top_k: int = 3,
    ) -> List[str]:
        """Mine sibling technical concepts, enforcing strict anti-collision and category consistency."""
        norm_correct = correct_concept.lower().strip()

        # Sanitize every candidate from the folder, rejecting duplicates and colliding terms
        folder_candidates = []
        seen = {norm_correct}
        for raw in candidate_pool:
            cleaned = clean_academic_concept(raw)
            if cleaned and cleaned.lower() not in seen and not are_options_too_similar(cleaned, correct_concept):
                seen.add(cleaned.lower())
                folder_candidates.append(cleaned)

        # Domain-aligned curated sibling concepts
        is_mode_req = bool(anchor and anchor.get("anchor_type") == "mode_requirement") or "hadoop and hdfs" in norm_correct
        if is_mode_req:
            domain_distractors = DOMAIN_DISTRACTOR_MAP.get("bigdata_runtime", []) + get_domain_distractors_for_context(f"{context_text} {correct_concept}")
        else:
            domain_distractors = get_domain_distractors_for_context(f"{context_text} {correct_concept}")

        domain_siblings = [
            d for d in domain_distractors
            if d.lower() not in seen and not are_options_too_similar(d, correct_concept)
        ]

        if embedding_model is not None:
            try:
                combined_pool = folder_candidates + domain_siblings
                if combined_pool:
                    target_vec = embedding_model.encode([correct_concept], convert_to_numpy=True)[0]
                    cand_vecs = embedding_model.encode(combined_pool, convert_to_numpy=True)

                    dot_products = np.dot(cand_vecs, target_vec)
                    norms = (np.linalg.norm(cand_vecs, axis=1) * np.linalg.norm(target_vec)) + 1e-9
                    sims = dot_products / norms

                    ranked = []
                    for idx, cand in enumerate(combined_pool):
                        similarity = float(sims[idx])
                        if similarity >= 0.92 or similarity < 0.25:
                            continue
                        if not are_options_too_similar(cand, correct_concept):
                            ranked.append((similarity, cand))

                    ranked.sort(key=lambda x: x[0], reverse=True)

                    distractors: List[str] = []
                    for sim, cand in ranked:
                        if cand.lower() not in seen and not are_options_too_similar(cand, correct_concept):
                            if not any(are_options_too_similar(cand, prev) for prev in distractors):
                                distractors.append(cand)
                                if len(distractors) == top_k:
                                    return distractors

                    for cand in domain_siblings:
                        if cand.lower() not in seen and not are_options_too_similar(cand, correct_concept):
                            if not any(are_options_too_similar(cand, prev) for prev in distractors):
                                distractors.append(cand)
                                if len(distractors) == top_k:
                                    return distractors

                    if len(distractors) < top_k and model is not None and tokenizer is not None and self.config.enable_refinement:
                        try:
                            prompt = f"List three alternative technical concepts in the same field as \"{correct_concept}\" separated by commas: "
                            device = torch.device(self.config.device)
                            inputs = tokenizer(prompt, return_tensors="pt", max_length=64, truncation=True).to(device)
                            with torch.no_grad():
                                out_tokens = model.generate(input_ids=inputs["input_ids"], max_new_tokens=32, num_beams=1)
                            decoded = tokenizer.decode(out_tokens[0], skip_special_tokens=True).strip()
                            for item in re.split(r"[,;\n]+", decoded):
                                cand = clean_academic_concept(item.strip())
                                if cand and cand.lower() not in seen and not are_options_too_similar(cand, correct_concept):
                                    if not any(are_options_too_similar(cand, prev) for prev in distractors):
                                        distractors.append(cand)
                                        if len(distractors) == top_k:
                                            return distractors
                        except Exception as exc:
                            logger.debug("Flan-T5 sibling distractor generation fallback: %s", exc)

                    if distractors:
                        return distractors
            except Exception as exc:
                logger.warning("Embedding distractor calculation fallback: %s", exc)

        # Fallback pool with anti-collision
        distractors = []
        for cand in domain_siblings + folder_candidates:
            if cand.lower() != norm_correct and not are_options_too_similar(cand, correct_concept):
                if not any(are_options_too_similar(cand, prev) for prev in distractors):
                    distractors.append(cand)
                    if len(distractors) == top_k:
                        return distractors

        return distractors[:top_k]

    def regenerate_question_with_context(
        self,
        anchor: Dict[str, Any],
        chunk: Dict[str, Any],
        candidate_pool: List[str],
        embedding_model=None,
        folder_name: str = "",
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
                + f", {clean_correct} in {enriched_anchor.get('subject', 'the course material')} is: "
                + f"{enriched_anchor.get('definition', '').rstrip('. ;')}."
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

        is_valid, _ = validate_quiz_question(candidate_q, source_chunk_text=chunk.get("text", ""))
        return candidate_q if is_valid else None

    def generate_quiz_for_folder(
        self,
        db: Session,
        current_user: User,
        folder_id: str,
        num_questions: int = 10,
        document_ids: Optional[List[str]] = None,
    ) -> QuizResponse:
        """Execute end-to-end quiz generation for an authorized user folder with strict validation and deduplication."""
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

        index_dir = get_folder_index_dir(BASE_STORAGE_DIR, str(current_user.id), str(folder.id))
        index = FolderVectorIndex(index_dir)

        if document_ids:
            doc_id_set = {str(did).strip() for did in document_ids if str(did).strip()}
            chunks = [c for c in (index.chunks or []) if str(c.get("document_id")) in doc_id_set]
        else:
            chunks = list(index.chunks) if index.chunks else []

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
            )

        # 1. Extract Fact Anchors
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

        # 2. Build candidate pool for distractors from fact anchors and domain libraries
        candidate_pool: List[str] = [fa["concept"] for fa in fact_anchors]
        folder_sample = " ".join([c.get("text", "")[:100] for c in chunks[:5]])
        for dp in get_domain_distractors_for_context(folder.name + " " + folder_sample):
            if dp not in candidate_pool:
                candidate_pool.append(dp)

        # 3. Model initialization
        model, tokenizer = get_atml_model_and_tokenizer(self.config)
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
                model=model,
                tokenizer=tokenizer,
            )

            # Mine 3 sibling distractors with strict anti-collision
            context_text = f"{folder.name} {chunk.get('source_filename', '')} {chunk.get('text', '')[:200]}"
            distractors = self.mine_semantic_distractors(
                correct_concept=concept,
                candidate_pool=candidate_pool,
                embedding_model=embedding_model,
                context_text=context_text,
                anchor=anchor,
                model=model,
                tokenizer=tokenizer,
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

            # Fill with domain defaults if needed, respecting anti-collision
            context_domain_pool = get_domain_distractors_for_context(chunk.get("text", "") + " " + clean_correct)
            for cd in context_domain_pool:
                if len(sanitized_distractors) >= 3:
                    break
                if cd.lower() not in seen_opt and not are_options_too_similar(cd, clean_correct):
                    if not any(are_options_too_similar(cd, prev) for prev in sanitized_distractors):
                        seen_opt.add(cd.lower())
                        sanitized_distractors.append(cd)

            if len(sanitized_distractors) < 3:
                continue

            all_options = [clean_correct] + sanitized_distractors[:3]
            random.shuffle(all_options)
            correct_idx = all_options.index(clean_correct)

            letters = ["A", "B", "C", "D"]
            formatted_options = [f"{letters[i]}) {opt}" for i, opt in enumerate(all_options)]

            page_num = chunk.get("page_number")
            doc_name = chunk.get("source_filename") or "Course Material"
            clean_def_preview = definition.rstrip(". ;")

            if anchor.get("anchor_type") == "mode_requirement" and anchor.get("mode"):
                explanation = (
                    f"According to {doc_name}"
                    + (f" (Page {page_num})" if page_num else "")
                    + f", in {anchor['mode']} of {anchor.get('subject', 'the system')}, "
                    + f"{clean_correct} is not required because all files are processed using the local file system."
                )
            else:
                subj_tag = anchor.get("subject") or "the course material"
                explanation = (
                    f"According to {doc_name}"
                    + (f" (Page {page_num})" if page_num else "")
                    + f", {clean_correct} in {subj_tag} is: {clean_def_preview}."
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
            is_valid, failures = validate_quiz_question(q_item, source_chunk_text=chunk.get("text", ""))
            if not is_valid:
                logger.debug("Question validation failed (%s). Attempting regeneration with context...", failures)
                regenerated_q = self.regenerate_question_with_context(
                    anchor=anchor,
                    chunk=chunk,
                    candidate_pool=candidate_pool,
                    embedding_model=embedding_model,
                    folder_name=folder.name,
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
                f"Generated {len(questions)} high-quality validated questions. "
                f"Additional candidate questions were discarded to prevent incomplete or ambiguous items from limited source material."
            )

        return QuizResponse(
            folder_id=folder.id,
            folder_name=folder.name,
            total_questions=len(questions),
            questions=questions,
            note=note,
        )

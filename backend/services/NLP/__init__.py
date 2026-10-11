"""NLP services package for Nalanda AI.

Modules:
- document_chunking_service: Boundary-aware document text chunking and metadata generation.
- vector_index_service: SentenceTransformer embedding management and isolated vector indexing.
- semantic_search_service: End-to-end scoped semantic search, multimodal asset binding, and relevance scoring.
"""
from backend.services.NLP.document_chunking_service import (
    chunk_document_data,
    create_chunks_from_paragraphs,
    detect_section_heading,
    split_text_into_paragraphs,
)
from backend.services.NLP.vector_index_service import (
    FolderVectorIndex,
    get_embedding_model,
    get_folder_index_dir,
)
from backend.services.NLP.semantic_search_service import (
    execute_semantic_search,
    get_or_create_folder_index,
    index_single_document,
    sync_folder_index,
    detect_query_intent,
    extract_direct_answer,
)

__all__ = [
    "chunk_document_data",
    "create_chunks_from_paragraphs",
    "detect_section_heading",
    "split_text_into_paragraphs",
    "FolderVectorIndex",
    "get_embedding_model",
    "get_folder_index_dir",
    "execute_semantic_search",
    "get_or_create_folder_index",
    "index_single_document",
    "sync_folder_index",
    "detect_query_intent",
    "extract_direct_answer",
]

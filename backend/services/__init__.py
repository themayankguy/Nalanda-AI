"""Nalanda AI Services package.

Exports core storage and extraction services, with NLP modules organized under backend.services.NLP.
"""
from backend.services.storage_service import (
    BASE_STORAGE_DIR,
    sanitize_filename,
    validate_file,
    get_document_paths,
)
from backend.services.nalanda_extraction_service import (
    run_extraction_subprocess,
    parse_and_store_extraction_output,
    process_document_extraction,
)

# Re-export NLP modules for backward compatibility and clean convenience
from backend.services.NLP import (
    document_chunking_service,
    vector_index_service,
    semantic_search_service,
)

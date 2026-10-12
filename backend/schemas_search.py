"""Pydantic schemas for local semantic search, multimodal RAG retrieval, and structured answers."""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class SemanticSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="Natural-language search query")
    folder_id: str = Field(..., min_length=1, description="Target folder ID for user- and folder-scoped retrieval")
    top_k: int = Field(default=5, ge=1, le=20, description="Maximum number of relevant source passages to return")


class FigureItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    figure_index: int
    page_number: Optional[int] = None
    filename: str
    figure_url: str
    caption: Optional[str] = None
    source: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None


class TableItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    table_index: int
    page_number: Optional[int] = None
    html: Optional[str] = None
    markdown: Optional[str] = None


class SearchResultItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    document_id: str
    source_filename: str
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    relevance_score: float = Field(..., description="Cosine similarity score (0.0 to 1.0)")
    direct_answer: Optional[str] = None
    formatted_context: Optional[str] = None
    figures: List[FigureItem] = Field(default_factory=list)
    tables: List[TableItem] = Field(default_factory=list)


class SemanticSearchResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    query: str
    folder_id: str
    detected_intent: str = Field(default="conceptual", description="'conceptual', 'visual', or 'tabular'")
    direct_answer: Optional[str] = None
    primary_figure: Optional[FigureItem] = None
    primary_table: Optional[TableItem] = None
    total_results: int
    results: List[SearchResultItem]
    refinement_metadata: Optional[Dict[str, Any]] = Field(default=None, description="Diagnostic ATML answer refinement metadata")

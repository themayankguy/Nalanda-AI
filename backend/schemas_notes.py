"""Pydantic schemas for ATML Comprehensive Notes Generation."""
from typing import List, Optional
from pydantic import BaseModel, Field


class NotesSectionSchema(BaseModel):
    """Represents a structured subsection in the study guide."""
    id: str = Field(..., description="Unique section identifier")
    title: str = Field(..., description="Section title")
    icon: str = Field(default="📌", description="Display icon for section")
    content_markdown: str = Field(..., description="Formatted markdown content")
    citations: List[str] = Field(default_factory=list, description="Associated source citations")


class NotesResponseSchema(BaseModel):
    """Complete comprehensive study guide response for a folder."""
    folder_id: str = Field(..., description="Target folder ID")
    folder_name: str = Field(..., description="Folder name")
    generated_at: str = Field(..., description="ISO timestamp of generation")
    documents_analyzed: List[str] = Field(default_factory=list, description="List of source filenames parsed")
    total_sections: int = Field(..., description="Number of distinct sections generated")
    markdown_content: str = Field(..., description="Complete aggregated study guide in Markdown")
    sections: List[NotesSectionSchema] = Field(default_factory=list, description="Structured subsections for collapsible rendering")

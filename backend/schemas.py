"""Pydantic schemas for Nalanda AI request and response validation.

CRITICAL REQUIREMENT:
Extracted text, OCR output, raw Markdown, JSON extraction results, and intermediate
processing files must remain internal to the backend. They must NEVER be included
in document API responses or frontend state.
"""
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, EmailStr, Field, ConfigDict


# --- Auth Schemas ---

class UserRegisterRequest(BaseModel):
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6, max_length=128)


class UserLoginRequest(BaseModel):
    email_or_username: str
    password: str


class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    email: str
    username: str
    created_at: datetime


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


# --- Folder Schemas ---

class FolderCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)


class FolderResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    user_id: str
    document_count: int = 0
    created_at: datetime
    updated_at: datetime


# --- Document Schemas (Document Metadata Only) ---

class DocumentMetadataResponse(BaseModel):
    """Document-level metadata returned to the frontend.
    
    Contains NO extracted content, raw text, or internal server paths.
    """
    model_config = ConfigDict(from_attributes=True)

    id: str
    folder_id: str
    original_filename: str
    file_type: str
    file_size_bytes: int
    status: str  # pending, processing, completed, failed
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class DocumentUploadResponse(BaseModel):
    id: str
    folder_id: str
    original_filename: str
    status: str
    message: str


class DocumentStatusResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    status: str
    error_message: Optional[str] = None
    updated_at: datetime

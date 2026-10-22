# Nalanda AI — Technical End-to-End Workflow Specification

This document details the complete, step-by-step workflow of the **Nalanda AI** system—from the moment an authenticated student uploads a document in the web interface through preprocessing, layout decompilation, OCR extraction, text normalization, semantic chunking, vector indexing, downstream AI processing (Study Notes, Practice Quizzes, and Ilayas AI Conversational Chat), to frontend visualization and PDF export.

---

## A. Main End-to-End Flowchart

The following flowchart captures the entire sequential pipeline, including all conditional branches, fallback paths, and error handling states:

```mermaid
flowchart TD
    %% Entry Stage
    START([Student Selects Document in Frontend]) --> UI_INPUT[Drag-and-Drop or File Input Dialog in app.js]
    UI_INPUT --> UI_VALIDATE{Active Folder Selected?}
    UI_VALIDATE -- No --> UI_TOAST_ERR[Show Toast Error: 'Please select a folder first']
    UI_VALIDATE -- Yes --> UI_UPLOAD_REQ[Construct FormData & Issue POST /api/folders/{id}/documents/upload]

    %% Backend Gateway & Validation
    UI_UPLOAD_REQ --> API_ROUTE[backend/routes/document_routes.py: upload_document]
    API_ROUTE --> DB_AUTH{Verify User & Folder Ownership}
    DB_AUTH -- Unauthorized / Not Found --> HTTP_404[Raise HTTP 404 / 401]
    DB_AUTH -- Authorized --> VAL_FILE[backend/services/storage_service.py: validate_file]
    
    VAL_FILE --> EXT_CHECK{Extension in .pdf, .jpg, .png, etc.?}
    EXT_CHECK -- Invalid Extension --> HTTP_400_EXT[Raise HTTP 400 Bad Request]
    EXT_CHECK -- Valid --> SIZE_CHECK{File Size <= 50 MB?}
    SIZE_CHECK -- Exceeds Limit --> HTTP_400_SIZE[Raise HTTP 400 File Size Exceeded]
    
    SIZE_CHECK -- Valid --> SAVE_ORIG[storage_service.py: Write file to storage/.../original/]
    SAVE_ORIG --> DB_REC[Insert Document Record status='pending' in SQLite]
    DB_REC --> QUEUE_TASK[Add process_document_extraction to FastAPI BackgroundTasks]
    QUEUE_TASK --> HTTP_202[Return HTTP 202 Accepted DocumentUploadResponse to Frontend]
    HTTP_202 --> UI_POLL[Frontend app.js starts polling /api/folders/{id}/documents]

    %% Background Ingestion Worker
    QUEUE_TASK -.-> BG_WORKER[backend/services/nalanda_extraction_service.py: process_document_extraction]
    BG_WORKER --> DB_PROC[Update SQLite Document status='processing']
    DB_PROC --> SUBPROC[Invoke subprocess: sys.executable nalanda_extractor.py input_file --output out_dir]

    %% Subprocess Extraction Execution
    SUBPROC --> EXT_START[nalanda_extractor.py: process_file]
    EXT_START --> FILE_TYPE{Is Digital PDF or Image/Scanned?}

    %% Digital PDF Branch
    FILE_TYPE -- PDF Document --> PDF_PAGES[PyMuPDF: Iterate pages via fitz.open]
    PDF_PAGES --> CHAR_CHECK{len of page text >= 50 chars?}
    CHAR_CHECK -- Yes: Digital PDF Page --> FITZ_TEXT[PyMuPDF page.get_text: Direct Layout Extraction]
    FITZ_TEXT --> FITZ_TABLE[PyMuPDF page.find_tables: Extract native tables]
    CHAR_CHECK -- No: Scanned Page --> RENDER_PNG[Render page as PNG pixmap via fitz.Matrix 1.5x]
    RENDER_PNG --> PADDLE_OCR[PaddleOCR PP-StructureV3: OCR text & table recognition]
    
    %% Standalone Image Branch
    FILE_TYPE -- Image File --> PADDLE_IMG[PaddleOCR PP-StructureV3 Pipeline on image]
    
    %% Extracted Elements Aggregation
    FITZ_TEXT & FITZ_TABLE & PADDLE_OCR & PADDLE_IMG --> EXTRACT_FIGS[PyMuPDF / Pillow: Extract raster diagrams to figures/]
    EXTRACT_FIGS --> WRITE_OUT[Write artifacts to disk: markdown/, txt/, json/, tables/, figures/, stats.json]

    %% Subprocess Evaluation & Ingestion
    WRITE_OUT --> SUBPROC_RET{Subprocess Exit Code == 0?}
    SUBPROC_RET -- Non-Zero / Timeout --> DB_FAIL[Update Document status='failed' with sanitized error]
    DB_FAIL --> UI_FAIL[Frontend displays 'Failed' badge with Retry option]
    
    SUBPROC_RET -- Success (0) --> PARSE_OUT[nalanda_extraction_service.py: parse_and_store_extraction_output]
    PARSE_OUT --> ATML_AE_CHECK{AutoencoderOutlierDetector: Check latent reconstruction error}
    ATML_AE_CHECK -- High Error --> FLAG_OOD[Log warning: is_mismatched_document=True in metadata]
    ATML_AE_CHECK -- In-Domain --> DB_EXTRACTION
    FLAG_OOD --> DB_EXTRACTION[Store Markdown, JSON path, HTML tables in SQLite DocumentExtraction]
    DB_EXTRACTION --> DB_COMPLETE[Update Document status='completed']
    DB_COMPLETE --> UI_INDEXED[Frontend Poller receives status='completed' -> Displays 'Indexed' pill]

    %% Incremental NLP Processing & Dynamic Index Synchronization
    DB_COMPLETE --> INC_INDEX[backend/services/NLP/semantic_search_service.py: index_single_document]
    INC_INDEX --> TEXT_NORM[backend/services/NLP/text_normalizer.py: normalize_ocr_text]
    TEXT_NORM --> CHUNKER[backend/services/NLP/document_chunking_service.py: chunk_document_data]
    CHUNKER --> EMBED_GEN[backend/services/NLP/vector_index_service.py: SentenceTransformer all-MiniLM-L6-v2]
    EMBED_GEN --> SAVE_INDEX[Append to embeddings.npy & metadata.json in vector_index/]

    %% Incremental Knowledge Base & Conflict Detection Pipeline
    DB_COMPLETE --> KNOWL_INTEG[backend/services/NLP/knowledge_integration_service.py: integrate_document_knowledge]
    KNOWL_INTEG --> CONCEPT_SYNTH[Synthesize/Enrich KnowledgeConcept domain entities]
    CONCEPT_SYNTH --> ITEM_DEDUP{Exact SHA-256 match in KnowledgeItem?}
    ITEM_DEDUP -- Duplicate --> SKIP_ITEM[Append source provenance to existing KnowledgeItem]
    ITEM_DEDUP -- New Fact --> CONFLICT_CHECK{Contradiction with existing items?}
    CONFLICT_CHECK -- Yes --> FLAG_CONFLICT[Set is_conflict=True & store conflict_details JSON]
    CONFLICT_CHECK -- No --> SAVE_ITEM[Persist atomic KnowledgeItem]
    SKIP_ITEM & FLAG_CONFLICT & SAVE_ITEM --> AUDIT_LOG[Persist counts to DocumentIngestionLog]

    %% Downstream Service 1: Study Notes Generation
    SAVE_INDEX & AUDIT_LOG -.-> USER_NOTES[User Clicks 'Study Guide' Tab]
    USER_NOTES --> NOTES_ROUTE[GET /api/folders/{id}/notes]
    NOTES_ROUTE --> NOTES_GEN[backend/services/atml/notes_generator.py: NotesGenerator]
    NOTES_GEN --> CACHE_CHECK{SHA-256 Hash matches folder_note_caches?}
    CACHE_CHECK -- Yes (Hit) --> CACHE_RETURN[Return cached notes instantly in < 5ms]
    CACHE_CHECK -- No (Miss) --> SYNTH_NOTES[Synthesize 5 Academic Sections: Overview, Concepts, Comparison, Procedures, Exam]
    SYNTH_NOTES --> SAVE_NOTE_CACHE[Persist in SQLite folder_note_caches]
    SAVE_NOTE_CACHE & CACHE_RETURN --> RENDER_NOTES_UI[Frontend app.js renders interactive Markdown Notes]
    RENDER_NOTES_UI --> PDF_EXPORT[User Clicks 'Download PDF']
    PDF_EXPORT --> PDF_ROUTE[POST /api/folders/{id}/notes/export-pdf]
    PDF_ROUTE --> PYMUPDF_STORY[backend/services/atml/pdf_exporter.py: fitz.Story A4 paginated PDF compiler]
    PYMUPDF_STORY --> DOWNLOAD_BLOB[Browser downloads publication-ready PDF]

    %% Downstream Service 2: Practice Quiz Runner
    SAVE_INDEX -.-> USER_QUIZ[User Clicks 'Practice Quiz' Tab]
    USER_QUIZ --> QUIZ_ROUTE[GET /api/folders/{id}/quiz?num_questions=10]
    QUIZ_ROUTE --> QUIZ_GEN[backend/services/atml/quiz_generator.py: QuizGenerator]
    QUIZ_GEN --> COPULA_MINE[Mine Fact-Anchors from lecture chunks via copula regex]
    COPULA_MINE --> MANIFOLD_DIST[Bandpass Cosine Manifold Filter: 0.35 <= cos < 0.92]
    MANIFOLD_DIST --> MCQ_BUILD[Assemble 4-Option MCQs with slide page citations]
    MCQ_BUILD --> RENDER_QUIZ_UI[Frontend app.js interactive Quiz Runner with instant grading]

    %% Downstream Service 3: Ilayas AI Conversational Chat
    SAVE_INDEX -.-> USER_CHAT[User Submits Query in Chat]
    USER_CHAT --> CHAT_ROUTE[POST /api/conversations/{id}/messages]
    CHAT_ROUTE --> CTX_RES[backend/services/NLP/chat_context_service.py: MAQD anaphora/ordinal resolution]
    CTX_RES --> INTENT_ROUTER{Classify Pedagogical Intent}
    INTENT_ROUTER -- Taxonomical 'Types of X' --> HCG_CRAWL[backend/services/NLP/hierarchical_concept_gatherer.py: 100% Subtopic Crawl]
    INTENT_ROUTER -- Visual 'Diagram of X' --> DIAG_MATCH[Locate matching figure & generate tokenized image URL]
    INTENT_ROUTER -- Procedural / Conceptual --> DENSE_SEARCH[NumPy dot-product matrix cosine search]
    HCG_CRAWL & DIAG_MATCH & DENSE_SEARCH --> ANS_REFINE[backend/services/atml/answer_refiner.py: FLAN-T5 Synthesis]
    ANS_REFINE --> OUT_GUARD[backend/services/atml/output_validator.py: Set-theoretic grounding validation]
    OUT_GUARD --> DB_CHAT_SAVE[Persist turn in chat_messages table]
    DB_CHAT_SAVE --> UI_STREAM[Frontend app.js renders typewriter response with inline citations & modals]
```

---

## B. Stage-by-Stage Technical Explanation

### Stage 1: User Interaction & File Selection
- **What happens:** The student logs into the Nalanda AI dashboard, selects or creates an active Course Folder (e.g., *"Big Data Analytics"*), and clicks the upload area or drags a course file into the drop zone.
- **Technology/library used:** Vanilla JavaScript (ES6+), HTML5 Drag-and-Drop API.
- **How it works:** The file input element (`#file-input`) is triggered via `triggerFileInput()`. When a file is dropped or selected, the change event handler dispatches the file object to `handleFileUpload(file)`. If no folder is active in application state (`activeFolderId === null`), an error toast notification is displayed immediately.
- **Input:** User interaction event with `File` object from browser DOM.
- **Output:** Validated `activeFolderId` and `File` object packed into a standard `FormData` payload.
- **Next step:** Sent asynchronously via HTTP POST to the backend gateway.
- **Code location:** [frontend/app.js](file:///Users/mayanksmac/Desktop/capstone/frontend/app.js#L1495-L1534), functions `triggerFileInput()` and `handleFileUpload()`.

---

### Stage 2: Input Handling & Gateway Receiving
- **What happens:** The backend receives the multipart form upload, authenticates the student's session token, and ensures the target course folder exists.
- **Technology/library used:** FastAPI (`UploadFile`, `File`, `BackgroundTasks`, `Depends`), SQLAlchemy 2.x.
- **How it works:** The request hits `@router.post("/api/folders/{folder_id}/documents/upload")`. The `get_current_user` dependency verifies the Bearer session token against the SQLite `session_tokens` table. A database query confirms that the target `Folder` exists and belongs to the authenticated user (`Folder.user_id == current_user.id`).
- **Input:** HTTP POST request with `multipart/form-data`, path parameter `folder_id`, and `Authorization: Bearer <token>`.
- **Output:** Authenticated `current_user` model, verified `Folder` model, and raw byte buffer from `await file.read()`.
- **Next step:** Passed to file validation in the storage service.
- **Code location:** [backend/routes/document_routes.py](file:///Users/mayanksmac/Desktop/capstone/backend/routes/document_routes.py#L47-L83), function `upload_document()`.

---

### Stage 3: File Validation & Security Guards
- **What happens:** Filename characters are sanitized, and the file's extension and byte length are strictly validated against security rules.
- **Technology/library used:** Python standard library (`pathlib.Path`, `os.path`, `re`).
- **How it works:** 
  1. `sanitize_filename(filename)` strips path traversal tokens (`..`, `/`, `\`) and control characters, truncating the filename to a maximum of 255 characters.
  2. `validate_file(filename, file_size)` extracts the lowercase suffix and checks if it belongs to `SUPPORTED_EXTENSIONS`:
     $$\text{SUPPORTED\_EXTENSIONS} = \{\text{.pdf}, \text{.jpg}, \text{.jpeg}, \text{.png}, \text{.webp}, \text{.bmp}, \text{.tif}, \text{.tiff}\}$$
  3. Verifies that `file_size <= MAX_FILE_SIZE_BYTES` ($50 \times 1024 \times 1024 = 50\text{ MB}$).
  4. Returns `file_type = "pdf"` or `"image"`. If validation fails, an HTTP 400 Bad Request exception is raised immediately.
- **Input:** Raw filename string and integer `file_size`.
- **Output:** Tuple `(is_valid: bool, file_type: str, error_message: str)`.
- **Next step:** Secure directory allocation and original file disk persistence.
- **Code location:** [backend/services/storage_service.py](file:///Users/mayanksmac/Desktop/capstone/backend/services/storage_service.py#L26-L61), functions `sanitize_filename()` and `validate_file()`.

---

### Stage 4: Storage Allocation & Asynchronous Job Dispatch
- **What happens:** Allocates an isolated filesystem path for the document, saves the original file to disk, writes a pending database record, and queues extraction in background workers.
- **Technology/library used:** Python standard library (`uuid`, `open`), SQLAlchemy 2.x, FastAPI `BackgroundTasks`.
- **How it works:**
  1. A unique UUID4 `doc_id` is generated.
  2. `get_document_paths()` creates isolated user/folder/document storage directories:
     $$\text{storage/users/}\{user\_id\}\text{/folders/}\{folder\_id\}\text{/documents/}\{doc\_id\}\text{/original/}$$
     $$\text{storage/users/}\{user\_id\}\text{/folders/}\{folder\_id\}\text{/documents/}\{doc\_id\}\text{/extracted/}$$
  3. The byte content is written to `original_file_path`.
  4. A `Document` row is committed to SQLite with `status="pending"`.
  5. `background_tasks.add_task(process_document_extraction, document.id)` schedules extraction.
  6. Returns an HTTP 202 Accepted response with a `DocumentUploadResponse` schema to the client.
- **Input:** Validated file bytes, `current_user.id`, `folder.id`, and `doc_id`.
- **Output:** Persisted original file on disk, initial SQLite record, and queued background task.
- **Next step:** Asynchronous worker takes over; client begins polling for progress.
- **Code location:** [backend/routes/document_routes.py](file:///Users/mayanksmac/Desktop/capstone/backend/routes/document_routes.py#L84-L129), function `upload_document()`.

---

### Stage 5: Ingestion Subprocess Isolation
- **What happens:** The background worker transitions the document status to `"processing"` and launches the extraction engine in a sandboxed operating system subprocess.
- **Technology/library used:** Python `subprocess` module, SQLAlchemy `SessionLocal`.
- **How it works:** `process_document_extraction(document_id)` opens an independent database session. It updates `Document.status = "processing"`. It then invokes `run_extraction_subprocess()` using `sys.executable` and `shell=False`:
  ```bash
  python nalanda_extractor.py <path_to_original_file> --output <path_to_extracted_dir>
  ```
  The process is bounded by a 300-second timeout (`SUBPROCESS_TIMEOUT_SECONDS = 300`).
- **Input:** `document_id: str`.
- **Output:** Spawned subprocess execution of `nalanda_extractor.py`.
- **Next step:** Execution of extraction algorithms within `nalanda_extractor.py`.
- **Code location:** [backend/services/nalanda_extraction_service.py](file:///Users/mayanksmac/Desktop/capstone/backend/services/nalanda_extraction_service.py#L31-L60), function `run_extraction_subprocess()`.

---

### Stage 6: OCR, Layout Decompilation & Multimodal Extraction
- **What happens:** Decompiles text, bounding boxes, tables, and visual diagrams from digital PDFs, scanned pages, or image files.
- **Technology/library used:** PyMuPDF (`fitz`), PaddleOCR (`PPStructureV3`), Pillow (`PIL`).
- **How it works:**
  1. **File Type Detection:**
     - **Digital PDF:** Opened via `fitz.open()`. Iterates over each page. Checks character density:
       $$\text{len}(\text{page.get\_text}("text")) \ge \text{MIN\_TEXT\_CHARS } (50\text{ characters})$$
       - **If True (Digital):** PyMuPDF extracts text directly. Native tables are detected via `page.find_tables()` and converted into HTML `<table>` elements and Markdown.
       - **If False (Scanned Slide / Low Text):** The page is rendered into a high-resolution PNG pixmap (`fitz.Matrix(1.5, 1.5)`). PaddleOCR's `PPStructureV3` analyzes layout, performing text detection, character recognition, and table boundary reconstruction.
     - **Image File:** Passes directly to `PPStructureV3`, executing full OCR and table cell reconstruction.
  2. **Diagram & Figure Extraction:**
     - PyMuPDF inspects image blocks on the page via `page.get_images()`.
     - Raster visual elements with bounding areas $\ge 10,000\text{ px}^2$ ($100 \times 100\text{ px}$) are saved as PNG files into the `figures/` directory.
     - Detects captions via regex (`^(?:Figure|Fig\.?)\s*\d+.*`).
  3. **File System Serialization:** Writes extracted artifacts to disk:
     - `markdown/<stem>.md`
     - `txt/<stem>.txt`
     - `json/<stem>.json` (contains page numbers, bounding boxes, text, tables, and timings)
     - `tables/<stem>_page_<p>_table_<i>.html`
     - `processing_summary.json`
- **Input:** Original PDF or image file on local disk.
- **Output:** Serialized Markdown, plain text, structured JSON metadata, table HTML files, and diagram images.
- **Next step:** Subprocess finishes and returns exit code to extraction service.
- **Code location:** [nalanda_extractor.py](file:///Users/mayanksmac/Desktop/capstone/nalanda_extractor.py#L266-L360), functions `process_pdf()`, `process_image()`, and `run_ppstructure()`.

---

### Stage 7: Extraction Ingestion & Database Persistence
- **What happens:** The backend parses the generated JSON and Markdown outputs, runs ATML autoencoder outlier compatibility analysis, populates the `document_extractions` database table, and marks the document `"completed"`.
- **Technology/library used:** SQLAlchemy 2.x, Python `json` module, PyTorch (`AutoencoderOutlierDetector`).
- **How it works:** 
  1. `parse_and_store_extraction_output()` reads `<extracted_dir>/json/<stem>.json` and `<extracted_dir>/markdown/<stem>.md`.
  2. Verifies that `json_path` exists. If the subprocess failed (exit code $\ne 0$ or recorded failure in `processing_summary.json`), `Document.status` is set to `"failed"` with a sanitized error string.
  3. **ATML Latent Outlier Compatibility Check:** If the course folder already has $\ge 5$ indexed vectors, `AutoencoderOutlierDetector.get_or_train_folder_model()` evaluates the incoming document's embedding reconstruction error against the folder's 384-dimensional latent manifold:
     $$\mathcal{L}_{\text{recon}}(\mathbf{x}) = \|\mathbf{x} - g_\psi(f_\theta(\mathbf{x}))\|_2^2$$
     Documents exhibiting anomalous reconstruction errors are flagged with `is_mismatched_document=True` in `metadata_json` (without blocking ingestion).
  4. Creates or updates a `DocumentExtraction` record containing `full_text`, `markdown_content`, `table_files_json`, `metadata_json`, `total_pages`, `total_characters`, and `total_tables`.
  5. Commits `Document.status = "completed"`.
- **Input:** Disk files inside `<extracted_dir>/` and folder embedding subspace.
- **Output:** Persisted `DocumentExtraction` database row, outlier compatibility metadata, and `status="completed"`.
- **Next step:** Incremental vector indexing and multi-document knowledge base integration.
- **Code location:** [backend/services/nalanda_extraction_service.py](file:///Users/mayanksmac/Desktop/capstone/backend/services/nalanda_extraction_service.py#L62-L196), function `parse_and_store_extraction_output()`.

---

### Stage 8: Text Normalization & OCR Line-Healing
- **What happens:** Raw text is healed from PDF margin splits and OCR hyphenation artifacts before chunking or embedding.
- **Technology/library used:** Python `re` module (Regular Expressions).
- **How it works:**
  1. **Hyphen Healing:** Rejoins words broken across line wraps:
     ```python
     re.sub(r"(\b[a-zA-Z]{2,})-\s*\n\s*([a-zA-Z]{2,}\b)", r"\1\2", text)
     ```
  2. **Soft Line-Break Unwrapping:** Collapses intra-paragraph newlines into single spaces while preserving terminal sentence punctuation (`.`, `?`, `!`, `:`).
  3. **Structural Fence Guard:** Uses `STRUCTURAL_LINE_RE` to bypass unwrapping for Markdown headers (`#`), table borders (`|`), bullet markers (`-`, `•`), and code fences (```` ``` ````).
  4. **Banner Stripping:** Strips repetitive slide footers/headers (e.g., `^Page \d+ of \d+`).
- **Input:** Raw extracted text from PDF/OCR pages.
- **Output:** Cleaned, continuous, structure-preserving Markdown string.
- **Next step:** Semantic slide-boundary token chunking.
- **Code location:** [backend/services/NLP/text_normalizer.py](file:///Users/mayanksmac/Desktop/capstone/backend/services/NLP/text_normalizer.py#L38-L110), function `normalize_ocr_text()`.

---

### Stage 9: Semantic Chunking & Heading Inheritance
- **What happens:** Splits documents into slide-atomic chunks with context overlap and hierarchical section heading inheritance.
- **Technology/library used:** Python standard library (`hashlib`, `re`).
- **How it works:**
  1. `chunk_document_data()` loads the per-page extraction JSON.
  2. Each slide is initially treated as an independent semantic unit, bound to its source filename and page number.
  3. Slides exceeding the token limit are split using a **500-token sliding window** with a **50-token overlap cushion** ($W_{\text{overlap}} = 50$).
  4. **Heading Inheritance:** Tracks the most recent section heading (`3.1 Line Chart`) and prefixes it to child bullets, ensuring sub-bullets retain context.
  5. Computes a deterministic SHA-256 `text_hash` for each chunk to support incremental index caching.
- **Input:** Document extraction JSON, page numbers, and normalized text.
- **Output:** List of chunk dictionaries containing `chunk_id`, `document_id`, `page_number`, `section_title`, `text`, and `text_hash`.
- **Next step:** Dense vector embedding generation.
- **Code location:** [backend/services/NLP/document_chunking_service.py](file:///Users/mayanksmac/Desktop/capstone/backend/services/NLP/document_chunking_service.py#L76-L160), function `chunk_document_data()`.

---

### Stage 10: Incremental Dense Vector Embeddings & Index Storage
- **What happens:** Converts semantic text chunks into 384-dimensional dense vectors and appends them incrementally to the folder's vector index.
- **Technology/library used:** `sentence-transformers` (`all-MiniLM-L6-v2`), PyTorch, NumPy.
- **How it works:**
  1. Instead of recomputing vectors for every document, `index_single_document(db, document)` isolates newly extracted chunks for the uploaded document.
  2. `FolderVectorIndex.add_document_chunks()` validates chunk uniqueness via `text_hash`.
  3. `all-MiniLM-L6-v2` encodes new chunks, projecting representations onto the 384-dimensional unit hypersphere:
     $$\mathbf{e}_j = \frac{\mathbf{u}_j}{\|\mathbf{u}_j\|_2} \in \mathbb{S}^{383}, \quad \|\mathbf{e}_j\|_2 = 1.0$$
  4. Atomically appends vectors into `embeddings.npy` and metadata into `metadata.json` in the course folder's `vector_index/` directory.
- **Input:** Document text chunks and folder index path.
- **Output:** Synchronized `embeddings.npy` and `metadata.json` on disk.
- **Next step:** Incremental knowledge base integration.
- **Code location:** [backend/services/NLP/semantic_search_service.py](file:///Users/mayanksmac/Desktop/capstone/backend/services/NLP/semantic_search_service.py#L425-L480) and [backend/services/NLP/vector_index_service.py](file:///Users/mayanksmac/Desktop/capstone/backend/services/NLP/vector_index_service.py#L89-L210).

---

### Stage 10b: Incremental Knowledge Base Integration & Conflict Tracking
- **What happens:** Maps extracted facts into domain knowledge concepts, eliminates duplicates across documents, and detects factual contradictions between syllabus slide decks.
- **Technology/library used:** SQLAlchemy 2.x, `sentence-transformers`, Python `hashlib`.
- **How it works:**
  1. `KnowledgeIntegrationService.integrate_document_knowledge(db, document)` parses slide chunks into atomic statements.
  2. **Concept Synthesis:** Identifies core concepts and either retrieves or creates canonical `KnowledgeConcept` rows (e.g., *"HDFS Replication"*).
  3. **Exact Deduplication:** Hashes normalized text via SHA-256 (`compute_normalized_hash`). Duplicate items from other slides or documents simply append source provenance references (`provenance_sources_json`) without duplicating rows.
  4. **Conflict & Contradiction Detection:** Evaluates semantic similarity against existing concept items. If two items share a high lexical overlap but assert contradictory numerical thresholds, dates, or opposite polarity (e.g., *"default 3 replicas"* vs *"default 2 replicas"*), the item is flagged with `is_conflict = True` and detailed in `conflict_details`.
  5. **Audit Trail Persistence:** Records summary counts (`items_matched`, `concepts_enriched`, `conflicts_flagged`) into the SQLite `document_ingestion_logs` table.
- **Input:** Uploaded document extraction and existing folder knowledge graph.
- **Output:** Persisted `KnowledgeConcept`, `KnowledgeItem`, and `DocumentIngestionLog` records.
- **Next step:** System is fully ready for multi-turn conversational chat, adaptive quizzes, and study guides.
- **Code location:** [backend/services/NLP/knowledge_integration_service.py](file:///Users/mayanksmac/Desktop/capstone/backend/services/NLP/knowledge_integration_service.py#L380-L540).

---

### Stage 11: Study Notes Synthesis & PDF Generation
- **What happens:** Compiles all lecture materials in a folder into a 5-section academic study guide, caches the result, and exports a formatted PDF.
- **Technology/library used:** Python `hashlib`, PyMuPDF (`fitz.Story`, `fitz.DocumentWriter`), SQLite `folder_note_caches`.
- **How it works:**
  1. `NotesGenerator.generate_notes_for_folder()` calculates a composite SHA-256 hash of all completed document IDs and update timestamps:
     $$\mathcal{H}_{\text{folder}} = \text{SHA256}\left(\sum d.\text{id} \mathbin{\Vert} d.\text{updated\_at}\right)$$
  2. If $\mathcal{H}_{\text{folder}}$ exists in `folder_note_caches`, cached notes are returned instantly ($< 5\text{ ms}$).
  3. Otherwise, it synthesizes 5 academic sections:
     - *Section 1: Executive Overview & Comprehensive Syllabus Roadmap*
     - *Section 2: Core Concepts & Definitional Registry (exact page provenance)*
     - *Section 3: Comparative Analysis Matrix (Markdown comparison tables)*
     - *Section 4: Architectural Workflows & Procedural Pipelines*
     - *Section 5: High-Yield Examination Revision Points*
  4. Caches the Markdown and section JSON in SQLite.
  5. **PDF Export:** When requested via `/api/folders/{id}/notes/export-pdf`, `NotesPdfExporter` transforms the Markdown into styled academic HTML, feeds it into PyMuPDF's `fitz.Story` pagination engine, and streams an A4 binary PDF byte stream.
- **Input:** Completed folder documents, vector index chunks, or PDF export request.
- **Output:** Structured `NotesResponse` JSON and downloadable `.pdf` binary document.
- **Next step:** Rendered in frontend study notes viewer or downloaded to the user's computer.
- **Code location:** [backend/services/atml/notes_generator.py](file:///Users/mayanksmac/Desktop/capstone/backend/services/atml/notes_generator.py#L1023-L1150) and [backend/services/atml/pdf_exporter.py](file:///Users/mayanksmac/Desktop/capstone/backend/services/atml/pdf_exporter.py#L275-L306).

---

### Stage 12: Practice Quiz Synthesis & Manifold Distractor Optimization
- **What happens:** Automatically synthesizes 4-option multiple-choice diagnostic quizzes with plausible distractors grounded in lecture embeddings.
- **Technology/library used:** Python `re`, `sentence-transformers`, NumPy.
- **How it works:**
  1. `QuizGenerator.generate_quiz_for_folder()` scans chunks for factual copula patterns:
     $$\mathcal{R}_{\text{fact}} = \text{^([A-Z][\w\s\-]{2,35})\s+(?:is|are|refers to|represents|denotes)\s+([^.]+)}$$
  2. Extracts the canonical entity $E$ (e.g., `"Line Chart"`) and correct definition $A$.
  3. **Bandpass Cosine Manifold Filter:** To generate 3 plausible distractors, candidate terms from the folder are embedded using `all-MiniLM-L6-v2` and filtered:
     $$\text{ValidDistractor}(d, A) \iff 0.35 \le \cos(\mathbf{e}_d, \mathbf{e}_A) < 0.92 \land \text{Levenshtein}(d, A) > 5$$
     - *Lower bound ($\ge 0.35$):* Rejects absurd, out-of-domain terms.
     - *Upper bound ($< 0.92$):* Rejects synonyms and duplicate definitions.
  4. Shuffles options, binds slide provenance citations, and returns structured MCQs.
- **Input:** Course folder vector index and lecture text chunks.
- **Output:** `QuizResponseSchema` containing question stems, 4 options, answer indices, and pedagogical explanations.
- **Next step:** Interactive Quiz Runner modal in frontend.
- **Code location:** [backend/services/atml/quiz_generator.py](file:///Users/mayanksmac/Desktop/capstone/backend/services/atml/quiz_generator.py#L70-L240), class `QuizGenerator`.

---

### Stage 13: Ilayas AI Conversational Chat & Disambiguation
- **What happens:** Resolves conversational pronouns and ordinals, classifies intent, retrieves source context, refines answers with FLAN-T5, and streams responses to the user.
- **Technology/library used:** `chat_context_service.py` (MAQD), `hierarchical_concept_gatherer.py` (HCG), `answer_refiner.py` (`google/flan-t5-base`/`small`), `output_validator.py`.
- **How it works:**
  1. **Anaphora Resolution (MAQD):** When a user asks *"Explain the second point"* or *"What are its advantages?"*, `ChatContextService` checks the last 6 turns from SQLite `chat_messages`. It maps `"second"` $\rightarrow$ Item 2 from the previous list, rewriting the query into an explicit search string in **3.8 ms**.
  2. **Pedagogical Intent Classification:**
     - *Taxonomical ("types of X"):* Routes to `HierarchicalConceptGatherer` (HCG), which crawls all slide headers and bullet points to achieve **100% subtopic recall**.
     - *Visual ("diagram of X"):* Matches diagram bounding boxes and generates tokenized `/api/figures/{id}` image links.
     - *Definitional/Procedural:* Executes NumPy dot-product cosine similarity search.
  3. **Neural Refinement:** `AnswerRefiner` passes retrieved chunks to FLAN-T5 for natural language answer synthesis.
  4. **Grounding Guard:** `OutputValidator` validates that named entities match the source text before writing to SQLite and streaming to the browser.
- **Input:** User query string and active `conversation_id`.
- **Output:** `ChatMessageResponse` containing structured markdown answer, direct answer lead, citation list, and diagram card.
- **Next step:** Frontend renders typewriter animation, citation pills, and modal previews.
- **Code location:** [backend/services/NLP/chat_service.py](file:///Users/mayanksmac/Desktop/capstone/backend/services/NLP/chat_service.py#L190-L250) and [backend/services/NLP/chat_context_service.py](file:///Users/mayanksmac/Desktop/capstone/backend/services/NLP/chat_context_service.py#L90-L155).

---

### Stage 14: Frontend Rendering, State Persistence & Downloads
- **What happens:** The user views indexed documents, reads synthesized study guides, downloads PDFs, takes interactive practice quizzes, and chats with Ilayas AI.
- **Technology/library used:** Vanilla JavaScript (ES6+), DOM APIs, CSS3 Glassmorphism tokens (`styles.css`).
- **How it works:**
  1. **Status Polling:** `checkAndStartPolling()` polls `GET /api/folders/{id}/documents` every 3 seconds while documents are pending/processing. Once completed, the badge updates to `"Indexed"`.
  2. **Notes Reader:** Markdown is rendered into styled HTML with custom tabs, collapsible units, and citation cards.
  3. **PDF Download:** Clicking "Download PDF" issues a POST to `/api/folders/{id}/notes/export-pdf`, receiving binary PDF data and triggering a browser download blob.
  4. **Interactive Quiz Runner:** Presents one question at a time, animates selection states, calculates real-time scores, and reveals slide provenance citations for incorrect choices.
  5. **Chat View:** Streams Ilayas AI responses with a typewriter effect, attaches zoomable diagram modals, and formats comparison tables.
- **Input:** JSON responses and PDF binary streams from backend REST endpoints.
- **Output:** Interactive visual components rendered in the browser DOM.
- **Next step:** User studies, tests their knowledge, exports materials, or uploads further course decks.
- **Code location:** [frontend/app.js](file:///Users/mayanksmac/Desktop/capstone/frontend/app.js#L1267-L1490), functions `loadDocuments()`, `renderDocumentsTable()`, and UI event listeners.

---

## C. Library-to-Workflow Mapping

The following table summarizes every significant library and framework used across the actual execution path:

| Workflow Stage | Library / Technology | Version / Source | Exact Purpose in Nalanda AI | Input $\rightarrow$ Output |
| :--- | :--- | :--- | :--- | :--- |
| **Client UI & Routing** | Vanilla HTML5 / ES6 JavaScript | Browser Native | Single-Page Application state, DOM event handling, typewriter chat, quiz runner. | User gestures $\rightarrow$ HTTP REST API requests |
| **API Gateway & Routing** | FastAPI | `0.115.x` | High-performance asynchronous REST API routing, request validation, and background queues. | HTTP Requests $\rightarrow$ Validated Pydantic Schemas |
| **Server Engine** | Uvicorn | ASGI Standard | Asynchronous ASGI server hosting backend services on `127.0.0.1:8000`. | Network sockets $\rightarrow$ FastAPI application events |
| **Relational Database** | SQLite (`sqlite3`) | Standard Library | Local persistent storage for users, session tokens, course folders, documents, extractions, and chat history (`data/nalanda.db`). | SQL Queries $\rightarrow$ Relational database rows |
| **ORM & Persistence** | SQLAlchemy | `2.x` | Declarative models (`User`, `Folder`, `Document`, `DocumentExtraction`), session lifecycle, cascading deletes. | Python Objects $\rightarrow$ SQLite transactions |
| **Data Validation** | Pydantic | `v2` | Request/response schemas (`DocumentUploadResponse`, `NotesResponseSchema`, `QuizResponseSchema`). | JSON payloads $\rightarrow$ Strongly-typed Python objects |
| **Digital PDF Parsing** | PyMuPDF (`fitz`) | `>= 1.24.0` | Digital PDF page decompilation, text extraction, native table parsing (`page.find_tables`), and diagram extraction. | Digital `.pdf` file $\rightarrow$ Text, tables, figures |
| **OCR & Layout Engine** | PaddleOCR / PP-StructureV3 | `paddleocr>=2.7`, `paddlepaddle==3.2.2` | Layout analysis, OCR character recognition, and HTML table reconstruction for scanned PDF pages and images. | Scanned PNG/JPG $\rightarrow$ OCR text and HTML tables |
| **Image Verification** | Pillow (PIL) | `>= 10.0` | Verifies image dimensions, detects formats, and manages diagram image previews. | Image file bytes $\rightarrow$ Dimension & format verification |
| **Process Isolation** | Python `subprocess` | Standard Library | Sandboxed, timeout-guarded execution of `nalanda_extractor.py` without blocking the main event loop. | CLI command list $\rightarrow$ `CompletedProcess` exit code & outputs |
| **Text Normalization** | Python `re` (Regex) | Standard Library | Heals OCR split hyphens (`vi-\nsualization`), unwraps soft newlines, strips slide banners, and protects code/tables. | Fragmented raw text $\rightarrow$ Clean Markdown string |
| **Semantic Chunking** | Python Standard Library | Custom Service | Implements 500-token sliding window with 50-token overlap cushion and heading inheritance across slide boundaries. | Continuous text $\rightarrow$ Slide-atomic chunk dictionaries |
| **Vector Embeddings** | SentenceTransformers | `all-MiniLM-L6-v2` | Encodes text passages and search queries into 384-dimensional unit hypersphere vectors ($\mathbb{S}^{383}$). | Normalized text passages $\rightarrow$ 384-D float32 vectors |
| **Vector Storage & Math** | NumPy | `>= 1.24` | Serialized disk storage (`embeddings.npy`) and sub-millisecond matrix dot-product cosine similarity retrieval (`np.dot`). | Embedding matrix $\mathbf{E} \times \mathbf{q} \rightarrow$ Similarity scores |
| **Latent Autoencoders & OOD** | PyTorch (`torch.nn`) | `>= 2.0` | Undercomplete and variational autoencoders for OCR denoising and $< 2\text{ms}$ out-of-distribution anomaly rejection. | 384-D chunk embeddings $\rightarrow$ Latent bottleneck $\rightarrow$ Reconstruction error |
| **Knowledge Graph & Conflicts** | SQLAlchemy / Custom Service | `knowledge_integration_service.py` | Synthesizes `KnowledgeConcept`, deduplicates atomic `KnowledgeItem` records, flags factual conflicts, and logs audit stats. | Extracted slide chunks $\rightarrow$ Relational knowledge entities & conflict logs |
| **Discourse Resolution** | Python `re` / Custom Service | `chat_context_service.py` | Multi-Turn Anaphoric Query Disambiguation (MAQD): resolves pronouns and ordinals in 3.8 ms. | Query + Chat History $\rightarrow$ Augmented search query |
| **Topology Concept Crawl** | Python `re` / Custom Service | `hierarchical_concept_gatherer.py` | Multi-pass cross-slide heading crawl achieving 100% subtopic recall for taxonomical queries. | Course folder index $\rightarrow$ Complete subtopic taxonomy |
| **Neural Refinement** | Hugging Face Transformers | `google/flan-t5-base`/`small` | Local instruction-tuned Seq2Seq model synthesizing grounded answers from retrieved context. | Retrieved context + Query $\rightarrow$ Formatted direct answer |
| **Assessment Optimization**| NumPy / Vector Index | `quiz_generator.py` | Bandpass cosine manifold filter ($0.35 \le \cos < 0.92$) selecting plausible, non-trivial MCQ distractors. | True answer vector $\rightarrow$ 3 plausible distractors |
| **Study Notes Synthesis** | Python Standard Library | `notes_generator.py` | Compiles 5-section academic study guides with deterministic SHA-256 fingerprint caching in SQLite. | Folder document extractions $\rightarrow$ 5-section Markdown notes |
| **Academic PDF Export** | PyMuPDF (`fitz.Story`) | `>= 1.24.0` | Compiles Markdown study guides into paginated, publication-grade A4 PDF documents with custom typography. | Styled HTML notes $\rightarrow$ Binary PDF byte stream |
| **Authentication & Crypto** | Python `hashlib` / `secrets` | Standard Library | PBKDF2-HMAC-SHA256 password hashing, secure token generation, and document state fingerprinting. | Passwords & states $\rightarrow$ Cryptographic hashes |

---

## D. Error Handling & Fallback Paths

The workflow incorporates strict error recovery at each critical transition:

```
[UPLOAD STAGE]
  ├── Invalid Extension (> not in SUPPORTED_EXTENSIONS) ──► HTTP 400 Bad Request
  ├── File Size Limit (> 50 MB) ─────────────────────────► HTTP 400 File Size Exceeded
  └── Disk Write Failure ────────────────────────────────► HTTP 500 Internal Server Error

[EXTRACTION STAGE]
  ├── Subprocess Timeout (> 300 seconds) ────────────────► Document.status = 'failed' (Sanitized: "Extraction timed out")
  ├── Subprocess Non-Zero Exit Code ─────────────────────► Document.status = 'failed' (Sanitized: "Document extraction failed")
  ├── Missing Output JSON ───────────────────────────────► Document.status = 'failed' (Internal logging, safe user message)
  └── User Clicks 'Retry' in Frontend ──────────────────► POST /api/documents/{id}/retry ──► Reschedules background task

[ATML OUTLIER & INGESTION STAGE]
  ├── Autoencoder Subspace Mismatch (> high reconstruction error) ──► Flags is_mismatched_document=True in metadata; completes ingestion safely
  ├── Cross-Document Contradiction Detected ─────────────► Sets is_conflict=True & populates conflict_details JSON in KnowledgeItem
  └── Single-Document Index Serialization Contention ────► Atomically retries vector file append; leaves database record intact

[RETRIEVAL & AI STAGE]
  ├── Vector Index Missing / Desynchronized ─────────────► sync_folder_index() automatically rebuilds index from SQLite extractions
  ├── FLAN-T5 Seq2Seq Model Timeout / Memory Pressure ──► Falls back to deterministic answer_structurer.py rule templates
  ├── Anaphora Resolution Ambiguity ─────────────────────► Preserves raw query unmodified; logs non-blocking warning
  └── Quiz Generation Fact Sparsity ─────────────────────► Expands search to all folder slides or falls back to definitional sentences
```

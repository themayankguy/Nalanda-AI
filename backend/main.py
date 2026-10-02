"""Main FastAPI application for Nalanda AI."""
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.database import engine, Base
from backend.routes.auth_routes import router as auth_router
from backend.routes.folder_routes import router as folder_router
from backend.routes.document_routes import router as document_router
from backend.routes.search_routes import router as search_router

# Ensure all database tables exist on startup
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Nalanda AI Document Management API",
    description="Knowledge and document management system backend with folder-based document upload and background extraction.",
    version="1.0.0",
)

# Enable CORS for local development and web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routers
app.include_router(auth_router)
app.include_router(folder_router)
app.include_router(document_router)
app.include_router(search_router)

# Mount frontend static directory if it exists
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIR)), name="assets")

    @app.get("/", include_in_schema=False)
    def read_root():
        index_file = FRONTEND_DIR / "index.html"
        if index_file.exists():
            return FileResponse(str(index_file))
        return {"message": "Nalanda AI API is running. Frontend index.html not found."}


@app.get("/health", tags=["system"])
def health_check():
    return {"status": "healthy", "service": "Nalanda AI"}

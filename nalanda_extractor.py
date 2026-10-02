#!/usr/bin/env python3
"""Nalanda AI document ingestion pipeline.

Extracts text and tables from PDFs and images. Digital PDF pages use PyMuPDF;
scanned pages and images use PaddleOCR PP-StructureV3. Writes Markdown, TXT,
JSON metadata, extracted table HTML files, and processing statistics.
"""

# These flags must be set before importing Paddle/PaddleOCR.
import os
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "0")
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("FLAGS_enable_pir_api", "0")

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from html import escape

import fitz  # PyMuPDF
import paddle
from paddleocr import PPStructureV3

SUPPORTED_IMAGES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
BULLET_CHARS = "•●○◦▪▫‣⁃∙·"
MIN_TEXT_CHARS = 50
PDF_RENDER_SCALE = 1.5


def make_output_dirs(output_dir: Path) -> dict[str, Path]:
    dirs = {
        "root": output_dir,
        "markdown": output_dir / "markdown",
        "txt": output_dir / "txt",
        "json": output_dir / "json",
        "tables": output_dir / "tables",
        "pages": output_dir / "pages",
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    return dirs


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_bullet_only(line: str) -> bool:
    return bool(line and re.match(rf"^\s*[{re.escape(BULLET_CHARS)}]\s*$", line))


def extract_inline_bullet(line: str) -> str | None:
    match = re.match(rf"^\s*[{re.escape(BULLET_CHARS)}]\s*(.+?)\s*$", line or "")
    return match.group(1).strip() if match else None


def reconstruct_bullets(text: str) -> str:
    if not text:
        return ""
    output = []
    pending_bullet = False
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            if not pending_bullet:
                output.append("")
            continue
        if is_bullet_only(stripped):
            pending_bullet = True
            continue
        if pending_bullet:
            output.append("- " + stripped)
            pending_bullet = False
            continue
        inline = extract_inline_bullet(stripped)
        if inline:
            output.append("- " + inline)
            continue
        markdown_bullet = re.match(r"^[-*+]\s+(.+?)\s*$", stripped)
        if markdown_bullet:
            output.append("- " + markdown_bullet.group(1).strip())
            continue
        numbered = re.match(r"^(\d+)[.)]\s+(.+?)\s*$", stripped)
        if numbered:
            output.append(f"{numbered.group(1)}. {numbered.group(2).strip()}")
            continue
        alphabetic = re.match(r"^([A-Za-z])[.)]\s+(.+?)\s*$", stripped)
        if alphabetic:
            output.append(f"{alphabetic.group(1)}. {alphabetic.group(2).strip()}")
            continue
        output.append(line)
    if pending_bullet:
        output.append("-")
    return "\n".join(output)


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = reconstruct_bullets(text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return "\n".join(line.rstrip() for line in text.splitlines()).strip()


def markdown_table(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    width = max((len(row) for row in rows), default=0)
    if width == 0:
        return ""
    normalized = [row + [""] * (width - len(row)) for row in rows]
    safe = [[cell.replace("|", "\\|").replace("\n", " ") for cell in row] for row in normalized]
    lines = ["| " + " | ".join(safe[0]) + " |", "| " + " | ".join(["---"] * width) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in safe[1:])
    return "\n".join(lines)


def extract_native_pdf_tables(page) -> list[dict]:
    tables = []
    try:
        finder = page.find_tables()
        for index, table in enumerate(getattr(finder, "tables", []), start=1):
            try:
                data = table.extract() or []
            except Exception:
                data = []
            rows = [["" if cell is None else str(cell).strip() for cell in row] for row in data]
            if not rows:
                continue
            html_rows = []
            for row_index, row in enumerate(rows):
                tag = "th" if row_index == 0 else "td"
                html_rows.append("<tr>" + "".join(f"<{tag}>{escape(cell)}</{tag}>" for cell in row) + "</tr>")
            html = "<table>" + "".join(html_rows) + "</table>"
            tables.append({
                "table_index": index,
                "source": "PyMuPDF",
                "data": rows,
                "markdown": markdown_table(rows),
                "html": html,
                "bbox": list(table.bbox) if getattr(table, "bbox", None) else None,
            })
    except Exception as exc:
        print(f"  Native table detection skipped: {str(exc)[:160]}")
    return tables


def run_ppstructure(pipeline, image_path: Path, page_number: int | None = None) -> dict:
    started = time.perf_counter()
    results = pipeline.predict(input=str(image_path))
    markdown_parts, tables, raw_results = [], [], []
    for result in results:
        try:
            markdown_data = result.markdown
            if isinstance(markdown_data, dict):
                text = markdown_data.get("markdown_texts", "")
                if isinstance(text, list):
                    text = "\n\n".join(map(str, text))
            else:
                text = str(markdown_data or "")
        except Exception:
            text = ""
        if text:
            markdown_parts.append(text)

        try:
            result_json = result.json
            if callable(result_json):
                result_json = result_json()
            if isinstance(result_json, str):
                result_json = json.loads(result_json)
            if not isinstance(result_json, dict):
                result_json = {}
        except Exception:
            result_json = {}
        raw_results.append(result_json)

        for table_index, table in enumerate(result_json.get("table_res_list", []), start=1):
            table_ocr = table.get("table_ocr_pred", {}) or {}
            tables.append({
                "page": page_number,
                "table_index": table_index,
                "source": "PP-StructureV3",
                "html": table.get("pred_html", "") or "",
                "cell_texts": table_ocr.get("rec_texts", []) or [],
                "cell_scores": table_ocr.get("rec_scores", []) or [],
                "cell_boxes": table.get("cell_box_list", []) or [],
            })

    return {
        "text": clean_text("\n\n".join(markdown_parts)),
        "tables": tables,
        "raw_results": raw_results,
        "processing_time": time.perf_counter() - started,
    }


def render_pdf_page(page, output_path: Path) -> None:
    pixmap = page.get_pixmap(matrix=fitz.Matrix(PDF_RENDER_SCALE, PDF_RENDER_SCALE), alpha=False)
    pixmap.save(str(output_path))


def process_pdf(pdf_path: Path, pipeline, dirs: dict[str, Path]) -> dict:
    started = time.perf_counter()
    document = fitz.open(str(pdf_path))
    pages = []
    direct_time_total = 0.0
    ocr_time_total = 0.0
    try:
        total_pages = len(document)
        print(f"\nProcessing PDF: {pdf_path.name} ({total_pages} pages)")
        for index, page in enumerate(document):
            page_number = index + 1
            page_started = time.perf_counter()
            text = (page.get_text("text") or "").strip()
            direct_time = time.perf_counter() - page_started

            if len(text) >= MIN_TEXT_CHARS:
                text = clean_text(text)
                tables = extract_native_pdf_tables(page)
                for table in tables:
                    table["page"] = page_number
                direct_time_total += direct_time
                method, used_ocr, elapsed = "PyMuPDF", False, direct_time
            else:
                image_path = dirs["pages"] / f"{pdf_path.stem}_page_{page_number}.png"
                render_pdf_page(page, image_path)
                ocr = run_ppstructure(pipeline, image_path, page_number)
                text, tables = ocr["text"], ocr["tables"]
                ocr_time_total += ocr["processing_time"]
                method, used_ocr, elapsed = "PP-StructureV3", True, ocr["processing_time"]

            print(f"  Page {page_number}/{total_pages}: {method}; {len(text)} chars; {len(tables)} tables; {elapsed:.2f}s")
            pages.append({
                "page": page_number,
                "method": method,
                "ocr": used_ocr,
                "characters": len(text),
                "processing_time": elapsed,
                "text": text,
                "tables": tables,
            })
    finally:
        document.close()
    return {
        "filename": pdf_path.name,
        "file_type": "pdf",
        "file_sha256": file_sha256(pdf_path),
        "total_pages": len(pages),
        "pages": pages,
        "total_direct_time": direct_time_total,
        "total_ocr_time": ocr_time_total,
        "total_processing_time": time.perf_counter() - started,
    }


def process_image(image_path: Path, pipeline) -> dict:
    started = time.perf_counter()
    print(f"\nProcessing image: {image_path.name}")
    result = run_ppstructure(pipeline, image_path, page_number=1)
    print(f"  OCR: {len(result['text'])} chars; {len(result['tables'])} tables; {result['processing_time']:.2f}s")
    return {
        "filename": image_path.name,
        "file_type": "image",
        "file_sha256": file_sha256(image_path),
        "total_pages": 1,
        "pages": [{
            "page": 1,
            "method": "PP-StructureV3",
            "ocr": True,
            "characters": len(result["text"]),
            "processing_time": result["processing_time"],
            "text": result["text"],
            "tables": result["tables"],
        }],
        "total_direct_time": 0.0,
        "total_ocr_time": result["processing_time"],
        "total_processing_time": time.perf_counter() - started,
    }


def build_markdown(result: dict) -> str:
    sections = [f"# {Path(result['filename']).stem}"]
    for page in result.get("pages", []):
        page_content = [f"## Page {page['page']}"]
        text = clean_text(page.get("text", ""))
        if text:
            page_content.append(text)
        for table_index, table in enumerate(page.get("tables", []), start=1):
            page_content.append(f"### Table {table_index}")
            if table.get("markdown"):
                page_content.append(table["markdown"])
            elif table.get("html"):
                page_content.append(table["html"])
            elif table.get("cell_texts"):
                page_content.append("\n".join(map(str, table["cell_texts"])))
        sections.append("\n\n".join(page_content))
    return "\n\n---\n\n".join(sections).strip() + "\n"


def save_outputs(result: dict, dirs: dict[str, Path]) -> dict:
    stem = Path(result["filename"]).stem
    markdown = build_markdown(result)
    txt = "\n\n".join(clean_text(page.get("text", "")) for page in result.get("pages", []))
    markdown_path = dirs["markdown"] / f"{stem}.md"
    txt_path = dirs["txt"] / f"{stem}.txt"
    json_path = dirs["json"] / f"{stem}.json"
    markdown_path.write_text(markdown, encoding="utf-8")
    txt_path.write_text(txt.strip() + "\n", encoding="utf-8")
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    table_files = []
    for page in result.get("pages", []):
        for index, table in enumerate(page.get("tables", []), start=1):
            html = table.get("html", "")
            if not html:
                continue
            table_path = dirs["tables"] / f"{stem}_page_{page['page']}_table_{index}.html"
            document = f"<!doctype html><html><head><meta charset='utf-8'><title>Nalanda table</title><style>body{{font-family:Arial,sans-serif;padding:24px}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #555;padding:8px;text-align:left}}</style></head><body>{html}</body></html>"
            table_path.write_text(document, encoding="utf-8")
            table_files.append(str(table_path))

    return {
        "markdown_path": str(markdown_path),
        "txt_path": str(txt_path),
        "json_path": str(json_path),
        "table_files": table_files,
    }


def build_pipeline(device: str):
    print("\nLoading PP-StructureV3 model. First run may download model files...")
    started = time.perf_counter()
    pipeline = PPStructureV3(
        lang="en",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        use_table_recognition=True,
        use_formula_recognition=False,
        device=device,
    )
    print(f"Model ready in {time.perf_counter() - started:.2f}s")
    return pipeline


def get_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        return "gpu:0" if paddle.device.is_compiled_with_cuda() else "cpu"
    except Exception:
        return "cpu"


def collect_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        files = [input_path]
    elif input_path.is_dir():
        files = sorted(path for path in input_path.rglob("*") if path.is_file())
    else:
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    return [p for p in files if p.suffix.lower() == ".pdf" or p.suffix.lower() in SUPPORTED_IMAGES]


def main() -> int:
    global PDF_RENDER_SCALE, MIN_TEXT_CHARS
    parser = argparse.ArgumentParser(description="Nalanda AI PDF/image text and table extraction")
    parser.add_argument("input", nargs="?", help="Path to a PDF/image file or a folder containing files; if omitted, you will be prompted")
    parser.add_argument("--output", default="nalanda_output", help="Output directory (default: nalanda_output)")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "gpu:0"], help="Inference device")
    parser.add_argument("--render-scale", type=float, default=PDF_RENDER_SCALE, help="PDF OCR rendering scale; higher may improve OCR but takes longer")
    parser.add_argument("--min-text-chars", type=int, default=MIN_TEXT_CHARS, help="Selectable PDF characters threshold before OCR")
    args = parser.parse_args()

    # Interactive mode: running `python nalanda_extractor.py` asks for the input.
    if not args.input:
        print("\nNALANDA AI - INPUT SELECTION")
        print("Enter the path to a PDF/image file or a folder containing files.")
        print("Tip: you can drag a file or folder from Finder into Terminal to paste its path.")
        try:
            selected_input = input("Input path: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nInput cancelled.")
            return 2
        # Remove matching quotes often included when copying paths from Finder.
        if len(selected_input) >= 2 and selected_input[0] == selected_input[-1] and selected_input[0] in {"'", '"'}:
            selected_input = selected_input[1:-1].strip()
        if not selected_input:
            print("No input path provided.", file=sys.stderr)
            return 2
        args.input = selected_input

    PDF_RENDER_SCALE = args.render_scale
    MIN_TEXT_CHARS = args.min_text_chars

    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    dirs = make_output_dirs(output_dir)
    files = collect_files(input_path)
    if not files:
        print(f"No supported PDF/image files found in: {input_path}", file=sys.stderr)
        return 2

    device = get_device(args.device)
    print("=" * 70)
    print("NALANDA AI - DOCUMENT INGESTION")
    print("=" * 70)
    print(f"Python: {sys.version.split()[0]} | Paddle: {paddle.__version__} | Device: {device}")
    print(f"Input files: {len(files)} | Output: {output_dir}")
    pipeline = build_pipeline(device)

    successful = 0
    failures = []
    for file_path in files:
        try:
            result = process_pdf(file_path, pipeline, dirs) if file_path.suffix.lower() == ".pdf" else process_image(file_path, pipeline)
            result["saved"] = save_outputs(result, dirs)
            successful += 1
            print("  Saved Markdown:", result["saved"]["markdown_path"])
            print("  Saved TXT:     ", result["saved"]["txt_path"])
            print("  Saved JSON:    ", result["saved"]["json_path"])
        except Exception as exc:
            failures.append({"file": str(file_path), "error": str(exc)})
            print(f"ERROR processing {file_path.name}: {exc}", file=sys.stderr)

    summary = {
        "files_found": len(files),
        "files_processed": successful,
        "failures": failures,
        "output_directory": str(output_dir),
    }
    (output_dir / "processing_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n" + "=" * 70)
    print(f"Finished: {successful}/{len(files)} files processed")
    print(f"Outputs: {output_dir}")
    if failures:
        print(f"Failed files: {len(failures)} (see processing_summary.json)")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

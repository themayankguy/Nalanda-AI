"""Service for compiling ATML Markdown Study Notes into publication-grade academic PDF documents."""
import html
import os
import re
import tempfile
import logging
from typing import Optional
import fitz

logger = logging.getLogger("nalanda.pdf_exporter")


class NotesPdfExporter:
    """Compiles Markdown study guides into high-resolution, paginated PDF documents using PyMuPDF."""

    def __init__(self):
        pass

    def markdown_to_html(self, markdown_text: str, folder_name: str) -> str:
        """Converts Markdown text into structured, print-ready HTML with academic styling."""
        lines = markdown_text.split("\n")
        html_lines = []
        in_list = False
        in_table = False
        in_code_block = False
        code_block_lines = []
        table_header_done = False

        for line in lines:
            line_s = line.strip()

            # Handle Fenced Code Blocks (```lang ... ```)
            if line_s.startswith("```"):
                if in_list:
                    html_lines.append("</ul>")
                    in_list = False
                if in_table:
                    html_lines.append("</table>")
                    in_table = False
                    table_header_done = False

                if not in_code_block:
                    in_code_block = True
                    code_block_lines = []
                else:
                    in_code_block = False
                    escaped_code = html.escape("\n".join(code_block_lines))
                    html_lines.append(f"<pre><code>{escaped_code}</code></pre>")
                continue

            if in_code_block:
                code_block_lines.append(line)
                continue

            if not line_s:
                if in_list:
                    html_lines.append("</ul>")
                    in_list = False
                if in_table:
                    html_lines.append("</table>")
                    in_table = False
                    table_header_done = False
                continue

            # Markdown Table Parsing
            if line_s.startswith("|") and line_s.endswith("|"):
                if in_list:
                    html_lines.append("</ul>")
                    in_list = False
                cells = [c.strip() for c in line_s.split("|")[1:-1]]
                if all(set(c).issubset({"-", ":", " "}) for c in cells):
                    table_header_done = True
                    continue
                if not in_table:
                    html_lines.append('<table border="1" cellspacing="0" cellpadding="4">')
                    in_table = True
                    table_header_done = False

                tag = "th" if not table_header_done else "td"
                cell_html = "".join(
                    f"<{tag}>{re.sub(r'\\*\\*(.*?)\\*\\*', r'<b>\\1</b>', c)}</{tag}>"
                    for c in cells
                )
                html_lines.append(f"<tr>{cell_html}</tr>")
                continue
            elif in_table:
                html_lines.append("</table>")
                in_table = False
                table_header_done = False

            # Markdown Headers
            if line_s.startswith("# "):
                if in_list:
                    html_lines.append("</ul>")
                    in_list = False
                html_lines.append(f"<h1>{line_s[2:]}</h1>")
            elif line_s.startswith("## "):
                if in_list:
                    html_lines.append("</ul>")
                    in_list = False
                html_lines.append(f"<h2>{line_s[3:]}</h2>")
            elif line_s.startswith("### "):
                if in_list:
                    html_lines.append("</ul>")
                    in_list = False
                html_lines.append(f"<h3>{line_s[4:]}</h3>")
            elif line_s.startswith("#### "):
                if in_list:
                    html_lines.append("</ul>")
                    in_list = False
                html_lines.append(f"<h4>{line_s[5:]}</h4>")
            elif line_s.startswith("> "):
                if in_list:
                    html_lines.append("</ul>")
                    in_list = False
                cleaned = line_s[2:].replace("**", "<b>").replace("**", "</b>")
                html_lines.append(f"<blockquote>{cleaned}</blockquote>")
            elif line_s.startswith("- ") or line_s.startswith("* "):
                if not in_list:
                    html_lines.append("<ul>")
                    in_list = True
                content = line_s[2:]
                content = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", content)
                content = re.sub(r"\*(.*?)\*", r"<i>\1</i>", content)
                content = re.sub(r"`([^`]+)`", r"<code>\1</code>", content)
                html_lines.append(f"<li>{content}</li>")
            elif re.match(r"^\d+\.\s", line_s):
                if in_list:
                    html_lines.append("</ul>")
                    in_list = False
                num_part = line_s.split(".")[0] + "."
                rest = line_s[len(num_part):].strip()
                rest = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", rest)
                rest = re.sub(r"\*(.*?)\*", r"<i>\1</i>", rest)
                html_lines.append(f"<p><b>{num_part}</b> {rest}</p>")
            else:
                if in_list:
                    html_lines.append("</ul>")
                    in_list = False
                content = line_s
                content = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", content)
                content = re.sub(r"\*(.*?)\*", r"<i>\1</i>", content)
                content = re.sub(r"`([^`]+)`", r"<code>\1</code>", content)
                html_lines.append(f"<p>{content}</p>")

        if in_list:
            html_lines.append("</ul>")
        if in_table:
            html_lines.append("</table>")
        if in_code_block:
            escaped_code = html.escape("\n".join(code_block_lines))
            html_lines.append(f"<pre><code>{escaped_code}</code></pre>")

        body = "\n".join(html_lines)
        styled_html = f"""
        <style>
        body {{
            font-family: sans-serif;
            font-size: 8.5pt;
            color: #1e293b;
            line-height: 1.38;
        }}
        h1 {{
            color: #1e3a8a;
            font-size: 14pt;
            margin: 6pt 0 4pt 0;
            border-bottom: 2px solid #3b82f6;
            padding-bottom: 3pt;
        }}
        h2 {{
            color: #1e40af;
            font-size: 11.5pt;
            margin: 10pt 0 4pt 0;
            border-bottom: 1px solid #cbd5e1;
            padding-bottom: 2pt;
        }}
        h3 {{
            color: #2563eb;
            font-size: 10pt;
            margin: 8pt 0 3pt 0;
        }}
        h4 {{
            color: #3b82f6;
            font-size: 9pt;
            margin: 6pt 0 2pt 0;
        }}
        p {{
            margin: 2pt 0 3pt 0;
        }}
        ul {{
            margin: 2pt 0 4pt 12pt;
            padding: 0;
        }}
        li {{
            margin-bottom: 2pt;
        }}
        blockquote {{
            background: #f0fdf4;
            border-left: 3px solid #059669;
            padding: 3pt 6pt;
            margin: 3pt 0;
            color: #065f46;
            font-size: 8pt;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 6pt 0;
            font-size: 7pt;
            table-layout: fixed;
        }}
        th, td {{
            border: 1px solid #cbd5e1;
            padding: 3pt 4pt;
            vertical-align: top;
            word-wrap: break-word;
            overflow-wrap: break-word;
        }}
        th {{
            background: #f1f5f9;
            color: #0f172a;
            font-weight: bold;
            text-align: left;
        }}
        th:nth-child(1), td:nth-child(1) {{ width: 17%; }}
        th:nth-child(2), td:nth-child(2) {{ width: 13%; }}
        th:nth-child(3), td:nth-child(3) {{ width: 22%; }}
        th:nth-child(4), td:nth-child(4) {{ width: 22%; }}
        th:nth-child(5), td:nth-child(5) {{ width: 26%; }}
        b {{
            color: #0f172a;
        }}
        pre {{
            background: #f8fafc;
            border: 1px solid #cbd5e1;
            border-radius: 3pt;
            padding: 4pt 6pt;
            margin: 4pt 0;
            font-family: monospace;
            font-size: 7.5pt;
            color: #0f172a;
            white-space: pre-wrap;
            word-break: break-all;
        }}
        code {{
            background: #f1f5f9;
            color: #1e40af;
            padding: 1pt 3pt;
            font-family: monospace;
            font-size: 7.5pt;
        }}
        </style>
        {body}
        """
        return styled_html

    def export_markdown_to_pdf_bytes(self, markdown_text: str, folder_name: str) -> bytes:
        """Renders Markdown study notes into a binary PDF byte stream."""
        html_content = self.markdown_to_html(markdown_text, folder_name)

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            story = fitz.Story(html=html_content)
            writer = fitz.DocumentWriter(tmp_path)
            # A4 standard dimensions: 595.28 x 841.89 points
            page_rect = fitz.Rect(0, 0, 595, 842)
            content_rect = fitz.Rect(36, 36, 595 - 36, 842 - 36)

            more = 1
            while more:
                device = writer.begin_page(page_rect)
                more, _ = story.place(content_rect)
                story.draw(device)
                writer.end_page()

            writer.close()

            with open(tmp_path, "rb") as f:
                pdf_bytes = f.read()

            logger.info("Successfully exported PDF for folder '%s' (%d bytes)", folder_name, len(pdf_bytes))
            return pdf_bytes
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

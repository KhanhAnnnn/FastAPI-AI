import json
import re
import sys
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from app.config import RAG_MAX_EXTRACTION_CHARS, RAG_MAX_PDF_PAGES

_PDF_OBJECT_REFERENCE = re.compile(rb"/Root\s+\d+\s+\d+\s+R\b")


class WorkerFailure(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _extract(source: Path, output_dir: Path) -> dict[str, Any]:
    try:
        reader = PdfReader(source, strict=True)
        if reader.is_encrypted:
            raise WorkerFailure("encrypted_pdf", "Password-protected PDFs are unsupported.")
        total_pages = len(reader.pages)
    except WorkerFailure:
        raise
    except Exception as exc:
        raise WorkerFailure("invalid_pdf", "PDF structure could not be parsed.") from exc

    if total_pages == 0:
        raise WorkerFailure("invalid_pdf", "PDF contains no pages.")
    if total_pages > RAG_MAX_PDF_PAGES:
        raise WorkerFailure("page_limit", "PDF exceeds the configured page limit.")

    page_metadata: list[dict[str, Any]] = []
    extracted_pages: list[str] = []
    total_characters = 0
    output_offset = 0
    try:
        for page_number, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text()
                content = page.get_contents()
                has_content = content is not None and bool(content.get_data().strip())
            except Exception as exc:
                raise WorkerFailure(
                    "page_extraction_error",
                    f"Text extraction failed on page {page_number}.",
                ) from exc

            normalized_text = text if text and text.strip() else ""
            if normalized_text:
                if extracted_pages:
                    output_offset += 2
                text_start = output_offset
                text_end = text_start + len(normalized_text)
                extracted_pages.append(normalized_text)
                output_offset = text_end
                total_characters += len(normalized_text)
                if total_characters > RAG_MAX_EXTRACTION_CHARS:
                    raise WorkerFailure(
                        "character_limit",
                        "Extracted text exceeds the configured character limit.",
                    )
                extraction_status = "text"
            else:
                text_start = None
                text_end = None
                extraction_status = "no_text_layer" if has_content else "empty"

            page_metadata.append(
                {
                    "page_number": page_number,
                    "has_text": bool(normalized_text),
                    "extraction_status": extraction_status,
                    "character_count": len(normalized_text),
                    "text_start": text_start,
                    "text_end": text_end,
                }
            )
    except WorkerFailure:
        raise
    except Exception as exc:
        raise WorkerFailure(
            "page_extraction_error", "A PDF page could not be processed."
        ) from exc

    text_output = "\n\n".join(extracted_pages)
    (output_dir / "extracted.txt").write_text(text_output, encoding="utf-8", newline="")
    metadata = {
        "total_pages": total_pages,
        "pages_with_text": sum(page["has_text"] for page in page_metadata),
        "pages_without_text": sum(not page["has_text"] for page in page_metadata),
        "pages": page_metadata,
    }
    (output_dir / "pages.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="",
    )
    return metadata


def main() -> int:
    try:
        _extract(Path(sys.argv[1]), Path(sys.argv[2]))
        return 0
    except WorkerFailure as exc:
        print(json.dumps({"code": exc.code, "message": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

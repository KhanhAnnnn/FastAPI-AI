import hashlib
import json
from io import BytesIO
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from pypdf import PdfReader, PdfWriter

from app.main import app
from app.rag import extraction_service, storage


def make_pdf(pages: list[str | None]) -> bytes:
    objects: list[bytes] = []
    page_ids = [3 + index * 2 for index in range(len(pages))]
    content_ids = [page_id + 1 for page_id in page_ids]
    objects.extend(
        [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            (
                f"<< /Type /Pages /Kids [{' '.join(f'{page_id} 0 R' for page_id in page_ids)}] "
                f"/Count {len(pages)} >>"
            ).encode("ascii"),
        ]
    )

    unicode_map: dict[int, str] = {}
    next_code = 1
    for page_id, content_id, text in zip(page_ids, content_ids, pages, strict=True):
        objects.append(
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Resources << /Font << /F1 {3 + len(pages) * 2} 0 R >> >> "
                f"/Contents {content_id} 0 R >>"
            ).encode("ascii")
        )
        if text is None:
            stream = b"q 0 0 10 10 re f Q"
        elif text == "":
            stream = b""
        else:
            character_codes = bytearray()
            for character in text:
                unicode_map[next_code] = character
                character_codes.extend(next_code.to_bytes(2, "big"))
                next_code += 1
            stream = (
                b"BT /F1 12 Tf 72 720 Td <"
                + character_codes.hex().upper().encode("ascii")
                + b"> Tj ET"
            )
        objects.append(
            b"<< /Length "
            + str(len(stream)).encode("ascii")
            + b" >>\nstream\n"
            + stream
            + b"\nendstream"
        )

    font_id = 3 + len(pages) * 2
    cid_font_id = font_id + 1
    cmap_id = font_id + 2
    objects.extend(
        [
            (
                f"<< /Type /Font /Subtype /Type0 /BaseFont /TestFont "
                f"/Encoding /Identity-H /DescendantFonts [{cid_font_id} 0 R] "
                f"/ToUnicode {cmap_id} 0 R >>"
            ).encode("ascii"),
            (
                b"<< /Type /Font /Subtype /CIDFontType2 /BaseFont /TestFont "
                b"/CIDSystemInfo << /Registry (Adobe) /Ordering (Identity) "
                b"/Supplement 0 >> /DW 1000 >>"
            ),
        ]
    )
    cmap_lines = [
        b"/CIDInit /ProcSet findresource begin",
        b"12 dict begin",
        b"begincmap",
        b"/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def",
        b"/CMapName /Test-UCS def",
        b"/CMapType 2 def",
        b"1 begincodespacerange",
        b"<0000> <FFFF>",
        b"endcodespacerange",
        f"{len(unicode_map)} beginbfchar".encode("ascii"),
    ]
    for code, character in unicode_map.items():
        cmap_lines.append(
            f"<{code:04X}> <{character.encode('utf-16-be').hex().upper()}>".encode(
                "ascii"
            )
        )
    cmap_lines.extend([b"endbfchar", b"endcmap", b"CMapName currentdict /CMap defineresource pop", b"end", b"end"])
    cmap = b"\n".join(cmap_lines)
    objects.append(
        b"<< /Length "
        + str(len(cmap)).encode("ascii")
        + b" >>\nstream\n"
        + cmap
        + b"\nendstream"
    )

    result = bytearray(b"%PDF-1.7\n")
    offsets = [0]
    for number, body in enumerate(objects, start=1):
        offsets.append(len(result))
        result.extend(f"{number} 0 obj\n".encode("ascii"))
        result.extend(body)
        result.extend(b"\nendobj\n")
    xref_offset = len(result)
    result.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    result.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        result.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    result.extend(b"trailer\n")
    result.extend(f"<< /Size {len(objects) + 1} /Root 1 0 R >>\n".encode("ascii"))
    result.extend(f"startxref\n{xref_offset}\n%%EOF\n".encode("ascii"))
    return bytes(result)


def encrypt_pdf(data: bytes) -> bytes:
    writer = PdfWriter(clone_from=PdfReader(BytesIO(data), strict=True))
    writer.encrypt("test-password")
    result = BytesIO()
    writer.write(result)
    return result.getvalue()


@pytest.fixture
def rag_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "rag-storage"
    monkeypatch.setattr(storage, "RAG_UPLOAD_ROOT", root)
    monkeypatch.setattr(extraction_service, "RAG_UPLOAD_ROOT", root)
    return root


def save_source(root: Path, data: bytes, filename: str = "source.pdf") -> str:
    document_id = str(uuid4())
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{document_id}.pdf").write_bytes(data)
    (root / f"{document_id}.upload.json").write_text(
        json.dumps(
            {
                "document_id": document_id,
                "original_filename": filename,
                "stored_filename": f"{document_id}.pdf",
            }
        ),
        encoding="utf-8",
    )
    return document_id


def extract(client: TestClient, document_id: str):
    return client.post(f"/rag/documents/{document_id}/extract")


def test_extracts_single_page_and_persists_utf8_text(rag_root: Path) -> None:
    source = make_pdf(["Hello from PDF"])
    document_id = save_source(rag_root, source, "single.pdf")

    response = extract(TestClient(app), document_id)

    assert response.status_code == 200, response.text
    result = response.json()
    assert result["total_pages"] == 1
    assert result["pages_with_text"] == 1
    assert result["pages"][0]["extraction_status"] == "text"
    output = (rag_root / result["text_filename"]).read_bytes()
    assert output.decode("utf-8") == "Hello from PDF"
    assert hashlib.sha256((rag_root / f"{document_id}.pdf").read_bytes()).hexdigest() == hashlib.sha256(source).hexdigest()


def test_extracts_multiple_pages_and_page_offsets(rag_root: Path) -> None:
    document_id = save_source(rag_root, make_pdf(["Page one", "Page two", "Page three"]))

    response = extract(TestClient(app), document_id)

    assert response.status_code == 200, response.text
    result = response.json()
    assert result["total_pages"] == 3
    assert result["pages_with_text"] == 3
    text = (rag_root / result["text_filename"]).read_text(encoding="utf-8")
    assert text == "Page one\n\nPage two\n\nPage three"
    for page in result["pages"]:
        assert text[page["text_start"] : page["text_end"]].startswith(
            f"Page {['one', 'two', 'three'][page['page_number'] - 1]}"
        )


def test_extracts_vietnamese_unicode_without_loss(rag_root: Path) -> None:
    expected = "Tiếng Việt có dấu: Trường học, Nguyễn Ái Quốc"
    document_id = save_source(rag_root, make_pdf([expected]), "tieng-viet.pdf")

    response = extract(TestClient(app), document_id)

    assert response.status_code == 200, response.text
    result = response.json()
    assert (rag_root / result["text_filename"]).read_text(encoding="utf-8") == expected
    assert result["pages"][0]["character_count"] == len(expected)


def test_distinguishes_blank_page_and_page_without_extractable_text(
    rag_root: Path,
) -> None:
    document_id = save_source(rag_root, make_pdf(["Readable", "", None]))

    response = extract(TestClient(app), document_id)

    assert response.status_code == 200, response.text
    result = response.json()
    assert result["total_pages"] == 3
    assert result["pages_with_text"] == 1
    assert result["pages_without_text"] == 2
    assert [page["extraction_status"] for page in result["pages"]] == [
        "text",
        "empty",
        "no_text_layer",
    ]
    assert result["pages"][2]["has_text"] is False


def test_structurally_malformed_pdf_returns_client_error(rag_root: Path) -> None:
    document_id = save_source(rag_root, b"%PDF-1.7\nnot a valid PDF")

    response = extract(TestClient(app), document_id)

    assert response.status_code == 400
    assert "PDF structure" in response.json()["detail"]


def test_encrypted_pdf_is_reported_as_unsupported(rag_root: Path) -> None:
    document_id = save_source(
        rag_root, encrypt_pdf(make_pdf(["Encrypted text"]))
    )

    response = extract(TestClient(app), document_id)

    assert response.status_code == 422
    assert "Password-protected" in response.json()["detail"]


def test_page_limit_is_enforced(rag_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_MAX_PDF_PAGES", "1")
    document_id = save_source(rag_root, make_pdf(["one", "two"]))

    response = extract(TestClient(app), document_id)

    assert response.status_code == 413
    assert "page limit" in response.json()["detail"]
    assert not list(rag_root.glob("*.txt"))


def test_extracted_character_limit_is_enforced(
    rag_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RAG_MAX_EXTRACTION_CHARS", "5")
    document_id = save_source(rag_root, make_pdf(["longer than limit"]))

    response = extract(TestClient(app), document_id)

    assert response.status_code == 413
    assert "character limit" in response.json()["detail"]
    assert not list(rag_root.glob("*.txt"))


def test_extraction_timeout_stops_worker(
    rag_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(extraction_service, "RAG_EXTRACTION_TIMEOUT_SECONDS", 0.001)
    document_id = save_source(rag_root, make_pdf(["timeout fixture"]))

    response = extract(TestClient(app), document_id)

    assert response.status_code == 408
    assert "time limit" in response.json()["detail"]
    assert not list(rag_root.glob("*.txt"))


def test_same_document_extraction_is_rejected_while_locked(rag_root: Path) -> None:
    document_id = save_source(rag_root, make_pdf(["locked document"]))
    lock = extraction_service._get_lock(document_id)
    assert lock.acquire(blocking=False)
    try:
        response = extract(TestClient(app), document_id)
    finally:
        with extraction_service._document_locks_guard:
            lock.release()
            extraction_service._document_locks.pop(document_id, None)

    assert response.status_code == 409
    assert "already running" in response.json()["detail"]
    assert not list(rag_root.glob("*.txt"))


@pytest.mark.parametrize("document_id", ["00000000-0000-0000-0000-000000000000", "../../secret"])
def test_missing_or_malformed_document_id_cannot_escape_storage(
    rag_root: Path, document_id: str
) -> None:
    response = extract(TestClient(app), document_id)
    assert response.status_code == 404
    assert list(rag_root.iterdir()) == [] if rag_root.exists() else True


def test_extraction_can_be_repeated_without_overwriting_previous_text(
    rag_root: Path,
) -> None:
    document_id = save_source(rag_root, make_pdf(["Repeatable text"]))
    client = TestClient(app)

    first = extract(client, document_id)
    second = extract(client, document_id)

    assert first.status_code == second.status_code == 200
    first_result, second_result = first.json(), second.json()
    assert first_result["extraction_id"] != second_result["extraction_id"]
    assert first_result["text_filename"] != second_result["text_filename"]
    assert (rag_root / first_result["text_filename"]).read_text(encoding="utf-8") == "Repeatable text"
    manifest = json.loads(
        (rag_root / second_result["metadata_filename"]).read_text(encoding="utf-8")
    )
    assert manifest["document_id"] == document_id
    assert manifest["original_filename"] == "source.pdf"
    assert manifest["source_filename"] == f"{document_id}.pdf"
    assert manifest["extraction_id"] == second_result["extraction_id"]
    assert manifest["pages"][0]["page_number"] == 1
    assert manifest["pages"][0]["extraction_status"] == "text"


def test_metadata_and_counts_match_extracted_text(rag_root: Path) -> None:
    document_id = save_source(rag_root, make_pdf(["Alpha", "", "Beta"]))

    response = extract(TestClient(app), document_id)

    assert response.status_code == 200
    result = response.json()
    metadata = json.loads((rag_root / result["metadata_filename"]).read_text(encoding="utf-8"))
    assert metadata["total_pages"] == len(metadata["pages"]) == 3
    assert metadata["pages_with_text"] == 2
    assert metadata["pages_without_text"] == 1
    assert metadata["pages"][0]["character_count"] == 5
    assert metadata["pages"][1]["extraction_status"] == "empty"
    assert metadata["pages"][2]["character_count"] == 4
    text = (rag_root / result["text_filename"]).read_text(encoding="utf-8")
    assert text == "Alpha\n\nBeta"
    assert text[metadata["pages"][0]["text_start"] : metadata["pages"][0]["text_end"]] == "Alpha"
    assert text[metadata["pages"][2]["text_start"] : metadata["pages"][2]["text_end"]] == "Beta"


def test_upload_regression_metadata_supports_extraction(rag_root: Path) -> None:
    response = TestClient(app).post(
        "/rag/documents/upload",
        files={"file": ("uploaded.pdf", make_pdf(["Uploaded source"]), "application/pdf")},
    )
    assert response.status_code == 201
    document_id = response.json()["document_id"]
    upload_metadata = json.loads(
        (rag_root / f"{document_id}.upload.json").read_text(encoding="utf-8")
    )
    assert upload_metadata["original_filename"] == "uploaded.pdf"
    extraction_response = extract(TestClient(app), document_id)
    assert extraction_response.status_code == 200
    assert extraction_response.json()["original_filename"] == "uploaded.pdf"


def test_existing_generation_routes_and_health_remain_registered() -> None:
    client = TestClient(app)
    assert client.get("/health").json() == {"status": "ok"}
    paths = client.get("/openapi.json").json()["paths"]
    assert {
        "/generate/text",
        "/generate/audio",
        "/generate/image",
        "/generate/video",
        "/generate/3d",
        "/rag/documents/upload",
    }.issubset(paths)


def test_fixture_generator_emits_strict_parseable_pdfs() -> None:
    for contents in (make_pdf(["test"]), make_pdf(["", None]), encrypt_pdf(make_pdf(["secret"]))):
        reader = PdfReader(BytesIO(contents), strict=True)
        if reader.is_encrypted:
            assert reader.is_encrypted
        else:
            assert len(reader.pages) >= 1

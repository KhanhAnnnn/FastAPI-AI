from __future__ import annotations

from uuid import uuid4
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.rag.chat_schemas import ChatRequest
from app.rag.search_schemas import SearchResponse, SearchResultItem
from app.rag.services.chat_service import ChatService, get_chat_service
from app.rag.services.retrieval_service import compute_lexical_overlap


class DummyRetrievalService:
    def __init__(self, results: list[SearchResultItem] | None = None) -> None:
        self.results = results or []

    async def search(self, **kwargs) -> SearchResponse:
        return SearchResponse(
            query=kwargs.get("query", ""),
            results=self.results,
            total_results=len(self.results),
        )


def test_chat_validation_rejects_invalid_inputs() -> None:
    client = TestClient(app)

    # Empty query
    assert client.post("/rag/chat", json={"query": ""}).status_code == 422
    assert client.post("/rag/chat", json={"query": "   \n\t  "}).status_code == 422

    # Query too long
    assert client.post("/rag/chat", json={"query": "x" * 2001}).status_code == 422

    # top_k out of bounds
    assert client.post("/rag/chat", json={"query": "hi", "top_k": 0}).status_code == 422
    assert client.post("/rag/chat", json={"query": "hi", "top_k": 21}).status_code == 422

    # score_threshold out of bounds
    assert client.post("/rag/chat", json={"query": "hi", "score_threshold": -0.1}).status_code == 422
    assert client.post("/rag/chat", json={"query": "hi", "score_threshold": 1.1}).status_code == 422


@pytest.mark.anyio
async def test_chat_refusal_without_evidence_english() -> None:
    retrieval = DummyRetrievalService(results=[])
    chat_svc = ChatService(retrieval_service=retrieval)

    request = ChatRequest(query="What is the capital of Mars?")
    response = await chat_svc.chat(request)

    assert response.sources == []
    assert response.retrieval_count == 0
    assert "could not find sufficient information" in response.answer.lower()
    assert response.latency_ms["generation_ms"] == 0.0


@pytest.mark.anyio
async def test_chat_refusal_without_evidence_vietnamese() -> None:
    retrieval = DummyRetrievalService(results=[])
    chat_svc = ChatService(retrieval_service=retrieval)

    request = ChatRequest(query="Thủ đô của Sao Hỏa là gì?")
    response = await chat_svc.chat(request)

    assert response.sources == []
    assert response.retrieval_count == 0
    assert "không tìm thấy đủ thông tin" in response.answer.lower()
    assert response.latency_ms["generation_ms"] == 0.0


@pytest.mark.anyio
async def test_chat_grounded_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    doc_id = uuid4()
    item = SearchResultItem(
        document_id=doc_id,
        page_number=2,
        chunk_id="chunk-001",
        score=0.88,
        text="FastAPI is a modern, fast web framework for building APIs with Python.",
    )
    retrieval = DummyRetrievalService(results=[item])
    chat_svc = ChatService(retrieval_service=retrieval)

    # Mock model_manager.run to verify generation call without spinning up full PyTorch CUDA
    async def mock_run(model_name, loader, func):
        assert model_name == "text"
        return "FastAPI is a fast web framework for Python (Trang 2).", {"duration_ms": 120.0}

    from app.core.model_manager import model_manager
    monkeypatch.setattr(model_manager, "run", mock_run)

    request = ChatRequest(query="What is FastAPI?")
    response = await chat_svc.chat(request)

    assert response.retrieval_count == 1
    assert len(response.sources) == 1
    assert response.sources[0].document_id == doc_id
    assert response.sources[0].page_number == 2
    assert "Trang 2" in response.answer
    assert response.latency_ms["generation_ms"] >= 0
    assert response.latency_ms["total_ms"] >= 0


def test_compute_lexical_overlap_generalized_and_unbiased() -> None:
    # 1. Unbiased towards arbitrary prefixes: query with "Trung tâm Aurora" vs without
    q1 = "Trung tâm Aurora sử dụng những công nghệ nào trong nền tảng NovaDesk?"
    q2 = "Nền tảng NovaDesk sử dụng những công nghệ nào?"
    target_section = (
        "3. Hệ thống hỗ trợ học tập\n"
        "Aurora vận hành nền tảng NovaDesk để quản lý bài tập và phản hồi. "
        "Hệ thống sử dụng FastAPI cho API backend, PostgreSQL lưu dữ liệu nghiệp vụ và Qdrant phục vụ tìm kiếm tài liệu theo ngữ nghĩa."
    )
    unrelated_section = (
        "1. Tổng quan\n"
        "Trung tâm Aurora là một dự án giáo dục công nghệ giả định tại thành phố Đà Nẵng. "
        "Người phụ trách dự án là Nguyễn Minh Khoa."
    )

    s1_target = compute_lexical_overlap(q1, target_section)
    s1_unrelated = compute_lexical_overlap(q1, unrelated_section)
    assert s1_target > s1_unrelated, f"Target section ({s1_target}) must outscore unrelated section ({s1_unrelated})"

    s2_target = compute_lexical_overlap(q2, target_section)
    s2_unrelated = compute_lexical_overlap(q2, unrelated_section)
    assert s2_target > s2_unrelated

    # 2. Score must always be bounded strictly in [0.0, 1.0]
    assert 0.0 <= compute_lexical_overlap("random query", "") <= 1.0
    assert 0.0 <= compute_lexical_overlap("", "some text") <= 1.0


def test_extract_focused_context_preserves_multientity_relations() -> None:
    from app.rag.services.chat_service import extract_focused_context

    sample_doc = (
        "HỒ SƠ DỰ ÁN GIẢ ĐỊNH: TRUNG TÂM AURORA\n\n"
        "1. Tổng quan\n"
        "Trung tâm Aurora là một dự án giáo dục công nghệ tại Đà Nẵng. Khởi động ngày 12/03/2024.\n\n"
        "2. Quy mô đào tạo năm 2025\n"
        "Tổng cộng trung tâm mở 18 lớp và ghi nhận 292 học viên hoàn thành trong năm 2025. "
        "Chỉ tiêu ban đầu là 300 học viên, kết quả thấp hơn mục tiêu 8 học viên.\n\n"
        "3. Hệ thống hỗ trợ học tập\n"
        "Aurora vận hành nền tảng NovaDesk để quản lý bài tập và phản hồi. "
        "Hệ thống sử dụng FastAPI cho API backend, PostgreSQL lưu dữ liệu nghiệp vụ và Qdrant phục vụ tìm kiếm tài liệu theo ngữ nghĩa. "
        "Tài liệu PDF được lưu trong vùng riêng; vector embedding được tạo bằng một mô hình chạy trên CPU.\n\n"
        "4. Sự cố năm 2025\n"
        "Ngày 18/11/2025, NovaDesk gặp sự cố làm 12 tài liệu ở trạng thái FAILED trong 47 phút."
    )

    # NovaDesk query MUST select Section 3 in full with FastAPI, PostgreSQL, and Qdrant
    q_novadesk = "Trung tâm Aurora sử dụng những công nghệ nào trong nền tảng NovaDesk và mỗi công nghệ đảm nhiệm vai trò gì?"
    ctx_novadesk = extract_focused_context(sample_doc, q_novadesk, max_chars=1200)

    assert "NovaDesk" in ctx_novadesk
    assert "FastAPI" in ctx_novadesk
    assert "API backend" in ctx_novadesk
    assert "PostgreSQL" in ctx_novadesk
    assert "lưu dữ liệu nghiệp vụ" in ctx_novadesk
    assert "Qdrant" in ctx_novadesk
    assert "tìm kiếm tài liệu theo ngữ nghĩa" in ctx_novadesk

    # Student count query MUST select Section 2
    q_students = "Tổng số học viên hoàn thành năm 2025 thấp hơn mục tiêu bao nhiêu người?"
    ctx_students = extract_focused_context(sample_doc, q_students, max_chars=1200)
    assert "292" in ctx_students
    assert "300" in ctx_students
    assert "8 học viên" in ctx_students

    # Incident query MUST select Section 4
    q_incident = "Sự cố ngày 18/11/2025 làm bao nhiêu tài liệu bị lỗi trong bao lâu?"
    ctx_incident = extract_focused_context(sample_doc, q_incident, max_chars=1200)
    assert "18/11/2025" in ctx_incident
    assert "12 tài liệu" in ctx_incident
    assert "47 phút" in ctx_incident


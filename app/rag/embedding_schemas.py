from typing import Literal
from uuid import UUID

from pydantic import BaseModel


class EmbeddedDocument(BaseModel):
    document_id: UUID
    extraction_id: UUID
    run_id: UUID
    status: Literal["completed"]
    model_id: str
    model_revision: str
    code_revision: str
    chunk_count: int
    embedding_dimension: int
    device: Literal["cpu"]
    model_load_seconds: float
    embedding_seconds: float
    artifact_directory: str
    manifest_filename: str

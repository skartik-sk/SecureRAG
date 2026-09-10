from pydantic import BaseModel, Field


class WorkspaceCreate(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    description: str | None = None
    is_private: bool = False


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    conversation_id: str | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]
    conversation_id: str
    refused: bool


class WorkspaceOut(BaseModel):
    slug: str
    name: str
    description: str | None
    is_private: bool
    documents: int
    chunks: int


class DocumentOut(BaseModel):
    id: str
    filename: str
    status: str
    chunk_count: int
    error: str | None
    source: str

# pydantic_models.py
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime

class ModelName(str, Enum):
    # Add ALL your available models, ordered from fastest to slowest
    PHI3_MINI = "phi3:mini"        # ← New: Fastest (3.8B)
    GEMMA2_2B = "gemma2:2b"        # ← New: Very fast (2B)  
    LLAMA3_2_1B = "llama3.2:1b"    # ← Fast (1.3B)
    LLAMA3_1 = "llama3.1"          # ← Slower (4.9B)
    GPT_OSS_20B = "gpt-oss:20b"    # ← Slowest (13B)

# Set the default to your FASTEST model
class QueryInput(BaseModel):
    question: str
    session_id: str = Field(default=None)
    model: ModelName = Field(default=ModelName.PHI3_MINI)  # ← Default to fastest

class QueryResponse(BaseModel):
    answer: str
    session_id: str
    model: ModelName

class DocumentInfo(BaseModel):
    id: int
    filename: str
    upload_timestamp: datetime

class DeleteFileRequest(BaseModel):
    file_id: int
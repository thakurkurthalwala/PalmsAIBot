from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime

class ModelName(str, Enum):
     # FREE LOCAL MODELS (via Ollama)
    LLAMA3 = "llama3"
    LLAMA3_1 = "llama3.1"  
    MISTRAL = "mistral"
    GEMMA = "gemma"
    PHI3 = "phi3"
    # Recommended Paid Models (via OpenAI)
    GPT4 = "gpt-4"
    GPT4_O = "gpt-4o"
    GPT4_O_MINI = "gpt-4o-mini"

class QueryInput(BaseModel):
    question: str
    session_id: str = Field(default=None)
    model: ModelName = Field(default=ModelName.GPT4_O_MINI)

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
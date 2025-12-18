from typing import List, Optional,TypedDict, Literal
from pydantic import BaseModel,Field
from datetime import datetime

class UserCreate(BaseModel):
    name:str
    email:str
    password:str
    role:str
    department:str

class UserOut(BaseModel):
    id: int
    name: str
    email: str
    role: str
    department: Optional[str] = None
    class Config:
        from_attributes = True

class ShowUser(BaseModel):
    name:str
    email:str
    role:str
    department:str 
    class Config():
        from_attributes = True
        
class TokenData(BaseModel):
    email: Optional[str] = None
    role: Optional[str] = None
    department: Optional[str] = None
    

class LoginRequest(BaseModel):
    email: str
    password: str
    
class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    
class DocumentUploadResponse(BaseModel):
    """Response when document is uploaded"""
    id: int
    filename: str
    original_filename: str
    file_type: str
    department: str
    uploaded_by: int
    file_size: int
    chunk_count: int
    vectorized: str
    uploaded_at: datetime
    message: str  # Success message
    description: Optional[str] = None
    class Config:
        from_attributes = True


class DocumentOut(BaseModel):
    """Single document info"""
    id: int
    filename: str
    original_filename: str
    file_type: str
    department: str
    description: str  # NEW
    uploaded_by: int
    file_size: int
    chunk_count: int
    vectorized: str
    uploaded_at: datetime
    
    class Config:
        from_attributes = True


class DocumentList(BaseModel):
    """List of documents response"""
    documents: List[DocumentOut]
    total: int
    accessible_departments: List[str]


class ChatRequest(BaseModel):
    """User's chat question"""
    query: str
    
    # @validator('query')
    def query_not_empty(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Query cannot be empty')
        return v.strip()


class ChatResponse(BaseModel):
    """Bot's chat response"""
    query: str
    response: str
    query_type: str  # "sql" or "rag"
    sources: List[str]  # List of document names used
    department: str  # Which department data was used
    sql_query: Optional[str] = None  # If SQL query was executed, show it
    data_preview: Optional[List[dict]] = None  # First few rows of SQL result

class QueryType(BaseModel):
    """Structured output from LLM for query classification"""
    type: Literal["sql", "rag"] = Field(
        description="Type of query: 'sql' for data analysis questions, 'general' for other questions"
    )  # Must be either "sql" or "rag"
    confidence: float = Field(ge=0.0, le=1.0)  # 0.0 to 1.0
    reasoning: str 
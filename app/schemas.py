from typing import List, Optional
from pydantic import BaseModel

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
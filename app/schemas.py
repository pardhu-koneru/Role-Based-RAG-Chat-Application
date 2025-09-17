from typing import List, Optional
from pydantic import BaseModel

class User(BaseModel):
    name:str
    email:str
    password:str
    role:str
    department:str

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
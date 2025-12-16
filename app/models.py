from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from database import Base

class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    password = Column(String, nullable=False)  # store bcrypt hashed
    role = Column(String, default="user")    # e.g., "HR", "Finance"
    department = Column(String, nullable=True)
    
    chats = relationship("ChatHistory", back_populates="user")




class Document(Base):
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)  # Saved filename on disk
    original_filename = Column(String(255), nullable=False)  # User's original name
    file_type = Column(String(50), nullable=False)  # md, xlsx, csv, pdf
    file_path = Column(String(500), nullable=False)  # Full path to file
    department = Column(String(50), nullable=False)  # finance, marketing, hr, engineering, general
    uploaded_by = Column(Integer, ForeignKey("users.id"), nullable=False)
    file_size = Column(Integer)  # Size in bytes
    chunk_count = Column(Integer, default=0)  # Number of chunks created
    vectorized = Column(String(10), default="no")  # "yes" or "no"
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    description = Column(String(500), nullable=True)  # Optional schema information
    # Relationship to User
    uploader = relationship("User", backref="documents")
    
    
class ChatHistory(Base):
    """Chat history table - stores user conversations"""
    __tablename__ = "chat_history"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    query = Column(String(1000), nullable=False)  # User's question
    response = Column(String(5000), nullable=False)  # Bot's answer
    query_type = Column(String(50))  # "rag" or "sql"
    sources = Column(String(1000))  # Document names used (comma separated)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="chats")
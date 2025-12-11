from sqlalchemy import Column, Integer, String, ForeignKey
from .database import Base
from sqlalchemy.orm import relationship


class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    password = Column(String, nullable=False)  # store bcrypt hashed
    role = Column(String, default="user")    # e.g., "HR", "Finance"
    department = Column(String, nullable=True)
    documents = relationship("Document", back_populates="owner")


class Document(Base):
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    content = Column(String)
    department = Column(String, index=True)
    owner_id = Column(Integer, ForeignKey("users.id"))  # keep this!
    owner = relationship("User", back_populates="documents")# which department this doc belongs to
   
    
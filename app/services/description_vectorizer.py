"""
Description Vectorization Service
Save as: app/services/description_vectorizer.py

This vectorizes ONLY the descriptions (not the CSV content)
"""

from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_core.documents import Document as LangchainDocument
from langchain_ollama import OllamaEmbeddings
from pathlib import Path
from config import settings
from models import Document


class DescriptionVectorizer:
    """Vectorize document descriptions for retrieval"""
    
    def __init__(self):
        print("🔧 Initializing Description Vectorizer...")
        
        # Use Ollama for free local embeddings
        self.embeddings = OllamaEmbeddings(
            model=settings.OLLAMA_MODEL,
            base_url=settings.OLLAMA_BASE_URL
        )
        print(f"✅ Ollama embeddings initialized: {settings.OLLAMA_MODEL}")
        
        # Vector store directory
        self.vectorstore_dir = Path("vectorstores")
        self.vectorstore_dir.mkdir(exist_ok=True)
    
    
    def vectorize_description(self, document: Document):
        """
        Vectorize ONLY the description of a document
        This is used for retrieval when user asks SQL queries
        """
        print(f"\n📝 Vectorizing description for: {document.original_filename}")
        print(f"📄 Description: {document.description[:100]}...")
        
        # Create LangChain document with description + metadata
        langchain_doc = LangchainDocument(
            page_content=document.description,  # Only description, not CSV content!
            metadata={
                "document_id": document.id,
                "filename": document.original_filename,
                "file_path": document.file_path,
                "department": document.department,
                "file_type": document.file_type
            }
        )
        
        # Get or create vectorstore for department
        vectorstore_path = self.vectorstore_dir / f"{document.department}_descriptions"
        
        if vectorstore_path.exists():
            print(f"📂 Adding to existing description store")
            vectorstore = Chroma(
                persist_directory=str(vectorstore_path),
                embedding_function=self.embeddings
            )
            vectorstore.add_documents([langchain_doc])
        else:
            print(f"📂 Creating new description store")
            vectorstore = Chroma.from_documents(
                documents=[langchain_doc],
                embedding=self.embeddings,
                persist_directory=str(vectorstore_path)
            )
        
        print(f"✅ Description vectorized!\n")
    
    
    def search_relevant_files(self, department: str, query: str, top_k: int = 2):
        """
        Search for relevant CSV/XLSX files based on query
        Returns file paths and metadata
        """
        vectorstore_path = self.vectorstore_dir / f"{department}_descriptions"
        
        if not vectorstore_path.exists():
            print(f"⚠️  No description store for {department}")
            return []
        
        print(f"\n🔍 Searching descriptions in {department}...")
        print(f"📝 Query: {query}")
        
        vectorstore = Chroma(
            persist_directory=str(vectorstore_path),
            embedding_function=self.embeddings
        )
        
        # Search for similar descriptions
        results = vectorstore.similarity_search(query, k=top_k)
        
        files = []
        for doc in results:
            files.append({
                "document_id": doc.metadata["document_id"],
                "filename": doc.metadata["filename"],
                "file_path": doc.metadata["file_path"],
                "description": doc.page_content
            })
        
        print(f"✅ Found {len(files)} relevant files\n")
        return files
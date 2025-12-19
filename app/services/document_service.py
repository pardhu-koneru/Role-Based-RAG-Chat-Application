"""
Document Service - Handles vectorization

This file:
1. Splits documents into chunks
2. Creates embeddings using Gemini
3. Stores in ChromaDB vector database
"""

# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
# from langchain.schema import Document as LangchainDocument
from langchain_core.documents import Document as LangchainDocument

from pathlib import Path
from typing import List

from services.file_parser import FileParser
from models import Document
from config import settings


class DocumentService:
    """Service to process and vectorize documents"""
    
    def __init__(self):
        print("🔧 Initializing Document Service...")
        
        # 1. Setup Gemini embeddings
        self.embeddings =OllamaEmbeddings(
            model="nomic-embed-text",
            base_url="http://localhost:11434"
        )
        print("✅ Ollama embeddings initialized")
        
        # 2. Setup text splitter (breaks text into chunks)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Each chunk = 1000 characters
            chunk_overlap=200,  # Overlap 200 chars between chunks
            length_function=len,
        )
        print("✅ Text splitter initialized")
        
        # 3. Create vectorstore directory
        self.vectorstore_dir = Path("vectorstores")
        self.vectorstore_dir.mkdir(exist_ok=True)
        print(f"✅ Vectorstore directory: {self.vectorstore_dir}\n")
    
    
    def process_and_vectorize(self, document: Document) -> int:
        """
        Main function: Process document and store in vector database
        
        Steps:
        1. Parse file (convert to text)
        2. Split into chunks
        3. Create embeddings
        4. Store in ChromaDB
        
        Returns:
            int: Number of chunks created
        """
        
        print(f"\n{'='*60}")
        print(f"🔄 PROCESSING: {document.original_filename}")
        print(f"📁 Department: {document.department}")
        print(f"{'='*60}\n")
        
        # STEP 1: Parse file and get text content
        print("Step 1: Reading file...")
        text_content, dataframe = FileParser.parse_file(document.file_path)
        print(f"📝 Extracted {len(text_content)} characters\n")
        
        # STEP 2: Split text into chunks
        print("Step 2: Splitting into chunks...")
        chunks = self.text_splitter.split_text(text_content)
        print(f"✂️  Created {len(chunks)} chunks\n")
        
        # STEP 3: Create LangChain Document objects with metadata
        print("Step 3: Adding metadata to chunks...")
        langchain_docs = []
        
        for i, chunk in enumerate(chunks):
            doc = LangchainDocument(
                page_content=chunk,
                metadata={
                    "source": document.original_filename,
                    "document_id": document.id,
                    "department": document.department,
                    "chunk_index": i,
                    "file_type": document.file_type
                }
            )
            langchain_docs.append(doc)
        
        print(f"✅ Added metadata to {len(langchain_docs)} chunks\n")
        
        # STEP 4: Store in vector database
        print("Step 4: Storing in vector database...")
        vectorstore_path = self.vectorstore_dir / f"{document.department}_vectorstore"
        
        # Check if vectorstore already exists for this department
        if vectorstore_path.exists():
            # Add to existing vectorstore
            print(f"📂 Adding to existing vectorstore: {vectorstore_path}")
            vectorstore = Chroma(
                persist_directory=str(vectorstore_path),
                embedding_function=self.embeddings
            )
            vectorstore.add_documents(langchain_docs)
            
        else:
            # Create new vectorstore
            print(f"📂 Creating new vectorstore: {vectorstore_path}")
            vectorstore = Chroma.from_documents(
                documents=langchain_docs,
                embedding=self.embeddings,
                persist_directory=str(vectorstore_path)
            )
        
        print(f"✅ Vectorization complete!\n")
        print(f"{'='*60}\n")
        
        return len(chunks)
    
    
    def get_vectorstore(self, department: str):
        """
        Get vector database for a specific department
        
        Args:
            department: 'finance', 'hr', 'marketing', etc.
            
        Returns:
            ChromaDB vectorstore or None
        """
        vectorstore_path = self.vectorstore_dir / f"{department}_vectorstore"
        
        if not vectorstore_path.exists():
            print(f"⚠️  No vectorstore found for {department}")
            return None
        
        print(f"📂 Loading vectorstore: {department}")
        return Chroma(
            persist_directory=str(vectorstore_path),
            embedding_function=self.embeddings
        )
    
    
    def search_similar_chunks(self, department: str, query: str, top_k: int = 5):
        """
        Search for similar chunks in vector database
        
        Args:
            department: Which department's data to search
            query: User's question
            top_k: How many chunks to return
            
        Returns:
            List of (chunk_text, metadata) tuples
        """
        vectorstore = self.get_vectorstore(department)
        
        if not vectorstore:
            return []
        
        print(f"🔍 Searching in {department} vectorstore...")
        print(f"📝 Query: {query}")
        print(f"🎯 Top {top_k} results\n")
        
        # Perform similarity search using the retriever
        retriever = vectorstore.as_retriever(
            search_type="similarity",  # "similarity" or "mmr" (Maximum Marginal Relevance)
            search_kwargs={"k": top_k}  # Number of documents to return
        )
        
        # Invoke the retriever to get actual documents
        docs = retriever.invoke(query)
        
        # Extract text and metadata
        chunks = []
        for doc in docs:
            chunks.append({
                "text": doc.page_content,
                "source": doc.metadata.get("source", "Unknown"),
                "chunk_index": doc.metadata.get("chunk_index", 0)
            })
        
        print(f"✅ Found {len(chunks)} relevant chunks\n")
        return chunks
    
    
    def delete_document_from_vectorstore(self, document: Document):
        """
        Delete document's chunks from vector database
        """
        vectorstore = self.get_vectorstore(document.department)
        
        if not vectorstore:
            print(f"⚠️  No vectorstore to delete from")
            return
        
        print(f"🗑️  Deleting chunks from vectorstore...")
        
        # Get all chunks belonging to this document
        results = vectorstore.get(
            where={"document_id": document.id}
        )
        
        if results and results['ids']:
            vectorstore.delete(ids=results['ids'])
            print(f"✅ Deleted {len(results['ids'])} chunks")
        else:
            print(f"⚠️  No chunks found to delete")


    
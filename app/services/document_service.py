"""
Document Service - Handles vectorization

This file:
1. Splits documents into chunks (optimized for markdown and text files)
2. Creates embeddings using Ollama
3. Stores in ChromaDB vector database
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document as LangchainDocument

from pathlib import Path
from typing import List, Tuple

from services.file_parser import FileParser
from models import Document
from config import settings


class DocumentService:
    """Service to process and vectorize documents"""
    
    def __init__(self):
        print("🔧 Initializing Document Service...")
        
        # 1. Setup Ollama embeddings
        self.embeddings = OllamaEmbeddings(
            model="nomic-embed-text",
            base_url="http://localhost:11434"
        )
        print("✅ Ollama embeddings initialized")
        
        # 2. Setup text splitter with optimized parameters
        # Chunk size: 800-1000 chars (balanced for context and efficiency)
        # Overlap: 250 chars (~25-30%) to maintain semantic continuity
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=900,
            chunk_overlap=250,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]  # Prioritize meaningful boundaries
        )
        print("✅ Text splitter initialized (900 chars, 250 overlap)")
        
        # 3. Setup Markdown header splitter for .md files
        # Headers act as semantic boundaries for markdown documents
        self.markdown_header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
                ("####", "Header 4"),
            ]
        )
        print("✅ Markdown header splitter initialized")
        
        # 4. Create vectorstore directory
        self.vectorstore_dir = Path("vectorstores")
        self.vectorstore_dir.mkdir(exist_ok=True)
        print(f"✅ Vectorstore directory: {self.vectorstore_dir}\n")
    
    
    def _split_markdown(self, text_content: str, filename: str) -> List[Tuple[str, dict]]:
        """
        Smart splitting for markdown files using header-based strategy
        
        Args:
            text_content: Raw markdown text
            filename: Original filename
            
        Returns:
            List of (chunk_text, metadata) tuples
        """
        print("  📊 Markdown splitting strategy: Header-aware two-tier approach")
        
        # TIER 1: Split by markdown headers to preserve document structure
        try:
            markdown_docs = self.markdown_header_splitter.split_text(text_content)
            print(f"  ✂️  Tier 1: Split into {len(markdown_docs)} header sections")
        except Exception as e:
            print(f"  ⚠️  Header splitting failed, using recursive splitting: {str(e)}")
            markdown_docs = []
        
        chunks_with_metadata = []
        
        # TIER 2: Apply recursive splitting to large header sections
        for doc in markdown_docs:
            # Check if this section needs further splitting
            if len(doc.page_content) > 1200:  # If larger than optimal chunk size
                # Further split large sections
                sub_chunks = self.text_splitter.split_text(doc.page_content)
                
                # Get header context from metadata
                header_context = []
                if "Header 1" in doc.metadata:
                    header_context.append(doc.metadata["Header 1"])
                if "Header 2" in doc.metadata:
                    header_context.append(doc.metadata["Header 2"])
                if "Header 3" in doc.metadata:
                    header_context.append(doc.metadata["Header 3"])
                if "Header 4" in doc.metadata:
                    header_context.append(doc.metadata["Header 4"])
                
                for chunk in sub_chunks:
                    chunks_with_metadata.append((
                        chunk,
                        {
                            "section_path": " > ".join(header_context) if header_context else "Root",
                            "header_level": len(header_context)
                        }
                    ))
            else:
                # Keep smaller sections as-is with header context
                header_context = []
                if "Header 1" in doc.metadata:
                    header_context.append(doc.metadata["Header 1"])
                if "Header 2" in doc.metadata:
                    header_context.append(doc.metadata["Header 2"])
                if "Header 3" in doc.metadata:
                    header_context.append(doc.metadata["Header 3"])
                if "Header 4" in doc.metadata:
                    header_context.append(doc.metadata["Header 4"])
                
                chunks_with_metadata.append((
                    doc.page_content,
                    {
                        "section_path": " > ".join(header_context) if header_context else "Root",
                        "header_level": len(header_context)
                    }
                ))
        
        # Fallback: If header splitting didn't work, use recursive splitting
        if not chunks_with_metadata:
            print(f"  ℹ️  Using recursive fallback for markdown file")
            chunks = self.text_splitter.split_text(text_content)
            chunks_with_metadata = [(chunk, {"section_path": "Root", "header_level": 0}) for chunk in chunks]
        
        print(f"  ✅ Markdown splitting complete: {len(chunks_with_metadata)} final chunks")
        return chunks_with_metadata
    
    
    def _split_text(self, text_content: str, filename: str) -> List[Tuple[str, dict]]:
        """
        Standard splitting for .txt and other text files
        
        Args:
            text_content: Raw text content
            filename: Original filename
            
        Returns:
            List of (chunk_text, metadata) tuples
        """
        print("  📊 Text file splitting strategy: Recursive character splitting (optimized)")
        
        chunks = self.text_splitter.split_text(text_content)
        print(f"  ✂️  Split into {len(chunks)} chunks")
        
        chunks_with_metadata = [
            (chunk, {"section_path": "Root", "header_level": 0}) 
            for chunk in chunks
        ]
        
        return chunks_with_metadata
    
    def process_and_vectorize(self, document: Document) -> int:
        """
        Main function: Process document and store in vector database
        
        Steps:
        1. Parse file (convert to text)
        2. Split into chunks using intelligent strategy
        3. Create embeddings
        4. Store in ChromaDB
        
        Returns:
            int: Number of chunks created
        """
        
        print(f"\n{'='*60}")
        print(f"🔄 PROCESSING: {document.original_filename}")
        print(f"📁 Department: {document.department}")
        print(f"📄 File Type: {document.file_type}")
        print(f"{'='*60}\n")
        
        # STEP 1: Parse file and get text content
        print("Step 1: Reading file...")
        text_content, dataframe = FileParser.parse_file(document.file_path)
        print(f"📝 Extracted {len(text_content)} characters\n")
        
        # STEP 2: Split text into chunks with intelligent strategy
        print("Step 2: Splitting into chunks...")
        
        # Determine splitting strategy based on file type
        if document.file_type.lower() == '.md' or document.original_filename.lower().endswith('.md'):
            print("➡️  Using Markdown-optimized splitting strategy\n")
            chunks_with_metadata = self._split_markdown(text_content, document.original_filename)
        else:
            print("➡️  Using Standard text splitting strategy\n")
            chunks_with_metadata = self._split_text(text_content, document.original_filename)
        
        # STEP 3: Create LangChain Document objects with metadata
        print("Step 3: Adding metadata to chunks...")
        langchain_docs = []
        
        for i, (chunk_text, chunk_meta) in enumerate(chunks_with_metadata):
            doc = LangchainDocument(
                page_content=chunk_text,
                metadata={
                    "source": document.original_filename,
                    "document_id": document.id,
                    "department": document.department,
                    "chunk_index": i,
                    "file_type": document.file_type,
                    "section_path": chunk_meta.get("section_path", "Root"),
                    "header_level": chunk_meta.get("header_level", 0)
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
        
        return len(langchain_docs)
    
    
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
        Search for similar chunks in vector database with enhanced metadata
        
        Args:
            department: Which department's data to search
            query: User's question
            top_k: How many chunks to return
            
        Returns:
            List of chunks with enhanced metadata
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
        
        # Extract text and metadata with section context
        chunks = []
        for doc in docs:
            chunks.append({
                "text": doc.page_content,
                "source": doc.metadata.get("source", "Unknown"),
                "section_path": doc.metadata.get("section_path", "Root"),
                "header_level": doc.metadata.get("header_level", 0),
                "chunk_index": doc.metadata.get("chunk_index", 0),
                "file_type": doc.metadata.get("file_type", "unknown")
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


    
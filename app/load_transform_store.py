# enhanced_load_transform_store.py
from fastapi import HTTPException
from sqlalchemy.orm import Session
from typing import List, Dict, Any
from . import models
import logging
import hashlib
import os
import re
from dotenv import load_dotenv

# LangChain imports
from langchain.schema import Document as LangChainDocument
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from pydantic import BaseModel

load_dotenv()

# Environment variables
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_data")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")

class ChatRequest(BaseModel):
    query: str
    max_chunks: int = 10  # Increased for admin queries
    similarity_threshold: float = 0.3

class ChatResponse(BaseModel):
    query: str
    department: str
    relevant_chunks: List[str]
    similarity_scores: List[float]
    response: str
    total_chunks_found: int
    collection_stats: Dict[str, Any]

class EnhancedLangChainDocumentService:
    """Enhanced service for comprehensive markdown processing"""
    
    def __init__(self):
        self.persist_directory = CHROMA_DB_PATH
        
        # Better embedding model for technical content
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Setup markdown-aware text splitters
        self.setup_text_splitters()
        
        os.makedirs(self.persist_directory, exist_ok=True)
        logging.info("Enhanced LangChain Document Service initialized")
    
    def setup_text_splitters(self):
        """Setup optimized text splitters for FinTech documentation"""
        # Markdown header splitter - captures section hierarchy
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "H1"),
                ("##", "H2"), 
                ("###", "H3"),
                ("####", "H4"),
                ("#####", "H5")
            ],
            strip_headers=False  # Keep headers for context
        )
        
        # Recursive splitter for large sections
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            separators=[
                "\n\n### ",  # Section breaks
                "\n\n#### ", # Subsection breaks  
                "\n\n",      # Paragraph breaks
                "\n* ",      # List items
                "\n",        # Line breaks
                ". ",        # Sentence breaks
                " "          # Word breaks
            ],
            chunk_size=600,      # Smaller chunks for precision
            chunk_overlap=50,    # Reduced overlap
            length_function=len,
            keep_separator=True  # Preserve structure
        )
    
    def preprocess_markdown_content(self, content: str) -> str:
        """Enhanced preprocessing for technical documentation"""
        # Clean excessive whitespace but preserve structure
        content = re.sub(r'\n{4,}', '\n\n\n', content)
        
        # Preserve code blocks and tables
        content = re.sub(r'``````', r'\n``````\n', content, flags=re.DOTALL)
        
        # Clean up table formatting while preserving structure
        content = re.sub(r'\|\s*\|\s*\|', '| |', content)
        content = re.sub(r'\|\s*-+\s*\|', '|---|', content)
        
        # Normalize bullet points
        content = re.sub(r'^\s*[\*\-\+]\s+', '* ', content, flags=re.MULTILINE)
        
        # Clean up numbered lists
        content = re.sub(r'^\s*(\d+\.)\s+', r'\1 ', content, flags=re.MULTILINE)
        
        return content.strip()
    
    def _format_headers_as_string(self, headers_dict: Dict) -> str:
        """Convert header dictionary to string format for ChromaDB compatibility"""
        if not headers_dict:
            return ""
        
        # Format as "H1: Title > H2: Subtitle > H3: Section"
        header_parts = []
        for level, title in headers_dict.items():
            header_parts.append(f"{level}: {title}")
        
        return " > ".join(header_parts)
    
    def load_documents_from_db(self, current_user, db: Session) -> List[LangChainDocument]:
        """Load documents with department awareness for admin aggregation"""
        try:
            # Import here to avoid circular imports
            
            
            # Get documents based on user role
            if current_user.role == "admin":
                docs = db.query(models.Document).all()
                logging.info(f"Admin user - loading {len(docs)} documents from all departments")
            else:
                docs = db.query(models.Document).filter(
                    models.Document.department == current_user.department
                ).all()
                logging.info(f"Regular user - loading {len(docs)} documents from {current_user.department}")
            
            if not docs:
                return []
            
            # Process each document separately with department context
            langchain_docs = []
            for doc in docs:
                # Preprocess content
                clean_content = self.preprocess_markdown_content(doc.content)
                
                # Create document with enhanced metadata
                langchain_doc = LangChainDocument(
                    page_content=clean_content,
                    metadata={
                        "id": doc.id,
                        "department": doc.department,
                        "title": getattr(doc, 'title', f'{doc.department}_Document_{doc.id}'),
                        "created_at": str(getattr(doc, 'created_at', '')),
                        "source": f"{doc.department}_doc_{doc.id}",
                        "content_type": "technical_markdown",
                        "content_length": len(clean_content),
                        "is_admin_view": current_user.role == "admin"
                    }
                )
                langchain_docs.append(langchain_doc)
            
            logging.info(f"Processed {len(langchain_docs)} documents")
            return langchain_docs
            
        except Exception as e:
            logging.error(f"Error loading documents from database: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    
    def chunk_documents_enhanced(self, documents: List[LangChainDocument]) -> List[LangChainDocument]:
        """Enhanced chunking optimized for technical documentation"""
        try:
            all_chunks = []
            
            for doc in documents:
                logging.info(f"Processing document from {doc.metadata['department']}: {doc.metadata['title']}")
                
                # First pass: Split by markdown headers
                try:
                    header_chunks = self.markdown_splitter.split_text(doc.page_content)
                except Exception as e:
                    logging.warning(f"Markdown splitting failed, using recursive: {str(e)}")
                    header_chunks = [LangChainDocument(
                        page_content=doc.page_content,
                        metadata={}
                    )]
                
                # Second pass: Process each header section
                for i, header_chunk in enumerate(header_chunks):
                    chunk_text = header_chunk.page_content
                    
                    # Convert header metadata to string format - THIS IS WHERE THE FIX IS APPLIED
                    header_str = self._format_headers_as_string(header_chunk.metadata)
                    
                    if len(chunk_text) > 600:
                        # Split large sections
                        sub_chunks = self.recursive_splitter.split_text(chunk_text)
                        
                        for j, sub_chunk in enumerate(sub_chunks):
                            if len(sub_chunk.strip()) > 30:  # Filter tiny chunks
                                chunk_doc = LangChainDocument(
                                    page_content=sub_chunk,
                                    metadata={
                                        **doc.metadata,
                                        "chunk_id": f"{doc.metadata['id']}_{i}_{j}",
                                        "chunk_type": "section_chunk",
                                        "header_hierarchy": header_str,  # STRING format instead of dict
                                        "chunk_size": len(sub_chunk),
                                        "section_index": i,
                                        "subsection_index": j
                                    }
                                )
                                all_chunks.append(chunk_doc)
                    else:
                        # Keep smaller sections intact
                        if len(chunk_text.strip()) > 30:
                            chunk_doc = LangChainDocument(
                                page_content=chunk_text,
                                metadata={
                                    **doc.metadata,
                                    "chunk_id": f"{doc.metadata['id']}_{i}",
                                    "chunk_type": "header_chunk",
                                    "header_hierarchy": header_str,  # STRING format instead of dict
                                    "chunk_size": len(chunk_text),
                                    "section_index": i
                                }
                            )
                            all_chunks.append(chunk_doc)
            
            logging.info(f"Created {len(all_chunks)} total chunks")
            
            # Log chunk distribution by department
            dept_counts = {}
            for chunk in all_chunks:
                dept = chunk.metadata.get('department', 'Unknown')
                dept_counts[dept] = dept_counts.get(dept, 0) + 1
            
            for dept, count in dept_counts.items():
                logging.info(f"Department {dept}: {count} chunks")
            
            return all_chunks
            
        except Exception as e:
            logging.error(f"Error chunking documents: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Chunking error: {str(e)}")
    
    def get_collection_name(self, department: str) -> str:
        """Generate collection name - single collection for admin aggregation"""
        if department == "ALL":
            return "admin_all_departments"
        return f"dept_{department.lower().replace(' ', '_').replace('-', '_')}"
    
    def get_or_create_vectorstore(self, documents: List[LangChainDocument], department: str) -> Chroma:
        """Create vectorstore with proper handling for admin multi-department access"""
        try:
            collection_name = self.get_collection_name(department)
            
            # For admin users, create a comprehensive collection
            if department == "ALL":
                logging.info("Creating/updating admin collection with all department data")
            
            # Try to load existing vectorstore
            try:
                vectorstore = Chroma(
                    collection_name=collection_name,
                    embedding_function=self.embeddings,
                    persist_directory=self.persist_directory
                )
                
                existing_count = len(vectorstore.get()['ids']) if vectorstore.get()['ids'] else 0
                
                # For admin collections, check if we need to update with new departments
                if existing_count > 0:
                    if department != "ALL" or existing_count > 50:  # Threshold for admin refresh
                        logging.info(f"Using existing vectorstore with {existing_count} chunks")
                        return vectorstore
                    
            except Exception as e:
                logging.info(f"Creating new vectorstore: {str(e)}")
            
            # Create enhanced chunks
            chunks = self.chunk_documents_enhanced(documents)
            
            if not chunks:
                raise HTTPException(status_code=400, detail="No chunks created from documents")
            
            # Clear existing data for refresh
            try:
                vectorstore = Chroma(
                    collection_name=collection_name,
                    embedding_function=self.embeddings,
                    persist_directory=self.persist_directory
                )
                # Clear existing documents
                existing_data = vectorstore.get()
                if existing_data['ids']:
                    vectorstore.delete(ids=existing_data['ids'])
                    logging.info(f"Cleared {len(existing_data['ids'])} existing chunks")
            except:
                pass  # Collection doesn't exist yet
            
            # Create new vectorstore
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                collection_name=collection_name,
                persist_directory=self.persist_directory
            )
            
            logging.info(f"Created vectorstore '{collection_name}' with {len(chunks)} chunks")
            return vectorstore
            
        except Exception as e:
            logging.error(f"Error creating vectorstore: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Vectorstore error: {str(e)}")
    
    def similarity_search(self, vectorstore: Chroma, query: str, max_chunks: int = 10) -> tuple:
        """Enhanced similarity search with department-aware ranking"""
        try:
            # Get more results for better filtering
            search_multiplier = 2
            results = vectorstore.similarity_search_with_score(
                query=query,
                k=max_chunks * search_multiplier
            )
            
            if not results:
                return [], [], []
            
            # Process and rank results
            processed_results = []
            for doc, distance in results:
                # Convert distance to similarity
                # similarity = max(0, 1 - distance)
                similarity_raw = 1 - distance          # now in [-1,1]
                similarity     = (similarity_raw + 1) / 2  # now in [0,1]
                # Boost scores for more relevant sections
                boost = 1.0
                content_lower = doc.page_content.lower()
                query_lower = query.lower()
                
                # Boost for exact matches in headers
                header_hierarchy = doc.metadata.get('header_hierarchy', '')
                if any(word in header_hierarchy.lower() for word in query_lower.split()):
                    boost += 0.1
                
                # Boost for technical content relevance
                tech_keywords = ['api', 'service', 'database', 'authentication', 'security', 'architecture']
                if any(keyword in content_lower for keyword in tech_keywords):
                    if any(keyword in query_lower for keyword in tech_keywords):
                        boost += 0.05
                
                adjusted_similarity = min(1.0, similarity * boost)
                processed_results.append((doc, adjusted_similarity))
            
            # Sort by adjusted similarity
            processed_results.sort(key=lambda x: x[1], reverse=True)
            
            # Take top results
            top_results = processed_results[:max_chunks]
            
            documents = [result[0] for result in top_results]
            scores = [result[1] for result in top_results]
            
            contents = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            
            return contents, scores, metadatas
            
        except Exception as e:
            logging.error(f"Error during similarity search: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")
    
    def get_vectorstore_stats(self, vectorstore: Chroma, department: str) -> Dict[str, Any]:
        """Get comprehensive vectorstore statistics"""
        try:
            all_docs = vectorstore.get()
            
            if not all_docs['ids']:
                return {"total_chunks": 0, "department": department}
            
            metadatas = all_docs['metadatas']
            documents = all_docs['documents']
            
            # Department distribution for admin view
            dept_distribution = {}
            chunk_types = {}
            
            for metadata in metadatas:
                dept = metadata.get('department', 'Unknown')
                chunk_type = metadata.get('chunk_type', 'unknown')
                
                dept_distribution[dept] = dept_distribution.get(dept, 0) + 1
                chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
            
            chunk_lengths = [len(doc) for doc in documents]
            
            stats = {
                "department": department,
                "total_chunks": len(all_docs['ids']),
                "department_distribution": dept_distribution,
                "chunk_types": chunk_types,
                "avg_chunk_length": sum(chunk_lengths) / len(chunk_lengths) if chunk_lengths else 0,
                "min_chunk_length": min(chunk_lengths) if chunk_lengths else 0,
                "max_chunk_length": max(chunk_lengths) if chunk_lengths else 0,
                "collection_name": vectorstore._collection.name
            }
            
            return stats
            
        except Exception as e:
            logging.error(f"Error getting vectorstore stats: {str(e)}")
            return {"error": str(e), "department": department}

def generate_enhanced_rag_response(query: str, relevant_chunks: List[str], metadatas: List[Dict]) -> str:
    """Generate contextual response with department attribution"""
    if not relevant_chunks:
        return "I couldn't find relevant information to answer your query in the available documents."
    
    # Group chunks by department for admin queries
    dept_chunks = {}
    for chunk, metadata in zip(relevant_chunks[:8], metadatas[:8]):
        dept = metadata.get('department', 'Unknown')
        if dept not in dept_chunks:
            dept_chunks[dept] = []
        
        # Get header context as string (now fixed)
        header_context = metadata.get('header_hierarchy', '')  # Now a string, not dict
        
        dept_chunks[dept].append({
            'content': chunk,
            'title': metadata.get('title', 'Document'),
            'headers': header_context,
            'source': metadata.get('source', 'Unknown')
        })
    
    # Build response with department sections
    response_sections = []
    all_sources = set()
    
    for dept, chunks in dept_chunks.items():
        if len(dept_chunks) > 1:  # Multi-department (admin) response
            response_sections.append(f"**From {dept} Department:**")
        
        for chunk_info in chunks[:3]:  # Top 3 per department
            if chunk_info['headers']:
                section_header = f"*{chunk_info['headers']}*"
                response_sections.append(f"{section_header}\n{chunk_info['content']}")
            else:
                response_sections.append(chunk_info['content'])
            
            all_sources.add(chunk_info['source'])
        
        if len(dept_chunks) > 1:
            response_sections.append("")  # Add spacing between departments
    
    context = "\n\n".join(response_sections)
    
    # Create final response
    if len(dept_chunks) > 1:
        dept_list = ", ".join(dept_chunks.keys())
        response = f"""Based on information from multiple departments ({dept_list}):

{context}

**Query:** "{query}"

**Sources:** {', '.join(sorted(all_sources))}

*This comprehensive response draws from your organization's knowledge base across departments.*"""
    else:
        single_dept = list(dept_chunks.keys())[0]
        response = f"""Based on information from the {single_dept} department:

{context}

**Query:** "{query}"

**Sources:** {', '.join(sorted(all_sources))}

*This response is generated from your department's technical documentation.*"""
    
    return response

# Create enhanced service instance
document_service = EnhancedLangChainDocumentService()

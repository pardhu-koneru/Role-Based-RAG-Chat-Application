from fastapi import APIRouter, Depends, status, HTTPException

from .. import schemas, database,oauth2

from ..load_transform_store import ChatRequest,ChatResponse,document_service,generate_enhanced_rag_response
# from app.load_transform_store import get_or_create_vectorstore,similarity_search,get_vectorstore_stats
from langchain_chroma import Chroma

import logging
import chromadb
from sqlalchemy.orm import Session


router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
def chat_with_rag(
    request: ChatRequest,
    current_user: schemas.TokenData = Depends(oauth2.get_current_user),
    db: Session = Depends(database.get_db)
):
    """
    RAG-powered chat endpoint using LangChain components
    """
    try:
        # Step 1: Load documents from database as LangChain Documents
        documents = document_service.load_documents_from_db(current_user, db)
        
        if not documents:
            raise HTTPException(status_code=404, detail="No documents found for your department")
        
        department = current_user.department if current_user.role != "admin" else "ALL"
        
        # Step 2: Get or create vectorstore
        vectorstore = document_service.get_or_create_vectorstore(documents, department)
        
        # Step 3: Perform similarity search
        relevant_chunks, similarities, metadatas = document_service.similarity_search(
            vectorstore, 
            request.query,
            max_chunks=request.max_chunks
        )
        
        # Step 4: Filter by similarity threshold
        filtered_chunks = []
        filtered_similarities = []
        filtered_metadatas = []
        
        for chunk, score, metadata in zip(relevant_chunks, similarities, metadatas):
            if score >= request.similarity_threshold:
                filtered_chunks.append(chunk)
                filtered_similarities.append(score)
                filtered_metadatas.append(metadata)
        
        # Step 5: Generate RAG response
        rag_response = generate_enhanced_rag_response(request.query, filtered_chunks, filtered_metadatas)
        
        # Step 6: Get vectorstore statistics
        collection_stats = document_service.get_vectorstore_stats(vectorstore, department)
        
        return ChatResponse(
            query=request.query,
            department=department,
            relevant_chunks=filtered_chunks,
            similarity_scores=filtered_similarities,
            response=rag_response,
            total_chunks_found=len(filtered_chunks),
            collection_stats=collection_stats
        )
        
    except Exception as e:
        logging.error(f"Chat endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/chat/collections")
def list_collections(current_user: schemas.TokenData = Depends(oauth2.get_current_user)):
    """
    List all collections in the vector store
    """
    try:

        client = chromadb.PersistentClient(path=document_service.persist_directory)
        collections = client.list_collections()
        
        collection_info = []
        for collection in collections:
            vectorstore = Chroma(
                client=client,
                collection_name=collection.name,
                embedding_function=document_service.embeddings
            )
            stats = document_service.get_vectorstore_stats(vectorstore, collection.name)
            collection_info.append({
                "name": collection.name,
                "stats": stats
            })
        
        return {
            "total_collections": len(collections),
            "collections": collection_info
        }
        
    except Exception as e:
        logging.error(f"Error listing collections: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/chat/reset/{department}")
def reset_department_vectorstore(
    department: str,
    current_user: schemas.TokenData = Depends(oauth2.get_current_user)
):
    """
    Reset vectorstore for a department (Admin only)
    """
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Only admins can reset vectorstores")
    
    try:
        
        
        collection_name = document_service.get_collection_name(department)
        client = chromadb.PersistentClient(path=document_service.persist_directory)
        
        try:
            client.delete_collection(name=collection_name)
            return {"message": f"Vectorstore for department {department} reset successfully"}
        except Exception as e:
            return {"message": f"No vectorstore found for department {department}"}
            
    except Exception as e:
        logging.error(f"Error resetting vectorstore: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

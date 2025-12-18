from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, Form
from sqlalchemy.orm import Session
from pathlib import Path
from datetime import datetime
import shutil
import os

from database import get_db
from auth.auth_jwt import get_current_user
from schemas import TokenData, DocumentUploadResponse, DocumentOut, DocumentList
from models import Document, User
from services.document_service import DocumentService
from services.description_vectorizer import DescriptionVectorizer
# Create router
router = APIRouter(prefix="/api/documents", tags=["Documents"])

# Initialize document service
doc_service = DocumentService()
 

@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    department: str = Form(...),
    description: str = Form(None),
    current_user: TokenData = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Upload a document to a specific department
    
    Steps:
    1. Validate file type
    2. Check user permissions
    3. Save file to disk
    4. Save info to database
    5. Vectorize the document
    """
    
    print(f"\n📤 Uploading file: {file.filename}")
    print(f"👤 User: {current_user.email}")
    print(f"🏢 Department: {department}\n")
    
    # Step 1: Check if file type is allowed
    allowed_types = ['.md', '.xlsx', '.xls', '.csv']
    file_extension = Path(file.filename).suffix.lower()
    
    if file_extension not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"File type {file_extension} not allowed. Use: {allowed_types}"
        )
    
    # Step 2: Check if department is valid
    valid_departments = ['finance', 'marketing', 'hr', 'engineering', 'general']
    department = department.lower()
    
    if department not in valid_departments:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid department. Choose from: {valid_departments}"
        )
    
    # Step 3: Check user permissions (only upload to own department, unless admin)
    if current_user.role not in ['admin', 'clevel']:
        if current_user.department != department:
            raise HTTPException(
                status_code=403,
                detail=f"You can only upload to {current_user.department} department"
            )
    
    # Step 4: Create folder to save file
    save_folder = Path(f"uploads/{department}")
    save_folder.mkdir(parents=True, exist_ok=True)
    
    # Step 5: Create unique filename (with timestamp)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_filename = file.filename.replace(" ", "_")
    new_filename = f"{timestamp}_{safe_filename}"
    file_path = save_folder / new_filename
    
    # Step 6: Save file to disk
    try:
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        print(f"✅ Saved to: {file_path}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    
    # Step 7: Get file size
    file_size = os.path.getsize(file_path)
    
    # Step 8: Get user ID from database
    user = db.query(User).filter(User.email == current_user.email).first()
    
    # Step 9: Save document info to database

    try:
        new_document = Document(
            filename=new_filename,
            original_filename=file.filename,
            file_type=file_extension[1:],  # Remove the dot
            file_path=str(file_path),
            department=department,
            uploaded_by=user.id,
            file_size=file_size,
            chunk_count=0,
            vectorized="no",
            description=description
        )
        
        db.add(new_document)
        db.commit()
        db.refresh(new_document)
        print(f"✅ Saved to database with ID: {new_document.id}")
        
    except Exception as e:
        # If database fails, delete the uploaded file
        os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    
    # Step 10: Process and vectorize document
    if(new_document.vectorized == "no" and new_document.file_type not in ['xlsx', 'xls', 'csv']):
        try:
            print("🔄 Starting vectorization...")
            chunk_count = doc_service.process_and_vectorize(new_document)
            
            # Update database with chunk count
            new_document.chunk_count = chunk_count
            new_document.vectorized = "yes"
            db.commit()
            
            print(f"✅ Vectorized into {chunk_count} chunks\n")
            
        except Exception as e:
            print(f"❌ Vectorization failed: {str(e)}")
    elif(new_document.file_type in ['xlsx', 'xls', 'csv']):
        try:
            print("🔄 Starting description vectorization for CSV/XLSX...")
            desc_vectorizer = DescriptionVectorizer()
            desc_vectorizer.vectorize_description(new_document)
            
            # Update database to indicate description vectorized
            new_document.vectorized = "yes"
            db.commit()
            
            print(f"✅ Description vectorization complete\n")
            
        except Exception as e:
            print(f"❌ Description vectorization failed: {str(e)}")

    # Return response
    return DocumentUploadResponse(
        id=new_document.id,
        filename=new_document.filename,
        original_filename=new_document.original_filename,
        file_type=new_document.file_type,
        department=new_document.department,
        uploaded_by=new_document.uploaded_by,
        file_size=new_document.file_size,
        chunk_count=new_document.chunk_count,
        vectorized=new_document.vectorized,
        uploaded_at=new_document.uploaded_at,
        description=new_document.description,
        message=f"✅ Document uploaded and split into {new_document.chunk_count} chunks"
    )


# ============================================================
# 2. LIST ALL DOCUMENTS (based on user role)
# ============================================================
@router.get("/", response_model=DocumentList)
async def list_documents(
    current_user: TokenData = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    List all documents the user can access
    
    Access Rules:
    - employee: only 'general' documents
    - finance/hr/marketing/engineering: their department + 'general'
    - admin/clevel: ALL documents
    """
    
    print(f"\n📋 Listing documents for: {current_user.email}")
    print(f"🎭 Role: {current_user.role}\n")
    
    # Decide which departments user can access
    if current_user.role in ['admin', 'clevel']:
        # Admins and C-level see everything
        accessible = ['finance', 'marketing', 'hr', 'engineering', 'general']
    elif current_user.role == 'employee' or current_user =='user':
        # Employees only see general documents
        accessible = [str(current_user.department)]
    
    print(f"🔓 Accessible departments: {accessible}")
    
    # Get documents from database
    documents = db.query(Document).filter(
        Document.department.in_(accessible)
    ).order_by(Document.uploaded_at.desc()).all()
    
    print(f"📄 Found {len(documents)} documents\n")
    
    return DocumentList(
        documents=documents,
        total=len(documents),
        accessible_departments=accessible
    )


# ============================================================
# 3. GET SINGLE DOCUMENT DETAILS
# ============================================================
@router.get("/{document_id}", response_model=DocumentOut)
async def get_document(
    document_id: int,
    current_user: TokenData = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get details of a specific document
    """
    
    print(f"\n🔍 Getting document ID: {document_id}")
    print(f"👤 User: {current_user.email}\n")
    
    # Find document in database
    document = db.query(Document).filter(Document.id == document_id).first()
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Check if user can access this document
    if current_user.role in ['admin', 'clevel']:
        # Admins can access everything
        accessible = ['finance', 'marketing', 'hr', 'engineering', 'general']
    elif current_user.role == 'employee' or current_user.role == 'user':
        accessible = [current_user.department]
    
    if document.department not in accessible:
        raise HTTPException(
            status_code=403,
            detail="You don't have permission to access this document"
        )
    
    print(f"✅ Access granted\n")
    return document


# ============================================================
# 4. DELETE DOCUMENT
# ============================================================
@router.delete("/{document_id}")
async def delete_document(
    document_id: int,
    current_user: TokenData = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Delete a document
    - Users can delete their own documents
    - Admins can delete any document
    """
    
    print(f"\n🗑️  Deleting document ID: {document_id}")
    print(f"👤 User: {current_user.email}\n")
    
    # Find document
    document = db.query(Document).filter(Document.id == document_id).first()
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Get current user from database
    user = db.query(User).filter(User.email == current_user.email).first()
    
    # Check permissions
    if current_user.role != 'admin' and document.uploaded_by != user.id:
        raise HTTPException(
            status_code=403,
            detail="You can only delete your own documents"
        )
    
    try:
        # Step 1: Delete from vector database
        if document.vectorized == "yes":
            print("Deleting from vector store...")
            doc_service.delete_document_from_vectorstore(document)
        
        # Step 2: Delete physical file
        file_path = Path(document.file_path)
        if file_path.exists():
            os.remove(file_path)
            print(f"Deleted file: {file_path}")
        
        # Step 3: Delete from database
        db.delete(document)
        db.commit()
        
        print(f"✅ Document deleted successfully\n")
        
        return {
            "success": True,
            "message": "Document deleted successfully",
            "document_id": document_id,
            "filename": document.original_filename
        }
        
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete document: {str(e)}"
        )
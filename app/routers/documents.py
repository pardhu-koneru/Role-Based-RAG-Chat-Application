from fastapi import APIRouter, HTTPException,Depends
from .. import auth_jwt, docs, schemas,models,database
from sqlalchemy.orm import Session
from ..database import get_db

router = APIRouter()
@router.post("/load_documents")
async def load_documents(db: Session = Depends(get_db), current_user: schemas.UserCreate = Depends(auth_jwt.get_admin_access)):
    try:
        # Check if documents already exist in the database
        existing_documents_count = db.query(models.Document).count()
        if existing_documents_count > 0:
            return {"message": "Documents already loaded. Skipping to prevent redundancy."}

        docs.read_and_store_documents(db)
        return {"message": "Documents loaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error occurred: {str(e)}")

@router.post("/load_documents")
async def load_documents(db: Session = Depends(get_db), current_user: schemas.UserCreate= Depends(auth_jwt.get_admin_access)):
    try:
        docs.read_and_store_documents(db)
        return {"message": "Documents loaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error occurred: {str(e)}")
    

@router.get("/get_content")
def get_content(
    current_user: schemas.TokenData = Depends(auth_jwt.get_current_user),
    db: Session = Depends(database.get_db),
):
    """
    Load documents based on user role and department.
    - If user is admin -> load all documents.
    - If user is normal -> load only their department docs.
    - Multiple rows from the same department are concatenated.
    """

    if current_user.role == "admin":
        docs = db.query(models.Document).all()
    else:
        docs = db.query(models.Document).filter(
            models.Document.department == current_user.department
        ).all()

    if not docs:
        return {"message": "No documents found."}

    # Concatenate all content
    combined_content = "\n\n".join([doc.content for doc in docs])

    return {
        "department": current_user.department if current_user.role != "admin" else "ALL",
        "total_docs": len(docs),
        "combined_content": combined_content
    }

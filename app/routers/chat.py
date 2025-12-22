"""
Chat Router with LangGraph
Save as: app/routers/chat.py
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from database import get_db
from auth.auth_jwt import get_current_user
from schemas import TokenData, ChatRequest, ChatResponse
from services.chat_workflow import ChatWorkflow
from models import ChatHistory,User

router = APIRouter(prefix="/api/chat", tags=["Chat"])

# Initialize workflow
workflow = ChatWorkflow()


@router.post("/query", response_model=ChatResponse)
async def chat_query(
    request: ChatRequest,
    current_user: TokenData = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Main chat endpoint - routes through LangGraph workflow
    
    Flow:
    1. Classify query (SQL vs RAG)
    2. If SQL:
       - Retrieve relevant files via description embeddings
       - Generate pandas code
       - Execute on CSV
       - Format response
    3. If RAG:
       - Regular RAG pipeline
    """
    
    print("\n" + "="*70)
    print(f"💬 NEW CHAT REQUEST")
    print(f"👤 User: {current_user.email} ({current_user.role})")
    print(f"🏢 Department: {current_user.department}")
    print(f"❓ Query: {request.query}")
    print("="*70)
    
    try:
        # Determine accessible department(s)
        if current_user.role in ['admin', 'clevel']:
            # Admins get access to all departments for comprehensive search
            department = "all"  
        else:
            department = current_user.department
        
        # Run through LangGraph workflow
        result = workflow.run(
            query=request.query,
            department=department,
            user_id=current_user.email
        )
        
        # Save to chat history
        try:
            history = ChatHistory(
                user_id=db.query(User).filter(User.email == current_user.email).first().id,
                query=request.query,
                response=result["final_response"],
                query_type=result["query_type"],
                sources=",".join(result["sources"]),
            )
            db.add(history)
            db.commit()
        except Exception as e:
            print(f"⚠️  Failed to save history: {e}")
        
        # Prepare response
        response = ChatResponse(
            query=request.query,
            response=result["final_response"],
            query_type=result["query_type"],
            sources=result["sources"],
            department=department,
            sql_query=result.get("generated_sql"),
            data_preview=result.get("sql_results", [])[:5] if result["query_type"] == "sql" else None
        )
        
        print("\n✅ CHAT COMPLETED")
        print(f"📊 Type: {result['query_type']}")
        print(f"📁 Sources: {len(result['sources'])}")
        print("="*70 + "\n")
        
        return response
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}\n")
        raise HTTPException(
            status_code=500,
            detail=f"Chat processing failed: {str(e)}"
        )

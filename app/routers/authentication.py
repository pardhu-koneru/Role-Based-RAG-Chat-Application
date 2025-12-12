from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from schemas import LoginRequest,TokenResponse,UserCreate,UserOut
from database import get_db,Base,engine
from models import User
from auth.token import create_access_token
from auth.hashing import Hash
from repository import user

router = APIRouter(prefix="/api/authentication", tags=["Authentication"])
Base.metadata.create_all(bind=engine)

@router.post("/register", response_model=UserOut)
def register(request: UserCreate, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == request.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    request.password = Hash.bcrypt(request.password)
    
    return user.create(request,db)

@router.post("/login",response_model = TokenResponse)
def login(request: LoginRequest, db: Session = Depends(get_db)):

    user = db.query(User).filter(User.email == request.email).first()
    if not user:
        raise HTTPException(401, "Invalid credentials")

    print("User typed password:", request.password, len(request.password))
    
    if not Hash.verify(user.password, request.password):
        raise HTTPException(401, "Invalid credentials")

    token = create_access_token({
        "sub": user.email,
        "role": user.role,
        "department": user.department
    })

    return {"access_token": token, "token_type": "bearer"}

from fastapi import APIRouter,HTTPException
from .. import database, oauth2, schemas, models,oauth2
from sqlalchemy.orm import Session
from fastapi import APIRouter,Depends,status
from ..repository import user

router = APIRouter(
    prefix="/user",
    tags=['Users']
)

get_db = database.get_db


@router.post('/', response_model=schemas.ShowUser)
def create_user(request: schemas.User, db: Session = Depends(get_db), current_user: schemas.User = Depends(oauth2.get_admin_access)):
    existing_user = db.query(models.User).filter(models.User.email == request.email).first()
    if existing_user:
        raise HTTPException( 
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )   
    return user.create(request, db)

@router.get('/{id}',response_model=schemas.ShowUser)
def get_user(id:int,db: Session = Depends(get_db),current_user: schemas.User = Depends(oauth2.get_admin_access)):
    return user.show(id,db)

@router.delete('/{email}', status_code=status.HTTP_204_NO_CONTENT)
def delete_user(email: str, db: Session = Depends(get_db), current_user: schemas.User = Depends(oauth2.get_admin_access)):
    user_to_delete = db.query(models.User).filter(models.User.email == email).first()
    if not user_to_delete:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with email {email} not found"
        )
    db.delete(user_to_delete)
    db.commit()
    return None
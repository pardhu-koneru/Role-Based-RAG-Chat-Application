from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from . import token,schemas

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

def get_current_user(data: str = Depends(oauth2_scheme)):
    """
    Get current user from token and return user info with role.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    # Verify token and get payload (should include email and role)  
    token_data = token.verify_token(data, credentials_exception)
    return token_data

def get_admin_access(current_user: schemas.TokenData = Depends(get_current_user)):
    """
    Dependency to ensure only admin users can access protected routes.
    """
    if not current_user or current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required. Insufficient permissions."
        )
    return current_user

def get_user_access(current_user: schemas.TokenData = Depends(get_current_user)):
    """
    Dependency to ensure authenticated users (both admin and regular) can access routes.
    """
    # This function allows both admin and regular users
    # You can add additional checks here if needed
    return current_user


from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from .token import verify_token
from schemas import TokenData

# Create a security scheme
security = HTTPBearer()

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> TokenData:
    
    token = credentials.credentials
    
    payload = verify_token(token)
    if not payload:
        print("Token verification failed")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    
    return payload


async def get_admin_access(
    current_user: TokenData = Depends(get_current_user)
) -> TokenData:
    
    
    if current_user.role != "admin":
        print(f"Access denied - user is not admin")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    return current_user

import bcrypt

class Hash:
    @staticmethod
    def bcrypt(password: str) -> str:
        """Hash a password using bcrypt."""
        try:
            password_bytes = password.encode('utf-8')
            
            # Generate salt and hash
            salt = bcrypt.gensalt()
            hashed = bcrypt.hashpw(password_bytes, salt)
            
            # Convert bytes back to string for storage
            hashed_str = hashed.decode('utf-8')
           
            return hashed_str
            
        except Exception as e:
             
            import traceback
            traceback.print_exc()
  

    @staticmethod
    def verify(hashed_password: str, plain_password: str) -> bool:
        """Verify a plain password against a hashed password."""
        try: 
            # Convert strings to bytes
            plain_bytes = plain_password.encode('utf-8')
            hashed_bytes = hashed_password.encode('utf-8')
            
            # Verify password
            result = bcrypt.checkpw(plain_bytes, hashed_bytes)
            
            return result
            
        except Exception as e:
            print(f"ERROR in Hash.verify: {e}")
            import traceback
            traceback.print_exc()
            return False
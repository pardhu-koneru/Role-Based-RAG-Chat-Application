import os
from . import models, database
from sqlalchemy.orm import Session

def read_and_store_documents(db: Session):
    # Create database session
    
    # Base directory where department folders are located
    base_dir = "./app/data"
    
    try:
        # Walk through all directories
        for department in os.listdir(base_dir):
            dept_path = os.path.join(base_dir, department)
            
            # Check if it's a directory
            if os.path.isdir(dept_path):
                # Process all .md files in the department folder
                for file in os.listdir(dept_path):
                    if file.endswith('.md'):
                        file_path = os.path.join(dept_path, file)
                        
                        # Read the content of the markdown file
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            
                        # Create new document record
                        doc = models.Document(
                            title=file.replace('.md', ''),
                            content=content,
                            department=department
                        )
                        
                        # Add to database
                        db.add(doc)
                
                # Commit after processing each department
                db.commit()
                
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        db.rollback()
    
    finally:
        db.close()

if __name__ == "__main__":
    read_and_store_documents()


   
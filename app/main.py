from fastapi import FastAPI
import uvicorn
import models
from routers import admin, authentication,documents,chat
from database import engine
from fastapi.middleware.cors import CORSMiddleware



app = FastAPI(
    title="FinSolve RAG Chatbot API",
    description="Role-Based Access Control Chatbot with RAG",
    version="1.0.0"
)

# CORS middleware (for frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

models.Base.metadata.create_all(engine)

app.include_router(authentication.router)
app.include_router(admin.router)
app.include_router(documents.router)
app.include_router(chat.router)





@app.get("/")
def root():
    return {"message": "API is running"}

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
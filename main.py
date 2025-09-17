from fastapi import FastAPI
import uvicorn
from app import models
from app.routers import authentication,user,chat,documents
from app.database import engine



app = FastAPI()

models.Base.metadata.create_all(engine)

app.include_router(authentication.router)
app.include_router(user.router)
app.include_router(documents.router)
app.include_router(chat.router)




@app.get("/")
def root():
    return {"message": "API is running"}

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
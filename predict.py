from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn 
import joblib


loaded_model = joblib.load('model.pkl')
app = FastAPI() 

@app.get("/")
def read_root():
    return {"mensaje": "Hola Mundo"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
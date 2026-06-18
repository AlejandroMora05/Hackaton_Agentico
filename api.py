# api.py
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from typing import List
import uvicorn
from agent import ask
from scrapper.grades import get_grades_report, LoginTimeoutError

app = FastAPI(title="Copiloto UdeA API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    question: str
    history: List[dict] = []

@app.post("/api/chat")
async def chat(req: ChatRequest):
    try:
        result = await run_in_threadpool(ask, req.question, req.history)
        return result
    except Exception as e:
        msg = str(e)
        if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
            answer = "El servicio de IA alcanzó su límite de solicitudes. Espera unos minutos e intenta de nuevo."
        else:
            answer = "Ocurrió un error inesperado. Por favor intenta de nuevo."
        return {"answer": answer, "sources": []}

@app.post("/api/grades")
async def grades():
    try:
        return await run_in_threadpool(get_grades_report)
    except LoginTimeoutError:
        return {"error": "No se detectó el inicio de sesión a tiempo. Intenta de nuevo."}
    except Exception:
        return {"error": "Ocurrió un error al consultar tus notas. Intenta de nuevo."}

app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

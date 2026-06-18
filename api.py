# api.py
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from typing import List
import uvicorn
from agent import ask
from scrapper.login_session import (
    NotReadyError,
    SessionNotFoundError,
    analyze_map,
    analyze_notas,
    cancel_session,
    goto_target,
    start_session,
)

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

class SessionRequest(BaseModel):
    session_id: str

# Las dos rutas ("Ver mis notas" y "Ver mi mapa") comparten la misma sesión
# de login institucional (scrapper/login_session.py): iniciar sesión una vez
# desde cualquiera de los dos botones sirve para el otro también.

@app.post("/api/session/start")
async def session_start(target: str):
    return await start_session(target)

@app.post("/api/session/goto")
async def session_goto(req: SessionRequest, target: str):
    try:
        await goto_target(req.session_id, target)
        return {"ok": True}
    except SessionNotFoundError as e:
        return {"error": str(e), "session_expired": True}
    except Exception:
        return {"error": "No se pudo navegar a esa sección. Intenta de nuevo."}

@app.post("/api/session/cancel")
async def session_cancel(req: SessionRequest):
    await cancel_session(req.session_id)
    return {"ok": True}

@app.post("/api/grades/analyze")
async def grades_analyze(req: SessionRequest):
    try:
        return await analyze_notas(req.session_id)
    except SessionNotFoundError as e:
        return {"error": str(e), "session_expired": True}
    except NotReadyError as e:
        return {"error": str(e)}
    except Exception:
        return {"error": "Ocurrió un error al consultar tus notas. Intenta de nuevo."}

@app.post("/api/map/analyze")
async def map_analyze(req: SessionRequest):
    try:
        return await analyze_map(req.session_id)
    except SessionNotFoundError as e:
        return {"error": str(e), "session_expired": True}
    except NotReadyError as e:
        return {"error": str(e)}
    except Exception:
        return {"error": "Ocurrió un error al analizar tu información. Intenta de nuevo."}

app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

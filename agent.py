# agent.py
import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List, Optional
from llm_factory import get_llm

load_dotenv()

# ── 1. Cargar vectorstore y retriever híbrido (BM25 + vectores) ───────────────
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# Reconstruye los documentos desde Chroma para alimentar BM25
_data = vectorstore.get(include=["documents", "metadatas"])
_all_docs = [
    Document(page_content=text, metadata=meta)
    for text, meta in zip(_data["documents"], _data["metadatas"])
]

bm25_retriever = BM25Retriever.from_documents(_all_docs, k=6)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

# 50% peso a cada uno; EnsembleRetriever deduplica resultados automáticamente
retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5, 0.5]
)

# ── 2. LLM ────────────────────────────────────────────────────────────────────
llm = get_llm()  # usa LLM_PROVIDER del .env

# ── 3. Prompt del sistema ─────────────────────────────────────────────────────
SYSTEM_PROMPT = """Eres un copiloto administrativo de la Universidad de Antioquia (UdeA).
Tu función es responder preguntas sobre el reglamento estudiantil y procesos de matrícula
ÚNICAMENTE con base en los fragmentos del reglamento que te son proporcionados.

Reglas:
- Responde siempre en español, de forma clara y amable.
- Cita el artículo o sección cuando lo encuentres en el contexto.
- Si la información no está en los fragmentos, di exactamente:
  "No encontré información sobre eso en el reglamento de la UdeA."
- Nunca inventes información.
- Usa el historial de conversación para dar respuestas coherentes y no repetir saludos.

Contexto del reglamento:
{context}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])

# ── 4. Estado del grafo ───────────────────────────────────────────────────────
class AgentState(TypedDict):
    question: str
    history: List[dict]
    context: str
    answer: str
    sources: List[str]
    grades_mode: bool

# ── 5. Detección de intención de notas ───────────────────────────────────────
_GRADES_KEYWORDS = [
    "notas", "nota", "calificacion", "calificaciones",
    "como voy", "cómo voy", "mis materias", "rendimiento",
    "definitiva", "acumulada", "porcentaje evaluado",
    "perder materia", "ganar materia", "perdiendo", "ganando",
    "cuanto saque", "cuánto saqué", "cuanto llevo", "cuánto llevo",
    "ver notas", "consultar notas", "mis notas",
]

def _is_grades_question(question: str) -> bool:
    q = question.lower()
    return any(kw in q for kw in _GRADES_KEYWORDS)

def _route_intent(state: AgentState) -> str:
    return "grades" if _is_grades_question(state["question"]) else "rag"

# ── 6. Nodo de notas (Playwright) ─────────────────────────────────────────────
def _format_grades(report: dict) -> str:
    lines = [
        f"Aquí están tus notas, **{report.get('estudiante', '')}**:",
        f"*{report.get('programa', '')} — Semestre {report.get('semestre', '')}*",
        "",
    ]
    for m in report.get("materias", []):
        icon = {"green": "✅", "orange": "⚠️", "red": "❌"}.get(m.get("color"), "•")
        lines.append(f"{icon} **{m['nombre']}**")
        lines.append(f"   {m['mensaje']}")
        acum = m.get("nota_acumulada")
        pct = m.get("porcentaje_evaluado")
        if acum is not None:
            lines.append(f"   Nota acumulada: **{acum}** | Evaluado: **{pct}%**")
        if m.get("nota_definitiva") is not None:
            lines.append(f"   Nota definitiva: **{m['nota_definitiva']}**")
        lines.append("")
    return "\n".join(lines)

def get_grades(state: AgentState) -> AgentState:
    """Abre el navegador, espera login y devuelve el reporte de notas."""
    from scrapper.grades import get_grades_report, LoginTimeoutError
    print(" Abriendo navegador para consultar notas...")
    try:
        report = get_grades_report()
        answer = _format_grades(report)
    except LoginTimeoutError:
        answer = (
            "⏱️ No completaste el inicio de sesión a tiempo. "
            "Por favor intenta de nuevo y accede al portal antes de que pase el tiempo."
        )
    except Exception as e:
        answer = f"Ocurrió un error al consultar tus notas: {e}"
    return {**state, "answer": answer, "sources": [], "grades_mode": True}

# ── 7. Nodos RAG ──────────────────────────────────────────────────────────────
def retrieve(state: AgentState) -> AgentState:
    """Busca fragmentos relevantes en el vectorstore."""
    print(" Buscando en documentos...")
    docs = retriever.invoke(state["question"])

    context = "\n\n".join([doc.page_content for doc in docs])
    sources = list(set([
        f"{doc.metadata.get('source', '').split('/')[-1]} (pág. {doc.metadata.get('page', '?')})"
        for doc in docs
    ]))

    return {**state, "context": context, "sources": sources}

def generate(state: AgentState) -> AgentState:
    """Genera la respuesta con el LLM usando el contexto recuperado."""
    print(" Generando respuesta...")
    history_messages = []
    for msg in state["history"]:
        if msg["role"] == "user":
            history_messages.append(HumanMessage(content=msg["content"]))
        else:
            history_messages.append(AIMessage(content=msg["content"]))

    chain = prompt | llm
    response = chain.invoke({
        "context": state["context"],
        "question": state["question"],
        "history": history_messages
    })
    return {**state, "answer": response.content}

# ── 8. Construir el grafo ─────────────────────────────────────────────────────
def build_agent():
    graph = StateGraph(AgentState)

    graph.add_node("get_grades", get_grades)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)

    graph.add_conditional_edges(START, _route_intent, {"grades": "get_grades", "rag": "retrieve"})
    graph.add_edge("get_grades", END)
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()

agent = build_agent()

# ── 9. Función pública para usar desde la interfaz ───────────────────────────
def ask(question: str, history: list = None) -> dict:
    result = agent.invoke({
        "question": question,
        "history": history or [],
        "context": "",
        "answer": "",
        "sources": [],
        "grades_mode": False,
    })
    return {
        "answer": result["answer"],
        "sources": result["sources"]
    }

# ── Test rápido ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_q = "¿Cuáles son las causales de pérdida de la calidad de estudiante?"
    print(f"\n Pregunta: {test_q}\n")
    result = ask(test_q)
    print(f" Respuesta:\n{result['answer']}")
    print(f"\n Fuentes: {', '.join(result['sources'])}")

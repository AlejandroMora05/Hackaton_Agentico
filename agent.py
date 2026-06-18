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
    search_query: str
    history: List[dict]
    context: str
    answer: str
    sources: List[str]
    grades_mode: bool
    academic_context: Optional[dict]

# ── 5. Detección de intención de notas ───────────────────────────────────────
_GRADES_KEYWORDS = [
    "notas", "nota", "calificacion", "calificaciones",
    "como voy", "cómo voy", "mis materias", "rendimiento",
    "definitiva", "acumulada", "porcentaje evaluado",
    "perder materia", "ganar materia", "perdiendo", "ganando",
    "cuanto saque", "cuánto saqué", "cuanto llevo", "cuánto llevo",
    "ver notas", "consultar notas", "mis notas",
]

_NEXT_SEMESTER_KEYWORDS = [
    "proximo semestre", "próximo semestre", "siguiente semestre",
    "que puedo ver", "qué puedo ver", "que puedo matricular", "qué puedo matricular",
    "que puedo cursar", "qué puedo cursar", "que puedo tomar", "qué puedo tomar",
    "que puedo inscribir", "qué puedo inscribir", "materias disponibles",
    "que materias quedan", "qué materias quedan", "que me falta", "qué me falta",
    "que desbloqueo", "qué desbloqueo", "que se habilita", "qué se habilita",
]

def _is_grades_question(question: str) -> bool:
    q = question.lower()
    return any(kw in q for kw in _GRADES_KEYWORDS)

def _is_next_semester_question(question: str) -> bool:
    q = question.lower()
    return any(kw in q for kw in _NEXT_SEMESTER_KEYWORDS)

def _route_intent(state: AgentState) -> str:
    question = state["question"]
    if _is_next_semester_question(question):
        return "next_semester"
    if _is_grades_question(question):
        return "grades"
    return "rag"

# ── 6. Nodos de notas y pénsum (usan lo ya analizado desde la interfaz) ──────
def _format_grades(report: dict) -> str:
    materias = report.get("materias", [])
    if not materias:
        return (
            "No encontré materias en curso este semestre en lo que ya analizaste. "
            "Revisa que hayas hecho click en \"Ver mis notas\" y que se haya cargado correctamente."
        )
    lines = [
        f"Aquí están tus notas, **{report.get('estudiante', '')}**:",
        f"*{report.get('programa', '')} — Semestre {report.get('semestre', '')}*",
        "",
    ]
    for m in materias:
        icon = {"green": "✅", "orange": "⚠️", "red": "❌"}.get(m.get("color"), "•")
        lines.append(f"{icon} **{m['nombre']}**")
        lines.append(f"   {m.get('mensaje', '')}")
        acum = m.get("nota_acumulada")
        pct = m.get("porcentaje_evaluado")
        if acum is not None:
            lines.append(f"   Nota acumulada: **{acum}** | Evaluado: **{pct}%**")
        if m.get("nota_definitiva") is not None:
            lines.append(f"   Nota definitiva: **{m['nota_definitiva']}**")
        lines.append("")
    return "\n".join(lines)

def _materias_cursando_from_map(map_report: dict) -> list:
    materias = []
    for nivel in map_report.get("niveles", []):
        materias.extend(m for m in nivel["materias"] if m.get("cursando"))
    for grupo in map_report.get("electivas", []):
        materias.extend(m for m in grupo["materias"] if m.get("cursando"))
    return materias

_NO_DATA_GRADES_MSG = (
    "Todavía no he visto tus notas. Haz click en \"Ver mis notas\" o \"Ver mi mapa\" en el menú, "
    "espera a que termine de analizar, y vuelve a preguntarme."
)

def get_grades(state: AgentState) -> AgentState:
    """Responde con las notas ya analizadas desde la interfaz (no abre un
    navegador nuevo: reutiliza lo que el estudiante ya cargó con los botones
    "Ver mis notas" / "Ver mi mapa")."""
    ctx = state.get("academic_context") or {}
    grades_report = ctx.get("grades")
    map_report = ctx.get("map")

    if grades_report and grades_report.get("materias"):
        answer = _format_grades(grades_report)
    elif map_report and map_report.get("niveles"):
        pseudo_report = {
            "estudiante": map_report.get("estudiante", ""),
            "programa": map_report.get("programa", ""),
            "semestre": map_report.get("semestre", ""),
            "materias": _materias_cursando_from_map(map_report),
        }
        answer = _format_grades(pseudo_report)
    else:
        answer = _NO_DATA_GRADES_MSG

    return {**state, "answer": answer, "sources": [], "grades_mode": True}

def next_semester(state: AgentState) -> AgentState:
    """Estima qué materias quedarían habilitadas el próximo semestre, usando
    el pénsum ya analizado desde "Ver mi mapa" (necesita el grafo de
    prerrequisitos, que la vista de solo-notas no tiene)."""
    ctx = state.get("academic_context") or {}
    map_report = ctx.get("map")

    if not map_report or not map_report.get("niveles"):
        answer = (
            "Para decirte qué materias podrías ver el próximo semestre necesito tu pénsum. "
            "Haz click en \"Ver mi mapa\", espera a que cargue, y vuelve a preguntarme."
        )
        return {**state, "answer": answer, "sources": [], "grades_mode": True}

    from scrapper.map_data import materias_disponibles_proximo_semestre
    resultado = materias_disponibles_proximo_semestre(map_report)

    if resultado["advertencia"]:
        answer = resultado["advertencia"]
    elif not resultado["candidatas"]:
        answer = (
            "Con lo que ya tienes en curso, no encontré materias nuevas cuyos prerrequisitos "
            "queden completamente cubiertos para el próximo semestre."
        )
    else:
        candidatas = sorted(resultado["candidatas"], key=lambda c: (c["nivel"] or 999, c["nombre"]))
        lines = [resultado["supuesto"], "", "Estas son las materias que ya tendrías habilitadas:", ""]
        for c in candidatas:
            lines.append(f"- **{c['nombre']}** ({c['creditos'] if c['creditos'] is not None else '—'} cr · nivel {c['nivel']})")
        lines.append("")
        lines.append(
            "Esto no incluye electivas (no tienen un nivel fijo en el pénsum). Si ya aprobaste "
            "alguna de estas o tienes materias pendientes de semestres anteriores que no se "
            "reflejan aquí, dímelo y ajusto la lista."
        )
        answer = "\n".join(lines)

    return {**state, "answer": answer, "sources": [], "grades_mode": True}

# ── 7. Nodos RAG ──────────────────────────────────────────────────────────────
_REWRITE_PROMPT = """Tienes una pregunta de un estudiante de la Universidad de Antioquia.

Si la pregunta está relacionada con el reglamento estudiantil, matrícula, sanciones, calidad de estudiante, procesos académicos o administrativos de la universidad, reformúlala usando vocabulario técnico y formal del reglamento (términos como: causal, sanción, expulsión, matrícula, calidad de estudiante, cancelación, período académico, rendimiento, disciplinario).

Si la pregunta NO está relacionada con la universidad ni su reglamento, devuélvela EXACTAMENTE igual, sin cambios.

Devuelve ÚNICAMENTE la pregunta (reformulada o igual), sin explicaciones ni comillas.

Pregunta: {question}"""

def rewrite_query(state: AgentState) -> AgentState:
    """Reformula la pregunta del usuario en lenguaje documental para mejorar el retrieval."""
    print(" Reescribiendo query...")
    response = llm.invoke(_REWRITE_PROMPT.format(question=state["question"]))
    rewritten = response.content.strip()
    print(f"   original : {state['question']}")
    print(f"   reescrita: {rewritten}")
    return {**state, "search_query": rewritten}

def retrieve(state: AgentState) -> AgentState:
    """Busca fragmentos relevantes en el vectorstore."""
    print(" Buscando en documentos...")
    docs = retriever.invoke(state["search_query"])

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
    graph.add_node("next_semester", next_semester)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)

    graph.add_conditional_edges(START, _route_intent, {
        "grades": "get_grades",
        "next_semester": "next_semester",
        "rag": "rewrite_query",
    })
    graph.add_edge("get_grades", END)
    graph.add_edge("next_semester", END)
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()

agent = build_agent()

# ── 9. Función pública para usar desde la interfaz ───────────────────────────
def ask(question: str, history: list = None, academic_context: dict = None) -> dict:
    result = agent.invoke({
        "question": question,
        "search_query": "",
        "history": history or [],
        "context": "",
        "answer": "",
        "sources": [],
        "grades_mode": False,
        "academic_context": academic_context,
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

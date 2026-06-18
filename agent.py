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
from typing import TypedDict, List
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

# ── 5. Nodos del grafo ────────────────────────────────────────────────────────
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

# ── 6. Construir el grafo ─────────────────────────────────────────────────────
def build_agent():
    graph = StateGraph(AgentState)

    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()

agent = build_agent()

# ── 7. Función pública para usar desde la interfaz ───────────────────────────
def ask(question: str, history: list = None) -> dict:
    result = agent.invoke({
        "question": question,
        "history": history or [],
        "context": "",
        "answer": "",
        "sources": []
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
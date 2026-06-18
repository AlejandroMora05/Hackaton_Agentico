# ingest.py
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

DOCS_PATH = "./docs"

def load_pdfs(path: str):
    all_docs = []
    archivos = [f for f in os.listdir(path) if f.endswith(".pdf")]

    if not archivos:
        raise FileNotFoundError(f"No se encontraron PDFs en {path}")

    for filename in archivos:
        full_path = os.path.join(path, filename)
        loader = PyMuPDFLoader(full_path)
        pages = loader.load()
        all_docs.extend(pages)
        print(f" {filename}: {len(pages)} páginas cargadas")

    print(f"\n Total documentos cargados: {len(all_docs)}")
    return all_docs

# ── 2. Dividir en chunks ──────────────────────────────────────────────────────
def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(docs)
    print(f" Total chunks generados: {len(chunks)}")
    return chunks

# ── 3. Crear vectorstore ──────────────────────────────────────────────────────
def create_vectorstore(chunks):
    print("\n Cargando modelo de embeddings (puede tardar la primera vez)...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    print(" Generando embeddings e indexando...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    print(" Vectorstore guardado en ./chroma_db")
    return vectorstore

def test_retriever(vectorstore):
    print("\n Prueba de búsqueda...")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    resultados = retriever.invoke("¿Cuándo se pierde la calidad de estudiante?")

    print(f"\n Top {len(resultados)} fragmentos encontrados:\n")
    for i, doc in enumerate(resultados, 1):
        fuente = doc.metadata.get("source", "desconocida").split("/")[-1]
        pagina = doc.metadata.get("page", "?")
        print(f"--- Fragmento {i} | {fuente} | pág. {pagina} ---")
        print(doc.page_content[:300])
        print()

if __name__ == "__main__":
    docs   = load_pdfs(DOCS_PATH)
    chunks = split_docs(docs)
    vs     = create_vectorstore(chunks)
    test_retriever(vs)
    print("\n Ingestión completa. Listo para construir el agente.")
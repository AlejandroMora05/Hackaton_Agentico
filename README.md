# Asistente Académico UdeA

Copiloto conversacional para estudiantes de la Universidad de Antioquia. Combina recuperación aumentada por generación (RAG) sobre reglamentos académicos con scraping autenticado del portal institucional para responder preguntas sobre notas, pénsum y matrículas.

---

## Características principales

- **Chat RAG**: responde preguntas sobre reglamentos, guías de matrícula y normativas usando búsqueda híbrida (BM25 + vectores semánticos).
- **Notas en tiempo real**: inicia sesión en el portal UdeA mediante Playwright, extrae y analiza las calificaciones del estudiante.
- **Mapa curricular interactivo**: visualiza el pénsum con estado por materia (aprobada, en curso, en riesgo) y muestra prerrequisitos y correquisitos.
- **Detección de intención**: enruta automáticamente entre consultas de notas, planificación del próximo semestre y preguntas generales.
- **Múltiples proveedores LLM**: Google Gemini 2.5 Flash (por defecto) o NVIDIA Llama 3.3 70B, seleccionable por variable de entorno.

---

## Arquitectura

```
Frontend (HTML/JS)          Backend (FastAPI)          Agente (LangGraph)
┌─────────────────┐         ┌───────────────┐          ┌──────────────────┐
│  index.html     │──chat──▶│   api.py      │──query──▶│   agent.py       │
│  (chat UI)      │         │   :8000       │          │  Intent Router   │
├─────────────────┤         ├───────────────┤          │  ├─ RAG (docs)   │
│  map.html       │──map───▶│  /api/map     │          │  ├─ Notas        │
│  (SVG pénsum)   │         │  /api/grades  │          │  └─ Próx. sem.   │
└─────────────────┘         └──────┬────────┘          └────────┬─────────┘
                                   │                            │
                            ┌──────▼────────┐          ┌────────▼─────────┐
                            │  scrapper/    │          │  ChromaDB        │
                            │  Playwright   │          │  + BM25          │
                            │  → Portal UdeA│          │  (reglamentos)   │
                            └───────────────┘          └──────────────────┘
```

---

## Requisitos

- Python 3.10+
- Una cuenta activa en el portal académico UdeA
- API key de Google Gemini o NVIDIA

---

## Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/AlejandroMora05/Hackaton_Agentico.git
cd Hackaton_Agentico

# 2. Crear entorno virtual e instalar dependencias
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Instalar navegador para Playwright
playwright install chromium

# 4. Configurar variables de entorno
cp .env.example .env             # editar con tus claves
```

### Variables de entorno (`.env`)

```env
LLM_PROVIDER=gemini              # gemini | nvidia
GOOGLE_API_KEY=tu_clave_gemini
NVIDIA_API_KEY=tu_clave_nvidia
```

### Indexar documentos (una sola vez)

Asegurate que los PDFs de reglamentos estén en `./docs/` y ejecuta:

```bash
python ingest.py
```

Esto genera la base vectorial en `./chroma_db/`.

---

## Uso

```bash
# Iniciar el servidor
python api.py
```

Abre [http://localhost:8000](http://localhost:8000) en tu navegador.

### Flujo de uso

1. **Preguntas generales**: escribe directamente en el chat (reglamentos, matrículas, requisitos de grado).
2. **Ver mis notas**: haz clic en el botón correspondiente → se abrirá un navegador para autenticarte en el portal → el agente analizará tus calificaciones.
3. **Mapa curricular**: haz clic en "Ver mi mapa" → visualiza tu avance en el pénsum con colores por estado.

---

## Estructura del proyecto

```
Hackaton_Agentico/
├── api.py              # Servidor FastAPI (punto de entrada principal)
├── agent.py            # Agente LangGraph con enrutamiento de intención
├── llm_factory.py      # Abstracción de proveedores LLM
├── ingest.py           # Pipeline de indexación de PDFs
├── app.py              # Interfaz alternativa con Gradio
├── scrapper/
│   ├── login_session.py    # Gestión de sesión Playwright
│   ├── grades.py           # Parser HTML de notas
│   ├── pensum.py           # Parser HTML de pénsum
│   └── map_data.py         # Combina notas + pénsum
├── static/
│   ├── index.html          # Chat principal
│   └── map.html            # Visualización del mapa curricular
├── docs/               # PDFs de reglamentos (no incluidos en el repo)
├── chroma_db/          # Base vectorial generada por ingest.py (ignorada en git)
├── .auth/              # Cookies de sesión persistentes (ignoradas en git)
└── requirements.txt
```

---

## Stack tecnológico

| Capa | Tecnología |
|---|---|
| Backend | FastAPI, LangGraph, LangChain |
| LLMs | Google Gemini 2.5 Flash / NVIDIA Llama 3.3 70B |
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` (HuggingFace) |
| Vector store | ChromaDB + BM25 (EnsembleRetriever) |
| Scraping | Playwright (async) + BeautifulSoup4 |
| Frontend | HTML, Tailwind CSS, Vanilla JS, marked.js |

---

## Contribuir

1. Haz fork del repositorio.
2. Crea una rama: `git checkout -b feature/mi-mejora`.
3. Abre un Pull Request describiendo los cambios.

---


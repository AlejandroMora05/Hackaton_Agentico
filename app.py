# app.py
import gradio as gr
from agent import ask

# ── Función que conecta la UI con el agente ───────────────────────────────────
def responder(pregunta: str, historial: list):
    if not pregunta.strip():
        return historial, ""

    result = ask(pregunta)

    fuentes_md = "\n".join([f"- {s}" for s in result["sources"]])
    respuesta_completa = f"""{result['answer']}

---
**Fuentes consultadas:**
{fuentes_md}"""

    historial.append((pregunta, respuesta_completa))
    return historial, ""

# ── Preguntas de ejemplo ──────────────────────────────────────────────────────
EJEMPLOS = [
    "¿Cuáles son las causales de pérdida de la calidad de estudiante?",
    "¿Cómo se realiza el proceso de matrícula?",
    "¿Qué es el rendimiento académico insuficiente?",
    "¿Cuáles son los derechos de los estudiantes?",
    "¿Qué sanciones disciplinarias existen?",
]

# ── Interfaz Gradio ───────────────────────────────────────────────────────────
with gr.Blocks(
    title="Copiloto UdeA",
    
) as demo:

    # Header
    gr.Markdown("""
    # Copiloto Administrativo UdeA
    Consulta el **reglamento estudiantil** y los **procesos de matrícula**
    de la Universidad de Antioquia en lenguaje natural.
    > Las respuestas se basan únicamente en los documentos oficiales cargados.
    """)

    # Chat
    chatbot = gr.Chatbot(
        label="Conversación",
        height=450,
        
        
        avatar_images=(None, "https://www.udea.edu.co/favicon.ico")
    )

    # Input
    with gr.Row():
        txt_input = gr.Textbox(
            placeholder="Escribe tu pregunta sobre el reglamento...",
            label="",
            scale=5,
            autofocus=True
        )
        btn_enviar = gr.Button("Enviar ➤", variant="primary", scale=1)

    # Ejemplos
    gr.Markdown("#### Preguntas frecuentes")
    gr.Examples(
        examples=EJEMPLOS,
        inputs=txt_input,
        label=""
    )

    # Footer
    gr.Markdown("""
    ---
    Powered by **LangGraph** · **Gemini 2.5 Flash** · **ChromaDB**
    Fuente: [normativa.udea.edu.co](https://normativa.udea.edu.co)
    """)

    # Estado del historial
    estado = gr.State([])

    # Eventos
    btn_enviar.click(
        fn=responder,
        inputs=[txt_input, estado],
        outputs=[chatbot, txt_input]
    )
    txt_input.submit(
        fn=responder,
        inputs=[txt_input, estado],
        outputs=[chatbot, txt_input]
    )

if __name__ == "__main__":
    demo.launch(share=False, theme=gr.themes.Soft(primary_hue="blue")) 
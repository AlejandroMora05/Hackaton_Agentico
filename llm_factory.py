# llm_factory.py
import os
from dotenv import load_dotenv

load_dotenv()

def get_llm(provider: str = None):
    provider = provider or os.getenv("LLM_PROVIDER", "gemini")

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.2
        )
    elif provider == "nvidia":
        from langchain_nvidia_ai_endpoints import ChatNVIDIA
        return ChatNVIDIA(
            model="nvidia/llama-3.1-nemotron-super-49b-v1",
            nvidia_api_key=os.getenv("NVIDIA_API_KEY"),
            temperature=0.2
        )

    raise ValueError(f"Proveedor desconocido: {provider}")
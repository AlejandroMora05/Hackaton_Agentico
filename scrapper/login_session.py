# scrapper/login_session.py
"""Sesión de login persistente, compartida entre "Ver mis notas" y "Ver mi
mapa" (un solo login institucional sirve para ambas).

A diferencia del scraper antiguo (que abría el navegador y sondeaba solo),
aquí el flujo es guiado por el estudiante desde la UI, en pasos explícitos:

  1. El estudiante pide iniciar sesión. Si ya tenemos una sesión guardada en
     disco (de un análisis anterior, de cualquiera de los dos botones) y
     todavía está viva, la reutilizamos en un navegador headless: sin abrir
     ninguna ventana, en segundos.
  2. Si no hay sesión guardada o ya expiró, abrimos UNA sola pestaña visible
     en el login institucional (y la traemos al frente con xdotool, para que
     no quede oculta detrás de otras ventanas). El estudiante inicia sesión
     ahí.
  3. El estudiante pide ir a notas o a pénsum: navegamos esa misma pestaña
     directamente a la URL correspondiente (sin que el estudiante tenga que
     encontrarla él mismo) y la volvemos a traer al frente. Si esa app pide
     contraseña otra vez, el estudiante la ingresa ahí.
  4. El estudiante pide "Analizar": esperamos unos segundos, leemos el HTML
     ya cargado, y guardamos las cookies en disco (compartidas entre ambos
     botones) para la próxima vez.

No se abren pestañas adicionales ni se navega nada hasta que el estudiante
lo pide explícitamente desde la UI.

Usa la API async de Playwright (en vez de la sync usada antes en grades.py)
porque la sesión debe sobrevivir entre peticiones HTTP distintas, y los
objetos sync de Playwright están atados al hilo donde se crearon.
"""
import asyncio
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Set

from playwright.async_api import Browser, BrowserContext, Page, Playwright, async_playwright

from scrapper.grades import NOTAS_URL, parse_grades_html
from scrapper.map_data import build_map_report
from scrapper.pensum import PENSUM_READY_MARKER, PENSUM_URL, parse_pensum_html

LOGIN_URL = "https://www.udea.edu.co/wps/portal/udea/web/inicio/login"
NOTAS_READY_MARKER = "MATERIAS QUE ESTAS CURSANDO"
ANALYZE_DELAY_SECONDS = 3
SESSION_MAX_AGE_SECONDS = 15 * 60

# Cookies de una sesión institucional ya autenticada, guardadas localmente
# (fuera de git) y compartidas entre "Ver mis notas" y "Ver mi mapa", para no
# pedirle login al estudiante en cada análisis.
STORAGE_STATE_PATH = Path(__file__).resolve().parent.parent / ".auth" / "udea_storage_state.json"

TARGETS = {
    "pensum": {"url": PENSUM_URL, "marker": PENSUM_READY_MARKER, "case_insensitive": True},
    "notas": {"url": NOTAS_URL, "marker": NOTAS_READY_MARKER, "case_insensitive": False},
}


def _matches_marker(target: str, html: str) -> bool:
    cfg = TARGETS[target]
    haystack = html.upper() if cfg["case_insensitive"] else html
    return cfg["marker"] in haystack


class SessionNotFoundError(Exception):
    """La sesión expiró, se cerró o nunca existió."""


class NotReadyError(Exception):
    """La pestaña no muestra todavía la página esperada."""


def _list_window_ids() -> Set[str]:
    """IDs de todas las ventanas visibles en el escritorio actual (xdotool)."""
    try:
        result = subprocess.run(
            ["xdotool", "search", "--onlyvisible", "--name", "."],
            capture_output=True, text=True, timeout=3,
        )
        return set(result.stdout.split())
    except Exception:
        return set()


def _focus_window(window_id: str) -> None:
    try:
        subprocess.run(["xdotool", "windowactivate", "--sync", window_id], timeout=3)
        subprocess.run(["xdotool", "windowraise", window_id], timeout=3)
    except Exception:
        pass  # el enfoque automático es solo una mejora de UX, no algo crítico


def _focus_new_window(before_ids: Set[str]) -> Optional[str]:
    """Encuentra la ventana que apareció después de lanzar el navegador y la
    trae al frente, para que el estudiante no la pierda detrás de otras."""
    new_ids = _list_window_ids() - before_ids
    if not new_ids:
        return None
    window_id = sorted(new_ids)[-1]
    _focus_window(window_id)
    return window_id


@dataclass
class _Session:
    playwright: Playwright
    browser: Browser
    context: BrowserContext
    page: Page
    created_at: float
    cached_html: Optional[str] = None
    window_id: Optional[str] = None


_sessions: Dict[str, _Session] = {}


async def _try_restore_session(target: str) -> Optional[_Session]:
    """Intenta reanudar la sesión guardada en disco, sin abrir ventana
    visible. Devuelve None si no hay sesión guardada o ya no es válida para
    el destino pedido."""
    if not STORAGE_STATE_PATH.exists():
        return None

    playwright = await async_playwright().start()
    browser = await playwright.chromium.launch(headless=True)
    context = await browser.new_context(storage_state=str(STORAGE_STATE_PATH))
    page = await context.new_page()
    await page.goto(TARGETS[target]["url"], wait_until="domcontentloaded")
    content = await page.content()

    if _matches_marker(target, content):
        return _Session(playwright, browser, context, page, time.time(), cached_html=content)

    await browser.close()
    await playwright.stop()
    return None


async def start_session(target: str) -> Dict:
    restored = await _try_restore_session(target)
    if restored is not None:
        session_id = str(uuid.uuid4())
        _sessions[session_id] = restored
        return {"session_id": session_id, "ready": True}

    before_ids = await asyncio.to_thread(_list_window_ids)

    playwright = await async_playwright().start()
    browser = await playwright.chromium.launch(headless=False, args=["--start-maximized"])
    context = await browser.new_context(no_viewport=True)
    page = await context.new_page()
    await page.goto(LOGIN_URL)

    window_id = await asyncio.to_thread(_focus_new_window, before_ids)

    session_id = str(uuid.uuid4())
    _sessions[session_id] = _Session(playwright, browser, context, page, time.time(), window_id=window_id)
    return {"session_id": session_id, "ready": False}


async def goto_target(session_id: str, target: str) -> None:
    """Navega la pestaña ya abierta directamente a la app pedida (en vez de
    pedirle al estudiante que la busque) y la vuelve a traer al frente."""
    session = await _check_session(session_id)

    await session.page.goto(TARGETS[target]["url"], wait_until="domcontentloaded")
    if session.window_id:
        await asyncio.to_thread(_focus_window, session.window_id)


async def _close_session(session_id: str) -> None:
    session = _sessions.pop(session_id, None)
    if session is None:
        return
    await session.browser.close()
    await session.playwright.stop()


async def cancel_session(session_id: str) -> None:
    await _close_session(session_id)


async def _check_session(session_id: str) -> _Session:
    session = _sessions.get(session_id)
    if session is None:
        raise SessionNotFoundError("La sesión expiró o no existe. Vuelve a iniciar sesión.")
    if time.time() - session.created_at > SESSION_MAX_AGE_SECONDS:
        await _close_session(session_id)
        raise SessionNotFoundError("La sesión expiró por inactividad. Vuelve a iniciar sesión.")
    return session


async def _capture_target_html(session: _Session, target: str) -> str:
    if session.cached_html is not None:
        return session.cached_html

    await asyncio.sleep(ANALYZE_DELAY_SECONDS)
    html = await session.page.content()
    if not _matches_marker(target, html):
        nombre = "pénsum" if target == "pensum" else "notas"
        raise NotReadyError(
            f"No vimos tu {nombre} cargado en la pestaña. Ve a esa sección, inicia sesión si te lo pide, y haz click en Analizar otra vez."
        )
    return html


async def _persist_and_close(session_id: str, session: _Session) -> None:
    STORAGE_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    await session.context.storage_state(path=str(STORAGE_STATE_PATH))
    await _close_session(session_id)


async def _fetch_creditos_by_codigo(session: _Session) -> Dict[str, Optional[int]]:
    """Consulta el pénsum (misma sesión SSO, sin pedirle nada al estudiante)
    solo para conocer los créditos de cada materia: la página de notas no
    los incluye, y el promedio ponderado los necesita."""
    try:
        await session.page.goto(PENSUM_URL, wait_until="domcontentloaded")
        pensum_html = await session.page.content()
    except Exception:
        return {}

    if PENSUM_READY_MARKER not in pensum_html.upper():
        return {}

    pensum = parse_pensum_html(pensum_html)
    creditos_por_codigo: Dict[str, Optional[int]] = {}
    for nivel in pensum["niveles"]:
        for materia in nivel["materias"]:
            creditos_por_codigo[materia["codigo"]] = materia["creditos"]
    for grupo in pensum["electivas"]:
        for materia in grupo["materias"]:
            creditos_por_codigo[materia["codigo"]] = materia["creditos"]
    return creditos_por_codigo


async def analyze_notas(session_id: str) -> Dict:
    session = await _check_session(session_id)
    notas_html = await _capture_target_html(session, "notas")
    creditos_por_codigo = await _fetch_creditos_by_codigo(session)
    await _persist_and_close(session_id, session)

    report = parse_grades_html(notas_html)
    for materia in report["materias"]:
        materia["creditos"] = creditos_por_codigo.get(materia["codigo"])
    return report


async def analyze_map(session_id: str) -> Dict:
    session = await _check_session(session_id)

    pensum_html = await _capture_target_html(session, "pensum")

    await session.page.goto(NOTAS_URL, wait_until="domcontentloaded")
    notas_html = await session.page.content()
    if not _matches_marker("notas", notas_html):
        raise NotReadyError("No se pudo cargar tus notas todavía. Inténtalo de nuevo en unos segundos.")

    await _persist_and_close(session_id, session)
    return build_map_report(pensum_html, notas_html)

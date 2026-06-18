# scrapper/map_data.py
"""Construye el mapa del pensum (niveles + electivas) cruzado con el estado
de las materias que el estudiante está cursando este semestre.

Reutiliza una sola sesión de navegador autenticada (mismo SSO de la UdeA)
para no pedirle al estudiante que inicie sesión dos veces: primero se
captura el pénsum y, sin cerrar el navegador, se captura notas.
"""
import time
from typing import Dict

from playwright.sync_api import sync_playwright

from scrapper.grades import (
    LOGIN_URL,
    LOGIN_WAIT_SECONDS,
    NOTAS_URL,
    POLL_INTERVAL_SECONDS,
    LoginTimeoutError,
    parse_grades_html,
)
from scrapper.pensum import parse_pensum_html, wait_for_pensum_html

NOTAS_READY_MARKER = "MATERIAS QUE ESTAS CURSANDO"


def _wait_for_notas_html(page, deadline: float) -> str:
    while time.time() < deadline:
        page.goto(NOTAS_URL, wait_until="domcontentloaded")
        content = page.content()
        if NOTAS_READY_MARKER in content:
            return content
        time.sleep(POLL_INTERVAL_SECONDS)
    raise LoginTimeoutError("No se detectó el inicio de sesión a tiempo.")


def open_browser_and_get_html(timeout_seconds: int = LOGIN_WAIT_SECONDS) -> Dict[str, str]:
    """Abre el login institucional una sola vez y captura, en orden, el HTML
    autenticado del pénsum y el de notas (misma sesión SSO)."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()

        login_page = context.new_page()
        login_page.goto(LOGIN_URL)

        check_page = context.new_page()
        deadline = time.time() + timeout_seconds

        pensum_html = wait_for_pensum_html(check_page, deadline)
        if pensum_html is None:
            browser.close()
            raise LoginTimeoutError("No se detectó el inicio de sesión a tiempo.")

        notas_html = _wait_for_notas_html(check_page, deadline)

        browser.close()
        return {"pensum_html": pensum_html, "notas_html": notas_html}


def build_map_report(pensum_html: str, notas_html: str) -> Dict:
    pensum = parse_pensum_html(pensum_html)
    notas = parse_grades_html(notas_html)

    estado_por_codigo = {m["codigo"]: m for m in notas["materias"] if m.get("codigo")}

    def con_estado(materia: Dict) -> Dict:
        estado = estado_por_codigo.get(materia["codigo"])
        if estado is None:
            return {**materia, "cursando": False}
        return {
            **materia,
            "cursando": True,
            "color": estado["color"],
            "mensaje": estado["mensaje"],
            "porcentaje_evaluado": estado["porcentaje_evaluado"],
            "nota_acumulada": estado["nota_acumulada"],
            "nota_definitiva": estado["nota_definitiva"],
        }

    niveles = [
        {**nivel, "materias": [con_estado(m) for m in nivel["materias"]]}
        for nivel in pensum["niveles"]
    ]
    electivas = [
        {**grupo, "materias": [con_estado(m) for m in grupo["materias"]]}
        for grupo in pensum["electivas"]
    ]

    return {
        "estudiante": pensum["estudiante"],
        "documento": pensum["documento"],
        "programa": pensum["programa"],
        "semestre": notas.get("semestre", ""),
        "niveles": niveles,
        "electivas": electivas,
    }


def get_map_report(timeout_seconds: int = LOGIN_WAIT_SECONDS) -> Dict:
    html = open_browser_and_get_html(timeout_seconds)
    return build_map_report(html["pensum_html"], html["notas_html"])


if __name__ == "__main__":
    import json

    print(json.dumps(get_map_report(), indent=2, ensure_ascii=False))

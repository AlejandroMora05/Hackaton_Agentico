# scrapper/grades.py
"""Trazabilidad de notas UdeA.

Como la página de notas requiere sesión iniciada en el portal institucional,
este módulo abre una ventana de navegador real (Playwright) para que el
estudiante inicie sesión manualmente. Una vez dentro, captura el HTML ya
autenticado y lo parsea para calcular, por materia, qué tan cerca está de
ganarla.
"""
import re
import time
from typing import Dict, List, Optional

from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

NOTAS_URL = "https://tsone.udea.edu.co/php_notas_estudiante/?app=consultar"
LOGIN_URL = "https://www.udea.edu.co/wps/portal/udea/web/inicio/login"
APPROVAL_GRADE = 3.0
MAX_GRADE = 5.0
LOGIN_WAIT_SECONDS = 300
POLL_INTERVAL_SECONDS = 3


class LoginTimeoutError(Exception):
    """El estudiante no completó el inicio de sesión a tiempo."""


def open_browser_and_get_html(timeout_seconds: int = LOGIN_WAIT_SECONDS) -> str:
    """Abre una ventana de navegador en el login institucional para que el
    estudiante inicie sesión manualmente. Mientras tanto, en una pestaña
    aparte (sin interrumpir el login), sondea la página de notas hasta que
    la sesión quede activa, y devuelve su HTML ya autenticado."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()

        login_page = context.new_page()
        login_page.goto(LOGIN_URL)

        check_page = context.new_page()
        deadline = time.time() + timeout_seconds
        html = None
        while time.time() < deadline:
            check_page.goto(NOTAS_URL, wait_until="domcontentloaded")
            content = check_page.content()
            if "MATERIAS QUE ESTAS CURSANDO" in content:
                html = content
                break
            time.sleep(POLL_INTERVAL_SECONDS)

        browser.close()

        if html is None:
            raise LoginTimeoutError("No se detectó el inicio de sesión a tiempo.")
        return html


def _parse_float(text: str) -> Optional[float]:
    try:
        return float(text.strip().replace("%", ""))
    except (ValueError, AttributeError):
        return None


def compute_status(
    porcentaje_evaluado: float,
    nota_acumulada: float,
    nota_definitiva: Optional[float],
) -> Dict:
    """Determina el color y el mensaje de trazabilidad de una materia.

    La materia se evalúa según si todavía es matemáticamente posible
    ganarla con el porcentaje que falta por evaluar:
      - rojo: ya no alcanza ni sacando 5.0 en todo lo que falta.
      - naranja: alcanza, pero necesita más de la nota mínima (3.0) en el resto.
      - verde: con 3.0 o menos en el resto ya gana la materia.
    """
    porcentaje_restante = max(0.0, 100.0 - porcentaje_evaluado)

    if porcentaje_restante <= 0:
        definitiva = nota_definitiva if nota_definitiva is not None else nota_acumulada
        if definitiva >= APPROVAL_GRADE:
            return {
                "color": "green",
                "nota_necesaria": None,
                "mensaje": f"Materia ganada con nota definitiva {definitiva:.1f}.",
            }
        return {
            "color": "red",
            "nota_necesaria": None,
            "mensaje": f"Materia perdida con nota definitiva {definitiva:.1f}.",
        }

    nota_necesaria = (APPROVAL_GRADE - nota_acumulada) / (porcentaje_restante / 100.0)

    if nota_necesaria > MAX_GRADE:
        return {
            "color": "red",
            "nota_necesaria": round(nota_necesaria, 2),
            "mensaje": "Ya no es matemáticamente posible ganar la materia.",
        }
    if nota_necesaria > APPROVAL_GRADE:
        return {
            "color": "orange",
            "nota_necesaria": round(nota_necesaria, 2),
            "mensaje": f"Necesitas un promedio de {nota_necesaria:.2f} en lo que falta para ganar.",
        }
    if nota_necesaria > 0:
        return {
            "color": "green",
            "nota_necesaria": round(nota_necesaria, 2),
            "mensaje": f"Vas bien: con un promedio de {nota_necesaria:.2f} en lo que falta, ya ganas.",
        }
    return {
        "color": "green",
        "nota_necesaria": 0,
        "mensaje": "Ya ganaste la materia matemáticamente, sin importar lo que falta.",
    }


def parse_grades_html(html: str) -> Dict:
    """Parsea el HTML de la página de notas y devuelve la trazabilidad por materia."""
    soup = BeautifulSoup(html, "html.parser")

    info_text = soup.find("p", class_="card-text").get_text(" ", strip=True)
    info = {"estudiante": "", "programa": "", "semestre": ""}
    estudiante_match = re.search(r"Estudiante\s*:\s*(.*?)\s*Programa\s*:", info_text)
    programa_match = re.search(r"Programa\s*:\s*(.*?)\s*Semestre\s*:", info_text)
    semestre_match = re.search(r"Semestre\s*:\s*(\S+)", info_text)
    if estudiante_match:
        info["estudiante"] = estudiante_match.group(1).strip()
    if programa_match:
        info["programa"] = programa_match.group(1).strip()
    if semestre_match:
        info["semestre"] = semestre_match.group(1).strip()

    materias: List[Dict] = []
    for card in soup.select(".pt-3 > .card"):
        header = card.find("div", class_="card-header")
        nombre_materia = header.get_text(strip=True) if header else "Materia"

        evaluaciones = []
        table = card.find("table")
        if table and table.find("tbody"):
            for row in table.find("tbody").find_all("tr"):
                cols = row.find_all("td")
                if len(cols) != 3:
                    continue
                nota_text = cols[2].get_text(strip=True)
                nota = _parse_float(nota_text)
                evaluaciones.append({
                    "nombre": cols[0].get_text(strip=True),
                    "porcentaje": _parse_float(cols[1].get_text(strip=True)) or 0.0,
                    "nota": nota,
                    "reportada": nota is not None,
                })

        notas_pane = card.select_one(".tab-pane[id^='nav-notas-']")
        resumen_text = ""
        if notas_pane:
            paragraphs = notas_pane.find_all("p")
            if paragraphs:
                resumen_text = paragraphs[-1].get_text(" ", strip=True)

        porcentaje_match = re.search(r"Porcentaje evaluado\s*:\s*([\d.]+)", resumen_text)
        acumulada_match = re.search(r"Nota acumulada\s*:\s*([\d.]+)", resumen_text)
        definitiva_match = re.search(r"Nota definitiva\s*:\s*([\d.]+)", resumen_text)

        porcentaje_evaluado = float(porcentaje_match.group(1)) if porcentaje_match else 0.0
        nota_acumulada = float(acumulada_match.group(1)) if acumulada_match else 0.0
        nota_definitiva = float(definitiva_match.group(1)) if definitiva_match else None

        estado = compute_status(porcentaje_evaluado, nota_acumulada, nota_definitiva)

        materias.append({
            "nombre": nombre_materia,
            "evaluaciones": evaluaciones,
            "porcentaje_evaluado": porcentaje_evaluado,
            "nota_acumulada": round(nota_acumulada, 3),
            "nota_definitiva": nota_definitiva,
            **estado,
        })

    return {**info, "materias": materias}


def get_grades_report(timeout_seconds: int = LOGIN_WAIT_SECONDS) -> Dict:
    """Punto de entrada único: abre el navegador, espera el login y devuelve el reporte parseado."""
    html = open_browser_and_get_html(timeout_seconds)
    return parse_grades_html(html)


if __name__ == "__main__":
    import json

    print(json.dumps(get_grades_report(), indent=2, ensure_ascii=False))

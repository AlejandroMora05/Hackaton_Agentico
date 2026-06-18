# scrapper/pensum.py
"""Scraper del pénsum del estudiante en UdeA.

Misma idea que scrapper/grades.py: la página requiere sesión institucional,
así que se abre un navegador real para que el estudiante inicie sesión y,
una vez autenticado, se captura y parsea el HTML.
"""
import re
import time
from typing import Dict, List, Optional

from bs4 import BeautifulSoup
from playwright.sync_api import Page

PENSUM_URL = "https://ayudame2.udea.edu.co/php_pensum_estudiante/?app=consultar"
PENSUM_READY_MARKER = "PÉNSUM DEL PROGRAMA"


def wait_for_pensum_html(page: Page, deadline: float, poll_interval_seconds: float = 3) -> Optional[str]:
    """Navega repetidamente a la página de pénsum (en la pestaña dada) hasta
    detectar contenido autenticado o hasta que se acabe el tiempo."""
    while time.time() < deadline:
        page.goto(PENSUM_URL, wait_until="domcontentloaded")
        content = page.content()
        if PENSUM_READY_MARKER in content.upper():
            return content
        time.sleep(poll_interval_seconds)
    return None


def _parse_int(text: str) -> Optional[int]:
    try:
        return int(float(text.strip()))
    except (ValueError, AttributeError):
        return None


def _parse_materia_row(tds: List) -> Dict:
    nombre_cell = tds[0].get_text(" ", strip=True)
    match = re.match(r"^(.*?)\s*\[(\d+)\]\s*$", nombre_cell)
    nombre, codigo = (match.group(1).strip(), match.group(2)) if match else (nombre_cell, "")

    creditos = _parse_int(tds[1].get_text(strip=True)) if len(tds) > 1 else None

    prerequisitos: List[str] = []
    correquisitos: List[str] = []
    if len(tds) >= 5:
        prerequisitos = [a.get_text(strip=True) for a in tds[3].find_all("a")]
        correquisitos = [a.get_text(strip=True) for a in tds[4].find_all("a")]

    return {
        "codigo": codigo,
        "nombre": nombre,
        "creditos": creditos,
        "prerequisitos": prerequisitos,
        "correquisitos": correquisitos,
    }


def parse_pensum_html(html: str) -> Dict:
    """Parsea el HTML del pénsum en niveles fijos (materias básicas) y grupos
    de electivas (que no tienen un nivel fijo, el estudiante elige)."""
    soup = BeautifulSoup(html, "html.parser")
    card_body = soup.select_one("#main_div .card-body")

    info_text = card_body.find("p", class_="card-text").get_text(" ", strip=True)
    info = {"estudiante": "", "documento": "", "programa": ""}
    estudiante_match = re.search(r"Estudiante\s*:\s*(.*?)\s*Documento de identidad\s*:", info_text)
    documento_match = re.search(r"Documento de identidad\s*:\s*(\S+)\s*Programa\s*:", info_text)
    programa_match = re.search(r"Programa\s*:\s*(.*)$", info_text)
    if estudiante_match:
        info["estudiante"] = estudiante_match.group(1).strip()
    if documento_match:
        info["documento"] = documento_match.group(1).strip()
    if programa_match:
        info["programa"] = programa_match.group(1).strip()

    niveles: List[Dict] = []
    electivas: List[Dict] = []
    seccion_actual = ""

    # La página anida el bloque de electivas (h4 + sus divs) dentro del último
    # div.table-responsive de "materias básicas", así que se recorre todo el
    # árbol en orden de documento en vez de solo los hijos directos.
    for child in card_body.find_all(["h4", "div"], recursive=True):
        if child.name == "h4":
            seccion_actual = child.get_text(strip=True)
            continue

        if "table-responsive" not in (child.get("class") or []):
            continue

        heading = child.find("h5", recursive=False)
        if heading is None:
            continue  # es un div contenedor (p.ej. el wrapper de electivas), no una tabla
        heading_text = heading.get_text(" ", strip=True)
        table = child.find("table")
        if not table or not table.find("tbody"):
            continue

        materias = [_parse_materia_row(tr.find_all("td")) for tr in table.find("tbody").find_all("tr")]

        if "BÁSICAS" in seccion_actual:
            nivel_match = re.search(r"NIVEL\s*(\d+)", heading_text)
            nivel = int(nivel_match.group(1)) if nivel_match else None
            niveles.append({"nivel": nivel, "materias": materias})
        else:
            grupo_match = re.match(r"^(.*?)\s*\(\d+\s*materias?\)", heading_text)
            grupo = grupo_match.group(1).strip() if grupo_match else heading_text
            requisito_el = child.find("p")
            requisito = requisito_el.get_text(strip=True) if requisito_el else ""
            electivas.append({"grupo": grupo, "requisito": requisito, "materias": materias})

    niveles.sort(key=lambda n: n["nivel"] if n["nivel"] is not None else 999)
    return {**info, "niveles": niveles, "electivas": electivas}

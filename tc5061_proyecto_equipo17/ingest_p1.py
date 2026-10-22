import json
from pathlib import Path
from urllib.parse import urlparse

import typer
from loguru import logger

from tc5061_proyecto_equipo17.config import RAW_DATA_DIR
from tc5061_proyecto_equipo17.dataset import descargar_archivo

app = typer.Typer(help="Descarga reproducible de datos para P1 - Equipo 17")

CATALOGO = Path(__file__).resolve().parent / "urls_p1.json"


def meses_esperados():
    return [
        f"datos_abiertos_{anio}_{mes:02d}.csv"
        for anio in range(2024, 2027)
        for mes in range(1, 13)
        if (2024, 10) <= (anio, mes) <= (2026, 9)
    ]


def descargar_p1():
    catalogo = json.loads(CATALOGO.read_text(encoding="utf-8"))
    esperados = meses_esperados()

    if set(catalogo) != set(esperados):
        faltantes = sorted(set(esperados) - set(catalogo))
        extras = sorted(set(catalogo) - set(esperados))
        raise ValueError(
            f"Catálogo incompleto. Faltantes: {faltantes}. Extras: {extras}"
        )

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    for numero, nombre in enumerate(esperados, start=1):
        url = catalogo[nombre]
        parsed = urlparse(url)

        if (
            parsed.scheme != "https"
            or parsed.hostname not in {"mibici.net", "www.mibici.net"}
            or Path(parsed.path).name != nombre
        ):
            raise ValueError(f"URL no válida para {nombre}: {url}")

        logger.info(f"[{numero}/24] Verificando {nombre}")
        descargar_archivo(RAW_DATA_DIR / nombre, url)

    incompletos = [
        nombre
        for nombre in esperados
        if not (RAW_DATA_DIR / nombre).is_file()
        or (RAW_DATA_DIR / nombre).stat().st_size == 0
    ]

    if incompletos:
        raise RuntimeError(f"Archivos faltantes o vacíos: {incompletos}")

    logger.success("Los 24 archivos de P1 están disponibles.")
    return RAW_DATA_DIR


@app.command()
def main():
    descargar_p1()


if __name__ == "__main__":
    app()

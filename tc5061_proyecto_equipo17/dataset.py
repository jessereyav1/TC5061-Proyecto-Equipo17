
import hashlib
import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import typer
from loguru import logger

from tc5061_proyecto_equipo17.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer(help="Pipeline de datos MiBici - Equipo 17")

COLUMNAS_REQUERIDAS = {
    "viaje_id",
    "inicio_del_viaje",
    "fin_del_viaje",
    "origen_id",
    "destino_id",
}

COLUMNAS_SALIDA = [
    "viaje_id",
    "inicio_del_viaje",
    "fin_del_viaje",
    "origen_id",
    "destino_id",
    "duracion_minutos",
    "fecha",
    "hora",
    "dia_semana",
    "mes",
]


def calcular_sha256(ruta: Path) -> str:
    """Calcula el hash SHA-256 de un archivo."""
    sha = hashlib.sha256()

    with ruta.open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1024 * 1024), b""):
            sha.update(bloque)

    return sha.hexdigest()


def normalizar_columna(nombre: str) -> str:
    """Convierte nombres de columnas a snake_case sin acentos."""
    nombre = unicodedata.normalize("NFKD", str(nombre))
    nombre = "".join(
        caracter for caracter in nombre
        if not unicodedata.combining(caracter)
    )
    nombre = nombre.strip().lower()
    nombre = re.sub(r"[^a-z0-9]+", "_", nombre)

    return nombre.strip("_")


def procesar_archivo(ruta_csv: Path) -> dict:
    """Procesa un CSV de MiBici y genera un archivo Parquet."""

    logger.info(f"Leyendo archivo: {ruta_csv.name}")

    df = pd.read_csv(
        ruta_csv,
        encoding="cp1252",
        low_memory=False,
        dtype={
            "Viaje_Id": "string",
            "Usuario_Id": "string",
            "Origen_Id": "string",
            "Destino_Id": "string",
        },
    )

    registros_originales = len(df)

    df.columns = [normalizar_columna(c) for c in df.columns]

    faltantes = COLUMNAS_REQUERIDAS - set(df.columns)

    if faltantes:
        raise ValueError(
            f"Columnas obligatorias faltantes: {sorted(faltantes)}"
        )

    df = df.drop_duplicates(subset=["viaje_id"])

    for columna in ["inicio_del_viaje", "fin_del_viaje"]:
        df[columna] = pd.to_datetime(
            df[columna],
            errors="coerce",
        )

    for columna in ["origen_id", "destino_id"]:
        df[columna] = pd.to_numeric(
            df[columna],
            errors="coerce",
        ).astype("Int64")

    df = df.dropna(
        subset=[
            "inicio_del_viaje",
            "fin_del_viaje",
            "origen_id",
            "destino_id",
        ]
    ).copy()

    df["duracion_minutos"] = (
        df["fin_del_viaje"] - df["inicio_del_viaje"]
    ).dt.total_seconds() / 60

    df = df[df["duracion_minutos"] > 0].copy()

    df["fecha"] = df["inicio_del_viaje"].dt.date
    df["hora"] = df["inicio_del_viaje"].dt.hour
    df["dia_semana"] = df["inicio_del_viaje"].dt.dayofweek
    df["mes"] = df["inicio_del_viaje"].dt.month

    # Excluimos Usuario_Id, genero y año de nacimiento.
    # Conservamos el ID del viaje solo para trazabilidad interna.
    df = df[COLUMNAS_SALIDA].copy()

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    ruta_parquet = (
        PROCESSED_DATA_DIR / f"{ruta_csv.stem}_procesado.parquet"
    )

    df.to_parquet(
        ruta_parquet,
        engine="pyarrow",
        index=False,
        compression="snappy",
    )

    resultado = {
        "archivo_origen": ruta_csv.name,
        "sha256_origen": calcular_sha256(ruta_csv),
        "archivo_procesado": ruta_parquet.name,
        "sha256_procesado": calcular_sha256(ruta_parquet),
        "registros_originales": registros_originales,
        "registros_procesados": len(df),
        "registros_descartados": registros_originales - len(df),
        "columnas_procesadas": COLUMNAS_SALIDA,
    }

    logger.success(
        f"{ruta_csv.name}: {len(df):,} registros procesados"
    )

    return resultado


@app.command()
def main(
    input_dir: Path = RAW_DATA_DIR,
    output_manifest: Path = (
        PROCESSED_DATA_DIR / "manifest.json"
    ),
):
    """Procesa todos los archivos CSV disponibles en data/raw."""

    archivos = sorted(input_dir.glob("*.csv"))

    if not archivos:
        logger.error(f"No se encontraron CSV en {input_dir}")
        raise typer.Exit(code=1)

    resultados = []

    for archivo in archivos:
        resultados.append(procesar_archivo(archivo))

    manifiesto = {
        "proyecto": "TC5061-Proyecto-Equipo17",
        "dataset": "MiBici Guadalajara",
        "fecha_procesamiento_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "total_archivos": len(resultados),
        "total_registros": sum(
            item["registros_procesados"]
            for item in resultados
        ),
        "archivos": resultados,
    }

    output_manifest.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_manifest.write_text(
        json.dumps(
            manifiesto,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    logger.success(
        f"Manifiesto generado: {output_manifest}"
    )
    logger.success("Pipeline MiBici completado.")


if __name__ == "__main__":
    app()


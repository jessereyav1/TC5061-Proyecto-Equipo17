
import hashlib
import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd
import typer
from loguru import logger

from tc5061_proyecto_equipo17.config import (
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)


# ============================================================
# CONFIGURACION GENERAL
# ============================================================

app = typer.Typer(
    help="Pipeline de datos MiBici Guadalajara - Equipo 17"
)

# URLs oficiales verificadas de enero y febrero de 2025.
URLS_MIBICI = {
    "datos_abiertos_2025_01.csv": (
        "https://www.mibici.net/site/assets/files/97653/"
        "datos_abiertos_2025_01.csv"
    ),
    "datos_abiertos_2025_02.csv": (
        "https://www.mibici.net/site/assets/files/99879/"
        "datos_abiertos_2025_02.csv"
    ),
}

# Se mantienen por compatibilidad con el codigo anterior.
NOMBRE_CSV = "datos_abiertos_2025_01.csv"
URL_MIBICI = URLS_MIBICI[NOMBRE_CSV]

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


# ============================================================
# FUNCIONES DE UTILIDAD
# ============================================================

def calcular_sha256(ruta: Path) -> str:
    """Calcula el hash SHA-256 de un archivo."""

    sha = hashlib.sha256()

    with ruta.open("rb") as archivo:
        for bloque in iter(
            lambda: archivo.read(1024 * 1024),
            b"",
        ):
            sha.update(bloque)

    return sha.hexdigest()


def normalizar_columna(nombre: str) -> str:
    """Convierte nombres de columnas a snake_case sin acentos."""

    nombre = unicodedata.normalize("NFKD", str(nombre))

    nombre = "".join(
        caracter
        for caracter in nombre
        if not unicodedata.combining(caracter)
    )

    nombre = nombre.strip().lower()
    nombre = re.sub(r"[^a-z0-9]+", "_", nombre)

    return nombre.strip("_")


# ============================================================
# DESCARGA AUTOMATICA
# ============================================================

def descargar_datos(
    destino: Path,
    url: str = URL_MIBICI,
) -> Path:
    """
    Descarga un CSV oficial de MiBici si no existe.

    Utiliza un archivo temporal para evitar conservar
    descargas incompletas.

    Verifica que el archivo tenga contenido y que incluya
    las columnas obligatorias.
    """

    destino.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Evitar descargas repetidas.
    if destino.is_file():

        if destino.stat().st_size == 0:
            raise ValueError(
                f"El archivo existente esta vacio: {destino}"
            )

        logger.info(
            f"El CSV ya existe. Se omite descarga: {destino.name}"
        )

        return destino

    temporal = destino.with_name(
        destino.name + ".part"
    )

    logger.info(
        f"Descargando datos desde: {url}"
    )

    solicitud = Request(
        url,
        headers={
            "User-Agent": "MiBici-Equipo17/1.0"
        },
    )

    try:

        with urlopen(
            solicitud,
            timeout=60,
        ) as respuesta:

            with temporal.open("wb") as archivo:

                while True:

                    bloque = respuesta.read(
                        1024 * 1024
                    )

                    if not bloque:
                        break

                    archivo.write(bloque)

        if temporal.stat().st_size == 0:
            raise ValueError(
                "La descarga produjo un archivo vacio."
            )

        # Verificacion basica del encabezado.
        with temporal.open(
            "r",
            encoding="cp1252",
        ) as archivo:

            encabezado = archivo.readline().strip()

        columnas = {
            normalizar_columna(
                columna.strip('"')
            )
            for columna in encabezado.split(",")
        }

        faltantes = COLUMNAS_REQUERIDAS - columnas

        if faltantes:
            raise ValueError(
                "El archivo descargado no tiene el "
                "formato esperado. "
                f"Faltan: {sorted(faltantes)}"
            )

        # Solo reemplazar el destino cuando la
        # descarga haya terminado correctamente.
        temporal.replace(destino)

    except (
        HTTPError,
        URLError,
        OSError,
        ValueError,
    ) as error:

        logger.error(
            f"Error al descargar MiBici: {error}"
        )

        raise

    finally:

        if temporal.exists():
            temporal.unlink()

    logger.success(
        f"Descarga completada: {destino.name} "
        f"({destino.stat().st_size:,} bytes)"
    )

    return destino


def descargar_archivos_mibici(
    input_dir: Path,
) -> list[Path]:
    """
    Descarga automaticamente los archivos oficiales
    de enero y febrero de 2025.

    Si ya existen, evita descargarlos nuevamente.
    """

    input_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    archivos = []

    logger.info(
        "Verificando archivos oficiales de MiBici..."
    )

    for nombre_archivo, url in URLS_MIBICI.items():

        destino = input_dir / nombre_archivo

        archivo = descargar_datos(
            destino=destino,
            url=url,
        )

        archivos.append(archivo)

    logger.success(
        "Verificacion de descargas completada."
    )

    return archivos


# ============================================================
# LIMPIEZA Y TRANSFORMACION
# ============================================================

def procesar_archivo(ruta_csv: Path) -> dict:
    """
    Procesa un CSV de MiBici y genera un archivo Parquet.

    Aplica limpieza, validaciones y transformaciones.
    """

    logger.info(
        f"Leyendo archivo: {ruta_csv.name}"
    )

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

    # Normalizar nombres de columnas.
    df.columns = [
        normalizar_columna(columna)
        for columna in df.columns
    ]

    faltantes = COLUMNAS_REQUERIDAS - set(df.columns)

    if faltantes:
        raise ValueError(
            "Columnas obligatorias faltantes: "
            f"{sorted(faltantes)}"
        )

    # Eliminar viajes duplicados.
    df = df.drop_duplicates(
        subset=["viaje_id"]
    )

    # Convertir fechas.
    for columna in [
        "inicio_del_viaje",
        "fin_del_viaje",
    ]:

        df[columna] = pd.to_datetime(
            df[columna],
            errors="coerce",
        )

    # Convertir identificadores de estaciones.
    for columna in [
        "origen_id",
        "destino_id",
    ]:

        df[columna] = pd.to_numeric(
            df[columna],
            errors="coerce",
        ).astype("Int64")

    # Eliminar registros con datos esenciales faltantes.
    df = df.dropna(
        subset=[
            "inicio_del_viaje",
            "fin_del_viaje",
            "origen_id",
            "destino_id",
        ]
    ).copy()

    # Calcular duracion del viaje en minutos.
    df["duracion_minutos"] = (
        df["fin_del_viaje"]
        - df["inicio_del_viaje"]
    ).dt.total_seconds() / 60

    # Excluir duraciones invalidas.
    df = df[
        df["duracion_minutos"] > 0
    ].copy()

    # Variables temporales.
    df["fecha"] = (
        df["inicio_del_viaje"].dt.date
    )

    df["hora"] = (
        df["inicio_del_viaje"].dt.hour
    )

    df["dia_semana"] = (
        df["inicio_del_viaje"].dt.dayofweek
    )

    df["mes"] = (
        df["inicio_del_viaje"].dt.month
    )

    # Excluir identificador de usuario y
    # variables demograficas.
    # Se conserva viaje_id para trazabilidad.
    df = df[COLUMNAS_SALIDA].copy()

    # Crear directorio de salida.
    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    ruta_parquet = (
        PROCESSED_DATA_DIR
        / f"{ruta_csv.stem}_procesado.parquet"
    )

    # Exportar a Parquet comprimido.
    df.to_parquet(
        ruta_parquet,
        engine="pyarrow",
        index=False,
        compression="snappy",
    )

    # Construir registro de trazabilidad.
    resultado = {
        "archivo_origen": ruta_csv.name,
        "sha256_origen": calcular_sha256(
            ruta_csv
        ),
        "archivo_procesado": ruta_parquet.name,
        "sha256_procesado": calcular_sha256(
            ruta_parquet
        ),
        "registros_originales": registros_originales,
        "registros_procesados": len(df),
        "registros_descartados": (
            registros_originales - len(df)
        ),
        "columnas_procesadas": COLUMNAS_SALIDA,
    }

    # Registrar la URL oficial de ambos meses.
    if ruta_csv.name in URLS_MIBICI:

        resultado["url_origen"] = (
            URLS_MIBICI[ruta_csv.name]
        )

    logger.success(
        f"{ruta_csv.name}: "
        f"{len(df):,} registros procesados"
    )

    return resultado


# ============================================================
# PIPELINE PRINCIPAL
# ============================================================

@app.command()
def main(
    input_dir: Path = RAW_DATA_DIR,
    output_manifest: Path = (
        PROCESSED_DATA_DIR / "manifest.json"
    ),
    descargar: bool = True,
):
    """
    Pipeline MiBici Guadalajara.

    1. Descarga enero y febrero de 2025 si faltan.
    2. Procesa los CSV disponibles.
    3. Genera archivos Parquet.
    4. Calcula hashes SHA-256.
    5. Genera un manifiesto JSON.
    """

    logger.info(
        "Iniciando pipeline MiBici - Equipo 17"
    )

    input_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Descargar automaticamente ambos meses.
    if descargar:

        descargar_archivos_mibici(
            input_dir=input_dir
        )

    # Detectar archivos CSV disponibles.
    archivos = sorted(
        input_dir.glob("*.csv")
    )

    if not archivos:

        logger.error(
            "No se encontraron archivos CSV "
            f"en {input_dir}"
        )

        raise typer.Exit(code=1)

    logger.info(
        f"Archivos CSV encontrados: {len(archivos)}"
    )

    resultados = []

    # Procesar cada archivo.
    for archivo in archivos:

        resultado = procesar_archivo(
            ruta_csv=archivo
        )

        resultados.append(resultado)

    # Crear manifiesto de trazabilidad.
    manifiesto = {
        "proyecto": "TC5061-Proyecto-Equipo17",
        "dataset": "MiBici Guadalajara",
        "fuente": (
            "https://www.mibici.net/es/datos-abiertos/"
        ),
        "fecha_procesamiento_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
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

    logger.success(
        "Pipeline MiBici completado."
    )

    logger.info(
        "Resumen del procesamiento:"
    )

    for resultado in resultados:

        logger.info(
            f"  {resultado['archivo_origen']}: "
            f"{resultado['registros_procesados']:,} "
            "viajes"
        )

    logger.info(
        "Total de viajes procesados: "
        f"{manifiesto['total_registros']:,}"
    )


if __name__ == "__main__":
    app()


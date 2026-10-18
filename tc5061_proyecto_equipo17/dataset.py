
import csv
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
    EXTERNAL_DATA_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)


# ============================================================
# CONFIGURACION GENERAL
# ============================================================

app = typer.Typer(
    help="Pipeline de datos MiBici Guadalajara - Equipo 17"
)

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

NOMBRE_NOMENCLATURA = "nomenclatura_2026_09.csv"

URL_NOMENCLATURA = (
    "https://www.mibici.net/site/assets/files/1118/"
    "nomenclatura_2026_09.csv"
)

# Compatibilidad con el codigo anterior.
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

COLUMNAS_ID_ESTACION = {
    "id",
    "id_estacion",
    "estacion_id",
    "station_id",
    "codigo",
    "codigo_estacion",
}

COLUMNAS_CAPACIDAD = {
    "capacidad",
    "capacidad_estacion",
    "capacidad_total",
    "capacity",
    "dpcapacity",
    "total_anclajes",
    "anclajes",
    "dock_count",
}


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


def detectar_formato_csv(ruta: Path):
    """
    Detecta la codificacion y el separador del CSV.

    Retorna:
        encoding, separador, columnas_normalizadas
    """

    with ruta.open("rb") as archivo:
        contenido = archivo.read(65536)

    if not contenido:
        raise ValueError(
            f"Archivo CSV vacio: {ruta}"
        )

    texto = None
    encoding_usado = None

    for encoding in ("utf-8-sig", "cp1252"):
        try:
            texto = contenido.decode(encoding)
            encoding_usado = encoding
            break
        except UnicodeDecodeError:
            continue

    if texto is None:
        raise ValueError(
            f"No se pudo identificar la codificacion: {ruta.name}"
        )

    encabezado = texto.splitlines()[0]

    if encabezado.lstrip().lower().startswith(
        ("<html", "<!doctype")
    ):
        raise ValueError(
            f"El archivo parece HTML, no CSV: {ruta.name}"
        )

    separador = max(
        (",", ";", "\t"),
        key=lambda sep: encabezado.count(sep),
    )

    columnas = next(
        csv.reader(
            [encabezado],
            delimiter=separador,
        )
    )

    columnas_normalizadas = [
        normalizar_columna(columna)
        for columna in columnas
    ]

    if len(columnas_normalizadas) < 2:
        raise ValueError(
            f"Encabezado CSV invalido: {ruta.name}"
        )

    return (
        encoding_usado,
        separador,
        columnas_normalizadas,
    )


# ============================================================
# DESCARGA DE ARCHIVOS
# ============================================================

def descargar_archivo(
    destino: Path,
    url: str,
) -> Path:
    """
    Descarga un archivo oficial de MiBici.

    Evita descargas repetidas y utiliza un archivo temporal
    para no conservar descargas incompletas.
    """

    destino.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if destino.is_file():
        if destino.stat().st_size == 0:
            raise ValueError(
                f"El archivo existente esta vacio: {destino}"
            )

        logger.info(
            f"Archivo existente, se omite descarga: {destino.name}"
        )

        return destino

    temporal = destino.with_name(
        destino.name + ".part"
    )

    logger.info(
        f"Descargando: {url}"
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
                f"La descarga esta vacia: {destino.name}"
            )

        temporal.replace(destino)

    except (
        HTTPError,
        URLError,
        OSError,
        ValueError,
    ) as error:

        logger.error(
            f"Error descargando {destino.name}: {error}"
        )
        raise

    finally:
        if temporal.exists():
            temporal.unlink()

    logger.success(
        f"Descarga completada: {destino.name}"
    )

    return destino


def descargar_datos(
    destino: Path,
    url: str = URL_MIBICI,
) -> Path:
    """
    Descarga y valida un archivo CSV de viajes.
    """

    existia = destino.is_file()

    ruta = descargar_archivo(
        destino=destino,
        url=url,
    )

    try:
        _, _, columnas = detectar_formato_csv(ruta)

        faltantes = (
            COLUMNAS_REQUERIDAS - set(columnas)
        )

        if faltantes:
            raise ValueError(
                f"Columnas faltantes en {ruta.name}: "
                f"{sorted(faltantes)}"
            )

    except Exception:
        if not existia and ruta.exists():
            ruta.unlink()
        raise

    return ruta


def descargar_archivos_mibici(
    input_dir: Path,
) -> list[Path]:
    """
    Descarga los CSV oficiales de enero y febrero de 2025.
    """

    input_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    archivos = []

    for nombre_archivo, url in URLS_MIBICI.items():

        destino = input_dir / nombre_archivo

        ruta = descargar_datos(
            destino=destino,
            url=url,
        )

        archivos.append(ruta)

    logger.success(
        "Archivos de viajes verificados correctamente."
    )

    return archivos


# ============================================================
# NOMENCLATURA DE ESTACIONES
# ============================================================

def validar_nomenclatura(
    ruta: Path,
) -> dict:
    """
    Valida la nomenclatura oficial de MiBici.

    Comprueba que existan identificadores de estacion
    y una columna de capacidad.

    El archivo oficial de septiembre de 2026 utiliza:
        id, name, latitude, longitude, dpcapacity
    """

    encoding, separador, columnas = (
        detectar_formato_csv(ruta)
    )

    tiene_id = bool(
        set(columnas) & COLUMNAS_ID_ESTACION
    )

    tiene_capacidad = bool(
        set(columnas) & COLUMNAS_CAPACIDAD
    )

    if not tiene_id or not tiene_capacidad:
        raise ValueError(
            "La nomenclatura no contiene columnas "
            "reconocidas de identificador y capacidad. "
            f"Columnas encontradas: {columnas}"
        )

    df = pd.read_csv(
        ruta,
        encoding=encoding,
        sep=separador,
        low_memory=False,
    )

    if df.empty:
        raise ValueError(
            "La nomenclatura no contiene estaciones."
        )

    df.columns = [
        normalizar_columna(columna)
        for columna in df.columns
    ]

    columna_id = next(
        columna
        for columna in df.columns
        if columna in COLUMNAS_ID_ESTACION
    )

    columna_capacidad = next(
        columna
        for columna in df.columns
        if columna in COLUMNAS_CAPACIDAD
    )

    ids = pd.to_numeric(
        df[columna_id],
        errors="coerce",
    )

    capacidades = pd.to_numeric(
        df[columna_capacidad],
        errors="coerce",
    )

    if ids.isna().all():
        raise ValueError(
            "No se encontraron identificadores validos."
        )

    if capacidades.isna().all():
        raise ValueError(
            "No se encontraron capacidades validas."
        )

    logger.success(
        f"Nomenclatura validada: {len(df):,} estaciones"
    )

    return {
        "archivo_origen": ruta.name,
        "url_origen": URL_NOMENCLATURA,
        "sha256_origen": calcular_sha256(ruta),
        "registros": len(df),
        "columna_id": columna_id,
        "columna_capacidad": columna_capacidad,
        "columnas": list(df.columns),
        "codificacion": encoding,
        "separador": separador,
    }


def descargar_nomenclatura() -> Path:
    """
    Descarga automaticamente la nomenclatura de estaciones.

    Guarda el archivo en:
        data/external/nomenclatura_2026_09.csv
    """

    EXTERNAL_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    destino = (
        EXTERNAL_DATA_DIR
        / NOMBRE_NOMENCLATURA
    )

    existia = destino.is_file()

    ruta = descargar_archivo(
        destino=destino,
        url=URL_NOMENCLATURA,
    )

    try:
        validar_nomenclatura(ruta)

    except Exception:
        if not existia and ruta.exists():
            ruta.unlink()
        raise

    return ruta


# ============================================================
# LIMPIEZA Y TRANSFORMACION DE VIAJES
# ============================================================

def procesar_archivo(
    ruta_csv: Path,
) -> dict:
    """
    Procesa un CSV de viajes y genera un archivo Parquet.

    Realiza:
    - Normalizacion de columnas.
    - Eliminacion de duplicados.
    - Conversion de fechas e identificadores.
    - Limpieza de registros invalidos.
    - Generacion de variables temporales.
    - Exportacion a Parquet.
    - Registro de hashes SHA-256.
    """

    logger.info(
        f"Leyendo archivo: {ruta_csv.name}"
    )

    encoding, separador, _ = (
        detectar_formato_csv(ruta_csv)
    )

    df = pd.read_csv(
        ruta_csv,
        encoding=encoding,
        sep=separador,
        low_memory=False,
        dtype={
            "Viaje_Id": "string",
            "Usuario_Id": "string",
            "Origen_Id": "string",
            "Destino_Id": "string",
        },
    )

    registros_originales = len(df)

    df.columns = [
        normalizar_columna(columna)
        for columna in df.columns
    ]

    faltantes = (
        COLUMNAS_REQUERIDAS - set(df.columns)
    )

    if faltantes:
        raise ValueError(
            "Columnas obligatorias faltantes: "
            f"{sorted(faltantes)}"
        )

    # Eliminar duplicados.
    df = df.drop_duplicates(
        subset=["viaje_id"]
    )

    # Convertir fechas.
    for columna in (
        "inicio_del_viaje",
        "fin_del_viaje",
    ):

        df[columna] = pd.to_datetime(
            df[columna],
            errors="coerce",
        )

    # Convertir identificadores de estaciones.
    for columna in (
        "origen_id",
        "destino_id",
    ):

        df[columna] = pd.to_numeric(
            df[columna],
            errors="coerce",
        ).astype("Int64")

    # Eliminar registros incompletos.
    df = df.dropna(
        subset=[
            "inicio_del_viaje",
            "fin_del_viaje",
            "origen_id",
            "destino_id",
        ]
    ).copy()

    # Calcular duracion del viaje.
    df["duracion_minutos"] = (
        df["fin_del_viaje"]
        - df["inicio_del_viaje"]
    ).dt.total_seconds() / 60

    # Eliminar duraciones invalidas.
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

    # Excluir identificadores personales
    # y variables demograficas.
    df = df[COLUMNAS_SALIDA].copy()

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    ruta_parquet = (
        PROCESSED_DATA_DIR
        / f"{ruta_csv.stem}_procesado.parquet"
    )

    df.to_parquet(
        ruta_parquet,
        engine="pyarrow",
        index=False,
        compression="snappy",
    )

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
    Pipeline de datos MiBici - Equipo 17.

    1. Descarga enero y febrero de 2025.
    2. Descarga la nomenclatura de estaciones.
    3. Valida los archivos CSV.
    4. Limpia y transforma los viajes.
    5. Genera archivos Parquet.
    6. Calcula hashes SHA-256.
    7. Genera manifest.json.
    """

    logger.info(
        "Iniciando pipeline MiBici - Equipo 17"
    )

    input_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Descargar archivos oficiales.
    if descargar:

        descargar_archivos_mibici(
            input_dir=input_dir
        )

        ruta_nomenclatura = (
            descargar_nomenclatura()
        )

    else:

        ruta_nomenclatura = (
            EXTERNAL_DATA_DIR
            / NOMBRE_NOMENCLATURA
        )

        if not ruta_nomenclatura.is_file():
            raise FileNotFoundError(
                "No se encontro la nomenclatura "
                "y la descarga esta deshabilitada."
            )

    # Validar nomenclatura.
    info_nomenclatura = (
        validar_nomenclatura(
            ruta_nomenclatura
        )
    )

    # Procesar exclusivamente los meses seleccionados.
    archivos = sorted(
        input_dir / nombre
        for nombre in URLS_MIBICI
        if (input_dir / nombre).is_file()
    )

    if len(archivos) != len(URLS_MIBICI):

        faltantes = [
            nombre
            for nombre in URLS_MIBICI
            if not (input_dir / nombre).is_file()
        ]

        logger.error(
            f"Archivos de viajes faltantes: {faltantes}"
        )

        raise typer.Exit(code=1)

    resultados = []

    for archivo in archivos:

        resultado = procesar_archivo(
            ruta_csv=archivo
        )

        resultados.append(resultado)

    # Generar manifiesto de versionado.
    manifiesto = {
        "proyecto": "TC5061-Proyecto-Equipo17",
        "dataset": "MiBici Guadalajara",
        "fuente": (
            "https://www.mibici.net/es/datos-abiertos/"
        ),
        "fecha_procesamiento_utc": (
            datetime.now(timezone.utc).isoformat()
        ),
        "total_archivos": len(resultados),
        "total_registros": sum(
            item["registros_procesados"]
            for item in resultados
        ),
        "archivos": resultados,
        "nomenclatura": info_nomenclatura,
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
            f"{resultado['registros_procesados']:,} viajes"
        )

    logger.info(
        "Total de viajes procesados: "
        f"{manifiesto['total_registros']:,}"
    )

    logger.info(
        "Registros en nomenclatura: "
        f"{info_nomenclatura['registros']:,}"
    )


if __name__ == "__main__":
    app()


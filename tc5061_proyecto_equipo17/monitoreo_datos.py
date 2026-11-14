"""Actividad 10: conjuntos de referencia y actual para el monitoreo de drift (P2).

Define dos periodos de datos abiertos de MiBici y deja listas las tablas de variables
del modelo P2 que recibirán Evidently (drift) y el A/B testing:

- Referencia: enero de 2025, el periodo con el que se entrena el modelo.
- Actual: julio de 2025, un periodo posterior y de otra temporada. Se divide en una
  ventana de reentrenamiento (1-21 de julio) y una de evaluación (22-31 de julio) para
  que las dos versiones del modelo se comparen sobre datos que ninguna vio al entrenar.

Los datos del periodo actual se guardan en subcarpetas de ``data/processed`` para no
mezclarse con los de entrenamiento: ``train_p2`` lee ``data/processed/*.parquet``.

Uso (desde la raíz del repositorio)::

    python -m tc5061_proyecto_equipo17.monitoreo_datos ingesta-actual
    python -m tc5061_proyecto_equipo17.monitoreo_datos construir-features
"""

from datetime import datetime, timezone
import json
from pathlib import Path

from loguru import logger
import pandas as pd
import typer

from tc5061_proyecto_equipo17.config import (
    ACTUAL_CORTE_EVAL,
    ACTUAL_DATA_DIR,
    ACTUAL_FIN,
    ACTUAL_INICIO,
    EXTERNAL_DATA_DIR,
    MONITOREO_DATA_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    REF_FIN,
    REF_INICIO,
)
from tc5061_proyecto_equipo17.dataset import (
    NOMBRE_NOMENCLATURA,
    calcular_sha256,
    descargar_datos,
    procesar_archivo,
)
from tc5061_proyecto_equipo17.features import (
    P2_FEATURES,
    P2_OBJETIVO,
    p2_construir_dataset,
)

app = typer.Typer(help="Datos de referencia y actual para el monitoreo de drift (Actividad 10).")

# Periodo actual: julio de 2025 (URL publicada en la página oficial de datos abiertos).
NOMBRE_CSV_ACTUAL = "datos_abiertos_2025_07.csv"
URL_CSV_ACTUAL = "https://mibici.net/site/assets/files/108898/datos_abiertos_2025_07.csv"
# Periodo de referencia: el parquet de enero que ya genera el pipeline de ingesta.
PARQUET_REFERENCIA = "datos_abiertos_2025_01_procesado.parquet"
PARQUET_ACTUAL = "datos_abiertos_2025_07_procesado.parquet"

ARCHIVO_REFERENCIA = "p2_referencia.parquet"
ARCHIVO_ACTUAL = "p2_actual.parquet"
ARCHIVO_MANIFIESTO = "manifest_monitoreo.json"

VENTANA_REENTRENAMIENTO = "reentrenamiento"
VENTANA_EVALUACION = "evaluacion"


def asignar_ventana(horas: pd.Series, corte: str = ACTUAL_CORTE_EVAL) -> pd.Series:
    """Marca cada hora como ventana de reentrenamiento o de evaluación.

    Las horas anteriores a ``corte`` son de reentrenamiento y las demás de evaluación.
    La última hora de reentrenamiento queda sin ventana (``NaN``): su etiqueta (riesgo
    de la hora siguiente) se calcula con la primera hora de evaluación y usarla para
    entrenar filtraría información de esa ventana.
    """
    corte_ts = pd.Timestamp(corte)
    ventana = pd.Series(VENTANA_EVALUACION, index=horas.index, dtype="object")
    ventana[horas < corte_ts] = VENTANA_REENTRENAMIENTO
    ventana[horas == corte_ts - pd.Timedelta(hours=1)] = None
    return ventana


def construir_conjunto(
    viajes: pd.DataFrame, estaciones: pd.DataFrame, inicio: str, fin: str
) -> pd.DataFrame:
    """Construye la tabla de variables de P2 (y su etiqueta) para un periodo.

    Reutiliza ``p2_construir_dataset``, que es lo que usa el entrenamiento; así el
    monitoreo compara exactamente las mismas variables que ve el modelo.
    """
    conjunto = p2_construir_dataset(viajes, estaciones, inicio, fin).reset_index()
    columnas = ["estacion_id", "hora_ts", *P2_FEATURES, P2_OBJETIVO]
    return conjunto[columnas]


def resumen_conjunto(conjunto: pd.DataFrame) -> dict:
    """Cifras para documentar un conjunto: filas, estaciones, fechas y tasa de riesgo."""
    return {
        "filas": int(len(conjunto)),
        "estaciones": int(conjunto["estacion_id"].nunique()),
        "hora_min": conjunto["hora_ts"].min().isoformat(),
        "hora_max": conjunto["hora_ts"].max().isoformat(),
        "tasa_riesgo": round(float(conjunto[P2_OBJETIVO].mean()), 4),
        "nulos_en_variables": int(conjunto[P2_FEATURES].isna().sum().sum()),
    }


def ingesta_actual(
    directorio_raw: Path | None = None,
    directorio_salida: Path | None = None,
    descargar: bool = True,
) -> dict:
    """Descarga el CSV del periodo actual y lo procesa a Parquet en ``data/processed/actual``."""
    directorio_raw = directorio_raw or RAW_DATA_DIR
    directorio_salida = directorio_salida or ACTUAL_DATA_DIR
    directorio_raw.mkdir(parents=True, exist_ok=True)

    ruta_csv = directorio_raw / NOMBRE_CSV_ACTUAL
    if descargar:
        descargar_datos(destino=ruta_csv, url=URL_CSV_ACTUAL)
    elif not ruta_csv.is_file():
        raise FileNotFoundError(f"No se encontró {ruta_csv} y la descarga está deshabilitada.")

    resultado = procesar_archivo(ruta_csv, directorio_salida=directorio_salida)
    resultado["url_origen"] = URL_CSV_ACTUAL
    resultado["periodo"] = {"inicio": ACTUAL_INICIO, "fin": ACTUAL_FIN}
    return resultado


def construir_features(
    carpeta_referencia: Path | None = None,
    carpeta_actual: Path | None = None,
    carpeta_salida: Path | None = None,
    ruta_nomenclatura: Path | None = None,
) -> dict:
    """Genera ``p2_referencia.parquet`` y ``p2_actual.parquet`` y su manifiesto."""
    carpeta_referencia = carpeta_referencia or PROCESSED_DATA_DIR
    carpeta_actual = carpeta_actual or ACTUAL_DATA_DIR
    carpeta_salida = carpeta_salida or MONITOREO_DATA_DIR
    ruta_nomenclatura = ruta_nomenclatura or (EXTERNAL_DATA_DIR / NOMBRE_NOMENCLATURA)

    ruta_ref = carpeta_referencia / PARQUET_REFERENCIA
    ruta_act = carpeta_actual / PARQUET_ACTUAL
    for ruta in (ruta_ref, ruta_act, ruta_nomenclatura):
        if not ruta.is_file():
            raise FileNotFoundError(f"Falta {ruta}. Ejecuta antes la ingesta de datos.")

    estaciones = pd.read_csv(ruta_nomenclatura)

    logger.info("Construyendo variables de P2 del periodo de referencia")
    referencia = construir_conjunto(pd.read_parquet(ruta_ref), estaciones, REF_INICIO, REF_FIN)

    logger.info("Construyendo variables de P2 del periodo actual")
    actual = construir_conjunto(pd.read_parquet(ruta_act), estaciones, ACTUAL_INICIO, ACTUAL_FIN)
    actual["ventana"] = asignar_ventana(actual["hora_ts"])

    carpeta_salida.mkdir(parents=True, exist_ok=True)
    ruta_salida_ref = carpeta_salida / ARCHIVO_REFERENCIA
    ruta_salida_act = carpeta_salida / ARCHIVO_ACTUAL
    referencia.to_parquet(ruta_salida_ref, index=False)
    actual.to_parquet(ruta_salida_act, index=False)

    manifiesto = {
        "descripcion": "Conjuntos de referencia y actual para monitoreo de drift (P2)",
        "fecha_generacion_utc": datetime.now(timezone.utc).isoformat(),
        "variables": P2_FEATURES,
        "objetivo": P2_OBJETIVO,
        "referencia": {
            "periodo": {"inicio": REF_INICIO, "fin": REF_FIN},
            "archivo": ARCHIVO_REFERENCIA,
            "sha256": calcular_sha256(ruta_salida_ref),
            **resumen_conjunto(referencia),
        },
        "actual": {
            "periodo": {"inicio": ACTUAL_INICIO, "fin": ACTUAL_FIN},
            "archivo": ARCHIVO_ACTUAL,
            "sha256": calcular_sha256(ruta_salida_act),
            "corte_evaluacion": ACTUAL_CORTE_EVAL,
            "filas_por_ventana": {
                str(k): int(v)
                for k, v in actual["ventana"].value_counts(dropna=False).items()
            },
            **resumen_conjunto(actual),
        },
    }
    (carpeta_salida / ARCHIVO_MANIFIESTO).write_text(
        json.dumps(manifiesto, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifiesto


@app.command("ingesta-actual")
def ingesta_actual_cmd(descargar: bool = True):
    """Descarga y procesa el CSV de julio de 2025 (periodo actual)."""
    resultado = ingesta_actual(descargar=descargar)
    logger.success(
        f"{resultado['archivo_origen']}: {resultado['registros_procesados']:,} viajes procesados"
    )


@app.command("construir-features")
def construir_features_cmd():
    """Genera las tablas de variables de P2 para referencia y actual."""
    manifiesto = construir_features()
    for nombre in ("referencia", "actual"):
        datos = manifiesto[nombre]
        logger.success(
            f"{nombre}: {datos['filas']:,} filas, {datos['estaciones']} estaciones, "
            f"tasa de riesgo {datos['tasa_riesgo']:.1%}"
        )


if __name__ == "__main__":
    app()

"""Reporte de drift con Evidently (P2, referencia enero vs actual julio).

Compara el conjunto de referencia (enero 2025) con el conjunto actual (julio 2025)
para las variables del modelo P2 y exporta, para cada modo de comparacion:

- Modo "magnitud": Wasserstein (normed) para numericas y Jensen-Shannon para
  categoricas, umbral 0.1. Mide *cuanto* cambian las distribuciones.
- Modo "significancia": K-S para numericas y chi-cuadrada para categoricas,
  umbral de p-valor 0.05. Mide si el cambio es *estadisticamente detectable*.

Artefactos por modo:
- Reporte HTML interactivo de Evidently.
- CSV con el drift (o p-valor) por variable.
- Grafica PNG de las variables con mayor drift (para el informe en PDF).
- Snapshot completo en JSON y un resumen en Markdown.

Uso (desde la raiz del repositorio)::

    python -m tc5061_proyecto_equipo17.monitoreo_drift reporte                 # ambos
    python -m tc5061_proyecto_equipo17.monitoreo_drift reporte --modo magnitud
    python -m tc5061_proyecto_equipo17.monitoreo_drift reporte --modo significancia
"""

from __future__ import annotations

from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")

from evidently import DataDefinition, Dataset, Report
from evidently.metrics import ValueDrift
from evidently.presets import DataDriftPreset, DataSummaryPreset
from evidently.presets.dataset_stats import ValueStatsTests
from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer

from tc5061_proyecto_equipo17.config import FIGURES_DIR, MONITOREO_DATA_DIR, REPORTS_DIR
from tc5061_proyecto_equipo17.features import P2_FEATURES, P2_OBJETIVO

ARCHIVO_REFERENCIA = "p2_referencia.parquet"
ARCHIVO_ACTUAL = "p2_actual.parquet"
DIRECTORIO_REPORTE = REPORTS_DIR / "drift"

MODO_MAGNITUD = "magnitud"
MODO_SIGNIFICANCIA = "significancia"

# Configuracion por modo: metodo, umbral y nombres de salida.
MODOS: dict[str, dict] = {
    MODO_MAGNITUD: {
        "num_method": None,        # default Evidently: Wasserstein distance (normed)
        "cat_method": None,        # default Evidently: Jensen-Shannon distance
        "objetivo_method": None,   # default Evidently para la etiqueta
        "threshold": 0.1,
        "archivos": {
            "html": "reporte_drift.html",
            "csv": "drift_por_variable.csv",
            "json": "drift_snapshot.json",
            "resumen": "resumen_drift.md",
            "png": "drift_por_variable.png",
        },
    },
    MODO_SIGNIFICANCIA: {
        "num_method": "ks",         # K-S p-value (numericas)
        "cat_method": "chisquare",  # chi-cuadrada p-value (categoricas)
        "objetivo_method": "chisquare",
        "threshold": 0.05,
        "archivos": {
            "html": "reporte_drift_significancia.html",
            "csv": "drift_significancia.csv",
            "json": "drift_snapshot_significancia.json",
            "resumen": "resumen_drift_significancia.md",
            "png": "drift_significancia.png",
        },
    },
}

# Variables discretas que Evidently debe tratar como categoricas.
VARIABLES_CATEGORICAS = ["hora_dia", "dia_semana", "fin_de_semana", "riesgo_ahora", P2_OBJETIVO]
VARIABLES_NUMERICAS = [c for c in P2_FEATURES if c not in VARIABLES_CATEGORICAS]

# Descriptivos numericos que se desactivan en el resumen: con ~273k filas y
# estacionalidad, las tolerancias relativas por defecto (+/-10%) no aplican
# (min/max son estadisticos de orden y la media de neto_ult_3h es ~0).
# Nota: DataSummaryPreset(include_tests=False) lanza KeyError en Evidently
# 0.7.x; la forma correcta de suprimir solo estas pruebas es column_tests.
TESTS_DESCRIPTIVOS_VACIOS = ValueStatsTests(
    min_tests=[],
    max_tests=[],
    mean_tests=[],
    std_tests=[],
    q25_tests=[],
    q50_tests=[],
    q75_tests=[],
)

# El metric_name de Evidently cambia el orden de sus argumentos segun como
# resuelva los metodos: en modo magnitud emite (column, threshold, method) y en
# significancia (column, method, threshold). Se captura el contenido completo y
# se parsea sin depender del orden.
PATRON_VALUE_DRIFT = re.compile(r"ValueDrift\((?P<args>.+)\)")
PATRON_TEST_COLUMNA = re.compile(r"Value Drift for column (?P<col>.+)")

app = typer.Typer(help="Reporte de drift con Evidently (Actividad 10).")


def _cargar_conjuntos(carpeta: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lee los Parquet de referencia y actual con las columnas del modelo P2."""
    ruta_ref = carpeta / ARCHIVO_REFERENCIA
    ruta_act = carpeta / ARCHIVO_ACTUAL
    for ruta in (ruta_ref, ruta_act):
        if not ruta.is_file():
            raise FileNotFoundError(
                f"Falta {ruta}. Ejecuta 'monitoreo_datos ingesta-actual' y "
                "'monitoreo_datos construir-features'."
            )
    columnas = [*P2_FEATURES, P2_OBJETIVO]
    referencia = pd.read_parquet(ruta_ref)[columnas]
    actual = pd.read_parquet(ruta_act)[columnas]
    return referencia, actual


def _definicion_datos() -> DataDefinition:
    """Fija el tipo de cada columna para que el drift sea reproducible."""
    return DataDefinition(
        numerical_columns=VARIABLES_NUMERICAS,
        categorical_columns=VARIABLES_CATEGORICAS,
    )


def _estado_a_texto(estado) -> str:
    """Normaliza el estado de una prueba de Evidently a 'FAIL'/'SUCCESS'/..."""
    return str(getattr(estado, "value", estado)).split(".")[-1].upper()


def _construir_informe(modo: str) -> Report:
    """Arma el Report de Evidently segun el modo de comparacion."""
    config = MODOS[modo]
    if config["objetivo_method"]:
        objetivo = ValueDrift(
            column=P2_OBJETIVO,
            method=config["objetivo_method"],
            threshold=config["threshold"],
        )
    else:
        objetivo = ValueDrift(column=P2_OBJETIVO)
    return Report(
        metrics=[
            # El resumen conserva sus visualizaciones; solo se desactivan las
            # pruebas descriptivas (min/max/media/cuartiles) de las numericas.
            DataSummaryPreset(
                column_tests={
                    columna: TESTS_DESCRIPTIVOS_VACIOS for columna in VARIABLES_NUMERICAS
                }
            ),
            DataDriftPreset(
                columns=P2_FEATURES,
                num_method=config["num_method"],
                cat_method=config["cat_method"],
                threshold=config["threshold"],
            ),
            objetivo,
        ],
        include_tests=True,
    )


def _extraer_drift(snapshot, modo: str) -> tuple[pd.DataFrame, dict]:
    """Extrae el drift por variable y el resumen a nivel de conjunto."""
    resultado = snapshot.dict()
    metricas = resultado.get("metrics", [])
    pruebas = resultado.get("tests", [])

    filas: dict[str, dict] = {}
    for metrica in metricas:
        coincidencia = PATRON_VALUE_DRIFT.match(metrica.get("metric_name", ""))
        if not coincidencia:
            continue
        # Los valores de los argumentos no contienen comas (p. ej.
        # "Wasserstein distance (normed)", "Jensen-Shannon distance", "ks"),
        # por lo que se pueden separar de forma segura.
        argumentos = dict(
            parte.split("=", 1) for parte in coincidencia.group("args").split(",")
        )
        columna = argumentos["column"]
        filas[columna] = {
            "variable": columna,
            "metodo": argumentos["method"],
            "drift_score": float(metrica.get("value")),
            "umbral": float(argumentos["threshold"]),
        }

    detectado: dict[str, bool] = {}
    for prueba in pruebas:
        # El nombre es "Value Drift for column <col>"; la descripcion solo trae el
        # score. Se revisan ambos por robustez.
        for texto in (str(prueba.get("name") or ""), str(prueba.get("description") or "")):
            coincidencia = PATRON_TEST_COLUMNA.search(texto)
            if coincidencia:
                detectado[coincidencia.group("col")] = (
                    _estado_a_texto(prueba.get("status")) == "FAIL"
                )
                break

    for columna, fila in filas.items():
        if columna in detectado:
            fila["drift_detectado"] = detectado[columna]
        elif modo == MODO_SIGNIFICANCIA:
            # Con un p-valor hay drift cuando el valor es MENOR que el umbral.
            fila["drift_detectado"] = fila["drift_score"] < fila["umbral"]
        else:
            fila["drift_detectado"] = fila["drift_score"] >= fila["umbral"]

    tabla = pd.DataFrame(filas.values())
    if not tabla.empty:
        tabla["es_objetivo"] = tabla["variable"] == P2_OBJETIVO
        tabla["unidad"] = "p-valor" if modo == MODO_SIGNIFICANCIA else "distancia"
        tabla = tabla.sort_values("drift_score", ascending=False).reset_index(drop=True)

    resumen = {"columnas_con_drift": None, "proporcion_con_drift": None, "drift_conjunto": None}
    for metrica in metricas:
        if metrica.get("metric_name", "").startswith("DriftedColumnsCount"):
            valor = metrica.get("value") or {}
            resumen["columnas_con_drift"] = int(valor.get("count", 0))
            resumen["proporcion_con_drift"] = float(valor.get("share", 0.0))
    if resumen["proporcion_con_drift"] is not None:
        resumen["drift_conjunto"] = resumen["proporcion_con_drift"] >= 0.5
    return tabla, resumen


def _grafica(tabla: pd.DataFrame, ruta: Path, modo: str) -> None:
    """Genera la grafica de barras del drift (o significancia) por variable."""
    if tabla.empty:
        return
    datos = tabla.copy()
    if modo == MODO_SIGNIFICANCIA:
        # Los p-valores son diminutos: se grafican como -log10(p) para que sean
        # legibles. -log10(0.05) ~ 1.30 marca el umbral.
        datos["valor"] = -np.log10(datos["drift_score"].clip(lower=1e-300))
        umbral = -np.log10(float(datos["umbral"].iloc[0]))
        xlabel = "-log10(p-valor)  (drift si supera el umbral, p < 0.05)"
        titulo = "Significancia del drift: referencia (enero) vs actual (julio)"
    else:
        datos["valor"] = datos["drift_score"]
        umbral = float(datos["umbral"].iloc[0])
        xlabel = "Drift score (distancia normalizada)"
        titulo = "Drift por variable: referencia (enero) vs actual (julio)"

    datos = datos.sort_values("valor")
    colores = ["#c0392b" if d else "#95a5a6" for d in datos["drift_detectado"]]

    fig, ax = plt.subplots(figsize=(9, 0.5 * len(datos) + 1.5))
    ax.barh(datos["variable"], datos["valor"], color=colores)
    ax.axvline(umbral, color="black", linestyle="--", linewidth=1, label=f"Umbral ({umbral:.4g})")
    ax.set_xlabel(xlabel)
    ax.set_title(titulo)
    ax.legend(loc="lower right")
    fig.tight_layout()
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def _escribir_resumen(tabla: pd.DataFrame, resumen: dict, ruta: Path, modo: str) -> None:
    """Redacta un resumen en Markdown a partir de los resultados."""
    con_drift = tabla[tabla["drift_detectado"]].sort_values("drift_score", ascending=False)
    if modo == MODO_SIGNIFICANCIA:
        titulo = "# Resumen de significancia del drift (K-S / chi-cuadrada, p < 0.05)"
        columna = "p-valor"
        nota = (
            "Con ~273k filas, un p-valor < 0.05 indica una diferencia detectable "
            "estadisticamente, no necesariamente una diferencia de magnitud relevante. "
            "Comparar con reports/drift/drift_por_variable.csv (modo magnitud)."
        )
    else:
        titulo = "# Resumen del drift (referencia = enero 2025, actual = julio 2025)"
        columna = "Drift score"
        nota = (
            "Distancias normalizadas (Wasserstein / Jensen-Shannon); drift si el "
            "valor >= umbral 0.1."
        )

    lineas = [
        titulo,
        "",
        nota,
        "",
        f"- Variables comparadas: {len(tabla)}",
        f"- Variables con drift: {len(con_drift)}",
        f"- Proporcion de columnas con drift: {resumen['proporcion_con_drift']:.1%}",
        f"- Drift a nivel de conjunto: {'si' if resumen['drift_conjunto'] else 'no'}",
        "",
        "## Variables con drift (ordenadas por magnitud del score)",
        "",
        f"| Variable | Metodo | {columna} | Umbral |",
        "|---|---|---:|---:|",
    ]
    for _, fila in con_drift.iterrows():
        lineas.append(
            f"| {fila['variable']} | {fila['metodo']} | "
            f"{fila['drift_score']:.4f} | {fila['umbral']} |"
        )
    lineas += ["", "## Variables sin drift", "", f"| Variable | {columna} |", "|---|---:|"]
    for _, fila in tabla[~tabla["drift_detectado"]].iterrows():
        lineas.append(f"| {fila['variable']} | {fila['drift_score']:.4f} |")
    ruta.write_text("\n".join(lineas) + "\n", encoding="utf-8")


def _generar(modo: str, carpeta: Path) -> None:
    """Genera todos los artefactos de un modo de comparacion."""
    archivos = MODOS[modo]["archivos"]
    referencia, actual = _cargar_conjuntos(carpeta)
    logger.info(f"[{modo}] Referencia: {len(referencia):,} filas | Actual: {len(actual):,} filas")

    definicion = _definicion_datos()
    ds_referencia = Dataset.from_pandas(referencia, data_definition=definicion)
    ds_actual = Dataset.from_pandas(actual, data_definition=definicion)

    snapshot = _construir_informe(modo).run(
        current_data=ds_actual, reference_data=ds_referencia
    )

    DIRECTORIO_REPORTE.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(DIRECTORIO_REPORTE / archivos["html"]))
    snapshot.save_json(str(DIRECTORIO_REPORTE / archivos["json"]))

    tabla, resumen = _extraer_drift(snapshot, modo)
    tabla.to_csv(DIRECTORIO_REPORTE / archivos["csv"], index=False)
    _grafica(tabla, FIGURES_DIR / archivos["png"], modo)
    _escribir_resumen(tabla, resumen, DIRECTORIO_REPORTE / archivos["resumen"], modo)

    logger.success(f"[{modo}] HTML: {DIRECTORIO_REPORTE / archivos['html']}")
    logger.success(f"[{modo}] CSV: {DIRECTORIO_REPORTE / archivos['csv']}")
    logger.success(f"[{modo}] PNG: {FIGURES_DIR / archivos['png']}")
    if resumen["proporcion_con_drift"] is not None:
        logger.info(
            f"[{modo}] Columnas con drift: {resumen['columnas_con_drift']} "
            f"({resumen['proporcion_con_drift']:.1%})"
        )


@app.callback()
def _main() -> None:
    """Reporte de drift de datos del modelo P2 (Actividad 10)."""


@app.command("reporte")
def reporte(carpeta: Path | None = None, modo: str = "ambos") -> None:
    """Genera el reporte de drift (modo magnitud y/o significancia)."""
    carpeta = carpeta or MONITOREO_DATA_DIR
    if modo == "ambos":
        modos = [MODO_MAGNITUD, MODO_SIGNIFICANCIA]
    elif modo in MODOS:
        modos = [modo]
    else:
        raise typer.BadParameter("--modo debe ser 'magnitud', 'significancia' o 'ambos'.")
    for nombre in modos:
        _generar(nombre, carpeta)


if __name__ == "__main__":
    app()
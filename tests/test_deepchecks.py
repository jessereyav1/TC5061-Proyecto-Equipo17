"""Validación de datos de P3 con DeepChecks.

Revisa la matriz de variables por estación (la entrada del clustering): tipos, nulos,
duplicados, columnas constantes y drift entre entrenamiento (enero) y prueba (febrero).
Las pruebas con datos sintéticos corren en cualquier computadora; la prueba con los datos
reales de MiBici se omite si los parquet no están en data/processed/ (no se suben al repo).
"""

import warnings

import numpy as np

# DeepChecks 0.19 usa np.Inf, que NumPy 2 eliminó. Este alias lo mantiene funcionando.
if not hasattr(np, "Inf"):
    np.Inf = np.inf

import pandas as pd
import pytest

pytest.importorskip("deepchecks")
warnings.filterwarnings("ignore", category=UserWarning, module="deepchecks")
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import DataDuplicates, FeatureDrift, IsSingleValue, PercentOfNulls

from tc5061_proyecto_equipo17.config import PROCESSED_DATA_DIR
from tc5061_proyecto_equipo17.features import matriz_clustering_estaciones

UMBRAL_DRIFT_ESTABLE = 0.3  # entre meses con la misma estructura
UMBRAL_DRIFT_ALTO = 0.5  # drift que sí queremos detectar


def _dataset(matriz: pd.DataFrame) -> Dataset:
    return Dataset(matriz.reset_index(drop=True), cat_features=[])


def _max_drift(entrenamiento: pd.DataFrame, prueba: pd.DataFrame) -> float:
    resultado = FeatureDrift().run(_dataset(entrenamiento), _dataset(prueba)).value
    return max(valor["Drift score"] for valor in resultado.values())


@pytest.fixture
def matrices(viajes_enero_febrero):
    enero, febrero = viajes_enero_febrero
    entrenamiento = matriz_clustering_estaciones(enero)
    prueba = matriz_clustering_estaciones(febrero).reindex(
        columns=entrenamiento.columns, fill_value=0.0
    )
    return entrenamiento, prueba


# ---- integridad: tipos, nulos, duplicados, constantes ----


def test_todas_las_variables_son_numericas(matrices):
    entrenamiento, _ = matrices
    assert all(pd.api.types.is_float_dtype(t) for t in entrenamiento.dtypes)


def test_no_hay_valores_faltantes(matrices):
    entrenamiento, prueba = matrices
    for matriz in (entrenamiento, prueba):
        nulos = PercentOfNulls().run(_dataset(matriz)).value
        assert (nulos["Percent of nulls in sample"] == 0).all()


def test_no_hay_filas_duplicadas(matrices):
    entrenamiento, _ = matrices
    assert DataDuplicates().run(_dataset(entrenamiento)).value == 0.0


def test_ninguna_variable_es_constante(matrices):
    entrenamiento, _ = matrices
    # El valor de cada columna es su número de valores distintos: debe ser más de uno
    valores_distintos = IsSingleValue().run(_dataset(entrenamiento)).value
    assert all(n > 1 for n in valores_distintos.values())


# ---- drift entre entrenamiento y prueba ----


def test_sin_drift_entre_meses_con_la_misma_estructura(matrices):
    entrenamiento, prueba = matrices
    assert _max_drift(entrenamiento, prueba) < UMBRAL_DRIFT_ESTABLE


# ---- los chequeos detectan problemas reales (casos límite) ----


def test_detecta_nulos_inyectados(matrices):
    entrenamiento, _ = matrices
    con_nulos = entrenamiento.copy()
    con_nulos.iloc[0, 0] = np.nan
    nulos = PercentOfNulls().run(_dataset(con_nulos)).value
    assert nulos["Percent of nulls in sample"].max() > 0


def test_detecta_duplicados_inyectados(matrices):
    entrenamiento, _ = matrices
    duplicado = pd.concat([entrenamiento, entrenamiento.iloc[:5]])
    assert DataDuplicates().run(_dataset(duplicado)).value > 0


def test_detecta_drift_inyectado(matrices):
    entrenamiento, prueba = matrices
    # Febrero con todas las variables desplazadas: el patrón de uso cambió por completo
    assert _max_drift(entrenamiento, prueba + 0.3) > UMBRAL_DRIFT_ALTO


# ---- datos reales (se omite si no están descargados) ----

ENERO = PROCESSED_DATA_DIR / "datos_abiertos_2025_01_procesado.parquet"
FEBRERO = PROCESSED_DATA_DIR / "datos_abiertos_2025_02_procesado.parquet"


@pytest.mark.skipif(not (ENERO.exists() and FEBRERO.exists()), reason="datos reales no disponibles")
def test_datos_reales_enero_febrero():
    entrenamiento = matriz_clustering_estaciones(pd.read_parquet(ENERO))
    prueba = matriz_clustering_estaciones(pd.read_parquet(FEBRERO)).reindex(
        columns=entrenamiento.columns, fill_value=0.0
    )

    nulos = PercentOfNulls().run(_dataset(entrenamiento)).value
    assert (nulos["Percent of nulls in sample"] == 0).all()
    assert entrenamiento.shape[1] == 21

    drift = FeatureDrift().run(_dataset(entrenamiento), _dataset(prueba)).value
    puntajes = pd.Series({col: v["Drift score"] for col, v in drift.items()})
    print("\nDrift enero-febrero (mayor a menor):\n", puntajes.sort_values(ascending=False).head(5))
    assert puntajes.between(0, 1).all()

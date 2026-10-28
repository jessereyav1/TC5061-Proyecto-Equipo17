"""Pruebas de reproducibilidad del entrenamiento de P3.

Con la misma semilla, dos entrenamientos deben dar las mismas etiquetas y las mismas
métricas. Se usan las funciones reales de `train_p3` sobre datos sintéticos.
"""

import numpy as np
import pandas as pd
import pytest
import typer

from tc5061_proyecto_equipo17.features import matriz_clustering_estaciones
from tc5061_proyecto_equipo17.modeling.train_p3 import _ajustar_y_evaluar, _construir_modelo


def _entrenar(algoritmo, k, semilla, X, X_test):
    modelo, _, _, _ = _construir_modelo(algoritmo, k, semilla)
    etiquetas, metricas = _ajustar_y_evaluar(modelo, X, X_test)
    return np.asarray(etiquetas), metricas


@pytest.fixture
def matrices(viajes_enero_febrero):
    enero, febrero = viajes_enero_febrero
    X = matriz_clustering_estaciones(enero)
    X_test = matriz_clustering_estaciones(febrero).reindex(columns=X.columns, fill_value=0.0)
    return X, X_test


@pytest.mark.parametrize("algoritmo", ["kmeans", "kmeans-escalado", "gmm"])
def test_misma_semilla_mismas_etiquetas_y_metricas(algoritmo, matrices):
    X, X_test = matrices
    etiquetas_1, metricas_1 = _entrenar(algoritmo, 3, 42, X, X_test)
    etiquetas_2, metricas_2 = _entrenar(algoritmo, 3, 42, X, X_test)

    assert np.array_equal(etiquetas_1, etiquetas_2)
    assert metricas_1.keys() == metricas_2.keys()
    for nombre in metricas_1:
        assert metricas_1[nombre] == pytest.approx(metricas_2[nombre]), nombre


def test_metricas_estan_en_rangos_validos(matrices):
    X, X_test = matrices
    _, metricas = _entrenar("kmeans", 3, 42, X, X_test)
    assert -1.0 <= metricas["train_silhouette"] <= 1.0
    assert -1.0 <= metricas["test_silhouette"] <= 1.0
    assert metricas["train_davies_bouldin"] >= 0.0
    assert -1.0 <= metricas["test_ari_ene_feb"] <= 1.0
    assert metricas["train_n_estaciones"] == len(X)


def test_con_datos_estructurados_kmeans_recupera_los_tres_tipos(matrices):
    # Los datos sintéticos tienen 3 tipos de estación: con k=3 la silhouette debe ser alta
    X, X_test = matrices
    _, metricas = _entrenar("kmeans", 3, 42, X, X_test)
    assert metricas["train_silhouette"] > 0.5


def test_algoritmo_desconocido_lanza_error():
    with pytest.raises(typer.BadParameter):
        _construir_modelo("dbscan", 3, 42)


def test_k_mayor_al_numero_de_estaciones_falla():
    # Caso límite: no se puede pedir más grupos que muestras
    X = pd.DataFrame(np.random.default_rng(0).random((3, 4)))
    modelo, _, _, _ = _construir_modelo("kmeans", 10, 42)
    with pytest.raises(ValueError):
        modelo.fit(X)

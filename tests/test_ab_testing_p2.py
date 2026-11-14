"""Pruebas unitarias del A/B Testing del Equipo 17."""

import numpy as np
import pytest

from scripts.ab_testing_p2 import calcular_metricas


def test_predicciones_perfectas():
    y = np.array([0, 1, 0, 1])
    pred = np.array([0, 1, 0, 1])
    proba = np.array([0.1, 0.9, 0.2, 0.8])

    m = calcular_metricas(y, pred, proba)

    assert m["precision"] == pytest.approx(1.0)
    assert m["recall"] == pytest.approx(1.0)
    assert m["f1"] == pytest.approx(1.0)
    assert m["pr_auc"] == pytest.approx(1.0)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 0, 0, 2)


def test_matriz_confusion_con_errores():
    y = np.array([1, 1, 1, 0, 0, 0])
    pred = np.array([1, 1, 0, 1, 0, 0])
    proba = np.array([0.9, 0.8, 0.2, 0.7, 0.1, 0.3])

    m = calcular_metricas(y, pred, proba)

    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 1, 1, 2)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3)


def test_sin_predicciones_positivas():
    y = np.array([0, 1, 0, 1])
    pred = np.array([0, 0, 0, 0])
    proba = np.array([0.1, 0.2, 0.1, 0.3])

    m = calcular_metricas(y, pred, proba)

    assert m["tp"] == 0
    assert m["fp"] == 0
    assert m["fn"] == 2
    assert m["tn"] == 2
    assert m["precision"] == 0
    assert m["recall"] == 0
    assert m["f1"] == 0


def test_pr_auc_utiliza_probabilidades():
    y = np.array([0, 1, 0, 1])
    pred = np.array([0, 0, 0, 0])

    proba_buena = np.array([0.1, 0.4, 0.2, 0.3])
    proba_mala = np.array([0.4, 0.1, 0.3, 0.2])

    buena = calcular_metricas(y, pred, proba_buena)
    mala = calcular_metricas(y, pred, proba_mala)

    assert buena["pr_auc"] > mala["pr_auc"]
    assert buena["recall"] == mala["recall"] == 0

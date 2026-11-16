"""Pruebas de la separación de datos de la versión alternativa de P2 (Actividad 10)."""

import pandas as pd

from tc5061_proyecto_equipo17.features import P2_FEATURES, P2_OBJETIVO
from tc5061_proyecto_equipo17.modeling.train_p2_v2 import conjuntos_v2
from tc5061_proyecto_equipo17.monitoreo_datos import (
    VENTANA_EVALUACION,
    VENTANA_REENTRENAMIENTO,
    asignar_ventana,
)


def _tabla(horas: list[str]) -> pd.DataFrame:
    """Tabla mínima con las columnas de P2 para una estación y las horas indicadas."""
    horas_ts = pd.to_datetime(horas)
    tabla = pd.DataFrame({"estacion_id": 1, "hora_ts": horas_ts})
    for columna in P2_FEATURES:
        tabla[columna] = 0
    tabla[P2_OBJETIVO] = [i % 2 for i in range(len(horas_ts))]
    return tabla


def _conjuntos():
    referencia = _tabla(["2025-01-01 00:00", "2025-01-31 23:00"])
    actual = _tabla(
        ["2025-07-01 00:00", "2025-07-21 22:00", "2025-07-21 23:00", "2025-07-22 00:00", "2025-07-31 23:00"]
    )
    actual["ventana"] = asignar_ventana(actual["hora_ts"])
    return referencia, actual


def test_train_incluye_enero_y_la_ventana_de_reentrenamiento():
    referencia, actual = _conjuntos()
    train, _ = conjuntos_v2(referencia, actual)
    esperadas = set(referencia["hora_ts"]) | set(
        actual.loc[actual["ventana"] == VENTANA_REENTRENAMIENTO, "hora_ts"]
    )
    assert set(train["hora_ts"]) == esperadas
    assert len(train) == 4


def test_evaluacion_es_solo_la_ventana_del_ab_testing():
    referencia, actual = _conjuntos()
    _, evaluacion = conjuntos_v2(referencia, actual)
    assert set(evaluacion["hora_ts"]) == set(
        actual.loc[actual["ventana"] == VENTANA_EVALUACION, "hora_ts"]
    )
    assert (evaluacion["hora_ts"] >= pd.Timestamp("2025-07-22")).all()


def test_no_hay_fuga_entre_entrenamiento_y_evaluacion():
    """Ninguna hora de evaluación ni la hora excluida (21 jul 23:00) entra al entrenamiento."""
    referencia, actual = _conjuntos()
    train, evaluacion = conjuntos_v2(referencia, actual)
    assert set(train["hora_ts"]).isdisjoint(evaluacion["hora_ts"])
    excluida = pd.Timestamp("2025-07-21 23:00")
    assert excluida not in set(train["hora_ts"]) and excluida not in set(evaluacion["hora_ts"])


def test_conjuntos_tienen_las_columnas_del_modelo():
    referencia, actual = _conjuntos()
    train, evaluacion = conjuntos_v2(referencia, actual)
    for conjunto in (train, evaluacion):
        assert [c for c in P2_FEATURES if c not in conjunto.columns] == []
        assert P2_OBJETIVO in conjunto.columns
        assert "ventana" not in conjunto.columns

"""Pruebas del módulo de inferencia P2."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from tc5061_proyecto_equipo17.model_serving import (
    FEATURES,
    predict_p2,
)


@pytest.fixture
def datos_validos():
    return pd.DataFrame([{
        "hora_dia": 12,
        "dia_semana": 2,
        "fin_de_semana": 0,
        "capacidad": 20,
        "pct_ocupacion": 0.5,
        "riesgo_ahora": 0,
        "neto": -2,
        "neto_ult_3h": -4.0,
    }])


def test_variables_requeridas():
    assert len(FEATURES) == 8
    assert len(set(FEATURES)) == 8


def test_rechaza_variable_faltante(datos_validos):
    datos = datos_validos.drop(columns=["capacidad"])

    with pytest.raises(ValueError, match="capacidad"):
        predict_p2(datos)


def test_prediccion_con_modelo_simulado(datos_validos):
    class ModeloSimulado:
        def predict(self, datos):
            assert list(datos.columns) == list(FEATURES)
            return np.array([0], dtype=np.int64)

    with patch(
        "tc5061_proyecto_equipo17.model_serving.load_p2_model",
        return_value=ModeloSimulado(),
    ):
        resultado = predict_p2(datos_validos)

    assert resultado.tolist() == [0]
    assert resultado.dtype == np.int64

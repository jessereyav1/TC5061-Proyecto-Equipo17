"""Pruebas de la prueba de carga (Actividad 11): entradas, docker stats y resúmenes."""

import json

from app import ModelInput
import numpy as np
import pandas as pd
import pytest

from tc5061_proyecto_equipo17 import carga
from tc5061_proyecto_equipo17.features import P2_FEATURES

ETAPAS = [(60, 10, 5), (120, 25, 5)]


def test_payloads_versionados_pasan_la_validacion_de_la_api():
    """Las entradas de Locust deben ser aceptadas por el esquema de app.py (si no, serían 422)."""
    payloads = json.loads(carga.ARCHIVO_PAYLOADS.read_text(encoding="utf-8"))
    assert len(payloads) >= 100
    for payload in payloads:
        assert set(payload) == set(P2_FEATURES)
        ModelInput(**payload)


def test_seleccionar_payloads_descarta_filas_que_la_api_rechazaria():
    actual = pd.DataFrame(
        {
            "hora_dia": [8, 9, 10, 11],
            "dia_semana": [1, 1, 1, 1],
            "fin_de_semana": [0, 0, 0, 0],
            "capacidad": [10, 10, 10, 10],
            "pct_ocupacion": [0.5, 1.4, 0.2, 0.3],  # 1.4: fuera de rango
            "riesgo_ahora": [0, 1, 0, 0],
            "neto": [2, 1, 15, -3],  # 15: mayor que la capacidad
            "neto_ult_3h": [1.0, 2.0, 3.0, -4.0],
            "ventana": ["evaluacion"] * 3 + ["reentrenamiento"],
        }
    )
    payloads = carga.seleccionar_payloads(actual, n=10)
    assert len(payloads) == 1
    assert payloads[0]["hora_dia"] == 8
    assert isinstance(payloads[0]["neto"], int)
    ModelInput(**payloads[0])


@pytest.mark.parametrize(
    ("texto", "esperado"),
    [("153.2MiB", 153.2), ("1GiB", 1024.0), ("512KiB", 0.5), ("2.5GB", 2560.0)],
)
def test_leer_memoria_mb(texto, esperado):
    assert carga.leer_memoria_mb(texto) == pytest.approx(esperado)


def test_leer_porcentaje():
    assert carga.leer_porcentaje("87.45%") == pytest.approx(87.45)
    assert np.isnan(carga.leer_porcentaje("--"))


def test_leer_docker_stats():
    lineas = [
        json.dumps({"Name": "mibici-p2-api", "CPUPerc": "95.10%", "MemUsage": "210.5MiB / 1GiB"}),
        "",
    ]
    filas = carga.leer_docker_stats(lineas, momento=100.0)
    assert filas == [
        {"timestamp": 100.0, "contenedor": "mibici-p2-api", "cpu_pct": 95.1,
         "mem_mb": 210.5, "mem_limite_mb": 1024.0}
    ]


def test_asignar_etapa():
    segundos = pd.Series([0, 59.9, 60, 119, 130])
    assert carga.asignar_etapa(segundos, ETAPAS).tolist() == [0, 0, 1, 1, -1]


def _peticiones():
    """Escalón 1: 5 pred/s a 20 ms; escalón 2: 10 pred/s a 300 ms con algunas fallas."""
    filas = []
    for segundo in range(120):
        lentas = segundo >= 60
        for i in range(10 if lentas else 5):
            exito = not (lentas and i == 0 and segundo % 10 == 0)
            filas.append((1000 + segundo + i / 20, carga.PREDICT, 300.0 if lentas else 20.0,
                          exito, 25 if lentas else 10))
        filas.append((1000 + segundo + 0.9, "GET /", 5.0, True, 10))  # no es inferencia
    return pd.DataFrame(filas, columns=["timestamp", "nombre", "latencia_ms", "exito", "usuarios"])


def _recursos():
    return pd.DataFrame(
        [(1000 + s, contenedor, cpu, mem, 1024.0)
         for s in range(0, 120, 2)
         for contenedor, cpu, mem in ((carga.CONTENEDOR_API, 40.0 if s < 60 else 98.0, 200.0),
                                      (carga.CONTENEDOR_MLFLOW, 1.0, 150.0))],
        columns=["timestamp", "contenedor", "cpu_pct", "mem_mb", "mem_limite_mb"],
    )


def test_resumen_por_etapa_detecta_la_saturacion():
    etapas = carga.resumen_por_etapa(_peticiones(), _recursos(), ETAPAS)
    assert etapas["usuarios"].tolist() == [10, 25]
    primera, segunda = etapas.iloc[0], etapas.iloc[1]
    assert primera["rps_exitosas"] == pytest.approx(5.0)
    assert primera["latencia_p95_ms"] == pytest.approx(20.0)
    assert primera["cumple_slo"]
    assert primera["api_cpu_pct"] == pytest.approx(40.0)
    assert segunda["latencia_p95_ms"] == pytest.approx(300.0)
    assert not segunda["cumple_slo"]
    assert segunda["predicciones_fallidas"] == 5  # segundos 70, 80, ..., 110


def test_resumen_global_usa_los_nombres_de_finops_y_solo_cuenta_predict():
    peticiones, recursos = _peticiones(), _recursos()
    etapas = carga.resumen_por_etapa(peticiones, recursos, ETAPAS)
    resumen = carga.resumen_global(peticiones, recursos, etapas)

    plantilla = json.loads((carga.PROJ_ROOT / "reports" / "finops" / "parametros.json").read_text(encoding="utf-8")) \
        if (carga.PROJ_ROOT / "reports" / "finops" / "parametros.json").is_file() else None
    campos_carga = {"duracion_segundos", "peticiones_exitosas", "peticiones_fallidas",
                    "rps_promedio", "latencia_p95_ms"}
    campos_recursos = {"api_cpu_promedio_pct", "api_ram_promedio_mb",
                       "mlflow_cpu_promedio_pct", "mlflow_ram_promedio_mb"}
    assert campos_carga <= set(resumen["prueba_carga"])
    assert campos_recursos == set(resumen["recursos"])
    if plantilla:
        assert campos_carga <= set(plantilla["prueba_carga"])

    predict = peticiones[peticiones["nombre"] == carga.PREDICT]
    assert resumen["prueba_carga"]["peticiones_exitosas"] == int(predict["exito"].sum())
    assert resumen["prueba_carga"]["peticiones_fallidas"] == 6
    assert resumen["capacidad_sostenible"]["usuarios"] == 10
    assert resumen["recursos"]["mlflow_ram_promedio_mb"] == pytest.approx(150.0)


def test_serie_temporal_agrupa_en_ventanas():
    serie = carga.serie_temporal(_peticiones(), ventana_s=5)
    assert serie["segundo"].iloc[0] == 0
    assert serie["rps_exitosas"].iloc[0] == pytest.approx(5.0)
    assert serie["p95_ms"].iloc[-1] == pytest.approx(300.0)

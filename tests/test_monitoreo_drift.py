"""Pruebas de la extracción de drift del reporte de Evidently (Actividad 10)."""

from tc5061_proyecto_equipo17 import monitoreo_drift as drift
from tc5061_proyecto_equipo17.features import P2_OBJETIVO


class _SnapshotFalso:
    """Sustituto mínimo de un Snapshot: solo necesita dict()."""

    def __init__(self, metrics, tests=None):
        self._datos = {"metrics": metrics, "tests": tests or []}

    def dict(self):
        return self._datos


def _metrica(column, threshold, method, value):
    return {
        "metric_name": f"ValueDrift(column={column},threshold={threshold},method={method})",
        "value": value,
    }


def test_parsea_magnitud_sin_depender_del_orden_de_los_argumentos():
    snapshot = _SnapshotFalso(
        [
            _metrica("dia_semana", 0.1, "Jensen-Shannon distance", 0.0416),
            _metrica("pct_ocupacion", 0.1, "Wasserstein distance (normed)", 0.0331),
        ]
    )
    tabla, _ = drift._extraer_drift(snapshot, drift.MODO_MAGNITUD)
    assert list(tabla["variable"]) == ["dia_semana", "pct_ocupacion"]  # ordenado desc
    fila = tabla.set_index("variable").loc["pct_ocupacion"]
    assert fila["metodo"] == "Wasserstein distance (normed)"
    assert fila["drift_score"] == 0.0331
    assert fila["umbral"] == 0.1
    assert fila["unidad"] == "distancia"
    assert not fila["drift_detectado"]


def test_significancia_marca_drift_con_p_valor_bajo():
    snapshot = _SnapshotFalso(
        [
            _metrica("neto", 0.05, "ks", 0.1024),
            _metrica("pct_ocupacion", 0.05, "ks", 9.6e-25),
        ]
    )
    tabla, _ = drift._extraer_drift(snapshot, drift.MODO_SIGNIFICANCIA)
    por_variable = tabla.set_index("variable")
    assert por_variable.loc["pct_ocupacion", "drift_detectado"]
    assert not por_variable.loc["neto", "drift_detectado"]
    assert (tabla["unidad"] == "p-valor").all()


def test_marca_la_etiqueta():
    snapshot = _SnapshotFalso([_metrica(P2_OBJETIVO, 0.1, "Jensen-Shannon distance", 0.01)])
    tabla, _ = drift._extraer_drift(snapshot, drift.MODO_MAGNITUD)
    assert bool(tabla.loc[0, "es_objetivo"]) is True


def test_usa_el_estado_de_las_pruebas_cuando_existe():
    snapshot = _SnapshotFalso(
        metrics=[_metrica("dia_semana", 0.1, "Jensen-Shannon distance", 0.0416)],
        tests=[{"name": "Value Drift for column dia_semana", "status": "FAIL"}],
    )
    tabla, _ = drift._extraer_drift(snapshot, drift.MODO_MAGNITUD)
    assert bool(tabla.loc[0, "drift_detectado"]) is True


def test_resumen_de_conjunto_justo_en_el_umbral():
    snapshot = _SnapshotFalso(
        [
            {
                "metric_name": "DriftedColumnsCount(columns=a,b,c,d,drift_share=0.5)",
                "value": {"count": 4.0, "share": 0.5},
            }
        ]
    )
    _, resumen = drift._extraer_drift(snapshot, drift.MODO_SIGNIFICANCIA)
    assert resumen["columnas_con_drift"] == 4
    assert resumen["proporcion_con_drift"] == 0.5
    assert resumen["drift_conjunto"] is True  # >= 0.5, justo en el umbral


def test_ignora_metricas_que_no_son_value_drift():
    snapshot = _SnapshotFalso(
        [
            {"metric_name": "RowCount()", "value": 272681.0},
            _metrica("capacidad", 0.1, "Wasserstein distance (normed)", 0.0019),
        ]
    )
    tabla, _ = drift._extraer_drift(snapshot, drift.MODO_MAGNITUD)
    assert list(tabla["variable"]) == ["capacidad"]


def test_estado_a_texto_normaliza():
    assert drift._estado_a_texto("FAIL") == "FAIL"
    assert drift._estado_a_texto("SUCCESS") == "SUCCESS"

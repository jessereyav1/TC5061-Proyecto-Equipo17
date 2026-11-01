"""Pruebas unitarias de las funciones de features de P2 (riesgo de la siguiente hora).

Usan pocas filas sintéticas con el resultado calculado a mano, así que no dependen de
los datos reales de MiBici (no se suben al repositorio).
"""

import pandas as pd
import pytest

from tc5061_proyecto_equipo17.features import (
    P2_FEATURES,
    P2_OBJETIVO,
    p2_etiqueta_riesgo,
    p2_features,
    p2_flujo_horario,
)

DIA = "2025-01-01"


def _viajes(filas):
    """Viajes con origen, destino, inicio y fin, igual que los datos procesados."""
    viajes = pd.DataFrame(
        filas, columns=["origen_id", "destino_id",
                        "inicio_del_viaje", "fin_del_viaje"]
    )
    viajes["inicio_del_viaje"] = pd.to_datetime(viajes["inicio_del_viaje"])
    viajes["fin_del_viaje"] = pd.to_datetime(viajes["fin_del_viaje"])
    return viajes


def _flujo(pct_ocupacion, estacion=1, inicio="2025-01-03 22:00", neto=None):
    """Flujo de una estación por hora, como lo recibe p2_etiqueta_riesgo."""
    horas = pd.date_range(inicio, periods=len(pct_ocupacion), freq="h")
    indice = pd.MultiIndex.from_product(
        [[estacion], horas], names=["estacion_id", "hora_ts"]
    )
    return pd.DataFrame(
        {
            "capacidad": 20,
            "pct_ocupacion": pct_ocupacion,
            "neto": neto if neto is not None else [0] * len(pct_ocupacion),
        },
        index=indice,
    )


# ---- p2_flujo_horario ----


def test_flujo_salida_por_hora_de_inicio_y_llegada_por_hora_de_fin():
    # El viaje cruza de hora: sale en la hora 8 y llega en la hora 9
    viajes = _viajes([(1, 2, "2025-01-01 08:50", "2025-01-01 09:10")])
    flujo = p2_flujo_horario(viajes, DIA, DIA)
    assert flujo.loc[(1, pd.Timestamp("2025-01-01 08:00")), "salidas"] == 1
    assert flujo.loc[(2, pd.Timestamp("2025-01-01 09:00")), "llegadas"] == 1
    assert flujo.loc[(1, pd.Timestamp("2025-01-01 08:00")), "neto"] == -1
    assert flujo.loc[(2, pd.Timestamp("2025-01-01 09:00")), "neto"] == 1


def test_flujo_rejilla_completa_rellena_con_cero():
    viajes = _viajes([(1, 2, "2025-01-01 08:10", "2025-01-01 08:20")])
    flujo = p2_flujo_horario(viajes, DIA, DIA)
    assert len(flujo) == 2 * 24  # 2 estaciones por 24 horas
    assert flujo.loc[(1, pd.Timestamp("2025-01-01 15:00")), "salidas"] == 0
    assert flujo[["salidas", "llegadas", "neto"]].isna().sum().sum() == 0


def test_flujo_queda_ordenado_aunque_la_entrada_no_lo_este():
    # Etiqueta y ventana móvil dependen de este orden para no mirar al futuro
    viajes = _viajes(
        [
            (2, 1, "2025-01-01 15:00", "2025-01-01 15:20"),
            (1, 2, "2025-01-01 08:00", "2025-01-01 08:20"),
        ]
    )
    assert p2_flujo_horario(viajes, DIA, DIA).index.is_monotonic_increasing


def test_flujo_viaje_circular_en_la_misma_hora_tiene_neto_cero():
    viajes = _viajes([(1, 1, "2025-01-01 08:10", "2025-01-01 08:40")])
    fila = p2_flujo_horario(viajes, DIA, DIA).loc[(
        1, pd.Timestamp("2025-01-01 08:00"))]
    assert (fila["salidas"], fila["llegadas"], fila["neto"]) == (1, 1, 0)


def test_flujo_llegada_despues_de_fecha_fin_no_se_cuenta():
    # Caso límite: el viaje termina al día siguiente, fuera del periodo pedido
    viajes = _viajes([(1, 2, "2025-01-01 23:50", "2025-01-02 00:10")])
    flujo = p2_flujo_horario(viajes, DIA, DIA)
    assert flujo["salidas"].sum() == 1
    assert flujo["llegadas"].sum() == 0
    assert flujo.index.get_level_values("hora_ts").max() == pd.Timestamp(
        "2025-01-01 23:00"
    )


def test_flujo_sin_viajes_devuelve_tabla_vacia():
    flujo = p2_flujo_horario(_viajes([]), DIA, DIA)
    assert flujo.empty
    assert list(flujo.columns) == ["salidas", "llegadas", "neto"]


def test_flujo_viaje_sin_fin_cuenta_salida_pero_no_llegada():
    # Documenta el comportamiento actual: la llegada se pierde sin aviso
    viajes = _viajes([(1, 2, "2025-01-01 08:10", None)])
    flujo = p2_flujo_horario(viajes, DIA, DIA)
    assert flujo["salidas"].sum() == 1
    assert flujo["llegadas"].sum() == 0


# ---- p2_etiqueta_riesgo ----

@pytest.mark.parametrize(
    "pct, riesgo_esperado",
    [
        (0.09, 1),
        (0.10, 0),
        (0.50, 0),
        (0.90, 0),
        (0.91, 1),
    ],
)
def test_etiqueta_umbrales_exactos(pct, riesgo_esperado):
    resultado = p2_etiqueta_riesgo(_flujo([pct]))
    assert resultado["riesgo_ahora"].iloc[0] == riesgo_esperado


def test_etiqueta_es_el_riesgo_de_la_hora_siguiente():
    resultado = p2_etiqueta_riesgo(_flujo([0.5, 0.05, 0.5]))
    assert resultado[P2_OBJETIVO].iloc[:2].tolist() == [1, 0]


def test_etiqueta_ultima_hora_queda_sin_valor():
    resultado = p2_etiqueta_riesgo(_flujo([0.05, 0.5]))
    assert pd.isna(resultado[P2_OBJETIVO].iloc[-1])


def test_etiqueta_no_se_pasa_de_una_estacion_a_otra():
    # La estación 2 está en riesgo, pero la última hora de la 1 no debe heredarlo
    flujo = pd.concat([_flujo([0.5, 0.5], estacion=1),
                      _flujo([0.05, 0.5], estacion=2)])
    resultado = p2_etiqueta_riesgo(flujo)
    assert pd.isna(resultado.loc[1, P2_OBJETIVO].iloc[-1])
    assert resultado.loc[2, "riesgo_ahora"].iloc[0] == 1


def test_etiqueta_respeta_umbrales_personalizados():
    flujo = _flujo([0.15, 0.5, 0.85])
    resultado = p2_etiqueta_riesgo(flujo, umbral_bajo=0.20, umbral_alto=0.80)
    assert resultado["riesgo_ahora"].tolist() == [1, 0, 1]


def test_etiqueta_no_modifica_la_tabla_de_entrada():
    flujo = _flujo([0.05, 0.5])
    original = flujo.copy()
    p2_etiqueta_riesgo(flujo)
    pd.testing.assert_frame_equal(flujo, original)


# ---- p2_features ----


def test_features_calendario_viernes_sabado_y_domingo():
    # 2025-01-03 fue viernes; las horas van de viernes 23:00 a domingo 00:00
    flujo = _flujo([0.5] * 26, inicio="2025-01-03 23:00")
    resultado = p2_features(flujo).droplevel("estacion_id")

    viernes = resultado.loc[pd.Timestamp("2025-01-03 23:00")]
    assert (viernes["hora_dia"], viernes["dia_semana"], viernes["fin_de_semana"]) == (
        23,
        4,
        0,
    )
    sabado = resultado.loc[pd.Timestamp("2025-01-04 00:00")]
    assert (sabado["hora_dia"], sabado["dia_semana"], sabado["fin_de_semana"]) == (
        0,
        5,
        1,
    )
    domingo = resultado.loc[pd.Timestamp("2025-01-05 00:00")]
    assert (domingo["dia_semana"], domingo["fin_de_semana"]) == (6, 1)


def test_features_neto_ult_3h_suma_la_hora_actual_y_dos_anteriores():
    flujo = _flujo([0.5] * 4, neto=[1, 2, 3, 4])
    assert p2_features(flujo)["neto_ult_3h"].tolist()[
        2:] == [6, 9]  # 1+2+3 y 2+3+4


def test_features_primera_hora_usa_solo_esa_hora():
    # Caso límite: aún no hay dos horas anteriores y no debe salir NaN
    flujo = _flujo([0.5, 0.5], neto=[5, 1])
    assert p2_features(flujo)["neto_ult_3h"].iloc[0] == 5


def test_features_ventana_no_mezcla_estaciones():
    flujo = pd.concat(
        [
            _flujo([0.5, 0.5], estacion=1, neto=[10, 10]),
            _flujo([0.5, 0.5], estacion=2, neto=[1, 1]),
        ]
    )
    assert p2_features(flujo).loc[2, "neto_ult_3h"].tolist() == [1, 2]


def test_features_sin_fuga_temporal():
    # Cambiar la última hora no debe alterar ninguna variable de las horas anteriores
    base = _flujo([0.5] * 4, neto=[1, 2, 3, 4])
    futuro_distinto = _flujo([0.5] * 4, neto=[1, 2, 3, 999])
    pd.testing.assert_frame_equal(
        p2_features(base).iloc[:-1], p2_features(futuro_distinto).iloc[:-1]
    )


def test_features_tabla_final_tiene_las_columnas_y_no_incluye_el_objetivo():
    resultado = p2_features(p2_etiqueta_riesgo(_flujo([0.5, 0.05, 0.5])))
    assert set(P2_FEATURES) <= set(resultado.columns)
    assert P2_OBJETIVO in resultado.columns
    assert P2_OBJETIVO not in P2_FEATURES

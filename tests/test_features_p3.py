"""Pruebas unitarias de las funciones de features de P3 (clustering de estaciones)."""

import numpy as np
import pandas as pd
import pytest

from tc5061_proyecto_equipo17.features import (
    matriz_clustering_estaciones,
    perfil_horario_entre_semana,
    proporcion_fin_de_semana,
)


def _viajes(filas):
    return pd.DataFrame(filas, columns=["origen_id", "hora", "dia_semana"])


# ---- perfil_horario_entre_semana ----


def test_perfil_horario_filas_suman_uno_y_tiene_24_horas(viajes_sinteticos):
    perfil = perfil_horario_entre_semana(viajes_sinteticos)
    assert list(perfil.columns) == [f"h{h:02d}" for h in range(24)]
    assert np.allclose(perfil.sum(axis=1), 1.0)


def test_perfil_horario_calcula_proporciones_correctas():
    # Estación 1: 3 viajes a las 8 h y 1 a las 18 h, todos entre semana
    viajes = _viajes([(1, 8, 0), (1, 8, 1), (1, 8, 2), (1, 18, 3)])
    perfil = perfil_horario_entre_semana(viajes)
    assert perfil.loc[1, "h08"] == pytest.approx(0.75)
    assert perfil.loc[1, "h18"] == pytest.approx(0.25)
    assert perfil.loc[1, "h00"] == 0.0


def test_perfil_horario_ignora_fin_de_semana():
    # Los viajes de sábado (5) y domingo (6) no cuentan en el perfil entre semana
    viajes = _viajes([(1, 8, 0), (1, 9, 5), (1, 9, 6)])
    perfil = perfil_horario_entre_semana(viajes)
    assert perfil.loc[1, "h08"] == pytest.approx(1.0)
    assert perfil.loc[1, "h09"] == 0.0


def test_perfil_horario_estacion_solo_fin_de_semana_no_aparece():
    # Caso límite: una estación sin ningún viaje entre semana no tiene perfil
    viajes = _viajes([(1, 8, 0), (2, 10, 5), (2, 11, 6)])
    perfil = perfil_horario_entre_semana(viajes)
    assert 2 not in perfil.index
    assert 1 in perfil.index


# ---- proporcion_fin_de_semana ----


def test_proporcion_fin_de_semana_valores():
    viajes = _viajes([(1, 8, 0), (1, 8, 5), (1, 8, 6), (1, 8, 2), (2, 8, 1)])
    prop = proporcion_fin_de_semana(viajes)
    assert prop.loc[1] == pytest.approx(0.5)
    assert prop.loc[2] == 0.0  # caso límite: ningún viaje en fin de semana
    assert prop.name == "prop_finde"


def test_proporcion_fin_de_semana_todo_en_fin_de_semana():
    prop = proporcion_fin_de_semana(_viajes([(1, 10, 5), (1, 11, 6)]))
    assert prop.loc[1] == pytest.approx(1.0)


# ---- matriz_clustering_estaciones ----


def test_matriz_tiene_columna_prop_finde_y_sin_nulos(viajes_sinteticos):
    matriz = matriz_clustering_estaciones(viajes_sinteticos)
    assert "prop_finde" in matriz.columns
    assert not matriz.isna().any().any()
    assert matriz.index.is_unique


def test_matriz_quita_columnas_constantes(viajes_sinteticos):
    # Las horas 1-6 no tienen viajes en los datos sintéticos: son constantes (0) y se descartan
    matriz = matriz_clustering_estaciones(viajes_sinteticos)
    for hora in ["h01", "h02", "h03", "h04", "h05", "h06"]:
        assert hora not in matriz.columns
    assert (matriz.std() > 0).all()


def test_matriz_min_viajes_descarta_estaciones_pequenas(viajes_sinteticos):
    # Se agrega una estación con solo 3 viajes entre semana
    pocas = _viajes([(999, 8, 0), (999, 9, 1), (999, 10, 2)])
    viajes = pd.concat([viajes_sinteticos, pocas], ignore_index=True)

    con_todas = matriz_clustering_estaciones(viajes, min_viajes=0)
    filtrada = matriz_clustering_estaciones(viajes, min_viajes=10)

    assert 999 in con_todas.index
    assert 999 not in filtrada.index
    assert len(filtrada) == len(con_todas) - 1


def test_matriz_estacion_sin_viajes_entre_semana_se_excluye():
    # Caso límite: la unión es interna, así que solo quedan estaciones con ambos tipos de dato
    viajes = pd.concat(
        [
            _viajes([(1, 8, 0), (1, 9, 5), (2, 18, 1), (2, 17, 6), (3, 12, 2), (3, 13, 2)]),
            _viajes([(4, 10, 5), (4, 11, 6)]),  # solo fin de semana
        ],
        ignore_index=True,
    )
    matriz = matriz_clustering_estaciones(viajes)
    assert 4 not in matriz.index

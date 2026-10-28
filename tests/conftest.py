"""Datos sintéticos compartidos por las pruebas de P3.

Las pruebas no usan los datos reales de MiBici (no se suben al repositorio): generan
viajes pequeños con un patrón conocido para que corran igual en cualquier computadora.
"""

import numpy as np
import pandas as pd
import pytest


def crear_viajes(n_estaciones: int = 60, viajes_por_estacion: int = 120, semilla: int = 0):
    """Viajes sintéticos con columnas origen_id, hora y dia_semana (0=lunes ... 6=domingo).

    Tres tipos de estación: laboral (picos a las 8 y 18 h), recreativa (más fin de
    semana, tarde) y nocturna (horas tardías). Así el clustering tiene estructura real.
    """
    rng = np.random.default_rng(semilla)
    filas = []
    for estacion in range(n_estaciones):
        tipo = estacion % 3
        for _ in range(viajes_por_estacion):
            if tipo == 0:  # laboral
                hora = int(rng.choice([7, 8, 9, 17, 18, 19]))
                dia = int(rng.choice([0, 1, 2, 3, 4, 4, 4, 5]))
            elif tipo == 1:  # recreativa
                hora = int(rng.choice([10, 11, 12, 13, 14, 15]))
                dia = int(rng.choice([0, 5, 5, 6, 6, 6]))
            else:  # nocturna
                hora = int(rng.choice([20, 21, 22, 23, 0]))
                dia = int(rng.choice([1, 2, 3, 4, 5]))
            filas.append((estacion, hora, dia))
    return pd.DataFrame(filas, columns=["origen_id", "hora", "dia_semana"])


@pytest.fixture
def viajes_sinteticos():
    return crear_viajes()


@pytest.fixture
def viajes_enero_febrero():
    """Dos 'meses' con la misma estructura pero distinta semilla (entrenamiento y prueba)."""
    return crear_viajes(semilla=1), crear_viajes(semilla=2)

"""Genera escenarios FinOps con una tarifa documentada."""

import json
from pathlib import Path

import pandas as pd

from scripts.finops_p2 import costo_droplet

BASE = Path("reports/finops")


def generar_escenarios(precio, rps_base, horas_mes=720):
    """Genera escenarios ilustrativos de demanda."""
    if precio < 0 or rps_base <= 0 or horas_mes <= 0:
        raise ValueError("Parametros invalidos")

    resultados = []

    for nombre, factor in (
        ("Baja", 0.25),
        ("Normal", 0.50),
        ("Alta", 1.00),
    ):
        rps = rps_base * factor
        inferencias = rps * horas_mes * 3600

        resultados.append({
            "escenario": nombre,
            "rps_estimadas": rps,
            "inferencias_mes": int(inferencias),
            "costo_infraestructura_usd": precio,
            "costo_por_inferencia_usd": precio / inferencias,
            "costo_por_millon_usd": precio / inferencias * 1000000,
        })

    return pd.DataFrame(resultados)


def main():
    with (BASE / "referencia_nube.json").open(
        encoding="utf-8"
    ) as archivo:
        referencia = json.load(archivo)

    with (BASE / "parametros.json").open(
        encoding="utf-8"
    ) as archivo:
        parametros = json.load(archivo)

    carga = parametros["prueba_carga"]

    if carga["rps_promedio"] is None:
        print("No hay RPS medidas por Locust.")
        print("No se generara una estimacion de rendimiento real.")
        return

    horas_mes = 720
    precio = costo_droplet(
        horas_mes,
        referencia["precio_publicado_usd_hora"],
        referencia["precio_publicado_usd_mes"],
    )

    tabla = generar_escenarios(
        precio,
        float(parametros["capacidad_sostenible"]["rps_exitosas"]),
        horas_mes,
    )

    BASE.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(BASE / "escenarios_finops.csv", index=False)

    print("\nESCENARIOS FINOPS")
    print(tabla.to_string(index=False))
    print("\nArchivo: reports/finops/escenarios_finops.csv")


if __name__ == "__main__":
    main()

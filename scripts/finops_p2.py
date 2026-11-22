"""Analisis FinOps del sistema MiBici P2."""

import json
from pathlib import Path

import pandas as pd

CONFIG = Path("reports/finops/parametros.json")
SALIDA = Path("reports/finops")


def calcular_costo_por_inferencia(
    costo_hora,
    duracion_segundos,
    inferencias_exitosas,
):
    """Calcula el costo de infraestructura por inferencia."""

    if costo_hora < 0:
        raise ValueError("El costo por hora no puede ser negativo")

    if duracion_segundos <= 0:
        raise ValueError("La duracion debe ser positiva")

    if inferencias_exitosas <= 0:
        raise ValueError("Debe existir al menos una inferencia exitosa")

    costo_periodo = costo_hora * duracion_segundos / 3600

    return {
        "costo_periodo_usd": costo_periodo,
        "inferencias_exitosas": inferencias_exitosas,
        "costo_por_inferencia_usd": (
            costo_periodo / inferencias_exitosas
        ),
        "inferencias_por_dolar": (
            inferencias_exitosas / costo_periodo
            if costo_periodo > 0 else None
        ),
    }



def proyectar_escenarios(
    costo_hora,
    rps_base,
    horas_mes=720,
):
    """Proyecta costos de tres niveles de demanda.

    Los escenarios son hipoteticos, no resultados de Locust.
    """
    if costo_hora < 0 or rps_base <= 0 or horas_mes <= 0:
        raise ValueError("Parametros de proyeccion invalidos")

    escenarios = {
        "Baja": 0.25,
        "Normal": 0.50,
        "Alta": 1.00,
    }

    costo_mensual = costo_hora * horas_mes
    resultados = []

    for nombre, factor in escenarios.items():
        rps = rps_base * factor
        inferencias = rps * horas_mes * 3600

        resultados.append({
            "escenario": nombre,
            "rps_proyectadas": rps,
            "inferencias_mes": round(inferencias),
            "costo_mensual_usd": costo_mensual,
            "costo_por_inferencia_usd": (
                costo_mensual / inferencias
            ),
        })

    return pd.DataFrame(resultados)


def calcular_costo_mensual(
    costo_hora,
    horas_mes,
    almacenamiento_usd_mes,
    inferencias_mes,
):
    """Calcula el costo mensual total y su costo unitario."""

    if costo_hora < 0 or almacenamiento_usd_mes < 0:
        raise ValueError("Los costos no pueden ser negativos")

    if horas_mes <= 0 or inferencias_mes <= 0:
        raise ValueError("Horas e inferencias deben ser positivas")

    costo_compute = costo_hora * horas_mes
    costo_total = costo_compute + almacenamiento_usd_mes

    return {
        "compute_usd": costo_compute,
        "almacenamiento_usd": almacenamiento_usd_mes,
        "costo_total_usd": costo_total,
        "inferencias_mes": inferencias_mes,
        "costo_por_inferencia_usd": costo_total / inferencias_mes,
    }

def main():
    with CONFIG.open(encoding="utf-8") as archivo:
        config = json.load(archivo)

    infraestructura = config["infraestructura"]
    carga = config["prueba_carga"]

    costo_hora = infraestructura[
        "costo_infraestructura_usd_hora"
    ]
    duracion = carga["duracion_segundos"]
    exitosas = carga["peticiones_exitosas"]

    print("\nANALISIS FINOPS - EQUIPO 17")
    print("Servicios:", infraestructura["servicios"])

    pendientes = []

    if costo_hora is None:
        pendientes.append("Precio de infraestructura USD/h")

    if duracion is None:
        pendientes.append("Duracion de la prueba Locust")

    if exitosas is None:
        pendientes.append("Peticiones exitosas de Locust")

    if pendientes:
        print("\nPendiente de completar:")
        for dato in pendientes:
            print("-", dato)

        print("\nNo se calcularan costos sin datos verificables.")
        return

    resultado = calcular_costo_por_inferencia(
        float(costo_hora),
        float(duracion),
        int(exitosas),
    )

    SALIDA.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([resultado]).to_csv(
        SALIDA / "resultado_costos.csv",
        index=False,
    )

    print("\nRESULTADOS")
    for nombre, valor in resultado.items():
        print(f"{nombre}: {valor}")

    print("\nArchivo: reports/finops/resultado_costos.csv")


if __name__ == "__main__":
    main()


def costo_droplet(horas_activas, tarifa_hora, tope_mensual):
    """Costo estimado de un Droplet dentro de un mes."""
    if horas_activas < 0 or tarifa_hora < 0 or tope_mensual < 0:
        raise ValueError("Los valores no pueden ser negativos")

    return min(horas_activas * tarifa_hora, tope_mensual)

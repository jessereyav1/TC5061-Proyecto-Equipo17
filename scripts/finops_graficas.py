"""Genera graficas FinOps a partir de resultados calculados y Locust."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

SALIDA = Path("reports/finops")
SALIDA.mkdir(parents=True, exist_ok=True)

def main():
    escenarios = pd.read_csv(SALIDA / "escenarios_finops.csv")
    etapas = pd.read_csv("reports/carga/resumen_etapas.csv")

    # Grafica 1: costo por millon de inferencias
    fig, ax = plt.subplots(figsize=(9, 5))
    colores = ["#94A3B8", "#2563EB", "#0F766E"]

    barras = ax.bar(
        escenarios["escenario"],
        escenarios["costo_por_millon_usd"],
        color=colores,
        width=0.58,
    )

    ax.bar_label(barras, fmt="US$%.2f", padding=5)
    ax.set_ylabel("USD por millon de inferencias")
    ax.set_title("MiBici P2 | Proyeccion FinOps mensual")
    ax.grid(axis="y", alpha=0.18)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(SALIDA / "costo_por_millon.png", dpi=250)
    plt.close(fig)

    # Grafica 2: rendimiento observado y cumplimiento del SLO
    fig, ax = plt.subplots(figsize=(9, 5))

    colores = [
        "#0F766E" if cumple else "#DC2626"
        for cumple in etapas["cumple_slo"]
    ]

    barras = ax.bar(
        etapas["usuarios"].astype(str),
        etapas["rps_exitosas"],
        color=colores,
        width=0.6,
    )

    ax.bar_label(barras, fmt="%.2f", padding=4)
    ax.set_xlabel("Usuarios concurrentes")
    ax.set_ylabel("Predicciones exitosas por segundo")
    ax.set_title("MiBici P2 | Rendimiento medido con Locust")
    ax.grid(axis="y", alpha=0.18)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(SALIDA / "rendimiento_locust.png", dpi=250)
    plt.close(fig)

    print("Graficas generadas:")
    print("-", SALIDA / "costo_por_millon.png")
    print("-", SALIDA / "rendimiento_locust.png")


if __name__ == "__main__":
    main()

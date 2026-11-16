"""Graficas de resultados A/B del modelo P2."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CARPETA = Path("reports/ab_testing")
datos = pd.read_csv(CARPETA / "comparacion_modelos.csv")
datos = datos.set_index("modelo").loc[["champion", "challenger"]]

CARPETA.mkdir(parents=True, exist_ok=True)

# Figura 1: metricas de clasificacion
metricas = ["precision", "recall", "f1", "pr_auc"]
nombres = ["Precision", "Recall", "F1-score", "PR-AUC"]

x = np.arange(len(metricas))
ancho = 0.35

fig, ax = plt.subplots(figsize=(10, 5.5))

for posicion, modelo in enumerate(["champion", "challenger"]):
    desplazamiento = (posicion - 0.5) * ancho
    valores = datos.loc[modelo, metricas].to_numpy(dtype=float)

    barras = ax.bar(
        x + desplazamiento,
        valores,
        ancho,
        label=modelo.capitalize()
    )
    ax.bar_label(barras, fmt="%.3f", padding=3, fontsize=9)

ax.set_title("MiBici P2 | Comparacion A/B de modelos")
ax.set_ylabel("Valor de la metrica")
ax.set_xticks(x, nombres)
ax.set_ylim(0, 1.05)
ax.legend()
ax.grid(axis="y", alpha=0.2)
fig.tight_layout()

fig.savefig(
    CARPETA / "comparacion_metricas.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close(fig)

# Figura 2: errores de clasificacion
fig, ax = plt.subplots(figsize=(9, 5.5))

errores = ["fp", "fn"]
nombres = ["Falsos positivos", "Falsos negativos"]
x = np.arange(2)

for posicion, modelo in enumerate(["champion", "challenger"]):
    desplazamiento = (posicion - 0.5) * ancho
    valores = datos.loc[modelo, errores].to_numpy(dtype=int)

    barras = ax.bar(
        x + desplazamiento,
        valores,
        ancho,
        label=modelo.capitalize()
    )
    ax.bar_label(barras, padding=3, fontsize=10)

ax.set_title("MiBici P2 | Errores de clasificacion")
ax.set_ylabel("Cantidad de observaciones")
ax.set_xticks(x, nombres)
ax.legend()
ax.grid(axis="y", alpha=0.2)
fig.tight_layout()

fig.savefig(
    CARPETA / "comparacion_errores.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close(fig)

print("Graficas generadas correctamente:")
print("- reports/ab_testing/comparacion_metricas.png")
print("- reports/ab_testing/comparacion_errores.png")

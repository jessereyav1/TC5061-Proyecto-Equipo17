"""Comparacion offline A/B de modelos P2 del Equipo 17."""

import json
from pathlib import Path

import mlflow.sklearn
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from tc5061_proyecto_equipo17.features import P2_FEATURES

MODEL_NAME = "mibici-equipo17-p2-riesgo"
DATA_PATH = Path("data/processed/monitoreo/p2_actual.parquet")
OUTPUT_DIR = Path("reports/ab_testing")
EXPECTED_ROWS = 87713



def calcular_metricas(y, predicciones, probabilidades):
    """Calcula metricas binarias y la matriz de confusion."""
    tn, fp, fn, tp = confusion_matrix(
        y, predicciones, labels=[0, 1]
    ).ravel()

    return {
        "precision": float(precision_score(y, predicciones, zero_division=0)),
        "recall": float(recall_score(y, predicciones, zero_division=0)),
        "f1": float(f1_score(y, predicciones, zero_division=0)),
        "pr_auc": float(average_precision_score(y, probabilidades)),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }

def main():
    df = pd.read_parquet(DATA_PATH)
    evaluacion = df.loc[df["ventana"] == "evaluacion"].copy()

    if len(evaluacion) != EXPECTED_ROWS:
        raise ValueError(
            f"Se esperaban {EXPECTED_ROWS} filas; "
            f"se encontraron {len(evaluacion)}"
        )

    fechas = pd.to_datetime(evaluacion["hora_ts"])

    inicio = pd.Timestamp("2025-07-22")
    fin = pd.Timestamp("2025-08-01")

    if not fechas.ge(inicio).all() or not fechas.lt(fin).all():
        raise ValueError("Hay observaciones fuera de julio 22-31")

    if evaluacion.duplicated(["estacion_id", "hora_ts"]).any():
        raise ValueError("Hay estaciones y horas duplicadas")

    print("Ventana temporal validada: julio 22-31 de 2025")
    print("Registros duplicados: 0")

    columnas = list(P2_FEATURES)
    X = evaluacion.loc[:, columnas]
    y = evaluacion["riesgo_siguiente_hora"]

    if X.isna().any().any() or y.isna().any():
        raise ValueError("La evaluacion contiene valores nulos")

    if not set(y.unique()).issubset({0, 1}):
        raise ValueError("La etiqueta debe ser binaria")

    resultados = []

    for alias in ("champion", "challenger"):
        uri = f"models:/{MODEL_NAME}@{alias}"
        modelo = mlflow.sklearn.load_model(uri)

        if list(modelo.feature_names_in_) != columnas:
            raise ValueError(
                f"Variables incompatibles en {alias}"
            )

        probabilidades = modelo.predict_proba(X)[:, 1]
        predicciones = (probabilidades >= 0.5).astype(int)

        metricas = calcular_metricas(y, predicciones, probabilidades)
        resultados.append({
            "modelo": alias,
            "uri": uri,
            "observaciones": len(y),
            **metricas,
        })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tabla = pd.DataFrame(resultados)

    tabla.to_csv(
        OUTPUT_DIR / "comparacion_modelos.csv",
        index=False,
    )

    with (OUTPUT_DIR / "resultados_ab.json").open(
        "w", encoding="utf-8"
    ) as archivo:
        json.dump(
            resultados, archivo, indent=2, ensure_ascii=False
        )

    print("\nRESULTADOS A/B TESTING")
    print(
        tabla[
            ["modelo", "precision", "recall",
             "f1", "pr_auc", "tp", "fp", "fn", "tn"]
        ].round(4).to_string(index=False)
    )

    print("\nArchivos generados en:", OUTPUT_DIR)


if __name__ == "__main__":
    main()

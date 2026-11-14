"""Incertidumbre estadistica del A/B Testing P2."""

from pathlib import Path

import mlflow.sklearn
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import confusion_matrix

from tc5061_proyecto_equipo17.features import P2_FEATURES

RUTA = Path("data/processed/monitoreo/p2_actual.parquet")
SALIDA = Path("reports/ab_testing")
MODELO = "mibici-equipo17-p2-riesgo"
SEMILLA = 42
REPETICIONES = 5000


def metricas(conteos):
    tn, fp, fn, tp = conteos
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )
    return np.array([precision, recall, f1])


def main():
    df = pd.read_parquet(RUTA)
    df = df.loc[df["ventana"] == "evaluacion"].copy()
    df["dia"] = pd.to_datetime(df["hora_ts"]).dt.date

    if len(df) != 87713 or df["dia"].nunique() != 10:
        raise ValueError("Ventana de evaluacion inesperada")

    X = df[list(P2_FEATURES)]
    y = df["riesgo_siguiente_hora"].to_numpy()

    predicciones = {}
    conteos_diarios = {}

    for alias in ("champion", "challenger"):
        uri = f"models:/{MODELO}@{alias}"
        modelo = mlflow.sklearn.load_model(uri)
        pred = (modelo.predict_proba(X)[:, 1] >= 0.5).astype(int)
        predicciones[alias] = pred

        diarios = []
        for dia in sorted(df["dia"].unique()):
            mask = (df["dia"].to_numpy() == dia)
            diarios.append(
                confusion_matrix(
                    y[mask], pred[mask], labels=[0, 1]
                ).ravel()
            )
        conteos_diarios[alias] = np.asarray(diarios)

    # Bootstrap pareado por dia: ambos modelos usan los mismos dias.
    rng = np.random.default_rng(SEMILLA)
    muestras = rng.integers(
        0, 10, size=(REPETICIONES, 10)
    )

    diferencias = []
    for indices in muestras:
        a = conteos_diarios["champion"][indices].sum(axis=0)
        b = conteos_diarios["challenger"][indices].sum(axis=0)
        diferencias.append(metricas(b) - metricas(a))

    diferencias = np.asarray(diferencias)
    nombres = ["precision", "recall", "f1"]

    registros = []
    print("\nBOOTSTRAP PAREADO POR DIA (IC 95%)")
    for j, nombre in enumerate(nombres):
        a = conteos_diarios["champion"].sum(axis=0)
        b = conteos_diarios["challenger"].sum(axis=0)

        delta = metricas(b)[j] - metricas(a)[j]
        inferior, superior = np.quantile(
            diferencias[:, j], [0.025, 0.975]
        )

        print(
            f"{nombre}: delta={delta:+.5f}, "
            f"IC95%=[{inferior:+.5f}, {superior:+.5f}]"
        )

        registros.append({
            "metrica": nombre,
            "delta": float(delta),
            "ic95_inferior": float(inferior),
            "ic95_superior": float(superior),
        })

    # McNemar exacto sobre aciertos y errores pareados.
    correcto_a = predicciones["champion"] == y
    correcto_b = predicciones["challenger"] == y

    solo_a = int(np.sum(correcto_a & ~correcto_b))
    solo_b = int(np.sum(~correcto_a & correcto_b))

    pvalor = binomtest(
        min(solo_a, solo_b),
        n=solo_a + solo_b,
        p=0.5,
        alternative="two-sided",
    ).pvalue if solo_a + solo_b else 1.0

    print("\nMCNEMAR EXACTO (COMPLEMENTARIO)")
    print("Solo champion acierta:", solo_a)
    print("Solo challenger acierta:", solo_b)
    print("p-valor:", round(pvalor, 8))
    print("Nota: no ajusta la dependencia entre horas/estaciones.")

    SALIDA.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(registros).to_csv(
        SALIDA / "bootstrap_intervalos.csv", index=False
    )

    pd.DataFrame([{
        "solo_champion_acierta": solo_a,
        "solo_challenger_acierta": solo_b,
        "p_valor_mcnemar": pvalor,
    }]).to_csv(
        SALIDA / "mcnemar_resultado.csv", index=False
    )

    print("\nResultados guardados en:", SALIDA)


if __name__ == "__main__":
    main()

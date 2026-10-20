import hashlib
import json
import subprocess

from loguru import logger
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
import typer

from tc5061_proyecto_equipo17.config import (
    EXTERNAL_DATA_DIR,
    FECHA_CORTE_TEST,
    FECHA_FIN,
    FECHA_INICIO,
    FIGURES_DIR,
    MLFLOW_EXPERIMENT,
    PROCESSED_DATA_DIR,
    SEED,
)
from tc5061_proyecto_equipo17.features import (
    P2_FEATURES,
    P2_OBJETIVO,
    p2_construir_dataset,
)

app = typer.Typer()

ARCHIVO_NOMENCLATURA = "nomenclatura_2026_09.csv"


def _commit_actual() -> str:
    """Devuelve el hash del commit de Git con el que se ejecuta el run."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "desconocido"


def _version_datos() -> str:
    """Huella de los datos procesados a partir de los SHA-256 del manifiesto.

    Se combinan los hashes de los parquet (no el manifiesto completo, que incluye
    la fecha de procesamiento) para que la versión solo cambie si cambian los datos.
    """
    manifest = json.loads(
        (PROCESSED_DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
    hashes = "".join(sorted(a["sha256_procesado"]
                     for a in manifest["archivos"]))
    return hashlib.sha256(hashes.encode()).hexdigest()[:12]


def _metricas(y_true, pred, score) -> dict:
    """Métricas de clasificación en el conjunto de prueba."""
    return {
        "test_precision": precision_score(y_true, pred),
        "test_recall": recall_score(y_true, pred),
        "test_f1": f1_score(y_true, pred),
        "test_pr_auc": average_precision_score(y_true, score),
    }


def _registrar_run(nombre, params, y_true, pred, score) -> None:
    """Registra un run de P2 con parámetros, etiquetas, métricas y matriz de confusión."""
    with mlflow.start_run(run_name=nombre):
        mlflow.log_params(params)
        mlflow.set_tags(
            {"equipo": "17", "pregunta": "P2-clasificacion",
                "git_commit": _commit_actual()}
        )
        metricas = _metricas(y_true, pred, score)
        mlflow.log_metrics(metricas)

        fig, ax = plt.subplots(figsize=(4, 4))
        ConfusionMatrixDisplay.from_predictions(
            y_true, pred, ax=ax, colorbar=False)
        ax.set_title(nombre)
        ruta = FIGURES_DIR / f"{nombre}_matriz_confusion.png"
        fig.savefig(ruta, bbox_inches="tight")
        plt.close(fig)
        mlflow.log_artifact(str(ruta))

        logger.success(
            f"{nombre}: " + ", ".join(f"{k}={v:.3f}" for k, v in metricas.items()))


@app.command()
def main(n_estimators: int = 100, max_depth: int = 12):
    """Entrena el baseline de persistencia y un Random Forest para P2 y los registra en MLflow."""
    archivos = sorted(PROCESSED_DATA_DIR.glob("*_procesado.parquet"))
    viajes = pd.concat([pd.read_parquet(f)
                       for f in archivos], ignore_index=True)
    estaciones = pd.read_csv(EXTERNAL_DATA_DIR / ARCHIVO_NOMENCLATURA)

    ds = p2_construir_dataset(viajes, estaciones, FECHA_INICIO, FECHA_FIN)
    fechas = ds.index.get_level_values("hora_ts")
    train = ds[fechas < FECHA_CORTE_TEST]
    test = ds[fechas >= FECHA_CORTE_TEST]
    y_test = test[P2_OBJETIVO]
    logger.info(f"Train: {len(train):,} filas | Test: {len(test):,} filas")

    comunes = {
        "fecha_min": FECHA_INICIO,
        "fecha_max": FECHA_FIN,
        "fecha_corte_test": FECHA_CORTE_TEST,
        "version_datos": _version_datos(),
        "archivo_nomenclatura": ARCHIVO_NOMENCLATURA,
        "semilla": SEED,
        # Valores por defecto de p2_ocupacion_estimada y p2_etiqueta_riesgo
        "frac_inicial": 0.5,
        "umbral_bajo": 0.10,
        "umbral_alto": 0.90,
        "features": ",".join(P2_FEATURES),
        "n_train": len(train),
        "n_test": len(test),
    }

    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    # Run 1: baseline de persistencia (la siguiente hora igual a la actual)
    pred_base = test["riesgo_ahora"]
    _registrar_run(
        "p2-baseline-persistencia",
        {**comunes, "modelo": "persistencia"},
        y_test, pred_base, pred_base,
    )

    # Run 2: Random Forest con pesos balanceados por el desbalance de clases
    modelo = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )
    modelo.fit(train[P2_FEATURES], train[P2_OBJETIVO])
    score = modelo.predict_proba(test[P2_FEATURES])[:, 1]
    pred = (score >= 0.5).astype(int)
    _registrar_run(
        "p2-random-forest-balanced",
        {
            **comunes,
            "modelo": "RandomForestClassifier",
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "class_weight": "balanced",
        },
        y_test, pred, score,
    )


if __name__ == "__main__":
    app()

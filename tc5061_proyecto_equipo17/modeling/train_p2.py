"""P2: riesgo de que una estación se quede sin bicis o sin anclajes la siguiente hora.

Cada ejecución produce una sola corrida en MLflow (compatible con `mlflow run`):

- Con hiperparámetros (--n-estimators y/o --max-depth): un Random Forest candidato.
- Sin hiperparámetros: búsqueda en rejilla. La corrida es el padre y contiene una hija
  con el baseline de persistencia y una hija por cada combinación de la rejilla.

Partición temporal: los hiperparámetros se eligen entrenando con el 1-24 de enero y
validando con el 25-31 de enero; el modelo final se reentrena con todo enero y se
evalúa una sola vez en febrero (mismo test que la entrega anterior).
"""

import hashlib
from itertools import product
import json
from pathlib import Path
import subprocess
import tempfile
import time

from loguru import logger
import matplotlib

matplotlib.use("Agg")  # dibuja a archivo, sin abrir ventanas
import matplotlib.pyplot as plt
import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
import typer

from tc5061_proyecto_equipo17.config import (
    EXTERNAL_DATA_DIR,
    FECHA_CORTE_TEST,
    FECHA_CORTE_VAL,
    FECHA_FIN,
    FECHA_INICIO,
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

# Valores del Random Forest de la entrega anterior (se usan si solo llega un hiperparámetro)
N_ESTIMATORS_BASE = 100
MAX_DEPTH_BASE = 12

# Rejilla de la búsqueda: 2 x 3 = 6 corridas hijas
REJILLA = {"n_estimators": [100, 300], "max_depth": [8, 12, 16]}

# Umbral de decisión: igual que en la entrega anterior para comparar contra el baseline
UMBRAL = 0.5


def _commit_actual() -> str:
    """Hash del commit de Git, para rastrear qué código generó la corrida."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "desconocido"


def _version_datos() -> str:
    """Huella de los datos procesados a partir de los SHA-256 del manifiesto.

    Se combinan los hashes de los parquet (no el manifiesto completo, que incluye la
    fecha de procesamiento) para que la versión solo cambie si cambian los datos.
    """
    manifest = json.loads((PROCESSED_DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
    hashes = "".join(sorted(a["sha256_procesado"] for a in manifest["archivos"]))
    return hashlib.sha256(hashes.encode()).hexdigest()[:12]


def _metricas(y_true, pred, score, prefijo: str) -> dict:
    """Precision, recall, F1 y PR-AUC con el prefijo del conjunto (val_ o test_)."""
    return {
        f"{prefijo}_precision": precision_score(y_true, pred),
        f"{prefijo}_recall": recall_score(y_true, pred),
        f"{prefijo}_f1": f1_score(y_true, pred),
        f"{prefijo}_pr_auc": average_precision_score(y_true, score),
    }


def _etiquetas(familia: str, etapa: str) -> dict:
    """Etiquetas comunes del equipo para filtrar y comparar en la interfaz de MLflow."""
    return {
        "equipo": "17",
        "pregunta": "P2-clasificacion",
        "question": "p2",
        "model_family": familia,
        "stage_of_work": etapa,
        "git_commit": _commit_actual(),
    }


def _random_forest(n_estimators: int, max_depth: int) -> RandomForestClassifier:
    """Random Forest con pesos balanceados por el desbalance de clases (15.6% en riesgo)."""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )


def _evaluar_rf(n_estimators, max_depth, X_fit, y_fit, X_eval, y_eval, prefijo):
    """Entrena un RF y devuelve (modelo, métricas, score) sobre el conjunto de evaluación."""
    modelo = _random_forest(n_estimators, max_depth)
    inicio = time.perf_counter()
    modelo.fit(X_fit, y_fit)
    segundos = time.perf_counter() - inicio
    score = modelo.predict_proba(X_eval)[:, 1]
    pred = (score >= UMBRAL).astype(int)
    metricas = {**_metricas(y_eval, pred, score, prefijo), f"{prefijo}_train_segundos": segundos}
    return modelo, metricas, score


def _registrar_graficas(y_true, pred, score, titulo: str, tmp: Path) -> None:
    """plots/: matriz de confusión y curva precision-recall sobre el conjunto de prueba."""
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay.from_predictions(y_true, pred, ax=ax, colorbar=False)
    ax.set_title(titulo)
    fig.savefig(tmp / "matriz_confusion.png", bbox_inches="tight")
    plt.close(fig)
    mlflow.log_artifact(str(tmp / "matriz_confusion.png"), "plots")

    fig, ax = plt.subplots(figsize=(5, 4))
    PrecisionRecallDisplay.from_predictions(y_true, score, ax=ax)
    ax.set_title(f"Curva precision-recall (test) - {titulo}")
    fig.savefig(tmp / "curva_precision_recall.png", bbox_inches="tight")
    plt.close(fig)
    mlflow.log_artifact(str(tmp / "curva_precision_recall.png"), "plots")


def _registrar_artefactos(config_corrida: dict, train, test, tmp: Path) -> None:
    """data/: manifiesto y conjuntos con log_input; config/: parámetros de la corrida."""
    mlflow.log_artifact(str(PROCESSED_DATA_DIR / "manifest.json"), "data")
    fuente = str(PROCESSED_DATA_DIR)
    # Solo features agregadas por estación y hora: no hay datos personales de usuarios
    mlflow.log_input(
        mlflow.data.from_pandas(train.reset_index(), source=fuente, name="p2_train"),
        context="training",
    )
    mlflow.log_input(
        mlflow.data.from_pandas(test.reset_index(), source=fuente, name="p2_test"),
        context="testing",
    )

    (tmp / "config_corrida.json").write_text(
        json.dumps(config_corrida, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    mlflow.log_artifact(str(tmp / "config_corrida.json"), "config")


def _registrar_modelo(modelo, X_train) -> None:
    """Modelo con firma (columnas de entrada/salida) y ejemplo de entrada."""
    mlflow.sklearn.log_model(
        modelo,
        name="modelo",
        signature=infer_signature(X_train, modelo.predict(X_train)),
        input_example=X_train.head(5),
        # skops no confía por defecto en los árboles de sklearn; el modelo lo entrenamos
        # nosotros, así que se autoriza solo ese tipo
        skops_trusted_types=["sklearn.tree._tree.Tree"],
    )


@app.command()
def main(n_estimators: int = 0, max_depth: int = 0):
    """Entrena un RF candidato o, si no se pasan hiperparámetros, lanza la búsqueda."""
    archivos = sorted(PROCESSED_DATA_DIR.glob("*_procesado.parquet"))
    viajes = pd.concat([pd.read_parquet(f) for f in archivos], ignore_index=True)
    estaciones = pd.read_csv(EXTERNAL_DATA_DIR / ARCHIVO_NOMENCLATURA)

    ds = p2_construir_dataset(viajes, estaciones, FECHA_INICIO, FECHA_FIN)
    fechas = ds.index.get_level_values("hora_ts")
    train = ds[fechas < FECHA_CORTE_TEST]  # todo enero: modelo final
    test = ds[fechas >= FECHA_CORTE_TEST]  # febrero: se evalúa una sola vez
    fechas_train = train.index.get_level_values("hora_ts")
    train_busq = train[fechas_train < FECHA_CORTE_VAL]  # 1-24 de enero
    val = train[fechas_train >= FECHA_CORTE_VAL]  # 25-31 de enero
    logger.info(
        f"Train búsqueda: {len(train_busq):,} | Val: {len(val):,} | "
        f"Train final: {len(train):,} | Test: {len(test):,} filas"
    )

    X_busq, y_busq = train_busq[P2_FEATURES], train_busq[P2_OBJETIVO]
    X_val, y_val = val[P2_FEATURES], val[P2_OBJETIVO]
    X_train, y_train = train[P2_FEATURES], train[P2_OBJETIVO]
    X_test, y_test = test[P2_FEATURES], test[P2_OBJETIVO]

    comunes = {
        "fecha_min": FECHA_INICIO,
        "fecha_max": FECHA_FIN,
        "fecha_corte_val": FECHA_CORTE_VAL,
        "fecha_corte_test": FECHA_CORTE_TEST,
        "version_datos": _version_datos(),
        "archivo_nomenclatura": ARCHIVO_NOMENCLATURA,
        "semilla": SEED,
        # Valores por defecto de p2_ocupacion_estimada y p2_etiqueta_riesgo
        "frac_inicial": 0.5,
        "umbral_bajo": 0.10,
        "umbral_alto": 0.90,
        "umbral_decision": UMBRAL,
        "features": ",".join(P2_FEATURES),
        "n_train_busqueda": len(train_busq),
        "n_val": len(val),
        "n_train": len(train),
        "n_test": len(test),
    }

    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    if n_estimators > 0 or max_depth > 0:
        _candidato(
            n_estimators or N_ESTIMATORS_BASE,
            max_depth or MAX_DEPTH_BASE,
            comunes,
            (X_busq, y_busq, X_val, y_val, X_train, y_train, X_test, y_test),
            train,
            test,
        )
    else:
        _busqueda(
            comunes,
            (X_busq, y_busq, X_val, y_val, X_train, y_train, X_test, y_test),
            train,
            val,
            test,
        )


def _candidato(n_estimators, max_depth, comunes, conjuntos, train, test) -> None:
    """Una corrida: RF validado con 25-31 ene y reentrenado con todo enero para test."""
    X_busq, y_busq, X_val, y_val, X_train, y_train, X_test, y_test = conjuntos
    nombre = f"p2-rf-n{n_estimators}-d{max_depth}"
    config_corrida = {
        **comunes,
        "modelo": "RandomForestClassifier",
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "class_weight": "balanced",
    }

    with mlflow.start_run(run_name=nombre):
        mlflow.log_params(config_corrida)
        mlflow.set_tags(_etiquetas("random_forest", "candidate"))

        _, metricas_val, _ = _evaluar_rf(
            n_estimators, max_depth, X_busq, y_busq, X_val, y_val, "val"
        )
        modelo, metricas_test, score = _evaluar_rf(
            n_estimators, max_depth, X_train, y_train, X_test, y_test, "test"
        )
        mlflow.log_metrics({**metricas_val, **metricas_test})

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            _registrar_graficas(y_test, (score >= UMBRAL).astype(int), score, nombre, tmp)
            _registrar_artefactos(config_corrida, train, test, tmp)
        _registrar_modelo(modelo, X_train)

    logger.success(
        f"{nombre}: " + ", ".join(f"{k}={v:.3f}" for k, v in {**metricas_val, **metricas_test}.items())
    )


def _busqueda(comunes, conjuntos, train, val, test) -> None:
    """Corrida padre con el baseline y la rejilla como hijas; el padre guarda al ganador."""
    X_busq, y_busq, X_val, y_val, X_train, y_train, X_test, y_test = conjuntos
    combinaciones = list(product(REJILLA["n_estimators"], REJILLA["max_depth"]))
    criterio = "max val_recall; desempate val_pr_auc"

    with mlflow.start_run(run_name="p2-busqueda-rf"):
        mlflow.log_params(
            {
                **comunes,
                "modelo": "RandomForestClassifier",
                "class_weight": "balanced",
                "rejilla_n_estimators": str(REJILLA["n_estimators"]),
                "rejilla_max_depth": str(REJILLA["max_depth"]),
                "criterio_seleccion": criterio,
            }
        )
        mlflow.set_tags(_etiquetas("random_forest", "search"))

        # Hija 1: baseline de persistencia (la siguiente hora igual a la actual)
        with mlflow.start_run(run_name="p2-baseline-persistencia", nested=True):
            mlflow.log_params({**comunes, "modelo": "persistencia"})
            mlflow.set_tags(_etiquetas("persistencia", "baseline"))
            pred_val, pred_test = val["riesgo_ahora"], test["riesgo_ahora"]
            metricas_base = {
                **_metricas(y_val, pred_val, pred_val, "val"),
                **_metricas(y_test, pred_test, pred_test, "test"),
            }
            mlflow.log_metrics(metricas_base)
            with tempfile.TemporaryDirectory() as tmp:
                _registrar_graficas(
                    y_test, pred_test, pred_test, "p2-baseline-persistencia", Path(tmp)
                )
        logger.success(
            "baseline: " + ", ".join(f"{k}={v:.3f}" for k, v in metricas_base.items())
        )

        # Hijas de la rejilla: solo validación (test se reserva para el ganador)
        resultados = []
        for n, d in combinaciones:
            with mlflow.start_run(run_name=f"p2-rf-n{n}-d{d}-busqueda", nested=True):
                mlflow.log_params(
                    {**comunes, "modelo": "RandomForestClassifier", "n_estimators": n,
                     "max_depth": d, "class_weight": "balanced"}
                )
                mlflow.set_tags(_etiquetas("random_forest", "search"))
                _, metricas_val, _ = _evaluar_rf(n, d, X_busq, y_busq, X_val, y_val, "val")
                mlflow.log_metrics(metricas_val)
            resultados.append({"n_estimators": n, "max_depth": d, **metricas_val})
            logger.info(f"n={n}, d={d}: val_recall={metricas_val['val_recall']:.3f}")

        resumen = pd.DataFrame(resultados).sort_values(
            ["val_recall", "val_pr_auc"], ascending=False
        )
        mejor = resumen.iloc[0]
        mejor_n, mejor_d = int(mejor["n_estimators"]), int(mejor["max_depth"])

        # Ganador: se reentrena con todo enero y se evalúa una sola vez en febrero
        modelo, metricas_test, score = _evaluar_rf(
            mejor_n, mejor_d, X_train, y_train, X_test, y_test, "test"
        )
        mlflow.log_params({"mejor_n_estimators": mejor_n, "mejor_max_depth": mejor_d})
        mlflow.log_metrics(
            {
                "mejor_val_recall": mejor["val_recall"],
                "mejor_val_pr_auc": mejor["val_pr_auc"],
                **metricas_test,
            }
        )

        config_corrida = {
            **comunes,
            "modelo": "RandomForestClassifier",
            "n_estimators": mejor_n,
            "max_depth": mejor_d,
            "class_weight": "balanced",
            "rejilla": REJILLA,
            "criterio_seleccion": criterio,
        }
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            titulo = f"p2-busqueda-rf (n{mejor_n}-d{mejor_d})"
            _registrar_graficas(y_test, (score >= UMBRAL).astype(int), score, titulo, tmp)
            _registrar_artefactos(config_corrida, train, test, tmp)
            resumen.to_csv(tmp / "resumen_busqueda.csv", index=False)
            mlflow.log_artifact(str(tmp / "resumen_busqueda.csv"), "data")
        _registrar_modelo(modelo, X_train)

    logger.success(
        f"Mejor: n={mejor_n}, d={mejor_d} | "
        + ", ".join(f"{k}={v:.3f}" for k, v in metricas_test.items())
    )


if __name__ == "__main__":
    app()

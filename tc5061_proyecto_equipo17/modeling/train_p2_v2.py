"""P2 (Actividad 10): versión alternativa del modelo, reentrenada con datos del periodo actual.

La versión en producción (alias ``champion``) se entrenó solo con enero de 2025. Esta
versión se reentrena con enero + la ventana de reentrenamiento de julio (1-21) y se
registra en el Model Registry como una nueva versión del mismo modelo, con el alias
``challenger``. El alias ``champion`` no se toca: la decisión de cuál queda vigente sale
del A/B testing.

Para que el A/B testing compare solo el efecto de reentrenar (Continuous Training), por
defecto se usan los mismos hiperparámetros que la versión ``champion`` (se leen de sus
etiquetas en el registro). La ventana de evaluación (22-31 de julio) no se usa para
entrenar ni para elegir nada; solo se registran sus métricas como referencia.

Requiere haber generado antes los conjuntos de monitoreo::

    python -m tc5061_proyecto_equipo17.monitoreo_datos ingesta-actual
    python -m tc5061_proyecto_equipo17.monitoreo_datos construir-features
    python -m tc5061_proyecto_equipo17.modeling.train_p2_v2
"""

import json
from pathlib import Path
import tempfile

from loguru import logger
import mlflow
from mlflow import MlflowClient
import mlflow.sklearn
import pandas as pd
import typer

from tc5061_proyecto_equipo17.config import (
    ACTUAL_CORTE_EVAL,
    ACTUAL_FIN,
    ACTUAL_INICIO,
    MLFLOW_EXPERIMENT,
    MONITOREO_DATA_DIR,
    REF_FIN,
    REF_INICIO,
    SEED,
)
from tc5061_proyecto_equipo17.dataset import calcular_sha256
from tc5061_proyecto_equipo17.features import P2_FEATURES, P2_OBJETIVO
from tc5061_proyecto_equipo17.modeling.register_p2 import (
    ALIAS_PRODUCCION,
    ALIAS_RETADOR,
    NOMBRE_MODELO,
)
from tc5061_proyecto_equipo17.modeling.train_p2 import (
    MAX_DEPTH_BASE,
    N_ESTIMATORS_BASE,
    UMBRAL,
    _etiquetas,
    _metricas,
    _random_forest,
    _registrar_graficas,
    _registrar_modelo,
)
from tc5061_proyecto_equipo17.monitoreo_datos import (
    ARCHIVO_ACTUAL,
    ARCHIVO_MANIFIESTO,
    ARCHIVO_REFERENCIA,
    VENTANA_EVALUACION,
    VENTANA_REENTRENAMIENTO,
)

app = typer.Typer()

NOMBRE_CORRIDA = "p2-v2-reentrenado-ene-jul"
PREFIJO_EVAL = "actual_eval"


def conjuntos_v2(referencia: pd.DataFrame, actual: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa los datos de entrenamiento y de evaluación de la versión alternativa.

    - Entrenamiento: todo el periodo de referencia (enero) + la ventana de
      reentrenamiento del periodo actual (1-21 de julio).
    - Evaluación: la ventana de evaluación (22-31 de julio), la misma del A/B testing.

    Las filas del periodo actual sin ventana (la última hora antes del corte, cuya
    etiqueta se calcula con la primera hora de evaluación) no entran en ningún conjunto.
    """
    columnas = ["estacion_id", "hora_ts", *P2_FEATURES, P2_OBJETIVO]
    reentrenamiento = actual[actual["ventana"] == VENTANA_REENTRENAMIENTO]
    evaluacion = actual[actual["ventana"] == VENTANA_EVALUACION]
    train = pd.concat([referencia[columnas], reentrenamiento[columnas]], ignore_index=True)
    return train, evaluacion[columnas].reset_index(drop=True)


def _hiperparametros_champion(client: MlflowClient) -> tuple[int, int, str]:
    """Hiperparámetros de la versión en producción; si no existe, los de la entrega anterior."""
    try:
        version = client.get_model_version_by_alias(NOMBRE_MODELO, ALIAS_PRODUCCION)
    except mlflow.exceptions.MlflowException:
        logger.warning(f"No hay {NOMBRE_MODELO}@{ALIAS_PRODUCCION}; se usan los valores base.")
        return N_ESTIMATORS_BASE, MAX_DEPTH_BASE, "ninguna"
    return (
        int(version.tags.get("n_estimators", N_ESTIMATORS_BASE)),
        int(version.tags.get("max_depth", MAX_DEPTH_BASE)),
        version.version,
    )


def _leer_conjuntos(carpeta: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Lee las tablas de monitoreo y comprueba que coinciden con su manifiesto."""
    manifiesto = json.loads((carpeta / ARCHIVO_MANIFIESTO).read_text(encoding="utf-8"))
    for clave, archivo in (("referencia", ARCHIVO_REFERENCIA), ("actual", ARCHIVO_ACTUAL)):
        ruta = carpeta / archivo
        if calcular_sha256(ruta) != manifiesto[clave]["sha256"]:
            raise RuntimeError(f"{archivo} no coincide con {ARCHIVO_MANIFIESTO}; regenera los datos.")
    referencia = pd.read_parquet(carpeta / ARCHIVO_REFERENCIA)
    actual = pd.read_parquet(carpeta / ARCHIVO_ACTUAL)
    return referencia, actual, manifiesto


def _registrar_version(client: MlflowClient, run_id: str, linaje: dict, descripcion: str) -> str:
    """Registra el modelo de la corrida como nueva versión y le asigna el alias challenger."""
    modelos = mlflow.search_logged_models(
        experiment_ids=[client.get_experiment_by_name(MLFLOW_EXPERIMENT).experiment_id],
        filter_string=f"source_run_id = '{run_id}'",
        output_format="list",
    )
    version = mlflow.register_model(f"models:/{modelos[0].model_id}", NOMBRE_MODELO, tags=linaje)
    client.update_model_version(NOMBRE_MODELO, version.version, description=descripcion)
    client.set_registered_model_alias(NOMBRE_MODELO, ALIAS_RETADOR, version.version)
    return version.version


@app.command()
def main(
    n_estimators: int = typer.Option(0, help="0 = los mismos que la versión champion"),
    max_depth: int = typer.Option(0, help="0 = los mismos que la versión champion"),
    registrar: bool = typer.Option(True, help="Registrar la versión con el alias challenger"),
):
    """Reentrena el Random Forest de P2 con enero + 1-21 de julio y lo registra."""
    client = MlflowClient()
    n_champion, d_champion, version_champion = _hiperparametros_champion(client)
    n_estimators = n_estimators or n_champion
    max_depth = max_depth or d_champion

    referencia, actual, manifiesto = _leer_conjuntos(MONITOREO_DATA_DIR)
    train, evaluacion = conjuntos_v2(referencia, actual)
    X_train, y_train = train[P2_FEATURES], train[P2_OBJETIVO]
    X_eval, y_eval = evaluacion[P2_FEATURES], evaluacion[P2_OBJETIVO]
    n_julio = int((actual["ventana"] == VENTANA_REENTRENAMIENTO).sum())
    logger.info(
        f"Train: {len(train):,} filas ({len(referencia):,} de enero + {n_julio:,} de julio) | "
        f"Evaluación: {len(evaluacion):,} filas"
    )

    config_corrida = {
        "modelo": "RandomForestClassifier",
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "class_weight": "balanced",
        "semilla": SEED,
        "umbral_decision": UMBRAL,
        "features": ",".join(P2_FEATURES),
        "periodo_entrenamiento": f"{REF_INICIO}..{REF_FIN} + {ACTUAL_INICIO}..(<{ACTUAL_CORTE_EVAL})",
        "periodo_evaluacion": f"{ACTUAL_CORTE_EVAL}..{ACTUAL_FIN}",
        "n_train": len(train),
        "n_train_referencia": len(referencia),
        "n_train_actual": n_julio,
        "n_eval": len(evaluacion),
        "sha256_referencia": manifiesto["referencia"]["sha256"][:12],
        "sha256_actual": manifiesto["actual"]["sha256"][:12],
        "version_base": version_champion,
    }

    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=NOMBRE_CORRIDA) as corrida:
        mlflow.log_params(config_corrida)
        mlflow.set_tags(
            {**_etiquetas("random_forest", "retrain"), "tipo_version": "reentrenada_continuous_training"}
        )

        modelo = _random_forest(n_estimators, max_depth)
        modelo.fit(X_train, y_train)
        score = modelo.predict_proba(X_eval)[:, 1]
        pred = (score >= UMBRAL).astype(int)
        metricas = _metricas(y_eval, pred, score, PREFIJO_EVAL)
        mlflow.log_metrics(metricas)

        fuente = str(MONITOREO_DATA_DIR)
        mlflow.log_input(
            mlflow.data.from_pandas(train, source=fuente, name="p2_v2_train_ene_jul"),
            context="training",
        )
        mlflow.log_input(
            mlflow.data.from_pandas(evaluacion, source=fuente, name="p2_actual_evaluacion"),
            context="evaluation",
        )
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            _registrar_graficas(y_eval, pred, score, NOMBRE_CORRIDA, tmp)
            (tmp / "config_corrida.json").write_text(
                json.dumps(config_corrida, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            mlflow.log_artifact(str(tmp / "config_corrida.json"), "config")
        mlflow.log_artifact(str(MONITOREO_DATA_DIR / ARCHIVO_MANIFIESTO), "data")
        _registrar_modelo(modelo, X_train)
        run_id = corrida.info.run_id

    logger.success(f"{NOMBRE_CORRIDA}: " + ", ".join(f"{k}={v:.3f}" for k, v in metricas.items()))
    if not registrar:
        return

    linaje = {
        "experimento": MLFLOW_EXPERIMENT,
        "run_id": run_id,
        "run_name": NOMBRE_CORRIDA,
        "git_commit": _etiquetas("", "")["git_commit"],
        "tipo_version": "reentrenada_continuous_training",
        "version_base": version_champion,
        "datos_entrenamiento": config_corrida["periodo_entrenamiento"],
        "n_estimators": str(n_estimators),
        "max_depth": str(max_depth),
        "umbral_decision": str(UMBRAL),
        **{k: f"{v:.4f}" for k, v in metricas.items()},
    }
    descripcion = (
        f"Random Forest (n_estimators={n_estimators}, max_depth={max_depth}) reentrenado con "
        f"enero 2025 + 1-21 de julio 2025 (Actividad 10, Continuous Training). Mismos "
        f"hiperparámetros que la versión {version_champion}; solo cambian los datos. En la "
        f"ventana de evaluación 22-31 de julio: recall {metricas[f'{PREFIJO_EVAL}_recall']:.4f}, "
        f"precision {metricas[f'{PREFIJO_EVAL}_precision']:.4f}, "
        f"PR-AUC {metricas[f'{PREFIJO_EVAL}_pr_auc']:.4f}."
    )
    version = _registrar_version(client, run_id, linaje, descripcion)
    logger.success(f"{NOMBRE_MODELO} v{version} registrado con alias @{ALIAS_RETADOR}")

    # Verificación: se carga por alias y acepta las mismas columnas que la versión en producción
    cargado = mlflow.sklearn.load_model(f"models:/{NOMBRE_MODELO}@{ALIAS_RETADOR}")
    assert list(cargado.feature_names_in_) == P2_FEATURES
    logger.success(
        f"Carga verificada: models:/{NOMBRE_MODELO}@{ALIAS_RETADOR} -> "
        f"{type(cargado).__name__} con {cargado.n_estimators} árboles y profundidad {cargado.max_depth}"
    )


if __name__ == "__main__":
    app()

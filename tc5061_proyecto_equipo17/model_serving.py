"""Carga e inferencia del modelo P2 registrado en MLflow."""

import os
from functools import lru_cache
from pathlib import Path

import mlflow.pyfunc

MODEL_NAME = "mibici-equipo17-p2-riesgo"
MODEL_VERSION = "1"

REGISTRY_URI = f"models:/{MODEL_NAME}/{MODEL_VERSION}"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_MODEL_PATH = PROJECT_ROOT / "artifacts" / "modelo_p2"

FEATURES = (
    "hora_dia",
    "dia_semana",
    "fin_de_semana",
    "capacidad",
    "pct_ocupacion",
    "riesgo_ahora",
    "neto",
    "neto_ult_3h",
)


def get_model_uri():
    """Obtiene el origen del modelo.

    Prioridad:
    1. Variable de entorno P2_MODEL_URI.
    2. Modelo exportado localmente.
    """
    configured_uri = os.getenv("P2_MODEL_URI")

    if configured_uri:
        return configured_uri

    if not (LOCAL_MODEL_PATH / "MLmodel").is_file():
        raise FileNotFoundError(
            "No se encontro el modelo P2 exportado en "
            f"{LOCAL_MODEL_PATH}. "
            "Exporta el modelo desde MLflow Registry o "
            "configura P2_MODEL_URI."
        )

    return str(LOCAL_MODEL_PATH)


@lru_cache(maxsize=1)
def load_p2_model():
    """Carga el modelo una sola vez por proceso."""
    return mlflow.pyfunc.load_model(get_model_uri())


def predict_p2(dataframe):
    """Genera predicciones utilizando las variables de P2."""
    missing = set(FEATURES) - set(dataframe.columns)

    if missing:
        raise ValueError(
            f"Faltan variables requeridas: {sorted(missing)}"
        )

    ordered = dataframe.loc[:, list(FEATURES)]
    model = load_p2_model()

    return model.predict(ordered)

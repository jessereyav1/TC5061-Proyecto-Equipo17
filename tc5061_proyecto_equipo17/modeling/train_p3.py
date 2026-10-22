"""P3: clustering de estaciones MiBici con K-means, registrado en MLflow."""

import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

from loguru import logger
import matplotlib

matplotlib.use("Agg")  # dibuja a archivo, sin abrir ventanas
import matplotlib.pyplot as plt
import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
import typer

from tc5061_proyecto_equipo17.config import (
    EXTERNAL_DATA_DIR,
    FECHA_CORTE_TEST,
    MLFLOW_EXPERIMENT,
    PROCESSED_DATA_DIR,
    SEED,
)
from tc5061_proyecto_equipo17.features import matriz_clustering_estaciones

app = typer.Typer()

ARCHIVO_NOMENCLATURA = "nomenclatura_2026_09.csv"


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


def _construir_modelo(algoritmo: str, k: int, semilla: int):
    """Devuelve (modelo, familia, tipo de escalado, sufijo para el nombre de la corrida)."""
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=semilla)
    if algoritmo == "kmeans":
        return kmeans, "kmeans", "ninguno", "sin-escalar"
    if algoritmo == "kmeans-escalado":
        return make_pipeline(StandardScaler(), kmeans), "kmeans", "StandardScaler", "escalado"
    if algoritmo == "gmm":
        gmm = GaussianMixture(
            n_components=k, covariance_type="diag", n_init=10, random_state=semilla
        )
        return gmm, "gmm", "ninguno", "diag"
    raise typer.BadParameter(f"Algoritmo desconocido: {algoritmo}")


def _metricas_modelo(modelo, X) -> dict:
    """Métricas propias de cada algoritmo (no comparables entre familias)."""
    if isinstance(modelo, GaussianMixture):
        return {"train_bic": modelo.bic(X), "train_aic": modelo.aic(X)}
    kmeans = modelo[-1] if isinstance(modelo, Pipeline) else modelo
    return {"train_inertia": kmeans.inertia_}


@app.command()
def main(
    archivo: str = "datos_abiertos_2025_01_procesado.parquet",
    archivo_test: str = "datos_abiertos_2025_02_procesado.parquet",
    algoritmo: str = "kmeans",  # kmeans | kmeans-escalado | gmm
    k: int = 4,
    semilla: int = SEED,
    min_viajes: int = 0,
    etapa: str = "baseline",  # baseline | search | candidate
):
    viajes = pd.read_parquet(PROCESSED_DATA_DIR / archivo)
    X = matriz_clustering_estaciones(viajes, min_viajes=min_viajes)

    # Prueba (febrero): mismas columnas que el entrenamiento; si falta una hora, vale 0
    viajes_test = pd.read_parquet(PROCESSED_DATA_DIR / archivo_test)
    X_test = matriz_clustering_estaciones(viajes_test, min_viajes=min_viajes).reindex(
        columns=X.columns, fill_value=0.0
    )

    modelo, familia, escalado, sufijo = _construir_modelo(algoritmo, k, semilla)

    config_corrida = {
        "modelo": familia,
        "k": k,
        "n_init": 10,
        "semilla": semilla,
        "algoritmo": algoritmo,
        "escalado": escalado,
        "min_viajes": min_viajes,
        "archivo_datos": archivo,
        "archivo_test": archivo_test,
        "fecha_corte_test": FECHA_CORTE_TEST,
        "version_datos": _version_datos(),
    }

    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=f"p3-{familia}-k{k}-{sufijo}"):
        etiquetas = modelo.fit_predict(X)

        mlflow.log_params(
            {
                **config_corrida,
                "fecha_min": str(viajes["fecha"].min()),
                "fecha_max": str(viajes["fecha"].max()),
                "n_features": X.shape[1],
            }
        )
        mlflow.set_tags(
            {
                "equipo": "17",
                "pregunta": "P3-clustering",
                "question": "p3",
                "model_family": familia,
                "stage_of_work": etapa,
                "git_commit": _commit_actual(),
            }
        )
        mlflow.log_metrics(
            {
                "train_silhouette": silhouette_score(X, etiquetas),
                "train_davies_bouldin": davies_bouldin_score(X, etiquetas),
                **_metricas_modelo(modelo, X),
                "train_n_estaciones": len(X),
            }
        )

        # Prueba: el modelo ajustado con enero se aplica a febrero (sin reentrenar)
        etiquetas_test = pd.Series(modelo.predict(X_test), index=X_test.index)
        comunes = X.index.intersection(X_test.index)
        etiquetas_train = pd.Series(etiquetas, index=X.index)
        mlflow.log_metrics(
            {
                "test_silhouette": silhouette_score(X_test, etiquetas_test),
                "test_davies_bouldin": davies_bouldin_score(X_test, etiquetas_test),
                "test_n_estaciones": len(X_test),
                # Estabilidad: ¿las estaciones de ambos meses caen en los mismos grupos?
                "test_ari_ene_feb": adjusted_rand_score(
                    etiquetas_train[comunes], etiquetas_test[comunes]
                ),
            }
        )

        # Versión de los datos usados (entrenamiento y prueba)
        mlflow.log_input(
            mlflow.data.from_pandas(
                X, source=str(PROCESSED_DATA_DIR / archivo), name="p3_train"
            ),
            context="training",
        )
        mlflow.log_input(
            mlflow.data.from_pandas(
                X_test, source=str(PROCESSED_DATA_DIR / archivo_test), name="p3_test"
            ),
            context="testing",
        )

        # Modelo con firma (columnas de entrada/salida) y ejemplo de entrada
        mlflow.sklearn.log_model(
            modelo,
            name="modelo",
            signature=infer_signature(X, modelo.predict(X)),
            input_example=X.head(5),
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            X_cl = X.assign(cluster=etiquetas)

            # plots/: perfil horario promedio de cada cluster
            horas = [c for c in X.columns if c.startswith("h")]
            X_cl.groupby("cluster")[horas].mean().T.plot(figsize=(9, 4))
            plt.xlabel("Hora del día")
            plt.ylabel("Proporción de viajes entre semana")
            plt.title(f"Perfil horario por cluster (k={k})")
            plt.tight_layout()
            plt.savefig(tmp / "perfiles_por_cluster.png")
            plt.close()
            mlflow.log_artifact(str(tmp / "perfiles_por_cluster.png"), "plots")

            # plots/: mapa, solo si existe la nomenclatura local
            nomenclatura = EXTERNAL_DATA_DIR / ARCHIVO_NOMENCLATURA
            if nomenclatura.exists():
                est = pd.read_csv(nomenclatura).set_index("id").join(X_cl["cluster"], how="inner")
                fig, ax = plt.subplots(figsize=(7, 7))
                for c, g in est.groupby("cluster"):
                    ax.scatter(g["longitude"], g["latitude"], s=18, label=f"Cluster {c}")
                ax.set_aspect("equal")
                ax.legend()
                ax.set_title(f"Clusters de estaciones (k={k})")
                fig.savefig(tmp / "mapa_clusters.png")
                plt.close(fig)
                mlflow.log_artifact(str(tmp / "mapa_clusters.png"), "plots")

            # data/: asignación estación -> cluster y manifiesto de la versión de datos
            X_cl[["cluster"]].to_csv(tmp / "asignacion_clusters.csv")
            mlflow.log_artifact(str(tmp / "asignacion_clusters.csv"), "data")
            mlflow.log_artifact(str(PROCESSED_DATA_DIR / "manifest.json"), "data")

            # config/: parámetros con los que se ejecutó la corrida
            (tmp / "config_corrida.json").write_text(
                json.dumps(config_corrida, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            mlflow.log_artifact(str(tmp / "config_corrida.json"), "config")

    logger.success("Corrida P3 registrada en MLflow.")


if __name__ == "__main__":
    app()

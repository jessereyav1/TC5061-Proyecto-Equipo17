"""P3: clustering de estaciones MiBici con K-means, registrado en MLflow."""

from pathlib import Path
import subprocess
import tempfile

from loguru import logger
import matplotlib

matplotlib.use("Agg")  # dibuja a archivo, sin abrir ventanas
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
import typer

from tc5061_proyecto_equipo17.config import EXTERNAL_DATA_DIR, PROCESSED_DATA_DIR
from tc5061_proyecto_equipo17.features import matriz_clustering_estaciones

app = typer.Typer()


def _commit_actual() -> str:
    """Hash corto del commit, para rastrear qué código generó la corrida."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "desconocido"


@app.command()
def main(
    archivo: str = "datos_abiertos_2025_01_procesado.parquet",
    k: int = 4,
    semilla: int = 42,
    min_viajes: int = 0,
):
    viajes = pd.read_parquet(PROCESSED_DATA_DIR / archivo)
    X = matriz_clustering_estaciones(viajes, min_viajes=min_viajes)

    mlflow.set_experiment("mibici-equipo17")
    with mlflow.start_run(run_name=f"p3-kmeans-k{k}-sin-escalar"):
        modelo = KMeans(n_clusters=k, n_init=10, random_state=semilla)
        etiquetas = modelo.fit_predict(X)

        mlflow.log_params(
            {
                "modelo": "KMeans",
                "k": k,
                "n_init": 10,
                "semilla": semilla,
                "escalado": "ninguno",
                "min_viajes": min_viajes,
                "archivo_datos": archivo,
                "fecha_min": str(viajes["fecha"].min()),
                "fecha_max": str(viajes["fecha"].max()),
                "n_features": X.shape[1],
            }
        )
        mlflow.set_tags(
            {"equipo": "17", "pregunta": "P3-clustering", "git_commit": _commit_actual()}
        )
        mlflow.log_metrics(
            {
                "train_silhouette": silhouette_score(X, etiquetas),
                "train_davies_bouldin": davies_bouldin_score(X, etiquetas),
                "train_inertia": modelo.inertia_,
                "train_n_estaciones": len(X),
            }
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            X_cl = X.assign(cluster=etiquetas)

            # Artefacto 1: perfil horario promedio de cada cluster
            horas = [c for c in X.columns if c.startswith("h")]
            X_cl.groupby("cluster")[horas].mean().T.plot(figsize=(9, 4))
            plt.xlabel("Hora del día")
            plt.ylabel("Proporción de viajes entre semana")
            plt.title(f"Perfil horario por cluster (k={k})")
            plt.tight_layout()
            plt.savefig(tmp / "perfiles_por_cluster.png")
            plt.close()

            # Artefacto 2: asignación estación -> cluster
            X_cl[["cluster"]].to_csv(tmp / "asignacion_clusters.csv")

            # Artefacto 3 (opcional): mapa, solo si existe la nomenclatura local
            nomenclatura = EXTERNAL_DATA_DIR / "nomenclatura_2026_09.csv"
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

            mlflow.log_artifacts(str(tmp))

    logger.success("Corrida P3 registrada en MLflow.")


if __name__ == "__main__":
    app()
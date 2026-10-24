"""P3: clustering de estaciones MiBici registrado en MLflow.

Dos modos:
- Corrida completa (baseline o candidata): modelo, firma, gráficas y manifiesto.
- Búsqueda de k (--buscar-k): una corrida padre con una hija por cada k, y después
  reentrena el mejor k como corrida completa.
"""

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

# Rango de k útil para decidir el rebalanceo: con menos de 3 tipos de estación no hay
# diferenciación operativa y con más de 6 son difíciles de interpretar y de usar.
K_OPERATIVO = (3, 6)


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
        # El escalador va dentro del Pipeline: se ajusta solo con enero (fit) y en
        # febrero solo se aplica (predict), así no hay fuga de datos.
        return make_pipeline(StandardScaler(), kmeans), "kmeans", "StandardScaler", "escalado"
    if algoritmo == "gmm":
        # Covarianza diagonal: con 21 variables y ~370 estaciones, la completa tendría
        # demasiados parámetros por grupo.
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


def _cargar_matrices(archivo: str, archivo_test: str, variables: str, min_viajes: int):
    """Matrices de variables de enero (entrenamiento) y febrero (prueba)."""
    viajes = pd.read_parquet(PROCESSED_DATA_DIR / archivo)
    X = matriz_clustering_estaciones(viajes, min_viajes=min_viajes)

    # Prueba (febrero): mismas columnas que el entrenamiento; si falta una hora, vale 0
    viajes_test = pd.read_parquet(PROCESSED_DATA_DIR / archivo_test)
    X_test = matriz_clustering_estaciones(viajes_test, min_viajes=min_viajes).reindex(
        columns=X.columns, fill_value=0.0
    )

    # "solo-horas" descarta la proporción de fin de semana (prop_finde)
    if variables == "solo-horas":
        horas_cols = [c for c in X.columns if c.startswith("h")]
        X, X_test = X[horas_cols], X_test[horas_cols]
    elif variables != "completas":
        raise typer.BadParameter(f"Variables desconocidas: {variables}")
    return viajes, X, X_test


def _ajustar_y_evaluar(modelo, X, X_test):
    """Ajusta con enero, evalúa en enero y en febrero. Devuelve (etiquetas_train, métricas)."""
    etiquetas = modelo.fit_predict(X)
    etiquetas_test = pd.Series(modelo.predict(X_test), index=X_test.index)
    comunes = X.index.intersection(X_test.index)
    etiquetas_train = pd.Series(etiquetas, index=X.index)
    metricas = {
        "train_silhouette": silhouette_score(X, etiquetas),
        "train_davies_bouldin": davies_bouldin_score(X, etiquetas),
        **_metricas_modelo(modelo, X),
        "train_n_estaciones": len(X),
        "test_silhouette": silhouette_score(X_test, etiquetas_test),
        "test_davies_bouldin": davies_bouldin_score(X_test, etiquetas_test),
        "test_n_estaciones": len(X_test),
        # Estabilidad: ¿las estaciones de ambos meses caen en los mismos grupos?
        "test_ari_ene_feb": adjusted_rand_score(
            etiquetas_train[comunes], etiquetas_test[comunes]
        ),
    }
    return etiquetas, metricas


def _config_corrida(
    familia, k, semilla, algoritmo, variables, escalado, min_viajes, archivo, archivo_test
) -> dict:
    return {
        "modelo": familia,
        "k": k,
        "n_init": 10,
        "semilla": semilla,
        "algoritmo": algoritmo,
        "variables": variables,
        "escalado": escalado,
        "min_viajes": min_viajes,
        "archivo_datos": archivo,
        "archivo_test": archivo_test,
        "fecha_corte_test": FECHA_CORTE_TEST,
        "version_datos": _version_datos(),
    }


def _etiquetas_mlflow(familia: str, etapa: str) -> dict:
    return {
        "equipo": "17",
        "pregunta": "P3-clustering",
        "question": "p3",
        "model_family": familia,
        "stage_of_work": etapa,
        "git_commit": _commit_actual(),
    }


def _corrida_completa(
    archivo, archivo_test, algoritmo, variables, k, semilla, min_viajes, etapa, nombre_extra=""
):
    """Una corrida con todos los artefactos: modelo con firma, gráficas, datos y configuración."""
    viajes, X, X_test = _cargar_matrices(archivo, archivo_test, variables, min_viajes)
    modelo, familia, escalado, sufijo = _construir_modelo(algoritmo, k, semilla)
    if variables == "solo-horas":
        sufijo = f"{sufijo}-solo-horas"
    if nombre_extra:
        sufijo = f"{sufijo}-{nombre_extra}"
    config_corrida = _config_corrida(
        familia, k, semilla, algoritmo, variables, escalado, min_viajes, archivo, archivo_test
    )

    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=f"p3-{familia}-k{k}-{sufijo}"):
        etiquetas, metricas = _ajustar_y_evaluar(modelo, X, X_test)

        mlflow.log_params(
            {
                **config_corrida,
                "fecha_min": str(viajes["fecha"].min()),
                "fecha_max": str(viajes["fecha"].max()),
                "n_features": X.shape[1],
            }
        )
        mlflow.set_tags(_etiquetas_mlflow(familia, etapa))
        mlflow.log_metrics(metricas)

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


def _busqueda_k(
    archivo, archivo_test, algoritmo, variables, semilla, min_viajes, k_min, k_max
) -> int:
    """Búsqueda de k con nested runs: un padre con el resumen y una hija por cada k.

    Las hijas son ligeras (parámetros y métricas, sin modelo ni gráficas). El padre guarda
    la curva de métricas contra k y el k elegido. Devuelve el mejor k.
    """
    if k_min < 2 or k_max < k_min:
        raise typer.BadParameter("Se requiere 2 <= k_min <= k_max")

    _, X, X_test = _cargar_matrices(archivo, archivo_test, variables, min_viajes)
    _, familia, escalado, _ = _construir_modelo(algoritmo, k_min, semilla)
    filas = []

    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=f"p3-busqueda-k-{algoritmo}-{variables}"):
        mlflow.log_params(
            {
                **_config_corrida(
                    familia, f"{k_min}-{k_max}", semilla, algoritmo, variables, escalado,
                    min_viajes, archivo, archivo_test,
                ),
                "k_min": k_min,
                "k_max": k_max,
                "k_operativo": f"{K_OPERATIVO[0]}-{K_OPERATIVO[1]}",
            }
        )
        mlflow.set_tags(_etiquetas_mlflow(familia, "search"))

        for k in range(k_min, k_max + 1):
            modelo, _, _, _ = _construir_modelo(algoritmo, k, semilla)
            with mlflow.start_run(run_name=f"p3-{familia}-k{k}-busqueda", nested=True):
                _, metricas = _ajustar_y_evaluar(modelo, X, X_test)
                mlflow.log_params(
                    {"k": k, "algoritmo": algoritmo, "variables": variables,
                     "n_features": X.shape[1]}
                )
                mlflow.set_tags(_etiquetas_mlflow(familia, "search"))
                mlflow.log_metrics(metricas)
            filas.append({"k": k, **metricas})
            logger.info(
                f"k={k}: test_silhouette={metricas['test_silhouette']:.3f}, "
                f"ARI={metricas['test_ari_ene_feb']:.3f}"
            )

        resumen = pd.DataFrame(filas).set_index("k")

        # Criterio: mayor silhouette en prueba dentro del rango operativo de k
        operativos = resumen.loc[K_OPERATIVO[0]:K_OPERATIVO[1]]
        if operativos.empty:
            operativos = resumen
        mejor_k = int(operativos["test_silhouette"].idxmax())
        mlflow.log_param("mejor_k", mejor_k)
        mlflow.log_param(
            "criterio_mejor_k",
            f"max test_silhouette con k en [{K_OPERATIVO[0]}, {K_OPERATIVO[1]}]",
        )
        mlflow.log_metrics(
            {
                "mejor_k_test_silhouette": float(resumen.loc[mejor_k, "test_silhouette"]),
                "mejor_k_test_ari_ene_feb": float(resumen.loc[mejor_k, "test_ari_ene_feb"]),
            }
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
            resumen[["train_silhouette", "test_silhouette"]].plot(ax=ax1, marker="o")
            ax1.axvline(mejor_k, color="gray", linestyle="--", label=f"mejor k = {mejor_k}")
            ax1.set_xlabel("k")
            ax1.set_ylabel("Silhouette")
            ax1.set_title("Separación de grupos (más alto es mejor)")
            ax1.legend()
            resumen["test_ari_ene_feb"].plot(ax=ax2, marker="o", color="tab:green")
            ax2.axvline(mejor_k, color="gray", linestyle="--")
            ax2.set_xlabel("k")
            ax2.set_ylabel("ARI enero-febrero")
            ax2.set_title("Estabilidad entre meses (más alto es mejor)")
            fig.tight_layout()
            fig.savefig(tmp / "curva_k.png")
            plt.close(fig)
            mlflow.log_artifact(str(tmp / "curva_k.png"), "plots")

            resumen.to_csv(tmp / "resumen_busqueda.csv")
            mlflow.log_artifact(str(tmp / "resumen_busqueda.csv"), "data")
            mlflow.log_artifact(str(PROCESSED_DATA_DIR / "manifest.json"), "data")

    logger.info(f"Mejor k de la búsqueda: {mejor_k}")
    return mejor_k


@app.command()
def main(
    archivo: str = "datos_abiertos_2025_01_procesado.parquet",
    archivo_test: str = "datos_abiertos_2025_02_procesado.parquet",
    algoritmo: str = "kmeans",  # kmeans | kmeans-escalado | gmm
    variables: str = "completas",  # completas | solo-horas
    k: int = 4,
    semilla: int = SEED,
    min_viajes: int = 0,
    etapa: str = "baseline",  # baseline | search | candidate
    buscar_k: bool = False,
    k_min: int = 2,
    k_max: int = 8,
    reentrenar_mejor: bool = True,
):
    if buscar_k:
        mejor_k = _busqueda_k(
            archivo, archivo_test, algoritmo, variables, semilla, min_viajes, k_min, k_max
        )
        if reentrenar_mejor:
            # El padre ya se cerró: la corrida completa del mejor k queda fuera del anidado
            _corrida_completa(
                archivo, archivo_test, algoritmo, variables, mejor_k, semilla, min_viajes,
                etapa="candidate", nombre_extra="mejor-k",
            )
    else:
        _corrida_completa(
            archivo, archivo_test, algoritmo, variables, k, semilla, min_viajes, etapa
        )
    logger.success("Corrida P3 registrada en MLflow.")


if __name__ == "__main__":
    app()

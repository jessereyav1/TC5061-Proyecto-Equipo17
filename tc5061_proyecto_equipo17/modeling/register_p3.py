"""P3: registra el modelo elegido en el Model Registry y lo lleva a producción (alias champion).

Por defecto toma la corrida candidata más reciente que resultó de la búsqueda de k
(nombre terminado en "-mejor-k"). También se puede indicar una corrida con --run-id.
"""

from loguru import logger
import mlflow
from mlflow import MlflowClient
import mlflow.sklearn
import typer

from tc5061_proyecto_equipo17.config import MLFLOW_EXPERIMENT

app = typer.Typer()

NOMBRE_MODELO = "mibici-equipo17-p3-grupos"
ALIAS_PRODUCCION = "champion"

DESCRIPCION_MODELO = (
    "P3 - Clustering de estaciones MiBici. K-means sobre el perfil horario de salidas "
    "entre semana y la proporción de viajes en fin de semana de cada estación. Agrupa las "
    "estaciones en tipos de uso para apoyar el rebalanceo de bicicletas. Se entrena con "
    "enero de 2025 y se evalúa con febrero de 2025."
)


def _buscar_corrida(client: MlflowClient, experimento_id: str, run_id: str | None):
    """Devuelve la corrida a registrar: la indicada o la última del mejor k."""
    if run_id:
        return client.get_run(run_id)
    corridas = client.search_runs(
        [experimento_id],
        filter_string=(
            "tags.question = 'p3' AND tags.stage_of_work = 'candidate' "
            "AND attributes.run_name LIKE '%-mejor-k'"
        ),
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not corridas:
        raise typer.BadParameter(
            "No hay corridas '-mejor-k'. Ejecuta antes: train_p3 --buscar-k (o usa --run-id)."
        )
    return corridas[0]


@app.command()
def main(run_id: str = typer.Option(None, help="Corrida a registrar (por defecto, la del mejor k)")):
    client = MlflowClient()
    experimento = client.get_experiment_by_name(MLFLOW_EXPERIMENT)
    if experimento is None:
        raise typer.BadParameter(f"No existe el experimento {MLFLOW_EXPERIMENT}")

    corrida = _buscar_corrida(client, experimento.experiment_id, run_id)
    logger.info(f"Corrida elegida: {corrida.info.run_name} ({corrida.info.run_id})")

    # El modelo guardado con log_model es un "logged model" ligado a la corrida
    modelos = mlflow.search_logged_models(
        experiment_ids=[experimento.experiment_id],
        filter_string=f"source_run_id = '{corrida.info.run_id}'",
        output_format="list",
    )
    if not modelos:
        raise RuntimeError("La corrida elegida no tiene un modelo guardado.")

    # Linaje: corrida de origen (automática), commit y versión de datos (como etiquetas)
    linaje = {
        "experimento": MLFLOW_EXPERIMENT,
        "run_id": corrida.info.run_id,
        "git_commit": corrida.data.tags.get("git_commit", "desconocido"),
        "version_datos": corrida.data.params.get("version_datos", "desconocida"),
        "k": corrida.data.params.get("k", ""),
        "algoritmo": corrida.data.params.get("algoritmo", ""),
        "test_silhouette": f"{corrida.data.metrics.get('test_silhouette', float('nan')):.4f}",
        "test_ari_ene_feb": f"{corrida.data.metrics.get('test_ari_ene_feb', float('nan')):.4f}",
    }

    version = mlflow.register_model(f"models:/{modelos[0].model_id}", NOMBRE_MODELO, tags=linaje)
    client.update_registered_model(NOMBRE_MODELO, description=DESCRIPCION_MODELO)
    client.update_model_version(
        NOMBRE_MODELO,
        version.version,
        description=(
            f"K-means con k={linaje['k']} elegido de la búsqueda de k "
            f"(silhouette en prueba {linaje['test_silhouette']}, "
            f"estabilidad ene-feb {linaje['test_ari_ene_feb']})."
        ),
    )

    # Producción mediante alias: se puede cambiar de versión sin cambiar quien consume el modelo
    client.set_registered_model_alias(NOMBRE_MODELO, ALIAS_PRODUCCION, version.version)
    logger.success(f"{NOMBRE_MODELO} v{version.version} registrado con alias @{ALIAS_PRODUCCION}")

    # Verificación: cargar el modelo por nombre y alias
    uri = f"models:/{NOMBRE_MODELO}@{ALIAS_PRODUCCION}"
    cargado = mlflow.sklearn.load_model(uri)
    logger.success(f"Carga verificada: {uri} -> {type(cargado).__name__} con {cargado.n_clusters} grupos")


if __name__ == "__main__":
    app()

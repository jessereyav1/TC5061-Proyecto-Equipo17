"""P2: registra el modelo elegido en el Model Registry y lo lleva a producción (alias champion).

Por defecto toma la búsqueda más reciente (corrida padre "p2-busqueda-rf"), que guarda el
Random Forest ganador reentrenado con todo enero. También se puede indicar una corrida con
--run-id y, de forma opcional, una segunda corrida como alias challenger.
"""

from loguru import logger
import mlflow
from mlflow import MlflowClient
import mlflow.sklearn
import typer

from tc5061_proyecto_equipo17.config import MLFLOW_EXPERIMENT

app = typer.Typer()

NOMBRE_MODELO = "mibici-equipo17-p2-riesgo"
ALIAS_PRODUCCION = "champion"
ALIAS_RETADOR = "challenger"

DESCRIPCION_MODELO = (
    "P2 - Clasificación del riesgo de que una estación MiBici se quede sin bicis o sin "
    "anclajes la siguiente hora (proxy: ocupación estimada menor al 10% o mayor al 90%). "
    "Random Forest con pesos balanceados sobre hora, día, capacidad, ocupación estimada, "
    "riesgo actual y flujo neto. Devuelve la probabilidad de riesgo para que el equipo de "
    "rebalanceo atienda primero las estaciones más probables. Se elige por recall, porque "
    "es peor no llegar a una estación vacía que hacer una visita de más. Entrena con enero "
    "de 2025 (validación 25-31 de enero) y se evalúa con febrero de 2025."
)


def _buscar_corrida(client: MlflowClient, experimento_id: str, run_id: str | None):
    """Devuelve la corrida a registrar: la indicada o la última búsqueda terminada."""
    if run_id:
        return client.get_run(run_id)
    corridas = client.search_runs(
        [experimento_id],
        filter_string=(
            "tags.question = 'p2' AND tags.stage_of_work = 'search' "
            "AND attributes.run_name = 'p2-busqueda-rf' AND attributes.status = 'FINISHED'"
        ),
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not corridas:
        raise typer.BadParameter(
            "No hay corridas 'p2-busqueda-rf'. Ejecuta antes la búsqueda (o usa --run-id)."
        )
    return corridas[0]


def _registrar_version(client: MlflowClient, experimento_id: str, corrida, alias: str) -> str:
    """Registra el modelo de la corrida con su linaje, le pone el alias y devuelve la versión."""
    # El modelo guardado con log_model es un "logged model" ligado a la corrida
    modelos = mlflow.search_logged_models(
        experiment_ids=[experimento_id],
        filter_string=f"source_run_id = '{corrida.info.run_id}'",
        output_format="list",
    )
    if not modelos:
        raise RuntimeError(f"La corrida {corrida.info.run_name} no tiene un modelo guardado.")

    params, metricas = corrida.data.params, corrida.data.metrics
    # Linaje: corrida de origen (automática), commit y versión de datos (como etiquetas)
    linaje = {
        "experimento": MLFLOW_EXPERIMENT,
        "run_id": corrida.info.run_id,
        "run_name": corrida.info.run_name,
        "git_commit": corrida.data.tags.get("git_commit", "desconocido"),
        "version_datos": params.get("version_datos", "desconocida"),
        # En la búsqueda, mlflow run guarda n_estimators=0 y max_depth=0 (los del MLproject);
        # los valores reales del ganador están en mejor_*
        "n_estimators": params.get("mejor_n_estimators", params.get("n_estimators", "")),
        "max_depth": params.get("mejor_max_depth", params.get("max_depth", "")),
        "umbral_decision": params.get("umbral_decision", ""),
        **{
            k: f"{metricas.get(k, float('nan')):.4f}"
            for k in ("test_recall", "test_precision", "test_f1", "test_pr_auc")
        },
    }

    version = mlflow.register_model(f"models:/{modelos[0].model_id}", NOMBRE_MODELO, tags=linaje)
    client.update_model_version(
        NOMBRE_MODELO,
        version.version,
        description=(
            f"Random Forest con n_estimators={linaje['n_estimators']} y "
            f"max_depth={linaje['max_depth']} (corrida {linaje['run_name']}). En prueba: "
            f"recall {linaje['test_recall']}, precision {linaje['test_precision']}, "
            f"PR-AUC {linaje['test_pr_auc']}."
        ),
    )
    client.set_registered_model_alias(NOMBRE_MODELO, alias, version.version)
    logger.success(f"{NOMBRE_MODELO} v{version.version} registrado con alias @{alias}")
    return version.version


@app.command()
def main(
    run_id: str = typer.Option(None, help="Corrida a registrar (por defecto, la última búsqueda)"),
    challenger_run_id: str = typer.Option(None, help="Corrida opcional para el alias challenger"),
):
    client = MlflowClient()
    experimento = client.get_experiment_by_name(MLFLOW_EXPERIMENT)
    if experimento is None:
        raise typer.BadParameter(f"No existe el experimento {MLFLOW_EXPERIMENT}")

    corrida = _buscar_corrida(client, experimento.experiment_id, run_id)
    logger.info(f"Corrida elegida: {corrida.info.run_name} ({corrida.info.run_id})")
    _registrar_version(client, experimento.experiment_id, corrida, ALIAS_PRODUCCION)
    client.update_registered_model(NOMBRE_MODELO, description=DESCRIPCION_MODELO)

    # Producción mediante alias: se puede cambiar de versión sin cambiar quien consume el modelo
    if challenger_run_id:
        retador = client.get_run(challenger_run_id)
        logger.info(f"Challenger: {retador.info.run_name} ({retador.info.run_id})")
        _registrar_version(client, experimento.experiment_id, retador, ALIAS_RETADOR)

    # Verificación: cargar el modelo por nombre y alias
    uri = f"models:/{NOMBRE_MODELO}@{ALIAS_PRODUCCION}"
    cargado = mlflow.sklearn.load_model(uri)
    logger.success(
        f"Carga verificada: {uri} -> {type(cargado).__name__} con {cargado.n_estimators} "
        f"árboles, profundidad {cargado.max_depth} y {cargado.n_features_in_} features"
    )


if __name__ == "__main__":
    app()

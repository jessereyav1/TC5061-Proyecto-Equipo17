from pathlib import Path
import pandas as pd
from loguru import logger
from tqdm import tqdm
import typer

from tc5061_proyecto_equipo17.config import PROCESSED_DATA_DIR

# ---- P3: clustering de estaciones ----

PRIMER_DIA_FIN_DE_SEMANA = 5


def perfil_horario_entre_semana(viajes: pd.DataFrame) -> pd.DataFrame:
    """Perfil horario de salidas por estación, de lunes a viernes.

    Para cada estación (origen_id) calcula qué proporción de sus viajes entre semana
    sale en cada hora del día (columnas h00..h23). Cada fila suma 1.

    Por qué sirve para P3: al normalizar por el total de la estación, se comparan
    patrones de uso (por ejemplo, pico de mañana contra pico de tarde) y no el tamaño
    de la estación.

    Solo usa los viajes recibidos; para evitar fuga de información, el llamador debe
    filtrarlos al periodo de entrenamiento.
    """
    entre_semana = viajes[viajes["dia_semana"] < PRIMER_DIA_FIN_DE_SEMANA]
    conteo = (
        entre_semana.groupby(["origen_id", "hora"]).size()
        .unstack(fill_value=0)
        .reindex(columns=range(24), fill_value=0)
    )
    perfil = conteo.div(conteo.sum(axis=1), axis=0)
    perfil.columns = [f"h{h:02d}" for h in perfil.columns]
    return perfil


def proporcion_fin_de_semana(viajes: pd.DataFrame) -> pd.Series:
    """Proporción de los viajes de cada estación que ocurren en sábado o domingo.

    Por qué sirve para P3: distingue estaciones de uso laboral (casi nada en fin de
    semana) de las recreativas, sin crear 24 columnas más con pocos datos por estación.
    """
    totales = viajes["origen_id"].value_counts()
    finde = viajes.loc[
        viajes["dia_semana"] >= PRIMER_DIA_FIN_DE_SEMANA, "origen_id"
    ].value_counts()
    return (finde / totales).fillna(0.0).rename("prop_finde")


def matriz_clustering_estaciones(viajes: pd.DataFrame, min_viajes: int = 0) -> pd.DataFrame:
    """Matriz de variables por estación para el clustering de P3.

    Combina el perfil horario entre semana con la proporción de fin de semana. Descarta
    las estaciones con menos de `min_viajes` viajes y las columnas constantes (horas sin
    ningún viaje), que no aportan información.
    """
    if min_viajes > 0:
        totales = viajes["origen_id"].value_counts()
        activas = totales[totales >= min_viajes].index
        viajes = viajes[viajes["origen_id"].isin(activas)]

    matriz = pd.concat(
        [perfil_horario_entre_semana(viajes), proporcion_fin_de_semana(viajes)],
        axis=1,
        join="inner",
    )
    return matriz.loc[:, matriz.std() > 0]


app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "features.csv",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Generating features from dataset...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Features generation complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()

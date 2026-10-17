from pathlib import Path

from loguru import logger
import pandas as pd
from tqdm import tqdm
import typer

from tc5061_proyecto_equipo17.config import PROCESSED_DATA_DIR


app = typer.Typer()


def p2_flujo_horario(
    viajes: pd.DataFrame, fecha_inicio: str, fecha_fin: str
) -> pd.DataFrame:
    """Cuenta salidas, llegadas y flujo neto por estación y hora (P2).

    Cada viaje genera dos eventos: una salida en ``origen_id`` a la hora de
    ``inicio_del_viaje`` y una llegada en ``destino_id`` a la hora de
    ``fin_del_viaje``. Las llegadas se ubican por la hora de fin (no por la
    columna ``hora``, que proviene del inicio del viaje).

    Se construye una rejilla completa estación × hora entre ``fecha_inicio``
    (00:00) y ``fecha_fin`` (23:00) y se rellenan con 0 las horas sin
    movimiento, para que las ventanas temporales posteriores no salten horas.

    Pertinencia para P2: el flujo neto (llegadas − salidas) es la base del
    proxy de ocupación con el que se aproxima el riesgo de quedarse sin
    bicicletas o sin anclajes.

    Fuga temporal: cada fila solo agrega eventos ocurridos dentro de esa hora;
    no usa información de horas posteriores.

    Args:
        viajes: Viajes procesados con ``origen_id``, ``destino_id``,
            ``inicio_del_viaje`` y ``fin_del_viaje``.
        fecha_inicio: Primer día del periodo (``"AAAA-MM-DD"``).
        fecha_fin: Último día del periodo, inclusive (``"AAAA-MM-DD"``).

    Returns:
        DataFrame con índice (``estacion_id``, ``hora_ts``) y columnas
        ``salidas``, ``llegadas`` y ``neto``.
    """
    salidas = (
        viajes.groupby(["origen_id", viajes["inicio_del_viaje"].dt.floor("h")])
        .size()
        .rename_axis(["estacion_id", "hora_ts"])
        .rename("salidas")
    )
    llegadas = (
        viajes.groupby(["destino_id", viajes["fin_del_viaje"].dt.floor("h")])
        .size()
        .rename_axis(["estacion_id", "hora_ts"])
        .rename("llegadas")
    )

    estaciones = sorted(set(viajes["origen_id"]) | set(viajes["destino_id"]))
    horas = pd.date_range(
        fecha_inicio, pd.Timestamp(fecha_fin) + pd.Timedelta(hours=23), freq="h"
    )
    rejilla = pd.MultiIndex.from_product(
        [estaciones, horas], names=["estacion_id", "hora_ts"]
    )

    flujo = (
        pd.concat([salidas, llegadas], axis=1)
        .fillna(0)
        .reindex(rejilla, fill_value=0)
        .astype(int)
    )
    flujo["neto"] = flujo["llegadas"] - flujo["salidas"]
    return flujo


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

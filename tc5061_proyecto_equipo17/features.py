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


def p2_ocupacion_estimada(
    flujo: pd.DataFrame, estaciones: pd.DataFrame, frac_inicial: float = 0.5
) -> pd.DataFrame:
    """Estima la ocupación de cada estación por hora con un proxy (P2).

    Como MiBici no publica el inventario de bicicletas, la ocupación se
    aproxima así: cada día inicia con ``frac_inicial`` de la capacidad y se le
    suma el flujo neto acumulado del día (llegadas − salidas). El acumulado se
    reinicia a las 00:00, lo que coincide con la ventana de baja operación
    nocturna en la que el sistema se rebalancea.

    Se excluyen, con aviso en el log, las estaciones sin capacidad válida
    (faltante o igual a 0) para evitar divisiones entre cero.

    Pertinencia para P2: el porcentaje de ocupación es la variable con la que
    se define el riesgo de quedarse sin bicicletas o sin anclajes.

    Limitaciones: la ocupación estimada puede ser negativa o superar la
    capacidad; se interpreta como que la estación ya se habría vaciado o
    llenado. La capacidad proviene de la nomenclatura de 2026-09 y se asume
    constante en el periodo analizado.

    Fuga temporal: el acumulado en la hora ``t`` solo suma horas del mismo día
    hasta ``t`` inclusive.

    Args:
        flujo: Salida de ``p2_flujo_horario``.
        estaciones: Nomenclatura con columnas ``id`` y ``dpcapacity``.
        frac_inicial: Fracción de la capacidad con la que inicia cada día.

    Returns:
        ``flujo`` con las columnas ``capacidad``, ``neto_acum_dia``,
        ``ocupacion`` y ``pct_ocupacion``.
    """
    capacidad = estaciones.set_index("id")["dpcapacity"]
    flujo = flujo.copy()
    flujo["capacidad"] = flujo.index.get_level_values(
        "estacion_id").map(capacidad)

    invalida = flujo["capacidad"].isna() | (flujo["capacidad"] <= 0)
    if invalida.any():
        n = flujo[invalida].index.get_level_values("estacion_id").nunique()
        logger.warning(f"Se excluyen {n} estaciones sin capacidad válida.")
        flujo = flujo[~invalida].copy()

    est = flujo.index.get_level_values("estacion_id")
    dia = flujo.index.get_level_values("hora_ts").normalize()
    flujo["neto_acum_dia"] = flujo.groupby([est, dia])["neto"].cumsum()
    flujo["ocupacion"] = frac_inicial * \
        flujo["capacidad"] + flujo["neto_acum_dia"]
    flujo["pct_ocupacion"] = flujo["ocupacion"] / flujo["capacidad"]
    return flujo


def p2_etiqueta_riesgo(
    flujo: pd.DataFrame, umbral_bajo: float = 0.10, umbral_alto: float = 0.90
) -> pd.DataFrame:
    """Construye la etiqueta de riesgo de la siguiente hora (P2).

    Una estación está en riesgo en la hora ``t`` si su ocupación estimada es
    menor que ``umbral_bajo`` (riesgo de quedarse sin bicicletas) o mayor que
    ``umbral_alto`` (riesgo de quedarse sin anclajes).

    - ``riesgo_ahora``: riesgo al cierre de la hora ``t``. Es conocido en el
      momento de la predicción y puede usarse como feature.
    - ``riesgo_siguiente_hora``: riesgo en ``t + 1``. Es la **variable
      objetivo**; por definición mira al futuro y nunca debe usarse como
      feature. La última hora de cada estación queda en ``NaN``.

    Pertinencia para P2: permite al equipo de rebalanceo anticipar con una
    hora de margen qué estaciones necesitan atención.

    Args:
        flujo: Salida de ``p2_ocupacion_estimada``.
        umbral_bajo: Fracción de ocupación por debajo de la cual hay riesgo.
        umbral_alto: Fracción de ocupación por encima de la cual hay riesgo.

    Returns:
        ``flujo`` con las columnas ``riesgo_ahora`` y ``riesgo_siguiente_hora``.
    """
    flujo = flujo.copy()
    pct = flujo["pct_ocupacion"]
    flujo["riesgo_ahora"] = ((pct < umbral_bajo) |
                             (pct > umbral_alto)).astype(int)
    flujo["riesgo_siguiente_hora"] = (
        flujo.groupby(level="estacion_id")["riesgo_ahora"].shift(-1)
    )
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

from pathlib import Path
import pandas as pd
from loguru import logger
import pandas as pd
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

P2_FEATURES = [
    "hora_dia",
    "dia_semana",
    "fin_de_semana",
    "capacidad",
    "pct_ocupacion",
    "riesgo_ahora",
    "neto",
    "neto_ult_3h",
]
P2_OBJETIVO = "riesgo_siguiente_hora"


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


def p2_features(flujo: pd.DataFrame) -> pd.DataFrame:
    """Agrega las variables predictoras de P2.

    - ``hora_dia``, ``dia_semana``, ``fin_de_semana``: capturan los picos de
      uso(8: 00 y 18: 00) y la diferencia entre días laborales y fin de semana.
    - ``neto_ult_3h``: suma del flujo neto de la hora actual y las dos
      anteriores en la misma estación; indica si la estación viene vaciándose
      o llenándose.

    Junto con ``capacidad``, ``pct_ocupacion``, ``riesgo_ahora`` y ``neto``
    forman ``P2_FEATURES``.

    Fuga temporal: todas las variables se conocen al cierre de la hora ``t``.
    La ventana de ``neto_ult_3h`` solo mira hacia atrás y se calcula por
    estación para no mezclar series.

    Args:
        flujo: Salida de ``p2_etiqueta_riesgo``.

    Returns:
        ``flujo`` con las columnas ``hora_dia``, ``dia_semana``,
        ``fin_de_semana`` y ``neto_ult_3h``.
    """
    flujo = flujo.copy()
    hora = flujo.index.get_level_values("hora_ts")
    flujo["hora_dia"] = hora.hour
    flujo["dia_semana"] = hora.dayofweek
    flujo["fin_de_semana"] = (flujo["dia_semana"] >= 5).astype(int)
    flujo["neto_ult_3h"] = flujo.groupby(level="estacion_id")["neto"].transform(
        lambda s: s.rolling(3, min_periods=1).sum()
    )
    return flujo


def p2_construir_dataset(
    viajes: pd.DataFrame,
    estaciones: pd.DataFrame,
    fecha_inicio: str,
    fecha_fin: str,
) -> pd.DataFrame:
    """Construye el dataset completo de P2: features y etiqueta.

    Encadena ``p2_flujo_horario`` → ``p2_ocupacion_estimada`` →
    ``p2_etiqueta_riesgo`` → ``p2_features`` y descarta las filas sin
    etiqueta(la última hora de cada estación).

    Args:
        viajes: Viajes procesados.
        estaciones: Nomenclatura con ``id`` y ``dpcapacity``.
        fecha_inicio: Primer día del periodo(``"AAAA-MM-DD"``).
        fecha_fin: Último día del periodo, inclusive(``"AAAA-MM-DD"``).

    Returns:
        DataFrame con índice(``estacion_id``, ``hora_ts``), las columnas de
        ``P2_FEATURES`` y ``P2_OBJETIVO`` como entero.
    """
    flujo = p2_flujo_horario(viajes, fecha_inicio, fecha_fin)
    flujo = p2_ocupacion_estimada(flujo, estaciones)
    flujo = p2_etiqueta_riesgo(flujo)
    flujo = p2_features(flujo)
    flujo = flujo.dropna(subset=[P2_OBJETIVO])
    flujo[P2_OBJETIVO] = flujo[P2_OBJETIVO].astype(int)
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

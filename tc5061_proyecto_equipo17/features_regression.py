from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
import typer
from loguru import logger

from tc5061_proyecto_equipo17.config import PROCESSED_DATA_DIR


app = typer.Typer(
    help="Módulo de Ingeniería de Características para el Modelo de Regresión - MiBici Equipo 17"
)


class FeatureRegressionEngineer:
    """
    Clase orientada a objetos responsable de construir características temporales/de series de tiempo
    (lags, rolling statistics, encoding cíclico). Nota: Se elimina el escalado para evitar fugas de datos
    (data leakage) y porque los modelos basados en árboles (Random Forest) no lo requieren.
    """

    def __init__(
        self,
        processed_dir: Path = PROCESSED_DATA_DIR,
    ):
        self.processed_dir = processed_dir

    def cargar_dataset_horario(self, input_filename: str = "dataset_regresion_horario.parquet") -> pd.DataFrame:
        """Carga el dataset de regresión horario generado por dataset_regression.py."""
        ruta_dataset = self.processed_dir / input_filename
        if not ruta_dataset.is_file():
            logger.error(f"El archivo {ruta_dataset} no existe. Ejecute primero dataset_regression.py.")
            raise FileNotFoundError(f"Archivo no encontrado: {ruta_dataset}")

        logger.info(f"Cargando dataset base de regresión: {ruta_dataset.name}")
        df = pd.read_parquet(ruta_dataset)
        return df.sort_values(by=["origen_id", "fecha_hora_inicio"]).reset_index(drop=True)

    def construir_y_guardar_features(
        self,
        df: pd.DataFrame,
        output_filename: str = "features_regresion.parquet",
    ) -> Tuple[pd.DataFrame, Path]:
        """
        Genera variables explicativas de series de tiempo (lags, rolling features, ciclicidad)
        por estación (origen_id) y guarda el resultado sin aplicar escalado.
        """
        logger.info("Generando características de series de tiempo (lags, rolling statistics y encoding cíclico)...")
        df_feat = df.copy()

        # 1. Variables de Rezago (Lags de demanda) por estación
        lags = [1, 2, 24, 168]  # 1h, 2h, 1 día antes, 1 semana antes
        for lag in lags:
            df_feat[f"lag_demanda_{lag}h"] = (
                df_feat.groupby("origen_id")["demanda_viajes"].shift(lag)
            )

        # 2. Estadísticas Móviles (Rolling Windows)
        ventanas = [3, 6]
        for v in ventanas:
            df_feat[f"rolling_mean_{v}h"] = (
                df_feat.groupby("origen_id")["lag_demanda_1h"]
                .transform(lambda x: x.rolling(v, min_periods=1).mean())
            )
            df_feat[f"rolling_std_{v}h"] = (
                df_feat.groupby("origen_id")["lag_demanda_1h"]
                .transform(lambda x: x.rolling(v, min_periods=1).std().fillna(0))
            )

        # 3. Transformación Cíclica (Seno/Coseno) para hora y mes
        df_feat["hora_sin"] = np.sin(2 * np.pi * df_feat["hora"] / 24.0)
        df_feat["hora_cos"] = np.cos(2 * np.pi * df_feat["hora"] / 24.0)
        df_feat["mes_sin"] = np.sin(2 * np.pi * df_feat["mes"] / 12.0)
        df_feat["mes_cos"] = np.cos(2 * np.pi * df_feat["mes"] / 12.0)

        # Eliminar registros iniciales con NaNs producidos por los lags mayores (e.g. 168h)
        filas_originales = len(df_feat)
        df_feat = df_feat.dropna().reset_index(drop=True)
        logger.info(f"Registros después de remoción de NaNs por lags: {len(df_feat):,} (descartados: {filas_originales - len(df_feat):,})")

        # Guardar dataset con features procesadas
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.processed_dir / output_filename
        df_feat.to_parquet(output_path, engine="pyarrow", index=False, compression="snappy")

        logger.success(f"Dataset de features guardado en: {output_path.name}")

        return df_feat, output_path


@app.command()
def main(
    input_filename: str = "dataset_regresion_horario.parquet",
    output_filename: str = "features_regresion.parquet",
):
    """
    Ejecuta el pipeline de ingeniería de características para el modelo de regresión.
    """
    logger.info("Iniciando pipeline de Ingeniería de Características...")
    engineer = FeatureRegressionEngineer()

    # 1. Cargar datos horarios
    df_base = engineer.cargar_dataset_horario(input_filename=input_filename)

    # 2. Generar features y guardar
    engineer.construir_y_guardar_features(
        df=df_base,
        output_filename=output_filename,
    )


if __name__ == "__main__":
    app()

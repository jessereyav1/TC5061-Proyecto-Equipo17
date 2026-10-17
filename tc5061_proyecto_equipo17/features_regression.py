import joblib
from pathlib import Path
from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
import typer
from loguru import logger
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from tc5061_proyecto_equipo17.config import PROCESSED_DATA_DIR, MODELS_DIR


app = typer.Typer(
    help="Módulo de Ingeniería de Características y Escalado para el Modelo de Regresión - MiBici Equipo 17"
)


class FeatureRegressionEngineer:
    """
    Clase orientada a objetos responsable de construir características temporales/de series de tiempo
    (lags, rolling statistics, encoding cíclico) y aplicar escalado a las variables continuas.
    """

    def __init__(
        self,
        processed_dir: Path = PROCESSED_DATA_DIR,
        models_dir: Path = MODELS_DIR,
        scaler_type: str = "standard",
    ):
        self.processed_dir = processed_dir
        self.models_dir = models_dir
        self.scaler_type = scaler_type
        self.scaler = self._obtener_escalador(scaler_type)

    def _obtener_escalador(self, tipo: str):
        """Retorna la instancia del escalador seleccionado."""
        if tipo == "standard":
            return StandardScaler()
        elif tipo == "minmax":
            return MinMaxScaler()
        elif tipo == "robust":
            return RobustScaler()
        else:
            raise ValueError(f"Tipo de escalador '{tipo}' no soportado. Usa: standard, minmax o robust.")

    def cargar_dataset_horario(self, input_filename: str = "dataset_regresion_horario.parquet") -> pd.DataFrame:
        """Carga el dataset de regresión horario generado por dataset_regression.py."""
        ruta_dataset = self.processed_dir / input_filename
        if not ruta_dataset.is_file():
            logger.error(f"El archivo {ruta_dataset} no existe. Ejecute primero dataset_regression.py.")
            raise FileNotFoundError(f"Archivo no encontrado: {ruta_dataset}")

        logger.info(f"Cargando dataset base de regresión: {ruta_dataset.name}")
        df = pd.read_parquet(ruta_dataset)
        return df.sort_values(by=["origen_id", "fecha_hora_inicio"]).reset_index(drop=True)

    def construir_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Genera variables explicativas de series de tiempo (lags, rolling features, ciclicidad)
        por estación (origen_id).
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

        return df_feat

    def escalar_y_guardar(
        self,
        df_feat: pd.DataFrame,
        features_num: List[str],
        output_filename: str = "features_regresion_escalados.parquet",
        scaler_filename: str = "scaler_regresion.joblib",
    ) -> Tuple[pd.DataFrame, Path, Path]:
        """
        Ajusta el escalador sobre las características numéricas seleccionadas,
        transforma los datos y guarda el objeto escalador para la fase de inferencia.
        """
        logger.info(f"Escalando {len(features_num)} columnas numéricas con {self.scaler_type.capitalize()}Scaler...")
        
        df_scaled = df_feat.copy()
        
        # Ajustar y transformar variables numéricas
        df_scaled[features_num] = self.scaler.fit_transform(df_scaled[features_num])

        # Guardar dataset con features procesadas y escaladas
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.processed_dir / output_filename
        df_scaled.to_parquet(output_path, engine="pyarrow", index=False, compression="snappy")

        # Guardar objeto escalador
        self.models_dir.mkdir(parents=True, exist_ok=True)
        scaler_path = self.models_dir / scaler_filename
        joblib.dump(self.scaler, scaler_path)

        logger.success(f"Dataset de features escalado guardado en: {output_path.name}")
        logger.success(f"Objeto Escalador guardado en: {scaler_path.name}")

        return df_scaled, output_path, scaler_path


@app.command()
def main(
    input_filename: str = "dataset_regresion_horario.parquet",
    output_filename: str = "features_regresion_escalados.parquet",
    scaler_type: str = "standard",
):
    """
    Ejecuta el pipeline de ingeniería de características y escalado numérico para el modelo de regresión.
    """
    logger.info("Iniciando pipeline de Ingeniería de Características y Escalado...")
    engineer = FeatureRegressionEngineer(scaler_type=scaler_type)

    # 1. Cargar datos horarios
    df_base = engineer.cargar_dataset_horario(input_filename=input_filename)

    # 2. Generar lags, rolling stats y transformaciones cíclicas
    df_features = engineer.construir_features(df_base)

    # 3. Definir columnas continuas/numéricas a escalar
    columnas_a_escalar = [
        "lag_demanda_1h", "lag_demanda_2h", "lag_demanda_24h", "lag_demanda_168h",
        "rolling_mean_3h", "rolling_std_3h", "rolling_mean_6h", "rolling_std_6h"
    ]

    # 4. Escalar y exportar artefactos
    engineer.escalar_y_guardar(
        df_feat=df_features,
        features_num=columnas_a_escalar,
        output_filename=output_filename,
    )


if __name__ == "__main__":
    app()

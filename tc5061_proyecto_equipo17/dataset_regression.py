import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import typer
from loguru import logger

from tc5061_proyecto_equipo17.config import PROCESSED_DATA_DIR


app = typer.Typer(
    help="Pipeline exclusivo de preparación de datos para el modelo de regresión - MiBici Equipo 17"
)


class DatasetRegresionProcessor:
    """
    Clase orientada a objetos para consolidar los archivos procesados
    en data/processed/ y estructurar el dataset a nivel horario por estación (Origen_Id)
    para la predicción de demanda de viajes.
    """

    def __init__(self, processed_dir: Path = PROCESSED_DATA_DIR):
        self.processed_dir = processed_dir

    def cargar_datos_procesados(self) -> pd.DataFrame:
        """Carga y concatena todos los archivos Parquet procesados previamente."""
        archivos_parquet = list(self.processed_dir.glob("*_procesado.parquet"))
        
        if not archivos_parquet:
            logger.error(
                f"No se encontraron archivos Parquet procesados en: {self.processed_dir}"
            )
            return pd.DataFrame()

        logger.info(
            f"Consolidando {len(archivos_parquet)} archivos procesados para el modelo de regresión..."
        )
        dfs = [pd.read_parquet(f) for f in archivos_parquet]
        return pd.concat(dfs, ignore_index=True)

    def construir_dataset_regresion(
        self, output_filename: str = "dataset_regresion_horario.parquet"
    ) -> Path:
        """
        Agrupa los datos por estación de origen y hora para calcular
        la variable objetivo (conteo de viajes / demanda de salidas).
        """
        df = self.cargar_datos_procesados()
        if df.empty:
            logger.error(
                "No se pudo generar el dataset. Verifica que existan archivos Parquet en data/processed/."
            )
            raise ValueError("Sin datos procesados para construir la regresión.")

        # Asegurar formato datetime
        df["inicio_del_viaje"] = pd.to_datetime(df["inicio_del_viaje"])

        # Truncar la marca temporal a nivel horario
        df["fecha_hora_inicio"] = df["inicio_del_viaje"].dt.floor("h")

        # Agregación espacial y temporal (Variable objetivo: conteo de salidas / viajes)
        logger.info("Agregando transacciones a nivel horario por estación (Origen_Id)...")
        regresion_df = (
            df.groupby(["origen_id", "fecha_hora_inicio"])
            .size()
            .reset_index(name="demanda_viajes")
        )

        # Generar características temporales para el modelo de regresión
        regresion_df["hora"] = regresion_df["fecha_hora_inicio"].dt.hour
        regresion_df["dia"] = regresion_df["fecha_hora_inicio"].dt.day  # Día del mes (1-31)
        regresion_df["dia_semana"] = regresion_df["fecha_hora_inicio"].dt.dayofweek  # 0: Lunes, 6: Domingo
        regresion_df["mes"] = regresion_df["fecha_hora_inicio"].dt.month
        regresion_df["anio"] = regresion_df["fecha_hora_inicio"].dt.year
        regresion_df["es_fin_de_semana"] = regresion_df["dia_semana"].isin([5, 6]).astype(int)

        # Crear directorio si no existe y exportar Parquet
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.processed_dir / output_filename
        
        regresion_df.to_parquet(
            output_path,
            engine="pyarrow",
            index=False,
            compression="snappy",
        )

        logger.success(
            f"Dataset de regresión generado con éxito: {output_path.name} "
            f"({len(regresion_df):,} filas agregadas con características temporales actualizadas)"
        )

        return output_path


@app.command()
def main(
    output_filename: str = "dataset_regresion_horario.parquet",
):
    """
    Ejecuta el pipeline exclusivo para generar el dataset del modelo de regresión.
    """
    logger.info("Iniciando generación de dataset de regresión...")
    procesador = DatasetRegresionProcessor()
    procesador.construir_dataset_regresion(output_filename=output_filename)


if __name__ == "__main__":
    app()

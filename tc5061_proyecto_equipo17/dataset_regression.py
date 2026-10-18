from pathlib import Path
import pandas as pd
import typer
from loguru import logger

from tc5061_proyecto_equipo17.config import PROCESSED_DATA_DIR

app = typer.Typer(
    help="Pipeline optimizado con manejo condicional de fechas (Oct 2024 - Sep 2026) - MiBici Equipo 17"
)

RAW_DATA_DIR = PROCESSED_DATA_DIR.parent / "raw"


class DatasetRegresionProcessor:
    """
    Clase orientada a objetos para procesar de forma incremental los archivos
    con un condicional específico para el formato de fecha de octubre 2024.
    """

    def __init__(self, raw_dir: Path = RAW_DATA_DIR, processed_dir: Path = PROCESSED_DATA_DIR):
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir

    def construir_dataset_regresion(
        self, output_filename: str = "dataset_regresion_horario.parquet"
    ) -> Path:
        """
        Filtra y procesa los archivos mensuales de Oct 2024 a Sep 2026 aplicando
        el parseo de fecha condicional según el archivo.
        """
        todos_csv = list(self.raw_dir.glob("datos_abiertos_*.csv"))
        
        if not todos_csv:
            todos_csv = list(self.raw_dir.glob("*.csv"))

        if not todos_csv:
            logger.error(f"No se encontraron archivos CSV en la ruta: {self.raw_dir}")
            raise FileNotFoundError(f"No hay archivos CSV en {self.raw_dir}")

        # Filtrar exclusivamente el rango de interés: Octubre 2024 (202410) a Septiembre 2026 (202609)
        archivos_csv = []
        for f in todos_csv:
            partes = f.stem.split("_")
            if len(partes) >= 3:
                try:
                    anio, mes = int(partes[-2]), int(partes[-1])
                    val = anio * 100 + mes
                    if 202410 <= val <= 202609:
                        archivos_csv.append(f)
                except ValueError:
                    continue

        if not archivos_csv:
            raise ValueError("No se encontraron archivos CSV en el rango de Octubre 2024 a Septiembre 2026.")

        logger.info(f"Procesando {len(archivos_csv)} archivos con manejo condicional de fechas...")
        
        dfs_agregados = []

        for f in sorted(archivos_csv):
            logger.info(f"Leyendo y agregando archivo: {f.name}")
            try:
                df_temp = pd.read_csv(f, low_memory=False, encoding="latin-1")
                df_temp.columns = df_temp.columns.str.strip().str.lower()

                date_col = next((c for c in ["inicio_del_viaje", "fecha_retiro", "tra_fecharetiro", "trip_start_time"] if c in df_temp.columns), None)
                origin_col = next((c for c in ["origen_id", "origen", "ret_estacion", "from_station_id"] if c in df_temp.columns), None)

                if not date_col or not origin_col:
                    logger.warning(f"Columnas requeridas no encontradas en {f.name}. Omitiendo.")
                    continue

                # CONDICIONAL DE FECHAS: Octubre 2024 usa formato DD/MM/YYYY (dayfirst=True)
                if "2024_10" in f.name:
                    df_temp["inicio_del_viaje"] = pd.to_datetime(df_temp[date_col], dayfirst=True, errors="coerce")
                else:
                    df_temp["inicio_del_viaje"] = pd.to_datetime(df_temp[date_col], errors="coerce")

                df_temp["origen_id"] = pd.to_numeric(df_temp[origin_col], errors="coerce")
                
                df_temp = df_temp.dropna(subset=["inicio_del_viaje", "origen_id"])
                df_temp["fecha_hora_inicio"] = df_temp["inicio_del_viaje"].dt.floor("h")

                parcial = (
                    df_temp.groupby(["origen_id", "fecha_hora_inicio"])
                    .size()
                    .reset_index(name="demanda_viajes")
                )
                dfs_agregados.append(parcial)

            except Exception as e:
                logger.error(f"Error al procesar el archivo {f.name}: {e}")

        if not dfs_agregados:
            raise ValueError("No se pudo procesar ningún archivo CSV.")

        logger.info("Consolidando todas las agregaciones mensuales...")
        regresion_df = pd.concat(dfs_agregados, ignore_index=True)

        regresion_df = (
            regresion_df.groupby(["origen_id", "fecha_hora_inicio"], as_index=False)["demanda_viajes"]
            .sum()
        )

        # Generar características temporales
        regresion_df["hora"] = regresion_df["fecha_hora_inicio"].dt.hour
        regresion_df["dia"] = regresion_df["fecha_hora_inicio"].dt.day
        regresion_df["dia_semana"] = regresion_df["fecha_hora_inicio"].dt.dayofweek
        regresion_df["mes"] = regresion_df["fecha_hora_inicio"].dt.month
        regresion_df["anio"] = regresion_df["fecha_hora_inicio"].dt.year
        regresion_df["es_fin_de_semana"] = regresion_df["dia_semana"].isin([5, 6]).astype(int)

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
            f"({len(regresion_df):,} filas agregadas desde {regresion_df['fecha_hora_inicio'].min()} hasta {regresion_df['fecha_hora_inicio'].max()})"
        )

        return output_path


@app.command()
def main(
    output_filename: str = "dataset_regresion_horario.parquet",
):
    logger.info("Iniciando pipeline con manejo condicional de fechas...")
    procesador = DatasetRegresionProcessor()
    procesador.construir_dataset_regresion(output_filename=output_filename)


if __name__ == "__main__":
    app()

import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple, Dict, Any
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import typer
from loguru import logger
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from tc5061_proyecto_equipo17.config import PROCESSED_DATA_DIR, MODELS_DIR


app = typer.Typer(
    help="Módulo de Entrenamiento e Integración MLOps con división temporal por fechas y MLflow - MiBici Equipo 17"
)


def obtener_git_commit() -> str:
    """Obtiene dinámicamente el hash del commit actual de Git para garantizar trazabilidad."""
    try:
        commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")
        return commit_hash
    except Exception as e:
        logger.warning(f"No se pudo obtener el commit de Git: {e}")
        return "unknown"


class RegressionModelTrainer:
    """
    Clase orientada a objetos responsable de gestionar la división temporal por fechas 
    (Train <= Jul 2026, Val = Ago 2026, Test = Sep 2026), el entrenamiento del modelo 
    y el registro de parámetros, métricas, artefactos y tags en MLflow.
    """

    def __init__(
        self,
        experiment_name: str = "mibici-equipo17",
        processed_dir: Path = PROCESSED_DATA_DIR,
        models_dir: Path = MODELS_DIR,
    ):
        self.experiment_name = experiment_name
        self.processed_dir = processed_dir
        self.models_dir = models_dir
        mlflow.set_experiment(self.experiment_name)

    def cargar_datos(self, input_filename: str = "features_regresion.parquet") -> pd.DataFrame:
        """Carga el dataset enriquecido con lags, rolling stats y variables sin escalado."""
        ruta_features = self.processed_dir / input_filename
        if not ruta_features.is_file():
            logger.error(f"El archivo {ruta_features} no existe. Ejecute primero features_regression.py.")
            raise FileNotFoundError(f"Archivo no encontrado: {ruta_features}")

        logger.info(f"Cargando dataset con características: {ruta_features.name}")
        df = pd.read_parquet(ruta_features)
        return df.sort_values(by="fecha_hora_inicio").reset_index(drop=True)

    def split_temporal_por_fechas(
        self, df: pd.DataFrame, target_col: str = "demanda_viajes"
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, List[str]]:
        """
        Realiza una división temporal estricta por calendario para evitar data leakage:
        - Entrenamiento: Octubre 2024 hasta Julio 2026
        - Validación: Agosto 2026
        - Prueba (Test): Septiembre 2026
        """
        exclude_cols = ["origen_id", "fecha_hora_inicio", target_col]
        feature_cols = [col for col in df.columns if col not in exclude_cols]

        # Filtros de fecha estrictos
        train_df = df[df["fecha_hora_inicio"] < pd.to_datetime("2026-08-01")]
        val_df = df[(df["fecha_hora_inicio"] >= pd.to_datetime("2026-08-01")) & (df["fecha_hora_inicio"] < pd.to_datetime("2026-09-01"))]
        test_df = df[df["fecha_hora_inicio"] >= pd.to_datetime("2026-09-01")]

        X_train, y_train = train_df[feature_cols], train_df[target_col]
        X_val, y_val = val_df[feature_cols], val_df[target_col]
        X_test, y_test = test_df[feature_cols], test_df[target_col]

        logger.info(
            f"División temporal por fechas completada:\n"
            f"  - Entrenamiento (Oct 2024 - Jul 2026): {len(X_train):,} filas\n"
            f"  - Validación (Ago 2026): {len(X_val):,} filas\n"
            f"  - Prueba / Test (Sep 2026): {len(X_test):,} filas"
        )
        return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols

    def calcular_metricas(self, y_true: np.ndarray, y_pred: np.ndarray, prefijo: str = "") -> Dict[str, float]:
        """Calcula métricas clave de evaluación de regresión (RMSE, MAE, R2, MAPE)."""
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        mask = y_true != 0
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if np.any(mask) else np.nan

        p = f"{prefijo}_" if prefijo else ""
        return {
            f"{p}rmse": float(rmse),
            f"{p}mae": float(mae),
            f"{p}r2": float(r2),
            f"{p}mape": float(mape),
        }

    def generar_grafico_importancia(
        self, model: Any, feature_names: List[str]
    ) -> str:
        """Genera un gráfico de importancia de características y devuelve la ruta temporal."""
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(10, 6))
        plt.title("Importancia de Características (Feature Importance)")
        plt.bar(range(len(feature_names)), importances[indices], align="center")
        plt.xticks(
            range(len(feature_names)),
            [feature_names[i] for i in indices],
            rotation=45,
            ha="right",
        )
        plt.tight_layout()

        temp_dir = Path(tempfile.mkdtemp())
        fig_path = str(temp_dir / "feature_importance.png")
        plt.savefig(fig_path)
        plt.close()
        return fig_path

    def entrenar_y_registrar(
        self,
        n_estimators: int = 150,
        max_depth: int = 20,
        min_samples_split: int = 5,
        min_samples_leaf: int = 2,
        max_features: str = "sqrt",
        random_state: int = 42,
        run_name: str = "RandomForest_TemporalSplit",
        data_filename: str = "features_regresion.parquet",
    ):
        """
        Ejecuta el pipeline completo de entrenamiento con división por fechas,
        registrando hiperparámetros, métricas de validación y test, artefactos y tags en MLflow.
        """
        df = self.cargar_datos(input_filename=data_filename)

        fecha_min = str(df["fecha_hora_inicio"].min())
        fecha_max = str(df["fecha_hora_inicio"].max())
        git_commit = obtener_git_commit()

        X_train, X_val, X_test, y_train, y_val, y_test, feature_cols = self.split_temporal_por_fechas(df)

        with mlflow.start_run(run_name=run_name):
            logger.info(f"Entrenando modelo {run_name} con división temporal estricta...")

            params = {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "min_samples_split": min_samples_split,
                "min_samples_leaf": min_samples_leaf,
                "max_features": max_features,
                "random_state": random_state,
                "split_strategy": "date_calendar_split",
                "train_period": "Oct 2024 - Jul 2026",
                "val_period": "Aug 2026",
                "test_period": "Sep 2026",
                "data_version_file": data_filename,
                "fecha_min_datos": fecha_min,
                "fecha_max_datos": fecha_max,
                "n_features": len(feature_cols),
            }

            mlflow.set_tag("team", "equipo-17")
            mlflow.set_tag("question", "P1")
            mlflow.set_tag("git_commit", git_commit)

            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                random_state=random_state,
                n_jobs=-1,
            )
            model.fit(X_train, y_train)

            # Evaluación en Validación (Agosto 2026)
            y_pred_val = model.predict(X_val)
            metricas_val = self.calcular_metricas(y_val, y_pred_val, prefijo="val")

            # Evaluación en Test (Septiembre 2026)
            y_pred_test = model.predict(X_test)
            metricas_test = self.calcular_metricas(y_test, y_pred_test, prefijo="test")

            # Consolidar todas las métricas
            metricas_totales = {**metricas_val, **metricas_test}

            logger.info(
                f"Métricas Validación (Ago 2026) -> RMSE: {metricas_val['val_rmse']:.4f} | R2: {metricas_val['val_r2']:.4f}"
            )
            logger.info(
                f"Métricas Prueba (Sep 2026)    -> RMSE: {metricas_test['test_rmse']:.4f} | R2: {metricas_test['test_r2']:.4f}"
            )

            mlflow.log_params(params)
            mlflow.log_metrics(metricas_totales)

            fig_path = self.generar_grafico_importancia(model, feature_cols)
            mlflow.log_artifact(fig_path)

            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                skops_trusted_types=["sklearn.tree._tree.Tree"],
            )

            logger.success(f"Entrenamiento y registro completados con éxito en MLflow ('{self.experiment_name}')")


@app.command()
def main(
    n_estimators: int = 150,
    max_depth: int = 20,
    min_samples_split: int = 5,
    min_samples_leaf: int = 2,
    max_features: str = "sqrt",
    run_name: str = "RandomForest_TemporalSplit",
):
    """
    Ejecuta el entrenamiento del modelo de regresión con división temporal basada en fechas específicas.
    """
    logger.info("Iniciando pipeline de entrenamiento con división por fechas (Jul/Ago/Sep 2026)...")
    trainer = RegressionModelTrainer()
    trainer.entrenar_y_registrar(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        run_name=run_name,
    )


if __name__ == "__main__":
    app()
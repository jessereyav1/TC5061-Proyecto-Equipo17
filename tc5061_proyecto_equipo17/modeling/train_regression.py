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
    help="Módulo de Entrenamiento e Integración MLOps con MLflow para el Modelo de Regresión - MiBici Equipo 17"
)


class RegressionModelTrainer:
    """
    Clase orientada a objetos responsable de gestionar la división temporal de datos,
    el entrenamiento del modelo de regresión y el registro de parámetros, métricas y artefactos en MLflow.
    """

    def __init__(
        self,
        experiment_name: str = "MiBici_Demanda_Regresion",
        processed_dir: Path = PROCESSED_DATA_DIR,
        models_dir: Path = MODELS_DIR,
    ):
        self.experiment_name = experiment_name
        self.processed_dir = processed_dir
        self.models_dir = models_dir
        mlflow.set_experiment(self.experiment_name)

    def cargar_datos(self, input_filename: str = "features_regresion_escalados.parquet") -> pd.DataFrame:
        """Carga el dataset enriquecido con lags, rolling stats y variables escaladas."""
        ruta_features = self.processed_dir / input_filename
        if not ruta_features.is_file():
            logger.error(f"El archivo {ruta_features} no existe. Ejecute primero features_regression.py.")
            raise FileNotFoundError(f"Archivo no encontrado: {ruta_features}")

        logger.info(f"Cargando dataset con características: {ruta_features.name}")
        df = pd.read_parquet(ruta_features)
        return df.sort_values(by="fecha_hora_inicio").reset_index(drop=True)

    def split_temporal(
        self, df: pd.DataFrame, target_col: str = "demanda_viajes", test_ratio: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]:
        """
        Realiza una división basada en el tiempo (Time Series Split) para evitar data leakage.
        """
        exclude_cols = ["origen_id", "fecha_hora_inicio", target_col]
        feature_cols = [col for col in df.columns if col not in exclude_cols]

        n_samples = len(df)
        split_idx = int(n_samples * (1 - test_ratio))

        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]

        X_train, y_train = train_df[feature_cols], train_df[target_col]
        X_test, y_test = test_df[feature_cols], test_df[target_col]

        logger.info(
            f"División temporal realizada - Entrenar: {len(X_train):,} filas | Evaluar: {len(X_test):,} filas"
        )
        return X_train, X_test, y_train, y_test, feature_cols

    def calcular_metricas(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calcula métricas clave de evaluación de regresión (RMSE, MAE, R2, MAPE)."""
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        mask = y_true != 0
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if np.any(mask) else np.nan

        return {
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2),
            "mape": float(mape),
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
        n_estimators: int = 100,
        max_depth: int = 15,
        random_state: int = 42,
        run_name: str = "RandomForest_Baseline",
    ):
        """
        Ejecuta el pipeline completo de entrenamiento y loguea parámetros, métricas y artefactos en MLflow.
        """
        df = self.cargar_datos()
        X_train, X_test, y_train, y_test, feature_cols = self.split_temporal(df)

        with mlflow.start_run(run_name=run_name):
            logger.info(f"Entrenando modelo {run_name} con {n_estimators} árboles...")

            params = {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "random_state": random_state,
                "n_jobs": -1,
            }

            model = RandomForestRegressor(**params)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            metricas = self.calcular_metricas(y_test, y_pred)

            logger.info(f"Métricas alcanzadas en Test -> RMSE: {metricas['rmse']:.4f} | MAE: {metricas['mae']:.4f} | R2: {metricas['r2']:.4f}")

            mlflow.log_params(params)
            mlflow.log_metrics(metricas)

            fig_path = self.generar_grafico_importancia(model, feature_cols)
            mlflow.log_artifact(fig_path)

            # Registro del modelo con skops_trusted_types para scikit-learn tree models
            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                skops_trusted_types=["sklearn.tree._tree.Tree"],
            )

            logger.success("Experimento completado y registrado exitosamente en MLflow.")


@app.command()
def main(
    n_estimators: int = 100,
    max_depth: int = 15,
    run_name: str = "RandomForest_Baseline",
):
    """
    Ejecuta el entrenamiento del modelo de regresión y registra los resultados en MLflow.
    """
    logger.info("Iniciando ejecución de entrenamiento del modelo de regresión...")
    trainer = RegressionModelTrainer()
    trainer.entrenar_y_registrar(
        n_estimators=n_estimators,
        max_depth=max_depth,
        run_name=run_name,
    )


if __name__ == "__main__":
    app()

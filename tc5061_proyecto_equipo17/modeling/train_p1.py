"""P1: Regresión de demanda MiBici con Random Forest, registrado en MLflow."""

import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from typing import Any, List, Tuple

from loguru import logger
import matplotlib

matplotlib.use("Agg")  # Dibuja a archivo sin abrir ventanas gráficas
import matplotlib.pyplot as plt
import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn
import numpy as np
import pandas as pd
import typer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tc5061_proyecto_equipo17.config import (
    MLFLOW_EXPERIMENT,
    PROCESSED_DATA_DIR,
    SEED,
)

app = typer.Typer(
    help="Módulo de Entrenamiento P1 (Regresión) con división temporal y MLflow - MiBici Equipo 17"
)


def _commit_actual() -> str:
    """Hash del commit de Git, para rastrear qué código generó la corrida."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "desconocido"


def _version_datos() -> str:
    """Huella de los datos procesados a partir de los SHA-256 del manifiesto."""
    try:
        manifest_path = PROCESSED_DATA_DIR / "manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            hashes = "".join(sorted(a["sha256_procesado"] for a in manifest["archivos"]))
            return hashlib.sha256(hashes.encode()).hexdigest()[:12]
    except Exception:
        pass
    return "desconocido"


class RegressionModelTrainer:
    """
    Clase responsable de gestionar la división temporal por fechas para P1
    (Train <= Jul 2026, Val = Ago 2026, Test = Sep 2026), el entrenamiento del modelo
    y el registro avanzado en MLflow (incluyendo soporte para Nested Runs).
    """

    def __init__(
        self,
        experiment_name: str = MLFLOW_EXPERIMENT,
        processed_dir: Path = PROCESSED_DATA_DIR,
    ):
        self.experiment_name = experiment_name
        self.processed_dir = processed_dir
        mlflow.set_experiment(self.experiment_name)

    def cargar_datos(self, input_filename: str = "features_p1.parquet") -> pd.DataFrame:
        """Carga el dataset enriquecido con características para P1."""
        ruta_features = self.processed_dir / input_filename
        if not ruta_features.is_file():
            logger.error(f"El archivo {ruta_features} no existe. Ejecute primero features_p1.py.")
            raise FileNotFoundError(f"Archivo no encontrado: {ruta_features}")

        logger.info(f"Cargando dataset con características: {ruta_features.name}")
        df = pd.read_parquet(ruta_features)
        return df.sort_values(by="fecha_hora_inicio").reset_index(drop=True)

    def split_temporal_por_fechas(
        self, df: pd.DataFrame, target_col: str = "demanda_viajes"
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, List[str]]:
        """
        División temporal estricta por calendario para evitar data leakage:
        - Entrenamiento: Octubre 2024 hasta Julio 2026
        - Validación: Agosto 2026
        - Prueba (Test): Septiembre 2026
        """
        exclude_cols = ["origen_id", "fecha_hora_inicio", target_col]
        feature_cols = [col for col in df.columns if col not in exclude_cols]

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

    def calcular_metricas(self, y_true: np.ndarray, y_pred: np.ndarray, prefijo: str = "") -> dict:
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

    def generar_grafico_importancia(self, model: Any, feature_names: List[str]) -> str:
        """Genera un gráfico de importancia de características."""
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(10, 6))
        plt.title("Importancia de Características - P1 Regresión")
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

    def generar_grafico_residuos(self, y_true: np.ndarray, y_pred: np.ndarray, set_name: str = "Test") -> str:
        """Genera un gráfico de análisis de errores (Residuos vs Predicciones)."""
        plt.figure(figsize=(8, 6))
        residuos = y_true - y_pred
        plt.scatter(y_pred, residuos, alpha=0.3, color="purple")
        plt.axhline(y=0, color="red", linestyle="--", linewidth=1.5)
        plt.title(f"Análisis de Residuos - P1 {set_name}")
        plt.xlabel("Valores Predichos")
        plt.ylabel("Residuos (Real - Predicho)")
        plt.tight_layout()

        temp_dir = Path(tempfile.mkdtemp())
        fig_path = str(temp_dir / f"residuals_analysis_{set_name.lower()}.png")
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
        random_state: int = SEED,
        run_name: str = "",
        data_filename: str = "features_p1.parquet",
        etapa: str = "baseline",
        nested: bool = False,
    ):
        """
        Ejecuta el pipeline completo de entrenamiento con división por fechas,
        registrando hiperparámetros, métricas de validación/test, artefactos avanzados
        y registrando el modelo en el Model Registry de MLflow.
        """
        df = self.cargar_datos(input_filename=data_filename)

        fecha_min = str(df["fecha_hora_inicio"].min())
        fecha_max = str(df["fecha_hora_inicio"].max())
        
        # Nombre dinámico y descriptivo basado en hiperparámetros
        if not run_name:
            run_name = f"p1-random-forest-n{n_estimators}-depth{max_depth}"

        X_train, X_val, X_test, y_train, y_val, y_test, feature_cols = self.split_temporal_por_fechas(df)

        config_corrida = {
            "modelo": "random_forest",
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
            "data_version": _version_datos(),
            "fecha_min_datos": fecha_min,
            "fecha_max_datos": fecha_max,
            "n_features": len(feature_cols),
        }

        with mlflow.start_run(run_name=run_name, nested=nested):
            logger.info(f"Entrenando modelo {run_name} (nested={nested}) con división temporal estricta...")

            mlflow.set_tags(
                {
                    "equipo": "17",
                    "pregunta": "P1-regresion",
                    "question": "p1",
                    "model_family": "random_forest",
                    "stage_of_work": etapa,
                    "git_commit": _commit_actual(),
                }
            )

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

            metricas_totales = {**metricas_val, **metricas_test}

            logger.info(
                f"Métricas Validación (Ago 2026) -> RMSE: {metricas_val['val_rmse']:.4f} | R2: {metricas_val['val_r2']:.4f}"
            )
            logger.info(
                f"Métricas Prueba (Sep 2026)    -> RMSE: {metricas_test['test_rmse']:.4f} | R2: {metricas_test['test_r2']:.4f}"
            )

            mlflow.log_params(config_corrida)
            mlflow.log_metrics(metricas_totales)

            # Registro de datasets de entrada en el Data Inspector de MLflow
            mlflow.log_input(
                mlflow.data.from_pandas(X_train, source=str(PROCESSED_DATA_DIR / data_filename), name="p1_train"),
                context="training",
            )
            mlflow.log_input(
                mlflow.data.from_pandas(X_test, source=str(PROCESSED_DATA_DIR / data_filename), name="p1_test"),
                context="testing",
            )

            # Registro explícito en el Model Registry oficial del equipo con soporte skops
            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                registered_model_name="mibici-equipo17-p1-demanda",
                signature=infer_signature(X_train, model.predict(X_train)),
                input_example=X_train.head(5),
                skops_trusted_types=["sklearn.tree._tree.Tree"],
            )

            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)

                # plots/: Gráfico de importancia de características
                fig_imp = self.generar_grafico_importancia(model, feature_cols)
                mlflow.log_artifact(fig_imp, "plots")

                # plots/: Análisis de residuos en validación
                fig_res_val = self.generar_grafico_residuos(y_val, y_pred_val, set_name="Validacion")
                mlflow.log_artifact(fig_res_val, "plots")

                # plots/: Análisis de residuos en prueba
                fig_res_test = self.generar_grafico_residuos(y_test, y_pred_test, set_name="Prueba")
                mlflow.log_artifact(fig_res_test, "plots")

                # config/: Parámetros de la corrida en JSON
                (tmp / "config_corrida.json").write_text(
                    json.dumps(config_corrida, indent=2, ensure_ascii=False), encoding="utf-8"
                )
                mlflow.log_artifact(str(tmp / "config_corrida.json"), "config")

                # data/: Manifiesto si está disponible
                manifest_path = PROCESSED_DATA_DIR / "manifest.json"
                if manifest_path.is_file():
                    mlflow.log_artifact(str(manifest_path), "data")

            logger.success(f"Entrenamiento y registro de P1 completados con éxito en MLflow ('{run_name}')")

    def busqueda_hiperparametros(self, data_filename: str = "features_p1.parquet"):
        """
        Ejecuta una búsqueda ligera de hiperparámetros con Nested Runs:
        Una corrida padre (parent run) que engloba múltiples corridas hijas (child runs).
        """
        # Grilla rápida con árboles ligeros para que termine en menos de 1-2 minutos
        combinaciones = [
            {"n_estimators": 20, "max_depth": 10, "min_samples_split": 5},
            {"n_estimators": 40, "max_depth": 15, "min_samples_split": 5},
            {"n_estimators": 50, "max_depth": 20, "min_samples_split": 10},
        ]

        with mlflow.start_run(run_name="p1-grid-search-parent"):
            logger.info("Iniciando corrida padre (Parent Run) para búsqueda rápida de hiperparámetros P1...")
            
            mlflow.set_tags(
                {
                    "equipo": "17",
                    "pregunta": "P1-regresion",
                    "question": "p1",
                    "model_family": "random_forest",
                    "stage_of_work": "search",
                    "git_commit": _commit_actual(),
                }
            )
            mlflow.log_param("search_strategy", "grid_search_rapido")
            mlflow.log_param("total_combinations", len(combinaciones))

            for idx, params in enumerate(combinaciones, start=1):
                logger.info(f"Ejecutando combinación hija {idx}/{len(combinaciones)}: {params}")
                
                with mlflow.start_run(
                    run_name=f"p1-child-n{params['n_estimators']}-depth{params['max_depth']}",
                    nested=True
                ):
                    self.entrenar_y_registrar(
                        n_estimators=params["n_estimators"],
                        max_depth=params["max_depth"],
                        min_samples_split=params["min_samples_split"],
                        data_filename=data_filename,
                        etapa="search",
                        nested=True,
                    )

            logger.success("Búsqueda rápida de hiperparámetros con Nested Runs finalizada con éxito.")


@app.command()
def main(
    n_estimators: int = 150,
    max_depth: int = 20,
    min_samples_split: int = 5,
    min_samples_leaf: int = 2,
    max_features: str = "sqrt",
    run_name: str = "",
):
    """Ejecuta el entrenamiento individual o detecta si se pide búsqueda mediante centinela."""
    trainer = RegressionModelTrainer()
    
    # Valor centinela positivo (999) para saltar la validación de números negativos de run_training.py
    if n_estimators == 999:
        logger.info("Activando búsqueda de hiperparámetros (Nested Runs) con centinela positivo...")
        trainer.busqueda_hiperparametros()
    else:
        # Detección automática de la etapa según la rúbrica:
        # Si coincide con el baseline oficial, se marca como "baseline", de lo contrario es un modelo "candidate".
        if n_estimators == 150 and max_depth == 20:
            etapa = "baseline"
        else:
            etapa = "candidate"
        
        logger.info("Iniciando pipeline de entrenamiento P1 individual...")
        trainer.entrenar_y_registrar(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            run_name=run_name,
            etapa=etapa,
            nested=False,
        )


if __name__ == "__main__":
    app()

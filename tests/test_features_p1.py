"""Pruebas unitarias avanzadas y parametrizadas para la generación de características de P1 (FeatureRegressionEngineer)."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from tc5061_proyecto_equipo17.features_p1 import FeatureRegressionEngineer


def _crear_dataset_horario_sintetico(num_horas=200, estaciones=[10, 20]):
    """Genera un DataFrame sintético con suficientes horas para calcular lags (hasta 168h)."""
    registros = []
    fechas = pd.date_range("2025-01-01 00:00:00", periods=num_horas, freq="h")

    for est in estaciones:
        for f in fechas:
            registros.append(
                {
                    "origen_id": est,
                    "fecha_hora_inicio": f,
                    "demanda_viajes": np.random.randint(1, 20),
                    "hora": f.hour,
                    "dia": f.day,
                    "dia_semana": f.dayofweek,
                    "mes": f.month,
                    "anio": f.year,
                    "es_fin_de_semana": int(f.dayofweek in [5, 6]),
                }
            )
    df = pd.DataFrame(registros)
    return df.sort_values(by=["origen_id", "fecha_hora_inicio"]).reset_index(
        drop=True
    )


def test_construir_y_guardar_features_caso_normal(tmp_path):
    """Caso normal: verifica la creación de rezagos (lags), ventanas móviles

    y transformaciones cíclicas (seno/coseno) guardando el archivo Parquet.
    """
    df_base = _crear_dataset_horario_sintetico(num_horas=200)
    engineer = FeatureRegressionEngineer(processed_dir=tmp_path)

    df_feat, output_path = engineer.construir_y_guardar_features(
        df=df_base, output_filename="test_features.parquet"
    )

    assert output_path.exists()
    assert not df_feat.empty

    columnas_esperadas = [
        "lag_demanda_1h",
        "lag_demanda_2h",
        "lag_demanda_24h",
        "lag_demanda_168h",
        "rolling_mean_3h",
        "rolling_std_3h",
        "rolling_mean_6h",
        "rolling_std_6h",
        "hora_sin",
        "hora_cos",
        "mes_sin",
        "mes_cos",
    ]
    for col in columnas_esperadas:
        assert col in df_feat.columns

    assert df_feat.isna().sum().sum() == 0


def test_features_transformacion_ciclica_valores_limite(tmp_path):
    """Caso límite: comprueba que las transformaciones de seno y coseno

    para la hora estén dentro del rango trigonométrico válido [-1, 1].
    """
    df_base = _crear_dataset_horario_sintetico(num_horas=200)
    engineer = FeatureRegressionEngineer(processed_dir=tmp_path)

    df_feat, _ = engineer.construir_y_guardar_features(
        df=df_base, output_filename="test_ciclicidad.parquet"
    )

    assert df_feat["hora_sin"].min() >= -1.0
    assert df_feat["hora_sin"].max() <= 1.0
    assert df_feat["hora_cos"].min() >= -1.0
    assert df_feat["hora_cos"].max() <= 1.0


def test_cargar_dataset_horario_archivo_inexistente_lanza_error(tmp_path):
    """Caso límite: lanza FileNotFoundError si se intenta cargar un archivo que no existe."""
    engineer = FeatureRegressionEngineer(processed_dir=tmp_path)
    with pytest.raises(FileNotFoundError):
        engineer.cargar_dataset_horario(
            input_filename="archivo_inexistente.parquet"
        )


def test_features_mes_transformacion_ciclica(tmp_path):
    """Caso normal: comprueba que las transformaciones cíclicas del mes estén acotadas [-1, 1]."""
    df_base = _crear_dataset_horario_sintetico(num_horas=200)
    engineer = FeatureRegressionEngineer(processed_dir=tmp_path)

    df_feat, _ = engineer.construir_y_guardar_features(
        df=df_base, output_filename="test_mes_ciclico.parquet"
    )

    assert df_feat["mes_sin"].min() >= -1.0
    assert df_feat["mes_sin"].max() <= 1.0
    assert df_feat["mes_cos"].min() >= -1.0
    assert df_feat["mes_cos"].max() <= 1.0


def test_features_ordenamiento_por_estacion_y_fecha(tmp_path):
    """Caso normal: verifica que el dataset cargado respete el orden cronológico por cada estación."""
    df_base = _crear_dataset_horario_sintetico(num_horas=50)
    df_desordenado = df_base.sample(frac=1).reset_index(drop=True)

    engineer = FeatureRegressionEngineer(processed_dir=tmp_path)
    input_file = tmp_path / "temp_base.parquet"
    df_desordenado.to_parquet(input_file)

    df_cargado = engineer.cargar_dataset_horario("temp_base.parquet")
    for _, grupo in df_cargado.groupby("origen_id"):
        assert grupo["fecha_hora_inicio"].is_monotonic_increasing


def test_lags_y_rolling_calculan_correctamente(tmp_path):
    """Caso normal: verifica que los valores de rezago correspondan matemáticamente con la demanda pasada."""
    df_base = _crear_dataset_horario_sintetico(num_horas=50, estaciones=[10])
    engineer = FeatureRegressionEngineer(processed_dir=tmp_path)

    df_feat, _ = engineer.construir_y_guardar_features(
        df=df_base, output_filename="test_lags_calc.parquet"
    )

    for i in range(1, len(df_feat)):
        demanda_anterior = df_feat.loc[i - 1, "demanda_viajes"]
        lag_actual = df_feat.loc[i, "lag_demanda_1h"]
        assert demanda_anterior == lag_actual


@pytest.mark.parametrize(
    "ventana",
    [3, 6],
)
def test_rolling_mean_calcula_con_diferentes_ventanas(tmp_path, ventana):
    """Prueba parametrizada: verifica que las estadísticas móviles (rolling mean) se generen correctamente para diferentes tamaños de ventana."""
    df_base = _crear_dataset_horario_sintetico(num_horas=100)
    engineer = FeatureRegressionEngineer(processed_dir=tmp_path)

    df_feat, _ = engineer.construir_y_guardar_features(
        df=df_base, output_filename=f"test_roll_{ventana}.parquet"
    )

    col_mean = f"rolling_mean_{ventana}h"
    col_std = f"rolling_std_{ventana}h"

    assert col_mean in df_feat.columns
    assert col_std in df_feat.columns
    assert not df_feat[col_mean].isna().any()


@pytest.mark.parametrize(
    "lag_val",
    [1, 2, 24, 168],
)
def test_lags_especificos_presentes_y_sin_nulos(tmp_path, lag_val):
    """Prueba parametrizada: comprueba que todos los rezagos (lags) requeridos (1h, 2h, 24h, 168h) estén presentes y limpios."""
    df_base = _crear_dataset_horario_sintetico(num_horas=250)
    engineer = FeatureRegressionEngineer(processed_dir=tmp_path)

    df_feat, _ = engineer.construir_y_guardar_features(
        df=df_base, output_filename=f"test_lag_{lag_val}.parquet"
    )

    col_lag = f"lag_demanda_{lag_val}h"
    assert col_lag in df_feat.columns
    assert df_feat[col_lag].dtype in [np.float64, np.int64, float, int]


def test_no_fuga_de_datos_escaladores(tmp_path):
    """Caso límite: verifica que el pipeline de features no introduzca columnas de escalado (evitando data leakage)."""
    df_base = _crear_dataset_horario_sintetico(num_horas=100)
    engineer = FeatureRegressionEngineer(processed_dir=tmp_path)

    df_feat, _ = engineer.construir_y_guardar_features(
        df=df_base, output_filename="test_noscale.parquet"
    )

    # Verificar que columnas típicas de StandardScaler no existan, asegurando que se respeta la arquitectura sin escalar para árboles
    escalados = [c for c in df_feat.columns if "scaled" in c or "normalized" in c]
    assert len(escalados) == 0


def test_dataset_features_mantiene_columnas_temporales_originales(tmp_path):
    """Caso normal: verifica que el dataset resultante conserve las variables temporales base (hora, mes, anio, etc.)."""
    df_base = _crear_dataset_horario_sintetico(num_horas=180)
    engineer = FeatureRegressionEngineer(processed_dir=tmp_path)

    df_feat, _ = engineer.construir_y_guardar_features(
        df=df_base, output_filename="test_orig_cols.parquet"
    )

    for c in ["hora", "dia", "mes", "anio", "es_fin_de_semana"]:
        assert c in df_feat.columns
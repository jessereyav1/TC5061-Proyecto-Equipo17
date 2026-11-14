"""Pruebas de los conjuntos de referencia y actual para el monitoreo (Actividad 10)."""

import json

import pandas as pd
import pytest

import tc5061_proyecto_equipo17.dataset as dataset
import tc5061_proyecto_equipo17.monitoreo_datos as monitoreo
from tc5061_proyecto_equipo17 import config
from tc5061_proyecto_equipo17.features import P2_FEATURES, P2_OBJETIVO


def _viajes_sinteticos(inicio: str, dias: int) -> pd.DataFrame:
    """Un viaje cada 30 minutos entre las estaciones 1 y 2."""
    salidas = pd.date_range(inicio, periods=dias * 48, freq="30min")
    return pd.DataFrame(
        {
            "viaje_id": pd.array(range(len(salidas)), dtype="string"),
            "inicio_del_viaje": salidas,
            "fin_del_viaje": salidas + pd.Timedelta(minutes=10),
            "origen_id": pd.array([1 if i % 2 == 0 else 2 for i in range(len(salidas))], dtype="Int64"),
            "destino_id": pd.array([2 if i % 2 == 0 else 1 for i in range(len(salidas))], dtype="Int64"),
        }
    )


def test_periodos_no_se_traslapan_con_el_entrenamiento():
    """Referencia y actual son periodos distintos, y el actual es posterior a los datos del modelo."""
    assert pd.Timestamp(config.REF_FIN) < pd.Timestamp(config.ACTUAL_INICIO)
    assert pd.Timestamp(config.ACTUAL_INICIO) > pd.Timestamp(config.FECHA_FIN)
    assert pd.Timestamp(config.ACTUAL_INICIO) < pd.Timestamp(config.ACTUAL_CORTE_EVAL) <= pd.Timestamp(
        config.ACTUAL_FIN
    )


def test_url_del_periodo_actual_es_oficial_y_corresponde_al_mes():
    assert dataset.validar_url_mibici(monitoreo.URL_CSV_ACTUAL) == monitoreo.URL_CSV_ACTUAL
    assert monitoreo.URL_CSV_ACTUAL.endswith(monitoreo.NOMBRE_CSV_ACTUAL)
    assert monitoreo.NOMBRE_CSV_ACTUAL == "datos_abiertos_2025_07.csv"


def test_asignar_ventana_separa_reentrenamiento_y_evaluacion():
    horas = pd.Series(
        pd.to_datetime(
            [
                "2025-07-01 00:00",
                "2025-07-21 22:00",
                "2025-07-21 23:00",  # última hora de reentrenamiento: su etiqueta usa la hora de evaluación
                "2025-07-22 00:00",
                "2025-07-31 22:00",
            ]
        )
    )
    ventana = monitoreo.asignar_ventana(horas, corte="2025-07-22")
    assert ventana.iloc[0] == monitoreo.VENTANA_REENTRENAMIENTO
    assert ventana.iloc[1] == monitoreo.VENTANA_REENTRENAMIENTO
    assert pd.isna(ventana.iloc[2])
    assert ventana.iloc[3] == monitoreo.VENTANA_EVALUACION
    assert ventana.iloc[4] == monitoreo.VENTANA_EVALUACION


def test_ingesta_actual_no_toca_los_datos_de_entrenamiento(tmp_path, monkeypatch):
    """El Parquet del periodo actual va a su carpeta y no a data/processed."""
    carpeta_processed = tmp_path / "processed"
    carpeta_processed.mkdir()
    monkeypatch.setattr(dataset, "PROCESSED_DATA_DIR", carpeta_processed)

    carpeta_raw = tmp_path / "raw"
    carpeta_raw.mkdir()
    datos = pd.DataFrame(
        {
            "Viaje_Id": ["1", "2"],
            "Inicio_del_viaje": ["2025-07-01 08:00:00", "2025-07-01 09:00:00"],
            "Fin_del_viaje": ["2025-07-01 08:15:00", "2025-07-01 09:20:00"],
            "Origen_Id": [10, 20],
            "Destino_Id": [11, 21],
        }
    )
    datos.to_csv(carpeta_raw / monitoreo.NOMBRE_CSV_ACTUAL, index=False, encoding="cp1252")

    salida = tmp_path / "actual"
    resultado = monitoreo.ingesta_actual(
        directorio_raw=carpeta_raw, directorio_salida=salida, descargar=False
    )

    assert resultado["registros_procesados"] == 2
    assert (salida / monitoreo.PARQUET_ACTUAL).is_file()
    assert list(carpeta_processed.glob("*_procesado.parquet")) == []


def test_ingesta_actual_sin_archivo_y_sin_descarga_falla(tmp_path):
    with pytest.raises(FileNotFoundError):
        monitoreo.ingesta_actual(
            directorio_raw=tmp_path / "raw", directorio_salida=tmp_path / "actual", descargar=False
        )


def test_construir_features_genera_tablas_y_manifiesto(tmp_path):
    carpeta_ref = tmp_path / "processed"
    carpeta_act = tmp_path / "actual"
    carpeta_ref.mkdir()
    carpeta_act.mkdir()
    _viajes_sinteticos("2025-01-01", dias=31).to_parquet(carpeta_ref / monitoreo.PARQUET_REFERENCIA)
    _viajes_sinteticos("2025-07-01", dias=31).to_parquet(carpeta_act / monitoreo.PARQUET_ACTUAL)
    estaciones = tmp_path / "nomenclatura.csv"
    pd.DataFrame({"id": [1, 2], "dpcapacity": [10, 10]}).to_csv(estaciones, index=False)

    salida = tmp_path / "monitoreo"
    manifiesto = monitoreo.construir_features(
        carpeta_referencia=carpeta_ref,
        carpeta_actual=carpeta_act,
        carpeta_salida=salida,
        ruta_nomenclatura=estaciones,
    )

    referencia = pd.read_parquet(salida / monitoreo.ARCHIVO_REFERENCIA)
    actual = pd.read_parquet(salida / monitoreo.ARCHIVO_ACTUAL)

    # Mismas variables que ve el modelo, más la etiqueta.
    for tabla in (referencia, actual):
        assert set(P2_FEATURES + [P2_OBJETIVO]).issubset(tabla.columns)
        assert tabla[P2_FEATURES].isna().sum().sum() == 0

    # Cada tabla solo contiene su propio periodo.
    assert referencia["hora_ts"].min() >= pd.Timestamp(config.REF_INICIO)
    assert referencia["hora_ts"].max() < pd.Timestamp(config.REF_FIN) + pd.Timedelta(days=1)
    assert actual["hora_ts"].min() >= pd.Timestamp(config.ACTUAL_INICIO)

    # Las ventanas del A/B no se cruzan y la hora de separación queda fuera.
    entrenamiento = actual[actual["ventana"] == monitoreo.VENTANA_REENTRENAMIENTO]
    evaluacion = actual[actual["ventana"] == monitoreo.VENTANA_EVALUACION]
    corte = pd.Timestamp(config.ACTUAL_CORTE_EVAL)
    assert entrenamiento["hora_ts"].max() < corte - pd.Timedelta(hours=1)
    assert evaluacion["hora_ts"].min() == corte
    assert actual["ventana"].isna().sum() == 2  # una hora de separación por estación

    # El manifiesto describe lo que se generó.
    en_disco = json.loads((salida / monitoreo.ARCHIVO_MANIFIESTO).read_text(encoding="utf-8"))
    assert en_disco["referencia"]["filas"] == len(referencia) == manifiesto["referencia"]["filas"]
    assert en_disco["actual"]["filas"] == len(actual)
    assert en_disco["variables"] == P2_FEATURES


def test_construir_features_sin_datos_previos_falla(tmp_path):
    with pytest.raises(FileNotFoundError):
        monitoreo.construir_features(
            carpeta_referencia=tmp_path,
            carpeta_actual=tmp_path,
            carpeta_salida=tmp_path / "salida",
            ruta_nomenclatura=tmp_path / "no_existe.csv",
        )

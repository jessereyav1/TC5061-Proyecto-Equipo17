"""Pruebas unitarias avanzadas y parametrizadas para el procesamiento de datos de P1 (DatasetRegresionProcessor)."""

from pathlib import Path
import pandas as pd
import pytest

from tc5061_proyecto_equipo17.dataset_p1 import DatasetRegresionProcessor


def test_construir_dataset_regresion_caso_normal(tmp_path):
    """Caso normal: procesa archivos CSV en el rango de fechas válido (202410 a 202609)

    y genera el Parquet agrupado por hora y estación con sus variables temporales.
    """
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir()
    processed_dir.mkdir()

    csv_oct2024 = raw_dir / "datos_abiertos_2024_10.csv"
    df_oct = pd.DataFrame(
        {
            "inicio_del_viaje": [
                "01/10/2024 08:15:00",
                "01/10/2024 08:45:00",
                "01/10/2024 09:10:00",
            ],
            "origen_id": [10, 10, 20],
        }
    )
    df_oct.to_csv(csv_oct2024, index=False)

    processor = DatasetRegresionProcessor(
        raw_dir=raw_dir, processed_dir=processed_dir
    )
    output_path = processor.construir_dataset_regresion(
        "test_regresion.parquet"
    )

    assert output_path.exists()
    df_resultado = pd.read_parquet(output_path)
    assert (
        df_resultado.loc[
            df_resultado["origen_id"] == 10, "demanda_viajes"
        ].iloc[0]
        == 2
    )
    for col in [
        "hora",
        "dia",
        "dia_semana",
        "mes",
        "anio",
        "es_fin_de_semana",
    ]:
        assert col in df_resultado.columns


def test_construir_dataset_regresion_sin_archivos_csv_lanza_error(tmp_path):
    """Caso límite: lanza FileNotFoundError si no existen archivos CSV en el directorio."""
    raw_vacio = tmp_path / "raw_vacio"
    raw_vacio.mkdir()

    processor = DatasetRegresionProcessor(
        raw_dir=raw_vacio, processed_dir=tmp_path
    )
    with pytest.raises(FileNotFoundError):
        processor.construir_dataset_regresion()


def test_construir_dataset_regresion_archivos_fuera_de_rango(tmp_path):
    """Caso límite: ignora archivos fuera del rango estipulado (Oct 2024 - Sep 2026)

    y lanza ValueError si no queda ninguno válido.
    """
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    csv_viejo = raw_dir / "datos_abiertos_2024_09.csv"
    df = pd.DataFrame(
        {"inicio_del_viaje": ["01/09/2024 08:00:00"], "origen_id": [10]}
    )
    df.to_csv(csv_viejo, index=False)

    processor = DatasetRegresionProcessor(
        raw_dir=raw_dir, processed_dir=tmp_path
    )
    with pytest.raises(
        ValueError, match="No se encontraron archivos CSV en el rango"
    ):
        processor.construir_dataset_regresion()


def test_procesamiento_maneja_nombres_alternativos_columnas(tmp_path):
    """Caso límite/normal: valida que reconozca columnas con nombres alternativos (ej. 'trip_start_time')."""
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir()
    processed_dir.mkdir()

    csv_alt = raw_dir / "datos_abiertos_2025_01.csv"
    df = pd.DataFrame(
        {
            "trip_start_time": ["2025-01-01 10:00:00", "2025-01-01 10:30:00"],
            "from_station_id": [15, 15],
        }
    )
    df.to_csv(csv_alt, index=False)

    processor = DatasetRegresionProcessor(
        raw_dir=raw_dir, processed_dir=processed_dir
    )
    output_path = processor.construir_dataset_regresion("test_alt.parquet")
    df_res = pd.read_parquet(output_path)
    assert not df_res.empty
    assert df_res.loc[df_res["origen_id"] == 15, "demanda_viajes"].iloc[0] == 2


def test_procesamiento_ignora_registros_nulos(tmp_path):
    """Caso límite: descarta filas que tengan valores nulos en fecha u origen_id."""
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir()
    processed_dir.mkdir()

    csv_nulos = raw_dir / "datos_abiertos_2025_02.csv"
    df = pd.DataFrame(
        {
            "inicio_del_viaje": [
                "2025-02-01 10:00:00",
                None,
                "2025-02-01 11:00:00",
            ],
            "origen_id": [10, 20, None],
        }
    )
    df.to_csv(csv_nulos, index=False)

    processor = DatasetRegresionProcessor(
        raw_dir=raw_dir, processed_dir=processed_dir
    )
    output_path = processor.construir_dataset_regresion("test_nulos.parquet")
    df_res = pd.read_parquet(output_path)
    assert df_res["demanda_viajes"].sum() == 1


def test_fin_de_semana_identificacion_correcta(tmp_path):
    """Caso normal: verifica que el indicador de fin de semana funcione correctamente para sábado y domingo."""
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir()
    processed_dir.mkdir()

    csv_fds = raw_dir / "datos_abiertos_2025_01.csv"
    df = pd.DataFrame(
        {
            "inicio_del_viaje": [
                "2025-01-04 12:00:00",
                "2025-01-05 14:00:00",
            ],
            "origen_id": [10, 10],
        }
    )
    df.to_csv(csv_fds, index=False)

    processor = DatasetRegresionProcessor(
        raw_dir=raw_dir, processed_dir=processed_dir
    )
    output_path = processor.construir_dataset_regresion("test_fds.parquet")
    df_res = pd.read_parquet(output_path)
    assert all(df_res["es_fin_de_semana"] == 1)


@pytest.mark.parametrize(
    "nombre_archivo, valido",
    [
        ("datos_abiertos_2024_10.csv", True),
        ("datos_abiertos_2025_06.csv", True),
        ("datos_abiertos_2026_09.csv", True),
        ("datos_abiertos_2024_09.csv", False),
        ("datos_abiertos_2026_10.csv", False),
        ("archivo_desconocido.csv", False),
    ],
)
def test_filtrado_rangos_fechas_csv(tmp_path, nombre_archivo, valido):
    """Prueba parametrizada: valida que el procesador acepte exclusivamente archivos en el rango Oct 2024 - Sep 2026."""
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir(exist_ok=True)
    processed_dir.mkdir(exist_ok=True)

    csv_path = raw_dir / nombre_archivo
    df = pd.DataFrame(
        {"inicio_del_viaje": ["2025-01-01 10:00:00"], "origen_id": [10]}
    )
    df.to_csv(csv_path, index=False)

    processor = DatasetRegresionProcessor(
        raw_dir=raw_dir, processed_dir=processed_dir
    )

    if valido:
        out = processor.construir_dataset_regresion("test_param.parquet")
        assert out.exists()
    else:
        with pytest.raises(
            (ValueError, FileNotFoundError),
            match="No se encontraron archivos CSV",
        ):
            processor.construir_dataset_regresion("test_param.parquet")


def test_dataset_salida_contiene_todas_las_columnas_requeridas(tmp_path):
    """Caso normal: verifica que el parquet de salida contenga todas las columnas estructurales obligatorias."""
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir()
    processed_dir.mkdir()

    csv_path = raw_dir / "datos_abiertos_2025_03.csv"
    df = pd.DataFrame(
        {"inicio_del_viaje": ["2025-03-01 12:00:00"], "origen_id": [5]}
    )
    df.to_csv(csv_path, index=False)

    processor = DatasetRegresionProcessor(
        raw_dir=raw_dir, processed_dir=processed_dir
    )
    out = processor.construir_dataset_regresion("test_cols.parquet")
    df_res = pd.read_parquet(out)

    columnas_obligatorias = [
        "origen_id",
        "fecha_hora_inicio",
        "demanda_viajes",
        "hora",
        "dia",
        "dia_semana",
        "mes",
        "anio",
        "es_fin_de_semana",
    ]
    for col in columnas_obligatorias:
        assert col in df_res.columns


def test_agregacion_correcta_multiples_viajes_misma_hora(tmp_path):
    """Caso normal: comprueba que multiples viajes en la misma hora y estación se agrupen sumando la demanda."""
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir()
    processed_dir.mkdir()

    csv_path = raw_dir / "datos_abiertos_2025_04.csv"
    df = pd.DataFrame(
        {
            "inicio_del_viaje": [
                "2025-04-01 14:05:00",
                "2025-04-01 14:20:00",
                "2025-04-01 14:50:00",
            ],
            "origen_id": [100, 100, 100],
        }
    )
    df.to_csv(csv_path, index=False)

    processor = DatasetRegresionProcessor(
        raw_dir=raw_dir, processed_dir=processed_dir
    )
    out = processor.construir_dataset_regresion("test_suma.parquet")
    df_res = pd.read_parquet(out)

    assert len(df_res) == 1
    assert df_res["demanda_viajes"].iloc[0] == 3
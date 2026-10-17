
import hashlib

from tc5061_proyecto_equipo17.dataset import (
    calcular_sha256,
    normalizar_columna,
)


def test_normalizar_columnas():
    """Verifica la normalización de columnas de MiBici."""
    assert normalizar_columna("Viaje_Id") == "viaje_id"
    assert normalizar_columna("Usuario_Id") == "usuario_id"
    assert normalizar_columna("Género") == "genero"
    assert normalizar_columna("Año_de_nacimiento") == "ano_de_nacimiento"
    assert normalizar_columna("Inicio_del_viaje") == "inicio_del_viaje"
    assert normalizar_columna("Origen_Id") == "origen_id"


def test_normalizar_espacios_y_simbolos():
    """Verifica que se eliminen espacios y símbolos."""
    assert normalizar_columna(" Duración (minutos) ") == "duracion_minutos"
    assert normalizar_columna("ESTACIÓN ORIGEN") == "estacion_origen"


def test_calcular_sha256(tmp_path):
    """Comprueba la integridad del hash SHA-256."""
    archivo = tmp_path / "prueba.csv"
    contenido = b"viaje_id,origen_id\n1,10\n"
    archivo.write_bytes(contenido)

    hash_esperado = hashlib.sha256(contenido).hexdigest()
    hash_obtenido = calcular_sha256(archivo)

    assert hash_obtenido == hash_esperado
    assert len(hash_obtenido) == 64


def test_sha256_detecta_cambios(tmp_path):
    """Verifica que el hash cambie al modificar el archivo."""
    archivo = tmp_path / "prueba.csv"

    archivo.write_text("viaje_id\n1\n", encoding="utf-8")
    hash_original = calcular_sha256(archivo)

    archivo.write_text("viaje_id\n2\n", encoding="utf-8")
    hash_modificado = calcular_sha256(archivo)

    assert hash_original != hash_modificado


def test_procesamiento_csv_a_parquet(tmp_path, monkeypatch):
    """Verifica limpieza, transformación y generación de Parquet."""
    import pandas as pd
    import tc5061_proyecto_equipo17.dataset as dataset

    # Carpetas temporales: no modifican los datos reales.
    carpeta_raw = tmp_path / "raw"
    carpeta_processed = tmp_path / "processed"

    carpeta_raw.mkdir()
    carpeta_processed.mkdir()

    # Redirigir la salida del pipeline a la carpeta temporal.
    monkeypatch.setattr(
        dataset, "PROCESSED_DATA_DIR", carpeta_processed
    )

    # CSV pequeño con registros válidos e inválidos.
    datos = pd.DataFrame(
        {
            "Viaje_Id": ["1", "2", "2", "3", "4"],
            "Inicio_del_viaje": [
                "2025-01-01 08:00:00",
                "2025-01-01 09:00:00",
                "2025-01-01 09:00:00",
                "2025-01-01 10:00:00",
                "2025-01-01 11:00:00",
            ],
            "Fin_del_viaje": [
                "2025-01-01 08:15:00",
                "2025-01-01 09:20:00",
                "2025-01-01 09:20:00",
                "2025-01-01 09:50:00",
                "2025-01-01 11:10:00",
            ],
            "Origen_Id": [10, 20, 20, 30, None],
            "Destino_Id": [11, 21, 21, 31, 41],
        }
    )

    archivo_csv = carpeta_raw / "mibici_prueba.csv"
    datos.to_csv(archivo_csv, index=False, encoding="cp1252")

    # Ejecutar la función real del pipeline.
    resultado = dataset.procesar_archivo(archivo_csv)

    # De 5 registros, solo 2 deben conservarse.
    assert resultado["registros_originales"] == 5
    assert resultado["registros_procesados"] == 2
    assert resultado["registros_descartados"] == 3

    # Comprobar que se generó el archivo Parquet.
    archivo_parquet = (
        carpeta_processed / "mibici_prueba_procesado.parquet"
    )
    assert archivo_parquet.exists()

    # Comprobar el contenido del Parquet.
    procesados = pd.read_parquet(archivo_parquet)

    assert len(procesados) == 2
    assert set(procesados["viaje_id"].astype(str)) == {"1", "2"}
    assert procesados["duracion_minutos"].tolist() == [15.0, 20.0]

    # Comprobar que se excluyen los datos personales.
    assert "usuario_id" not in procesados.columns
    assert "genero" not in procesados.columns
    assert "ano_de_nacimiento" not in procesados.columns

    # Verificar la integridad del archivo generado.
    assert (
        resultado["sha256_procesado"]
        == dataset.calcular_sha256(archivo_parquet)
    )


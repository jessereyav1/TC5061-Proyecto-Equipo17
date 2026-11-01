"""Pruebas de seguridad para descargas de MiBici."""

import pytest

from tc5061_proyecto_equipo17.dataset import (
    RedireccionMiBici,
    validar_url_mibici,
)


@pytest.mark.parametrize(
    "url",
    [
        "https://mibici.net/archivo.csv",
        "https://www.mibici.net/archivo.csv",
        "https://datos.mibici.net/archivo.csv",
    ],
)
def test_url_oficial_permitida(url):
    assert validar_url_mibici(url) == url


@pytest.mark.parametrize(
    "url",
    [
        "http://mibici.net/archivo.csv",
        "file:///etc/passwd",
        "https://mibici.net.ejemplo.com/archivo.csv",
        "https://usuario:clave@mibici.net/archivo.csv",
        "https://mibici.net:8080/archivo.csv",
        "https://evil.com/archivo.csv",
    ],
)
def test_url_insegura_rechazada(url):
    with pytest.raises(ValueError):
        validar_url_mibici(url)


def test_redireccion_externa_bloqueada():
    controlador = RedireccionMiBici()

    with pytest.raises(ValueError):
        controlador.redirect_request(
            None,
            None,
            302,
            "Found",
            {},
            "https://evil.com/datos.csv",
        )


def test_redireccion_http_bloqueada():
    controlador = RedireccionMiBici()

    with pytest.raises(ValueError):
        controlador.redirect_request(
            None,
            None,
            302,
            "Found",
            {},
            "http://mibici.net/datos.csv",
        )


def test_descarga_bloquea_url_insegura_antes_de_conectar(tmp_path):
    """Una URL insegura no debe generar conexiones ni archivos."""
    from unittest.mock import patch

    from tc5061_proyecto_equipo17.dataset import descargar_archivo

    destino = tmp_path / "prueba.csv"

    with patch(
        "tc5061_proyecto_equipo17.dataset.build_opener"
    ) as simulador:
        with pytest.raises(ValueError):
            descargar_archivo(
                destino=destino,
                url="file:///etc/passwd",
            )

        simulador.assert_not_called()

    assert not destino.exists()
    assert not destino.with_name("prueba.csv.part").exists()

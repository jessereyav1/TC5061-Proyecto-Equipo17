"""Pruebas unitarias del calculador FinOps de MiBici P2."""

import pytest

from scripts.finops_p2 import calcular_costo_por_inferencia


def test_calculo_basico():
    resultado = calcular_costo_por_inferencia(
        costo_hora=0.12,
        duracion_segundos=3600,
        inferencias_exitosas=36000,
    )

    assert resultado["costo_periodo_usd"] == pytest.approx(0.12)
    assert resultado["inferencias_exitosas"] == 36000
    assert resultado["costo_por_inferencia_usd"] == pytest.approx(
        0.0000033333333333
    )
    assert resultado["inferencias_por_dolar"] == pytest.approx(300000)


def test_duracion_en_segundos():
    resultado = calcular_costo_por_inferencia(
        costo_hora=0.36,
        duracion_segundos=600,
        inferencias_exitosas=3000,
    )

    assert resultado["costo_periodo_usd"] == pytest.approx(0.06)
    assert resultado["costo_por_inferencia_usd"] == pytest.approx(0.00002)


def test_rechaza_costo_negativo():
    with pytest.raises(ValueError):
        calcular_costo_por_inferencia(-0.10, 3600, 100)


def test_rechaza_duracion_invalida():
    with pytest.raises(ValueError):
        calcular_costo_por_inferencia(0.12, 0, 100)


def test_rechaza_cero_inferencias():
    with pytest.raises(ValueError):
        calcular_costo_por_inferencia(0.12, 3600, 0)


def test_costo_cero():
    resultado = calcular_costo_por_inferencia(
        costo_hora=0,
        duracion_segundos=3600,
        inferencias_exitosas=100,
    )

    assert resultado["costo_periodo_usd"] == 0
    assert resultado["costo_por_inferencia_usd"] == 0
    assert resultado["inferencias_por_dolar"] is None


def test_escenarios_generan_tres_niveles():
    from scripts.finops_p2 import proyectar_escenarios

    tabla = proyectar_escenarios(
        costo_hora=0.12,
        rps_base=10,
        horas_mes=720,
    )

    assert tabla["escenario"].tolist() == [
        "Baja", "Normal", "Alta"
    ]
    assert tabla["rps_proyectadas"].tolist() == [
        2.5, 5.0, 10.0
    ]


def test_costo_unitario_disminuye_con_volumen():
    from scripts.finops_p2 import proyectar_escenarios

    tabla = proyectar_escenarios(0.12, 10, 720)

    costos = tabla["costo_por_inferencia_usd"].tolist()

    assert costos[0] > costos[1] > costos[2]
    assert tabla["costo_mensual_usd"].tolist() == pytest.approx(
        [86.4, 86.4, 86.4]
    )


def test_escenarios_rechazan_parametros_invalidos():
    from scripts.finops_p2 import proyectar_escenarios

    with pytest.raises(ValueError):
        proyectar_escenarios(-0.12, 10)

    with pytest.raises(ValueError):
        proyectar_escenarios(0.12, 0)

    with pytest.raises(ValueError):
        proyectar_escenarios(0.12, 10, horas_mes=0)


def test_calculo_mensual_con_almacenamiento():
    from scripts.finops_p2 import calcular_costo_mensual

    resultado = calcular_costo_mensual(
        costo_hora=0.12,
        horas_mes=720,
        almacenamiento_usd_mes=5,
        inferencias_mes=12960000,
    )

    assert resultado["compute_usd"] == pytest.approx(86.4)
    assert resultado["almacenamiento_usd"] == pytest.approx(5)
    assert resultado["costo_total_usd"] == pytest.approx(91.4)
    assert resultado["costo_por_inferencia_usd"] == pytest.approx(
        91.4 / 12960000
    )


def test_calculo_mensual_sin_almacenamiento():
    from scripts.finops_p2 import calcular_costo_mensual

    resultado = calcular_costo_mensual(
        costo_hora=0.20,
        horas_mes=100,
        almacenamiento_usd_mes=0,
        inferencias_mes=100000,
    )

    assert resultado["costo_total_usd"] == pytest.approx(20)
    assert resultado["costo_por_inferencia_usd"] == pytest.approx(0.0002)


def test_calculo_mensual_rechaza_valores_invalidos():
    from scripts.finops_p2 import calcular_costo_mensual

    with pytest.raises(ValueError):
        calcular_costo_mensual(-0.12, 720, 5, 1000)

    with pytest.raises(ValueError):
        calcular_costo_mensual(0.12, 720, -5, 1000)

    with pytest.raises(ValueError):
        calcular_costo_mensual(0.12, 0, 5, 1000)

    with pytest.raises(ValueError):
        calcular_costo_mensual(0.12, 720, 5, 0)


def test_droplet_cobro_por_horas():
    from scripts.finops_p2 import costo_droplet

    assert costo_droplet(100, 0.03571, 24) == pytest.approx(3.571)
    assert costo_droplet(360, 0.03571, 24) == pytest.approx(12.8556)


def test_droplet_respeta_tope_mensual():
    from scripts.finops_p2 import costo_droplet

    assert costo_droplet(720, 0.03571, 24) == pytest.approx(24)
    assert costo_droplet(1000, 0.03571, 24) == pytest.approx(24)


def test_droplet_rechaza_valores_negativos():
    from scripts.finops_p2 import costo_droplet

    with pytest.raises(ValueError):
        costo_droplet(-1, 0.03571, 24)

    with pytest.raises(ValueError):
        costo_droplet(100, -0.03571, 24)

    with pytest.raises(ValueError):
        costo_droplet(100, 0.03571, -24)

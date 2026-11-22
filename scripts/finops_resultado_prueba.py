"""Costo equivalente de la prueba Locust MiBici P2."""

import json
from pathlib import Path

from scripts.finops_p2 import calcular_costo_por_inferencia

BASE = Path("reports/finops")


def main():
    carga = json.loads(
        Path("reports/carga/resumen_carga.json").read_text(
            encoding="utf-8"
        )
    )
    nube = json.loads(
        (BASE / "referencia_nube.json").read_text(
            encoding="utf-8"
        )
    )

    prueba = carga["prueba_carga"]

    resultado = calcular_costo_por_inferencia(
        costo_hora=nube["precio_publicado_usd_hora"],
        duracion_segundos=prueba["duracion_segundos"],
        inferencias_exitosas=prueba["peticiones_exitosas"],
    )

    resultado.update({
        "duracion_segundos": prueba["duracion_segundos"],
        "tarifa_referencia_usd_hora":
            nube["precio_publicado_usd_hora"],
        "costo_por_millon_usd":
            resultado["costo_por_inferencia_usd"] * 1_000_000,
        "tipo_resultado":
            "Estimacion prorrateada, no costo facturado",
    })

    BASE.mkdir(parents=True, exist_ok=True)

    salida = BASE / "resultado_prueba.json"
    salida.write_text(
        json.dumps(resultado, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("\nFINOPS - RESULTADO DE LA PRUEBA")
    print("Inferencias exitosas:", resultado["inferencias_exitosas"])
    print("Costo equivalente USD:",
          round(resultado["costo_periodo_usd"], 8))
    print("Costo por millon USD:",
          round(resultado["costo_por_millon_usd"], 4))
    print("Archivo generado:", salida)


if __name__ == "__main__":
    main()

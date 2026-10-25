"""Punto de entrada común para MLflow Projects - Equipo 17."""

import argparse
import subprocess
import sys

from tc5061_proyecto_equipo17.config import (
    PROCESSED_DATA_DIR,
    SEED,
)


MODULES = {
    "p1": "tc5061_proyecto_equipo17.modeling.train_p1",
    "p2": "tc5061_proyecto_equipo17.modeling.train_p2",
    "p3": "tc5061_proyecto_equipo17.modeling.train_p3",
}

SUPPORTED_MODELS = {
    "p1": {"rf"},
    "p2": {"rf"},
    "p3": {"kmeans"},
}


def ejecutar_modulo(modulo, argumentos=None):
    """Ejecuta un módulo Python y detiene el proceso si falla."""
    comando = [
        sys.executable,
        "-m",
        modulo,
    ]

    if argumentos:
        comando.extend(argumentos)

    print("Ejecutando:", " ".join(comando), flush=True)

    subprocess.run(comando, check=True)



def preparar_p1():
    """Descarga y prepara los 24 meses necesarios para P1."""
    import pandas as pd

    dataset = PROCESSED_DATA_DIR / "dataset_regresion_horario.parquet"
    features = PROCESSED_DATA_DIR / "features_p1.parquet"

    print("\nPreparando datos de P1...", flush=True)

    # Verificar y descargar los 24 CSV.
    ejecutar_modulo("tc5061_proyecto_equipo17.ingest_p1")

    def cobertura_completa(ruta):
        if not ruta.is_file():
            return False

        try:
            fechas = pd.read_parquet(
                ruta,
                columns=["fecha_hora_inicio"],
            )["fecha_hora_inicio"]

            fechas = pd.to_datetime(fechas)

            meses = set(fechas.dt.to_period("M").astype(str))

            esperados = {
                str(periodo)
                for periodo in pd.period_range(
                    "2024-10", "2026-09", freq="M"
                )
            }

            return esperados.issubset(meses)

        except (OSError, ValueError, KeyError, TypeError):
            return False

    # Regenerar dataset cuando esté incompleto.
    if not cobertura_completa(dataset):
        print("Generando dataset completo de P1...", flush=True)
        ejecutar_modulo(
            "tc5061_proyecto_equipo17.dataset_p1"
        )

        if not cobertura_completa(dataset):
            raise RuntimeError(
                "El dataset de P1 no contiene los 24 meses requeridos."
            )
        # Si cambia el dataset, hay que actualizar las features.
        if features.exists():
            features.unlink()
    else:
        print("Dataset de P1 con cobertura completa.", flush=True)

    # Regenerar características si faltan meses.
    if not cobertura_completa(features):
        print("Generando características de P1...", flush=True)
        ejecutar_modulo(
            "tc5061_proyecto_equipo17.features_p1"
        )

    if not cobertura_completa(features):
        raise RuntimeError(
            "Las características de P1 no contienen los 24 meses."
        )

    print("Preparación de P1 completada.", flush=True)

def main():
    parser = argparse.ArgumentParser(
        description="Entrenamiento reproducible MiBici - Equipo 17"
    )

    parser.add_argument(
        "--question",
        choices=MODULES.keys(),
        required=True,
    )

    parser.add_argument("--model", required=True)
    parser.add_argument("--n-estimators", type=int, default=0)
    parser.add_argument("--max-depth", type=int, default=0)
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=SEED)

    args = parser.parse_args()

    if args.model not in SUPPORTED_MODELS[args.question]:
        parser.error(
            f"Modelo '{args.model}' no soportado para {args.question}. "
            f"Disponibles: {sorted(SUPPORTED_MODELS[args.question])}"
        )

    parametros = []

    if args.question in ("p1", "p2"):
        if args.k != 0:
            parser.error("--k solamente aplica a P3")

        if args.n_estimators < 0 or args.max_depth < 0:
            parser.error("Los hiperparámetros no pueden ser negativos")

        if args.n_estimators > 0:
            parametros.extend([
                "--n-estimators",
                str(args.n_estimators),
            ])

        if args.max_depth > 0:
            parametros.extend([
                "--max-depth",
                str(args.max_depth),
            ])

    elif args.question == "p3":
        if args.n_estimators != 0 or args.max_depth != 0:
            parser.error(
                "--n-estimators y --max-depth no aplican a P3"
            )

        if args.k < 0:
            parser.error("--k no puede ser negativo")

        if args.k > 0:
            parametros.extend(["--k", str(args.k)])

        parametros.extend(["--semilla", str(args.seed)])

    if args.question == "p1":
        preparar_p1()

    ejecutar_modulo(
        MODULES[args.question],
        parametros,
    )


if __name__ == "__main__":
    main()

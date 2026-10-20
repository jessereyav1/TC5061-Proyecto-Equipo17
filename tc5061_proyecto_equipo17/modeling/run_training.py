"""Punto de entrada común para MLflow Projects - Equipo 17."""

import argparse
import subprocess
import sys

from tc5061_proyecto_equipo17.config import (
    PROCESSED_DATA_DIR,
    SEED,
)


MODULES = {
    "p1": "tc5061_proyecto_equipo17.modeling.train_regression",
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
    """Genera los archivos necesarios para entrenar P1."""

    dataset = PROCESSED_DATA_DIR / "dataset_regresion_horario.parquet"
    features = PROCESSED_DATA_DIR / "features_regresion.parquet"

    print("\nPreparando datos de P1...", flush=True)

    if not dataset.is_file():
        ejecutar_modulo(
            "tc5061_proyecto_equipo17.dataset_regression"
        )
    else:
        print("Dataset de regresión existente.", flush=True)

    if not features.is_file():
        ejecutar_modulo(
            "tc5061_proyecto_equipo17.features_regression"
        )
    else:
        print("Características de regresión existentes.", flush=True)

    if not features.is_file():
        raise FileNotFoundError(
            f"No se generó el archivo requerido: {features}"
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

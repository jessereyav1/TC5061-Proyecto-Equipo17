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
    "p3": {"kmeans", "kmeans-escalado", "gmm"},
}

# Opciones exclusivas de P3
VARIABLES_P3 = {"completas", "solo-horas"}
ETAPAS_P3 = {"baseline", "candidate"}


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
    features = PROCESSED_DATA_DIR / "features_regresion.parquet"

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
            "tc5061_proyecto_equipo17.dataset_regression"
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
            "tc5061_proyecto_equipo17.features_regression"
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
    # Opciones exclusivas de P3
    parser.add_argument("--variables", default="completas")
    parser.add_argument("--etapa", default="baseline")
    parser.add_argument("--buscar-k", type=int, choices=[0, 1], default=0)

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

        if (
            args.variables != "completas"
            or args.etapa != "baseline"
            or args.buscar_k != 0
        ):
            parser.error("--variables, --etapa y --buscar-k solo aplican a P3")

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

        if args.variables not in VARIABLES_P3:
            parser.error(f"--variables debe ser una de {sorted(VARIABLES_P3)}")

        if args.etapa not in ETAPAS_P3:
            parser.error(f"--etapa debe ser una de {sorted(ETAPAS_P3)}")

        # Para P3, --model es el algoritmo (kmeans, kmeans-escalado o gmm)
        parametros.extend([
            "--algoritmo", args.model,
            "--variables", args.variables,
            "--etapa", args.etapa,
        ])

        if args.buscar_k == 1:
            # La búsqueda recorre k; no se fija un k único
            parametros.append("--buscar-k")
        else:
            # MLflow guarda -P k en la corrida antes de entrenar y no permite cambiarlo
            # después, así que en P3 el k se pide siempre de forma explícita.
            if args.k < 2:
                parser.error("P3 requiere -P k=<entero mayor o igual a 2> (o -P buscar_k=1)")
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

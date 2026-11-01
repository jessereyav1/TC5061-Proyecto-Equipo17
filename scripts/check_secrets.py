"""Comprueba posibles secretos sin mostrar sus valores."""

import json
import subprocess
import sys
from pathlib import Path


def main():
    archivos = subprocess.run(
        ["git", "ls-files", "-z"],
        capture_output=True,
        check=True,
    ).stdout.split(b"\0")

    # Notebooks con salidas PNG verificadas durante la auditoria.
    # La exclusion es temporal y debe revisarse si cambian.
    notebooks_excluidos = {
        "notebooks/1.0-exploracion-inicial.ipynb",
        "notebooks/1.0-mdlc-perfiles-horarios.ipynb",
    }

    rutas = [
        a.decode("utf-8")
        for a in archivos
        if a and not a.startswith(b"reports/security/") and a.decode("utf-8") not in notebooks_excluidos
    ]

    comando = [
        "detect-secrets",
        "scan",
        *rutas,
    ]

    resultado = subprocess.run(
        comando,
        capture_output=True,
        text=True,
    )

    if resultado.returncode != 0:
        print("Error al ejecutar detect-secrets.", file=sys.stderr)
        return 2

    datos = json.loads(resultado.stdout)
    hallazgos = datos.get("results", {})
    total = sum(len(v) for v in hallazgos.values())

    print(f"Archivos con hallazgos: {len(hallazgos)}")
    print(f"Posibles secretos detectados: {total}")

    for archivo, coincidencias in hallazgos.items():
        for coincidencia in coincidencias:
            print(
                f"{archivo}: línea {coincidencia.get('line_number')} "
                f"({coincidencia.get('type')})"
            )

    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())

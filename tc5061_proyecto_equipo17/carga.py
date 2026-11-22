"""Prueba de carga de la API P2 con Locust y consumo de recursos con docker stats (Actividad 11).

Comandos (desde la raíz del repositorio, con el sistema levantado con ``docker compose up``)::

    python -m tc5061_proyecto_equipo17.carga generar-payloads   # entradas reales de julio 2025
    python -m tc5061_proyecto_equipo17.carga ejecutar           # Locust + docker stats + análisis
    python -m tc5061_proyecto_equipo17.carga analizar           # repite solo el análisis

``ejecutar`` lanza el escenario de ``carga/locustfile.py`` sin interfaz y, en paralelo,
toma una muestra de ``docker stats`` cada 2 segundos. Al terminar deja en
``reports/carga/``:

- ``peticiones.csv``: cada petición con su latencia (para percentiles exactos).
- ``recursos_docker.csv``: CPU y memoria de cada contenedor en el tiempo.
- ``locust_*.csv`` y ``reporte_locust.html``: salidas propias de Locust.
- ``resumen_etapas.csv``: desempeño y recursos en cada escalón de usuarios.
- ``resumen_carga.json`` y ``resumen_carga.md``: cifras para el análisis FinOps, con los
  mismos nombres de campo que ``reports/finops/parametros.json``.

y las gráficas en ``reports/figures/carga_*.png``.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
import urllib.request

from loguru import logger
import matplotlib

matplotlib.use("Agg")  # dibuja a archivo, sin abrir ventanas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer

from tc5061_proyecto_equipo17.config import (
    FIGURES_DIR,
    MONITOREO_DATA_DIR,
    PROJ_ROOT,
    REPORTS_DIR,
    SEED,
)
from tc5061_proyecto_equipo17.features import P2_FEATURES

app = typer.Typer(help="Prueba de carga de la API P2 (Actividad 11).")

LOCUSTFILE = PROJ_ROOT / "carga" / "locustfile.py"
ARCHIVO_PAYLOADS = PROJ_ROOT / "carga" / "payloads_p2.json"
SALIDA = REPORTS_DIR / "carga"

CONTENEDOR_API = "mibici-p2-api"
CONTENEDOR_MLFLOW = "mibici-p2-mlflow"

PREDICT = "POST /predict"
# Objetivo de servicio: el 95 % de las predicciones en menos de 200 ms y menos de 1 % de errores
SLO_P95_MS = 200.0
SLO_ERROR = 0.01
# Segundos al inicio de cada escalón que no se cuentan, mientras se crean los usuarios
SEGUNDOS_ESTABILIZACION = 10

# Mismas reglas que valida app.py (ModelInput) para que ninguna entrada real sea rechazada
ENTEROS = ["hora_dia", "dia_semana", "fin_de_semana", "riesgo_ahora", "neto"]
CAPACIDAD_MAXIMA = 200

# Paleta categórica validada (guía de visualización), en orden fijo
COLOR = {
    "superficie": "#fcfcfb",
    "texto": "#0b0b0b",
    "texto_sec": "#52514e",
    "rejilla": "#e6e5e1",
    "serie1": "#2a78d6",  # azul: API, éxito, p50
    "serie2": "#eb6834",  # naranja: MLflow, p95
    "error": "#e34948",  # rojo: fallas
    "limite": "#8a8984",  # gris: umbrales y límites
}


# ---------------------------------------------------------------------------
# Entradas de prueba
# ---------------------------------------------------------------------------


def seleccionar_payloads(actual: pd.DataFrame, n: int = 1000, semilla: int = SEED) -> list[dict]:
    """Toma ``n`` filas reales de la ventana de evaluación y las deja como JSON para /predict.

    Se descartan las filas que app.py rechazaría (flujo neto mayor que la capacidad, etc.):
    en la prueba de carga las entradas válidas deben ser válidas.
    """
    filas = actual.loc[actual["ventana"] == "evaluacion", P2_FEATURES].copy()
    validas = (
        (filas["capacidad"] > 0)
        & (filas["capacidad"] <= CAPACIDAD_MAXIMA)
        & filas["pct_ocupacion"].between(0, 1)
        & (filas["neto"].abs() <= filas["capacidad"])
        & (filas["neto_ult_3h"].abs() <= 3 * filas["capacidad"])
    )
    filas = filas[validas].sample(n=min(n, int(validas.sum())), random_state=semilla)
    filas[ENTEROS] = filas[ENTEROS].astype(int)
    filas["capacidad"] = filas["capacidad"].astype(float)
    filas["pct_ocupacion"] = filas["pct_ocupacion"].astype(float).round(6)
    filas["neto_ult_3h"] = filas["neto_ult_3h"].astype(float)
    # to_json convierte los tipos de numpy a tipos JSON nativos
    return json.loads(filas.to_json(orient="records"))


# ---------------------------------------------------------------------------
# docker stats
# ---------------------------------------------------------------------------

_UNIDADES = {"b": 1 / 1024**2, "kib": 1 / 1024, "kb": 1 / 1024, "mib": 1.0, "mb": 1.0,
             "gib": 1024.0, "gb": 1024.0, "tib": 1024.0**2}


def leer_porcentaje(texto: str) -> float:
    """``'12.34%'`` -> ``12.34``. Docker reporta ``'--'`` cuando aún no hay dato."""
    texto = texto.strip().rstrip("%")
    return float(texto) if texto not in ("", "--") else float("nan")


def leer_memoria_mb(texto: str) -> float:
    """``'153.2MiB'`` o ``'1.5GiB'`` -> megabytes (MiB)."""
    coincidencia = re.fullmatch(r"\s*([\d.]+)\s*([A-Za-z]+)\s*", texto)
    if not coincidencia:
        return float("nan")
    valor, unidad = coincidencia.groups()
    return float(valor) * _UNIDADES[unidad.lower()]


def leer_docker_stats(lineas: list[str], momento: float) -> list[dict]:
    """Convierte la salida de ``docker stats --format '{{json .}}'`` en filas."""
    filas = []
    for linea in lineas:
        if not linea.strip():
            continue
        dato = json.loads(linea)
        uso, limite = (parte for parte in dato["MemUsage"].split("/"))
        filas.append(
            {
                "timestamp": momento,
                "contenedor": dato["Name"],
                "cpu_pct": leer_porcentaje(dato["CPUPerc"]),
                "mem_mb": leer_memoria_mb(uso),
                "mem_limite_mb": leer_memoria_mb(limite),
            }
        )
    return filas


class MonitorRecursos(threading.Thread):
    """Toma muestras de ``docker stats`` cada ``intervalo`` segundos hasta que se detiene."""

    def __init__(self, contenedores: list[str], intervalo: float = 2.0):
        super().__init__(daemon=True)
        self.contenedores = contenedores
        self.intervalo = intervalo
        self.filas: list[dict] = []
        self._alto = threading.Event()

    def run(self):
        while not self._alto.is_set():
            momento = time.time()
            salida = subprocess.run(
                ["docker", "stats", "--no-stream", "--format", "{{json .}}", *self.contenedores],
                capture_output=True, text=True, check=False,
            )
            if salida.returncode == 0:
                self.filas.extend(leer_docker_stats(salida.stdout.splitlines(), momento))
            else:
                logger.warning(f"docker stats falló: {salida.stderr.strip()}")
            self._alto.wait(max(0.0, self.intervalo - (time.time() - momento)))

    def detener(self) -> pd.DataFrame:
        self._alto.set()
        self.join()
        return pd.DataFrame(self.filas)


# ---------------------------------------------------------------------------
# Análisis
# ---------------------------------------------------------------------------


def asignar_etapa(segundos: pd.Series, etapas: list[tuple]) -> pd.Series:
    """Número de escalón (0, 1, ...) según el segundo de la prueba; -1 al terminar."""
    fines = np.array([fin for fin, _, _ in etapas])
    indice = np.searchsorted(fines, segundos.to_numpy(), side="right")
    return pd.Series(np.where(indice < len(etapas), indice, -1), index=segundos.index)


def _percentil(serie: pd.Series, q: float) -> float:
    return float(np.percentile(serie, q)) if len(serie) else float("nan")


def resumen_por_etapa(
    peticiones: pd.DataFrame, recursos: pd.DataFrame, etapas: list[tuple]
) -> pd.DataFrame:
    """Desempeño de /predict y consumo de la API en cada escalón (sin su estabilización)."""
    inicio = peticiones["timestamp"].min()
    pet = peticiones.assign(segundo=peticiones["timestamp"] - inicio)
    pet["etapa"] = asignar_etapa(pet["segundo"], etapas)
    rec = recursos.assign(segundo=recursos["timestamp"] - inicio) if len(recursos) else recursos

    filas = []
    inicio_etapa = 0
    for i, (fin, usuarios, _) in enumerate(etapas):
        desde = inicio_etapa + SEGUNDOS_ESTABILIZACION
        inicio_etapa = fin
        ventana = pet[(pet["etapa"] == i) & (pet["segundo"] >= desde) & (pet["nombre"] == PREDICT)]
        if ventana.empty:
            continue
        duracion = fin - desde
        exitosas = ventana[ventana["exito"]]
        fila = {
            "etapa": i + 1,
            "usuarios": usuarios,
            "segundos_medidos": duracion,
            "predicciones_exitosas": len(exitosas),
            "predicciones_fallidas": int((~ventana["exito"]).sum()),
            "rps_exitosas": len(exitosas) / duracion,
            "latencia_p50_ms": _percentil(exitosas["latencia_ms"], 50),
            "latencia_p95_ms": _percentil(exitosas["latencia_ms"], 95),
            "latencia_p99_ms": _percentil(exitosas["latencia_ms"], 99),
            "tasa_error": (~ventana["exito"]).mean(),
        }
        if len(rec):
            api = rec[
                (rec["contenedor"] == CONTENEDOR_API) & (rec["segundo"] >= desde) & (rec["segundo"] < fin)
            ]
            fila["api_cpu_pct"] = api["cpu_pct"].mean()
            fila["api_mem_mb"] = api["mem_mb"].mean()
        fila["cumple_slo"] = bool(
            fila["latencia_p95_ms"] <= SLO_P95_MS and fila["tasa_error"] <= SLO_ERROR
        )
        filas.append(fila)
    return pd.DataFrame(filas)


def resumen_global(peticiones: pd.DataFrame, recursos: pd.DataFrame, etapas_df: pd.DataFrame) -> dict:
    """Cifras de la prueba completa con los nombres de ``reports/finops/parametros.json``.

    Solo cuentan como inferencias las peticiones ``POST /predict`` con entrada válida; las
    de salud y las entradas inválidas no.
    """
    inicio, fin = peticiones["timestamp"].min(), peticiones["timestamp"].max()
    duracion = float(fin - inicio)
    predict = peticiones[peticiones["nombre"] == PREDICT]
    exitosas = predict[predict["exito"]]

    def _medias(contenedor: str) -> tuple[float | None, float | None]:
        if not len(recursos):
            return None, None
        sel = recursos[(recursos["contenedor"] == contenedor) & recursos["timestamp"].between(inicio, fin)]
        return round(float(sel["cpu_pct"].mean()), 2), round(float(sel["mem_mb"].mean()), 1)

    api_cpu, api_mem = _medias(CONTENEDOR_API)
    mlflow_cpu, mlflow_mem = _medias(CONTENEDOR_MLFLOW)

    sostenibles = etapas_df[etapas_df["cumple_slo"]] if len(etapas_df) else etapas_df
    capacidad = None
    if len(sostenibles):
        mejor = sostenibles.loc[sostenibles["rps_exitosas"].idxmax()]
        capacidad = {
            "usuarios": int(mejor["usuarios"]),
            "rps_exitosas": round(float(mejor["rps_exitosas"]), 2),
            "latencia_p95_ms": round(float(mejor["latencia_p95_ms"]), 1),
            "api_cpu_pct": round(float(mejor.get("api_cpu_pct", float("nan"))), 1),
        }

    return {
        "prueba_carga": {
            "duracion_segundos": round(duracion, 1),
            "peticiones_exitosas": len(exitosas),
            "peticiones_fallidas": int((~predict["exito"]).sum()),
            "rps_promedio": round(len(exitosas) / duracion, 2) if duracion else None,
            "latencia_p95_ms": round(_percentil(exitosas["latencia_ms"], 95), 1),
            "latencia_p50_ms": round(_percentil(exitosas["latencia_ms"], 50), 1),
            "concurrencia_maxima": int(peticiones["usuarios"].max()),
            "peticiones_totales_todas_las_rutas": len(peticiones),
        },
        "recursos": {
            "api_cpu_promedio_pct": api_cpu,
            "api_ram_promedio_mb": api_mem,
            "mlflow_cpu_promedio_pct": mlflow_cpu,
            "mlflow_ram_promedio_mb": mlflow_mem,
        },
        "capacidad_sostenible": capacidad,
        "slo": {"latencia_p95_ms": SLO_P95_MS, "tasa_error_maxima": SLO_ERROR},
    }


def serie_temporal(peticiones: pd.DataFrame, ventana_s: int = 5) -> pd.DataFrame:
    """Peticiones por segundo, fallas y latencias de /predict en ventanas de ``ventana_s`` s."""
    inicio = peticiones["timestamp"].min()
    pet = peticiones[peticiones["nombre"] == PREDICT].copy()
    pet["bloque"] = ((pet["timestamp"] - inicio) // ventana_s * ventana_s).astype(int)
    grupos = pet.groupby("bloque")
    return pd.DataFrame(
        {
            "usuarios": grupos["usuarios"].max(),
            "rps_exitosas": grupos["exito"].sum() / ventana_s,
            "fallas_por_s": (~pet["exito"]).groupby(pet["bloque"]).sum() / ventana_s,
            "p50_ms": grupos["latencia_ms"].median(),
            "p95_ms": grupos["latencia_ms"].quantile(0.95),
        }
    ).reset_index().rename(columns={"bloque": "segundo"})


# ---------------------------------------------------------------------------
# Gráficas
# ---------------------------------------------------------------------------


def _estilo(ax, titulo: str, ylabel: str) -> None:
    ax.set_facecolor(COLOR["superficie"])
    ax.set_title(titulo, loc="left", fontsize=11, color=COLOR["texto"], fontweight="bold")
    ax.set_ylabel(ylabel, color=COLOR["texto_sec"], fontsize=9)
    ax.grid(axis="y", color=COLOR["rejilla"], linewidth=0.8)
    ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(COLOR["rejilla"])
    ax.tick_params(colors=COLOR["texto_sec"], labelsize=8)


def _guardar(fig, ruta: Path) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=150, facecolor=COLOR["superficie"], bbox_inches="tight")
    plt.close(fig)
    logger.success(f"Gráfica: {ruta}")


def grafica_linea_tiempo(serie: pd.DataFrame, ruta: Path) -> None:
    """Usuarios, peticiones por segundo y latencia a lo largo de la prueba (3 paneles)."""
    fig, ejes = plt.subplots(3, 1, figsize=(9, 8), sharex=True, facecolor=COLOR["superficie"])
    x = serie["segundo"]

    ejes[0].step(x, serie["usuarios"], where="post", color=COLOR["serie1"], linewidth=2)
    _estilo(ejes[0], "Usuarios virtuales activos", "usuarios")

    ejes[1].plot(x, serie["rps_exitosas"], color=COLOR["serie1"], linewidth=2, label="exitosas")
    ejes[1].plot(x, serie["fallas_por_s"], color=COLOR["error"], linewidth=2, label="fallidas")
    _estilo(ejes[1], "Predicciones por segundo", "peticiones / s")
    ejes[1].legend(frameon=False, fontsize=8, loc="upper left")

    ejes[2].plot(x, serie["p50_ms"], color=COLOR["serie1"], linewidth=2, label="p50")
    ejes[2].plot(x, serie["p95_ms"], color=COLOR["serie2"], linewidth=2, label="p95")
    ejes[2].axhline(SLO_P95_MS, color=COLOR["limite"], linestyle="--", linewidth=1)
    ejes[2].annotate(f"objetivo p95 = {SLO_P95_MS:.0f} ms", (x.iloc[0], SLO_P95_MS),
                     xytext=(2, 4), textcoords="offset points", fontsize=8, color=COLOR["texto_sec"])
    _estilo(ejes[2], "Latencia de POST /predict", "milisegundos")
    ejes[2].legend(frameon=False, fontsize=8, loc="upper left")
    ejes[2].set_xlabel("segundo de la prueba", color=COLOR["texto_sec"], fontsize=9)

    fig.tight_layout()
    _guardar(fig, ruta)


def grafica_saturacion(etapas_df: pd.DataFrame, ruta: Path) -> None:
    """Rendimiento y latencia p95 contra el número de usuarios (curva de saturación)."""
    fig, (izq, der) = plt.subplots(1, 2, figsize=(9, 3.6), facecolor=COLOR["superficie"])
    u = etapas_df["usuarios"]

    izq.plot(u, etapas_df["rps_exitosas"], color=COLOR["serie1"], linewidth=2, marker="o", markersize=6)
    _estilo(izq, "Rendimiento por escalón", "predicciones exitosas / s")
    ultimo = etapas_df.iloc[-1]
    izq.annotate(f"{ultimo['rps_exitosas']:.0f}/s", (ultimo["usuarios"], ultimo["rps_exitosas"]),
                 xytext=(-6, 8), textcoords="offset points", ha="right", fontsize=8, color=COLOR["texto"])

    der.plot(u, etapas_df["latencia_p95_ms"], color=COLOR["serie2"], linewidth=2, marker="o", markersize=6)
    der.axhline(SLO_P95_MS, color=COLOR["limite"], linestyle="--", linewidth=1)
    der.annotate(f"objetivo {SLO_P95_MS:.0f} ms", (u.iloc[0], SLO_P95_MS), xytext=(2, 4),
                 textcoords="offset points", fontsize=8, color=COLOR["texto_sec"])
    _estilo(der, "Latencia p95 por escalón", "milisegundos")

    for ax in (izq, der):
        ax.set_xlabel("usuarios virtuales", color=COLOR["texto_sec"], fontsize=9)
        ax.set_ylim(bottom=0)
    fig.tight_layout()
    _guardar(fig, ruta)


def grafica_recursos(recursos: pd.DataFrame, inicio: float, ruta: Path) -> None:
    """CPU y memoria de cada contenedor durante la prueba (2 paneles)."""
    fig, (cpu, mem) = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True, facecolor=COLOR["superficie"])
    for contenedor, color, etiqueta in (
        (CONTENEDOR_API, COLOR["serie1"], "API"),
        (CONTENEDOR_MLFLOW, COLOR["serie2"], "MLflow"),
    ):
        datos = recursos[recursos["contenedor"] == contenedor]
        x = datos["timestamp"] - inicio
        cpu.plot(x, datos["cpu_pct"], color=color, linewidth=2, label=etiqueta)
        mem.plot(x, datos["mem_mb"], color=color, linewidth=2, label=etiqueta)

    cpu.axhline(100, color=COLOR["limite"], linestyle="--", linewidth=1)
    cpu.annotate("límite de la API: 1 vCPU = 100 %", (0, 100), xytext=(2, 4),
                 textcoords="offset points", fontsize=8, color=COLOR["texto_sec"])
    _estilo(cpu, "CPU por contenedor (docker stats)", "% de un núcleo")
    cpu.legend(frameon=False, fontsize=8, loc="center right")
    _estilo(mem, "Memoria por contenedor (docker stats)", "MiB")
    mem.legend(frameon=False, fontsize=8, loc="center right")
    mem.set_ylim(bottom=0)
    mem.set_xlabel("segundo de la prueba", color=COLOR["texto_sec"], fontsize=9)
    fig.tight_layout()
    _guardar(fig, ruta)


# ---------------------------------------------------------------------------
# Comandos
# ---------------------------------------------------------------------------


def _etapas_del_locustfile() -> list[tuple]:
    """Lee los escalones de ``carga/locustfile.py`` para no duplicarlos."""
    spec = importlib.util.spec_from_file_location("locustfile_p2", LOCUSTFILE)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return list(modulo.ETAPAS)


def _escribir_resumen_md(resumen: dict, etapas_df: pd.DataFrame, ruta: Path) -> None:
    carga, rec, cap = resumen["prueba_carga"], resumen["recursos"], resumen["capacidad_sostenible"]
    lineas = [
        "# Resumen de la prueba de carga (API P2, 1 vCPU / 1 GB)",
        "",
        f"- Duración: {carga['duracion_segundos']} s, hasta {carga['concurrencia_maxima']} usuarios.",
        f"- Predicciones exitosas: {carga['peticiones_exitosas']:,}; fallidas: {carga['peticiones_fallidas']:,}.",
        f"- Rendimiento promedio: {carga['rps_promedio']} predicciones/s.",
        f"- Latencia de /predict: p50 {carga['latencia_p50_ms']} ms, p95 {carga['latencia_p95_ms']} ms.",
        f"- API: CPU promedio {rec['api_cpu_promedio_pct']} %, memoria {rec['api_ram_promedio_mb']} MiB.",
        f"- MLflow: CPU promedio {rec['mlflow_cpu_promedio_pct']} %, memoria {rec['mlflow_ram_promedio_mb']} MiB.",
        "",
        f"Objetivo de servicio: p95 <= {SLO_P95_MS:.0f} ms y errores <= {SLO_ERROR:.0%}.",
        (
            f"Capacidad sostenible: {cap['rps_exitosas']} predicciones/s con {cap['usuarios']} usuarios "
            f"(p95 {cap['latencia_p95_ms']} ms, CPU de la API {cap['api_cpu_pct']} %)."
            if cap else "Ningún escalón cumplió el objetivo de servicio."
        ),
        "",
        "## Por escalón",
        "",
        "| Usuarios | Pred./s | p50 ms | p95 ms | p99 ms | Errores | CPU API % | Mem API MiB | Cumple |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for _, f in etapas_df.iterrows():
        lineas.append(
            f"| {f['usuarios']} | {f['rps_exitosas']:.1f} | {f['latencia_p50_ms']:.0f} | "
            f"{f['latencia_p95_ms']:.0f} | {f['latencia_p99_ms']:.0f} | {f['tasa_error']:.2%} | "
            f"{f.get('api_cpu_pct', float('nan')):.0f} | {f.get('api_mem_mb', float('nan')):.0f} | "
            f"{'sí' if f['cumple_slo'] else 'no'} |"
        )
    ruta.write_text("\n".join(lineas) + "\n", encoding="utf-8")


@app.command("generar-payloads")
def generar_payloads_cmd(n: int = 1000):
    """Guarda en carga/payloads_p2.json ``n`` entradas reales de la ventana 22-31 de julio."""
    actual = pd.read_parquet(MONITOREO_DATA_DIR / "p2_actual.parquet")
    payloads = seleccionar_payloads(actual, n)
    ARCHIVO_PAYLOADS.write_text(json.dumps(payloads, indent=1), encoding="utf-8")
    logger.success(f"{len(payloads)} entradas en {ARCHIVO_PAYLOADS}")


@app.command("analizar")
def analizar(salida: Path = SALIDA):
    """Genera tablas, resumen y gráficas a partir de peticiones.csv y recursos_docker.csv."""
    peticiones = pd.read_csv(salida / "peticiones.csv")
    peticiones["exito"] = peticiones["exito"].astype(str).str.lower().eq("true")
    ruta_recursos = salida / "recursos_docker.csv"
    recursos = pd.read_csv(ruta_recursos) if ruta_recursos.is_file() else pd.DataFrame()

    etapas_df = resumen_por_etapa(peticiones, recursos, _etapas_del_locustfile())
    etapas_df.to_csv(salida / "resumen_etapas.csv", index=False)
    resumen = resumen_global(peticiones, recursos, etapas_df)
    (salida / "resumen_carga.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    _escribir_resumen_md(resumen, etapas_df, salida / "resumen_carga.md")

    grafica_linea_tiempo(serie_temporal(peticiones), FIGURES_DIR / "carga_linea_tiempo.png")
    grafica_saturacion(etapas_df, FIGURES_DIR / "carga_saturacion.png")
    if len(recursos):
        grafica_recursos(recursos, peticiones["timestamp"].min(), FIGURES_DIR / "carga_recursos.png")
    logger.success(f"Resumen: {salida / 'resumen_carga.md'}")
    print((salida / "resumen_carga.md").read_text(encoding="utf-8"))


@app.command("ejecutar")
def ejecutar(host: str = "http://localhost:8000", salida: Path = SALIDA, intervalo: float = 2.0):
    """Ejecuta el escenario de Locust midiendo docker stats y después analiza los resultados."""
    try:
        estado = json.load(urllib.request.urlopen(f"{host}/", timeout=5))
    except OSError as error:
        raise typer.BadParameter(f"La API no responde en {host}: ¿está levantado docker compose? ({error})")
    if not estado.get("modelo_cargado"):
        raise typer.BadParameter("La API responde pero sin el modelo cargado; revisa docker compose logs api.")

    salida.mkdir(parents=True, exist_ok=True)
    monitor = MonitorRecursos([CONTENEDOR_API, CONTENEDOR_MLFLOW], intervalo)
    monitor.start()
    comando = [
        sys.executable, "-m", "locust", "-f", str(LOCUSTFILE), "--headless", "--host", host,
        "--csv", str(salida / "locust"), "--csv-full-history",
        "--html", str(salida / "reporte_locust.html"), "--loglevel", "WARNING",
    ]
    logger.info("Prueba en curso (6 minutos): " + " ".join(comando))
    entorno = {**os.environ, "CARGA_PETICIONES_CSV": str(salida / "peticiones.csv")}
    resultado = subprocess.run(comando, env=entorno, check=False)
    # Locust termina con código 1 si hubo peticiones fallidas; eso es un resultado, no un error
    if resultado.returncode not in (0, 1):
        monitor.detener()
        raise typer.Exit(resultado.returncode)

    recursos = monitor.detener()
    recursos.to_csv(salida / "recursos_docker.csv", index=False)
    analizar(salida)


if __name__ == "__main__":
    app()

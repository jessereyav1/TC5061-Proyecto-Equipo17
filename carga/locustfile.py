"""Prueba de carga de la API de riesgo P2 (Actividad 11).

Escenario: operadores del equipo de rebalanceo y tableros que consultan el riesgo de las
estaciones. Cada usuario virtual manda peticiones con variables reales de MiBici (julio de
2025, ventana de evaluación del A/B testing), no con un mismo ejemplo repetido, para que
el modelo recorra caminos distintos en sus árboles.

Mezcla de peticiones de cada usuario:

- 90 % ``POST /predict`` con una entrada válida (la inferencia que se cuesta en FinOps).
- 5 % ``POST /predict`` con una entrada inválida: debe responder 422 sin llegar al modelo.
- 5 % ``GET /``: comprobación de salud, como la de un balanceador o del healthcheck.

La carga sube en escalones (``EscalonesDeCarga``) para encontrar el punto en que la API,
limitada a 1 vCPU y 1 GB en ``docker-compose.yml``, deja de responder a tiempo.

Se ejecuta con ``python -m tc5061_proyecto_equipo17.carga ejecutar``, que además mide
``docker stats``. Para usar la interfaz web de Locust::

    locust -f carga/locustfile.py --host http://localhost:8000
"""

import csv
import json
import os
from pathlib import Path
import random
import time

from locust import HttpUser, LoadTestShape, between, events, task

CARPETA = Path(__file__).resolve().parent
PAYLOADS = json.loads((CARPETA / "payloads_p2.json").read_text(encoding="utf-8"))

# Nombres con los que aparecen las peticiones en los reportes
PREDICT = "POST /predict"
PREDICT_INVALIDA = "POST /predict [invalida]"
SALUD = "GET /"

# Entradas que la validación de Pydantic de app.py debe rechazar con 422
INVALIDAS = [
    {**PAYLOADS[0], "hora_dia": 25},  # hora fuera de rango
    {**PAYLOADS[0], "dia_semana": 6, "fin_de_semana": 0},  # domingo marcado como entre semana
    {**PAYLOADS[0], "estacion": 120},  # campo que no existe en el esquema
]

# Escalones: (segundo en que termina, usuarios, usuarios nuevos por segundo)
ETAPAS = [
    (60, 10, 5),
    (120, 25, 5),
    (180, 50, 10),
    (240, 100, 10),
    (300, 150, 10),
    (360, 200, 10),
]

# Cada petición se guarda con su latencia para calcular percentiles exactos por escalón.
# La ruta se recibe del script que lanza la prueba; si no llega, no se guarda nada.
RUTA_PETICIONES = os.getenv("CARGA_PETICIONES_CSV")
_peticiones: list[tuple] = []
_entorno = None


class OperadorRebalanceo(HttpUser):
    """Usuario que consulta el riesgo de estaciones entre 0.5 y 1.5 segundos de pausa."""

    wait_time = between(0.5, 1.5)

    @task(18)
    def predecir(self):
        with self.client.post(
            "/predict", json=random.choice(PAYLOADS), name=PREDICT, catch_response=True
        ) as respuesta:
            if respuesta.status_code != 200:
                respuesta.failure(f"HTTP {respuesta.status_code}")
                return
            cuerpo = respuesta.json()
            if cuerpo.get("prediccion") not in (0, 1) or not 0 <= cuerpo.get("probabilidad", -1) <= 1:
                respuesta.failure(f"Respuesta fuera de contrato: {cuerpo}")

    @task(1)
    def entrada_invalida(self):
        with self.client.post(
            "/predict", json=random.choice(INVALIDAS), name=PREDICT_INVALIDA, catch_response=True
        ) as respuesta:
            # Aquí el éxito es que la API rechace la entrada
            if respuesta.status_code == 422:
                respuesta.success()
            else:
                respuesta.failure(f"Se esperaba 422 y llegó {respuesta.status_code}")

    @task(1)
    def salud(self):
        with self.client.get("/", name=SALUD, catch_response=True) as respuesta:
            if respuesta.status_code != 200 or not respuesta.json().get("modelo_cargado"):
                respuesta.failure("La API responde sin el modelo cargado")


class EscalonesDeCarga(LoadTestShape):
    """Sube de 10 a 200 usuarios en escalones de un minuto y termina a los 6 minutos."""

    def tick(self):
        segundo = self.get_run_time()
        for fin, usuarios, tasa in ETAPAS:
            if segundo < fin:
                return usuarios, tasa
        return None


@events.init.add_listener
def _guardar_entorno(environment, **_):
    global _entorno
    _entorno = environment


@events.request.add_listener
def _registrar_peticion(name, response_time, exception, **_):
    usuarios = _entorno.runner.user_count if _entorno and _entorno.runner else 0
    _peticiones.append((time.time(), name, round(response_time, 3), exception is None, usuarios))


@events.quitting.add_listener
def _escribir_peticiones(**_):
    if not RUTA_PETICIONES:
        return
    ruta = Path(RUTA_PETICIONES)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with ruta.open("w", newline="", encoding="utf-8") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(["timestamp", "nombre", "latencia_ms", "exito", "usuarios"])
        escritor.writerows(_peticiones)

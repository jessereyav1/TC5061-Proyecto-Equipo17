# Análisis FinOps — MiBici P2

**Equipo 17 — TC5061**

## 1. Objetivo

Estimar el costo de infraestructura por inferencia
del sistema MiBici P2, compuesto por una API FastAPI
y un servidor MLflow.

El análisis distingue entre costos proyectados y
mediciones reales de rendimiento.

## 2. Arquitectura

La solución contempla:

- API FastAPI para ejecutar las predicciones P2.
- Servidor MLflow para Model Registry.
- Base de datos SQLite y artefactos persistentes.
- Orquestación mediante Docker Compose.

La configuración propuesta establece un límite de
1 vCPU y 1 GB de memoria para la API.

## 3. Infraestructura de referencia

Proveedor: DigitalOcean.

Plan: Basic Droplet, CPU compartida.

Capacidad de referencia:
- 2 vCPU.
- 4 GiB RAM.
- 80 GiB SSD incluidos.

Tarifas de referencia:
- USD 0.03571 por hora.
- USD 24 por mes como límite de facturación base.

Fuente:
https://www.digitalocean.com/pricing/droplets

Fecha de consulta: 21 de noviembre de 2026.

La infraestructura es una hipótesis de despliegue,
no una instancia realmente contratada o validada.

## 4. Modelo de costos

Costo de cómputo mensual:

    C_compute = min(horas_activas * tarifa_hora,
                    límite_mensual)

Costo total mensual:

    C_total = C_compute + C_adicionales

Costo unitario:

    C_inferencia = C_total / N_inferencias_exitosas

Los costos adicionales incluyen almacenamiento o
transferencia que no estén cubiertos por el plan.

No se agrega por separado el SSD ya incluido
en el precio base de DigitalOcean.

## 5. Escenarios de demanda

Se consideran tres niveles:

- Baja: 25 % de la tasa de referencia.
- Normal: 50 % de la tasa de referencia.
- Alta: 100 % de la tasa de referencia.

Los escenarios representan proyecciones hipotéticas
con infraestructura de costo fijo.

La viabilidad de cada escenario debe verificarse
mediante pruebas de carga con Locust.

## 6. Integración de mediciones

Los datos necesarios de Locust son:

- Duración de la prueba.
- Total de solicitudes exitosas de POST /predict.
- Solicitudes fallidas.
- RPS de predicciones exitosas.
- Latencia promedio y percentil 95.
- Concurrencia alcanzada.

Los datos necesarios de docker stats son:

- CPU promedio de API.
- Memoria promedio de API.
- CPU promedio de MLflow.
- Memoria promedio de MLflow.

Las solicitudes a endpoints de salud no deben
contabilizarse como inferencias.

## 7. Optimizaciones propuestas

1. Revisar la utilización de CPU y memoria.
2. Evitar sobredimensionar la infraestructura.
3. Evaluar alternativas de menor costo cuando
   la demanda permanezca baja.
4. Mantener el modelo cargado en memoria para
   evitar cargarlo en cada solicitud.
5. Monitorear latencia p95 y errores antes de
   aumentar el tráfico.
6. Apagar entornos de desarrollo no utilizados,
   conservando los datos persistentes.

Las recomendaciones requieren validación operativa.

## 8. Reproducibilidad

Desde la raíz del repositorio:

    python -m pytest tests/test_finops_p2.py -v
    python -m scripts.finops_p2
    python -m scripts.finops_reporte_p2

Archivos de entrada:

- reports/finops/parametros.json
- reports/finops/referencia_nube.json

Archivos principales:

- scripts/finops_p2.py
- scripts/finops_reporte_p2.py
- tests/test_finops_p2.py

## 9. Estado y limitaciones

Se validaron 15 pruebas unitarias FinOps.

La última ejecución de la suite del proyecto
registró 131 pruebas aprobadas.

No se dispone todavía de mediciones finales
de Locust ni de consumo de recursos.

Por tanto, no se declara un costo por inferencia
real medido ni una capacidad máxima validada.

Los resultados económicos definitivos se
incorporarán cuando estén disponibles las
evidencias de las pruebas de carga.

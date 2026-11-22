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

El módulo FinOps cuenta con 15 pruebas unitarias aprobadas.

La última ejecución de la suite completa del equipo
registró 146 pruebas aprobadas y 12 advertencias.

Se integraron las mediciones experimentales de Locust
y Docker proporcionadas por el equipo.

Los resultados económicos utilizan tarifas públicas
de infraestructura y no representan gastos facturados.

El costo equivalente por inferencia se calculó con
las 5,312 predicciones exitosas registradas durante
la prueba de carga.

Para las proyecciones mensuales se utilizó la
capacidad observada de 8.42 RPS que cumplió el SLO.

La capacidad debe validarse mediante pruebas
prolongadas antes de considerarla garantizada
para una operación de producción.

Las proyecciones suponen demanda continua,
infraestructura fija y ausencia de costos adicionales.

## 10. Resultados experimentales de Locust

Las pruebas realizadas por el equipo sobre la API P2
registraron los siguientes resultados:

- Duración total: 359.1 segundos.
- Predicciones exitosas: 5,312.
- Predicciones fallidas: 0.
- Rendimiento promedio global: 14.79 RPS.
- Concurrencia máxima evaluada: 200 usuarios.
- Latencia p95 global: 11,534.9 ms.

El objetivo de servicio establecido fue una latencia
p95 inferior a 200 ms y una tasa de errores menor al 1 %.

Solo la etapa de 10 usuarios cumplió el objetivo,
con 8.42 predicciones exitosas por segundo y una
latencia p95 de aproximadamente 156 ms.

Por tanto, los escenarios económicos se construyeron
utilizando 8.42 RPS como capacidad observada que
cumple el SLO, no el rendimiento promedio global.

## 11. Consumo de recursos

Durante la prueba completa se registraron:

| Servicio | CPU promedio | RAM promedio |
|---|---:|---:|
| API FastAPI | 81.81 % | 246.8 MiB |
| MLflow | 5.02 % | 1,833.1 MiB |

La memoria combinada promedio fue de aproximadamente
2.03 GiB, sin incluir necesariamente la totalidad
del consumo del sistema operativo y otros procesos.

En la etapa de 10 usuarios, la API registró un
consumo de CPU cercano al 26.1 %.

Estos resultados sugieren que la CPU de la API
representa un factor importante bajo alta concurrencia.

## 12. Costo equivalente de la prueba

Se utilizó una tarifa de referencia de
USD 0.03571 por hora.

Con 359.1 segundos de duración y 5,312
predicciones exitosas:

- Costo equivalente de infraestructura: USD 0.00356207.
- Costo equivalente por inferencia: USD 0.0000006706.
- Costo equivalente por millón: USD 0.6706.

Este importe es un equivalente económico prorrateado,
no una factura ni un gasto real observado.

La prueba incluyó etapas que no cumplieron el SLO;
por tanto, este costo no representa una operación
continuamente conforme al objetivo de servicio.

## 13. Proyección mensual

Se utilizó un costo base de infraestructura
de USD 24 mensuales y 720 horas de operación.

| Escenario | RPS | Inferencias/mes | USD/millón |
|---|---:|---:|---:|
| Baja | 2.105 | 5,456,160 | 4.40 |
| Normal | 4.210 | 10,912,320 | 2.20 |
| Alta | 8.420 | 21,824,640 | 1.10 |

Los tres escenarios suponen tráfico continuo a
la tasa indicada durante todo el mes, sin
escalamiento de infraestructura.

No se consideran impuestos, excedentes de
transferencia ni servicios adicionales.

## 14. Interpretación FinOps

Los resultados evidencian que aumentar la utilización
de una infraestructura de costo fijo reduce el costo
unitario por inferencia.

Sin embargo, maximizar las solicitudes por segundo
no equivale a optimizar el servicio.

Durante las pruebas, la API aumentó su latencia
considerablemente a medida que creció la concurrencia,
aunque no se registraron errores de solicitudes.

Por ello, la evaluación FinOps debe considerar
simultáneamente costo, rendimiento y calidad
del servicio.

La etapa de 10 usuarios constituye el punto de
referencia observado que cumplió el SLO.

## 15. Recomendaciones finales

1. Mantener el SLO como restricción de capacidad.
2. Dimensionar la infraestructura con base en
   CPU, memoria y latencia p95.
3. Evitar usar 14.79 RPS como capacidad sostenible.
4. Monitorear periódicamente el costo por millón
   de inferencias exitosas.
5. Evaluar el consumo de memoria de MLflow y
   alternativas de despliegue.
6. Considerar mecanismos de escalamiento cuando
   crezca la demanda.
7. Repetir las pruebas durante periodos prolongados
   antes de considerar una capacidad permanente.

## 16. Evidencias generadas

- reports/finops/escenarios_finops.csv
- reports/finops/resultado_prueba.json
- reports/finops/costo_por_millon.png
- reports/finops/rendimiento_locust.png

Las gráficas económicas son proyecciones FinOps.
La gráfica de rendimiento proviene de Locust.

No se presentan como resultados de Evidently.

## 17. Conclusión

El análisis FinOps permitió estimar el costo por
inferencia del sistema MiBici P2 utilizando las
mediciones de carga del equipo y una referencia
económica documentada.

El principal hallazgo es que la optimización no
debe limitarse a reducir costos o aumentar RPS,
sino a mantener un equilibrio entre utilización
de infraestructura, latencia y confiabilidad.

Se recomienda utilizar la capacidad observada
de 8.42 RPS como referencia inicial para nuevas
evaluaciones, y validar su estabilidad antes de
adoptarla como capacidad de producción.

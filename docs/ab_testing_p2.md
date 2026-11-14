# Evaluación A/B y Continuous Training — MiBici P2

## Objetivo

Comparar el modelo vigente (Champion) contra el modelo
reentrenado (Challenger) y determinar si existe evidencia
suficiente para promover una nueva versión a producción.

## Metodología

- Evaluación offline con datos del 22 al 31 de julio de 2025.
- 87,713 observaciones de 367 estaciones.
- Mismas ocho variables de entrada y etiquetas reales.
- Umbral de clasificación: 0.5.
- Sin utilizar el periodo de evaluación para entrenamiento.
- Versiones recuperadas por alias desde MLflow Model Registry.

El modelo alternativo corresponde a la versión 3 en el
Registry de Diego y a la versión 2 en el Registry local
utilizado para reproducir esta evaluación.

## Resultados A/B

| Métrica | Champion | Challenger |
|---|---:|---:|
| Precision | 0.7138 | 0.7037 |
| Recall | 0.8976 | 0.9045 |
| F1-score | 0.7952 | 0.7916 |
| PR-AUC | 0.9063 | 0.9069 |
| Verdaderos positivos | 11707 | 11796 |
| Falsos positivos | 4695 | 4966 |
| Falsos negativos | 1335 | 1246 |
| Verdaderos negativos | 69976 | 69705 |

Challenger identifica 89 riesgos reales adicionales,
a cambio de 271 falsas alarmas adicionales.

## Resultados de Data Drift

El análisis de Evidently compara enero y julio de 2025.

- Magnitud: 0 de 9 variables superaron el umbral de 0.1.
- Significancia: 5 de 9 variables tuvieron p < 0.05.
- Los cambios detectados fueron de magnitud reducida.
- Variables prioritarias para seguimiento:
  pct_ocupacion, neto y neto_ult_3h.

La significancia estadística no demuestra por sí sola
un deterioro operativo del modelo.

## Recomendación de Continuous Training

Se recomienda conservar Champion de forma provisional.

Aunque Challenger mejora ligeramente el recall, también
incrementa las falsas alarmas y reduce el F1-score.
La promoción requiere definir el costo operativo de
falsos positivos y falsos negativos.

Se propone:

1. Monitorear mensualmente las distribuciones y métricas.
2. Analizar el desempeño por estación y franja horaria.
3. Detectar deterioro sostenido o cambios relevantes.
4. Reentrenar un candidato cuando exista evidencia.
5. Evaluar Champion y Challenger en la misma ventana.
6. Promover una versión únicamente tras validación.
7. Conservar trazabilidad y posibilidad de rollback.

El análisis presentado es una evaluación offline
histórica, no un experimento A/B aleatorizado en producción.

## Reproducibilidad

Desde la raíz del repositorio:

    python -m scripts.ab_testing_p2
    python -m scripts.graficas_ab_p2
    python -m pytest tests/test_ab_testing_p2.py -v

Las evidencias se encuentran en reports/ab_testing/.

La comparación requiere los datos procesados de monitoreo
y ambas versiones disponibles en un MLflow Model Registry.

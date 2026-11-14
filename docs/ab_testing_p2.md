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

## Análisis de incertidumbre estadística

Se aplicó bootstrap pareado por día, con 5,000
remuestreos y semilla 42, utilizando los diez días
del conjunto de evaluación.

| Métrica | Delta Challenger - Champion | IC 95% |
|---|---:|---|
| Precision | -0.01002 | [-0.01352, -0.00720] |
| Recall | +0.00682 | [+0.00565, +0.00825] |
| F1-score | -0.00363 | [-0.00578, -0.00182] |

Los intervalos no incluyen cero. Sin embargo, la
ventana comprende solo diez días, por lo que la
generalización a otros meses requiere cautela.

La prueba exacta de McNemar, como análisis
complementario, produjo:

- Observaciones donde solo Champion acertó: 436.
- Observaciones donde solo Challenger acertó: 254.
- p-valor: menor a 0.001.

McNemar no considera explícitamente la dependencia
entre estación-horas, por lo que no debe interpretarse
como una prueba definitiva de superioridad operacional.

Los resultados completos están en:

- reports/ab_testing/bootstrap_intervalos.csv
- reports/ab_testing/mcnemar_resultado.csv

## Criterio de decisión y costo operativo

Challenger mejora el recall y detecta 89 riesgos
reales adicionales, pero genera 271 falsos positivos
adicionales.

De las 360 alertas adicionales, aproximadamente
24.7 % corresponden a riesgos reales y 75.3 %
a falsas alarmas.

Si el objetivo fuera exclusivamente maximizar
recall, Challenger sería preferible.

Sin embargo, sin una cuantificación del costo
de falsos positivos y falsos negativos, no
se justifica su promoción automática.

Se recomienda conservar Champion provisionalmente,
definir una función de costo operativo y evaluar
nuevamente antes de cualquier promoción.

## Aclaración sobre el drift de neto

El análisis de Evidently reportó para neto:

- Magnitud del cambio: 0.0132.
- p-valor: 0.102.
- Drift significativo: no.

Por tanto, neto se propone como variable de
monitoreo preventivo, no como variable con
drift estadísticamente significativo.

## Línea base histórica de Champion

Resultados históricos reportados para febrero
de 2025:

| Métrica | Febrero | Julio |
|---|---:|---:|
| Precision | 0.728 | 0.7138 |
| Recall | 0.892 | 0.8976 |
| F1-score | 0.802 | 0.7952 |
| PR-AUC | 0.905 | 0.9063 |

Las métricas muestran variaciones moderadas,
sin evidencia clara de degradación severa
del desempeño entre ambos periodos.

## Trazabilidad en MLflow

Modelo registrado: mibici-equipo17-p2-riesgo.

### Champion

- Alias: champion.
- Versión local: 1.
- Run ID: 1197d147660245ba87f4b56f14a51eac.
- Estado: READY.

### Challenger

- Alias: challenger.
- Versión local: 2.
- Run ID: 10a8083a2c7f4c6294a56b6d074cdca4.
- Estado: READY.

En el Registry de Diego, Challenger corresponde
a la versión 3. Los números pueden variar
entre registros independientes; el alias y
los identificadores de ejecución permiten
documentar la procedencia de cada modelo.

## Conclusión de la revisión

El A/B Testing demuestra diferencias medibles entre
los modelos. Challenger aumenta la detección de riesgo,
mientras Champion mantiene mejores valores de precision,
F1-score y menor cantidad de falsas alarmas.

Se recomienda mantener Champion hasta contar con
criterios de costo operativo, monitoreo continuo
y evaluación adicional antes de promover Challenger.

# Versión alternativa del modelo P2 (Actividad 10)

Este documento describe la segunda versión del modelo P2 (riesgo de que una estación se
quede sin bicicletas o sin anclajes la hora siguiente), que se compara contra la versión
en producción en el A/B testing.

## Qué cambia respecto a la versión en producción

| | Versión en producción (`@champion`, v1) | Versión alternativa (`@challenger`, v3) |
|---|---|---|
| Datos de entrenamiento | Enero de 2025 | Enero de 2025 + 1 a 21 de julio de 2025 |
| Filas de entrenamiento | 273,424 | 458,025 (273,424 + 184,601) |
| Modelo | Random Forest, `class_weight="balanced"` | El mismo |
| Hiperparámetros | `n_estimators=100`, `max_depth=12` | Los mismos (se leen del registro) |
| Variables | `P2_FEATURES` (8) | Las mismas, en el mismo orden |
| Umbral de decisión | 0.5 | 0.5 |

**Solo cambian los datos.** Se dejan fijos los hiperparámetros para que el A/B testing
mida el efecto de reentrenar con datos recientes, es decir, la pregunta de Continuous
Training: *¿vale la pena reentrenar cuando los datos cambian?* Si también cambiáramos los
hiperparámetros, no sabríamos a cuál de los dos cambios atribuir la diferencia.

Los periodos vienen de `docs/periodos_monitoreo.md`. La ventana de evaluación (22 a 31 de
julio, 87,713 filas) **no se usa para entrenar ni para elegir nada**; es el conjunto
común del A/B testing. La última hora antes del corte (21 de julio a las 23:00) se
excluye, porque su etiqueta se calcula con la primera hora de evaluación.

La v2 del registro es un candidato de la entrega anterior (`n_estimators=200`,
`max_depth=8`) y conserva su linaje; el alias `challenger` pasó de la v2 a la v3.

## Cómo se genera

```bash
python -m tc5061_proyecto_equipo17.monitoreo_datos ingesta-actual
python -m tc5061_proyecto_equipo17.monitoreo_datos construir-features
python -m tc5061_proyecto_equipo17.modeling.train_p2_v2
```

El script `tc5061_proyecto_equipo17/modeling/train_p2_v2.py`:

1. Lee los hiperparámetros de `mibici-equipo17-p2-riesgo@champion`.
2. Lee `p2_referencia.parquet` y `p2_actual.parquet` y comprueba su SHA-256 contra
   `manifest_monitoreo.json`, para que todo el equipo use los mismos datos.
3. Entrena con enero + la ventana de reentrenamiento (`conjuntos_v2`).
4. Registra en MLflow la corrida `p2-v2-reentrenado-ene-jul`: parámetros, conjuntos con
   `log_input`, manifiesto, matriz de confusión, curva precision-recall y modelo con firma.
5. Registra el modelo como nueva versión de `mibici-equipo17-p2-riesgo`, con etiquetas de
   linaje (`run_id`, `git_commit`, `tipo_version`, `version_base`, periodo de datos) y el
   alias `challenger`. El alias `champion` no se modifica.
6. Carga el modelo por alias y verifica que recibe las mismas columnas que la API.

Opciones: `--n-estimators` y `--max-depth` para probar otros hiperparámetros, y
`--no-registrar` para entrenar sin crear una versión.

Como la API carga `models:/mibici-equipo17-p2-riesgo@champion`, si el A/B testing
favorece a la v3 basta mover el alias `champion` a la v3: no hay que cambiar código ni
reconstruir la imagen.

## Métricas en la ventana de evaluación (22 a 31 de julio)

Referencia rápida para el A/B testing (umbral 0.5, 87,713 filas, tasa de riesgo real del
periodo actual 14.4 %):

| Versión | Recall | Precision | F1 | PR-AUC |
|---|---:|---:|---:|---:|
| v1 `@champion` (solo enero) | 0.8976 | 0.7138 | 0.7952 | 0.9063 |
| v3 `@challenger` (enero + julio) | 0.9045 | 0.7037 | 0.7916 | 0.9069 |

Las diferencias son pequeñas: la v3 detecta un poco más de estaciones en riesgo (más
recall), que es la métrica prioritaria del proyecto, a cambio de algo más de falsas
alarmas (menos precisión). La comparación formal, con la decisión sobre qué versión queda
vigente, se documenta en el A/B testing.

## Pruebas

`tests/test_train_p2_v2.py` comprueba que el entrenamiento incluye enero y la ventana de
reentrenamiento, que la evaluación es solo la ventana del A/B testing, que no hay fuga de
información entre ambas (tampoco la hora excluida) y que los conjuntos tienen las
variables del modelo.

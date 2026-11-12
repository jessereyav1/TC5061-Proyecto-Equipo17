# Periodos de referencia y actual para el monitoreo (Actividad 10)

Este documento define los conjuntos de datos con los que se mide el *drift* del modelo
P2 (riesgo de que una estación se quede sin bicicletas o sin anclajes la hora siguiente)
y con los que se compararán dos versiones del modelo en el A/B testing.

## Periodos elegidos

| Conjunto | Periodo | Uso |
|---|---|---|
| **Referencia** | enero de 2025 (1 a 31) | Datos con los que se entrena el modelo en producción. |
| **Actual** | julio de 2025 (1 a 31) | Periodo posterior que representa el comportamiento en producción. |
| Actual: ventana de reentrenamiento | 1 a 21 de julio | Datos con los que se puede entrenar la versión alternativa del modelo. |
| Actual: ventana de evaluación | 22 a 31 de julio | Conjunto común para comparar las dos versiones (A/B testing). |

Las fechas viven en `tc5061_proyecto_equipo17/config.py` (`REF_INICIO`, `REF_FIN`,
`ACTUAL_INICIO`, `ACTUAL_FIN`, `ACTUAL_CORTE_EVAL`).

## Por qué estos periodos

- **Referencia = enero.** El modelo P2 se entrena con enero de 2025 (la validación es la
  última semana de enero), así que enero es la distribución que el modelo "conoce".
- **Actual = julio.** Está cinco meses después del entrenamiento y pertenece a otra
  temporada (el verano coincide con la temporada de lluvias en Guadalajara), por lo que es
  razonable esperar cambios en el uso del sistema.
- **No se usa febrero como periodo actual.** Febrero ya es el conjunto de prueba con el que
  se evaluó el modelo y es el mes contiguo a enero, así que mostraría poco cambio.
- **No se usan los registros de la API.** La API de la Actividad 8 no guarda las peticiones
  que recibe, por lo que se usan los datos abiertos de MiBici.
- **Ventanas dentro de julio.** La versión alternativa se reentrena con la ventana de
  reentrenamiento y las dos versiones se evalúan sobre la ventana de evaluación, que
  ninguna vio al entrenar. La última hora de la ventana de reentrenamiento (21 de julio a
  las 23:00) se excluye, porque su etiqueta se calcula con la primera hora de evaluación.

## Qué variables se comparan

Las ocho variables que recibe el modelo (`P2_FEATURES`: `hora_dia`, `dia_semana`,
`fin_de_semana`, `capacidad`, `pct_ocupacion`, `riesgo_ahora`, `neto`, `neto_ult_3h`),
más la etiqueta `riesgo_siguiente_hora`. Se construyen con las mismas funciones del
entrenamiento (`p2_construir_dataset`), una fila por estación y hora.

## Cómo se generan

Desde la raíz del repositorio, con el entorno virtual activo y habiendo ejecutado antes la
ingesta general (que deja el Parquet de enero en `data/processed/`):

```bash
python -m tc5061_proyecto_equipo17.monitoreo_datos ingesta-actual
python -m tc5061_proyecto_equipo17.monitoreo_datos construir-features
```

1. `ingesta-actual` descarga `datos_abiertos_2025_07.csv` desde el sitio oficial de MiBici
   y lo procesa con el mismo código de limpieza que enero y febrero. El resultado queda en
   `data/processed/actual/`, no en `data/processed/`, para que el entrenamiento existente
   (`train_p2`, que lee `data/processed/*_procesado.parquet`) no cambie.
2. `construir-features` genera en `data/processed/monitoreo/`:
   - `p2_referencia.parquet`: variables y etiqueta de enero.
   - `p2_actual.parquet`: variables y etiqueta de julio, con la columna `ventana`
     (`reentrenamiento`, `evaluacion` o vacía en la hora excluida).
   - `manifest_monitoreo.json`: periodos, filas, estaciones, tasa de riesgo, valores nulos
     y SHA-256 de cada archivo.

Los datos no se versionan en Git (`/data/` está en `.gitignore`); el manifiesto permite
comprobar que todos trabajan con los mismos archivos.

## Pruebas

`tests/test_monitoreo_datos.py` comprueba que los periodos no se traslapan con el
entrenamiento, que la URL es la oficial, que las ventanas no se cruzan, que el periodo
actual no se guarda en `data/processed/` y que las tablas contienen las variables del
modelo sin valores nulos.

## Resultados de la generación

| Conjunto | Filas | Estaciones | Tasa de riesgo | Valores nulos |
|---|---:|---:|---:|---:|
| Referencia (enero 2025) | 273,424 | 368 | 15.5 % | 0 |
| Actual (julio 2025) | 272,681 | 367 | 14.4 % | 0 |

Ventanas del periodo actual: 184,601 filas de reentrenamiento (1 al 21 de julio), 87,713
de evaluación (22 al 31 de julio) y 367 horas excluidas (una por estación; aparecen como
`None` en el manifiesto). Los viajes de julio procesados son 367,933.

Julio tiene una estación menos que enero con filas en la tabla. Las cifras salen de
`data/processed/monitoreo/manifest_monitoreo.json`.

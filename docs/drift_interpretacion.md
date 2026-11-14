# Interpretación del drift de datos — Actividad 10

## 1. Objetivo y metodología

- Modelo monitoreado: **P2 (clasificación de riesgo)**, servido por la API.
  Su periodo de entrenamiento es enero 2025 (`train_p2.py`: entrena con todo enero, valida 25-31 de enero), por lo que la referencia (enero) coincide con la distribución de entrenamiento.

Se comparó el **conjunto de referencia** (enero 2025, el periodo con el que se
entrena el modelo P2) contra el **conjunto actual** (julio 2025, un periodo
posterior y de otra temporada) usando **Evidently 0.7.21**.

- Variables: las 8 de `P2_FEATURES` más la etiqueta `riesgo_siguiente_hora`.
- **Corrida de magnitud** (`--modo magnitud`): **Wasserstein distance (normed)** para
  numéricas y **Jensen-Shannon distance** para categóricas (defaults de Evidently
  para conjuntos de más de 1000 filas), con **umbral 0.1** (drift si `score ≥ 0.1`).
- **Corrida de significancia** (`--modo significancia`): **K-S** para numéricas y
  **chi-cuadrada** para categóricas, con **umbral de p-valor 0.05** (drift si `p < 0.05`).
- Reportes: `reports/drift/reporte_drift.html` y
  `reports/drift/reporte_drift_significancia.html`.

## 2. Resultado global

| Indicador | Valor |
|---|---:|
| Variables comparadas | 9 |
| Variables con drift significativo (magnitud, umbral 0.1) | **0** |
| Proporción de columnas con drift (magnitud) | **0.0 %** |
| Drift a nivel de conjunto (magnitud) | **No** |

**No se detectó drift de datos de magnitud relevante entre enero y julio de 2025.**

## 3. Variables con mayor alteración estadística

Ordenadas por *drift score* (todas por debajo del umbral 0.1):

| # | Variable | Tipo | Método | Drift score | Umbral | ¿Drift? |
|---|---|---|---:|---:|---:|---|
| 1 | `dia_semana` | categórica | Jensen-Shannon | 0.0416 | 0.1 | No |
| 2 | `pct_ocupacion` | numérica | Wasserstein | 0.0331 | 0.1 | No |
| 3 | `neto_ult_3h` | numérica | Wasserstein | 0.0198 | 0.1 | No |
| 4 | `neto` | numérica | Wasserstein | 0.0132 | 0.1 | No |
| 5 | `riesgo_siguiente_hora` | categórica (objetivo) | Jensen-Shannon | 0.0109 | 0.1 | No |
| 6 | `riesgo_ahora` | categórica | Jensen-Shannon | 0.0109 | 0.1 | No |
| 7 | `capacidad` | numérica | Wasserstein | 0.0019 | 0.1 | No |
| 8 | `hora_dia` | categórica | Jensen-Shannon | 0.0000 | 0.1 | No |
| 9 | `fin_de_semana` | categórica | Jensen-Shannon | 0.0000 | 0.1 | No |

## 3.1. Significancia estadística (corrida sensible complementaria)

La corrida de magnitud mide *cuánto* cambian las distribuciones. Para saber si las
diferencias son *estadísticamente detectables* se repitió el análisis con **K-S**
(numéricas) y **chi-cuadrada** (categóricas), con **umbral de p-valor 0.05**.

| Indicador | Magnitud (score ≥ 0.1) | Significancia (p < 0.05) |
|---|---:|---:|
| Variables con drift | 0 / 9 | **5 / 9** |
| Columnas con drift (features) | 0 / 8 | **4 / 8 (50.0 %)** |
| Drift a nivel de conjunto | No | **Sí** |

Variables con drift significativo (ordenadas por p-valor):

| Variable | Método | p-valor | ¿Drift? |
|---|---|---:|---|
| `dia_semana` | chi-cuadrada | 0.0 | Sí |
| `riesgo_siguiente_hora` (objetivo) | chi-cuadrada | 9.54e-57 | Sí |
| `riesgo_ahora` | chi-cuadrada | 1.46e-56 | Sí |
| `pct_ocupacion` | K-S | 9.64e-25 | Sí |
| `neto_ult_3h` | K-S | 2.88e-06 | Sí |
| `neto` | K-S | 0.102 | No |
| `capacidad` | K-S | 0.994 | No |
| `hora_dia` | chi-cuadrada | 1.0 | No |
| `fin_de_semana` | chi-cuadrada | 1.0 | No |

**Lectura.** Con ~273 000 registros por conjunto, las pruebas de hipótesis detectan
como significativas diferencias que en magnitud son despreciables: las cuatro
features que "cambian" (`dia_semana`, `riesgo_ahora`, `pct_ocupacion`,
`neto_ult_3h`) tienen drift de magnitud ≤ 0.042, muy por debajo de 0.1. No hay
contradicción: **la magnitud es mínima pero medible**. La etiqueta también sale
significativa (p = 9.5e-57), pero su prevalencia apenas se movió (~15.5 % → ~14.4 %).
`neto` es el caso frontera (p = 0.102), y `capacidad`, `hora_dia` y `fin_de_semana`
son claramente estables.

**Conclusión conjunta.** El data drift es **leve**: sin cambios de magnitud relevante
(ninguna variable supera 0.1) pero con diferencias estadísticamente detectables,
atribuibles a la estacionalidad (temporada de lluvias) y a la composición del
calendario. Evidencia: `reports/drift/reporte_drift_significancia.html` y
`reports/drift/drift_significancia.csv`.

## 4. Lectura por variable

- **`dia_semana` (0.0416, el mayor).** La diferencia se explica casi por completo
  por la **composición del calendario**, no por un cambio de comportamiento:
  enero de 2025 tiene cinco miércoles, jueves y viernes y cuatro martes; julio de
  2025 tiene cinco martes, miércoles y jueves y cuatro viernes. Al comparar
  meses completos, la proporción de estación-horas por día de la semana cambia
  por construcción.
- **`hora_dia` y `fin_de_semana` (0.0000).** Son variables deterministas del
  calendario: cada hora aparece el mismo número de veces en un mes completo y
  ambos meses tienen 8 de 31 días de fin de semana. Su drift exactamente cero es
  una propiedad de la variable, no evidencia de estabilidad operativa.
- **`pct_ocupacion` (0.0331).** Es la variable **continua** más alterada; la
  ocupación estimada responde a la temporada (julio coincide con la temporada de
  lluvias en Guadalajara). Aun así, está muy por debajo del umbral.
- **`neto` (0.0132) y `neto_ult_3h` (0.0198).** El flujo neto cambia algo, de
  forma coherente con la variación estacional de viajes, pero sin alcanzar
  magnitud de drift.
- **`riesgo_ahora` y `riesgo_siguiente_hora` (0.0109 ambos).** El régimen de
  riesgo (proporción de estación-horas en riesgo) es estable: la tasa base pasó
  de ~15.5 % (enero) a ~14.4 % (julio). **No hay cambio de prior de la etiqueta**,
  por lo que no se requiere recalibrar el umbral por cambio de prevalencia.
- **`capacidad` (0.0019).** La capacidad de las estaciones prácticamente no
  cambió entre periodos.

## 5. Implicaciones para el desempeño del modelo P2

- **Los insumos del modelo siguen en distribución.** Como ninguna variable
  presenta drift de magnitud relevante, no hay razón para esperar una degradación
  del modelo P2 atribuible a *data drift*. El desempeño en julio debería ser
  comparable al observado en validación/prueba (enero–febrero).
- **La etiqueta no se desplazó.** Que `riesgo_siguiente_hora` tenga un drift de
  magnitud 0.0109 implica que la prevalencia del riesgo se mantiene; el modelo no
  "sobre-predice" ni "sub-predice" riesgo por un cambio de tasa base.
- **Las diferencias estadísticamente significativas (sección 3.1) son de magnitud
  mínima.** Que 5 de 9 variables salgan significativas con 273k filas refleja la
  sensibilidad de las pruebas de hipótesis, no un cambio operativo: los cuatro
  features "significativos" tienen drift de magnitud ≤ 0.042. No ameritan
  reentrenamiento por sí solos.
- **Si en el A/B testing se observa una caída de métricas, no será por data
  drift, sino por *concept drift*.** Es decir, la relación entre las variables y
  la etiqueta podría haber cambiado (por ejemplo, patrones de uso distintos en
  temporada de lluvias) aunque las distribuciones marginales sean estables. Esto
  justifica precisamente el A/B testing planeado: comparar el modelo `@champion`
  contra la versión reentrenada sobre la ventana de evaluación (22–31 de julio).
- **Variables a vigilar en el monitoreo continuo.** `pct_ocupacion`, `neto` y
  `neto_ult_3h` son las que más se movieron y las más sensibles a la temporada;
  conviene monitorearlas mes a mes. `dia_semana` no debe interpretarse como
  señal de negocio por su componente de calendario.

## 6. Limitaciones

- **Métodos de magnitud vs. significancia.** Wasserstein y Jensen-Shannon miden
  *cuánto* cambian las distribuciones; K-S y chi-cuadrada (sección 3.1) miden si
  el cambio es *estadísticamente detectable*. Con ~273 000 filas las pruebas de
  hipótesis detectan como significativas diferencias ínfimas (5 de 9 variables,
  p < 0.05) que no alcanzan magnitud de drift (0 de 9, score < 0.1). Por eso
  ambos resultados se reportan juntos y no se sustituyen.
- **Drift agregado oculta drift local.** Un análisis por estación o por hora del
  día podría revelar cambios concentrados que el promedio global no muestra.
- **`pct_ocupacion` es un proxy.** Es una ocupación estimada a partir del flujo,
  no un inventario real; su medición de drift hereda esa limitación.

## 7. Recomendaciones

1. ~~Complementar con una corrida sensible~~ **Hecho** (sección 3.1):
   `python -m tc5061_proyecto_equipo17.monitoreo_drift reporte --modo significancia`
   con `num_method="ks"`, `cat_method="chisquare"`, umbral de p-valor 0.05.
2. Analizar el drift **segmentado por estación y por hora** para descartar
   cambios localizados.
3. Mantener el A/B testing sobre la ventana de evaluación de julio para detectar
   *concept drift* (lo que el drift de datos no cubre).
4. Monitorear mensualmente `pct_ocupacion`, `neto` y `neto_ult_3h`.

## 8. Reproducibilidad

```bash
python -m tc5061_proyecto_equipo17.monitoreo_datos ingesta-actual
python -m tc5061_proyecto_equipo17.monitoreo_datos construir-features
python -m tc5061_proyecto_equipo17.monitoreo_drift reporte --modo ambos
```

Artefactos de magnitud: `reports/drift/reporte_drift.html`,
`reports/drift/drift_por_variable.csv`, `reports/figures/drift_por_variable.png`.

Artefactos de significancia: `reports/drift/reporte_drift_significancia.html`,
`reports/drift/drift_significancia.csv`, `reports/figures/drift_significancia.png`.
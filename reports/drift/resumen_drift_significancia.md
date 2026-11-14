# Resumen de significancia del drift (K-S / chi-cuadrada, p < 0.05)

Con ~273k filas, un p-valor < 0.05 indica una diferencia detectable estadisticamente, no necesariamente una diferencia de magnitud relevante. Comparar con reports/drift/drift_por_variable.csv (modo magnitud).

- Variables comparadas: 9
- Variables con drift: 5
- Proporcion de columnas con drift: 50.0%
- Drift a nivel de conjunto: si

## Variables con drift (ordenadas por magnitud del score)

| Variable | Metodo | p-valor | Umbral |
|---|---|---:|---:|
| neto_ult_3h | ks | 0.0000 | 0.05 |
| pct_ocupacion | ks | 0.0000 | 0.05 |
| riesgo_ahora | chisquare | 0.0000 | 0.05 |
| riesgo_siguiente_hora | chisquare | 0.0000 | 0.05 |
| dia_semana | chisquare | 0.0000 | 0.05 |

## Variables sin drift

| Variable | p-valor |
|---|---:|
| hora_dia | 1.0000 |
| fin_de_semana | 1.0000 |
| capacidad | 0.9942 |
| neto | 0.1024 |

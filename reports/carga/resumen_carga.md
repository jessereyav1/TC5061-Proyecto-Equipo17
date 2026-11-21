# Resumen de la prueba de carga (API P2, 1 vCPU / 1 GB)

- Duración: 359.1 s, hasta 200 usuarios.
- Predicciones exitosas: 5,312; fallidas: 0.
- Rendimiento promedio: 14.79 predicciones/s.
- Latencia de /predict: p50 4273.5 ms, p95 11534.9 ms.
- API: CPU promedio 81.81 %, memoria 246.8 MiB.
- MLflow: CPU promedio 5.02 %, memoria 1833.1 MiB.

Objetivo de servicio: p95 <= 200 ms y errores <= 1%.
Capacidad sostenible: 8.42 predicciones/s con 10 usuarios (p95 156.0 ms, CPU de la API 26.1 %).

## Por escalón

| Usuarios | Pred./s | p50 ms | p95 ms | p99 ms | Errores | CPU API % | Mem API MiB | Cumple |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 10 | 8.4 | 69 | 156 | 227 | 0.00% | 26 | 240 | sí |
| 25 | 17.1 | 218 | 1038 | 1297 | 0.00% | 76 | 242 | no |
| 50 | 16.2 | 1995 | 3049 | 3309 | 0.00% | 97 | 247 | no |
| 100 | 15.8 | 4981 | 5828 | 6104 | 0.00% | 97 | 249 | no |
| 150 | 15.4 | 8230 | 9126 | 9490 | 0.00% | 97 | 252 | no |
| 200 | 16.1 | 11289 | 12111 | 12325 | 0.00% | 97 | 253 | no |

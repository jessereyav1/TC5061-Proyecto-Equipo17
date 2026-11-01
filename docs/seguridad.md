# Auditoría de seguridad — Equipo 17

## 1. Objetivo

Integrar controles básicos de seguridad al repositorio del proyecto
MLOps MiBici Guadalajara, con énfasis en análisis estático, auditoría
de dependencias, detección de secretos y pruebas automatizadas.

Responsable: Rol 1 — Seguridad e infraestructura.

## 2. Entorno y herramientas

- Python 3.10.22
- Bandit 1.9.4
- pip-audit 2.10.1
- detect-secrets 1.5.0
- pytest

Instalación:

    python -m pip install -r requirements-security.txt

## 3. Comandos reproducibles

Ejecutar pruebas:

    make test

Analizar el código:

    make security-bandit

Auditar dependencias instaladas:

    make security-audit

Detectar posibles secretos:

    make security-secrets

Ejecutar auditorías:

    make security

Nota: make security puede terminar con error porque pip-audit
identifica vulnerabilidades conocidas pendientes. No deben
silenciarse sus resultados para simular una aprobación.

## 4. Análisis estático con Bandit

Se analizaron 2,522 líneas de código Python.

Resultados del análisis completo:

- Severidad alta: 0
- Severidad media: 0
- Severidad baja: 12

Los hallazgos bajos corresponden principalmente al uso de
subprocess en los módulos de entrenamiento y a una excepción
silenciada.

Estos hallazgos requieren revisión individual. No se consideran
automáticamente vulnerabilidades explotables.

### Corrección B310

Se identificó inicialmente una advertencia de severidad media
relacionada con urllib.urlopen en dataset.py.

Se implementaron controles para:

- Validar el protocolo HTTPS.
- Restringir las URL al dominio oficial mibici.net.
- Rechazar credenciales y puertos no autorizados.
- Validar redirecciones antes de seguirlas.

Después del cambio, Bandit no reportó hallazgos medios ni altos.

## 5. Auditoría de dependencias

Auditoría inicial:

- 47 vulnerabilidades conocidas.
- 5 paquetes afectados.

Se actualizaron:

- mlflow: 3.17.0 a 3.17.2
- mlflow-skinny: 3.17.0 a 3.17.2
- mlflow-tracing: 3.17.0 a 3.17.2
- GitPython: 3.2.0 a 3.2.1
- certifi: 2026.7.22 a 2026.10.23

Auditoría posterior:

- 43 vulnerabilidades conocidas.
- 2 paquetes afectados.

Paquetes pendientes:

| Paquete | Versión | Hallazgos | Situación |
|---|---|---:|---|
| filelock | 4.1.0 | 6 | Corrección 4.2.1 requiere Python 3.11 o superior |
| pillow | 12.3.0 | 37 | Corrección 13.0.0 no disponible en el entorno Python 3.10 |

Las vulnerabilidades pendientes deben evaluarse según su
aplicabilidad al proyecto. Mantener Python 3.10 permite conservar
el entorno actual, pero no elimina los riesgos reportados.

Se recomienda evaluar una migración controlada a una versión
compatible de Python y volver a ejecutar todas las pruebas y
los experimentos de MLflow.

## 6. Detección de secretos

Se utilizó detect-secrets para revisar posibles credenciales
dentro del repositorio.

Escaneo inicial:

- 6 coincidencias en 3 archivos.
- Una coincidencia en caché de pytest.
- Cinco coincidencias Base64 en dos notebooks de Jupyter.

Las cinco coincidencias de los notebooks se relacionaron con
campos image/png de salidas gráficas.

Se configuraron exclusiones temporales para:

- notebooks/1.0-exploracion-inicial.ipynb
- notebooks/1.0-mdlc-perfiles-horarios.ipynb

El análisis automatizado posterior terminó con cero hallazgos
en los archivos incluidos.

Limitación: la exclusión de notebooks completos también excluye
su código y texto. Se recomienda reemplazarla por excepciones
más específicas y revisar las exclusiones cuando cambien.

## 7. Revisión del historial Git

Se revisaron nombres de archivos sensibles en el historial
disponible y algunos patrones conocidos de credenciales.

Resultado:

- Cero coincidencias con los patrones evaluados.

Esta revisión es básica y no garantiza la ausencia de secretos
de otros formatos dentro de commits anteriores.

No se reescribió el historial Git.

## 8. Protección de archivos sensibles

El archivo .gitignore excluye, entre otros:

- Archivos .env y variantes.
- Datos locales y archivos generados.
- Modelos serializados.
- Artefactos y base de datos de MLflow.
- Entornos virtuales.
- Reportes locales de seguridad.

Los reportes ubicados en reports/security/ se mantienen fuera
del control de versiones.

## 9. Pruebas automatizadas

Se ejecutó:

    make test

Resultado: 17 pruebas aprobadas.

Distribución al momento de la auditoría:

- 5 pruebas existentes de datos.
- 12 pruebas nuevas de validación de URL y redirecciones.

Las pruebas de seguridad cubren URLs oficiales, protocolos
inseguros, dominios no autorizados, credenciales en URLs,
puertos no permitidos y redirecciones inválidas.

Pendiente: ampliar las pruebas de integración para verificar
el comportamiento completo de la función descargar_archivo
con respuestas HTTP simuladas, sin utilizar conexiones reales.

## 10. Riesgos pendientes y acciones recomendadas

1. Resolver o mitigar los hallazgos de filelock y pillow.
2. Evaluar una migración controlada de Python.
3. Revisar los 12 hallazgos bajos de Bandit.
4. Sustituir las exclusiones completas de notebooks por
   excepciones específicas.
5. Ampliar la revisión de secretos del historial Git.
6. Validar P1, P2 y P3 con las versiones actualizadas.
7. Incorporar las auditorías a integración continua.

## 11. Conclusión

La actividad incorporó controles reproducibles para revisar
el código, las dependencias y las posibles credenciales del
repositorio.

Se redujeron los hallazgos conocidos de pip-audit de 47 a 43
y se eliminó el hallazgo de severidad media inicialmente
reportado por Bandit.

Las pruebas disponibles continúan aprobando. Sin embargo,
persisten vulnerabilidades de dependencias y limitaciones de
cobertura que requieren seguimiento antes de considerar
completamente satisfecha la revisión de seguridad.

## 12. Validación de compatibilidad de modelos

Después de actualizar MLflow a la versión 3.17.2,
se ejecutaron nuevamente los tres pipelines del proyecto.

### P1 - Regresión

Modelo: Random Forest, 20 estimadores, profundidad 12.

- RMSE de prueba: 2.2563
- R2 de prueba: 0.5653
- Estado: ejecución exitosa y registro en MLflow.

### P2 - Clasificación

Modelo: Random Forest.

- F1 de prueba: 0.802
- Recall de prueba: 0.892
- Estado: ejecución exitosa y registro en MLflow.

### P3 - Agrupamiento

Modelo: K-Means, k=4, semilla 42.

- Estado: ejecución exitosa y registro en MLflow.

Las tres ejecuciones terminaron correctamente.

Se observaron advertencias no bloqueantes relacionadas
con inferencia de esquemas y fuentes de datos de MLflow.

Esta validación confirma la ejecución funcional de los
tres pipelines en el entorno actualizado. No constituye
una prueba de ausencia de vulnerabilidades.

# TC5061 - Proyecto Equipo 17
## Sistema de Machine Learning para MiBici Guadalajara

Proyecto académico desarrollado para analizar datos abiertos del sistema de bicicletas compartidas MiBici, en el Área Metropolitana de Guadalajara, mediante técnicas de aprendizaje automático y prácticas de MLOps.

El proyecto aborda tres problemas:

- **P1 - Regresión:** Predicción de demanda.
- **P2 - Clasificación:** Estimación de riesgos operativos.
- **P3 - Clustering:** Agrupamiento de estaciones según patrones de uso.

## 1. Fuente de datos

Los datos provienen del portal oficial de MiBici:

https://www.mibici.net/es/datos-abiertos/

La ingesta general utiliza registros de viajes de enero y febrero de 2025, además de información de nomenclatura y capacidad de estaciones. Para P1 se emplean 24 archivos mensuales, desde octubre de 2024 hasta septiembre de 2026.

Los archivos originales se descargan automáticamente y se procesan localmente.

## 2. Requisitos

- Python y pip.
- Git.
- Entorno virtual de Python.
- Dependencias indicadas en `requirements.txt`.

## 3. Instalación

Clonar el repositorio:

```bash
git clone https://github.com/jessereyav1/TC5061-Proyecto-Equipo17.git
cd TC5061-Proyecto-Equipo17
```

Crear y activar el entorno virtual en Ubuntu:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Instalar dependencias:

```bash
python -m pip install -r requirements.txt
```

## 4. Ejecución del pipeline de datos

Desde la raíz del repositorio, con el entorno virtual activado:

```bash
python -m tc5061_proyecto_equipo17.dataset
```

Este proceso:

1. Descarga los archivos oficiales de MiBici cuando no están disponibles localmente.
2. Valida las columnas y los registros.
3. Normaliza los nombres de variables.
4. Elimina duplicados y registros inválidos.
5. Calcula la duración de los viajes y genera variables temporales.
6. Almacena los resultados procesados en formato Parquet.
7. Genera un manifiesto JSON con metadatos y hashes SHA-256.

En la ejecución verificada se procesaron **752,314 viajes** correspondientes a enero y febrero de 2025. La nomenclatura consultada contiene **484 estaciones**.

La preparación específica de P1 se documenta en la sección 11.4.

## 5. Estructura del repositorio

```text
TC5061-Proyecto-Equipo17/
├── data/                         # Datos locales no versionados
├── docs/                         # Documentación técnica
├── notebooks/                    # Análisis exploratorios
├── references/                   # Material de referencia
├── reports/                      # Resultados y figuras
├── tc5061_proyecto_equipo17/
│   ├── dataset.py                # Ingesta general
│   ├── ingest_p1.py              # Descarga de los 24 CSV para P1
│   ├── urls_p1.json              # Catálogo de URL oficiales para P1
│   ├── dataset_p1.py             # Consolidación horaria de P1
│   ├── features_p1.py            # Características de regresión
│   ├── features.py               # Ingeniería de características
│   └── modeling/                 # Código de modelos y entrada común
│       ├── run_training.py       # Punto de entrada común MLflow
│       ├── train_p1.py           # Entrenamiento P1 (Regresión)
│       ├── train_p2.py           # Entrenamiento P2 (Clasificación)
│       └── train_p3.py           # Entrenamiento P3 (Clustering)
├── tests/                        # Pruebas automatizadas
│   ├── test_data_p1.py           # Pruebas unitarias para dataset_p1
│   ├── test_features_p1.py       # Pruebas unitarias para features_p1
│   └── ...
├── MLproject                     # Entradas de MLflow Projects
├── python_env.yaml               # Entorno reproducible
├── requirements-mlproject.txt    # Dependencias de MLflow Projects
├── requirements.txt
├── requirements-lock.txt
├── pyproject.toml
└── README.md
```

## 6. Pruebas automatizadas

El proyecto cuenta con un conjunto de pruebas automatizadas mediante **PyTest** para asegurar la calidad de los datos, la correcta ingeniería de características y la integridad de las canalizaciones (*pipelines*).

### 6.1. Pruebas de datos generales
Ejecutar desde la raíz del proyecto:

```bash
python -m pytest tests/test_data.py -v
```

Se verifican la normalización de datos, el cálculo de hashes SHA-256 y la transformación de CSV a Parquet.

### 6.2. Pruebas unitarias de P1 (Predicción de Demanda - Regresión)
Para ejecutar exclusivamente las pruebas unitarias correspondientes a los módulos de datos (`dataset_p1.py`) e ingeniería de características (`features_p1.py`):

```bash
python -m pytest tests/test_data_p1.py tests/test_features_p1.py -v
```

Estas pruebas evalúan un total de **24 casos de prueba (normales y límite)**:
- **Procesamiento de datos P1 (`tests/test_data_p1.py`):** Filtrado condicional del rango de fechas válido (Octubre 2024 a Septiembre 2026), mapeo de nombres de columnas alternativas (`trip_start_time`, `from_station_id`), manejo de registros con valores nulos, cálculo de la bandera de fin de semana y agregación correcta de la demanda por hora y estación.
- **Ingeniería de características P1 (`tests/test_features_p1.py`):** Generación de variables de rezago (*lags* de 1h, 2h, 24h, 168h), cálculo de estadísticas móviles en ventanas de 3h y 6h, acotamiento [-1, 1] de transformaciones trigonométricas (seno/coseno) para horas y meses, ordenamiento cronológico por estación y prevención de fuga de datos (*data leakage* temporal).

### 6.3. Ejecución de la suite completa
Para correr todas las pruebas del repositorio:

```bash
python -m pytest tests/ -v
# O bien utilizando el automatizador integrado:
make test
```

## 7. Seguimiento de experimentos

El proyecto utiliza MLflow para registrar experimentos de aprendizaje automático y facilitar la comparación de ejecuciones, parámetros y métricas.

La implementación de cada problema se documenta en sus respectivos módulos y notebooks.

## 8. Reproducibilidad y privacidad

Las dependencias se documentan mediante `requirements.txt` y `requirements-lock.txt`.

Los archivos de datos originales y procesados se mantienen fuera del control de versiones en el estado actual mediante `.gitignore`. Sin embargo, existen archivos de datos en commits históricos, cuya limpieza requiere coordinación con el equipo.

El pipeline general excluye del dataset procesado las columnas de identificador de usuario, género y año de nacimiento. La exclusión de estas columnas no garantiza por sí sola el anonimato de los registros.

La documentación detallada del procesamiento se encuentra en:
`docs/mibici_data_documentation.md`

## 9. Equipo de trabajo

**Equipo 17 — TC5061**

- Rol 1: Datos e infraestructura.
- Rol 2: Regresión.
- Rol 3: Clasificación.
- Rol 4: Clustering.

Cada integrante es responsable de documentar, validar e integrar sus componentes al repositorio colaborativo.

## 10. Referencias

MiBici Guadalajara. (s. f.). *Datos abiertos*. https://www.mibici.net/es/datos-abiertos/

DrivenData. (s. f.). *Cookiecutter Data Science*. https://cookiecutter-data-science.drivendata.org/

---

## 11. MLflow Projects e infraestructura reproducible

El Equipo 17 utiliza MLflow Projects para ejecutar los pipelines de datos y entrenamiento desde una interfaz común.

### 11.1. Entorno de ejecución

El archivo `MLproject` contiene dos puntos de entrada:

- `ingest`: descarga, valida y procesa los datos generales de MiBici (enero y febrero de 2025).
- `train`: selecciona P1, P2 o P3 y ejecuta el entrenamiento correspondiente. Cuando se selecciona P1, prepara primero sus 24 meses de datos.

El entorno se declara en `python_env.yaml` y utiliza `requirements-mlproject.txt` con versiones fijadas de las dependencias.

Versiones utilizadas en las pruebas:

- Python 3.10.22.
- MLflow 3.17.0.

Para utilizar `--env-manager=virtualenv`, es necesario disponer de `pyenv` y `virtualenv` de acuerdo con la configuración del proyecto.

### 11.2. Servidor MLflow con SQLite

Desde la raíz del repositorio, iniciar el servidor:

```bash
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --host 127.0.0.1 \
  --port 5000
```

En otra terminal, configurar la conexión:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

La interfaz estará disponible en http://127.0.0.1:5000.

La base de datos `mlflow.db` se encuentra excluida de Git.

### 11.2.1. Variante para Windows

En Windows, los comandos de las secciones siguientes cambian en tres puntos:

- Escribir cada comando en una sola línea. La barra invertida `\` es de bash; en PowerShell se usa el acento grave `` ` `` y en CMD el signo `^`.
- Definir la conexión con `$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"` en PowerShell, o con `set MLFLOW_TRACKING_URI=http://127.0.0.1:5000` en CMD, en lugar de `export`.
- Usar `--env-manager=local` en `mlflow run` si no se tiene `pyenv`. Esta opción usa el entorno virtual activo, que debe tener las dependencias instaladas con `pip install -r requirements-mlproject.txt` y `pip install -e .`.

Ejemplo en PowerShell:

```powershell
mlflow run . -e train -P question=p3 -P model=kmeans -P k=4 --experiment-name mibici-equipo17 --env-manager=local
```

### 11.3. Ejecutar la ingesta general

```bash
mlflow run . \
  -e ingest \
  --experiment-name mibici-equipo17 \
  --env-manager=virtualenv
```

Esta ingesta descarga los registros de enero y febrero de 2025, genera archivos Parquet y produce `manifest.json` para documentar el origen e integridad de los datos. Es diferente de la ingesta específica de P1.

### 11.4. Entrenamiento P1: regresión de demanda

P1 utiliza **24 meses de datos de MiBici**, desde octubre de 2024 hasta septiembre de 2026. Su preparación se activa automáticamente al ejecutar `train` con `question=p1`:

1. `ingest_p1.py` lee `urls_p1.json`, descarga los 24 CSV requeridos y comprueba que existan y no estén vacíos.
2. `dataset_p1.py` consolida la demanda por estación y hora en `data/processed/dataset_regresion_horario.parquet`.
3. `features_p1.py` genera rezagos, estadísticas móviles y otras características en `data/processed/features_p1.parquet`.
4. `train_p1.py` entrena el modelo y registra resultados en MLflow.

El punto de entrada evita regenerar archivos Parquet que ya tengan cobertura de los 24 meses. **Esta comprobación de cobertura no garantiza que los archivos sean íntegros o estén actualizados** con respecto a sus fuentes; se requiere una validación adicional para detectar esos casos.

Ejemplo reproducido en las pruebas de infraestructura (20 árboles, profundidad máxima 12):

```bash
mlflow run . \
  -e train \
  -P question=p1 \
  -P model=rf \
  -P n_estimators=20 \
  -P max_depth=12 \
  --experiment-name mibici-equipo17 \
  --env-manager=virtualenv
```

La división temporal es estricta:

| Partición | Periodo | Registros después de características |
|---|---|---:|
| Entrenamiento | Octubre de 2024 a julio de 2026 | 2,705,545 |
| Validación | Agosto de 2026 | 138,930 |
| Prueba | Septiembre de 2026 | 154,793 |

Resultados de la **prueba de infraestructura**, no necesariamente del modelo definitivo del responsable de P1:

| Métrica | Validación | Prueba |
|---|---:|---:|
| RMSE | 1.9985 | 2.2563 |
| R² | 0.5380 | 0.5653 |

En una ejecución verificada se consolidaron **3,078,203 registros estación-hora**, de los cuales **2,999,268** permanecieron después de la ingeniería de características. No hubo meses faltantes, valores nulos ni duplicados estación-hora en la validación estructural del dataset consolidado.

Se completó con éxito una ejecución integral de P1 desde un clon limpio del repositorio local: descarga, consolidación, características, entrenamiento y registro en MLflow. En esa prueba se reutilizó un entorno virtual que MLflow había creado anteriormente.

### 11.5. Entrenamiento P2: clasificación de riesgo

`train_p2.py` decide qué hacer según los hiperparámetros que recibe:

- **Sin hiperparámetros → búsqueda:** una corrida padre con una hija baseline (persistencia) y una hija por combinación (`n_estimators` ∈ {100, 300} × `max_depth` ∈ {8, 12, 16}). Elige con validación (25–31 de enero, `FECHA_CORTE_VAL`) por mayor recall, con PR-AUC como desempate; reentrena al ganador con todo enero y lo evalúa en febrero.
- **Con hiperparámetros → candidato:** un Random Forest en una sola corrida, con métricas de validación y de prueba.

Búsqueda (reproduce el modelo en producción, RF 100/12):

```bash
mlflow run . -e train -P question=p2 -P model=rf \
  --experiment-name mibici-equipo17 --env-manager=virtualenv
```

Candidatos evaluados:

```bash
mlflow run . -e train -P question=p2 -P model=rf -P n_estimators=100 -P max_depth=12 --experiment-name mibici-equipo17 --env-manager=virtualenv
mlflow run . -e train -P question=p2 -P model=rf -P n_estimators=300 -P max_depth=16 --experiment-name mibici-equipo17 --env-manager=virtualenv
mlflow run . -e train -P question=p2 -P model=rf -P n_estimators=200 -P max_depth=8  --experiment-name mibici-equipo17 --env-manager=virtualenv
```

Registro en el Model Registry (`mibici-equipo17-p2-riesgo`): toma la última búsqueda como `@champion`, opcionalmente otra corrida como `@challenger`, y verifica la carga por alias:

```bash
python -m tc5061_proyecto_equipo17.modeling.register_p2 --challenger-run-id <run_id de p2-rf-n200-d8>
```

**En Windows (PowerShell):** usar `$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"` en lugar de `export`, y `--env-manager=local` con un `.venv` que tenga las versiones de `requirements-lock.txt` (pandas 2.3.3, scikit-learn 1.7.2). Con pandas 3 cambia `version_datos` aunque los datos sean los mismos.

### 11.6. Entrenamiento P3: clustering de estaciones

En P3, `-P model` elige el algoritmo (`kmeans`, `kmeans-escalado` o `gmm`), `-P variables` las variables (`completas` o `solo-horas`) y `-P etapa` la etapa del trabajo (`baseline` o `candidate`).

`mlflow run` guarda `-P k` en la corrida antes de entrenar y MLflow no permite cambiar un parámetro ya guardado. Por eso P3 exige indicar `-P k` (entero mayor o igual a 2) o usar `-P buscar_k=1`.

Baseline:

```bash
mlflow run . \
  -e train \
  -P question=p3 \
  -P model=kmeans \
  -P k=4 \
  --experiment-name mibici-equipo17 \
  --env-manager=virtualenv
```

Candidatas:

```bash
# K-means con StandardScaler
mlflow run . -e train -P question=p3 -P model=kmeans-escalado -P etapa=candidate -P k=4 \
  --experiment-name mibici-equipo17 --env-manager=virtualenv

# Mezcla gaussiana (covarianza diagonal)
mlflow run . -e train -P question=p3 -P model=gmm -P etapa=candidate -P k=4 \
  --experiment-name mibici-equipo17 --env-manager=virtualenv

# K-means solo con el perfil horario
mlflow run . -e train -P question=p3 -P model=kmeans -P variables=solo-horas -P etapa=candidate -P k=4 \
  --experiment-name mibici-equipo17 --env-manager=virtualenv
```

Búsqueda de k con corridas anidadas. Crea una corrida padre con una hija por cada k de 2 a 8, elige el mejor k dentro del rango operativo 3-6 según el silhouette de prueba y lo reentrena como candidata:

```bash
mlflow run . \
  -e train \
  -P question=p3 \
  -P model=kmeans \
  -P buscar_k=1 \
  --experiment-name mibici-equipo17 \
  --env-manager=virtualenv
```

### 11.7. Model Registry

Los modelos registrados utilizarán los siguientes nombres:

- `mibici-equipo17-p1-demanda`
- `mibici-equipo17-p2-riesgo`
- `mibici-equipo17-p3-grupos`

Cada responsable seleccionará la versión de su modelo y le asignará el alias `champion`.

El alias permite actualizar la versión seleccionada sin cambiar la referencia utilizada para consumir el modelo.

El registro de modelos y la asignación de alias corresponden a los responsables de P1, P2 y P3.

### 11.8. Reproducibilidad y archivos excluidos

Los datos, modelos serializados, bases de datos locales, artefactos y credenciales no deben versionarse en Git.

Se comprobó la ejecución de la ingesta general mediante MLflow Projects utilizando el entorno local y un entorno virtual independiente. P2 y P3 se ejecutaron correctamente mediante MLflow Projects con el entorno local.

Para P1 se comprobó desde un clon limpio del repositorio local la descarga de sus 24 CSV, la reconstrucción de sus Parquet y el entrenamiento con registro en MLflow. Las métricas coincidieron con las de la ejecución realizada previamente con los mismos hiperparámetros.

Queda pendiente validar todos los entrenamientos desde un clon limpio y con una instalación del entorno totalmente nueva, así como mejorar la detección de archivos de datos desactualizados. También se requiere coordinar la limpieza de datasets versionados en commits antiguos de Git, sin alterar unilateralmente el historial compartido.

### 12 Gestión del Ciclo de Vida del Modelo (Model Registry)

#### Esquema Seleccionado: Aliases
Para la gestión del ciclo de vida del modelo correspondiente a **P1 (Predicción de Demanda MiBici)**, se optó por implementar el **esquema moderno de Aliases de MLflow** (utilizando la etiqueta `@champion`) en lugar del esquema tradicional de etapas (`Staging` / `Production`).

#### Justificación Técnica de la Elección:
1. **Flexibilidad y Desacoplamiento:** El uso de etapas tradicionales (`Production`/`Staging`) se encuentra oficialmente en estado de obsolescencia (*deprecated*) por MLflow. El esquema de aliases permite asignar etiquetas semánticas y dinámicas a versiones específicas del modelo sin imponer transiciones rígidas de estado.
2. **Consumo Simplificado en Entornos de Producción:** Permite que las aplicaciones consumidoras o APIs de inferencia carguen siempre la última versión validada utilizando un nombre estático y legible (`models:/mibici-equipo17-p1-demanda@champion`), eliminando el riesgo de depender de números de versión hardcodeados o cambios manuales de archivos.
3. **Soporte para Estrategias A/B y Despliegues Progresivos:** El esquema facilita la convivencia fluida de alias adicionales (como `@challenger` o `@candidate`) para pruebas A/B o despliegues en la sombra (*shadow deployments*).

#### Detalle del Modelo Registrado:
- **Nombre en Registry:** `mibici-equipo17-p1-demanda`
- **Versión en Producción:** `Version 2` (Baseline `n_estimators=150`, `max_depth=20`)
- **Alias Asignado:** `@champion`
- **Linaje:** Vinculado directamente al experimento `mibici-equipo17`, commit de Git activo y versión del dataset manifiesto SHA-256 (`18d854ce642b`).
- **Motivo de Selección:** *Se eligió esta versión por registrar el mejor desempeño absoluto y la mayor estabilidad predictiva tanto en el conjunto de validación ($RMSE = 1.9763$, $R^2 = 0.5482$) como en el de prueba ($RMSE = 2.2219$, $R^2 = 0.5785$) en comparación con las variantes candidatas evaluadas.*
- **Verificación de Carga:** *Se ejecuta satisfactoriamente el script `verificar_p1.py`, cargando con éxito el modelo en producción mediante la URI del alias (`models:/mibici-equipo17-p1-demanda@champion`).*

## Seguridad y pruebas automatizadas

El proyecto incorpora controles de seguridad mediante Bandit,
pip-audit y detect-secrets.

### Instalación

Instalar las herramientas de auditoría:

    python -m pip install -r requirements-security.txt

### Comandos

Ejecutar todas las pruebas:

    make test

Analizar el código Python:

    make security-bandit

Detectar posibles secretos:

    make security-secrets

Auditar las dependencias instaladas:

    make security-audit

Ejecutar todas las auditorías:

    make security

### Estado de seguridad

- Bandit: 0 hallazgos medios y altos; 12 bajos.
- pip-audit: 43 vulnerabilidades conocidas pendientes.
- PyTest: 17 pruebas aprobadas.
- detect-secrets: 0 hallazgos en archivos incluidos; dos notebooks
  permanecen excluidos temporalmente.

La ejecución completa de make security puede devolver un error
debido a las vulnerabilidades pendientes de pip-audit.

Para consultar los resultados, las limitaciones y el plan de
remediación, revisar docs/seguridad.md.

## Actividad 7 — Validación integral de calidad y seguridad

### Objetivo

Validar los componentes del pipeline MLOps de MiBici Guadalajara
mediante pruebas unitarias, verificaciones de calidad y drift de datos,
pruebas de reproducibilidad y controles básicos de seguridad.

### Resultados consolidados

La suite completa se ejecutó sobre la rama `main`, con los cambios
de los cuatro integrantes integrados en el mismo repositorio.

| Indicador | Resultado |
|---|---:|
| Pruebas aprobadas | **94** |
| Pruebas fallidas | **0** |
| Pruebas omitidas | **0** |
| Tiempo de ejecución | 19.28 segundos |

Entorno de validación: Ubuntu WSL, Python 3.10.22 y PyTest 9.1.1.

Comando utilizado:

    make test

Los resultados anteriores corresponden a una ejecución consolidada
y no a la suma de las evidencias individuales.

### Participación y evidencias individuales

| Responsable | Área | Resultado reportado |
|---|---|---|
| Esteban | P1 - Regresión | 28 pruebas aprobadas |
| Diego | P2 - Clasificación | 23 pruebas aprobadas |
| Marcos | P3, reproducibilidad y DeepChecks | 31 pruebas aprobadas |
| Rol 1 | Seguridad e infraestructura | 17 pruebas aprobadas |

Las cifras individuales corresponden a distintas ejecuciones y
pueden contener pruebas comunes.

### P1 — Regresión (Esteban)

Archivos principales:

- `tests/test_data_p1.py`
- `tests/test_features_p1.py`

Se verificaron la construcción del dataset de regresión, los
casos límite, el manejo de archivos y fechas, la normalización
de columnas, los valores nulos, las variables rolling y lag,
así como la prevención de fuga de información temporal.

Comando de referencia:

    python -m pytest tests/test_data_p1.py tests/test_features_p1.py -v

Resultado individual reportado: 28 pruebas aprobadas.

### P2 — Clasificación (Diego)

Archivo principal:

- `tests/test_features_p2.py`

Se verificaron los flujos de viajes, la construcción de rejillas
horarias, los umbrales de riesgo, las variables temporales y la
prevención de fuga de información hacia el objetivo.

Comando de referencia:

    python -m pytest tests/test_features_p2.py -v

Resultado individual reportado: 23 pruebas aprobadas.

### P3 — Agrupamiento y reproducibilidad (Marcos)

Archivos principales:

- `tests/test_features_p3.py`
- `tests/test_reproducibilidad.py`
- `tests/test_deepchecks.py`

Se validaron perfiles horarios, proporciones, matrices de
características, manejo de valores faltantes y consistencia
de resultados con semillas controladas.

Resultado individual reportado: 31 pruebas aprobadas en la suite
disponible en su entorno.

### DeepChecks — Calidad de datos y drift

Se verificaron la estructura de variables, valores faltantes,
duplicados y comportamiento de los datos ante alteraciones.

También se compararon datos reales de enero y febrero de 2025.

Comando:

    python -m pytest tests/test_deepchecks.py -v -rs

Resultado consolidado de este archivo: 9 pruebas aprobadas.

Indicadores de diferencia observados entre periodos:

| Variable | Indicador |
|---|---:|
| `prop_findes` | 0.197377 |
| `h12` | 0.090762 |
| `h17` | 0.085594 |
| `h20` | 0.080552 |
| `h10` | 0.074710 |

La interpretación de estos indicadores como drift significativo
depende de los criterios de las pruebas.

### Seguridad e infraestructura (Rol 1)

Se incorporaron controles para:

- Validar URL HTTPS y redirecciones en las descargas.
- Ejecutar pruebas de seguridad con PyTest.
- Revisar código Python con Bandit.
- Auditar dependencias mediante pip-audit.
- Detectar posibles secretos con detect-secrets.
- Proteger archivos sensibles con .gitignore.
- Ejecutar las auditorías desde el Makefile.

Resultados de la auditoría documentada:

- Bandit: 0 hallazgos medios o altos; 12 bajos.
- pip-audit: 43 hallazgos pendientes en el entorno evaluado.
- detect-secrets: 0 coincidencias en archivos incluidos, con dos
  notebooks temporalmente excluidos del análisis.

Consultar `docs/seguridad.md` para conocer los riesgos pendientes,
las limitaciones y las medidas recomendadas.

### Instalación de dependencias para pruebas

Para reproducir las verificaciones de la Actividad 7
se incorporó el archivo `requirements-test.txt`.

Incluye las dependencias del proyecto, herramientas de seguridad,
PyTest y las versiones necesarias para ejecutar DeepChecks
en Python 3.10.

Desde la raíz del repositorio, instalar:

    python -m pip install -r requirements-test.txt

Verificar las dependencias:

    python -m pip check

Ejecutar toda la suite:

    make test

El entorno validado utiliza `deepchecks==0.19.1` y
`setuptools==80.9.0`. Esta última versión proporciona
`pkg_resources`, necesario para la compatibilidad con
DeepChecks.

Se recomienda utilizar un entorno virtual para evitar
conflictos con otros proyectos.

### Conclusión

La ejecución consolidada confirmó 94 pruebas aprobadas, sin fallos
ni omisiones, cubriendo P1, P2, P3, reproducibilidad, DeepChecks
y controles básicos de seguridad.

Los resultados reflejan la funcionalidad validada en el entorno
evaluado; no implican ausencia de vulnerabilidades ni sustituyen
la revisión de los riesgos pendientes.

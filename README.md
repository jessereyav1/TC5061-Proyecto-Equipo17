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
│   ├── dataset_regression.py     # Consolidación horaria de P1
│   ├── features_regression.py    # Características de regresión
│   ├── features.py               # Ingeniería de características
│   └── modeling/                 # Código de modelos y entrada común
├── tests/                        # Pruebas automatizadas
├── MLproject                     # Entradas de MLflow Projects
├── python_env.yaml               # Entorno reproducible
├── requirements-mlproject.txt    # Dependencias de MLflow Projects
├── requirements.txt
├── requirements-lock.txt
├── pyproject.toml
└── README.md
```

## 6. Pruebas automatizadas

Ejecutar desde la raíz del proyecto:

```bash
python -m pytest tests/test_data.py -v
```

Se verificaron cinco pruebas relacionadas con la normalización de datos, el cálculo de hashes SHA-256 y la transformación de CSV a Parquet.

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
2. `dataset_regression.py` consolida la demanda por estación y hora en `data/processed/dataset_regresion_horario.parquet`.
3. `features_regression.py` genera rezagos, estadísticas móviles y otras características en `data/processed/features_regresion.parquet`.
4. `train_regression.py` entrena el modelo y registra resultados en MLflow.

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

```bash
mlflow run . \
  -e train \
  -P question=p2 \
  -P model=rf \
  -P n_estimators=100 \
  -P max_depth=12 \
  --experiment-name mibici-equipo17 \
  --env-manager=virtualenv
```

### 11.6. Entrenamiento P3: clustering de estaciones

```bash
mlflow run . \
  -e train \
  -P question=p3 \
  -P model=kmeans \
  -P k=4 \
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

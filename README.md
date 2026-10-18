
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

El pipeline utiliza registros de viajes de enero y febrero de 2025, además de información de nomenclatura y capacidad de estaciones.

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

## 5. Estructura del repositorio

```text
TC5061-Proyecto-Equipo17/
├── data/                         # Datos locales no versionados
├── docs/                         # Documentación técnica
├── notebooks/                    # Análisis exploratorios
├── references/                   # Material de referencia
├── reports/                      # Resultados y figuras
├── tc5061_proyecto_equipo17/
│   ├── dataset.py                # Pipeline de datos
│   ├── features.py               # Ingeniería de características
│   └── modeling/                 # Código de modelos
├── tests/                        # Pruebas automatizadas
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

Los archivos de datos originales y procesados se mantienen fuera del control de versiones mediante `.gitignore`.

El pipeline excluye del dataset procesado las columnas de identificador de usuario, género y año de nacimiento. La exclusión de estas columnas no garantiza por sí sola el anonimato de los registros.

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



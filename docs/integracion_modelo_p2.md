# Integración del modelo P2 con FastAPI y Docker

## Modelo seleccionado

- Proyecto: MiBici Guadalajara, Equipo 17.
- Objetivo: clasificación del riesgo para la siguiente hora.
- Algoritmo: Random Forest.
- Registry: `mibici-equipo17-p2-riesgo`.
- Versión: `1`.
- URI: `models:/mibici-equipo17-p2-riesgo/1`.

## Modelo exportado

El modelo fue recuperado desde MLflow Model Registry
y empaquetado como:

`mibici-equipo17-p2-v1.tar.gz`

El archivo comprimido debe obtenerse del responsable del Rol 1.

Desde la raíz del repositorio, extraerlo con:

    mkdir -p artifacts
    tar -xzf mibici-equipo17-p2-v1.tar.gz -C artifacts

La estructura esperada es:

    artifacts/
      modelo_p2/
        MLmodel
        model.skops
        requirements.txt
        conda.yaml
        python_env.yaml

La carpeta `artifacts/` se excluye de Git por medio de `.gitignore`.

## Variables de entrada

| Variable | Tipo |
|---|---|
| hora_dia | int32 |
| dia_semana | int32 |
| fin_de_semana | int64 |
| capacidad | int64 |
| pct_ocupacion | float64 |
| riesgo_ahora | int64 |
| neto | int64 |
| neto_ult_3h | float64 |

Todas son obligatorias.

La salida del modelo es un arreglo de enteros int64.

## Integración con FastAPI

El módulo reutilizable se encuentra en:

`tc5061_proyecto_equipo17/model_serving.py`

Para utilizarlo:

    from tc5061_proyecto_equipo17.model_serving import predict_p2

La función `predict_p2()` recibe un DataFrame de pandas
con las ocho variables requeridas.

La función `load_p2_model()` carga el modelo una sola vez
por proceso utilizando caché.

La API deberá convertir los datos validados por Pydantic
a un DataFrame con los tipos adecuados antes de llamar
a `predict_p2()`.

## Configuración del modelo

Por defecto se utiliza:

`artifacts/modelo_p2`

Opcionalmente, se puede establecer:

    export P2_MODEL_URI=models:/mibici-equipo17-p2-riesgo/1

Esta opción requiere acceso al almacenamiento de MLflow
que contiene el Model Registry y los artefactos.

No se debe asumir que la base SQLite local del Rol 1
estará disponible desde otras computadoras.

## Integración con Docker

Para una imagen autocontenida:

1. Obtener el paquete del modelo exportado.
2. Extraerlo en `artifacts/modelo_p2`.
3. Incorporar esa carpeta durante la construcción de la imagen.
4. Instalar las dependencias compatibles con el modelo.
5. Configurar la ruta del modelo si es necesario.
6. Ejecutar FastAPI y validar `/predict`.

El Dockerfile y el contexto de construcción deben permitir
incluir explícitamente el modelo. Revisar las reglas de
`.dockerignore`, si existen.

No incorporar al contenedor bases SQLite, datasets completos
ni información confidencial.

## Validaciones realizadas

- Registro en Model Registry: exitoso.
- Recuperación por nombre y versión: exitosa.
- Predicción real: exitosa.
- Exportación del modelo: exitosa.
- Carga desde la carpeta exportada: exitosa.
- Extracción y carga desde otra ubicación: exitosa.
- Pruebas unitarias del módulo: 3 aprobadas.

## Limitaciones

Las pruebas se realizaron en Ubuntu WSL con Python 3.10.

La carga del modelo en otra ubicación no demuestra por sí sola
compatibilidad con otros sistemas operativos ni con Docker.

El contenedor final deberá ejecutar una predicción real
con el modelo empaquetado antes de considerarse validado.

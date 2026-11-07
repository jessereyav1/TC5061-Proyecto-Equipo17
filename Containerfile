# Imagen de la API de riesgo de estaciones (P2) - Equipo 17.
# Funciona igual con Podman o Docker (estándar OCI):
#   podman build -t mibici-p2-api -f Containerfile .
#   docker build -t mibici-p2-api -f Containerfile .

# Misma versión mayor de Python con la que se desarrolló el proyecto (3.10)
FROM python:3.10-slim

# - sin .pyc ni buffer de salida: los logs aparecen al instante en "podman logs"
# - sin caché de pip: la imagen pesa menos
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# 1) Primero solo las dependencias: mientras requirements-api.txt no cambie, Docker/Podman
#    reutiliza esta capa y no vuelve a descargar todo en cada build.
COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

# 2) Después el código. La API es un solo archivo, app.py (raíz del repositorio); no se copian
#    datos, modelos ni notebooks (ver .dockerignore).
COPY app.py .

# El proceso no corre como root dentro del contenedor
RUN useradd --create-home --uid 10001 apiuser
USER apiuser

# Configuración por defecto; se sobrescribe con "-e VARIABLE=valor" al ejecutar.
# MLFLOW_TRACKING_URI apunta al servidor MLflow de la máquina anfitriona
# (host.containers.internal es el nombre con el que Podman ve a la máquina host).
# El nombre del modelo (mibici-equipo17-p2-riesgo) y el alias (champion) están definidos en app.py.
ENV MLFLOW_TRACKING_URI=http://host.containers.internal:5000 \
    APP_MODULE=app:app \
    PORT=8000

EXPOSE 8000

# "exec" hace que uvicorn sea el proceso principal y reciba la señal de parada (Ctrl+C)
CMD ["sh", "-c", "exec uvicorn ${APP_MODULE} --host 0.0.0.0 --port ${PORT}"]

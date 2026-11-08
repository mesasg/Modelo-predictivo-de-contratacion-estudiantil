# Imagen para ejecutar train.py y predict.py (Fase 2).
# No entrena durante la construcción: los datos, modelos y resultados se montan al ejecutar el contenedor.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencias primero para reutilizar la capa mientras requirements.txt no cambie
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ src/
COPY train.py predict.py pytest.ini ./
COPY tests/ tests/

# Sin ENTRYPOINT: el comando se indica al ejecutar (python train.py ... o python predict.py ...)
CMD ["python", "predict.py", "--help"]

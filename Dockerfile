# Usamos una imagen de Python ligera (slim)
FROM python:3.9-slim

# Evitar archivos .pyc y asegurar logs en tiempo real
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# INSTALACIÓN DE DEPENDENCIAS DEL SISTEMA
# - libgl1 y libglib2.0: Necesarias para OpenCV
# - gcc, g++, python3-dev: Necesarias para compilar Mahotas
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    gcc \
    g++ \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar e instalar requerimientos primero (para aprovechar el cache de capas)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del proyecto (incluyendo mobilenet_v2.onnx)
COPY . .

# Exponer el puerto que usa Flask (Render lo detectará automáticamente)
EXPOSE 5001

# Ejecutar con Gunicorn
# IMPORTANTE: --workers 1 para que el estado del clustering no se duplique
CMD ["gunicorn", "--bind", "0.0.0.0:5001", "--workers", "1", "--timeout", "120", "app:app"]
FROM python:3.10-slim

# 1. Configuraciones de entorno
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FLASK_ENV=production

WORKDIR /app

# 2. Instalar dependencias del sistema (necesarias para librerías de IA y OpenCV)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 3. Instalar dependencias de Python
# Copiamos primero requirements para aprovechar la caché de Docker
COPY requirements.txt .

RUN pip install --upgrade pip \
    && pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

# 4. Copiar el código fuente y archivos de configuración
COPY app.py models_architecture.py metrics.json ./
COPY utils ./utils

# 5. Manejo del modelo de IA
# Creamos la carpeta y copiamos el archivo .pth
RUN mkdir -p /app/models
COPY models/DenseNet121.pth /app/models/DenseNet121.pth

# 6. Configurar el script de entrada (Entrypoint)
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Puerto que usa Flask
EXPOSE 5000

# 7. Verificación de salud del contenedor
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/health')" || exit 1

# 8. Ejecución
ENTRYPOINT ["/entrypoint.sh"]
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "4", "--timeout", "120", "app:app"]
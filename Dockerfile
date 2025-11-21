# imagen oficial python ligera
FROM python:3.10-slim 

#directorio de trabajo, creación de carptea app dentro del contenedor
WORKDIR /app

COPY requirements-api.txt .

# instalación de dependencias, 
# paquetes de linux
# librerías del sistema 
# RUN apt-get update && apt-get install -y libgomp1
RUN apt-get update && apt-get install -y libgomp1


# instalación de dependencias de python
# no guarda caché dentro del directorio
RUN pip install --no-cache-dir -r requirements-api.txt


# copia todo lo demas del proyecto local 
# WORDIR (/ap) dentro del contenedor
COPY . .

# comando de arranque
CMD ["uvicorn", "api_prediccion:app", "--host", "0.0.0.0", "--port", "8000"]

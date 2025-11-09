# Librerías generales
import os
from scripts.data_loader import cargar_datos
from scripts.data_cleaning import (
    limpiar_columnas_no_usadas,
    convertir_tipos,
    normalizar_codigos,
    validar_integridad,
    rellenar_coordenadas
)
from scripts.data_merge import unir_catalogos

from scripts.data_features import generar_nuevas_columnas, resumen_causas
from scripts.data_saving import guardar_datos_limpios



# === RUTAS ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

# Archivos CSV
DATA_PATH_VUELOS = os.path.join(DATA_DIR, "flights.csv")
DATA_PATH_AEROPUERTOS = os.path.join(DATA_DIR, "airports.csv")
DATA_PATH_AEROLINEAS = os.path.join(DATA_DIR, "airlines.csv")

# Carpeta de salida
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed", "flights_clean.csv")

# === EJECUCIÓN ===
if __name__ == "__main__":
    print(f"🚀 Ejecutando script desde: {os.path.abspath(__file__)}")

    # Cargar los tres datasets (usa nrows=100000 para probar sin saturar memoria)
    vuelos = cargar_datos(DATA_PATH_VUELOS) #, nrows=100000)
    aeropuertos = cargar_datos(DATA_PATH_AEROPUERTOS)
    aerolineas = cargar_datos(DATA_PATH_AEROLINEAS)

    # === Validaciones básicas ===
    if vuelos is not None:
        print("\n📊 Información de flights.csv:")
        print(vuelos.info(memory_usage='deep', show_counts=True))
        print(vuelos.head(3))
    else:
        print("❌ Error al cargar flights.csv")

    if aeropuertos is not None:
        print("\n📍 Información de airports.csv:")
        print(aeropuertos.info(memory_usage='deep', show_counts=True))
        print(aeropuertos.head(3))
    else:
        print("❌ Error al cargar airports.csv")

    if aerolineas is not None:
        print("\n✈️ Información de airlines.csv:")
        print(aerolineas.info(memory_usage='deep', show_counts=True))
        print(aerolineas.head(3))
    else:
        print("❌ Error al cargar airlines.csv")

# === Limpieza de datos ===
# if vuelos is not None and aeropuertos is not None and aerolineas is not None:
    vuelos = limpiar_columnas_no_usadas(vuelos)
    vuelos, aeropuertos, aerolineas = convertir_tipos(vuelos, aeropuertos, aerolineas)
    vuelos, aerolineas, aeropuertos = normalizar_codigos(vuelos, aerolineas, aeropuertos)
    vuelos = validar_integridad(vuelos, aerolineas, aeropuertos)
    aeropuertos = rellenar_coordenadas(aeropuertos)

    v = unir_catalogos(vuelos, aerolineas, aeropuertos)
# else:
#     print("❌ No se pueden limpiar los datos porque no se cargaron correctamente.")
    
# # === Generar nuevas columnas ===
# if v is not None:
    v = unir_catalogos(vuelos, aerolineas, aeropuertos)
    v = generar_nuevas_columnas(v)
    dist_causas = resumen_causas(v)
    print(dist_causas)
# else:
#     print("❌ No se pueden generar nuevas columnas porque no se unieron los catalogos.")
    

    # === Guardar dataset final ===
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, "data", "processed")
    guardar_datos_limpios(v, OUTPUT_DIR)
else:
    print("❌ No se pueden guardar los datos porque no se limpiaron correctamente.")
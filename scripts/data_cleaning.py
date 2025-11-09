import pandas as pd
import numpy as np

# =========================
#  LIMPIEZA DE DATOS
# =========================

def limpiar_columnas_no_usadas(vuelos: pd.DataFrame) -> pd.DataFrame:
    """Elimina columnas irrelevantes para el modelo de retrasos."""
    columnas_eliminar = [
        "YEAR", "FLIGHT_NUMBER", "TAIL_NUMBER", "WHEELS_OFF",
        "TAXI_OUT", "ELAPSED_TIME", "AIR_TIME", "WHEELS_ON",
        "TAXI_IN", "CANCELLATION_REASON"
    ]
    print("🧹 Eliminando columnas no necesarias...")
    vuelos = vuelos.drop(columns=[c for c in columnas_eliminar if c in vuelos.columns], errors="ignore")
    print(f"✅ Columnas restantes: {len(vuelos.columns)}")
    return vuelos


def convertir_tipos(vuelos, aeropuertos, aerolineas):
    """Convierte columnas a tipos categóricos para optimizar memoria."""
    print("🔄 Convirtiendo tipos de datos...")

    for c in ["AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT"]:
        if c in vuelos.columns:
            vuelos[c] = vuelos[c].astype("category")

    for c in ["IATA_CODE", "AIRPORT", "CITY", "STATE", "COUNTRY"]:
        if c in aeropuertos.columns:
            aeropuertos[c] = aeropuertos[c].astype("category")

    for c in ["IATA_CODE", "AIRLINE"]:
        if c in aerolineas.columns:
            aerolineas[c] = aerolineas[c].astype("category")

    print("✅ Conversión de tipos finalizada.")
    return vuelos, aeropuertos, aerolineas


def normalizar_codigos(vuelos, aerolineas, aeropuertos):
    """Normaliza códigos y mayúsculas para evitar errores de join."""
    print("✈️ Normalizando códigos (espacios y mayúsculas)...")

    for df, col in [
        (vuelos, "AIRLINE"),
        (vuelos, "ORIGIN_AIRPORT"),
        (vuelos, "DESTINATION_AIRPORT"),
        (aerolineas, "IATA_CODE"),
        (aeropuertos, "IATA_CODE")
    ]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.upper()

    return vuelos, aerolineas, aeropuertos


def validar_integridad(vuelos, aerolineas, aeropuertos):
    """Calcula porcentaje de registros inválidos entre catálogos."""
    print("🔍 Validando integridad entre datasets...")

    set_aerolineas = set(aerolineas["IATA_CODE"])
    set_aeropuertos = set(aeropuertos["IATA_CODE"])

    mask_aerolinea_invalida = ~vuelos["AIRLINE"].isin(set_aerolineas)
    mask_origen_invalido = ~vuelos["ORIGIN_AIRPORT"].isin(set_aeropuertos)
    mask_destino_invalido = ~vuelos["DESTINATION_AIRPORT"].isin(set_aeropuertos)

    total = len(vuelos)
    print(f"  Aerolíneas no válidas: {(mask_aerolinea_invalida.mean() * 100):.3f}%")
    print(f"  Aeropuertos origen no válidos: {(mask_origen_invalido.mean() * 100):.3f}%")
    print(f"  Aeropuertos destino no válidos: {(mask_destino_invalido.mean() * 100):.3f}%")

    vuelos_validos = vuelos[
        (vuelos["CANCELLED"] == 0) &
        (vuelos["DIVERTED"] == 0) &
        (~mask_aerolinea_invalida) &
        (~mask_origen_invalido) &
        (~mask_destino_invalido)
    ].copy()

    print(f"✅ Registros válidos: {len(vuelos_validos):,} ({len(vuelos_validos)/total*100:.2f}% del total)")
    return vuelos_validos


def rellenar_coordenadas(aeropuertos):
    """Completa coordenadas faltantes para tres aeropuertos."""
    coords_faltantes = {
        "ECP": {"LATITUDE": 30.357106, "LONGITUDE": -85.795414},
        "PBG": {"LATITUDE": 44.6509, "LONGITUDE": -73.4681},
        "UST": {"LATITUDE": 29.9592, "LONGITUDE": -81.3398},
    }

    registros_actualizados = 0
    for codigo, valores in coords_faltantes.items():
        mask = aeropuertos["IATA_CODE"] == codigo
        if mask.any():
            aeropuertos.loc[mask, ["LATITUDE", "LONGITUDE"]] = (
                valores["LATITUDE"], valores["LONGITUDE"]
            )
            registros_actualizados += 1

    print(f"📍 Coordenadas actualizadas: {registros_actualizados}")
    return aeropuertos

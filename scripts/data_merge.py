import pandas as pd

def unir_catalogos(vuelos, aerolineas, aeropuertos):
    """Une vuelos con aerolíneas, aeropuertos origen y destino."""
    print("🔗 Iniciando merge de catálogos...")

    # Aerolíneas
    aerolineas_ren = aerolineas.rename(
        columns={"IATA_CODE": "AIRLINE", "AIRLINE": "AIRLINE_NAME"}
    )
    v = pd.merge(vuelos, aerolineas_ren, on="AIRLINE", how="left")

    # Aeropuerto origen
    aerop_origen = aeropuertos.rename(
        columns={
            "IATA_CODE": "ORIGIN_AIRPORT",
            "AIRPORT": "ORIGEN_AEROPUERTO",
            "CITY": "ORIGEN_CIUDAD",
            "STATE": "ORIGEN_ESTADO",
            "COUNTRY": "ORIGEN_PAIS",
            "LATITUDE": "ORIGEN_LAT",
            "LONGITUDE": "ORIGEN_LON",
        }
    )
    v = pd.merge(v, aerop_origen, on="ORIGIN_AIRPORT", how="left")

    # Aeropuerto destino
    aerop_dest = aeropuertos.rename(
        columns={
            "IATA_CODE": "DESTINATION_AIRPORT",
            "AIRPORT": "DEST_AEROPUERTO",
            "CITY": "DEST_CIUDAD",
            "STATE": "DEST_ESTADO",
            "COUNTRY": "DEST_PAIS",
            "LATITUDE": "DEST_LAT",
            "LONGITUDE": "DEST_LON",
        }
    )
    v = pd.merge(v, aerop_dest, on="DESTINATION_AIRPORT", how="left")

    print("✅ Merge completado.")
    print(f"Dimensiones finales: {v.shape[0]:,} filas × {v.shape[1]} columnas.")
    return v

import os
import pandas as pd

# =========================
# FUNCIÓN PRINCIPAL
# =========================

def guardar_datos_limpios(df: pd.DataFrame, path_salida: str, nombre_archivo: str = "flights_clean.csv"):
    """
    Guarda el DataFrame procesado en formato CSV dentro de la carpeta 'processed',
    eliminando columnas no necesarias para modelado.
    """
    print("\n💾 Iniciando guardado del archivo procesado...")

    columnas_eliminar = [
        "DIVERTED",
        "CANCELLED",
        "AIR_SYSTEM_DELAY",
        "SECURITY_DELAY",
        "AIRLINE_DELAY",
        "LATE_AIRCRAFT_DELAY",
        "WEATHER_DELAY"
    ]

    df = df.drop(columns=[c for c in columnas_eliminar if c in df.columns], errors="ignore")
    print(f"🧹 Columnas eliminadas antes de guardar: {', '.join(columnas_eliminar)}")

    # Crear carpeta si no existe
    if not os.path.exists(path_salida):
        os.makedirs(path_salida, exist_ok=True)
        print(f"📁 Carpeta creada: {path_salida}")

    ruta_final = os.path.join(path_salida, nombre_archivo)

    try:
        df.to_csv(ruta_final, index=False)
        print(f"✅ Archivo guardado correctamente en:\n   {ruta_final}")
        print(f"📊 Tamaño final: {len(df):,} filas × {len(df.columns)} columnas")
    except Exception as e:
        print(f"❌ Error al guardar el archivo: {e}")

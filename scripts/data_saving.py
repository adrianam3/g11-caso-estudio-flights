
import os
import pandas as pd

# =========================
# FUNCIÓN PRINCIPAL
# =========================

def guardar_datos_limpios(df: pd.DataFrame, path_salida: str, nombre_archivo: str = "flights_clean.csv"):
    """
    Guarda el DataFrame procesado en formato CSV dentro de la carpeta 'processed'.
    
    Parámetros:
    -----------
    df : pd.DataFrame
        DataFrame a guardar.
    path_salida : str
        Ruta completa o carpeta donde guardar el archivo.
    nombre_archivo : str
        Nombre del archivo CSV final (por defecto 'flights_clean.csv').
    """
    print("\n💾 Iniciando guardado del archivo procesado...")

    # Crear carpeta si no existe
    if not os.path.exists(path_salida):
        os.makedirs(path_salida, exist_ok=True)
        print(f"📁 Carpeta creada: {path_salida}")

    # Ruta completa del archivo
    ruta_final = os.path.join(path_salida, nombre_archivo)

    try:
        df.to_csv(ruta_final, index=False)
        print(f"✅ Archivo guardado correctamente en:\n   {ruta_final}")
        print(f"📊 Tamaño final: {len(df):,} filas × {len(df.columns)} columnas")
    except Exception as e:
        print(f"❌ Error al guardar el archivo: {e}")

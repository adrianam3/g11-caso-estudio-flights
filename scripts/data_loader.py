# import pandas as pd

# # creacion de funcion
# def cargar_datos(path):
#     print(f"Cargando datos desde {path}...")
    
#     try:
#         games = pd.read_csv(path)
#         print("Datos han sido cargados!!!")correctamente
#         return games
#     except FileNotFoundError:
#         print(f"Error: no se encontró el archivo en {path}")
#         print("Asegurate de tener el archivo en la carpeta 'data'.")
#         return None
#     except Exception as e:
#         print(f"Ocurrió un error inesperado {e}")
#         return None
    


import pandas as pd

def cargar_datos(path, nrows=None):
    """
    Carga un archivo CSV en un DataFrame.
    
    Parámetros:
        path (str): ruta del archivo CSV.
        nrows (int, opcional): número de filas a leer (útil para pruebas).
    
    Retorna:
        DataFrame o None si hay error.
    """
    print(f"\n📂 Cargando datos desde: {path}")
    
    try:
        vuelos = pd.read_csv(path, low_memory=False, nrows=nrows)
        print(f"✅ Datos cargados correctamente ({len(vuelos):,} filas, {len(vuelos.columns)} columnas).")
        return vuelos
    except FileNotFoundError:
        print(f"❌ Error: no se encontró el archivo en {path}")
        print("Verifica que el archivo esté en la carpeta 'data/'.")
        return None
    except Exception as e:
        print(f"⚠️ Error inesperado: {e}")
        return None

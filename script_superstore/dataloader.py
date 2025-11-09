
import pandas as pd
import os

# ruta absoluta de la carpeta donde esta el script (.../scripts/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ruta absoluta del archivo data.csv (.../data/data.csv)
EXCEL_PATH = os.path.join(SCRIPT_DIR, "..", "data", 'Sample - Superstore.xls')     

def cargar_datos_excel(path, lista_de_hojas):
    print(f"Cargando datos desde: {path}..")
    
    data_frame_cargados = {}  # Diccionario para almacenar los DataFrames cargados
    
    try:
        # Cargar cada hoja del archivo Excel en un DataFrame
        for hoja in lista_de_hojas:
            print(f"Cargando hoja: {hoja}..")
            
            df_temporal = pd.read_excel(path, sheet_name=hoja)
            data_frame_cargados[hoja] = df_temporal
    
        print("Datos cargados correctamente !!!.")
        return data_frame_cargados      #flights= dataframe
    except FileNotFoundError:
        print(f"Error: El archivo no se encontró en la ruta especificada: {path}")
        print("Verifique que el archivo exista y la ruta sea correcta carpeta 'data'.")
        return None
    except Exception as e:
        print(f"Ocurrió un error inseperado: {e}")
        return None
    

 # ¿este archivo se está ejcutando directamente por el usuario o esta siendo importado por otro script ?    
if __name__ == "__main__":
    # indica donde esta el script actual
    # print(f"El script se está ejecutando desde: {os.path.abspath(__file__)}")
    print(f"El script se está ejecutando desde: {EXCEL_PATH}")    
    #llamar a la función de arriba para cargar el csv cargar_datos
    hojas = ["Orders", "People", "Returns"]
    
    diccionario_dataframes = cargar_datos_excel(EXCEL_PATH, hojas)
    
    # dataframe_vuelos = cargar_datos(DATA_PATH)
  
    if diccionario_dataframes is not None:
        
        for nombre_hoja, df in diccionario_dataframes.items():
            print(f"\n---Primeras 5 filas de la hoja {nombre_hoja}--")
            print(df.head())
            print(f"\n---Información del dataframe de la hoja {nombre_hoja}--")
            df.info(show_counts=True)
        
 
        
         
# %%

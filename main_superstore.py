
import pandas as pd
from script_superstore.dataloader 
import os

# ruta absoluta de la carpeta donde esta el script (.../scripts/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ruta absoluta del archivo data.csv (.../data/data.csv)
EXCEL_PATH = os.path.join(SCRIPT_DIR, ".", "data", 'Sample - Superstore.xls')     
   

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
        
 
        
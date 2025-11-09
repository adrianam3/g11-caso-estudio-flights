import pandas as pd
import numpy as np
from script_superstore.data_loader import cargar_datos_excel
 
 
def union_dataframes(dataframes):
   
    try:
        df_pedidos = dataframes["Orders"]
        df_devoluciones = dataframes["Returns"]
        df_personas = dataframes["People"]
       
        print("Se está uniendo archivo Pedidos con Devoluciones...")
        df_maestro = pd.merge(
            left = df_pedidos,
            right = df_devoluciones,
            on = "Order ID",
            how = "left"
        )
       
        df_maestro["Returned"] = df_maestro["Returned"].fillna("No")
       
        print("Listo. Ahora se está uniendo archivo Maestro con Personas...")
       
        df_maestro = pd.merge(
            left = df_maestro,
            right = df_personas,
            on = "Region",
            how = "left"
        )
       
        print("Unión lista!!!")
       
        return df_maestro
   
    except Exception as e:
        print(f"Ha ocurrido un error: {e}")
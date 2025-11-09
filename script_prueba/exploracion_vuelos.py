# %% [markdown]
# # Predicción de Retrasos de Vuelos en la Industria Aérea (Estados Unidos)

# %% [markdown]
# ## Información del caso

# %% [markdown]
# ### **TEMA DEL CASO**

# %% [markdown]
# Predicción de retrasos en vuelos comerciales y análisis de factores asociados a la puntualidad aérea.

# %% [markdown]
# ### **ANTECEDENTES**

# %% [markdown]
# El dataset flights.csv contiene un histórico de más de 5 millones de registros de vuelos domésticos en Estados Unidos (año 2015), con detalles sobre aerolínea, aeropuerto de origen y destino, horarios programados y reales, y demoras registradas.
# 
# La Administración Federal de Aviación (FAA) y el Bureau of Transportation Statistics (BTS) recopilan esta información para monitorear el rendimiento operativo de las aerolíneas y la congestión en los aeropuertos.
# 
# El objetivo operativo es anticipar los retrasos significativos (>15 minutos) que generan pérdidas económicas, descontento en los pasajeros y desorganización logística.
# El reto consiste en limpiar y analizar un dataset grande y heterogéneo que contiene valores nulos, datos anómalos, codificaciones de aeropuertos no válidas y variables redundantes, para construir un modelo predictivo útil para las aerolíneas y operadores aeroportuarios.

# %% [markdown]
# ### **OBJETIVO**

# %% [markdown]
# Construir un sistema de analítica completo que permita:
# 
# 1. Implementar un pipeline de limpieza y transformación de datos robusto, capaz de procesar millones de registros de vuelos y detectar datos inválidos o inconsistentes.
# 
# 2. Realizar un análisis exploratorio (EDA) que identifique las aerolíneas, rutas y aeropuertos con mayores tasas de retraso.
# 
# 3. Entrenar un modelo de Machine Learning (clasificación) (por ejemplo, LightGBM o RandomForestClassifier) que prediga la probabilidad de que un vuelo llegue con un retraso mayor a 15 minutos.
# 
# 4. Desplegar el modelo y los resultados mediante una API de predicción y un dashboard analítico, que permita a los usuarios consultar retrasos esperados por ruta, aerolínea u horario.

# %% [markdown]
# ### **ACTIVIDADES**

# %% [markdown]
# **Pipeline**
# 
# 1. Descargar y cargar el dataset flights.csv junto con los catálogos airlines.csv y airports.csv.
# 
# 2. Implementar un proceso de limpieza que:
# 
#     2.1 Elimine registros de vuelos cancelados o desviados.
# 
#     2.2 Valide y normalice los códigos de aeropuertos (ORIGIN_AIRPORT, DESTINATION_AIRPORT) y aerolíneas (AIRLINE).
# 
#     2.3 Maneje valores faltantes o anómalos en las columnas de tiempo (SCHEDULED_DEPARTURE, ARRIVAL_DELAY).
# 
# 3. Realizar la ingeniería de características (feature engineering):
# 
#     3.1 Extraer variables como hora de salida, día de la semana, mes, distancia, ruta origen-destino.
# 
#     3.2 Crear la variable objetivo binaria DELAYED (1 si ARRIVAL_DELAY > 15, 0 en caso contrario).

# %% [markdown]
# ### **ANALISIS**

# %% [markdown]
# Librerias

# %%
# pandas para manejar df (tablas)
import pandas as pd
# missingno para visualizar datos faltantes en df
import missingno as msno
# numpy para manejar datos específicos con pandas (procesamiento computacional más rápido)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# %%
# import sklearn
# print("Scikit-learn version:", sklearn.__version__)

# %% [markdown]
# Cargar datos

# %%
# Definir la ruta base
ruta_excel = r"D:/OneDrive/DOCUMENTOS/Personales/2024/uniandes/8 S/seminario/g11-caso-estudio-complexivo/data/"


# %%
# Leer los archivos
vuelos = pd.read_csv(ruta_excel + "flights.csv")
aeropuertos = pd.read_csv(ruta_excel + "airports.csv")
aerolineas = pd.read_csv(ruta_excel + "airlines.csv")

# %%
# Mostrar las dimensiones
print(f" Vuelos: {vuelos.shape}")
print(f" Aeropuertos: {aeropuertos.shape}")
print(f" Aerolineas: {aerolineas.shape}")

# %%
# Vista preliminar
vuelos.head()


# %%
aeropuertos.head(100)

# %%
aerolineas.head(100)

# %%
# --- Vista general ---
vuelos.head()


# %% [markdown]
# Analisis inicial de los datos

# %%
vuelos.head(100)

# %%
vuelos.info()

# %% [markdown]
# Valores únicos (diagnóstico rápido)

# %%
print("Información general vuelos:")
vuelos.info()

print("\nNulos (top 20):")
vuelos.isna().sum().sort_values(ascending=False).head(20)


# %%
type(vuelos)

# %%
vuelos.columns

# %% [markdown]
# # Información general

# %%
print("Información general del dataset de vuelos:")
print("="*80)
vuelos.info()

# %%
print("\nValores nulos por columna:")
print(vuelos.isnull().sum().sort_values(ascending=False).head(100))

# %%
# Estadísticas descriptivas
vuelos.describe().T.head(100)

# %%
print("Número de valores únicos por columna en vuelos:")
print(vuelos.nunique().sort_values(ascending=False))


# %%
# vuelos = vuelos.rename(
#     columns={
#         "YEAR":"anio",
#         "MONTH":"mes",
#         "DAY":"dia",
#         "DAY_OF_WEEK":"dia_de_semana",
#         "AIRLINE":"aerolinea",
#         "FLIGHT_NUMBER":"numero_de_vuelo",
#         "TAIL_NUMBER":"numero_matricula_avion",
#         "ORIGIN_AIRPORT":"aeropuerto_de_origen",
#         "DESTINATION_AIRPORT":"aeropuerto_de_destino",
#         "SCHEDULED_DEPARTURE":"hora_programada_salida",
#         "DEPARTURE_TIME":"hora_de_salida",
#         "DEPARTURE_DELAY":"retraso_en_salida",
#         "TAXI_OUT":"rodaje_de_salida",
#         "WHEELS_OFF":"hora_de_despegue",
#         "SCHEDULED_TIME":"tiempo_programado_de_vuelo",
#         "ELAPSED_TIME":"tiempo_trascurrido",
#         "AIR_TIME":"tiempo_en_el_aire",
#         "DISTANCE":"distacia",
#         "WHEELS_ON":"hora_de_aterrizaje",
#         "TAXI_IN":"rodaje_en_llegada",
#         "SCHEDULED_ARRIVAL":"hora_programada_llegada",
#         "ARRIVAL_TIME":"hora_de_llegada",
#         "ARRIVAL_DELAY":"retraso_en_llegada",
#         "DIVERTED":"desviado",
#         "CANCELLED":"cancelado",
#         "CANCELLATION_REASON":"motivo_de_cancelacion",
#         "AIR_SYSTEM_DELAY":"retraso_por_sistema_aereo",
#         "SECURITY_DELAY":"retraso_por_seguridad",
#         "AIRLINE_DELAY":"retraso_por_aerolinea",
#         "LATE_AIRCRAFT_DELAY":"retraso_por_llegada_tardia_aeronave",
#         "WEATHER_DELAY":"retraso_por_meteorologicas",

#     }
# )

# %%
vuelos.head()

# %%
vuelos.nunique()

# %%
vuelos.ORIGIN_AIRPORT.unique()

# %%
vuelos[vuelos['ORIGIN_AIRPORT'] == '13933']


# %%
vuelos[vuelos['AIRLINE'] == 'B6']


# %% [markdown]
# revisar dataset airports

# %%
aeropuertos.head(100)

# %%
aeropuertos.info()

# %%
aeropuertos

# %% [markdown]
# revisar valores nulos en dtaframe aeropuertos

# %%
aeropuertos.isna().sum()

# %%
# porcentaje de valores ausentes de todas las columnas
round(aeropuertos.isnull().mean()*100, 3)



# %%
# contar datos con los nans
aeropuertos["LATITUDE"].value_counts(dropna=False)
# v["ORIGEN_LAT"].value_counts(dropna=True)

# %%
aeropuertos["LONGITUDE"].value_counts(dropna=False)

# %%
aeropuertos[aeropuertos["LATITUDE"].isna()]


# %%

aeropuertos[aeropuertos["LONGITUDE"].isna()]

# %% [markdown]
#  Cantidad de registros de los aeropuertos en flights.csv que no tienen coordenadas en airports.csv

# %%
# Lista de códigos a analizar
iata_codes = ["ECP", "PBG", "UST"]

conteo_origen = vuelos["ORIGIN_AIRPORT"].isin(iata_codes).sum()
conteo_destino = vuelos["DESTINATION_AIRPORT"].isin(iata_codes).sum()

print(f"Vuelos con ORIGIN en {iata_codes}: {conteo_origen}")
print(f"Vuelos con DESTINATION en {iata_codes}: {conteo_destino}")


# %% [markdown]
# Estadisticas de registros con aeropuetos con iata_codes = ["ECP", "PBG", "UST"]

# %%
# Lista de aeropuertos de interés
iata_codes = ["ECP", "PBG", "UST"]

# Total de registros en flights.csv
total_vuelos = len(vuelos)

# Conteos individuales
conteo_origen = vuelos["ORIGIN_AIRPORT"].isin(iata_codes).sum()
conteo_destino = vuelos["DESTINATION_AIRPORT"].isin(iata_codes).sum()

# Calcular porcentajes
porc_origen = (conteo_origen / total_vuelos) * 100
porc_destino = (conteo_destino / total_vuelos) * 100

# Mostrar resultados
print(f"Total de registros en flights.csv: {total_vuelos:,}")
print(f"Vuelos con ORIGIN en {iata_codes}: {conteo_origen:,} ({porc_origen:.4f}%)")
print(f"Vuelos con DESTINATION en {iata_codes}: {conteo_destino:,} ({porc_destino:.4f}%)")

# Si quieres el total combinado (sin preocuparte por posibles duplicados)
conteo_total = vuelos["ORIGIN_AIRPORT"].isin(iata_codes).sum() + vuelos["DESTINATION_AIRPORT"].isin(iata_codes).sum()
porc_total = (conteo_total / (2 * total_vuelos)) * 100  # se duplica el total para que no se sume doble
print(f"Participación combinada (ORIGIN + DESTINATION): {conteo_total:,} ({porc_total:.4f}%)")


# %%
vuelos.query("ORIGIN_AIRPORT in @iata_codes or DESTINATION_AIRPORT in @iata_codes") \
      [["ORIGIN_AIRPORT", "DESTINATION_AIRPORT"]] \
      .value_counts()


# %% [markdown]
# rellenar coordenadas faltantes en dataset aeropuertos

# %%
# Diccionario con las coordenadas faltantes
coords_faltantes = {
    "ECP": {"LATITUDE": 30.357106, "LONGITUDE": -85.795414},
    "PBG": {"LATITUDE": 44.6509, "LONGITUDE": -73.4681},
    "UST": {"LATITUDE": 29.9592, "LONGITUDE": -81.3398},
}


# %% [markdown]
# Actualizar el DataFrame directamente (sin modificar el CSV)

# %%
# Diccionario con las coordenadas faltantes
coords_faltantes = {
    "ECP": {"LATITUDE": 30.357106, "LONGITUDE": -85.795414},
    "PBG": {"LATITUDE": 44.6509, "LONGITUDE": -73.4681},
    "UST": {"LATITUDE": 29.9592, "LONGITUDE": -81.3398},
}

# Contador para registrar cuántos fueron actualizados
registros_actualizados = 0

# Actualizar el DataFrame con los valores del diccionario
for codigo, valores in coords_faltantes.items():
    mask = aeropuertos["IATA_CODE"] == codigo
    if mask.any():
        aeropuertos.loc[mask, ["LATITUDE", "LONGITUDE"]] = valores["LATITUDE"], valores["LONGITUDE"]
        registros_actualizados += 1

# Mostrar resumen
print(f"Se actualizaron {registros_actualizados} registros en el DataFrame 'aeropuertos'.")
print("Registros actualizados:")

# Mostrar solo los registros actualizados con coordenadas nuevas
registros = aeropuertos.loc[aeropuertos["IATA_CODE"].isin(coords_faltantes.keys()),
                            ["IATA_CODE", "AIRPORT", "LATITUDE", "LONGITUDE"]]

print(registros.to_string(index=False))



# %%
# v.columns

# %% [markdown]
# ### **Validación de integridad entre datasets**
# Comprobamos qué porcentaje de registros de vuelos no tienen correspondencia en los catálogos.

# %%
# Normalizar texto (evita errores por espacios o minúsculas)
for df, col in [(vuelos, "AIRLINE"), (vuelos, "ORIGIN_AIRPORT"), (vuelos, "DESTINATION_AIRPORT"),
                (aerolineas, "IATA_CODE"), (aeropuertos, "IATA_CODE")]:
    df[col] = df[col].astype(str).str.strip().str.upper()

# %%
# Crear conjuntos de referencia
set_aerolineas = set(aerolineas["IATA_CODE"])
set_aeropuertos = set(aeropuertos["IATA_CODE"])

# Crear máscaras de valores inválidos
mask_aerolinea_invalida = ~vuelos["AIRLINE"].isin(set_aerolineas)
mask_origen_invalido = ~vuelos["ORIGIN_AIRPORT"].isin(set_aeropuertos)
mask_destino_invalido = ~vuelos["DESTINATION_AIRPORT"].isin(set_aeropuertos)

# Calcular porcentajes
total = len(vuelos)
porc_aerolinea = mask_aerolinea_invalida.mean() * 100
porc_origen = mask_origen_invalido.mean() * 100
porc_destino = mask_destino_invalido.mean() * 100

print("Porcentaje de registros con códigos inválidos:")
print(f"Aerolíneas no válidas: {porc_aerolinea:.3f}%")
print(f"Aeropuertos de ORIGEN no válidos: {porc_origen:.3f}%")
print(f"Aeropuertos de DESTINO no válidos: {porc_destino:.3f}%")

# Porcentaje total de registros inválidos
mask_total_invalido = mask_aerolinea_invalida | mask_origen_invalido | mask_destino_invalido
porc_total_invalido = mask_total_invalido.mean() * 100

print(f"Registros totales con datos inválidos (aerolínea o aeropuerto): {porc_total_invalido:.3f}%")


# %%
set_aerolineas

# %%
set_aeropuertos

# %%
mask_aerolinea_invalida


# %%
mask_total_invalido

# %% [markdown]
# Revisión de las columnas CANCELLED y DIVERTED

# %%
vuelos.CANCELLED.unique()


# %%
vuelos.DIVERTED.unique()

# %%
print(vuelos["CANCELLED"].value_counts())
print(vuelos["DIVERTED"].value_counts())

# %%
vuelos_cancelados = vuelos[vuelos["CANCELLED"] == 0]
display(vuelos_cancelados.head(10))  # muestra las 10 primeras filas


# %%
total_cancelados = (vuelos["CANCELLED"] == 1).sum()
total_total = len(vuelos)
porcentaje = total_cancelados / total_total * 100

print(f" Vuelos cancelados: {total_cancelados:,} ({porcentaje:.2f}%) de {total_total:,} vuelos")



# %%
# Calcular totales
conteo_cancelados = vuelos["CANCELLED"].value_counts().sort_index()
etiquetas = ["No cancelado", "Cancelado"]
valores = conteo_cancelados.values
porcentajes = valores / valores.sum() * 100

# Crear gráfico
plt.figure(figsize=(6,4))
barras = plt.bar(etiquetas, valores, color=["skyblue", "salmon"])

# Añadir texto con totales y porcentaje encima de cada barra
for i, (v, p) in enumerate(zip(valores, porcentajes)):
    plt.text(i, v + (valores.max() * 0.3), f"{v:,}\n({p:.2f}%)",
             ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.title("Distribución total de vuelos cancelados")
plt.ylabel("Cantidad de vuelos")
plt.tight_layout()
plt.show()


# %%
# Datos
etiquetas = ["No cancelado", "Cancelado"]
valores = conteo_cancelados.values
porcentajes = valores / valores.sum() * 100

# Etiquetas combinadas: total + porcentaje
etiquetas_con_valor = [f"{etiquetas[i]}\n{valores[i]:,} ({porcentajes[i]:.2f}%)" for i in range(len(etiquetas))]

plt.figure(figsize=(6,6))
plt.pie(
    valores,
    labels=etiquetas_con_valor,
    startangle=90,
    colors=["lightgreen", "lightcoral"],
    wedgeprops={'edgecolor': 'white'},
    textprops={'fontsize': 11}
)
plt.title("Porcentaje y cantidad total de vuelos cancelados vs no cancelados")
plt.show()



# %%
# Calcular porcentaje de vuelos cancelados por aerolínea
# Cada barra representa el porcentaje de vuelos cancelados por aerolínea.
# Se usa .mean() porque el promedio de 0/1 equivale al porcentaje.

porc_cancelados = vuelos.groupby("AIRLINE")["CANCELLED"].mean().sort_values(ascending=False) * 100

# Crear gráfico de barras
plt.figure(figsize=(10,5))
plt.bar(porc_cancelados.index, porc_cancelados.values)
plt.title("Porcentaje de vuelos cancelados por aerolínea")
plt.xlabel("Aerolínea")
plt.ylabel("Porcentaje (%) de cancelaciones")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# %%
cancelaciones = vuelos.groupby("AIRLINE")["CANCELLED"].agg(["sum", "count"])
cancelaciones["porcentaje"] = cancelaciones["sum"] / cancelaciones["count"] * 100

cancelaciones = cancelaciones.sort_values("porcentaje", ascending=False)

plt.figure(figsize=(10,5))
plt.bar(cancelaciones.index, cancelaciones["porcentaje"], color="orange")
plt.title("Tasa de cancelaciones por aerolínea")
plt.ylabel("Porcentaje (%)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# %%
total_cancelados = (vuelos["CANCELLED"] == 1).sum()
total_total = len(vuelos)
porcentaje = total_cancelados / total_total * 100

print(f"Total de vuelos: {total_total:,}")
print(f"Vuelos cancelados: {total_cancelados:,} ({porcentaje:.2f}%)")
print(f"Vuelos no cancelados: {total_total - total_cancelados:,}")

# %% [markdown]
# vuelos desviados

# %%
# ========================
# Análisis de vuelos desviados
# ========================

# Calcular totales
conteo_desviados = vuelos["DIVERTED"].value_counts().sort_index()
etiquetas = ["No desviado", "Desviado"]
valores = conteo_desviados.values
porcentajes = valores / valores.sum() * 100

# Mostrar resumen en consola
total_desviados = (vuelos["DIVERTED"] == 1).sum()
total_total = len(vuelos)
porcentaje = total_desviados / total_total * 100

print(f"Total de vuelos: {total_total:,}")
print(f"Vuelos desviados: {total_desviados:,} ({porcentaje:.3f}%)")
print(f"Vuelos no desviados: {total_total - total_desviados:,}")



# %%
# ========================
# Gráfico de Barras (totales + porcentajes)
# ========================

plt.figure(figsize=(6,4))
barras = plt.bar(etiquetas, valores, color=["lightblue", "orange"])

# Mostrar totales y porcentajes encima de cada barra
for i, (v, p) in enumerate(zip(valores, porcentajes)):
    plt.text(i, v + (valores.max() * 0.2), f"{v:,}\n({p:.3f}%)",
             ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.title("Distribución total de vuelos desviados")
plt.ylabel("Cantidad de vuelos")
plt.tight_layout()
plt.show()

# %%


# ========================
# Gráfico de Pastel (porcentaje + totales)
# ========================

etiquetas_con_valor = [f"{etiquetas[i]}\n{valores[i]:,} ({porcentajes[i]:.3f}%)" for i in range(len(etiquetas))]

plt.figure(figsize=(6,6))
plt.pie(
    valores,
    labels=etiquetas_con_valor,
    startangle=90,
    colors=["lightgreen", "lightcoral"],
    wedgeprops={'edgecolor': 'white'},
    textprops={'fontsize': 11}
)
plt.title("Porcentaje y cantidad total de vuelos desviados vs no desviados")
plt.show()


# %% [markdown]
# ### **Limpieza de registros inválidos y cancelados**
# 
# Mantiene los vuelos NO cancelados (CANCELLED = 0)
# 
# Mantiene los vuelos NO desviados (DIVERTED = 0)
# 
# Excluye los registros con aeropuertos o aerolíneas inválidas  (~mask_aerolinea_invalida) & (~mask_origen_invalido) & (~mask_destino_invalido)
# 
# | Caso                       | CANCELLED | DIVERTED | ¿Se mantiene? | Motivo                |
# | -------------------------- | --------- | -------- | ------------- | --------------------- |
# | Vuelo normal               | 0         | 0        | Sí          | vuelo válido          |
# | Vuelo cancelado            | 1         | 0        | No          | cancelado             |
# | Vuelo desviado             | 0         | 1        | No          | no llegó al destino   |
# | Vuelo cancelado y desviado | 1         | 1        | No          | ambos casos inválidos |
# 

# %%
vuelos_limpios = vuelos[
    (vuelos["CANCELLED"] == 0) &
    (vuelos["DIVERTED"] == 0) &
    (~mask_aerolinea_invalida) &
    (~mask_origen_invalido) &
    (~mask_destino_invalido)
].copy()

print(f"Registros válidos: {len(vuelos_limpios):,} ({len(vuelos_limpios)/len(vuelos)*100:.2f}% del total)")


# %%
vuelos.columns

# %%
# v.columns

# %%
vuelos_limpios

# %% [markdown]
# ### **Merge de catálogos (aerolíneas y aeropuertos)**

# %%
# Aerolíneas: mapear AIRLINE -> nombre
aerolineas_ren = aerolineas.rename(columns={"IATA_CODE":"AIRLINE", "AIRLINE":"AIRLINE_NAME"})


# %%
aerolineas_ren.info()

# %%
# v = vuelos_limpios.merge(aerolineas_ren[["AIRLINE","NOMBRE_AEROLINEA"]], on="AIRLINE", how="left")

v = pd.merge(
    left = vuelos_limpios, 
    right = aerolineas_ren, 
    on = "AIRLINE", 
    how = "left"
)
v.info()

# %%
v.head()


# %%
# Aeropuertos ORIGEN
aerop_origen = aeropuertos.rename(columns={
    "IATA_CODE":"ORIGIN_AIRPORT",
    "AIRPORT":"ORIGEN_AEROPUERTO",
    "CITY":"ORIGEN_CIUDAD",
    "STATE":"ORIGEN_ESTADO",
    "COUNTRY":"ORIGEN_PAIS",
    "LATITUDE":"ORIGEN_LAT",
    "LONGITUDE":"ORIGEN_LON"
})


# %%
# v = v.merge(aerop_origen, on="ORIGIN_AIRPORT", how="left")

v = pd.merge(
    left = v, 
    right = aerop_origen, 
    on = "ORIGIN_AIRPORT", 
    how = "left"
)
v.info()

# %%
v.head()

# %%
# Aeropuertos DESTINO
aerop_dest = aeropuertos.rename(columns={
    "IATA_CODE":"DESTINATION_AIRPORT",
    "AIRPORT":"DEST_AEROPUERTO",
    "CITY":"DEST_CIUDAD",
    "STATE":"DEST_ESTADO",
    "COUNTRY":"DEST_PAIS",
    "LATITUDE":"DEST_LAT",
    "LONGITUDE":"DEST_LON"
})


# %%
# v = v.merge(aerop_dest, on="DESTINATION_AIRPORT", how="left")
v = pd.merge(
    left = v, 
    right = aerop_dest, 
    on = "DESTINATION_AIRPORT", 
    how = "left"
)
v.info()

# %%
v.head()

# %%
print("Post-merge:", v.shape)
v.head()

# %% [markdown]
# ### **Valores ausentes y únicos (post-merge)**

# %%
valores_ausentes=print(v.isnull().sum())

# %%
# Resumen de nulos (top)
nulos = v.isna().sum()#.sort_values(ascending=False)
display(nulos.head(100))

# %%
# % nulos por columna (útil para decidir imputación o descarte)
pct_nulos = (v.isna().mean()*100).sort_values(ascending=False)
display(pct_nulos.head(200))


# %%
# # Unicidad de columnas clave post-merge
# unicos_clave = df_vuelos_completo[["AIRLINE","NOMBRE_AEROLINEA","ORIGIN_AIRPORT","ORIGEN_AEROPUERTO",
#                   "DESTINATION_AIRPORT","DEST_AEROPUERTO"]].nunique().sort_values(ascending=False)
# display(unicos_clave)

#  Unicidad de columnas clave, filtrando las que existan
cols_clave = ["AIRLINE","NOMBRE_AEROLINEA",
              "ORIGIN_AIRPORT","ORIGEN_AEROPUERTO",
              "DESTINATION_AIRPORT","DEST_AEROPUERTO"]
cols_clave = [c for c in cols_clave if c in v.columns]  # <-- filtro evita KeyError
unicos_clave = v[cols_clave].nunique().sort_values(ascending=False)
display(unicos_clave)




# %%
v.shape

# %%
v.head()

# %%
v.info()

# %% [markdown]
# Una ves que se realizó limpieza y merge de los registros del datraframe, se alamceno en v, tengo estos campos que aun tienen las siguientes columnas con valores nulos: 
# 
# AIR_SYSTEM_DELAY 4227770 
# 
# SECURITY_DELAY 4227770 
# 
# AIRLINE_DELAY 4227770 
# 
# LATE_AIRCRAFT_DELAY 4227770 
# 
# WEATHER_DELAY 
# 
# sin embargo de un analisis previo que se realizo en excel, se pudo notar que la suma de estas columnas, da el valor del tiempo de retraso que esta en ARRIVAL_DELAY
# 
# Por loque se procede con lo siguiente: 
# 
# 1. Verificar si tienen al menos una causa con valor
# 
# 2. Generar un script para verificar y confirmar si la suma de estas columnas, coincide con el valor de ARRIVAL_DELAY 
# 
# 3. Generar una nueva columna basada en las columnas AIR_SYSTEM_DELAY SECURITY_DELAY AIRLINE_DELAY LATE_AIRCRAFT_DELAY WEATHER_DELAY, que ubique el motivo del retraso, el motivo de retraso 
# 

# %% [markdown]
# 1. Verificar si tienen al menos una causa con valor

# %%
# columnas de causas
causas_cols = [
    "AIR_SYSTEM_DELAY",
    "SECURITY_DELAY",
    "AIRLINE_DELAY",
    "LATE_AIRCRAFT_DELAY",
    "WEATHER_DELAY"
]

# considerar solo vuelos con retraso positivo
mask_pos = v["ARRIVAL_DELAY"] > 0

# filas con al menos una causa NO nula
tienen_alguna = v.loc[mask_pos, causas_cols].notna().any(axis=1)

# conteos
total_retrasados = mask_pos.sum()
con_alguna = tienen_alguna.sum()
sin_ninguna = total_retrasados - con_alguna

# porcentajes
porc_con_alguna = 100 * con_alguna / total_retrasados
porc_sin_ninguna = 100 * sin_ninguna / total_retrasados

print("📊 Análisis de causas registradas (solo ARRIVAL_DELAY>0):")
print(f"  Total de vuelos con retraso (>0 min): {total_retrasados:,}")
print(f"  Con al menos una causa registrada   : {con_alguna:,} ({porc_con_alguna:.2f}%)")
print(f"  Sin ninguna causa (todas NaN)       : {sin_ninguna:,} ({porc_sin_ninguna:.2f}%)")

# (opcional) ejemplo de filas con al menos una causa válida
v.loc[mask_pos & tienen_alguna, ["ARRIVAL_DELAY"] + causas_cols].head(10)


# %% [markdown]
# 2) Verificar que la suma de causas coincide con ARRIVAL_DELAY

# %%
# import numpy as np
# import pandas as pd

# --- columnas de causas (BTS) ---
causas_cols = [
    "AIR_SYSTEM_DELAY",       # NAS / Sistema aéreo
    "SECURITY_DELAY",         # Seguridad
    "AIRLINE_DELAY",          # Aerolínea
    "LATE_AIRCRAFT_DELAY",    # Aeronave llegada tardía
    "WEATHER_DELAY"           # Clima
]

# Asegurar numérico (por si vienen como object)
for c in causas_cols + ["ARRIVAL_DELAY"]:
    v[c] = pd.to_numeric(v[c], errors="coerce")

# Suma de causas (tratando NaN como 0 para la suma)
v["SUMA_CAUSAS"] = v[causas_cols].fillna(0).sum(axis=1)

# Comparamos SOLO cuando ARRIVAL_DELAY > 0 (BTS solo reporta causas si hubo retraso positivo)
mask_pos = v["ARRIVAL_DELAY"] > 0

# Diferencia absoluta
v["DIF_CAUSAS_ARR"] = (v["SUMA_CAUSAS"] - v["ARRIVAL_DELAY"]).abs()

# Métricas de consistencia
total_pos = mask_pos.sum()
coinciden = ((v.loc[mask_pos, "DIF_CAUSAS_ARR"] <= 1e-6) | (v.loc[mask_pos, "DIF_CAUSAS_ARR"] < 1)).sum()  # tolerancia 1 min
porc_coinciden = 100 * coinciden / total_pos if total_pos else np.nan
max_dif = v.loc[mask_pos, "DIF_CAUSAS_ARR"].max()
mean_dif = v.loc[mask_pos, "DIF_CAUSAS_ARR"].mean()

print("✔ Verificación suma de causas vs ARRIVAL_DELAY (solo ARRIVAL_DELAY>0):")
print(f"  Filas con ARRIVAL_DELAY>0           : {total_pos:,}")
print(f"  Coinciden (tolerancia ±1 min)       : {coinciden:,} ({porc_coinciden:.2f}%)")
print(f"  Diferencia absoluta media (min)     : {mean_dif:.3f}")
print(f"  Diferencia absoluta máxima (min)    : {max_dif:.3f}")

# (Opcional) Muestra algunas filas donde no coincide, para inspección
mismatch = v.loc[mask_pos & (v["DIF_CAUSAS_ARR"] > 1), 
                 ["ARRIVAL_DELAY", "SUMA_CAUSAS"] + causas_cols].head(10)
mismatch
if len(mismatch):
    print("\n  Ejemplos de no coincidencia:")
    display(mismatch)


# %%
# --- Comparación mejor explicada ---
total_pos = mask_pos.sum()
coinciden = ((v.loc[mask_pos, "DIF_CAUSAS_ARR"] <= 1e-6) | (v.loc[mask_pos, "DIF_CAUSAS_ARR"] < 1)).sum()
no_coinciden = total_pos - coinciden

porc_coinciden = 100 * coinciden / total_pos
porc_no_coinciden = 100 * no_coinciden / total_pos

print("📊 Verificación suma de causas vs ARRIVAL_DELAY (solo ARRIVAL_DELAY>0):")
print(f"  Total de vuelos con ARRIVAL_DELAY>0 : {total_pos:,}")
print(f"  Coinciden (±1 min)                   : {coinciden:,}  ({porc_coinciden:.2f}%)")
print(f"  No coinciden (faltan o NaN causas)   : {no_coinciden:,}  ({porc_no_coinciden:.2f}%)")

# Extra: proporción de filas con todas las causas nulas
sin_causas = v.loc[mask_pos, causas_cols].isna().all(axis=1).sum()
porc_sin_causas = 100 * sin_causas / total_pos
print(f"  Retrasos sin causas registradas      : {sin_causas:,}  ({porc_sin_causas:.2f}%)")


# %% [markdown]
# 3. Generar una nueva columna basada en las columnas AIR_SYSTEM_DELAY SECURITY_DELAY AIRLINE_DELAY LATE_AIRCRAFT_DELAY WEATHER_DELAY, que ubique el motivo del retraso, el motivo de retraso que se almacene en idioma español

# %%
# # --- columnas de causas ---
# causas_cols = [
#     "AIR_SYSTEM_DELAY",
#     "SECURITY_DELAY",
#     "AIRLINE_DELAY",
#     "LATE_AIRCRAFT_DELAY",
#     "WEATHER_DELAY"
# ]

# # --- mapeo de nombres en español ---
# map_es = {
#     "AIR_SYSTEM_DELAY":      "Sistema aéreo (NAS)",
#     "SECURITY_DELAY":        "Seguridad",
#     "AIRLINE_DELAY":         "Aerolínea",
#     "LATE_AIRCRAFT_DELAY":   "Aeronave llegada tardía",
#     "WEATHER_DELAY":         "Clima"
# }

# def motivo_retraso_concat(row):
#     arr = row["ARRIVAL_DELAY"]
    
#     # casos sin retraso
#     if pd.isna(arr):
#         return "Desconocido"
#     if arr <= 0:
#         return "Sin retraso"
    
#     # causas con valor > 0
#     causas_presentes = [map_es[c] for c in causas_cols if pd.notna(row[c]) and row[c] > 0]
    
#     if not causas_presentes:
#         return "Retraso sin causa reportada"
    
#     # concatenar si hay más de una
#     return " - ".join(causas_presentes)

# # --- crear columna ---
# v["MOTIVO_RETRASO"] = v.apply(motivo_retraso_concat, axis=1)

# # --- resumen ---
# conteo_motivos = v["MOTIVO_RETRASO"].value_counts(dropna=False)
# porc_motivos = (conteo_motivos / len(v) * 100).round(2)

# print("📊 Distribución de motivos de retraso (top 10):")
# display(pd.DataFrame({
#     "conteo": conteo_motivos.head(10),
#     "porcentaje": porc_motivos.head(10)
# }))



# %%
# # --- columnas de causas y mapeo (si no las tienes ya definidas) ---
# causas_cols = [
#     "AIR_SYSTEM_DELAY",
#     "SECURITY_DELAY",
#     "AIRLINE_DELAY",
#     "LATE_AIRCRAFT_DELAY",
#     "WEATHER_DELAY"
# ]
# map_es = {
#     "AIR_SYSTEM_DELAY":      "Sistema aéreo (NAS)",
#     "SECURITY_DELAY":        "Seguridad",
#     "AIRLINE_DELAY":         "Aerolínea",
#     "LATE_AIRCRAFT_DELAY":   "Aeronave llegada tardía",
#     "WEATHER_DELAY":         "Clima"
# }

# # --- MOTIVO_RETRASO concatenado (como definimos antes) ---
# def motivo_retraso_concat(row):
#     arr = row["ARRIVAL_DELAY"]
#     if pd.isna(arr):
#         return "Desconocido"
#     if arr <= 0:
#         return "Sin retraso"
#     causas_presentes = [map_es[c] for c in causas_cols if pd.notna(row[c]) and row[c] > 0]
#     if not causas_presentes:
#         return "Retraso sin causa reportada"
#     return " - ".join(causas_presentes)

# v["MOTIVO_RETRASO"] = v.apply(motivo_retraso_concat, axis=1)

# # --- NUEVO: CANTIDAD_CAUSAS (>0) ---
# v["CANTIDAD_CAUSAS"] = (
#     v[causas_cols]
#     .gt(0)          # True si la causa > 0
#     .sum(axis=1)    # contar cuántas
#     .astype("int")
# )

# # --- (opcional) forzar 0 cuando no hay retraso ---
# v.loc[v["ARRIVAL_DELAY"] <= 0, "CANTIDAD_CAUSAS"] = 0

# # --- Resumen rápido ---
# total = len(v)
# con_retraso = (v["ARRIVAL_DELAY"] > 0).sum()
# multi = ((v["ARRIVAL_DELAY"] > 0) & (v["CANTIDAD_CAUSAS"] >= 2)).sum()
# una  = ((v["ARRIVAL_DELAY"] > 0) & (v["CANTIDAD_CAUSAS"] == 1)).sum()
# cero = ((v["ARRIVAL_DELAY"] > 0) & (v["CANTIDAD_CAUSAS"] == 0)).sum()

# print("📊 Resumen de causas (solo ARRIVAL_DELAY>0):")
# print(f"  Vuelos con retraso:          {con_retraso:,}")
# print(f"  0 causas reportadas:         {cero:,}  ({cero/con_retraso*100:.2f}%)")
# print(f"  1 causa reportada:           {una:,}   ({una/con_retraso*100:.2f}%)")
# print(f"  ≥2 causas (multifactorial):  {multi:,} ({multi/con_retraso*100:.2f}%)")

# # --- Tabla de distribución completa 0..5 causas (en % del total con retraso) ---
# dist_causas = (
#     v.loc[v["ARRIVAL_DELAY"] > 0, "CANTIDAD_CAUSAS"]
#      .value_counts().sort_index()
#      .to_frame("conteo")
# )
# dist_causas["porcentaje"] = (dist_causas["conteo"] / con_retraso * 100).round(2)
# dist_causas


# %%
# =========================
# Configuración de causas
# =========================

# --- columnas de causas ---
causas_cols = [
    "AIR_SYSTEM_DELAY",
    "SECURITY_DELAY",
    "AIRLINE_DELAY",
    "LATE_AIRCRAFT_DELAY",
    "WEATHER_DELAY"
]

# --- mapeo de nombres en español ---
map_es = {
    "AIR_SYSTEM_DELAY":      "Sistema aéreo (NAS)",
    "SECURITY_DELAY":        "Seguridad",
    "AIRLINE_DELAY":         "Aerolínea",
    "LATE_AIRCRAFT_DELAY":   "Aeronave llegada tardía",
    "WEATHER_DELAY":         "Clima"
}

# se crea un a lista para manejar el orden fijo deseado para concatenar
prioridad_es = [
    "Aerolínea",
    "Aeronave llegada tardía",
    "Clima",
    "Seguridad",
    "Sistema aéreo (NAS)",  
]

#crea un diccionario interno con la prioridad numérica en base a
prio_idx = {nombre:i for i, nombre in enumerate(prioridad_es)}

# =========================
# MOTIVO_RETRASO (orden fijo, sin magnitud)
# =========================
# import pandas as pd
# import numpy as np

def motivo_retraso_concat_ordenado(row):
    arr = row["ARRIVAL_DELAY"]
    if pd.isna(arr):
        return "Desconocido"
    if arr <= 0:
        return "Sin retraso"

     # causas presentes (valor > 0)
     # Busca, para cada vuelo, qué causas tienen un valor positivo (>0), es decir, que aportaron al retraso.
     # Solo se guardan sus nombres (según el diccionario map_es).
    presentes = []
    for c in causas_cols:
        val = row.get(c)
        if pd.notna(val) and val > 0:
            presentes.append(map_es[c])

    if not presentes:
        return "Retraso sin causa reportada"

    # ordenar SOLO por prioridad fija
    # set(presentes) elimina duplicados.
    # sorted(..., key=lambda nombre: prio_idx.get(nombre, 999)) aplica el orden predefinido.
    # " - ".join(...) concatena las causas con el separador " - ".
    presentes_orden = sorted(set(presentes), key=lambda nombre: prio_idx.get(nombre, 999))
    return " - ".join(presentes_orden)

v["MOTIVO_RETRASO"] = v.apply(motivo_retraso_concat_ordenado, axis=1)

# =========================
# CANTIDAD_CAUSAS (>0)
# .gt(0) → convierte en True las columnas con valor > 0.
# .sum(axis=1) → suma los True de cada fila → total de causas activas.
# Si el vuelo no tiene retraso (ARRIVAL_DELAY <= 0), el valor se fija en 0.
# =========================
v["CANTIDAD_CAUSAS"] = (
    v[causas_cols].gt(0).sum(axis=1).astype(int)
)
v.loc[v["ARRIVAL_DELAY"] <= 0, "CANTIDAD_CAUSAS"] = 0

# =========================
# Resumen TOP 10 de motivos
# =========================
conteo_motivos = v["MOTIVO_RETRASO"].value_counts(dropna=False)
porc_motivos   = (conteo_motivos / len(v) * 100).round(2)

print("Distribución de motivos de retraso (top 10):")
display(pd.DataFrame({
    "conteo": conteo_motivos.head(10),
    "porcentaje": porc_motivos.head(10)
}))

# =========================
# Resumen como la captura (solo ARRIVAL_DELAY>0)
# =========================
mask_pos = v["ARRIVAL_DELAY"] > 0
con_retraso = mask_pos.sum()
cero  = (mask_pos & (v["CANTIDAD_CAUSAS"] == 0)).sum()
una   = (mask_pos & (v["CANTIDAD_CAUSAS"] == 1)).sum()
multi = (mask_pos & (v["CANTIDAD_CAUSAS"] >= 2)).sum()

print("\nResumen de causas (solo ARRIVAL_DELAY>0):")
print(f"  Vuelos con retraso:          {con_retraso:,}")
print(f"  0 causas reportadas:         {cero:,}   ({cero/con_retraso*100:.2f}%)")
print(f"  1 causa reportada:           {una:,}    ({una/con_retraso*100:.2f}%)")
print(f"  ≥2 causas (multifactorial):  {multi:,}  ({multi/con_retraso*100:.2f}%)")

dist_causas = (
    v.loc[mask_pos, "CANTIDAD_CAUSAS"]
     .value_counts().sort_index().to_frame("conteo")
)
dist_causas["porcentaje"] = (dist_causas["conteo"] / con_retraso * 100).round(2)
display(dist_causas)


# %%
v["MOTIVO_RETRASO"].value_counts().head(1000).sort_values(ascending=True)

# %%
v["MOTIVO_RETRASO"].value_counts().sort_index(ascending=True).head(1000)


# %%
motivos = (v.groupby("MOTIVO_RETRASO")
             .size()
             .reset_index(name="count")
             .assign(porcentaje=lambda x: (x["count"] / x["count"].sum() * 100).round(4))
             .sort_values(by="porcentaje", ascending=False))


# %%
motivos.sort_index()

# %% [markdown]
# visualización de motivos principales

# %%
# import matplotlib.pyplot as plt
# import seaborn as sns

top = v["MOTIVO_RETRASO"].value_counts().head(10).sort_values(ascending=True)
plt.figure(figsize=(9,5))
sns.barplot(x=top.values, y=top.index)
plt.title("Principales motivos de retraso")
plt.xlabel("Cantidad de vuelos")
plt.ylabel("Motivo")
plt.show()


# %%
import pandas as pd
import plotly.express as px

# Conteo correcto -> DataFrame con tipos numéricos
motivos = (
    v["MOTIVO_RETRASO"]
      .value_counts(dropna=False)                 # incluye NaN si existieran
      .rename_axis("MOTIVO_RETRASO")
      .reset_index(name="count")
)

# Asegurar tipo numérico (por si vino como string)
motivos["count"] = pd.to_numeric(motivos["count"], errors="coerce").fillna(0).astype("int64")

# Porcentaje
total = motivos["count"].sum()
motivos["porcentaje"] = (motivos["count"] / total * 100).round(2)

# Orden de mayor a menor
motivos = motivos.sort_values("porcentaje", ascending=False)

# excluir “Sin retraso” antes de graficar:
# motivos = motivos[motivos["MOTIVO_RETRASO"] != "Sin retraso"]


# Gráfico
fig = px.bar(
    motivos,
    x="porcentaje",
    y="MOTIVO_RETRASO",
    orientation="h",
    text="porcentaje",
    color="porcentaje",
    color_continuous_scale="Blues",
    title="Distribución de Motivos de Retraso (%)",
    labels={"porcentaje": "Porcentaje (%)", "MOTIVO_RETRASO": "Motivo de retraso"}
)
fig.update_traces(texttemplate="%{text}% ", textposition="outside")
fig.update_layout(yaxis=dict(categoryorder="total ascending"), plot_bgcolor="white", bargap=0.25, height=600, title_x=0.5)

fig.show()



# %% [markdown]
# ### Relación entre ARRIVAL_DELAY ves MOTIVO_RETRASO

# %% [markdown]
# Opción 1 — Boxplot (distribución de retrasos por motivo)
# 
# Ideal para visualizar la dispersión y valores atípicos por tipo de motivo.
# 
# ✈️ Cantidad de vuelos retrasados
# 
# ⏱️ Promedio y mediana de minutos de retraso (ARRIVAL_DELAY)
# 
# 📊 Desviación estándar (variabilidad)
# 
# 📈 Porcentaje del total de retrasos
# 
# Conclusión general
# 
# El gráfico muestra que los retrasos en llegada no solo dependen de una causa individual, sino que factores combinados (por ejemplo, Aerolínea + Aeronave llegada tardía + Clima) producen mayores tiempos de espera.
# Las causas puramente operativas (Aerolínea, Aeronave llegada tardía) son las más frecuentes, mientras que los factores externos (Clima, Seguridad) generan menos eventos, pero con gran variabilidad.

# %%
# import pandas as pd
# import plotly.express as px
# import gc

# 1) Filtro mínimo de columnas y tipos compactos
mask = v["ARRIVAL_DELAY"] > 0
df_plot = v.loc[mask, ["ARRIVAL_DELAY", "MOTIVO_RETRASO"]].copy()
df_plot["ARRIVAL_DELAY"] = pd.to_numeric(df_plot["ARRIVAL_DELAY"], downcast="float")
df_plot["MOTIVO_RETRASO"] = df_plot["MOTIVO_RETRASO"].astype("category")

# 2) Quedarse con Top 10 motivos para no sobrecargar
top_motivos = df_plot["MOTIVO_RETRASO"].value_counts().nlargest(100).index
df_plot = df_plot[df_plot["MOTIVO_RETRASO"].isin(top_motivos)]

# # 3) (Opcional) MUESTRA por categoría para ver puntos sin reventar memoria
# #    Descomenta si quieres puntos. Si no, deja el boxplot sin puntos (más liviano).
# max_per_cat = 3000  # ajusta según tu RAM
# df_plot = (df_plot
#            .groupby("MOTIVO_RETRASO", group_keys=False)
#            .apply(lambda g: g.sample(min(len(g), max_per_cat), random_state=42))
#           ).reset_index(drop=True)

# 4) Boxplot (sin puntos) — MUY liviano
fig = px.box(
    df_plot,
    x="MOTIVO_RETRASO",
    y="ARRIVAL_DELAY",
    color="MOTIVO_RETRASO",
    # points=False,  # usa "outliers" si quieres ver algunos puntos
    points="outliers",
    title="Distribución de retrasos en llegada por motivo (Top 10)",
    labels={"ARRIVAL_DELAY": "Retraso en llegada (min)", "MOTIVO_RETRASO": "Motivo del retraso"},
)
fig.update_layout(showlegend=False, height=600, xaxis_title="Motivo de retraso", yaxis_title="Minutos")
fig.show()

# 5) Limpieza de memoria
del df_plot
gc.collect()



# %% [markdown]
# Opción 2 — Barras: promedio de retraso por motivo (Top 10)

# %%
import pandas as pd
import plotly.express as px
import gc

# 1) Subset mínimo y tipos compactos
mask = v["ARRIVAL_DELAY"] > 0
df_avg = v.loc[mask, ["ARRIVAL_DELAY", "MOTIVO_RETRASO"]].copy()
df_avg["ARRIVAL_DELAY"] = pd.to_numeric(df_avg["ARRIVAL_DELAY"], downcast="float")
df_avg["MOTIVO_RETRASO"] = df_avg["MOTIVO_RETRASO"].astype("category")

# 2) Top 10 motivos por volumen
top_motivos = df_avg["MOTIVO_RETRASO"].value_counts().nlargest(100).index
df_avg = df_avg[df_avg["MOTIVO_RETRASO"].isin(top_motivos)]

# 3) Agregación muy compacta
df_avg = (df_avg
          .groupby("MOTIVO_RETRASO", as_index=False)
          .agg(promedio_retraso=("ARRIVAL_DELAY","mean"),
               vuelos=("ARRIVAL_DELAY","size"))
          .sort_values("promedio_retraso", ascending=False))

# 4) Gráfico
fig = px.bar(
    df_avg,
    x="promedio_retraso",
    y="MOTIVO_RETRASO",
    color="promedio_retraso",
    text="vuelos",
    orientation="h",
    title="Promedio de retraso en llegada por motivo (Top 10)",
    labels={"promedio_retraso":"Promedio de retraso (min)", "MOTIVO_RETRASO":"Motivo", "vuelos":"Vuelos"},
    color_continuous_scale="RdYlGn_r"
)
fig.update_traces(texttemplate="%{text:,}", textposition="outside")
fig.update_layout(height=600, yaxis={'categoryorder': 'total ascending'}, showlegend=False)
fig.show()

# 5) Liberar memoria
del df_avg
gc.collect()


# %% [markdown]
# Opción 3 — Scatter: ARRIVAL_DELAY vs DISTANCE coloreado por MOTIVO_RETRASO (muestra balanceada)

# %%
# import pandas as pd
# import plotly.express as px
# import numpy as np
# import gc

# 1) Subset mínimo y tipos compactos
mask = v["ARRIVAL_DELAY"] > 0
cols = ["ARRIVAL_DELAY", "DISTANCE", "MOTIVO_RETRASO", "AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT"]
df_sc = v.loc[mask, cols].copy()

df_sc["ARRIVAL_DELAY"] = pd.to_numeric(df_sc["ARRIVAL_DELAY"], downcast="float")
df_sc["DISTANCE"] = pd.to_numeric(df_sc["DISTANCE"], downcast="integer")
df_sc["MOTIVO_RETRASO"] = df_sc["MOTIVO_RETRASO"].astype("category")

# 2) Limitar a Top N motivos (para colores/leyenda manejables)
top_motivos = df_sc["MOTIVO_RETRASO"].value_counts().nlargest(100).index
df_sc = df_sc[df_sc["MOTIVO_RETRASO"].isin(top_motivos)]

# 3) Muestreo balanceado por motivo (evita MemoryError y sesgos visuales)
max_per_cat = 2500  # ajusta según tu RAM/fluidez
df_sc = (df_sc.groupby("MOTIVO_RETRASO", group_keys=False)
              .apply(lambda g: g.sample(min(len(g), max_per_cat), random_state=42))
              .reset_index(drop=True))

# (Opcional) recorte de outliers para visual más legible
df_sc["ARRIVAL_DELAY_CLIP"] = df_sc["ARRIVAL_DELAY"].clip(lower=-10, upper=240)

# 4) Gráfico
fig = px.scatter(
    df_sc,
    x="DISTANCE",
    y="ARRIVAL_DELAY_CLIP",
    color="MOTIVO_RETRASO",
    hover_data={"AIRLINE": True, "ORIGIN_AIRPORT": True, "DESTINATION_AIRPORT": True,
                "ARRIVAL_DELAY_CLIP": False, "ARRIVAL_DELAY": True, "DISTANCE": True},
    title="Retraso en llegada vs distancia (muestra balanceada por motivo)",
    labels={"DISTANCE":"Distancia (millas)", "ARRIVAL_DELAY_CLIP":"Retraso en llegada (min, recortado)"},
    opacity=0.6
)
fig.update_layout(height=600)
fig.show()

# 5) Liberar memoria
del df_sc
gc.collect()


# %% [markdown]
# Gráfico: Cantidad de vuelos por motivo de retraso (Top N)

# %%
# import pandas as pd
# import plotly.express as px
# import gc

# === Parámetros ===
TOP_N = 100  # cambia a 20/50/100 si quieres
SOLO_RETRASADOS = True  # True: ARRIVAL_DELAY>0, False: todos

# === Subset mínimo ===
cols_need = ["ARRIVAL_DELAY", "MOTIVO_RETRASO"]
df_plot_src = v.loc[:, cols_need]

# Filtro opcional: solo retrasados
if SOLO_RETRASADOS:
    df_plot_src = df_plot_src[df_plot_src["ARRIVAL_DELAY"] > 0]

# Optimización de memoria
if df_plot_src["MOTIVO_RETRASO"].dtype != "category":
    df_plot_src["MOTIVO_RETRASO"] = df_plot_src["MOTIVO_RETRASO"].astype("category")

# === Conteo correcto con nombres fijos ===
# Opción A (value_counts):
df_count = (
    df_plot_src["MOTIVO_RETRASO"]
      .value_counts(dropna=False)
      .reset_index(name="CANTIDAD_VUELOS")
      .rename(columns={"index": "MOTIVO_RETRASO"})
)

# (Equivalente con groupby.size(): 
# df_count = (df_plot_src.groupby("MOTIVO_RETRASO", observed=True)
#                        .size().reset_index(name="CANTIDAD_VUELOS"))
# )

# Top N
df_count = df_count.head(TOP_N)

# Verificación defensiva
print("Columnas df_count:", list(df_count.columns))
print(df_count.head(3))

# === Gráfico ===
fig = px.bar(
    df_count,
    x="CANTIDAD_VUELOS",
    y="MOTIVO_RETRASO",
    orientation="h",
    text="CANTIDAD_VUELOS",
    color="CANTIDAD_VUELOS",
    color_continuous_scale="Blues",
    title=("Cantidad de vuelos con retraso por motivo (Top "
           f"{TOP_N})" if SOLO_RETRASADOS else f"Cantidad total de vuelos por motivo (Top {TOP_N})"),
    labels={"CANTIDAD_VUELOS": "Cantidad de vuelos", "MOTIVO_RETRASO": "Motivo del retraso"}
)

fig.update_traces(texttemplate="%{text:,}", textposition="outside")
fig.update_layout(height=600, showlegend=False,
                  yaxis={'categoryorder': 'total ascending'})
fig.show()

del df_plot_src, df_count
gc.collect()



# %% [markdown]
# cantidad y el porcentaje de vuelos por motivo de retraso (MOTIVO_RETRASO).

# %%
import pandas as pd
import plotly.graph_objects as go

# -----------------------------
# Parámetro: excluir categorías "no explicativas"
# -----------------------------
EXCLUIR_NO_EXPLICATIVAS = False
CATS_NO_EXPLIC = {"Sin retraso", "Retraso sin causa reportada"}

# Serie base
motivos = v["MOTIVO_RETRASO"].copy()

if EXCLUIR_NO_EXPLICATIVAS:
    motivos = motivos[~motivos.isin(CATS_NO_EXPLIC)]

total = len(motivos)

# Conteo y % sin problemas de tipos
conteo = motivos.value_counts(dropna=False)
porc = motivos.value_counts(dropna=False, normalize=True) * 100

# Armar DataFrame ordenado
df_mot = (
    pd.DataFrame({
        "MOTIVO_RETRASO": conteo.index.astype(str),
        "CANTIDAD": conteo.values,                      # ya es numérico
        "PORCENTAJE": porc.reindex(conteo.index).values # alineado al mismo orden
    })
    .assign(PORCENTAJE=lambda d: d["PORCENTAJE"].round(2))
)

# Limitar a Top-N para mejor legibilidad (ajusta si quieres)
TOP_N = 100
df_plot = df_mot.head(TOP_N)

# -----------------------------
# Gráfico combinado (cantidad + %)
# -----------------------------
fig = go.Figure()

# Barras: Cantidad
fig.add_trace(go.Bar(
    x=df_plot["CANTIDAD"],
    y=df_plot["MOTIVO_RETRASO"],
    orientation="h",
    name="Cantidad de vuelos",
    text=df_plot["CANTIDAD"].map("{:,}".format),
    textposition="outside",
    marker=dict(color="rgba(0, 102, 204, 0.75)")
))

# Línea: Porcentaje (eje secundario)
fig.add_trace(go.Scatter(
    x=df_plot["PORCENTAJE"],
    y=df_plot["MOTIVO_RETRASO"],
    mode="lines+markers+text",
    name="% del total",
    text=df_plot["PORCENTAJE"].astype(str) + "%",
    textposition="middle right",
    marker=dict(size=8),
    line=dict(width=2),
    xaxis="x2"
))

# Layout con doble eje X
fig.update_layout(
    title="📊 Cantidad y porcentaje por motivo de retraso " + ("(solo con causa)" if EXCLUIR_NO_EXPLICATIVAS else "(incluye todo)"),
    height=750,
    bargap=0.25,
    template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    yaxis=dict(title="Motivo de retraso", categoryorder="total ascending"),
    xaxis=dict(title="Cantidad de vuelos"),
    xaxis2=dict(
        overlaying="x",
        side="top",
        title="% de vuelos",
        showgrid=False
    )
)

fig.show()

# (opcional) Imprimir el resumen en tabla como pediste
print("— Resumen TOP motivos —")
display(df_plot.rename(columns={"CANTIDAD":"conteo", "PORCENTAJE":"porcentaje"}))


# %%
# import pandas as pd
# import plotly.graph_objects as go

# ===============================
# 🔹 Filtrar vuelos con retraso real
# ===============================
v_retrasos = v[v["MOTIVO_RETRASO"] != "Sin retraso"].copy()

total = len(v_retrasos)

# ===============================
# 🔹 Calcular conteos y porcentajes
# ===============================
conteo = v_retrasos["MOTIVO_RETRASO"].value_counts(dropna=False)
porc = (conteo / total * 100).round(2)

# Crear DataFrame ordenado
df_mot = (
    pd.DataFrame({
        "MOTIVO_RETRASO": conteo.index.astype(str),
        "CANTIDAD": conteo.values,
        "PORCENTAJE": porc.values
    })
)

# Tomar top 15 motivos
df_plot = df_mot.head(15)

# ===============================
# 🔹 Gráfico combinado (cantidad + %)
# ===============================
fig = go.Figure()

# Barras: cantidad de vuelos
fig.add_trace(go.Bar(
    x=df_plot["CANTIDAD"],
    y=df_plot["MOTIVO_RETRASO"],
    orientation="h",
    name="Cantidad de vuelos",
    text=df_plot["CANTIDAD"].map("{:,}".format),
    textposition="outside",
    marker=dict(color="rgba(0, 102, 204, 0.75)")
))

# Línea: porcentaje (%)
fig.add_trace(go.Scatter(
    x=df_plot["PORCENTAJE"],
    y=df_plot["MOTIVO_RETRASO"],
    mode="lines+markers+text",
    name="% del total",
    text=df_plot["PORCENTAJE"].astype(str) + "%",
    textposition="middle right",
    marker=dict(size=8, color="orange"),
    line=dict(color="orange", width=2),
    xaxis="x2"
))

# ===============================
# 🔹 Layout
# ===============================
fig.update_layout(
    title="📊 Cantidad y porcentaje por motivo de retraso (excluyendo 'Sin retraso')",
    height=750,
    bargap=0.25,
    template="plotly_white",
    yaxis=dict(title="Motivo de retraso", categoryorder="total ascending"),
    xaxis=dict(title="Cantidad de vuelos"),
    xaxis2=dict(
        overlaying="x",
        side="top",
        title="% de vuelos",
        showgrid=False
    ),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

fig.show()

# ===============================
# 🔹 Resumen de motivos (Top 15)
# ===============================
print("📋 Distribución de motivos de retraso (sin 'Sin retraso'):")
display(df_plot.rename(columns={"CANTIDAD":"conteo","PORCENTAJE":"porcentaje"}))


# %% [markdown]
# Solo vuelos con retraso real (>1 minuto) y causa conocida.
# 
# Cantidad de vuelos por motivo.
# 
# Promedio de minutos de ARRIVAL_DELAY por motivo.
# 
# Visualización con dos ejes:
# 
# Eje inferior → cantidad de vuelos.
# 
# Eje superior → promedio de minutos.

# %%
# import pandas as pd
# import plotly.graph_objects as go

# ===============================
# 🔹 Filtrar vuelos con retraso (ARRIVAL_DELAY > 1)
# ===============================
v_delay = v[(v["ARRIVAL_DELAY"] > 1) & (v["MOTIVO_RETRASO"] != "Sin retraso")].copy()

total = len(v_delay)
print(f"✈️ Total de vuelos con ARRIVAL_DELAY > 1 y causa conocida: {total:,}")

# ===============================
# 🔹 Calcular conteo y promedio de ARRIVAL_DELAY
# ===============================
df_mot = (
    v_delay.groupby("MOTIVO_RETRASO", observed=False)
    .agg(
        CANTIDAD_VUELOS=("ARRIVAL_DELAY", "count"),
        PROMEDIO_RETRASO=("ARRIVAL_DELAY", "mean")
    )
    .reset_index()
    .sort_values("CANTIDAD_VUELOS", ascending=False)
)

# Calcular % de participación
df_mot["PORCENTAJE"] = (df_mot["CANTIDAD_VUELOS"] / total * 100).round(2)

# Limitar a los 15 principales motivos
df_plot = df_mot.head(100)

# ===============================
# 🔹 Gráfico combinado (cantidad + promedio minutos)
# ===============================
fig = go.Figure()

# Barras: cantidad de vuelos retrasados
fig.add_trace(go.Bar(
    x=df_plot["CANTIDAD_VUELOS"],
    y=df_plot["MOTIVO_RETRASO"],
    orientation='h',
    name="Cantidad de vuelos retrasados",
    text=df_plot["CANTIDAD_VUELOS"].map("{:,}".format),
    textposition="outside",
    marker=dict(color="rgba(0, 102, 204, 0.75)")
))

# Línea: promedio de minutos de retraso
fig.add_trace(go.Scatter(
    x=df_plot["PROMEDIO_RETRASO"],
    y=df_plot["MOTIVO_RETRASO"],
    mode="lines+markers+text",
    name="Promedio minutos de retraso",
    text=df_plot["PROMEDIO_RETRASO"].round(1).astype(str) + " min",
    textposition="middle right",
    marker=dict(size=8, color="orange"),
    line=dict(color="orange", width=2),
    xaxis="x2"
))

# ===============================
# 🔹 Layout del gráfico
# ===============================
fig.update_layout(
    title="✈️ Motivos de retraso (ARRIVAL_DELAY > 1 min): cantidad y promedio de minutos",
    height=750,
    bargap=0.25,
    template="plotly_white",
    yaxis=dict(title="Motivo de retraso", categoryorder="total ascending"),
    xaxis=dict(title="Cantidad de vuelos con retraso"),
    xaxis2=dict(
        overlaying="x",
        side="top",
        title="Promedio de minutos de ARRIVAL_DELAY",
        showgrid=False
    ),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

fig.show()

# ===============================
# 🔹 Resumen (Top motivos)
# ===============================
print("📋 Promedio de retraso por motivo (Top 15):")
display(df_plot[["MOTIVO_RETRASO", "CANTIDAD_VUELOS", "PORCENTAJE", "PROMEDIO_RETRASO"]])


# %% [markdown]
# mismo análisis, pero invertido visualmente:
# 
# 🔵 Barras → promedio de minutos de ARRIVAL_DELAY
# 
# 🟠 Línea → cantidad de vuelos retrasados
# 
# Así puedes identificar fácilmente qué causas generan retrasos más largos, no solo más frecuentes.

# %%
import pandas as pd
import plotly.graph_objects as go

# ===============================
# 🔹 Filtrar vuelos con retraso real
# ===============================
v_delay = v[(v["ARRIVAL_DELAY"] > 1) & (v["MOTIVO_RETRASO"] != "Sin retraso")].copy()

total = len(v_delay)
print(f"✈️ Total de vuelos con ARRIVAL_DELAY > 1 y causa conocida: {total:,}")

# ===============================
# 🔹 Calcular promedio y cantidad
# ===============================
df_mot = (
    v_delay.groupby("MOTIVO_RETRASO", observed=False)
    .agg(
        PROMEDIO_RETRASO=("ARRIVAL_DELAY", "mean"),
        CANTIDAD_VUELOS=("ARRIVAL_DELAY", "count")
    )
    .reset_index()
    .sort_values("PROMEDIO_RETRASO", ascending=False)
)

df_mot["PORCENTAJE"] = (df_mot["CANTIDAD_VUELOS"] / total * 100).round(2)

# Top motivos
df_plot = df_mot.head(15)

# ===============================
# 🔹 Gráfico invertido
# ===============================
fig = go.Figure()

# Barras → Promedio de minutos de ARRIVAL_DELAY
fig.add_trace(go.Bar(
    x=df_plot["PROMEDIO_RETRASO"],
    y=df_plot["MOTIVO_RETRASO"],
    orientation='h',
    name="Promedio minutos de ARRIVAL_DELAY",
    text=df_plot["PROMEDIO_RETRASO"].round(1).astype(str) + " min",
    textposition="outside",
    marker=dict(color="rgba(0, 102, 204, 0.75)")
))

# Línea → Cantidad de vuelos retrasados
fig.add_trace(go.Scatter(
    x=df_plot["CANTIDAD_VUELOS"],
    y=df_plot["MOTIVO_RETRASO"],
    mode="lines+markers+text",
    name="Cantidad de vuelos retrasados",
    text=df_plot["CANTIDAD_VUELOS"].map("{:,}".format),
    textposition="middle right",
    marker=dict(size=8, color="orange"),
    line=dict(color="orange", width=2),
    xaxis="x2"
))

# ===============================
# 🔹 Layout
# ===============================
fig.update_layout(
    title="📊 Promedio de minutos de ARRIVAL_DELAY y cantidad de vuelos por motivo (ARRIVAL_DELAY > 1)",
    height=750,
    bargap=0.25,
    template="plotly_white",
    yaxis=dict(title="Motivo de retraso", categoryorder="total ascending"),
    xaxis=dict(title="Promedio de minutos de ARRIVAL_DELAY"),
    xaxis2=dict(
        overlaying="x",
        side="top",
        title="Cantidad de vuelos retrasados",
        showgrid=False
    ),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

fig.show()

# ===============================
# 🔹 Tabla resumen
# ===============================
print("📋 Motivos con mayor promedio de retraso (Top 15):")
display(df_plot[["MOTIVO_RETRASO", "PROMEDIO_RETRASO", "CANTIDAD_VUELOS", "PORCENTAJE"]])


# %% [markdown]
# dos conjuntos de barras superpuestos, uno para el
# 🔵 promedio de minutos de retraso y otro para la
# 🟠 cantidad de vuelos retrasados, en el mismo eje horizontal.

# %%
# import pandas as pd
# import plotly.graph_objects as go

# ===============================
# 🔹 Filtrar vuelos con retraso real
# ===============================
v_delay = v[(v["ARRIVAL_DELAY"] > 1) & (v["MOTIVO_RETRASO"] != "Sin retraso")].copy()

total = len(v_delay)
print(f"✈️ Total de vuelos con ARRIVAL_DELAY > 1 y causa conocida: {total:,}")

# ===============================
# 🔹 Calcular métricas por motivo
# ===============================
df_mot = (
    v_delay.groupby("MOTIVO_RETRASO", observed=False)
    .agg(
        PROMEDIO_RETRASO=("ARRIVAL_DELAY", "mean"),
        CANTIDAD_VUELOS=("ARRIVAL_DELAY", "count")
    )
    .reset_index()
)

df_mot["PORCENTAJE"] = (df_mot["CANTIDAD_VUELOS"] / total * 100).round(2)

# Top motivos (orden por cantidad)
df_plot = df_mot.sort_values("CANTIDAD_VUELOS", ascending=False).head(15)

# ===============================
# 🔹 Gráfico con dos barras
# ===============================
fig = go.Figure()

# Barras 1: Promedio de minutos de ARRIVAL_DELAY (color azul)
fig.add_trace(go.Bar(
    x=df_plot["PROMEDIO_RETRASO"],
    y=df_plot["MOTIVO_RETRASO"],
    orientation='h',
    name="Promedio minutos de ARRIVAL_DELAY",
    text=df_plot["PROMEDIO_RETRASO"].round(1).astype(str) + " min",
    textposition="outside",
    marker=dict(color="rgba(30, 144, 255, 0.7)")  # Azul
))

# Barras 2: Cantidad de vuelos retrasados (color naranja)
fig.add_trace(go.Bar(
    x=df_plot["CANTIDAD_VUELOS"],
    y=df_plot["MOTIVO_RETRASO"],
    orientation='h',
    name="Cantidad de vuelos retrasados",
    text=df_plot["CANTIDAD_VUELOS"].map("{:,}".format),
    textposition="outside",
    marker=dict(color="rgba(255, 165, 0, 0.7)")  # Naranja
))

# ===============================
# 🔹 Layout del gráfico
# ===============================
fig.update_layout(
    title="✈️ Motivos de retraso (ARRIVAL_DELAY > 1 min): Promedio vs Cantidad de vuelos",
    height=750,
    barmode="group",  # Barras agrupadas
    template="plotly_white",
    yaxis=dict(title="Motivo de retraso", categoryorder="total ascending"),
    xaxis=dict(title="Valor"),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)

fig.show()

# ===============================
# 🔹 Tabla resumen
# ===============================
print("📋 Motivos con mayor impacto (Top 15):")
display(df_plot[["MOTIVO_RETRASO", "CANTIDAD_VUELOS", "PORCENTAJE", "PROMEDIO_RETRASO"]])


# %% [markdown]
# Barras + línea con doble eje X (recomendada)
# 
# 🔹 Ideal cuando las escalas son muy distintas.
# 🔹 Muestra barras para cantidad de vuelos y línea para promedio de minutos con ejes separados.

# %%
import plotly.graph_objects as go
import pandas as pd

# --- Filtrar retrasos reales ---
v_delay = v[(v["ARRIVAL_DELAY"] > 1) & (v["MOTIVO_RETRASO"] != "Sin retraso")].copy()
df = (
    v_delay.groupby("MOTIVO_RETRASO", observed=False)
    .agg(
        PROMEDIO_RETRASO=("ARRIVAL_DELAY", "mean"),
        CANTIDAD_VUELOS=("ARRIVAL_DELAY", "count")
    )
    .reset_index()
)
df["PORCENTAJE"] = (df["CANTIDAD_VUELOS"] / len(v_delay) * 100).round(2)
df = df.sort_values("CANTIDAD_VUELOS", ascending=False).head(100)

# --- Crear figura ---
fig = go.Figure()

# Barras → cantidad de vuelos retrasados
fig.add_trace(go.Bar(
    x=df["CANTIDAD_VUELOS"],
    y=df["MOTIVO_RETRASO"],
    orientation="h",
    name="Cantidad de vuelos retrasados",
    marker_color="orange",
    text=df["CANTIDAD_VUELOS"].map("{:,}".format),
    textposition="outside"
))

# Línea → promedio de minutos (segundo eje)
fig.add_trace(go.Scatter(
    x=df["PROMEDIO_RETRASO"],
    y=df["MOTIVO_RETRASO"],
    mode="lines+markers+text",
    name="Promedio de ARRIVAL_DELAY (min)",
    text=df["PROMEDIO_RETRASO"].round(1).astype(str) + " min",
    textposition="middle right",
    line=dict(color="royalblue", width=3),
    marker=dict(size=8, color="royalblue"),
    xaxis="x2"
))

fig.update_layout(
    title="✈️ Motivos de retraso: cantidad de vuelos vs. promedio de minutos",
    height=750,
    template="plotly_white",
    yaxis=dict(title="Motivo de retraso", categoryorder="total ascending"),
    xaxis=dict(title="Cantidad de vuelos retrasados"),
    xaxis2=dict(
        overlaying="x",
        side="top",
        title="Promedio de minutos de ARRIVAL_DELAY",
        showgrid=False
    ),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

fig.show()



# %% [markdown]
# Gráfico de dispersión (bubble chart)
# 
# 🔹 Muestra cada motivo como un punto:
# 
# eje X = cantidad de vuelos
# 
# eje Y = promedio de minutos
# 
# tamaño de burbuja = porcentaje del total

# %% [markdown]
# revisar dataframe final como queda con la nueva columna

# %%
import plotly.express as px

fig = px.scatter(
    df,
    x="CANTIDAD_VUELOS",
    y="PROMEDIO_RETRASO",
    size="PORCENTAJE",
    text="MOTIVO_RETRASO",
    color="PROMEDIO_RETRASO",
    color_continuous_scale="RdYlBu_r",
    title="✈️ Relación entre cantidad de vuelos retrasados y promedio de ARRIVAL_DELAY",
    labels={
        "CANTIDAD_VUELOS": "Cantidad de vuelos retrasados",
        "PROMEDIO_RETRASO": "Promedio de minutos de ARRIVAL_DELAY"
    }
)

fig.update_traces(textposition="top center")
fig.update_layout(height=700, template="plotly_white")
fig.show()



# %% [markdown]
# Heatmap
# 
# 🔹 Muestra la intensidad de ambas variables de forma bidimensional (ideal si luego tienes más variables como mes o aerolínea).

# %%
import plotly.express as px

fig = px.density_heatmap(
    df,
    x="PROMEDIO_RETRASO",
    y="MOTIVO_RETRASO",
    z="CANTIDAD_VUELOS",
    color_continuous_scale="YlOrRd",
    title="Mapa de calor: promedio de retraso vs cantidad de vuelos por motivo"
)

fig.update_layout(height=700, template="plotly_white")
fig.show()


# %%
v.info()

# %%
v.head()

# %%
v.describe()

# %% [markdown]
# ### **Nuevas columnas útiles (para análisis y futura predicción)**

# %%
# 8.1 Variable objetivo binaria (retraso en llegada > 15 min)
v["RETRASADO_LLEGADA"] = (v["ARRIVAL_DELAY"]   > 15).astype(int)
v["RETRASADO_SALIDA"]  = (v["DEPARTURE_DELAY"] > 15).astype(int)

# 8.2 Hora de salida programada
v["HORA_SALIDA"] = (v["SCHEDULED_DEPARTURE"] // 100).clip(0,23)

# 8.3 Hora de llegada programada
v["HORA_LLEGADA"] = (v["SCHEDULED_ARRIVAL"] // 100).clip(0,23)

# 8.4 Periodo del día
v["PERIODO_SALIDA"] = pd.cut(v["HORA_SALIDA"], bins=[0,6,12,18,24],
                             labels=["Madrugada","Mañana","Tarde","Noche"], right=False)
# 8.4 Periodo del día
v["PERIODO_LLEGADA"] = pd.cut(v["HORA_LLEGADA"], bins=[0,6,12,18,24],
                             labels=["Madrugada","Mañana","Tarde","Noche"], right=False)
# 8.5 Ruta (origen-destino)
v["RUTA"] = v["ORIGIN_AIRPORT"] + "_" + v["DESTINATION_AIRPORT"]


# %% [markdown]
# ### **usar missingno para vistas rápidas:**

# %%
# si no lo tienes: pip install missingno
# import missingno as msno
msno.matrix(v.sample(min(len(v), 100000), random_state=42))
plt.show()


# %%
msno.bar(v)
plt.show()


# %%
v.info()

# %%
v.head()

# %%
v.nunique()

# %%
v.RETRASADO_LLEGADA.nunique()

# %% [markdown]
# ### **Revisar si las columnas calculadas DEPARTURE_DELAY Y ARRIVAL_DELAY, estan correctamente calculadas
# 
# Si >95% coincide, usamos columnas originales; si no, preferimos calculadas o revisamos outliers

# %%
def hhmm_to_minutes(x):
    # maneja nulos
    if pd.isna(x): return np.nan
    x = int(x)
    return (x // 100) * 60 + (x % 100)

def diff_minutes(actual, sched):
    if np.isnan(actual) or np.isnan(sched): 
        return np.nan
    d = actual - sched
    # si pasó medianoche, ajustar (rango +- 24h)
    if d < - 12*60:  # gran negativo -> sumo 24h
        d += 24*60
    if d >  12*60:  # gran positivo -> resto 24h
        d -= 24*60
    return d

# Calcular minutos desde HHMM
v["DEP_TIME_MIN"]  = v["DEPARTURE_TIME"].apply(hhmm_to_minutes)
v["DEP_SCHED_MIN"] = v["SCHEDULED_DEPARTURE"].apply(hhmm_to_minutes)
v["ARR_TIME_MIN"]  = v["ARRIVAL_TIME"].apply(hhmm_to_minutes)
v["ARR_SCHED_MIN"] = v["SCHEDULED_ARRIVAL"].apply(hhmm_to_minutes)

# Recalcular delays
v["DELAY_DEP_CALC"] = v.apply(lambda r: diff_minutes(r["DEP_TIME_MIN"],  r["DEP_SCHED_MIN"]), axis=1)
v["DELAY_ARR_CALC"] = v.apply(lambda r: diff_minutes(r["ARR_TIME_MIN"],  r["ARR_SCHED_MIN"]), axis=1)

# Comparar con columnas provistas
cmp_dep = (v["DELAY_DEP_CALC"] - v["DEPARTURE_DELAY"]).abs()
cmp_arr = (v["DELAY_ARR_CALC"] - v["ARRIVAL_DELAY"]).abs()

print("Coincidencia DEPARTURE_DELAY vs calculado (tolerancia 1 min):",
      (cmp_dep <= 1).mean()*100, "%")
print("Coincidencia ARRIVAL_DELAY   vs calculado (tolerancia 1 min):",
      (cmp_arr <= 1).mean()*100, "%")

# Si >95% coincide, usamos columnas originales; si no, preferimos calculadas o revisamos outliers


# %%
v.head()

# %%
v.info()

# %%
# cambiar distancia a tipo de dato float ( no necesario ya que esta en int)
# v["DISTANCE"] = v["DISTANCE"].astype(float)

# %%
v.head()

# %%
v.groupby("PERIODO_SALIDA")["RETRASADO_LLEGADA"].mean()

# videogames["Genre"].value_counts(normalize=True)*100


# %%
v.groupby("PERIODO_LLEGADA")["RETRASADO_LLEGADA"].mean()

# %%
v["PERIODO_SALIDA"].value_counts(normalize=True)*100

# %%
v["PERIODO_LLEGADA"].value_counts(normalize=True)*100

# %%
v["RETRASADO_LLEGADA"].value_counts(normalize=True)*100

# %%
v["RETRASADO_SALIDA"].value_counts(normalize=True)*100

# %% [markdown]
# grafica el porcentaje de retrasos por PERIODO_SALIDA (Madrugada, Mañana, Tarde, Noche) y además anota en cada barra el % y el total de vuelos en ese periodo.

# %%
# import pandas as pd
# import matplotlib.pyplot as plt

# --- A) Asegurar PERIODO_SALIDA (si ya lo tienes, este bloque no altera nada) ---
if "PERIODO_SALIDA" not in v.columns:
    v["PERIODO_SALIDA"] = pd.cut(
        v["HOUR"],
        bins=[0, 6, 12, 18, 24],
        labels=["Madrugada", "Mañana", "Tarde", "Noche"],
        right=False
    )

# Orden fijo de categorías para mostrar siempre en el mismo orden
orden = ["Madrugada", "Mañana", "Tarde", "Noche"]
v["PERIODO_SALIDA"] = pd.Categorical(v["PERIODO_SALIDA"], categories=orden, ordered=True)

# --- B) Resumen por periodo: total y % retrasos ---
# RETRASADO_LLEGADA debe ser 0/1
resumen = v.groupby("PERIODO_SALIDA", observed=True).agg(
    TOTAL=("RETRASADO_LLEGADA", "size"),
    PORC_RETRASOS=("RETRASADO_LLEGADA", "mean")
).reset_index()

# Convertir a porcentaje
resumen["PORC_RETRASOS_PCT"] = resumen["PORC_RETRASOS"] * 100

# (Opcional) mostrar tabla
display(resumen)

# --- C) Gráfico de barras con anotaciones de % y total ---
plt.figure(figsize=(7.5, 4.5))
barras = plt.bar(resumen["PERIODO_SALIDA"], resumen["PORC_RETRASOS_PCT"])

plt.title("Porcentaje de vuelos retrasados por periodo del día")
plt.xlabel("Periodo del día")
plt.ylabel("Retrasos (%)")
plt.ylim(0, max(5, resumen["PORC_RETRASOS_PCT"].max() * 1.15))  # pequeño margen superior
plt.tight_layout()

# Anotaciones: encima de cada barra → "xx.xx%  |  n=TOTAL"
for i, (pct, total) in enumerate(zip(resumen["PORC_RETRASOS_PCT"], resumen["TOTAL"])):
    plt.text(
        i, pct + (resumen["PORC_RETRASOS_PCT"].max() * 0.02),  # un poco por encima
        f"{pct:.2f}%  |  n={total:,}",
        ha="center", va="bottom", fontsize=10, fontweight="bold"
    )

plt.show()


# %% [markdown]
# El gráfico de porcentaje de retrasos por periodo del día (SIMILAR al anterior).
# 
# Un gráfico de barras apiladas que compara vuelos a tiempo vs retrasados por cada periodo.
# 
# Esto te da una visión comparativa y analítica: volumen total y proporción visual de puntualidad.

# %%
# import pandas as pd
# import matplotlib.pyplot as plt

# --- A) Asegurar columna PERIODO_SALIDA ---
if "PERIODO_SALIDA" not in v.columns:
    v["PERIODO_SALIDA"] = pd.cut(
        v["HOUR"],
        bins=[0, 6, 12, 18, 24],
        labels=["Madrugada", "Mañana", "Tarde", "Noche"],
        right=False
    )

orden = ["Madrugada", "Mañana", "Tarde", "Noche"]
v["PERIODO_SALIDA"] = pd.Categorical(v["PERIODO_SALIDA"], categories=orden, ordered=True)

# --- B) Resumen de retrasos y puntualidad por periodo ---
resumen_stack = v.groupby("PERIODO_SALIDA", observed=True)["RETRASADO_LLEGADA"].value_counts().unstack(fill_value=0)
resumen_stack.columns = ["A_Tiempo", "Retrasado"]  # 0 = A tiempo, 1 = Retrasado
resumen_stack = resumen_stack.reindex(orden).reset_index()

# Calcular porcentajes para cada periodo
resumen_stack["Total"] = resumen_stack["A_Tiempo"] + resumen_stack["Retrasado"]
resumen_stack["Porc_Retrasado"] = resumen_stack["Retrasado"] / resumen_stack["Total"] * 100
resumen_stack["Porc_A_Tiempo"] = 100 - resumen_stack["Porc_Retrasado"]

# Mostrar tabla resumen
display(resumen_stack)

# --- C) Gráfico 1: Porcentaje de vuelos retrasados por periodo ---
plt.figure(figsize=(7.5, 4.5))
barras = plt.bar(resumen_stack["PERIODO_SALIDA"], resumen_stack["Porc_Retrasado"], color="tomato")

plt.title("Porcentaje de vuelos retrasados por periodo del día")
plt.xlabel("Periodo del día")
plt.ylabel("Retrasos (%)")
plt.ylim(0, max(5, resumen_stack["Porc_Retrasado"].max() * 1.15))
plt.tight_layout()

for i, (pct, total) in enumerate(zip(resumen_stack["Porc_Retrasado"], resumen_stack["Total"])):
    plt.text(i, pct + 0.3, f"{pct:.2f}% | n={total:,}", ha="center", va="bottom", fontsize=10, fontweight="bold")

plt.show()

# --- D) Gráfico 2: Barras apiladas (vuelos a tiempo vs retrasados) ---
plt.figure(figsize=(8, 5))

plt.bar(resumen_stack["PERIODO_SALIDA"], resumen_stack["A_Tiempo"], label="A Tiempo", color="skyblue")
plt.bar(resumen_stack["PERIODO_SALIDA"], resumen_stack["Retrasado"],
        bottom=resumen_stack["A_Tiempo"], label="Retrasado", color="salmon")

plt.title("Distribución de vuelos a tiempo vs retrasados por periodo del día")
plt.xlabel("Periodo del día")
plt.ylabel("Cantidad de vuelos")
plt.legend()
plt.tight_layout()

# Anotar totales encima de las barras
for i, total in enumerate(resumen_stack["Total"]):
    plt.text(i, total + (resumen_stack["Total"].max() * 0.01), f"{total:,}", ha="center", va="bottom", fontsize=9, fontweight="bold")

plt.show()


# %%
# import pandas as pd
# import plotly.express as px

# --- A) Asegurar columna PERIODO_SALIDA ---
if "PERIODO_SALIDA" not in v.columns:
    v["PERIODO_SALIDA"] = pd.cut(
        v["HOUR"],
        bins=[0, 6, 12, 18, 24],
        labels=["Madrugada", "Mañana", "Tarde", "Noche"],
        right=False
    )

orden = ["Madrugada", "Mañana", "Tarde", "Noche"]
v["PERIODO_SALIDA"] = pd.Categorical(v["PERIODO_SALIDA"], categories=orden, ordered=True)

# --- B) Resumen de retrasos y puntualidad ---
resumen_stack = v.groupby("PERIODO_SALIDA", observed=True)["RETRASADO_LLEGADA"].value_counts().unstack(fill_value=0)
resumen_stack.columns = ["A_Tiempo", "Retrasado"]
resumen_stack = resumen_stack.reindex(orden).reset_index()

# Totales y porcentajes
resumen_stack["Total"] = resumen_stack["A_Tiempo"] + resumen_stack["Retrasado"]
resumen_stack["Porc_Retrasado"] = resumen_stack["Retrasado"] / resumen_stack["Total"] * 100
resumen_stack["Porc_A_Tiempo"] = 100 - resumen_stack["Porc_Retrasado"]

# --- C) Gráfico 1: Porcentaje de vuelos retrasados por periodo ---
fig1 = px.bar(
    resumen_stack,
    x="PERIODO_SALIDA",
    y="Porc_Retrasado",
    text=resumen_stack["Porc_Retrasado"].apply(lambda x: f"{x:.2f}%"),
    title="Porcentaje de vuelos retrasados por periodo del día",
    labels={"PERIODO_SALIDA": "Periodo del día", "Porc_Retrasado": "Porcentaje de retrasos (%)"},
    color="Porc_Retrasado",
    color_continuous_scale="Reds"
)

fig1.update_traces(textposition="outside")
fig1.update_layout(
    yaxis=dict(title="Porcentaje de retrasos (%)"),
    xaxis=dict(title="Periodo del día"),
    coloraxis_colorbar=dict(title="% Retrasos"),
    uniformtext_minsize=10,
    uniformtext_mode="hide"
)
fig1.show()

# --- D) Gráfico 2: Barras apiladas (vuelos a tiempo vs retrasados) ---
# Convertir a formato largo para Plotly
resumen_long = resumen_stack.melt(
    id_vars=["PERIODO_SALIDA", "Total"],
    value_vars=["A_Tiempo", "Retrasado"],
    var_name="Estado",
    value_name="Cantidad"
)

fig2 = px.bar(
    resumen_long,
    x="PERIODO_SALIDA",
    y="Cantidad",
    color="Estado",
    text=resumen_long["Cantidad"].apply(lambda x: f"{x:,}"),
    title="Distribución de vuelos a tiempo vs retrasados por periodo del día",
    labels={"PERIODO_SALIDA": "Periodo del día", "Cantidad": "Cantidad de vuelos", "Estado": "Estado"},
    color_discrete_map={"A_Tiempo": "skyblue", "Retrasado": "salmon"}
)

fig2.update_traces(textposition="outside")
fig2.update_layout(barmode="stack", yaxis_title="Cantidad de vuelos", xaxis_title="Periodo del día")
fig2.show()


# %% [markdown]
# visualización combinada — es decir:
# 
# Gráfico de porcentaje de retrasos por periodo (barras)
# 
# Gráfico de vuelos a tiempo vs retrasados (barras apiladas)
# 
# Gráfico combinado (barras + línea) → Muestra el total de vuelos por periodo y el porcentaje de retrasos superpuesto
# 
# Todo interactivo con Plotly Express

# %%
# import pandas as pd
# import plotly.express as px

# --- A) Asegurar columna PERIODO_SALIDA ---
if "PERIODO_SALIDA" not in v.columns:
    v["PERIODO_SALIDA"] = pd.cut(
        v["HOUR"],
        bins=[0, 6, 12, 18, 24],
        labels=["Madrugada", "Mañana", "Tarde", "Noche"],
        right=False
    )

orden = ["Madrugada", "Mañana", "Tarde", "Noche"]
v["PERIODO_SALIDA"] = pd.Categorical(v["PERIODO_SALIDA"], categories=orden, ordered=True)

# --- B) Resumen de retrasos y puntualidad ---
resumen_stack = v.groupby("PERIODO_SALIDA", observed=True)["RETRASADO_LLEGADA"].value_counts().unstack(fill_value=0)
resumen_stack.columns = ["A_Tiempo", "Retrasado"]
resumen_stack = resumen_stack.reindex(orden).reset_index()

# Totales y porcentajes
resumen_stack["Total"] = resumen_stack["A_Tiempo"] + resumen_stack["Retrasado"]
resumen_stack["Porc_Retrasado"] = resumen_stack["Retrasado"] / resumen_stack["Total"] * 100
resumen_stack["Porc_A_Tiempo"] = 100 - resumen_stack["Porc_Retrasado"]

# --- C) Gráfico 1: Porcentaje de vuelos retrasados por periodo ---
fig1 = px.bar(
    resumen_stack,
    x="PERIODO_SALIDA",
    y="Porc_Retrasado",
    text=resumen_stack["Porc_Retrasado"].apply(lambda x: f"{x:.2f}%"),
    title="Porcentaje de vuelos retrasados por periodo de salida",
    labels={"PERIODO_SALIDA": "Periodo de salida", "Porc_Retrasado": "Porcentaje de retrasos (%)"},
    color="Porc_Retrasado",
    color_continuous_scale="Reds"
)

fig1.update_traces(textposition="outside")
fig1.update_layout(
    yaxis=dict(title="Porcentaje de retrasos (%)"),
    xaxis=dict(title="Periodo de salida"),
    coloraxis_colorbar=dict(title="% Retrasos"),
    uniformtext_minsize=10,
    uniformtext_mode="hide"
)
fig1.show()

# --- D) Gráfico 2: Barras apiladas (vuelos a tiempo vs retrasados) ---
resumen_long = resumen_stack.melt(
    id_vars=["PERIODO_SALIDA", "Total"],
    value_vars=["A_Tiempo", "Retrasado"],
    var_name="Estado",
    value_name="Cantidad"
)

fig2 = px.bar(
    resumen_long,
    x="PERIODO_SALIDA",
    y="Cantidad",
    color="Estado",
    text=resumen_long["Cantidad"].apply(lambda x: f"{x:,}"),
    title="Distribución de vuelos a tiempo vs retrasados por periodo de salida",
    labels={"PERIODO_SALIDA": "Periodo de salida", "Cantidad": "Cantidad de vuelos", "Estado": "Estado"},
    color_discrete_map={"A_Tiempo": "skyblue", "Retrasado": "salmon"}
)

fig2.update_traces(textposition="outside")
fig2.update_layout(barmode="stack", yaxis_title="Cantidad de vuelos", xaxis_title="Periodo de salida")
fig2.show()

# --- E) Gráfico 3: Combinado (barras = total de vuelos, línea = % retrasos) ---
import plotly.graph_objects as go

fig3 = go.Figure()

# Barras: total de vuelos
fig3.add_trace(go.Bar(
    x=resumen_stack["PERIODO_SALIDA"],
    y=resumen_stack["Total"],
    name="Total de vuelos",
    marker_color="lightblue",
    text=resumen_stack["Total"].apply(lambda x: f"{x:,}"),
    textposition="outside",
    yaxis="y1"
))

# Línea: porcentaje de retrasos
fig3.add_trace(go.Scatter(
    x=resumen_stack["PERIODO_SALIDA"],
    y=resumen_stack["Porc_Retrasado"],
    name="% Retrasos",
    mode="lines+markers+text",
    text=resumen_stack["Porc_Retrasado"].apply(lambda x: f"{x:.2f}%"),
    textposition="top center",
    line=dict(color="red", width=3),
    yaxis="y2"
))

fig3.update_layout(
    title="Total de vuelos vs porcentaje de retrasos por periodo de salida",
    xaxis=dict(title="Periodo de salida"),
    yaxis=dict(title="Total de vuelos", side="left", showgrid=False),
    yaxis2=dict(title="% Retrasos", overlaying="y", side="right", showgrid=False),
    legend=dict(x=0.75, y=1.1, orientation="h"),
    bargap=0.3,
    template="plotly_white"
)

fig3.show()


# %% [markdown]
# Gráfico  combinado , usando marcadores coloreados por umbral:
# 
# Verde: < 10%
# 
# Amarillo: 10% – 20%
# 
# Rojo: > 20%
# 
# Incluye barras (total de vuelos), línea con puntos (%, coloreados por umbral), texto de % sobre cada punto y una leyenda personalizada para los umbrales.
# 
# Este bloque asume que ya tienes resumen_stack armado como en el paso anterior (con columnas: PERIODO_SALIDA, Total, Porc_Retrasado). Si no, pega primero el bloque donde se construye resumen_stack.

# %%
# import numpy as np
# import plotly.graph_objects as go

# --- Umbrales y colores por punto ---
pct = resumen_stack["Porc_Retrasado"].values
colors = np.where(
    pct < 15, "green",
    np.where(pct <= 25, "gold", "red")
)

# --- Figura combinada ---
fig = go.Figure()

# Barras: total de vuelos (eje Y izquierdo)
fig.add_trace(go.Bar(
    x=resumen_stack["PERIODO_SALIDA"],
    y=resumen_stack["Total"],
    name="Total de vuelos",
    marker_color="lightblue",
    text=resumen_stack["Total"].apply(lambda x: f"{x:,}"),
    textposition="outside",
    yaxis="y1",
    hovertemplate="<b>%{x}</b><br>Total de vuelos: %{y:,}<extra></extra>"
))

# Línea + puntos: % retrasos (eje Y derecho) con color por umbral
fig.add_trace(go.Scatter(
    x=resumen_stack["PERIODO_SALIDA"],
    y=resumen_stack["Porc_Retrasado"],
    name="% Retrasos",
    mode="lines+markers+text",
    line=dict(color="rgba(0,0,0,0.35)", width=2),  # línea neutra
    marker=dict(size=12, color=colors, line=dict(color="black", width=1)),
    text=[f"{v:.2f}%" for v in resumen_stack["Porc_Retrasado"]],
    textposition="top center",
    yaxis="y2",
    hovertemplate="<b>%{x}</b><br>% Retrasos: %{y:.2f}%<extra></extra>"
))

# --- Leyenda personalizada de umbrales (trazas vacías solo para leyenda) ---
fig.add_trace(go.Scatter(
    x=[None], y=[None], mode="markers",
    marker=dict(size=12, color="green"),
    name="< 15% retrasos"
))
fig.add_trace(go.Scatter(
    x=[None], y=[None], mode="markers",
    marker=dict(size=12, color="gold"),
    name="15% – 25% retrasos"
))
fig.add_trace(go.Scatter(
    x=[None], y=[None], mode="markers",
    marker=dict(size=12, color="red"),
    name="> 20% retrasos"
))

# --- Layout con doble eje Y ---
fig.update_layout(
    title="Total de vuelos vs porcentaje de retrasos por periodo de salida (con umbrales)",
    xaxis=dict(title="Periodo de salida"),
    yaxis=dict(title="Total de vuelos", side="left", showgrid=False),
    yaxis2=dict(title="% Retrasos", side="right", overlaying="y", showgrid=False),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    bargap=0.3,
    template="plotly_white",
    margin=dict(l=60, r=60, t=60, b=40)
)

fig.show()


# %% [markdown]
# áfico combinado de barras + línea + puntos para que los umbrales de color se definan así:
# 
# 🟩 Verde → < 15% de retrasos
# 
# 🟨 Amarillo → entre 15% y 25%
# 
# 🟥 Rojo → > 25%
# 
# Y además añadiremos bandas de color de fondo (relleno) en el eje de porcentaje para que sea visualmente claro en qué rango se encuentra cada valor.
# 
# variable PERIODO_SALIDA

# %%
# import numpy as np
# import plotly.graph_objects as go

# --- Umbrales de color para los puntos ---
pct = resumen_stack["Porc_Retrasado"].values
colors = np.where(
    pct < 15, "green",
    np.where(pct <= 25, "gold", "red")
)

# --- Figura combinada ---
fig = go.Figure()

# === A) Barras: total de vuelos (eje Y izquierdo) ===
fig.add_trace(go.Bar(
    x=resumen_stack["PERIODO_SALIDA"],
    y=resumen_stack["Total"],
    name="Total de vuelos",
    marker_color="lightblue",
    text=resumen_stack["Total"].apply(lambda x: f"{x:,}"),
    textposition="outside",
    yaxis="y1",
    hovertemplate="<b>%{x}</b><br>Total de vuelos: %{y:,}<extra></extra>"
))

# === B) Línea + puntos: % de retrasos (eje Y derecho, color por umbral) ===
fig.add_trace(go.Scatter(
    x=resumen_stack["PERIODO_SALIDA"],
    y=resumen_stack["Porc_Retrasado"],
    name="% Retrasos",
    mode="lines+markers+text",
    line=dict(color="rgba(0,0,0,0.4)", width=2),
    marker=dict(size=12, color=colors, line=dict(color="black", width=1)),
    text=[f"{v:.2f}%" for v in resumen_stack["Porc_Retrasado"]],
    textposition="top center",
    yaxis="y2",
    hovertemplate="<b>%{x}</b><br>% Retrasos: %{y:.2f}%<extra></extra>"
))

# === C) Bandas de color en el eje de % retrasos ===
# Se añaden como shapes sobre el fondo
fig.update_layout(
    shapes=[
        # Verde: <15%
        dict(
            type="rect", xref="paper", x0=0, x1=1, yref="y2",
            y0=0, y1=15, fillcolor="rgba(0,255,0,0.12)", line_width=0,
            layer="below"
        ),
        # Amarillo: 15%–25%
        dict(
            type="rect", xref="paper", x0=0, x1=1, yref="y2",
            y0=15, y1=25, fillcolor="rgba(255,255,0,0.15)", line_width=0,
            layer="below"
        ),
        # Rojo: >25%
        dict(
            type="rect", xref="paper", x0=0, x1=1, yref="y2",
            y0=25, y1=max(30, resumen_stack["Porc_Retrasado"].max() * 1.2),
            fillcolor="rgba(255,0,0,0.10)", line_width=0,
            layer="below"
        ),
    ]
)

# === D) Leyenda personalizada de umbrales ===
fig.add_trace(go.Scatter(
    x=[None], y=[None], mode="markers",
    marker=dict(size=12, color="green"), name="< 15% retrasos"
))
fig.add_trace(go.Scatter(
    x=[None], y=[None], mode="markers",
    marker=dict(size=12, color="gold"), name="15% – 25% retrasos"
))
fig.add_trace(go.Scatter(
    x=[None], y=[None], mode="markers",
    marker=dict(size=12, color="red"), name="> 25% retrasos"
))

# === E) Configuración general ===
fig.update_layout(
    title="Total de vuelos vs porcentaje de retrasos por periodo de salida (con bandas y umbrales)",
    xaxis=dict(title="Periodo de salida"),
    yaxis=dict(title="Total de vuelos", side="left", showgrid=False),
    yaxis2=dict(
        title="% Retrasos",
        side="right",
        overlaying="y",
        showgrid=False,
        range=[0, max(30, resumen_stack['Porc_Retrasado'].max() * 1.2)]
    ),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    bargap=0.3,
    template="plotly_white",
    margin=dict(l=60, r=60, t=70, b=40)
)

fig.show()


# %% [markdown]
# versión final del gráfico combinado con:
# 
# ✅ Umbrales de color (verde <15%, amarillo 15–25%, rojo >25%)
# 
# ✅ Bandas de color en el eje derecho (% retrasos)
# 
# ✅ Título dinámico, que incluye automáticamente:
# 
# 
# El total de vuelos analizados
# 
# El promedio global de retrasos (%) en el dataset

# %%
# import numpy as np
# import plotly.graph_objects as go

# --- A) Cálculos generales para el título dinámico ---
total_vuelos = resumen_stack["Total"].sum()
promedio_retrasos = (resumen_stack["Retrasado"].sum() / total_vuelos) * 100

# --- B) Colores por umbral de % retrasos ---
pct = resumen_stack["Porc_Retrasado"].values
colors = np.where(
    pct < 15, "green",
    np.where(pct <= 25, "gold", "red")
)

# --- C) Figura combinada ---
fig = go.Figure()

# Barras: total de vuelos (eje izquierdo)
fig.add_trace(go.Bar(
    x=resumen_stack["PERIODO_SALIDA"],
    y=resumen_stack["Total"],
    name="Total de vuelos",
    marker_color="lightblue",
    text=resumen_stack["Total"].apply(lambda x: f"{x:,}"),
    textposition="outside",
    yaxis="y1",
    hovertemplate="<b>%{x}</b><br>Total de vuelos: %{y:,}<extra></extra>"
))

# Línea + puntos: % de retrasos (eje derecho)
fig.add_trace(go.Scatter(
    x=resumen_stack["PERIODO_SALIDA"],
    y=resumen_stack["Porc_Retrasado"],
    name="% Retrasos",
    mode="lines+markers+text",
    line=dict(color="rgba(0,0,0,0.4)", width=2),
    marker=dict(size=12, color=colors, line=dict(color="black", width=1)),
    text=[f"{v:.2f}%" for v in resumen_stack["Porc_Retrasado"]],
    textposition="top center",
    yaxis="y2",
    hovertemplate="<b>%{x}</b><br>% Retrasos: %{y:.2f}%<extra></extra>"
))

# --- D) Bandas de color en eje derecho (% retrasos) ---
fig.update_layout(
    shapes=[
        dict(type="rect", xref="paper", x0=0, x1=1, yref="y2",
             y0=0, y1=15, fillcolor="rgba(0,255,0,0.12)", line_width=0, layer="below"),
        dict(type="rect", xref="paper", x0=0, x1=1, yref="y2",
             y0=15, y1=25, fillcolor="rgba(255,255,0,0.15)", line_width=0, layer="below"),
        dict(type="rect", xref="paper", x0=0, x1=1, yref="y2",
             y0=25, y1=max(30, resumen_stack["Porc_Retrasado"].max() * 1.2),
             fillcolor="rgba(255,0,0,0.10)", line_width=0, layer="below"),
    ]
)

# --- E) Leyenda de umbrales ---
fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                         marker=dict(size=12, color="green"), name="< 15% retrasos"))
fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                         marker=dict(size=12, color="gold"), name="15% – 25% retrasos"))
fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                         marker=dict(size=12, color="red"), name="> 25% retrasos"))

# --- F) Configuración y título dinámico ---
fig.update_layout(
    title=(
        f"Total de vuelos vs porcentaje de retrasos por periodo de salida<br>"
        f"<sup>✈️ Total analizado: {total_vuelos:,} vuelos | Promedio global de retrasos: {promedio_retrasos:.2f}%</sup>"
    ),
    xaxis=dict(title="Periodo de salida"),
    yaxis=dict(title="Total de vuelos", side="left", showgrid=False),
    yaxis2=dict(
        title="% Retrasos",
        side="right",
        overlaying="y",
        showgrid=False,
        range=[0, max(30, resumen_stack['Porc_Retrasado'].max() * 1.2)]
    ),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    bargap=0.3,
    template="plotly_white",
    margin=dict(l=60, r=60, t=80, b=40)
)

fig.show()


# %% [markdown]
# gráfico de tendencia mensual que muestra cómo varía el porcentaje de vuelos retrasados a lo largo del año — ideal para identificar patrones estacionales o meses críticos 📈.
# 
# Vamos a mantener el mismo estilo profesional que los anteriores:
# ✅ Línea de tendencia de % de retrasos
# 
# ✅ Puntos coloreados por umbral (verde <15%, amarillo 15–25%, rojo >25%)
# 
# ✅ Bandas de color de fondo (zonas de desempeño)
# 
# ✅ Título dinámico con totales y promedio anual
# 

# %%
# import pandas as pd
# import numpy as np
# import plotly.graph_objects as go

# --- A) Asegurar que exista columna 'MONTH' ---
# Si aún no la tienes:
if "MONTH" not in v.columns:
    v["MONTH"] = pd.to_datetime(v["FL_DATE"]).dt.month

# --- B) Resumen mensual de retrasos ---
resumen_mes = (
    v.groupby("MONTH", observed=True)["RETRASADO_LLEGADA"]
    .agg(Total="size", Retrasados="sum")
    .reset_index()
)
resumen_mes["Porc_Retrasado"] = resumen_mes["Retrasados"] / resumen_mes["Total"] * 100

# Nombres de meses
meses = [
    "Enero","Febrero","Marzo","Abril","Mayo","Junio",
    "Julio","Agosto","Septiembre","Octubre","Noviembre","Diciembre"
]
resumen_mes["MES_NOMBRE"] = resumen_mes["MONTH"].apply(lambda m: meses[m-1])

# --- C) Calcular totales globales para el título dinámico ---
total_vuelos_anual = resumen_mes["Total"].sum()
promedio_retrasos_anual = (resumen_mes["Retrasados"].sum() / total_vuelos_anual) * 100

# --- D) Colores según umbrales de % retrasos ---
pct = resumen_mes["Porc_Retrasado"].values
colors = np.where(
    pct < 15, "green",
    np.where(pct <= 25, "gold", "red")
)

# --- E) Figura de tendencia mensual ---
fig = go.Figure()

# Línea + puntos (% retrasos)
fig.add_trace(go.Scatter(
    x=resumen_mes["MES_NOMBRE"],
    y=resumen_mes["Porc_Retrasado"],
    name="% Retrasos",
    mode="lines+markers+text",
    line=dict(color="rgba(0,0,0,0.4)", width=2),
    marker=dict(size=12, color=colors, line=dict(color="black", width=1)),
    text=[f"{v:.2f}%" for v in resumen_mes["Porc_Retrasado"]],
    textposition="top center",
    hovertemplate="<b>%{x}</b><br>% Retrasos: %{y:.2f}%<extra></extra>"
))

# Bandas de color (zonas de desempeño)
fig.update_layout(
    shapes=[
        # Verde (<15%)
        dict(type="rect", xref="paper", x0=0, x1=1, yref="y",
             y0=0, y1=15, fillcolor="rgba(0,255,0,0.12)", line_width=0, layer="below"),
        # Amarillo (15-25%)
        dict(type="rect", xref="paper", x0=0, x1=1, yref="y",
             y0=15, y1=25, fillcolor="rgba(255,255,0,0.15)", line_width=0, layer="below"),
        # Rojo (>25%)
        dict(type="rect", xref="paper", x0=0, x1=1, yref="y",
             y0=25, y1=max(30, resumen_mes["Porc_Retrasado"].max() * 1.2),
             fillcolor="rgba(255,0,0,0.10)", line_width=0, layer="below"),
    ]
)

# Leyenda de umbrales
fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                         marker=dict(size=12, color="green"), name="< 15% retrasos"))
fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                         marker=dict(size=12, color="gold"), name="15% – 25% retrasos"))
fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                         marker=dict(size=12, color="red"), name="> 25% retrasos"))

# --- F) Configuración general ---
fig.update_layout(
    title=(
        f"Tendencia mensual del porcentaje de vuelos retrasados<br>"
        f"<sup>✈️ Total anual: {total_vuelos_anual:,} vuelos | Promedio anual de retrasos: {promedio_retrasos_anual:.2f}%</sup>"
    ),
    xaxis=dict(title="Mes del año"),
    yaxis=dict(title="% Retrasos", range=[0, max(30, resumen_mes["Porc_Retrasado"].max() * 1.2)]),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    template="plotly_white",
    margin=dict(l=60, r=60, t=80, b=60)
)

fig.show()


# %% [markdown]
# gráfico combinado mensual:
# 
# Barras azules = total de vuelos por mes
# 
# Línea con puntos = % de retrasos (puntos verdes <15%, amarillos 15–25%, rojos >25%)
# 
# Bandas de fondo con los mismos umbrales
# 
# Título dinámico con total anual y promedio general de retrasos
# 
# Asume que tu DataFrame se llama flights y que la variable binaria de retraso es RETRASADO_LLEGADA (0/1). Si no tienes MONTH, el código la crea desde FL_DATE.

# %% [markdown]
# ### **Exploraciones clave (columnas importantes)**

# %%
# import pandas as pd
# import numpy as np
# import plotly.graph_objects as go

# --- A) Asegurar columna MONTH (1-12) ---
if "MONTH" not in v.columns:
    # Intenta derivar desde fecha; ajusta el nombre de la columna si tu fecha difiere
    if "FL_DATE" in v.columns:
        v["MONTH"] = pd.to_datetime(v["FL_DATE"], errors="coerce").dt.month
    else:
        raise ValueError("No existe la columna MONTH ni FL_DATE para derivarla. Crea MONTH antes de continuar.")

# --- B) Resumen mensual: Totales y % retrasos ---
resumen_mes = (
    v.groupby("MONTH", observed=True)["RETRASADO_LLEGADA"]
    .agg(Total="size", Retrasados="sum")
    .reset_index()
    .sort_values("MONTH")
)

resumen_mes["Porc_Retrasado"] = (resumen_mes["Retrasados"] / resumen_mes["Total"]) * 100
meses = ["Enero","Febrero","Marzo","Abril","Mayo","Junio","Julio","Agosto","Septiembre","Octubre","Noviembre","Diciembre"]
resumen_mes["MES_NOMBRE"] = resumen_mes["MONTH"].apply(lambda m: meses[m-1] if pd.notna(m) else "Desconocido")

# --- C) Título dinámico ---
total_vuelos_anual = resumen_mes["Total"].sum()
promedio_retrasos_anual = (resumen_mes["Retrasados"].sum() / total_vuelos_anual) * 100

# --- D) Colores por umbral: verde <15%, amarillo 15–25%, rojo >25% ---
pct = resumen_mes["Porc_Retrasado"].values
colors = np.where(pct < 15, "green", np.where(pct <= 25, "gold", "red"))

# --- E) Figura combinada (barras + línea) ---
fig = go.Figure()

# Barras: Total de vuelos (eje izquierdo)
fig.add_trace(go.Bar(
    x=resumen_mes["MES_NOMBRE"],
    y=resumen_mes["Total"],
    name="Total de vuelos",
    marker_color="lightblue",
    text=resumen_mes["Total"].apply(lambda x: f"{x:,}"),
    textposition="outside",
    yaxis="y1",
    hovertemplate="<b>%{x}</b><br>Total de vuelos: %{y:,}<extra></extra>"
))

# Línea + puntos: % Retrasos (eje derecho)
fig.add_trace(go.Scatter(
    x=resumen_mes["MES_NOMBRE"],
    y=resumen_mes["Porc_Retrasado"],
    name="% Retrasos",
    mode="lines+markers+text",
    line=dict(color="rgba(0,0,0,0.4)", width=2),
    marker=dict(size=12, color=colors, line=dict(color="black", width=1)),
    text=[f"{v:.2f}%" for v in resumen_mes["Porc_Retrasado"]],
    textposition="top center",
    yaxis="y2",
    hovertemplate="<b>%{x}</b><br>% Retrasos: %{y:.2f}%<extra></extra>"
))

# --- F) Bandas de fondo por umbral en eje de % (derecho) ---
y2_max = max(30, float(resumen_mes["Porc_Retrasado"].max() * 1.2))
fig.update_layout(
    shapes=[
        # Verde: <15%
        dict(type="rect", xref="paper", x0=0, x1=1, yref="y2",
             y0=0, y1=15, fillcolor="rgba(0,255,0,0.12)", line_width=0, layer="below"),
        # Amarillo: 15–25%
        dict(type="rect", xref="paper", x0=0, x1=1, yref="y2",
             y0=15, y1=25, fillcolor="rgba(255,255,0,0.15)", line_width=0, layer="below"),
        # Rojo: >25%
        dict(type="rect", xref="paper", x0=0, x1=1, yref="y2",
             y0=25, y1=y2_max, fillcolor="rgba(255,0,0,0.10)", line_width=0, layer="below"),
    ]
)

# --- G) Leyenda de umbrales (trazas dummy) ---
fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                         marker=dict(size=12, color="green"), name="< 15% retrasos"))
fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                         marker=dict(size=12, color="gold"), name="15% – 25% retrasos"))
fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                         marker=dict(size=12, color="red"), name="> 25% retrasos"))

# --- H) Layout final ---
fig.update_layout(
    title=(
        "Tendencia mensual: Total de vuelos vs % de retrasos<br>"
        f"<sup>✈️ Total anual: {total_vuelos_anual:,} vuelos | Promedio anual de retrasos: {promedio_retrasos_anual:.2f}%</sup>"
    ),
    xaxis=dict(title="Mes"),
    yaxis=dict(title="Total de vuelos", side="left", showgrid=False),
    yaxis2=dict(title="% Retrasos", side="right", overlaying="y", showgrid=False, range=[0, y2_max]),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    bargap=0.3,
    template="plotly_white",
    margin=dict(l=60, r=60, t=90, b=50)
)

fig.show()

# (Opcional) Guardar HTML interactivo:
# fig.write_html("tendencia_mensual_total_vs_pct_retrasos.html")


# %% [markdown]
# gráfico combinado mensual añadiendo una línea horizontal de referencia en el eje de porcentaje de retrasos.
# Esta línea marca el promedio anual de retrasos y te permitirá identificar fácilmente qué meses están por encima o por debajo del promedio.
# 
# Además, mantendremos:
# 
# Barras (total de vuelos)
# 
# Línea con puntos coloreados por umbrales
# 
# Bandas de color de fondo (verde <15%, amarillo 15–25%, rojo >25%)
# 
# Título dinámico con totales y promedio global
# 
# | Elemento               | Descripción                                                      |
# | ---------------------- | ---------------------------------------------------------------- |
# | 🟦 Barras azules       | Representan el total de vuelos por mes                           |
# | 🔴 Línea + puntos      | Muestran el porcentaje de vuelos retrasados                      |
# | 🟩🟨🟥 Bandas          | Zonas de desempeño: verde (<15%), amarillo (15–25%), rojo (>25%) |
# | 🔵 Línea punteada azul | Promedio anual de % de retrasos                                  |
# | 📊 Tooltip interactivo | Muestra totales y porcentajes al pasar el mouse                  |
# | 🧭 Doble eje Y         | Izquierdo = cantidad de vuelos / Derecho = % retrasos            |
# 

# %%
# import pandas as pd
# import numpy as np
# import plotly.graph_objects as go

# --- A) Asegurar columna MONTH (1-12) ---
if "MONTH" not in v.columns:
    if "FL_DATE" in v.columns:
        v["MONTH"] = pd.to_datetime(v["FL_DATE"], errors="coerce").dt.month
    else:
        raise ValueError("No existe la columna MONTH ni FL_DATE para derivarla.")

# --- B) Resumen mensual: totales y % de retrasos ---
resumen_mes = (
    v.groupby("MONTH", observed=True)["RETRASADO_LLEGADA"]
    .agg(Total="size", Retrasados="sum")
    .reset_index()
    .sort_values("MONTH")
)

resumen_mes["Porc_Retrasado"] = (resumen_mes["Retrasados"] / resumen_mes["Total"]) * 100
meses = ["Enero","Febrero","Marzo","Abril","Mayo","Junio","Julio","Agosto","Septiembre","Octubre","Noviembre","Diciembre"]
resumen_mes["MES_NOMBRE"] = resumen_mes["MONTH"].apply(lambda m: meses[m-1])

# --- C) Totales globales para título dinámico ---
total_vuelos_anual = resumen_mes["Total"].sum()
promedio_retrasos_anual = (resumen_mes["Retrasados"].sum() / total_vuelos_anual) * 100

# --- D) Colores por umbral ---
pct = resumen_mes["Porc_Retrasado"].values
colors = np.where(pct < 15, "green", np.where(pct <= 25, "gold", "red"))

# --- E) Figura combinada ---
fig = go.Figure()

# Barras = total de vuelos
fig.add_trace(go.Bar(
    x=resumen_mes["MES_NOMBRE"],
    y=resumen_mes["Total"],
    name="Total de vuelos",
    marker_color="lightblue",
    text=resumen_mes["Total"].apply(lambda x: f"{x:,}"),
    textposition="outside",
    yaxis="y1",
    hovertemplate="<b>%{x}</b><br>Total de vuelos: %{y:,}<extra></extra>"
))

# Línea = % de retrasos (puntos con colores por umbral)
fig.add_trace(go.Scatter(
    x=resumen_mes["MES_NOMBRE"],
    y=resumen_mes["Porc_Retrasado"],
    name="% Retrasos",
    mode="lines+markers+text",
    line=dict(color="rgba(0,0,0,0.4)", width=2),
    marker=dict(size=12, color=colors, line=dict(color="black", width=1)),
    text=[f"{v:.2f}%" for v in resumen_mes["Porc_Retrasado"]],
    textposition="top center",
    yaxis="y2",
    hovertemplate="<b>%{x}</b><br>% Retrasos: %{y:.2f}%<extra></extra>"
))

# --- F) Bandas de color de fondo (zonas de desempeño) ---
y2_max = max(30, float(resumen_mes["Porc_Retrasado"].max() * 1.2))
fig.update_layout(
    shapes=[
        dict(type="rect", xref="paper", x0=0, x1=1, yref="y2",
             y0=0, y1=15, fillcolor="rgba(0,255,0,0.12)", line_width=0, layer="below"),
        dict(type="rect", xref="paper", x0=0, x1=1, yref="y2",
             y0=15, y1=25, fillcolor="rgba(255,255,0,0.15)", line_width=0, layer="below"),
        dict(type="rect", xref="paper", x0=0, x1=1, yref="y2",
             y0=25, y1=y2_max, fillcolor="rgba(255,0,0,0.10)", line_width=0, layer="below"),
        # Línea horizontal de promedio anual
        dict(
            type="line", xref="paper", x0=0, x1=1, yref="y2",
            y0=promedio_retrasos_anual, y1=promedio_retrasos_anual,
            line=dict(color="blue", width=2, dash="dot")
        ),
    ]
)

# --- G) Leyenda personalizada de umbrales + línea de referencia ---
fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                         marker=dict(size=12, color="green"), name="< 15% retrasos"))
fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                         marker=dict(size=12, color="gold"), name="15% – 25% retrasos"))
fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                         marker=dict(size=12, color="red"), name="> 25% retrasos"))
fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
                         line=dict(color="blue", width=2, dash="dot"),
                         name=f"Promedio anual ({promedio_retrasos_anual:.2f}%)"))

# --- H) Layout final ---
fig.update_layout(
    title=(
        "Tendencia mensual: Total de vuelos vs % de retrasos<br>"
        f"<sup>✈️ Total anual: {total_vuelos_anual:,} vuelos | Promedio anual de retrasos: {promedio_retrasos_anual:.2f}%</sup>"
    ),
    xaxis=dict(title="Mes"),
    yaxis=dict(title="Total de vuelos", side="left", showgrid=False),
    yaxis2=dict(title="% Retrasos", side="right", overlaying="y", showgrid=False, range=[0, y2_max]),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    bargap=0.3,
    template="plotly_white",
    margin=dict(l=60, r=60, t=90, b=50)
)

fig.show()

# (Opcional)
# fig.write_html("tendencia_mensual_vuelos_vs_pct_retrasos.html")


# %% [markdown]
# ### **revisar si en los retrasos influye la distancia**

# %% [markdown]
# 1. Correlación simple (lineal y monótona)
# 

# %%
# import numpy as np
# Lectura:
# Pearson ≈ 0 sugiere relación lineal débil.
# Spearman > Pearson sugiere relación no lineal/monótona.

df = v[["DISTANCE","DEPARTURE_DELAY","ARRIVAL_DELAY"]].dropna()
pearson = df["DISTANCE"].corr(df["ARRIVAL_DELAY"], method="pearson")
spearman = df["DISTANCE"].corr(df["ARRIVAL_DELAY"], method="spearman")

print(f"Correlación Pearson (lineal):  {pearson:.3f}")
print(f"Correlación Spearman (monótona): {spearman:.3f}")


# %% [markdown]
# Dispersión + tendencia (muestra)

# %%
# import matplotlib.pyplot as plt
ms = v[["DISTANCE","ARRIVAL_DELAY"]].dropna().sample(min(20000, len(v)), random_state=42)

plt.figure(figsize=(7,5))
plt.scatter(ms["DISTANCE"], ms["ARRIVAL_DELAY"], alpha=0.15, s=8)
plt.xlabel("Distancia (millas)")
plt.ylabel("Retraso en llegada (min)")
plt.title("DISTANCE vs ARRIVAL_DELAY (muestra)")

# Recta de tendencia lineal rápida
m, b = np.polyfit(ms["DISTANCE"], ms["ARRIVAL_DELAY"], 1)
xs = np.linspace(ms["DISTANCE"].min(), ms["DISTANCE"].max(), 50)
plt.plot(xs, m*xs + b, linewidth=2)
plt.show()


# %% [markdown]
# ¿Distancia “explica” retraso controlando por hora/mes/ruta?
# 
# (chequeo rápido con regresión lineal; para clasificación usarás luego LightGBM)

# %%
# !pip install scikit-learn


# %%
# import sklearn
# print(sklearn.__version__)


# %%
# import pandas as pd
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split

tmp = v[["ARRIVAL_DELAY","DISTANCE","HORA_SALIDA","DAY_OF_WEEK","MONTH","ORIGIN_AIRPORT","DESTINATION_AIRPORT"]].dropna().copy()

# One-hot minimal para aeropuertos (muchas categorías; tomamos las top para ejemplo)
top_origen = tmp["ORIGIN_AIRPORT"].value_counts().nlargest(20).index
top_dest   = tmp["DESTINATION_AIRPORT"].value_counts().nlargest(20).index
tmp["ORIGIN_TOP"] = np.where(tmp["ORIGIN_AIRPORT"].isin(top_origen), tmp["ORIGIN_AIRPORT"], "OTROS")
tmp["DEST_TOP"]   = np.where(tmp["DESTINATION_AIRPORT"].isin(top_dest), tmp["DESTINATION_AIRPORT"], "OTROS")

X = pd.get_dummies(tmp[["DISTANCE","HORA_SALIDA","DAY_OF_WEEK","MONTH","ORIGIN_TOP","DEST_TOP"]], drop_first=True)
y = tmp["ARRIVAL_DELAY"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

lr = LinearRegression()
lr.fit(X_train, y_train)

coef = pd.Series(lr.coef_, index=X.columns).sort_values(ascending=False)
print("Top coeficientes:\n", coef.head(10))
print("\nR² (test):", lr.score(X_test, y_test))
print("\nCoeficiente de DISTANCE:", coef.get("DISTANCE", np.nan))


# %% [markdown]
# Promedio de retraso por “tramos” de distancia

# %%
# Cuantiles para armar bins balanceados por cantidad de vuelos
bins = np.quantile(v["DISTANCE"].dropna(), [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1])
labels = [f"Q{i}-{i+1}" for i in range(1,11)]
v["DIST_BIN"] = pd.cut(v["DISTANCE"], bins=bins, include_lowest=True, labels=labels)

ag = v.groupby("DIST_BIN").agg(
    vuelos=("ARRIVAL_DELAY", "size"),
    retraso_prom=("ARRIVAL_DELAY","mean"),
    porc_retrasados=("RETRASADO_LLEGADA","mean")
).reset_index()

print(ag)

# Gráfico 1: retraso promedio por tramo
plt.figure(figsize=(8,4))
plt.plot(ag["DIST_BIN"], ag["retraso_prom"], marker="o")
plt.title("Retraso promedio de llegada por tramo de distancia")
plt.xlabel("Tramo de distancia (cuantiles)")
plt.ylabel("Minutos promedio de retraso")
plt.grid(True, alpha=0.3)
plt.show()

# Gráfico 2: % de vuelos retrasados (>15 min) por tramo
plt.figure(figsize=(8,4))
plt.plot(ag["DIST_BIN"], ag["porc_retrasados"]*100, marker="o")
plt.title("% de vuelos retrasados por tramo de distancia")
plt.xlabel("Tramo de distancia (cuantiles)")
plt.ylabel("% retrasados")
plt.grid(True, alpha=0.3)
plt.show()


# %% [markdown]
# ¿Distancia “explica” retraso controlando por hora/mes/ruta?
# 
# (chequeo rápido con regresión lineal; para clasificación usarás luego LightGBM)

# %%
# import pandas as pd
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split

tmp = v[["ARRIVAL_DELAY","DISTANCE","HORA_SALIDA","DAY_OF_WEEK","MONTH","ORIGIN_AIRPORT","DESTINATION_AIRPORT"]].dropna().copy()

# One-hot minimal para aeropuertos (muchas categorías; tomamos las top para ejemplo)
top_origen = tmp["ORIGIN_AIRPORT"].value_counts().nlargest(20).index
top_dest   = tmp["DESTINATION_AIRPORT"].value_counts().nlargest(20).index
tmp["ORIGIN_TOP"] = np.where(tmp["ORIGIN_AIRPORT"].isin(top_origen), tmp["ORIGIN_AIRPORT"], "OTROS")
tmp["DEST_TOP"]   = np.where(tmp["DESTINATION_AIRPORT"].isin(top_dest), tmp["DESTINATION_AIRPORT"], "OTROS")

X = pd.get_dummies(tmp[["DISTANCE","HORA_SALIDA","DAY_OF_WEEK","MONTH","ORIGIN_TOP","DEST_TOP"]], drop_first=True)
y = tmp["ARRIVAL_DELAY"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

lr = LinearRegression()
lr.fit(X_train, y_train)

coef = pd.Series(lr.coef_, index=X.columns).sort_values(ascending=False)
print("Top coeficientes:\n", coef.head(10))
print("\nR² (test):", lr.score(X_test, y_test))
print("\nCoeficiente de DISTANCE:", coef.get("DISTANCE", np.nan))


# %% [markdown]
# Proporción de retrasos por buckets de distancia (binaria)
# 
# (útil para el caso de negocio)

# %%
# si no creaste RETRASADO_LLEGADA:
v["RETRASADO_LLEGADA"] = (v["ARRIVAL_DELAY"] > 15).astype(int)

tbl = v.groupby("DIST_BIN")["RETRASADO_LLEGADA"].agg(["mean","size"]).reset_index()
tbl["mean"] = tbl["mean"]*100
print(tbl)

plt.figure(figsize=(8,4))
plt.bar(tbl["DIST_BIN"], tbl["mean"])
plt.title("% de retrasados (>15 min) por tramo de distancia")
plt.xlabel("Tramo de distancia (cuantiles)")
plt.ylabel("% retrasados")
plt.show()


# %% [markdown]
# Hexbin (densidad) para evitar overplot

# %%
plt.figure(figsize=(7,5))
plt.hexbin(v["DISTANCE"], v["ARRIVAL_DELAY"], gridsize=60, cmap="viridis", mincnt=5)
plt.colorbar(label="Frecuencia")
plt.xlabel("Distancia (millas)")
plt.ylabel("Retraso en llegada (min)")
plt.title("Densidad DISTANCE vs ARRIVAL_DELAY")
plt.show()


# %%
v.head(10000)

# %%
v.info()

# %% [markdown]
# Aerolínea

# %%
porc_retraso_x_linea = v.groupby("AIRLINE_NAME")["RETRASADO_LLEGADA"].mean().sort_values(ascending=False)
porc_retraso_x_linea*100


# %%
porc_retraso_x_linea = v.groupby("AIRLINE_NAME")["RETRASADO_LLEGADA"].mean().sort_values(ascending=False)
plt.figure(figsize=(8,6)); sns.barplot(x=porc_retraso_x_linea.values, y=porc_retraso_x_linea.index, palette="Reds_r")
plt.title("Porcentaje de vuelos con retraso (>15 min) por Aerolínea"); plt.xlabel("Proporción"); plt.ylabel("Aerolínea")
plt.show()


# %% [markdown]
# Aeropuerto de origen (TOP 20 por % retraso)
# 

# %%
top_origen = (v.groupby(["ORIGIN_AIRPORT","ORIGEN_AEROPUERTO"])["RETRASADO_LLEGADA"]
                .mean().sort_values(ascending=False).head(20))
top_origen*100

# %%
top_origen = (v.groupby(["ORIGIN_AIRPORT","ORIGEN_AEROPUERTO"])["RETRASADO_LLEGADA"]
                .mean().sort_values(ascending=False).head(20))
top_origen.plot(kind="barh", figsize=(8,7), color="steelblue")
plt.title("Top 20 ORIGIN con mayor % de retraso en llegada"); plt.xlabel("Proporción retrasados")
plt.show()


# %% [markdown]
# Hora y periodo

# %%
sns.barplot(data=v, x="HORA_SALIDA", y="RETRASADO_LLEGADA", palette="coolwarm")
plt.title("Retrasos por hora programada de salida"); plt.ylabel("Proporción retrasados"); plt.show()

sns.barplot(data=v, x="PERIODO_SALIDA", y="RETRASADO_LLEGADA", order=["Madrugada","Mañana","Tarde","Noche"])
plt.title("Retrasos por período del día"); plt.ylabel("Proporción"); plt.show()


# %% [markdown]
# dos gráficos mejorados con Seaborn/Matplotlib que:
# 
# muestran la proporción (en %) directamente encima de cada barra
# 
# usan paleta verde→rojo (verde = menor % de retrasos, rojo = mayor)
# 
# están ordenados correctamente
# 
# evitan el cálculo de CIs para ir más rápido (ci=None)
# 
# Asumen que v tiene RETRASADO_LLEGADA (0/1), HORA_SALIDA (0–23) y PERIODO_SALIDA (Madrugada, Mañana, Tarde, Noche).

# %%
# import numpy as np
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# --- A) Sanitizar datos mínimos ---
# Asegurar binaria numérica (0/1)
v["RETRASADO_LLEGADA"] = pd.to_numeric(v["RETRASADO_LLEGADA"], errors="coerce").fillna(0).astype(int)

# Asegurar hora entera 0..23
v["HORA_SALIDA"] = pd.to_numeric(v["HORA_SALIDA"], errors="coerce").astype("Int64")
v = v[v["HORA_SALIDA"].between(0, 23)]

# Orden fijo para periodo
orden_periodo = ["Madrugada", "Mañana", "Tarde", "Noche"]
if "PERIODO_SALIDA" in v.columns:
    v["PERIODO_SALIDA"] = pd.Categorical(v["PERIODO_SALIDA"], categories=orden_periodo, ordered=True)

# --- Helper para anotar porcentaje (y opcionalmente n por categoría) ---
def anota_porcentaje(ax, counts=None, factor=100, fmt_dec=2):
    """
    Anota % encima de cada barra. Si pasas 'counts' (dict {x: n}),
    añade 'n=...' al texto.
    """
    for p in ax.patches:
        altura = p.get_height()
        x_centro = p.get_x() + p.get_width() / 2
        etiqueta = f"{altura*factor:.{fmt_dec}f}%"
        if counts is not None:
            # Para recuperar la categoría del eje x:
            cat = p.get_x() + p.get_width()/2
            # Mejor tomar el tick label correspondiente:
            # Usamos la posición del patch para encontrar el índice más cercano
            # (si el mapeo fuese complejo, una alternativa: precomputar con groupby)
        ax.annotate(
            etiqueta if counts is None else f"{etiqueta}",
            (x_centro, altura),
            ha="center", va="bottom", fontsize=10, fontweight="bold",
            xytext=(0, max(altura*0.02, 0.01)), textcoords="offset points"
        )

# ---- 1) Retrasos por HORA_SALIDA (0–23) ----
plt.figure(figsize=(10, 4.5))
ax = sns.barplot(
    data=v.sort_values("HORA_SALIDA"),
    x="HORA_SALIDA",
    y="RETRASADO_LLEGADA",
    ci=None,
    palette="RdYlGn_r"  # verde = mejor (menos retrasos), rojo = peor
)
ax.set_title("Retrasos por hora programada de salida")
ax.set_xlabel("Hora de salida")
ax.set_ylabel("Proporción de retrasos (%)")

# Formatear eje Y a %
vals = ax.get_yticks()
ax.set_yticklabels([f"{x*100:.0f}%" for x in vals])

# Anotar %
anota_porcentaje(ax, counts=None, factor=100, fmt_dec=2)

plt.tight_layout()
plt.show()

# ---- 2) Retrasos por PERIODO_SALIDA (Madrugada, Mañana, Tarde, Noche) ----
plt.figure(figsize=(7.5, 4.5))
ax2 = sns.barplot(
    data=v,
    x="PERIODO_SALIDA",
    y="RETRASADO_LLEGADA",
    order=orden_periodo,
    ci=None,
    palette="RdYlGn_r"
)
ax2.set_title("Retrasos por período del día")
ax2.set_xlabel("Período de salida")
ax2.set_ylabel("Proporción de retrasos (%)")

vals2 = ax2.get_yticks()
ax2.set_yticklabels([f"{x*100:.0f}%" for x in vals2])

anota_porcentaje(ax2, counts=None, factor=100, fmt_dec=2)

plt.tight_layout()
plt.show()


# %% [markdown]
# mejorada de los dos gráficos con Seaborn y Matplotlib, que ahora:
# 
# ✅ Muestran el porcentaje de retrasos encima de cada barra
# ✅ Incluyen también el número total de vuelos (n) en esa categoría
# ✅ Usan la paleta verde→rojo invertida (RdYlGn_r)
# ✅ Mantienen formato limpio y proporcional
# 
# Este código es ideal para tus notebooks de análisis exploratorio (EDA) antes del modelado.

# %%
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# --- A) Preparación de datos ---
# Asegurar que las columnas estén en el formato correcto
v["RETRASADO_LLEGADA"] = pd.to_numeric(v["RETRASADO_LLEGADA"], errors="coerce").fillna(0).astype(int)
v["HORA_SALIDA"] = pd.to_numeric(v["HORA_SALIDA"], errors="coerce").fillna(0).astype(int)

# Orden fijo para los periodos del día
orden_periodo = ["Madrugada", "Mañana", "Tarde", "Noche"]
if "PERIODO_SALIDA" in v.columns:
    v["PERIODO_SALIDA"] = pd.Categorical(v["PERIODO_SALIDA"], categories=orden_periodo, ordered=True)

# --- Helper para anotar porcentaje y total (n) encima de cada barra ---
def anotar_barras(ax, df, x_col, y_col, factor=100):
    """
    Añade etiquetas con porcentaje y cantidad total (n) encima de cada barra.
    """
    # Calcular totales y porcentajes por categoría
    resumen = df.groupby(x_col)[y_col].agg(
        total="count", promedio="mean"
    ).reset_index()

    for p, (_, fila) in zip(ax.patches, resumen.iterrows()):
        altura = p.get_height()
        porcentaje = fila["promedio"] * factor
        total = fila["total"]
        etiqueta = f"{porcentaje:.2f}% | n={total:,}"
        ax.annotate(
            etiqueta,
            (p.get_x() + p.get_width() / 2, altura),
            ha="center", va="bottom",
            fontsize=10, fontweight="bold",
            xytext=(0, 3), textcoords="offset points"
        )

# === 1️⃣ Retrasos por HORA_SALIDA ===
plt.figure(figsize=(10, 4.5))
ax = sns.barplot(
    data=v.sort_values("HORA_SALIDA"),
    x="HORA_SALIDA",
    y="RETRASADO_LLEGADA",
    ci=None,
    palette="RdYlGn_r"
)
ax.set_title("Retrasos por hora programada de salida", fontsize=13, fontweight="bold")
ax.set_xlabel("Hora de salida")
ax.set_ylabel("Proporción de retrasos (%)")

# Formatear eje Y a porcentaje
vals = ax.get_yticks()
ax.set_yticklabels([f"{x*100:.0f}%" for x in vals])

# Añadir porcentaje + n
anotar_barras(ax, v, "HORA_SALIDA", "RETRASADO_LLEGADA")

plt.tight_layout()
plt.show()

# === 2️⃣ Retrasos por PERIODO_SALIDA ===
plt.figure(figsize=(7.5, 4.5))
ax2 = sns.barplot(
    data=v,
    x="PERIODO_SALIDA",
    y="RETRASADO_LLEGADA",
    order=orden_periodo,
    ci=None,
    palette="RdYlGn_r"
)
ax2.set_title("Retrasos por período del día", fontsize=13, fontweight="bold")
ax2.set_xlabel("Período de salida")
ax2.set_ylabel("Proporción de retrasos (%)")

# Formatear eje Y
vals2 = ax2.get_yticks()
ax2.set_yticklabels([f"{x*100:.0f}%" for x in vals2])

# Añadir porcentaje + n
anotar_barras(ax2, v, "PERIODO_SALIDA", "RETRASADO_LLEGADA")

plt.tight_layout()
plt.show()


# %% [markdown]
# con ploty

# %%
# import pandas as pd
# import numpy as np
# import plotly.express as px

# =========================
# A) Copia limpia (no toca v)
# =========================
v_limpio = v.copy()

# Tipado/limpieza sin alterar v
v_limpio["RETRASADO_LLEGADA"] = pd.to_numeric(v_limpio["RETRASADO_LLEGADA"], errors="coerce").fillna(0).astype(int)
v_limpio["HORA_SALIDA"] = pd.to_numeric(v_limpio["HORA_SALIDA"], errors="coerce").astype("Int64")
v_limpio = v_limpio[v_limpio["HORA_SALIDA"].between(0, 23)]

# Si no existe PERIODO_SALIDA, lo creamos desde HORA_SALIDA
if "PERIODO_SALIDA" not in v_limpio.columns:
    v_limpio["PERIODO_SALIDA"] = pd.cut(
        v_limpio["HORA_SALIDA"].astype(int),
        bins=[0, 6, 12, 18, 24],
        labels=["Madrugada", "Mañana", "Tarde", "Noche"],
        right=False
    )

# Orden consistente
orden_periodo = ["Madrugada", "Mañana", "Tarde", "Noche"]
v_limpio["PERIODO_SALIDA"] = pd.Categorical(v_limpio["PERIODO_SALIDA"], categories=orden_periodo, ordered=True)

# ==========================================
# B) Resúmenes: % retrasos y totales (n)
# ==========================================

# Por HORA_SALIDA
res_hora = (v_limpio.groupby("HORA_SALIDA", observed=True)["RETRASADO_LLEGADA"]
            .agg(Total="size", Retrasados="sum").reset_index())
res_hora["Porc_Retrasado"] = res_hora["Retrasados"] / res_hora["Total"] * 100
res_hora = res_hora.sort_values("HORA_SALIDA")

# Por PERIODO_SALIDA
res_periodo = (v_limpio.groupby("PERIODO_SALIDA", observed=True)["RETRASADO_LLEGADA"]
               .agg(Total="size", Retrasados="sum").reset_index())
res_periodo["Porc_Retrasado"] = res_periodo["Retrasados"] / res_periodo["Total"] * 100
res_periodo = res_periodo.sort_values("PERIODO_SALIDA")

# ==========================================
# C) Plotly: Barras interactivas
# ==========================================

# --- 1) Retrasos por HORA_SALIDA ---
fig_hora = px.bar(
    res_hora,
    x="HORA_SALIDA",
    y="Porc_Retrasado",
    text=res_hora["Porc_Retrasado"].apply(lambda x: f"{x:.2f}%"),
    labels={"HORA_SALIDA": "Hora de salida", "Porc_Retrasado": "Retrasos (%)"},
    title="Retrasos por hora programada de salida",
    color="Porc_Retrasado",  # gradiente visual (opcional)
    color_continuous_scale="RdYlGn_r"
)
fig_hora.update_traces(textposition="outside")
fig_hora.update_layout(
    yaxis_title="Retrasos (%)",
    xaxis_title="Hora de salida",
    uniformtext_minsize=10, uniformtext_mode="hide",
    coloraxis_colorbar=dict(title="%")
)
# Tooltip con % y n
fig_hora.update_traces(
    hovertemplate="<b>Hora %{x}</b><br>% Retrasos: %{y:.2f}%<br>n: %{customdata:,}<extra></extra>",
    customdata=np.stack([res_hora["Total"]], axis=-1)
)
fig_hora.show()

# --- 2) Retrasos por PERIODO_SALIDA ---
fig_periodo = px.bar(
    res_periodo,
    x="PERIODO_SALIDA",
    y="Porc_Retrasado",
    text=res_periodo["Porc_Retrasado"].apply(lambda x: f"{x:.2f}%"),
    labels={"PERIODO_SALIDA": "Período de salida", "Porc_Retrasado": "Retrasos (%)"},
    title="Retrasos por período del día",
    color="Porc_Retrasado",
    color_continuous_scale="RdYlGn_r",
    category_orders={"PERIODO_SALIDA": orden_periodo}
)
fig_periodo.update_traces(textposition="outside")
fig_periodo.update_layout(
    yaxis_title="Retrasos (%)",
    xaxis_title="Período de salida",
    uniformtext_minsize=10, uniformtext_mode="hide",
    coloraxis_colorbar=dict(title="%")
)
fig_periodo.update_traces(
    hovertemplate="<b>%{x}</b><br>% Retrasos: %{y:.2f}%<br>n: %{customdata:,}<extra></extra>",
    customdata=np.stack([res_periodo["Total"]], axis=-1)
)
fig_periodo.show()


# %% [markdown]
# revisar valores de las columnas
# v["RETRASADO_LLEGADA"]= y v["HORA_SALIDA"]

# %%
v["RETRASADO_LLEGADA"].value_counts()

# %%
sum(v["RETRASADO_LLEGADA"].value_counts())

# %%
v["HORA_SALIDA"].value_counts().sort_values().sort_index()


# %%
sum(v["HORA_SALIDA"].value_counts())

# %% [markdown]
# retrasos x aerolinea

# %%
# === Copia de seguridad (no modifica v original) ===
v_retrasos = v.copy()

# Asegurar que la columna sea numérica binaria (0/1)
v_retrasos["RETRASADO_LLEGADA"] = pd.to_numeric(
    v_retrasos["RETRASADO_LLEGADA"], errors="coerce"
).fillna(0).astype(int)

# === Agrupación por aerolínea (puedes cambiar a ORIGIN_AIRPORT o DESTINATION_AIRPORT) ===
resumen_retrasos = (
    v_retrasos.groupby("AIRLINE", observed=True)["RETRASADO_LLEGADA"]
    .agg(Total="size", Retrasados="sum")
    .reset_index()
)

# Calcular % retrasados
resumen_retrasos["Porcentaje_Retrasos"] = (
    resumen_retrasos["Retrasados"] / resumen_retrasos["Total"] * 100
)

# Orden descendente por % de retrasos
resumen_retrasos = resumen_retrasos.sort_values("Porcentaje_Retrasos", ascending=False)

# Mostrar resultados
print(resumen_retrasos.head(10))

# %% [markdown]
# | Script                  | Agrupa por             | Muestra                | Ejemplo de uso                       |
# | ----------------------- | ---------------------- | ---------------------- | ------------------------------------ |
# | 1️⃣ `RETRASADO_LLEGADA` | Aerolínea / aeropuerto | % de retrasos          | Ranking de aerolíneas más retrasadas |
# 
# 

# %%
# import pandas as pd

# === Copia de seguridad (no modifica v original) ===
v_retrasos = v.copy()

# Asegurar que la columna sea numérica binaria (0/1)
v_retrasos["RETRASADO_LLEGADA"] = pd.to_numeric(
    v_retrasos["RETRASADO_LLEGADA"], errors="coerce"
).fillna(0).astype(int)

# === Agrupación por aerolínea (puedes cambiar a ORIGIN_AIRPORT o DESTINATION_AIRPORT) ===
resumen_retrasos = (
    v_retrasos.groupby("AIRLINE_NAME", observed=True)["RETRASADO_LLEGADA"]
    .agg(Total="size", Retrasados="sum")
    .reset_index()
)

# Calcular % retrasados
resumen_retrasos["Porcentaje_Retrasos"] = (
    resumen_retrasos["Retrasados"] / resumen_retrasos["Total"] * 100
)

# Orden descendente por % de retrasos
resumen_retrasos = resumen_retrasos.sort_values("Porcentaje_Retrasos", ascending=False)

# Mostrar resultados
print(resumen_retrasos.head(10))

# (Opcional) Gráfico interactivo con Plotly
import plotly.express as px

fig = px.bar(
    resumen_retrasos,
    x="AIRLINE_NAME",
    y="Porcentaje_Retrasos",
    text=resumen_retrasos["Porcentaje_Retrasos"].apply(lambda x: f"{x:.2f}%"),
    title="Porcentaje de vuelos retrasados por aerolínea",
    labels={"AIRLINE_NAME": "Aerolínea", "Porcentaje_Retrasos": "Retrasos (%)"},
    color="Porcentaje_Retrasos",
    color_continuous_scale="RdYlGn_r"
)
fig.update_traces(textposition="outside")
fig.show()


# %%
# import pandas as pd
# import plotly.express as px

# === Copia de seguridad (no modifica v original) ===
v_retrasos = v.copy()

# Asegurar que la columna sea numérica binaria (0/1)
v_retrasos["RETRASADO_LLEGADA"] = pd.to_numeric(
    v_retrasos["RETRASADO_LLEGADA"], errors="coerce"
).fillna(0).astype(int)

# === Agrupar por aerolínea ===
resumen_retrasos = (
    v_retrasos.groupby(["AIRLINE", "AIRLINE_NAME"], observed=True)["RETRASADO_LLEGADA"]
    .agg(Total="size", Retrasados="sum")
    .reset_index()
)

# Calcular % retrasados
resumen_retrasos["Porcentaje_Retrasos"] = (
    resumen_retrasos["Retrasados"] / resumen_retrasos["Total"] * 100
)

# Crear columna combinada (por ejemplo: "AA - American Airlines Inc.")
resumen_retrasos["AEROLINEA_FULL"] = resumen_retrasos["AIRLINE"] + " - " + resumen_retrasos["AIRLINE_NAME"]

# Ordenar descendente por % de retrasos
resumen_retrasos = resumen_retrasos.sort_values("Porcentaje_Retrasos", ascending=False)

# Mostrar resultados
print(resumen_retrasos[["AEROLINEA_FULL", "Porcentaje_Retrasos"]].head(10))

# === Gráfico interactivo con Plotly ===
fig = px.bar(
    resumen_retrasos,
    x="AEROLINEA_FULL",
    y="Porcentaje_Retrasos",
    text=resumen_retrasos["Porcentaje_Retrasos"].apply(lambda x: f"{x:.2f}%"),
    title="Porcentaje de vuelos retrasados por aerolínea",
    labels={"AEROLINEA_FULL": "Aerolínea", "Porcentaje_Retrasos": "Retrasos (%)"},
    color="Porcentaje_Retrasos",
    color_continuous_scale="RdYlGn_r"
)

# Mostrar etiquetas y ajustar diseño
fig.update_traces(textposition="outside")
fig.update_layout(
    xaxis_tickangle=-45,
    title_x=0.5,
    plot_bgcolor="white",
    height=600
)

fig.show()


# %% [markdown]
# | Script                  | Agrupa por             | Muestra                | Ejemplo de uso                       |
# | ----------------------- | ---------------------- | ---------------------- | ------------------------------------ |
# | 2️⃣ `HORA_SALIDA`       | Hora (0–23)            | % de retrasos por hora | Picos de retraso según horario       |
# 

# %%
import pandas as pd
import plotly.express as px

# === Copia limpia ===
v_horas = v.copy()

# Asegurar tipos numéricos
v_horas["RETRASADO_LLEGADA"] = pd.to_numeric(
    v_horas["RETRASADO_LLEGADA"], errors="coerce"
).fillna(0).astype(int)
v_horas["HORA_SALIDA"] = pd.to_numeric(
    v_horas["HORA_SALIDA"], errors="coerce"
).fillna(0).astype(int)

# Filtrar horas válidas
v_horas = v_horas[v_horas["HORA_SALIDA"].between(0, 23)]

# === Agrupación por hora ===
resumen_horas = (
    v_horas.groupby("HORA_SALIDA", observed=True)["RETRASADO_LLEGADA"]
    .agg(Total="size", Retrasados="sum")
    .reset_index()
)

# Calcular % retrasados
resumen_horas["Porcentaje_Retrasos"] = (
    resumen_horas["Retrasados"] / resumen_horas["Total"] * 100
)

# === Gráfico interactivo ===
fig = px.bar(
    resumen_horas,
    x="HORA_SALIDA",
    y="Porcentaje_Retrasos",
    text=resumen_horas["Porcentaje_Retrasos"].apply(lambda x: f"{x:.2f}%"),
    title="Porcentaje de vuelos retrasados por hora programada de salida",
    labels={"HORA_SALIDA": "Hora de salida", "Porcentaje_Retrasos": "Retrasos (%)"},
    color="Porcentaje_Retrasos",
    color_continuous_scale="RdYlGn_r"
)
fig.update_traces(textposition="outside")
fig.update_layout(yaxis_title="Retrasos (%)", xaxis_title="Hora de salida")
fig.show()


# %% [markdown]
# Dos graficos uno a lado del otro 

# %%
# import pandas as pd
# import numpy as np
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots

# =========================
# A) Copia limpia (no toca v)
# =========================
v_air = v.copy()
v_hour = v.copy()

# Tipado/limpieza sin alterar v
for df in (v_air, v_hour):
    df["RETRASADO_LLEGADA"] = pd.to_numeric(df["RETRASADO_LLEGADA"], errors="coerce").fillna(0).astype(int)

v_hour["HORA_SALIDA"] = pd.to_numeric(v_hour["HORA_SALIDA"], errors="coerce").astype("Int64")
v_hour = v_hour[v_hour["HORA_SALIDA"].between(0, 23)]

# =========================
# B) Resúmenes
# =========================

# --- Por aerolínea ---
res_air = (
    v_air.groupby("AIRLINE", observed=True)["RETRASADO_LLEGADA"]
         .agg(Total="size", Retrasados="sum")
         .reset_index()
)
res_air["Porc_Retrasos"] = res_air["Retrasados"] / res_air["Total"] * 100
# Top 15 por volumen
res_air = res_air.sort_values(["Total", "Porc_Retrasos"], ascending=[False, False]).head(15)
# Ordenar eje X por % retrasos (desc) para lectura
res_air = res_air.sort_values("Porc_Retrasos", ascending=False).reset_index(drop=True)

# --- Por hora (0–23) ---
res_hour = (
    v_hour.groupby("HORA_SALIDA", observed=True)["RETRASADO_LLEGADA"]
          .agg(Total="size", Retrasados="sum")
          .reset_index()
          .sort_values("HORA_SALIDA")
)
res_hour["Porc_Retrasos"] = res_hour["Retrasados"] / res_hour["Total"] * 100

# =========================
# C) Subplots (2 columnas)
# =========================
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=(
        "Top aerolíneas por % de retrasos (con volumen)",
        "Retrasos por hora programada de salida (0–23)"
    )
)

# --- C1) Barras por aerolínea ---
fig.add_trace(
    go.Bar(
        x=res_air["AIRLINE"],
        y=res_air["Porc_Retrasos"],
        text=[f"{x:.2f}%" for x in res_air["Porc_Retrasos"]],
        textposition="outside",
        marker=dict(
            color=res_air["Porc_Retrasos"],
            colorscale="RdYlGn_r",
            colorbar=dict(title="%")
        ),
        hovertemplate=(
            "<b>%{x}</b><br>"
            "% Retrasos: %{y:.2f}%<br>"
            "Retrasados: %{customdata[0]:,}<br>"
            "Total: %{customdata[1]:,}<extra></extra>"
        ),
        customdata=np.stack([res_air["Retrasados"], res_air["Total"]], axis=-1),
        name="% Retrasos (aerolínea)"
    ),
    row=1, col=1
)

# --- C2) Barras por hora ---
fig.add_trace(
    go.Bar(
        x=res_hour["HORA_SALIDA"],
        y=res_hour["Porc_Retrasos"],
        text=[f"{x:.2f}%" for x in res_hour["Porc_Retrasos"]],
        textposition="outside",
        marker=dict(
            color=res_hour["Porc_Retrasos"],
            colorscale="RdYlGn_r",
            showscale=False  # ya mostramos colorbar en el gráfico de la izquierda
        ),
        hovertemplate=(
            "<b>Hora %{x}</b><br>"
            "% Retrasos: %{y:.2f}%<br>"
            "Retrasados: %{customdata[0]:,}<br>"
            "Total: %{customdata[1]:,}<extra></extra>"
        ),
        customdata=np.stack([res_hour["Retrasados"], res_hour["Total"]], axis=-1),
        name="% Retrasos (hora)"
    ),
    row=1, col=2
)

# =========================
# D) Layout
# =========================
fig.update_layout(
    title=(
        "Dashboard: % de retrasos por Aerolínea y por Hora de salida<br>"
        "<sup>Etiquetas muestran %; tooltip muestra % y n</sup>"
    ),
    bargap=0.25,
    template="plotly_white",
    margin=dict(l=40, r=40, t=80, b=40),
    showlegend=False
)

fig.update_yaxes(title_text="% Retrasos", row=1, col=1)
fig.update_yaxes(title_text="% Retrasos", row=1, col=2)
fig.update_xaxes(title_text="Aerolínea", row=1, col=1)
fig.update_xaxes(title_text="Hora de salida", row=1, col=2)

fig.show()

# (Opcional) Guardar como HTML interactivo:
# fig.write_html("dashboard_retrasos_aerolinea_hora.html")


# %% [markdown]
# Día de la semana / Mes

# %%
sns.barplot(data=v, x="DAY_OF_WEEK", y="RETRASADO_LLEGADA")
plt.title("Retrasos por día de la semana (1=Lun … 7=Dom)"); plt.show()

sns.barplot(data=v, x="MONTH", y="RETRASADO_LLEGADA")
plt.title("Retrasos por mes"); plt.show()


# %% [markdown]
# Distancia y relación con retraso

# %%
muestra = v.sample(min(len(v), 5000), random_state=42)
sns.scatterplot(data=muestra, x="DISTANCE", y="ARRIVAL_DELAY", alpha=0.3)
plt.title("Distancia vs Retraso en llegada"); plt.show()


# %% [markdown]
# Agrupaciones solicitadas (volumen vs retrasados)

# %%
# Totales por dimensión
tot_x_origen      = v.groupby(["ORIGIN_AIRPORT","ORIGEN_AEROPUERTO"]).size().rename("TOTAL")
ret_x_origen      = v.groupby(["ORIGIN_AIRPORT","ORIGEN_AEROPUERTO"])["RETRASADO_LLEGADA"].sum().rename("RETRASADOS")
df_origen = pd.concat([tot_x_origen, ret_x_origen], axis=1)
df_origen["PORC_RETRASADOS"] = df_origen["RETRASADOS"] / df_origen["TOTAL"]

tot_x_linea      = v.groupby("AIRLINE_NAME").size().rename("TOTAL")
ret_x_linea      = v.groupby("AIRLINE_NAME")["RETRASADO_LLEGADA"].sum().rename("RETRASADOS")
df_linea = pd.concat([tot_x_linea, ret_x_linea], axis=1)
df_linea["PORC_RETRASADOS"] = df_linea["RETRASADOS"] / df_linea["TOTAL"]

tot_x_dest       = v.groupby(["DESTINATION_AIRPORT","DEST_AEROPUERTO"]).size().rename("TOTAL")
ret_x_dest       = v.groupby(["DESTINATION_AIRPORT","DEST_AEROPUERTO"])["RETRASADO_LLEGADA"].sum().rename("RETRASADOS")
df_dest = pd.concat([tot_x_dest, ret_x_dest], axis=1)
df_dest["PORC_RETRASADOS"] = df_dest["RETRASADOS"] / df_dest["TOTAL"]

# Por país/estado/ciudad (ORIGEN)
tot_x_estado = v.groupby("ORIGEN_ESTADO").size().rename("TOTAL")
ret_x_estado = v.groupby("ORIGEN_ESTADO")["RETRASADO_LLEGADA"].sum().rename("RETRASADOS")
df_estado = pd.concat([tot_x_estado, ret_x_estado], axis=1)
df_estado["PORC_RETRASADOS"] = df_estado["RETRASADOS"] / df_estado["TOTAL"]

tot_x_ciudad = v.groupby("ORIGEN_CIUDAD").size().rename("TOTAL")
ret_x_ciudad = v.groupby("ORIGEN_CIUDAD")["RETRASADO_LLEGADA"].sum().rename("RETRASADOS")
df_ciudad = pd.concat([tot_x_ciudad, ret_x_ciudad], axis=1)
df_ciudad["PORC_RETRASADOS"] = df_ciudad["RETRASADOS"] / df_ciudad["TOTAL"]

# Muestras rápidas
display(df_linea.sort_values("PORC_RETRASADOS", ascending=False).head(10))
display(df_origen.sort_values("PORC_RETRASADOS", ascending=False).head(10))
display(df_dest.sort_values("PORC_RETRASADOS", ascending=False).head(10))
display(df_estado.sort_values("PORC_RETRASADOS", ascending=False).head(10))
display(df_ciudad.sort_values("PORC_RETRASADOS", ascending=False).head(10))


# %% [markdown]
# ¿Podemos usar lat/lon para mapas?
# 
# Sí. Con plotly.express podemos mostrar orígenes con mayor % de retraso:
# 

# %% [markdown]
# x origen

# %%
# %pip install nbformat


# %%
# pip install plotly  (si no lo tienes)
import plotly.express as px


# Prepara agregación por ORIGEN con coordenadas
geo_origen = (v.groupby(["ORIGIN_AIRPORT","ORIGEN_AEROPUERTO","ORIGEN_CIUDAD","ORIGEN_ESTADO","ORIGEN_PAIS","ORIGEN_LAT","ORIGEN_LON"])
                .agg(TOTAL=("RETRASADO_LLEGADA","size"),
                     PORC_RETRASADOS=("RETRASADO_LLEGADA","mean"))
                .reset_index())

# Filtra aeropuertos con suficiente volumen (p.ej. > 5000 vuelos)
geo_top = geo_origen[geo_origen["TOTAL"] > 5000].copy()

fig = px.scatter_geo(
    geo_top,
    lat="ORIGEN_LAT", lon="ORIGEN_LON",
    color="PORC_RETRASADOS", size="TOTAL",
    hover_name="ORIGEN_AEROPUERTO",
    hover_data={"ORIGEN_CIUDAD":True, "ORIGEN_ESTADO":True, "TOTAL":":,", "PORC_RETRASADOS":":.2%"},
    scope="north america", projection="natural earth",
    title="Aeropuertos de origen con mayor % de retrasos (tamaño=volumen)"
)
fig.show()


# %% [markdown]
# x Destino

# %%
# --- Mapa de DESTINOS con Plotly ---
# Requiere que en 'v' existan las columnas agregadas en el merge:
# DEST_AEROPUERTO, DEST_CIUDAD, DEST_ESTADO, DEST_PAIS, DEST_LAT, DEST_LON
# y la etiqueta RETRASADO_LLEGADA (0/1)

# import plotly.express as px
# import numpy as np
# import pandas as pd

# 1) Agregación por destino
geo_dest = (
    v.groupby(["DESTINATION_AIRPORT","DEST_AEROPUERTO","DEST_CIUDAD","DEST_ESTADO","DEST_PAIS","DEST_LAT","DEST_LON"])
     .agg(TOTAL=("RETRASADO_LLEGADA","size"),
          PORC_RETRASADOS=("RETRASADO_LLEGADA","mean"))
     .reset_index()
)

# 2) Limpieza: quitar filas sin coordenadas o con 0 volumen
geo_dest = geo_dest.replace([np.inf, -np.inf], np.nan).dropna(subset=["DEST_LAT","DEST_LON"])
geo_dest = geo_dest[geo_dest["TOTAL"] > 0]

# 3) Umbral de volumen (evita ruido visual). Ajusta a conveniencia (p.ej. 2000/5000)
umbral = 5000
geo_dest_top = geo_dest[geo_dest["TOTAL"] >= umbral].copy()

# 4) Mapa
fig_dest = px.scatter_geo(
    geo_dest_top,
    lat="DEST_LAT", lon="DEST_LON",
    size="TOTAL",
    color="PORC_RETRASADOS",
    color_continuous_scale="RdYlGn_r",  # rojo=peor, verde=mejor
    hover_name="DEST_AEROPUERTO",
    hover_data={
        "DEST_CIUDAD": True,
        "DEST_ESTADO": True,
        "TOTAL": ":,",
        "PORC_RETRASADOS": ":.2%",
        "DEST_LAT": False,
        "DEST_LON": False
    },
    scope="north america",    # ajusta a tu necesidad
    projection="natural earth",
    title=f"Destinos con mayor % de retrasos (tamaño=volumen, umbral ≥ {umbral})"
)
fig_dest.update_layout(margin=dict(r=0,l=0,t=60,b=0))
fig_dest.show()

# 5) (Opcional) Guardar a HTML interactivo
# fig_dest.write_html("mapa_destinos.html")


# %%
v.info()

# %%
print(v.columns.tolist())

# %%
v[v['AIRLINE'] == 'B6'][['AIRLINE', 'FLIGHT_NUMBER', 'TAIL_NUMBER', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT']] 

# %%
v[(v['AIRLINE'] == 'B6') & (v['TAIL_NUMBER'] == 'N353JB')][['AIRLINE', 'TAIL_NUMBER', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT']]

# %%
v[(v['AIRLINE'] == 'B6') & (v['TAIL_NUMBER'] == 'N353JB')]['ORIGIN_AIRPORT'].unique()

# %%
v.isnull().sum()

# %%
type(v)

# %%
v["ARRIVAL_TIME"].dtypes

type(v["ARRIVAL_TIME"])

# %%
valores_unicos=print(v.nunique())

# %%
valores_ausentes=print(v.isnull().sum())

# %%
v.isnull().sum()
print()


# %% [markdown]
# Porcentaje de valores ausentes de todas las columnas

# %%
v.isnull().mean()*100 #porcentaje de valores nulos

# %%
round(v.isnull().mean()*100,2) #porcentaje de valores nulos

# %%
# resumen estadístico de datos numéricos
v.info()
v.describe(include='all')
v.head()


# %%
msno.bar(v)

# %% [markdown]
# matriz de valores ausentes para ver en que posición estan 

# %%
msno.matrix(v)

# %% [markdown]
# Un head map de correlación entre valores faltantes 

# %%
msno.heatmap(v)

# %% [markdown]
# ### Analizar de forma rápida los datos por columnas

# %%
v["ARRIVAL_TIME"].value_counts()

# %%
v["ARRIVAL_TIME"].value_counts(normalize=True)*100

# %% [markdown]
# ### Automatización del código values_count()

# %% [markdown]
# vuelos.columns()

# %%
v.columns

# %%
# for loop que itera en cada una de las columnas y muestra la distribución de datos en % 
for columna in v.columns:
    print(f"Se enseña la columna {columna}")
    print(v[columna].value_counts(normalize=True, dropna=False)*100)
    print()

# %%
for col in vuelos.columns:
    print(f"Se enseña la columna: {col}")
    print(vuelos[col].value_counts)
    print(vuelos[col].value_counts(normalize=True,dropna=False)*100)
    print()

# %% [markdown]
# sodigo para ver solo tipos de datos
# 

# %%
v.dtypes

# %% [markdown]
# En caso de que se quiera ordenar los datos de otra manera, se utilizaría el siguiente código: 

# %%
# .reset_index() se resetea el indice y se va como columna al df
# es necesario poner paréntesis antes ()
# .sort_values() para ordenar los datos por la columna "Genre" y el argumento ascending para cambiar el orden
(v["AIRLINE_NAME"].value_counts(normalize=True)*100).reset_index().sort_values(by="AIRLINE_NAME", ascending=False)

# %%
(v["AIRLINE_NAME"].value_counts(normalize=True)*100).reset_index().sort_values(by="proportion", ascending=False)

# %%
(vuelos["TAIL_NUMBER"].value_counts(normalize=True)*100).reset_index().sort_values(by="TAIL_NUMBER")

# %% [markdown]
# Resumen estadístico .describe()
# 
# de las columnas numericas
# 
# total de datos que se tienen en las columns, promedios 

# %%
(vuelos["DISTANCE"].value_counts(normalize=True)*100).reset_index().sort_values(by="DISTANCE")

# %%
# guardar valores únicos en variable
valores_unicos = v.nunique()
valores_unicos

# %%
v["DISTANCE"].isna().sum()



# %%
v[v["DISTANCE"].astype(str).str.strip() == '']

# %%
v["DISTANCE"].apply(type).value_counts()


# %%
v["DISTANCE"].value_counts().sort_index().head(100000)


# %%
# import numpy as np

# Buscar distancias muy parecidas (diferencia menor a 0.001)
distancias = v["DISTANCE"].dropna().unique()
similares = [(a, b) for a in distancias for b in distancias if a != b and abs(a - b) < 0.001]
print(similares[:10])


# %%
# Mostrar las dimensiones
print(f" Vuelos: {v.shape}")
print(f" Aeropuertos: {aeropuertos.shape}")
print(f" Aerolineas: {aerolineas.shape}")

# %%
v.describe()

# %%
v.info()

# %%
v.select_dtypes(include="number")

# %%
categirical_data=v.select_dtypes(include="object")
categirical_data.describe()

# %%
numerical_data=v.select_dtypes(include="float64")
numerical_data.describe()

# %% [markdown]
# ## cambiar tipo de dato a una columna
# 

# %%
vuelos["ARRIVAL_DELAY"].astype("object")    

# %% [markdown]
# limpiar la columna User_Score

# %%
vuelos["ARRIVAL_DELAY"].value_counts(dropna=False)

# %% [markdown]
# tipo de datos ausente en numpy, y que pandas lo reconoce

# %%
np.nan

# %% [markdown]
# Limpieza nombres columnas

# %%
# vuelos.columns
vuelos.columns = vuelos.columns.str.upper()
vuelos.columns = vuelos.columns.str.lower()

# %%
vuelos.head()

# %% [markdown]
# limpieza de nombres de columnas

# %%
vuelos["airline"].value_counts()

# %%
vuelos["airline"]=vuelos["airline"].str.lower().str.strip()

# %%
v.isna().sum()

# %%
# porcentaje de valores ausentes de todas las columnas
round(v.isnull().mean()*100, 3)

# %%
# contar datos con los nans
v["ORIGEN_LAT"].value_counts(dropna=False)
# v["ORIGEN_LAT"].value_counts(dropna=True)

# %%
# filtrar videogame_names igual a wii sports
# v[v["ORIGEN_LAT"] == "nan"]
v[v["ORIGEN_LAT"].isna()]

# %% [markdown]
# revisar coordenadas en airports.csv

# %%
aeropuertos[aeropuertos["LATITUDE"].isna()]

# %%
aeropuertos[aeropuertos["LONGITUDE"].isna()]

# %% [markdown]
# ### Limpieza de valores ausentes "vuelos_airline"

# %%
## indexa
vuelos[]

# %%
vuelos[vuelos["airline"] == "aa"]

# %%
vuelos[vuelos["origin_airport"] == "LAX"]

# %% [markdown]
# # and &
# # or |
# # not

# %% [markdown]
# filtrar datos origen LAX y AEROLINEA aa

# %%
origen_aerolinea = vuelos[(vuelos["origin_airport"] == "LAX") & (vuelos["airline"] == "aa")]
origen_aerolinea.info()

# %% [markdown]
# filtrar datos origen LAX o AEROLINEA aa

# %%
origen_aerolinea = vuelos[(vuelos["origin_airport"] == "LAX") | (vuelos["airline"] == "aa")]
origen_aerolinea.info()
origen_aerolinea

# %%
vuelos[vuelos.departure_delay.isna() & vuelos.departure_delay.isna()]

# %%
vuelos.dropna(subset=["departure_delay", "arrival_delay"], how="all").reset_index(drop=True)

# %%
vuelos.isna().sum()

# %%
vuelos.shape

# %%
vuelos.dropna().shape

# %%
vuelos.shape

# %%
vuelos.isna().sum()

# %%
vuelos.head()

# %% [markdown]
# limpuza de valores ausentes en 

# %%
vuelos_ej = vuelos.dropna(columns=["departure_delay", "arrival_delay"])

# %%
vuelos.isna().sum()

# %%
vuelos["tail_number"].value_counts().head(20) 

# %%
vuelos[vuelos.scheduled_time.isna()]

# %%
# vuelos.groupby(["airline"]).agg({"flight_number": "count"})  
vuelos.groupby("airline")["arrival_delay"].median()
# vuelos.groupby("airline")["flight_number"].count()

# %%
vuelos.groupby(["month","airline"])["origin_airport"].count()

# %%
vuelos.groupby(["month","airline"])["origin_airport"].count().reset_index().rename(columns={"origin_airport": "count"}) #cambiar nombre de columna
# vuelos.groupby(["month", "airline"])["origin_airport"].count().reset_index().rename(columns={"origin_airport": "count"}).sort_values(by="count", ascending=False)

# %%
vuelos.groupby(["year","month"])["arrival_delay"].median()

# %%
vuelos.arrival_delay.isna() 

# %%


# %%
vuelos.iloc[0]

# %% [markdown]
# revsiones aeropuestos origen y destino

# %%
import pandas as pd

# --- 1️⃣ Rutas de archivos ---
flights_path = "../data/flights.csv"
airports_path = "../data/airports.csv"

# --- 2️⃣ Cargar datasets ---
flights = pd.read_csv(flights_path, usecols=["ORIGIN_AIRPORT", "DESTINATION_AIRPORT"])
airports = pd.read_csv(airports_path, usecols=["IATA_CODE"])

# --- 3️⃣ Normalizar formatos (importante por espacios o mayúsculas) ---
flights["ORIGIN_AIRPORT"] = flights["ORIGIN_AIRPORT"].astype(str).str.strip().str.upper()
flights["DESTINATION_AIRPORT"] = flights["DESTINATION_AIRPORT"].astype(str).str.strip().str.upper()
airports["IATA_CODE"] = airports["IATA_CODE"].astype(str).str.strip().str.upper()

# --- 4️⃣ Crear conjunto de códigos válidos ---
airports_set = set(airports["IATA_CODE"])

# --- 5️⃣ Detectar aeropuertos de origen y destino inexistentes ---
mask_origin_not_in = ~flights["ORIGIN_AIRPORT"].isin(airports_set)
mask_dest_not_in   = ~flights["DESTINATION_AIRPORT"].isin(airports_set)

# --- 6️⃣ Calcular totales ---
total = len(flights)
origin_missing = mask_origin_not_in.sum()
dest_missing   = mask_dest_not_in.sum()

# --- 7️⃣ Calcular porcentajes ---
pct_origin = (origin_missing / total) * 100
pct_dest   = (dest_missing / total) * 100

print("📊 Resultados de validación:")
print(f"Total de registros en flights.csv: {total:,}")
print(f"Aeropuertos ORIGIN no encontrados: {origin_missing:,} ({pct_origin:.2f}%)")
print(f"Aeropuertos DESTINATION no encontrados: {dest_missing:,} ({pct_dest:.2f}%)")

# --- 8️⃣ (Opcional) Ver los códigos que faltan ---
missing_origin_codes = sorted(set(flights.loc[mask_origin_not_in, "ORIGIN_AIRPORT"]))
missing_dest_codes   = sorted(set(flights.loc[mask_dest_not_in, "DESTINATION_AIRPORT"]))

print("\n✈️ Códigos de ORIGIN no encontrados:")
print(missing_origin_codes[:2000])  # muestra los primeros 20

print("\n🏁 Códigos de DESTINATION no encontrados:")
print(missing_dest_codes[:2000])




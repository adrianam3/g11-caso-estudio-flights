import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


st.title("Mi primera app en Streamlit!!")

# ruta de los datos
DATA_PATH = r"D:\OneDrive\DOCUMENTOS\Personales\2024\uniandes\8 S\seminario\g11-caso-estudio-flights\data\flights_clean.csv"

# un decorador mágico
# le dice a streamlit que no recargue ese CSV cada vez, sino
# que lo guarde en la memoria
# esto nos ayuda en el rendimiento
@st.cache_data
def cargar_datos(path):
    return pd.read_csv(path)


vuelos_clean = cargar_datos(DATA_PATH)

# mostrar datos en df
st.header("Vistazo del dataset de videojuegos")
st.write("Los primeros 15 datos del dataset:")
st.dataframe(vuelos_clean.head(15))


# mostrar datos estáticos
st.write("Mostrando datos estáticos")
st.table(vuelos_clean.tail())


# .Series
ventas_por_anio = vuelos_clean.groupby("year_of_release")["total_sales"].sum()


# header
st.header("Evolución de Ventas Totales por Año")


# fig_ventas es la figura 
# ax_ventas son los ejes
fig_ventas, ax_ventas = plt.subplots(figsize=(12,6))

# dibujar nuestro gráfico de lineas
ventas_por_anio.plot(
    kind="line", 
    ax=ax_ventas, 
    title="Ventas totales de videojuegos por año (Millones)", 
    xlabel = "Año de lanzamiento", 
    ylabel = "Ventas totales (Millones)"
)

# aplicar la cuadricula del eje 
ax_ventas.grid(True)

# mostrar figura en streamlit
st.pyplot(fig_ventas)
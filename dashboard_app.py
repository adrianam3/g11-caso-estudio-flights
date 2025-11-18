import streamlit as st
import pandas as pd
import plotly.express as px

# configuracion de la página
st.set_page_config(layout="wide")

# ruta de los datos
DATA_PATH = r"D:\OneDrive\DOCUMENTOS\Personales\2024\uniandes\8 S\seminario\g11-caso-estudio-flights\data\processed\flights_clean.csv"

# cargar datos
@st.cache_data
def cargar_datos(path):
    """funcion para cargar datos con caché"""
    return pd.read_csv(path)

# datos cargados
flights_clean = cargar_datos(DATA_PATH)

# titulo
st.title("Dashboard de Vuelos")

# para mostrar texto con fuente pequeña
st.caption("Caso de Estudio Grupo 11 | Retraso de Vuelos | Integrantes: Mario Farinango - Adrián Merlo")

st.subheader("Análisis  Exploratorio de Datos y Predicción de Retrasos")

# crear pestañas
tab1, tab2 = st.tabs(["Análisis Exploratorio (EDA)", "Predicción de Retraso de Vuelos (ML)"])

# PESTAÑA 1
with tab1:
    st.header("Indicadores Clave del Retraso de Vuelos")
    
    col_filtro1, col_filtro2 = st.columns(2)
    
    # Filtro de aerolíneas
    with col_filtro1:
        
        aerolineas_lista = sorted(flights_clean["AIRLINE"].unique())

        # selector
        aerolinea_seleccionada = st.multiselect(
            "Selecciona Aerolineas:",
            options=aerolineas_lista,
            default=[]
        )
    
    with col_filtro2:
        min_month = int(flights_clean["MONTH"].min())
        max_month = int(flights_clean["MONTH"].max())
        
        rango_meses = st.slider(
            "Selecciona rango de meses",
            min_value=min_month,
            max_value=max_month,
            value=(min_month, max_month) # valor mínimo, valor máximo, default del rango
        )
    
    # Filtrar dataframe antes de calcular métricas
    # Se puede seleccionar uno o varias aerolineas
    if aerolinea_seleccionada:
        # aerolineas
        filtro_aerolineas = flights_clean["AIRLINE"].isin(aerolinea_seleccionada)
        
        # rango de meses seleccionado
        filtro_mes = (flights_clean["MONTH"]>= rango_meses[0]) & (flights_clean["MONTH"]<= rango_meses[1])
        
        # filtro unificados de los 2 anteriores 
        flights_filtrado = flights_clean[filtro_aerolineas & filtro_mes]
        
    else:
        # Si no selecciona se crea df vacío - para evitar errores
        flights_filtrado = pd.DataFrame(columns=flights_clean.columns)

    if not flights_filtrado.empty:    
        # KPIs
        # métricas para mostrar valores numéricos grandes
        flights_filtrado = flights_filtrado[flights_filtrado["MOTIVO_RETRASO"].fillna("").str.lower() != "sin retraso"]

        # Total de vuelos con retraso
        total_vuelos_retraso = len(flights_filtrado)

        # Retraso promedio - ARRIVAL_DELAY
        retraso_promedio = flights_filtrado["ARRIVAL_DELAY"].mean()

        # Comparación para retrasos 
        retraso_esperado = 15  # referencia, valor de comparación se puede cambiar
        delta_retraso = retraso_promedio - retraso_esperado

        # Aerolínea con más retrasos
        df_airline = flights_filtrado.groupby("AIRLINE")["ARRIVAL_DELAY"].mean().reset_index()
        retraso_prom_aerolinea = df_airline["ARRIVAL_DELAY"].mean()
        airline_delay = df_airline.sort_values("ARRIVAL_DELAY", ascending=False).iloc[0]
        airline_name = airline_delay["AIRLINE"]
        
        airline_delay_valor = airline_delay["ARRIVAL_DELAY"]

        # Referencia respecto al retraso promedio general
        delta_airline = airline_delay_valor - retraso_promedio

        # Mes más retrasado
        # Retraso promedio por mes (más relevante)
        delay_mes = flights_filtrado.groupby("MONTH")["ARRIVAL_DELAY"].mean()
        prom_retraso_mensual = delay_mes.mean()
        mes_delay = delay_mes.idxmax()
        mes_delay_valor = delay_mes.max()
        delta_mes = mes_delay_valor - retraso_promedio

        # Ruta con más retrasos
        ruta_top = (
            flights_filtrado
            .groupby(["ORIGIN_AIRPORT", "DESTINATION_AIRPORT"])["ARRIVAL_DELAY"]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
            .iloc[0]
        )
        ruta_origen = ruta_top["ORIGIN_AIRPORT"]
        ruta_destino = ruta_top["DESTINATION_AIRPORT"]
        ruta_delay = ruta_top["ARRIVAL_DELAY"]
        delta_ruta = ruta_delay - retraso_promedio
    else:
        st.warning("No hay datos para las aerolineas seleccionadas")
        total_vuelos_retraso = 0
        retraso_promedio = 0
        delta_retraso = 0
        airline_name = ""
        delta_airline = 0
        mes_delay = 0
        delta_mes = 0
        ruta_origen = ""
        ruta_destino = ""
        ruta_delay = 0
        delta_ruta = 0
        airline_delay_valor= 0
        mes_delay_valor = 0

    # Separador de
    st.markdown("---") 

    # Columnas

    col1, col2, col3, col4, col5 = st.columns(5)
    
    # METRICAS 

    # ---- Columna1 Total vuelos ----
    with col1:
        with st.container(border=True):
            st.metric(
                label="Vuelos con Retraso",
                value=f"{total_vuelos_retraso:,}",
                delta="+5%"  # ejemplo estático, se puede calcular real
            )
            st.caption("Total registros")

    # ---- Columna2: Retraso promedio ----
    with col2:
        with st.container(border=True):
            st.metric(
                label="Retraso Promedio",
                value=f"{retraso_promedio:.2f} min",
                delta=f"{delta_retraso:.2f} min"
            )
            st.caption("Promedio general")

    # ---- Columna3: Aerolínea más retrasada ----
    with col3:
        with st.container(border=True):
            st.metric(
                label="Aerolínea Más Retrasada",
                value=airline_name,
                delta=f"{delta_airline:.2f} min"
            )
            st.caption(f"{airline_delay_valor:.2f} min promedio")

    # ---- Columna4: Mes más retrasado ----
    with col4:
        with st.container(border=True):
            st.metric(
                label="Mes Más Retrasado",
                value=str(mes_delay),
                delta=f"{delta_mes:.2f} min"
            )
            st.caption(f"{mes_delay_valor:.2f} min promedio")

    # ---- Columna5: Ruta más afectada ----
    with col5:
        with st.container(border=True):
            st.metric(
                label="Ruta Más Afectada",
                value=f"{ruta_origen} → {ruta_destino}",
                delta=f"{delta_ruta:.2f} min"
            )
            st.caption(f"{ruta_delay:.2f} min promedio")
    
    st.markdown("---")
    
    # GRAFICOS VUELOS POR MESES
    
    st.subheader("Vuelos por mes y retrasos")
    
    # df de vuelos por meses
    vuelos_df = flights_clean.groupby("MONTH").agg(
        CANTIDAD_VUELOS=("ARRIVAL_DELAY", "size"),
        ARRIVAL_DELAY_MEAN=("ARRIVAL_DELAY", "mean"),
        ARRIVAL_DELAY_SUM=("ARRIVAL_DELAY", "sum")
    ).reset_index()
    
    # .melt trasnforma un df de "ancho" a "largo"
    vuelos_df_melt = vuelos_df.melt(
        id_vars="MONTH",
        value_vars=["CANTIDAD_VUELOS", "ARRIVAL_DELAY_MEAN", "ARRIVAL_DELAY_SUM"],
        var_name="METRICA",
        value_name="VALOR"
    )
    
    fig_vuelos_meses = px.line(
        vuelos_df_melt,
        x="MONTH",
        y="VALOR",
        color="METRICA",
        title="Métricas por mes" ,
        labels={
        "MONTH": "Mes",
        "VALOR": "Cantidad de Vuelos"
    },
    markers=True   
    )
    
    # añadir slider
    fig_vuelos_meses.update_layout(xaxis_rangeslider_visible=True)
    
    st.plotly_chart(fig_vuelos_meses, use_container_width=True)
    st.caption("Control de movimiento de meses")
    
    
   
    st.markdown("---")
    st.subheader("Análisis de restrasos de vuelos según el destino")
    
    col_graf1, col_graf2 = st.columns(2)
    
    with col_graf1:
    
        # Grafico de Barras
        flights_clean_destination = flights_clean.groupby("DESTINATION_AIRPORT").size().nlargest(10).reset_index(name="CANTIDAD_VUELOS")

        fig_bar_destino = px.bar(
            flights_clean_destination,
            x="DESTINATION_AIRPORT",
            y="CANTIDAD_VUELOS",
            title="Top 10 Retraso de vuelos ",
            labels={
                "DESTINATION_AIRPORT": "Aeropuerto Destino",
                "CANTIDAD_VUELOS": "Número de Vuelos"
            },
            color="CANTIDAD_VUELOS",
            color_continuous_scale="Blues",
            text_auto=".2s"
        )
        
        fig_bar_destino.update_layout(showlegend=False, title_x=0.5)
        
        st.plotly_chart(fig_bar_destino, use_container_width=True)
    
    
    with col_graf2:
        st.write("### Distribución real de vuelos")

        vuelos_por_aerolinea = flights_clean["AIRLINE"].value_counts().reset_index()
        vuelos_por_aerolinea.columns = ["GRUPO", "TOTAL"]
        vuelos_por_aerolinea["CATEGORIA"] = "AEROLINEA"

        vuelos_por_dia = flights_clean["DAY_OF_WEEK"].value_counts().reset_index()
        vuelos_por_dia.columns = ["GRUPO", "TOTAL"]
        vuelos_por_dia["CATEGORIA"] = "DIA"

        vuelos_por_dest = flights_clean["DESTINATION_AIRPORT"].value_counts().reset_index()
        vuelos_por_dest.columns = ["GRUPO", "TOTAL"]
        vuelos_por_dest["CATEGORIA"] = "DESTINO"

        data_treemap = pd.concat([
            vuelos_por_aerolinea,
            vuelos_por_dia,
            vuelos_por_dest
        ], ignore_index=True)

        fig_treemap = px.treemap(
            data_treemap,
            path=[px.Constant("Vuelos Totales"), "CATEGORIA", "GRUPO"],
            values="TOTAL",
            color="TOTAL",
            color_continuous_scale="Blues",
            title="Distribución de vuelos por Aerolínea, Día y Destino"
        )

        st.plotly_chart(fig_treemap, width="stretch")



    st.markdown("---")
    
    col_graf3, col_graf4 = st.columns(2)
    
    with col_graf3:
        
        st.subheader("Porcentaje de vuelos retrasados por aerolínea")

        v_retrasos = flights_clean.copy()

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
        
        st.plotly_chart(fig, width="stretch")
        
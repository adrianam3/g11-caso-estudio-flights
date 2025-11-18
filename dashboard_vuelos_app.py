import os
import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np
import plotly.graph_objects as go


# ============================
# CONFIGURACIÓN INICIAL
# ============================

st.set_page_config(
    page_title="Dashboard Retrasos de Vuelos 2015",
    layout="wide"
)

st.markdown(
    """
    <style>
    /* Permitir que el valor de la métrica (lo grande) pueda ocupar varias líneas */
    div[data-testid="stMetricValue"] > div {
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
        word-break: break-word !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.title("Dashboard de Retrasos de Vuelos (USA, 2015)")
st.caption("Caso de Estudio | Predicción de Retrasos de Vuelos")

# # ============================
# # CARGA DE DATOS
# # ============================

# # Ajusta esta ruta a donde tengas flights_clean_am.csv
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "flights_clean.csv")

# @st.cache_data
# def cargar_datos(path: str) -> pd.DataFrame:
#     """Carga el dataset de vuelos con caché."""
#     df = pd.read_csv(path)
#     return df

# try:
#     flights = cargar_datos(DATA_PATH)
# except FileNotFoundError:
#     st.error(f"No se encontró el archivo en: {DATA_PATH}")
#     st.stop()

# ============================
# CARGA DE DATOS
# ============================

# Ajusta esta ruta a donde tengas flights_clean.csv
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "flights_clean.csv")

@st.cache_data
def cargar_datos(path: str, n_muestra: int = 5000, seed: int = 42) -> pd.DataFrame:
    """
    Carga el dataset de vuelos con caché y devuelve una muestra
    aleatoria pero reproducible de n_muestra registros.
    """
    df = pd.read_csv(path)

    # Muestra aleatoria pero siempre igual gracias a random_state
    if n_muestra is not None and n_muestra < len(df):
        df = df.sample(n=n_muestra, random_state=seed)

    return df

try:
    flights = cargar_datos(DATA_PATH)  # por defecto 5 000 filas
except FileNotFoundError:
    st.error(f"No se encontró el archivo en: {DATA_PATH}")
    st.stop()


# ============================
# PREPARACIÓN BÁSICA
# ============================

# Diccionario para mostrar nombre del día
DIA_SEMANA_MAP = {
    1: "Lunes",
    2: "Martes",
    3: "Miércoles",
    4: "Jueves",
    5: "Viernes",
    6: "Sábado",
    7: "Domingo",
}

flights["DAY_OF_WEEK_NOMBRE"] = flights["DAY_OF_WEEK"].map(DIA_SEMANA_MAP)

# Aseguramos que estas columnas existan y sean numéricas
for col in ["RETRASADO_LLEGADA", "RETRASADO_SALIDA", "ARRIVAL_DELAY", "DISTANCE"]:
    if col in flights.columns:
        flights[col] = pd.to_numeric(flights[col], errors="coerce")

# ============================
# FILTROS GLOBALES (SIDEBAR)
# ============================

st.sidebar.header("Filtros")

# Mes
meses_disponibles = sorted(flights["MONTH"].dropna().unique())
meses_sel = st.sidebar.multiselect(
    "Mes",
    options=meses_disponibles,
    default=meses_disponibles
)

# Día de la semana
dias_disponibles = sorted(flights["DAY_OF_WEEK"].dropna().unique())
dias_sel = st.sidebar.multiselect(
    "Día de la semana",
    options=dias_disponibles,
    format_func=lambda x: DIA_SEMANA_MAP.get(x, x),
    default=dias_disponibles
)

# Aerolínea
aerolineas_disp = sorted(flights["AIRLINE_NAME"].dropna().unique())
aerolineas_sel = st.sidebar.multiselect(
    "Aerolínea",
    options=aerolineas_disp,
    default=aerolineas_disp
)

# Aeropuerto origen
origenes_disp = sorted(flights["ORIGEN_AEROPUERTO"].dropna().unique())
origenes_sel = st.sidebar.multiselect(
    "Aeropuerto origen",
    options=origenes_disp,
    default=origenes_disp
)

# Aeropuerto destino
destinos_disp = sorted(flights["DEST_AEROPUERTO"].dropna().unique())
destinos_sel = st.sidebar.multiselect(
    "Aeropuerto destino",
    options=destinos_disp,
    default=destinos_disp
)

# Período de salida
if "PERIODO_SALIDA" in flights.columns:
    periodos_disp = ["Madrugada", "Mañana", "Tarde", "Noche"]
    periodos_disp = [p for p in periodos_disp if p in flights["PERIODO_SALIDA"].unique()]
    periodos_sel = st.sidebar.multiselect(
        "Período de salida",
        options=periodos_disp,
        default=periodos_disp
    )
else:
    periodos_sel = None

st.sidebar.markdown("---")
st.sidebar.write("**Tipo de retraso a analizar**")
analizar_salida = st.sidebar.checkbox("Retraso en salida", value=True)
analizar_llegada = st.sidebar.checkbox("Retraso en llegada", value=True)

# ============================
# APLICAR FILTROS
# ============================

df = flights.copy()

df = df[df["MONTH"].isin(meses_sel)]
df = df[df["DAY_OF_WEEK"].isin(dias_sel)]
df = df[df["AIRLINE_NAME"].isin(aerolineas_sel)]
df = df[df["ORIGEN_AEROPUERTO"].isin(origenes_sel)]
df = df[df["DEST_AEROPUERTO"].isin(destinos_sel)]

if periodos_sel is not None:
    df = df[df["PERIODO_SALIDA"].isin(periodos_sel)]

if df.empty:
    st.warning("No hay datos para los filtros seleccionados.")
    st.stop()

# ============================
# FUNCIONES AUXILIARES
# ============================

def calcular_porcentaje_retraso(serie_retraso: pd.Series) -> float:
    """Calcula % de retraso (variable binaria 0/1) sobre la serie no nula."""
    serie_valida = serie_retraso.dropna()
    if len(serie_valida) == 0:
        return 0.0
    return float(serie_valida.mean() * 100)


def crear_barra_porcentaje_retraso(df_group, dim, col_retraso, titulo, x_label):
    """Crea gráfico de barras de % retraso por dimensión (aerolínea, aeropuerto, etc.)."""
    tmp = (
        df_group
        .groupby(dim, observed=True)[col_retraso]
        .mean()
        .reset_index()
        .rename(columns={col_retraso: "porc_retraso"})
    )
    tmp["porc_retraso"] = tmp["porc_retraso"] * 100
    tmp = tmp.sort_values("porc_retraso", ascending=False)

    fig = px.bar(
        tmp,
        x="porc_retraso",
        y=dim,
        orientation="h",
        title=titulo,
        labels={"porc_retraso": "% retrasos", dim: x_label},
        text=tmp["porc_retraso"].round(1).astype(str) + "%",
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    return fig

# ============================
# TABS PRINCIPALES
# ============================

tab_resumen, tab_aerolineas, tab_aeropuertos, tab_tiempo, tab_causas = st.tabs(
    ["Resumen", "Aerolíneas", "Aeropuertos", "Tiempo", "Causas de retraso"]
)

# ============================
# TAB 1 - RESUMEN
# ============================
with tab_resumen:
    st.subheader("Resumen Ejecutivo")

    col1, col2, col3, col4, col5 = st.columns(5)

    total_vuelos = len(df)
    total_vuelos_retrasos_sal = df[
        ((df["DEPARTURE_DELAY"]>0) == 1)  
    ]
    total_vuelos_retrasos_sal = len(total_vuelos_retrasos_sal) if analizar_salida else 0.0
    
    total_vuelos_retrasos_lle = df[
        ((df["ARRIVAL_DELAY"]>0) == 1)  
    ]
    total_vuelos_retrasos_lle = len(total_vuelos_retrasos_lle) if analizar_llegada else 0.0
    
    # % retraso llegada y salida
    porc_retraso_lleg = calcular_porcentaje_retraso(df["RETRASADO_LLEGADA"]) if analizar_llegada else 0.0
    porc_retraso_sal = calcular_porcentaje_retraso(df["RETRASADO_SALIDA"]) if analizar_salida else 0.0

    # Retraso promedio llegada (en minutos)
    retraso_prom_lleg = df["ARRIVAL_DELAY"].dropna()
    retraso_prom_lleg = float(retraso_prom_lleg.mean()) if not retraso_prom_lleg.empty else 0.0

    # Retraso máximo llegada
    retraso_max_lleg = df["ARRIVAL_DELAY"].dropna()
    retraso_max_lleg = float(retraso_max_lleg.max()) if not retraso_max_lleg.empty else 0.0

    # Distancia promedio
    dist_prom = df["DISTANCE"].dropna()
    dist_prom = float(dist_prom.mean()) if not dist_prom.empty else 0.0

    col1.metric("Total de vuelos", f"{total_vuelos:,}".replace(",", "."))
    col2.metric("% retrasos llegada", f"{porc_retraso_lleg:.1f}%", delta=f"{total_vuelos_retrasos_lle:,} con retrasos".replace(",", "."))
    col3.metric("% retrasos salida", f"{porc_retraso_sal:.1f}%", delta=f"{total_vuelos_retrasos_sal:,} con retrasos".replace(",", "."))
    col4.metric("Retraso promedio llegada (min)", f"{retraso_prom_lleg:.1f}")
    col5.metric("Distancia promedio (millas)", f"{dist_prom:.1f}")

    st.markdown("---")
    
    col6, col7, col8 = st.columns(3)
    # col6, col7, col8, col9, col10 = st.columns([2.5, 2.5, 1.5, 1.5, 1.5])

    # Aerolínea con más retrasos
    # Filtrar solo vuelos con retraso en llegada
    flights_delay_d = df[df["DEPARTURE_DELAY"] > 0]

    # Calcular retraso promedio por aerolínea usando solo vuelos retrasados
    df_airline_d = (
        flights_delay_d
        .groupby("AIRLINE_NAME")["DEPARTURE_DELAY"]
        .mean()
        .reset_index()
    )
    # Promedio entre aerolíneas (cada aerolínea un voto)
    retraso_prom_aerolinea_d = df_airline_d["DEPARTURE_DELAY"].mean()
    # Aerolínea con mayor retraso promedio llegada
    airline_delay_d = df_airline_d.sort_values("DEPARTURE_DELAY", ascending=False).iloc[0]
    airline_name_d = airline_delay_d["AIRLINE_NAME"]  if analizar_salida else " - "

    # Aerolínea con más retrasos llegada
    # Filtrar solo vuelos con retraso en llegada
    flights_delay = df[df["ARRIVAL_DELAY"] > 0]

    # Calcular retraso promedio por aerolínea usando solo vuelos retrasados
    df_airline = (
        flights_delay
        .groupby("AIRLINE_NAME")["ARRIVAL_DELAY"]
        .mean()
        .reset_index()
    )
    # Promedio entre aerolíneas (cada aerolínea un voto)
    retraso_prom_aerolinea = df_airline["ARRIVAL_DELAY"].mean()
    # Aerolínea con mayor retraso promedio llegada
    airline_delay = df_airline.sort_values("ARRIVAL_DELAY", ascending=False).iloc[0]
    airline_name_lle = airline_delay["AIRLINE_NAME"]  if analizar_llegada else ' - '

    #-->
    # Ruta con más retrasos
    # Retraso promedio - ARRIVAL_DELAY
    retraso_promedio = flights_delay["ARRIVAL_DELAY"].mean()
    
    ruta_top = (
        flights_delay
        .groupby(["ORIGEN_AEROPUERTO", "DEST_AEROPUERTO"])["ARRIVAL_DELAY"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
        .iloc[0]
    )
    ruta_origen = ruta_top["ORIGEN_AEROPUERTO"]
    ruta_destino = ruta_top["DEST_AEROPUERTO"]
    ruta_delay = ruta_top["ARRIVAL_DELAY"]
    delta_ruta = ruta_delay - retraso_promedio    
    
    #<--

    col6.metric("Aerolínea Más Retrasada a las Salida", f"{airline_name_d}")
    col7.metric("Aerolínea Más Retrasada a la Llegada", f"{airline_name_lle}")
    col8.metric("Ruta Más Afectada", f"{ruta_origen} → {ruta_destino}", delta=f"{delta_ruta:.2f} min")

    st.markdown("---")
    # =======================
    # GRÁFICO COMBINADO MES (% + TOTAL)
    # =======================

    st.markdown("---")

    # ==========================
    # Tendencia mensual: Total de vuelos vs % de retrasos
    # ==========================

    # Asegurar que MONTH exista (por si algún día cargas otro dataset)
    if "MONTH" not in df.columns:
        if "FL_DATE" in df.columns:
            df["MONTH"] = pd.to_datetime(df["FL_DATE"], errors="coerce").dt.month
        else:
            st.error("No existe la columna MONTH ni FL_DATE para derivarla.")
    else:
        # Por si viene como float/obj
        df["MONTH"] = pd.to_numeric(df["MONTH"], errors="coerce")

    # Resumen mensual: totales y % retrasos
    resumen_mes = (
        df.groupby("MONTH", observed=True)["RETRASADO_LLEGADA"]
        .agg(Total="size", Retrasados="sum")
        .reset_index()
        .sort_values("MONTH")
    )

    if resumen_mes.empty:
        st.info("No hay datos para generar la tendencia mensual con los filtros actuales.")
    else:
        resumen_mes["Porc_Retrasado"] = (
            resumen_mes["Retrasados"] / resumen_mes["Total"] * 100
        )

        meses = [
            "Enero","Febrero","Marzo","Abril","Mayo","Junio",
            "Julio","Agosto","Septiembre","Octubre","Noviembre","Diciembre"
        ]
        resumen_mes["MES_NOMBRE"] = resumen_mes["MONTH"].apply(
            lambda m: meses[int(m)-1] if pd.notna(m) and 1 <= int(m) <= 12 else "Desconocido"
        )

        # Totales globales para título dinámico
        total_vuelos_anual = resumen_mes["Total"].sum()
        promedio_retrasos_anual = (
            (resumen_mes["Retrasados"].sum() / total_vuelos_anual) * 100
            if total_vuelos_anual > 0 else 0.0
        )

        # Colores por umbral para los puntos
        pct = resumen_mes["Porc_Retrasado"].values
        colors = np.where(
            pct < 15,
            "#4CAF50",                          # verde
            np.where(pct <= 25, "#FFC107", "#F44336")  # amarillo / rojo
        )

        # fig = go.Figure()

        # # Barras = total de vuelos (eje izquierdo)
        # fig.add_trace(go.Bar(
        #     x=resumen_mes["MES_NOMBRE"],
        #     y=resumen_mes["Total"],
        #     name="Total de vuelos",
        #     marker_color="rgba(56, 128, 255, 0.85)",  # azul intenso
        #     text=resumen_mes["Total"].apply(lambda x: f"{x:,}"),
        #     textposition="inside",                    # dentro de la barra
        #     textfont=dict(color="white", size=11),
        #     yaxis="y1",
        #     hovertemplate="<b>%{x}</b><br>Total de vuelos: %{y:,}<extra></extra>"
        # ))

        # # Línea = % de retrasos (puntos con colores por umbral, eje derecho)
        # fig.add_trace(go.Scatter(
        #     x=resumen_mes["MES_NOMBRE"],
        #     y=resumen_mes["Porc_Retrasado"],
        #     name="% Retrasos",
        #     mode="lines+markers+text",
        #     line=dict(color="#FF9800", width=3),      # línea naranja
        #     marker=dict(
        #         size=10,
        #         color=colors,
        #         line=dict(color="#FFB300", width=1.5)
        #     ),
        #     text=[f"{v:.2f}%" for v in resumen_mes["Porc_Retrasado"]],
        #     textposition="top center",
        #     textfont=dict(color="white", size=11),
        #     yaxis="y2",
        #     hovertemplate="<b>%{x}</b><br>% Retrasos: %{y:.2f}%<extra></extra>"
        # ))

        # # Más espacio en el eje de %
        # y2_max = max(
        #     40,
        #     float(resumen_mes["Porc_Retrasado"].max() * 1.5),
        #     float(promedio_retrasos_anual * 1.3),
        # )

        # # Bandas de color + línea de promedio
        # fig.update_layout(
        #     shapes=[
        #         # Verde: <15%
        #         dict(type="rect", xref="paper", x0=0, x1=1, yref="y2",
        #              y0=0, y1=15, fillcolor="rgba(76,175,80,0.12)", line_width=0, layer="below"),
        #         # Amarillo: 15–25%
        #         dict(type="rect", xref="paper", x0=0, x1=1, yref="y2",
        #              y0=15, y1=25, fillcolor="rgba(255,193,7,0.15)", line_width=0, layer="below"),
        #         # Rojo: >25%
        #         dict(type="rect", xref="paper", x0=0, x1=1, yref="y2",
        #              y0=25, y1=y2_max, fillcolor="rgba(244,67,54,0.10)", line_width=0, layer="below"),
        #         # Línea horizontal del promedio anual
        #         dict(
        #             type="line", xref="paper", x0=0, x1=1, yref="y2",
        #             y0=promedio_retrasos_anual, y1=promedio_retrasos_anual,
        #             line=dict(color="#03A9F4", width=2, dash="dot")
        #         ),
        #     ]
        # )

        # # Leyenda personalizada de umbrales + línea de referencia
        # fig.add_trace(go.Scatter(
        #     x=[None], y=[None], mode="markers",
        #     marker=dict(size=10, color="#4CAF50"),
        #     name="< 15% retrasos"
        # ))
        # fig.add_trace(go.Scatter(
        #     x=[None], y=[None], mode="markers",
        #     marker=dict(size=10, color="#FFC107"),
        #     name="15% – 25% retrasos"
        # ))
        # fig.add_trace(go.Scatter(
        #     x=[None], y=[None], mode="markers",
        #     marker=dict(size=10, color="#F44336"),
        #     name="> 25% retrasos"
        # ))
        # fig.add_trace(go.Scatter(
        #     x=[None], y=[None], mode="lines",
        #     line=dict(color="#03A9F4", width=2, dash="dot"),
        #     name=f"Promedio anual ({promedio_retrasos_anual:.2f}%)"
        # ))

        # # Layout final
        # fig.update_layout(
        #     title=(
        #         "Tendencia mensual: Total de vuelos vs % de retrasos<br>"
        #         f"<sup>✈️ Total periodo filtrado: {total_vuelos_anual:,} vuelos | "
        #         f"Promedio de retrasos: {promedio_retrasos_anual:.2f}%</sup>"
        #     ),
        #     xaxis=dict(title="Mes"),
        #     yaxis=dict(title="Total de vuelos", side="left", showgrid=False),
        #     yaxis2=dict(
        #         title="% Retrasos",
        #         side="right",
        #         overlaying="y",
        #         showgrid=False,
        #         range=[0, y2_max]
        #     ),
        #     legend=dict(
        #         orientation="h",
        #         yanchor="bottom", y=1.02,
        #         xanchor="right", x=1
        #     ),
        #     bargap=0.3,
        #     plot_bgcolor="rgba(0,0,0,0)",   # fondo transparente (tema oscuro)
        #     paper_bgcolor="rgba(0,0,0,0)",
        #     margin=dict(l=60, r=60, t=90, b=50),
        # )
        # --- E) Figura combinada ---
        # Colores por umbral para los puntos
        pct = resumen_mes["Porc_Retrasado"].values
        colors = np.where(
            pct < 15,
            "#4CAF50",                          # verde
            np.where(pct <= 25, "#FFC107", "#F44336")  # amarillo / rojo
        )

        fig = go.Figure()

        # ==========================
        # BARRAS = Total de vuelos (CON texto dentro)
        # ==========================
        fig.add_trace(go.Bar(
            x=resumen_mes["MES_NOMBRE"],
            y=resumen_mes["Total"],
            name="Total de vuelos",
            marker_color="#1976D2",  # azul vivo
            text=resumen_mes["Total"].apply(lambda x: f"{x:,}"),
            textposition="inside",   # texto dentro de la barra
            textfont=dict(color="white", size=11),
            yaxis="y1",
            hovertemplate="<b>%{x}</b><br>Total de vuelos: %{y:,}<extra></extra>"
        ))

        # ==========================
        # LÍNEA = % de retrasos (CON texto, pero abajo del punto)
        # ==========================
        fig.add_trace(go.Scatter(
            x=resumen_mes["MES_NOMBRE"],
            y=resumen_mes["Porc_Retrasado"],
            name="% Retrasos",
            mode="lines+markers+text",
            line=dict(color="#F57C00", width=3),      # naranja intenso
            marker=dict(
                size=10,
                color=colors,
                line=dict(color="#BF360C", width=1.5)
            ),
            text=[f"{v:.2f}%" for v in resumen_mes["Porc_Retrasado"]],
            textposition="bottom center",             # <--- ya no se pisa con el texto de la barra
            textfont=dict(size=11),
            yaxis="y2",
            hovertemplate="<b>%{x}</b><br>% Retrasos: %{y:.2f}%<extra></extra>"
        ))

        # ==========================
        # Bandas + línea promedio
        # ==========================
        y2_max = max(
            45,  # más espacio vertical para los textos
            float(resumen_mes["Porc_Retrasado"].max() * 1.6),
            float(promedio_retrasos_anual * 1.4),
        )

        fig.update_layout(
            shapes=[
                # Verde: <15%
                dict(type="rect", xref="paper", x0=0, x1=1, yref="y2",
                     y0=0, y1=15, fillcolor="rgba(129,199,132,0.5)", line_width=0, layer="below"),
                # Amarillo: 15–25%
                dict(type="rect", xref="paper", x0=0, x1=1, yref="y2",
                     y0=15, y1=25, fillcolor="rgba(255,241,118,0.6)", line_width=0, layer="below"),
                # Rojo: >25%
                dict(type="rect", xref="paper", x0=0, x1=1, yref="y2",
                     y0=25, y1=y2_max, fillcolor="rgba(239,154,154,0.6)", line_width=0, layer="below"),
                # Línea horizontal de promedio anual
                dict(
                    type="line", xref="paper", x0=0, x1=1, yref="y2",
                    y0=promedio_retrasos_anual, y1=promedio_retrasos_anual,
                    line=dict(color="#00838F", width=2, dash="dot")
                ),
            ]
        )

        # Leyenda de umbrales + referencia
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=10, color="#4CAF50"),
            name="< 15% retrasos"
        ))
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=10, color="#FFC107"),
            name="15% – 25% retrasos"
        ))
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=10, color="#F44336"),
            name="> 25% retrasos"
        ))
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines",
            line=dict(color="#00838F", width=2, dash="dot"),
            name=f"Promedio anual ({promedio_retrasos_anual:.2f}%)"
        ))

        fig.update_layout(
            title=(
                "Tendencia mensual: Total de vuelos vs % de retrasos<br>"
                f"<sup>✈️ Total periodo filtrado: {total_vuelos_anual:,} vuelos | "
                f"Promedio de retrasos: {promedio_retrasos_anual:.2f}%</sup>"
            ),
            xaxis=dict(title="Mes"),
            yaxis=dict(title="Total de vuelos", side="left", showgrid=False),
            yaxis2=dict(
                title="% Retrasos",
                side="right",
                overlaying="y",
                showgrid=False,
                range=[0, y2_max]
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            bargap=0.3,
            template="plotly_white",
            margin=dict(l=60, r=60, t=90, b=50),
        )
        st.plotly_chart(fig, use_container_width=True)


    st.markdown("---")
    # % retrasos por mes
    st.markdown("### % de retrasos por mes (llegada)")

    df_mes = (
        df.groupby("MONTH", observed=True)["RETRASADO_LLEGADA"]
        .mean()
        .reset_index()
        .rename(columns={"RETRASADO_LLEGADA": "porc_retraso"})
    )
    df_mes["porc_retraso"] = df_mes["porc_retraso"] * 100

    fig_mes = px.bar(
        df_mes,
        x="MONTH",
        y="porc_retraso",
        labels={"MONTH": "Mes", "porc_retraso": "% vuelos con retraso de llegada"},
        title="% de vuelos con retraso de llegada por mes",
        text=df_mes["porc_retraso"].round(1).astype(str) + "%",
    )
    st.plotly_chart(fig_mes, use_container_width=True)

    # Histograma de retraso en llegada
    st.markdown("### Distribución del retraso en llegada (minutos)")

    df_delay = df[df["ARRIVAL_DELAY"].notna() & (df["ARRIVAL_DELAY"] > -20) & (df["ARRIVAL_DELAY"] < 300)]
    if not df_delay.empty:
        fig_hist = px.histogram(
            df_delay,
            x="ARRIVAL_DELAY",
            nbins=60,
            labels={"ARRIVAL_DELAY": "Retraso en llegada (min)"},
            title="Histograma de retraso en llegada (entre -20 y 300 min)",
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info("No hay datos de ARRIVAL_DELAY válidos para mostrar el histograma.")

    # Ranking de aerolíneas por puntualidad
    st.markdown("### Ranking de aerolíneas por % de retrasos en llegada")

    fig_rank_aero = crear_barra_porcentaje_retraso(
        df, dim="AIRLINE_NAME", col_retraso="RETRASADO_LLEGADA",
        titulo="% de vuelos con retraso en llegada por aerolínea",
        x_label="Aerolínea"
    )
    st.plotly_chart(fig_rank_aero, use_container_width=True)


# ============================
# TAB 2 - AEROLÍNEAS
# ============================
with tab_aerolineas:
    st.subheader("Análisis por Aerolínea")

    # Selector de aerolínea para detalle
    aerolinea_detalle = st.selectbox(
        "Selecciona una aerolínea para ver detalle",
        options=sorted(df["AIRLINE_NAME"].unique())
    )

    df_aero = df[df["AIRLINE_NAME"] == aerolinea_detalle]

    col1, col2, col3, col4 = st.columns(4)

    total_vuelos_aero = len(df_aero)
    porc_retraso_lleg_aero = calcular_porcentaje_retraso(df_aero["RETRASADO_LLEGADA"])
    retraso_prom_lleg_aero = df_aero["ARRIVAL_DELAY"].dropna()
    retraso_prom_lleg_aero = float(retraso_prom_lleg_aero.mean()) if not retraso_prom_lleg_aero.empty else 0.0

    col1.metric("Vuelos de la aerolínea", f"{total_vuelos_aero:,}".replace(",", "."))
    col2.metric("% retrasos llegada (aerolínea)", f"{porc_retraso_lleg_aero:.1f}%")
    col3.metric("Retraso prom. llegada (min)", f"{retraso_prom_lleg_aero:.1f}")

    # # Top 5 aeropuertos de origen de esa aerolínea
    # top_origen_aero = (
    #     df_aero["ORIGEN_AEROPUERTO"].value_counts()
    #     .head(5)
    #     .reset_index()
    #     .rename(columns={"index": "ORIGEN_AEROPUERTO", "ORIGEN_AEROPUERTO": "count"})
    # )

    # col4.write("Top 5 aeropuertos de origen")
    # col4.dataframe(top_origen_aero)
    # Top 5 aeropuertos de origen de esa aerolínea
    top_origen_aero = (
        df_aero["ORIGEN_AEROPUERTO"]
        .value_counts()
        .head(5)
        .reset_index()
    )

    # Renombrar columnas de forma explícita para evitar duplicados
    top_origen_aero.columns = ["ORIGEN_AEROPUERTO", "TOTAL_VUELOS"]

    col4.write("Top 5 aeropuertos de origen")
    col4.dataframe(top_origen_aero)

    st.markdown("---")

    # Ranking de aerolíneas por % retraso llegada
    st.markdown("### % de retrasos en llegada por aerolínea")

    fig_aero_lleg = crear_barra_porcentaje_retraso(
        df, dim="AIRLINE_NAME", col_retraso="RETRASADO_LLEGADA",
        titulo="% retrasos en llegada por aerolínea",
        x_label="Aerolínea"
    )
    st.plotly_chart(fig_aero_lleg, use_container_width=True)

    # Ranking por retraso promedio en llegada
    st.markdown("### Retraso promedio en llegada por aerolínea")

    df_avg_delay = (
        df.groupby("AIRLINE_NAME", observed=True)["ARRIVAL_DELAY"]
        .mean()
        .reset_index()
        .rename(columns={"ARRIVAL_DELAY": "retraso_prom"})
    )
    df_avg_delay = df_avg_delay.sort_values("retraso_prom", ascending=False)

    fig_aero_delay = px.bar(
        df_avg_delay,
        x="retraso_prom",
        y="AIRLINE_NAME",
        orientation="h",
        labels={"retraso_prom": "Retraso prom. llegada (min)", "AIRLINE_NAME": "Aerolínea"},
        title="Retraso promedio en llegada por aerolínea",
        text=df_avg_delay["retraso_prom"].round(1).astype(str),
    )
    fig_aero_delay.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig_aero_delay, use_container_width=True)


# ============================
# TAB 3 - AEROPUERTOS
# ============================
with tab_aeropuertos:
    st.subheader("Análisis por Aeropuertos")

    col1, col2 = st.columns(2)

    # Top 10 aeropuertos origen por volumen
    # top_origen = (
    #     df["ORIGEN_AEROPUERTO"].value_counts()
    #     .head(10)
    #     .reset_index()
    #     .rename(columns={"index": "ORIGEN_AEROPUERTO", "ORIGEN_AEROPUERTO": "count"})
    # )
    # fig_top_origen = px.bar(
    #     top_origen,
    #     x="ORIGEN_AEROPUERTO",
    #     y="count",
    #     labels={"ORIGEN_AEROPUERTO": "Aeropuerto origen", "count": "Número de vuelos"},
    #     title="Top 10 aeropuertos de origen por número de vuelos",
    # )
    
    top_origen = (
        df["ORIGEN_AEROPUERTO"]
        .value_counts()
        .head(10)
        .reset_index()
    )
    top_origen.columns = ["ORIGEN_AEROPUERTO", "TOTAL_VUELOS"]

    fig_top_origen = px.bar(
        top_origen,
        x="ORIGEN_AEROPUERTO",
        y="TOTAL_VUELOS",
        labels={"ORIGEN_AEROPUERTO": "Aeropuerto origen", "TOTAL_VUELOS": "Número de vuelos"},
        title="Top 10 aeropuertos de origen por número de vuelos",
    )

    col1.plotly_chart(fig_top_origen, use_container_width=True)

    # Top 10 aeropuertos destino por volumen
    # top_dest = (
    #     df["DEST_AEROPUERTO"].value_counts()
    #     .head(10)
    #     .reset_index()
    #     .rename(columns={"index": "DEST_AEROPUERTO", "DEST_AEROPUERTO": "count"})
    # )
    # fig_top_dest = px.bar(
    #     top_dest,
    #     x="DEST_AEROPUERTO",
    #     y="count",
    #     labels={"DEST_AEROPUERTO": "Aeropuerto destino", "count": "Número de vuelos"},
    #     title="Top 10 aeropuertos de destino por número de vuelos",
    # )
    
    top_dest = (
        df["DEST_AEROPUERTO"]
        .value_counts()
        .head(10)
        .reset_index()
    )
    top_dest.columns = ["DEST_AEROPUERTO", "TOTAL_VUELOS"]

    fig_top_dest = px.bar(
        top_dest,
        x="DEST_AEROPUERTO",
        y="TOTAL_VUELOS",
        labels={"DEST_AEROPUERTO": "Aeropuerto destino", "TOTAL_VUELOS": "Número de vuelos"},
        title="Top 10 aeropuertos de destino por número de vuelos",
    )

    col2.plotly_chart(fig_top_dest, use_container_width=True)

    st.markdown("---")

    col3, col4 = st.columns(2)

    # % retrasos de salida por aeropuerto origen
    fig_retraso_origen = crear_barra_porcentaje_retraso(
        df, dim="ORIGEN_AEROPUERTO", col_retraso="RETRASADO_SALIDA",
        titulo="% de retrasos de salida por aeropuerto origen",
        x_label="Aeropuerto origen"
    )
    col3.plotly_chart(fig_retraso_origen, use_container_width=True)

    # % retrasos de llegada por aeropuerto destino
    fig_retraso_dest = crear_barra_porcentaje_retraso(
        df, dim="DEST_AEROPUERTO", col_retraso="RETRASADO_LLEGADA",
        titulo="% de retrasos de llegada por aeropuerto destino",
        x_label="Aeropuerto destino"
    )
    col4.plotly_chart(fig_retraso_dest, use_container_width=True)


# ============================
# TAB 4 - TIEMPO Y PATRONES
# ============================
with tab_tiempo:
    st.subheader("Patrones Temporales de Retrasos")

    col1, col2 = st.columns(2)

    # % retrasos por período del día
    if "PERIODO_SALIDA" in df.columns:
        orden_periodos = ["Madrugada", "Mañana", "Tarde", "Noche"]
        df_periodo = (
            df.groupby("PERIODO_SALIDA", observed=True)["RETRASADO_LLEGADA"]
            .mean()
            .reset_index()
            .rename(columns={"RETRASADO_LLEGADA": "porc_retraso"})
        )
        df_periodo["porc_retraso"] = df_periodo["porc_retraso"] * 100
        df_periodo["PERIODO_SALIDA"] = pd.Categorical(
            df_periodo["PERIODO_SALIDA"],
            categories=[p for p in orden_periodos if p in df_periodo["PERIODO_SALIDA"].unique()],
            ordered=True
        )
        df_periodo = df_periodo.sort_values("PERIODO_SALIDA")

        fig_periodo = px.bar(
            df_periodo,
            x="PERIODO_SALIDA",
            y="porc_retraso",
            labels={"PERIODO_SALIDA": "Período de salida", "porc_retraso": "% retrasos llegada"},
            title="% de retrasos en llegada por período del día",
            text=df_periodo["porc_retraso"].round(1).astype(str) + "%",
        )
        col1.plotly_chart(fig_periodo, use_container_width=True)
    else:
        col1.info("No se encontró la columna PERIODO_SALIDA en el dataset.")

    # % retrasos por día de la semana
    df_dia = (
        df.groupby(["DAY_OF_WEEK", "DAY_OF_WEEK_NOMBRE"], observed=True)["RETRASADO_LLEGADA"]
        .mean()
        .reset_index()
        .rename(columns={"RETRASADO_LLEGADA": "porc_retraso"})
    )
    df_dia["porc_retraso"] = df_dia["porc_retraso"] * 100
    df_dia = df_dia.sort_values("DAY_OF_WEEK")

    fig_dia = px.bar(
        df_dia,
        x="DAY_OF_WEEK_NOMBRE",
        y="porc_retraso",
        labels={"DAY_OF_WEEK_NOMBRE": "Día de la semana", "porc_retraso": "% retrasos llegada"},
        title="% de retrasos en llegada por día de la semana",
        text=df_dia["porc_retraso"].round(1).astype(str) + "%",
    )
    col2.plotly_chart(fig_dia, use_container_width=True)

    st.markdown("---")

    # Heatmap Día vs Hora de salida
    st.markdown("### Mapa de calor: Día de la semana vs. Hora de salida")

    if "HORA_SALIDA" in df.columns:
        df_heat = (
            df.groupby(["DAY_OF_WEEK_NOMBRE", "HORA_SALIDA"], observed=True)["RETRASADO_LLEGADA"]
            .mean()
            .reset_index()
            .rename(columns={"RETRASADO_LLEGADA": "porc_retraso"})
        )
        df_heat["porc_retraso"] = df_heat["porc_retraso"] * 100

        fig_heat = px.density_heatmap(
            df_heat,
            x="HORA_SALIDA",
            y="DAY_OF_WEEK_NOMBRE",
            z="porc_retraso",
            color_continuous_scale="Reds",
            labels={
                "HORA_SALIDA": "Hora de salida programada",
                "DAY_OF_WEEK_NOMBRE": "Día de la semana",
                "porc_retraso": "% retrasos llegada",
            },
            title="% de retrasos en llegada por día y hora de salida",
        )
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("No se encontró la columna HORA_SALIDA en el dataset.")


# ============================
# TAB 5 - CAUSAS DE RETRASO
# ============================
with tab_causas:
    st.subheader("Causas de retraso")

    if "MOTIVO_RETRASO" not in df.columns:
        st.info("El dataset no contiene la columna MOTIVO_RETRASO.")
    else:
        # Solo vuelos con retraso en llegada
        df_retrasados = df[df["RETRASADO_LLEGADA"] == 1]

        if df_retrasados.empty:
            st.info("No hay vuelos marcados como retrasados en llegada para analizar causas.")
        else:
            # Conteo de causas
            top_causas = (
                df_retrasados["MOTIVO_RETRASO"]
                .value_counts()
                .reset_index()
            )
            # Renombramos columnas de forma explícita
            top_causas.columns = ["MOTIVO_RETRASO", "TOTAL_VUELOS_RETRASADOS"]

            st.markdown("### Ranking de causas de retraso (vuelos retrasados en llegada)")

            fig_causas = px.bar(
                top_causas.head(15),
                x="TOTAL_VUELOS_RETRASADOS",
                y="MOTIVO_RETRASO",
                orientation="h",
                labels={
                    "TOTAL_VUELOS_RETRASADOS": "Número de vuelos retrasados",
                    "MOTIVO_RETRASO": "Motivo de retraso"
                },
                title="Top causas de retraso en llegada",
                text=top_causas.head(15)["TOTAL_VUELOS_RETRASADOS"],
            )
            fig_causas.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig_causas, use_container_width=True)  # o width="stretch"

            # Tabla detallada
            st.markdown("### Tabla de frecuencia de motivos de retraso")
            st.dataframe(top_causas)

            # Pareto simple
            top_causas["porc_acum"] = (
                top_causas["TOTAL_VUELOS_RETRASADOS"].cumsum()
                / top_causas["TOTAL_VUELOS_RETRASADOS"].sum()
                * 100
            )

            st.markdown("### Porcentaje acumulado de causas (Pareto)")
            st.dataframe(
                top_causas[["MOTIVO_RETRASO", "TOTAL_VUELOS_RETRASADOS", "porc_acum"]]
            )


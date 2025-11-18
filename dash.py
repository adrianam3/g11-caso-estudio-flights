# ============================================================
# dashboard_flights.py – Dashboard Completo
# Dataset: flights_clean.csv
# Filtros independientes por TAB (arriba)
# ============================================================

import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

# ==========================
# Configuración de página
# ==========================
st.set_page_config(
    page_title="Dashboard – Retrasos de Vuelos",
    layout="wide"
)

st.title("✈️ Dashboard de Análisis – Retrasos de Vuelos")
st.caption("Dataset vuelos domésticos EE.UU. 2015 – flights_clean.csv")


# ==========================
# Rutas del proyecto
# ==========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

FLIGHTS_PATH = os.path.join(DATA_DIR, "flights_clean.csv")


# ==========================
# Carga de dataset
# ==========================
@st.cache_data
def cargar_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

try:
    df = cargar_csv(FLIGHTS_PATH)
except Exception as e:
    st.error(f"❌ No se pudo cargar flights_clean.csv\n{e}")
    st.stop()


# ==========================
# Limpieza mínima
# ==========================
if "ARRIVAL_DELAY" in df.columns:
    df["RETRASADO"] = (df["ARRIVAL_DELAY"] > 15).astype(int)
else:
    df["RETRASADO"] = 0  # fallback si no existe


# ============================================================
# TABS principales
# ============================================================
tab_overview, tab_airlines, tab_airports, tab_routes, tab_prediccion = st.tabs(
    ["📊 Resumen", "🛫 Aerolíneas", "🏙 Aeropuertos", "🛣 Rutas", "Predicción Retraso de Vuelos"]
)

# ############################################################
# TAB 1 – RESUMEN GENERAL
# ############################################################
with tab_overview:

    st.subheader("📌 Filtros – Resumen General")

    # Estado inicial
    if "flt1_airline" not in st.session_state:
        st.session_state["flt1_airline"] = df["AIRLINE_NAME"].unique().tolist()
    if "flt1_origin" not in st.session_state:
        st.session_state["flt1_origin"] = df["ORIGEN_AEROPUERTO"].unique().tolist()
    if "flt1_dest" not in st.session_state:
        st.session_state["flt1_dest"] = df["DEST_AEROPUERTO"].unique().tolist()
    if "flt1_month" not in st.session_state:
        st.session_state["flt1_month"] = (df["MONTH"].min(), df["MONTH"].max())
    if "flt1_hour" not in st.session_state:
        st.session_state["flt1_hour"] = (0, 23)

    col1, col2, col3 = st.columns(3)
    with col1:
        flt1_airline = st.multiselect(
            "Aerolínea",
            sorted(df["AIRLINE_NAME"].unique()),
            default=st.session_state["flt1_airline"],
            key="flt1_airline"
        )
    with col2:
        flt1_origin = st.multiselect(
            "Origen",
            sorted(df["ORIGEN_AEROPUERTO"].unique()),
            default=st.session_state["flt1_origin"],
            key="flt1_origin"
        )
    with col3:
        flt1_dest = st.multiselect(
            "Destino",
            sorted(df["DEST_AEROPUERTO"].unique()),
            default=st.session_state["flt1_dest"],
            key="flt1_dest"
        )

    col4, col5 = st.columns(2)
    with col4:
        flt1_month = st.slider(
            "Mes",
            int(df["MONTH"].min()), int(df["MONTH"].max()),
            value=st.session_state["flt1_month"],
            key="flt1_month"
        )
    with col5:
        flt1_hour = st.slider(
            "Hora salida",
            0, 23,
            value=st.session_state["flt1_hour"],
            key="flt1_hour"
        )

    # Aplicar filtros TAB 1
    df_overview = df[
        (df["AIRLINE_NAME"].isin(flt1_airline)) &
        (df["ORIGEN_AEROPUERTO"].isin(flt1_origin)) &
        (df["DEST_AEROPUERTO"].isin(flt1_dest)) &
        (df["MONTH"] >= flt1_month[0]) &
        (df["MONTH"] <= flt1_month[1])
    ]

    if "HORA_SALIDA" in df.columns:
        df_overview = df_overview[
            (df_overview["HORA_SALIDA"] >= flt1_hour[0]) &
            (df_overview["HORA_SALIDA"] <= flt1_hour[1])
        ]

    # ---------------- KPIs ----------------
    st.markdown("### 📊 Indicadores generales")

    colA, colB, colC, colD = st.columns(4)

    total = len(df_overview)
    retrasados = df_overview["RETRASADO"].mean() * 100
    delay_lleg = df_overview["ARRIVAL_DELAY"].mean() if "ARRIVAL_DELAY" in df_overview else 0
    delay_sal = df_overview["DEPARTURE_DELAY"].mean() if "DEPARTURE_DELAY" in df_overview else 0

    colA.metric("Total vuelos", f"{total:,}")
    colB.metric("% retrasados", f"{retrasados:0.1f}%")
    colC.metric("Retraso llegada", f"{delay_lleg:0.1f} min")
    colD.metric("Retraso salida", f"{delay_sal:0.1f} min")

    # ---------------- Gráfico por mes ----------------
    st.markdown("---")
    st.subheader("📅 % de vuelos retrasados por mes")

    df_mes = df_overview.groupby("MONTH")["RETRASADO"].mean().reset_index()
    df_mes["PORC"] = df_mes["RETRASADO"] * 100

    fig_mes = px.line(
        df_mes,
        x="MONTH", y="PORC",
        markers=True,
        title="% de retrasos por mes"
    )
    st.plotly_chart(fig_mes, use_container_width=True)


# ############################################################
# TAB 2 – AEROLÍNEAS
# ############################################################
with tab_airlines:

    st.subheader("📌 Filtros – Aerolíneas")

    if "flt2_airline" not in st.session_state:
        st.session_state["flt2_airline"] = df["AIRLINE_NAME"].unique().tolist()
    if "flt2_month" not in st.session_state:
        st.session_state["flt2_month"] = (df["MONTH"].min(), df["MONTH"].max())

    col1, col2 = st.columns(2)

    with col1:
        flt2_airline = st.multiselect(
            "Aerolínea",
            sorted(df["AIRLINE_NAME"].unique()),
            default=st.session_state["flt2_airline"],
            key="flt2_airline"
        )
    with col2:
        flt2_month = st.slider(
            "Mes",
            int(df["MONTH"].min()), int(df["MONTH"].max()),
            value=st.session_state["flt2_month"],
            key="flt2_month"
        )

    df_airlines_f = df[
        (df["AIRLINE_NAME"].isin(flt2_airline)) &
        (df["MONTH"] >= flt2_month[0]) &
        (df["MONTH"] <= flt2_month[1])
    ]

    st.markdown("### Ranking de aerolíneas por retrasos")

    df_air = df_airlines_f.groupby("AIRLINE_NAME")["RETRASADO"] \
        .mean().reset_index().sort_values("RETRASADO", ascending=False)
    df_air["PORC"] = df_air["RETRASADO"] * 100

    fig_air = px.bar(
        df_air.head(20),
        x="AIRLINE_NAME", y="PORC",
        title="% de vuelos retrasados por aerolínea",
        color="PORC"
    )
    st.plotly_chart(fig_air, use_container_width=True)

    st.dataframe(df_air.head(20))


# ############################################################
# TAB 3 – AEROPUERTOS
# ############################################################
with tab_airports:

    st.subheader("📌 Filtros – Aeropuertos")

    if "flt3_dest" not in st.session_state:
        st.session_state["flt3_dest"] = df["DEST_AEROPUERTO"].unique().tolist()

    flt3_dest = st.multiselect(
        "Aeropuerto destino",
        sorted(df["DEST_AEROPUERTO"].unique()),
        default=st.session_state["flt3_dest"],
        key="flt3_dest"
    )

    df_airp_f = df[df["DEST_AEROPUERTO"].isin(flt3_dest)]

    st.subheader("Top aeropuertos destino por % retrasos")

    df_airp = df_airp_f.groupby("DEST_AEROPUERTO")["RETRASADO"] \
        .mean().reset_index().sort_values("RETRASADO", ascending=False)
    df_airp["PORC"] = df_airp["RETRASADO"] * 100

    fig_airp = px.bar(
        df_airp.head(20),
        x="DEST_AEROPUERTO", y="PORC",
        title="Aeropuertos con mayor retraso",
        color="PORC"
    )
    st.plotly_chart(fig_airp, use_container_width=True)

    st.dataframe(df_airp.head(30))


# ############################################################
# TAB 4 – RUTAS
# ############################################################
with tab_routes:

    st.subheader("📌 Filtros – Rutas")

    if "flt4_origin" not in st.session_state:
        st.session_state["flt4_origin"] = df["ORIGEN_AEROPUERTO"].unique().tolist()
    if "flt4_dest" not in st.session_state:
        st.session_state["flt4_dest"] = df["DEST_AEROPUERTO"].unique().tolist()

    col1, col2 = st.columns(2)

    with col1:
        flt4_origin = st.multiselect(
            "Origen",
            sorted(df["ORIGEN_AEROPUERTO"].unique()),
            default=st.session_state["flt4_origin"],
            key="flt4_origin"
        )
    with col2:
        flt4_dest = st.multiselect(
            "Destino",
            sorted(df["DEST_AEROPUERTO"].unique()),
            default=st.session_state["flt4_dest"],
            key="flt4_dest"
        )

    df_routes_f = df[
        (df["ORIGEN_AEROPUERTO"].isin(flt4_origin)) &
        (df["DEST_AEROPUERTO"].isin(flt4_dest))
    ]

    st.subheader("🛣 Peores rutas origen → destino")

    df_routes = df_routes_f.groupby(
        ["ORIGEN_AEROPUERTO", "DEST_AEROPUERTO"]
    )["RETRASADO"].mean().reset_index().sort_values("RETRASADO", ascending=False)
    df_routes["PORC"] = df_routes["RETRASADO"] * 100

    st.dataframe(df_routes.head(30))

# ############################################################
# TAB 5 – PREDICCIÓN RETRASO VUELOS
# ############################################################
with tab_prediccion:

    st.subheader("📌 Predicción Retraso Vuelos – ")
    st.info("En construcción...")
    # Aquí iría el código para la predicción de retrasos de vuelos
    
    
    st.dataframe(df_routes.head(30))
    
import os
import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
import joblib
from pathlib import Path
from datetime import datetime
from plotly.subplots import make_subplots

import io

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

# titulo
st.title("✈️ Dashboard Predicción de Retrasos de Vuelos en la Industria Aérea. ✈️")
st.caption("Caso de Estudio | Grupo 11 | Integrantes: Farinango Mario / Adrián Merlo")
st.subheader("Análisis Exploratorio de Datos y Predicción Predicción de Retrasos de Vuelos")

# # ============================
# # CARGA DE DATOS (+ 5 M - registros )
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

# -------------------------
# Config y rutas 
# -------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "flight_delay_lgbm.joblib"
PREPROCESSOR_PATH = MODELS_DIR / "flight_delay_preprocessors.joblib"

# Log de predicciones
PREDICTIONS_LOG = PROJECT_ROOT / "predictions_log.csv"

# -------------------------
# Diccionario de aerolíneas 
# -------------------------
# AIRLINES_FULL = {
#     'AS': 'Alaska Airlines Inc.', 
#     'AA': 'American Airlines Inc.', 
#     'US': 'US Airways Inc.', 
#     'DL': 'Delta Air Lines Inc.', 
#     'NK': 'Spirit Air Lines', 
#     'UA': 'United Air Lines Inc.', 
#     'HA': 'Hawaiian Airlines Inc.', 
#     'B6': 'JetBlue Airways', 
#     'OO': 'Skywest Airlines Inc.', 
#     'EV': 'Atlantic Southeast Airlines', 
#     'F9': 'Frontier Airlines Inc.', 
#     'WN': 'Southwest Airlines Co.', 
#     'MQ': 'American Eagle Airlines Inc.', 
#     'VX': 'Virgin America'
# }



def hhmm_to_hhmmss(v):
    try:
        v = int(v)
        h = v // 100
        m = v % 100
        return f"{h:02d}:{m:02d}:00"
    except Exception:
        return "N/A"

def ensure_log():
    if not PREDICTIONS_LOG.exists():
        pd.DataFrame(columns=[
            "timestamp_utc","airline","origin_code","origin_name","dest_code","dest_name",
            "month","day_of_week","scheduled_dep","scheduled_arr","scheduled_time","distance","prob_delay"
        ]).to_csv(PREDICTIONS_LOG, index=False)

def append_log(record: dict):
    try:
        ensure_log()
        df_old = pd.read_csv(PREDICTIONS_LOG)
        df_new = pd.concat([df_old, pd.DataFrame([record])], ignore_index=True)
        df_new.to_csv(PREDICTIONS_LOG, index=False)
    except Exception as e:
        st.warning(f"No se pudo guardar el log de predicción: {e}")

def get_log_bytes():
    if PREDICTIONS_LOG.exists():
        return PREDICTIONS_LOG.read_bytes()
    return None

# -------------------------
# 1) Cargar artefactos
# -------------------------
@st.cache_resource
def load_artifacts():
    try:
        preprocessors = None
        model = None
        if PREPROCESSOR_PATH.exists():
            preprocessors = joblib.load(PREPROCESSOR_PATH)
        else:
            st.warning(f"No se encontró {PREPROCESSOR_PATH.name} en /models/")

        if MODEL_PATH.exists():
            try:
                model = joblib.load(MODEL_PATH)
            except Exception as e:
                st.warning(f"No se pudo cargar el modelo: {e}")
                st.warning("Si aparece 'Booster' object has no attribute 'handle' o errores de dll, reinstala la versión de lightgbm con la que entrenaste.")
                model = None
        else:
            st.warning(f"No se encontró {MODEL_PATH.name} en /models/")

        if model is None or preprocessors is None:
            return None

        artifacts = {
            "model": model,
            "label_encoders": preprocessors.get("label_encoders", {}),
            "scaler": preprocessors.get("scaler", None),
            "cat_cols": preprocessors.get("cat_features_names", []),
            "num_cols": preprocessors.get("num_features_names", [])
        }
        artifacts["feature_order"] = artifacts["cat_cols"] + artifacts["num_cols"]
        return artifacts

    except Exception as e:
        st.error(f"Error al cargar artefactos: {e}")
        return None



artifacts = load_artifacts()

# # === DEBUG opcional: columnas que espera el modelo ===
# if artifacts is not None:
#     st.sidebar.markdown("### 🔍 Debug modelo")
#     if st.sidebar.checkbox("Mostrar columnas esperadas por el modelo"):
#         st.write("✅ Columnas esperadas (feature_order):")
#         st.write(artifacts["feature_order"])

artifacts = load_artifacts()

# # === DEBUG opcional: columnas que espera el modelo ===
# if artifacts is not None:
#     st.sidebar.markdown("### 🔍 Debug modelo")

#     if st.sidebar.checkbox("Mostrar columnas esperadas por el modelo"):
#         st.write("✅ Columnas esperadas (feature_order, salida del preprocesador):")
#         st.write(artifacts["feature_order"])

#         # Si guardaste también las columnas de entrada:
#         input_feats = artifacts.get("input_features", None)
#         if input_feats is not None:
#             st.write("📥 Columnas de entrada que espera el preprocesador (input_features):")
#             st.write(input_feats)

#             # Comparar con lo que realmente estás enviando
#             st.write("🧪 Columnas de df_input (antes de preprocesar):")
#             st.write(df_input.columns.tolist())
#         else:
#             st.warning("⚠️ El artefacto no tiene 'input_features' guardado. Solo se muestran 'feature_order'.")
# -------------------------
@st.cache_data
def cargar_datos(path: str, n_muestra: int = 50000, seed: int = 42) -> pd.DataFrame:
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
    
# def cargar_tabla_ruta():
#     # tabla agregada por ruta (PARA PREDICCIONES)
#     tabla = (
#         flights.groupby(["AIRLINE","ORIGIN_AIRPORT","DESTINATION_AIRPORT"], dropna=False)
#           .agg(DISTANCIA_HAV=("DISTANCE","mean"),
#                SCHEDULED_TIME=("SCHEDULED_TIME","median"),
#                SCHEDULED_ARRIVAL=("SCHEDULED_ARRIVAL","median"))
#           .reset_index()
#     )

#     for c in ["DISTANCIA_HAV","SCHEDULED_TIME","SCHEDULED_ARRIVAL"]:
#         if c in tabla.columns:
#             tabla[c] = pd.to_numeric(tabla[c], errors="coerce")

#     return tabla

# tabla_rutas = cargar_tabla_ruta()
@st.cache_data
def cargar_tabla_ruta(df: pd.DataFrame) -> pd.DataFrame:
    """Tabla agregada por ruta (para predicciones) cacheada."""
    tabla = (
        df.groupby(["AIRLINE","ORIGIN_AIRPORT","DESTINATION_AIRPORT"], dropna=False)
          .agg(
              DISTANCIA_HAV=("DISTANCE", "mean"),
              SCHEDULED_TIME=("SCHEDULED_TIME", "median"),
              SCHEDULED_ARRIVAL=("SCHEDULED_ARRIVAL", "median")
          )
          .reset_index()
    )

    for c in ["DISTANCIA_HAV","SCHEDULED_TIME","SCHEDULED_ARRIVAL"]:
        if c in tabla.columns:
            tabla[c] = pd.to_numeric(tabla[c], errors="coerce")

    return tabla

# Usar la versión cacheada
tabla_rutas = cargar_tabla_ruta(flights)

@st.cache_data
def get_origen_df(df: pd.DataFrame) -> pd.DataFrame:
    """Catálogo de aeropuertos origen (code, name)."""
    return (
        df[["ORIGIN_AIRPORT", "ORIGEN_CIUDAD"]]
        .drop_duplicates()
        .rename(columns={"ORIGIN_AIRPORT": "code", "ORIGEN_CIUDAD": "name"})
    )

@st.cache_data
def get_destino_df(df: pd.DataFrame) -> pd.DataFrame:
    """Catálogo de aeropuertos destino (code, name)."""
    return (
        df[["DESTINATION_AIRPORT", "DEST_CIUDAD"]]
        .drop_duplicates()
        .rename(columns={"DESTINATION_AIRPORT": "code", "DEST_CIUDAD": "name"})
    )

@st.cache_data
def get_airline_options(df: pd.DataFrame):
    """Lista de opciones de aerolínea para el selectbox."""
    airline_codes = sorted(df["AIRLINE"].dropna().unique().tolist())
    airline_options = [f"{c} — {AIRLINES_FULL.get(c, c)}" for c in airline_codes]
    return airline_options

@st.cache_data
def get_origen_por_aerolinea(df: pd.DataFrame) -> pd.DataFrame:
    """
    Catálogo de (AIRLINE, ORIGIN_AIRPORT, ORIGEN_CIUDAD) único.
    Lo usamos para filtrar orígenes por aerolínea sin escanear todos los vuelos cada vez.
    """
    return df[["AIRLINE", "ORIGIN_AIRPORT", "ORIGEN_CIUDAD"]].drop_duplicates()


#
# PREDICCIONES
# 

# -------------------------
# Preprocess para inference
# -------------------------
def preprocess_data_for_api(df: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    df = df.copy()
    if artifacts is None:
        raise ValueError("Artifacts is None.")

    label_encoders = artifacts["label_encoders"]
    scaler = artifacts["scaler"]
    CAT_COLS = artifacts["cat_cols"]
    NUM_COLS = artifacts["num_cols"]
    FINAL_FEATURE_ORDER = artifacts["feature_order"]

    # RUTA
    df["RUTA"] = df["ORIGIN_AIRPORT"].astype(str) + "_" + df["DESTINATION_AIRPORT"].astype(str)

    # SALIDA cíclica
    sched_dep = pd.to_numeric(df["SCHEDULED_DEPARTURE"], errors="coerce").fillna(0).astype(int)
    hs = (sched_dep // 100).clip(0,23)
    ms = (sched_dep % 100).clip(0,59)
    minuto_dia_salida = hs * 60 + ms
    df["SALIDA_SIN"] = np.sin(2 * np.pi * minuto_dia_salida / (24*60))
    df["SALIDA_COS"] = np.cos(2 * np.pi * minuto_dia_salida / (24*60))

    # LLEGADA cíclica
    sched_arr = pd.to_numeric(df.get("SCHEDULED_ARRIVAL", 0), errors="coerce").fillna(0).astype(int)
    hl = (sched_arr // 100).clip(0,23)
    ml = (sched_arr % 100).clip(0,59)
    minuto_dia_llegada = hl * 60 + ml
    df["LLEGADA_SIN"] = np.sin(2 * np.pi * minuto_dia_llegada / (24*60))
    df["LLEGADA_COS"] = np.cos(2 * np.pi * minuto_dia_llegada / (24*60))

    # Mes cíclico
    df["MONTH_SIN"] = np.sin(2 * np.pi * df["MONTH"].astype(float) / 12)
    df["MONTH_COS"] = np.cos(2 * np.pi * df["MONTH"].astype(float) / 12)

    # DISTANCIA_HAV
    if "DISTANCE" in df.columns and "DISTANCIA_HAV" not in df.columns:
        df["DISTANCIA_HAV"] = df["DISTANCE"]

    # Label encoding con manejo de <unknown>
    for col in CAT_COLS:
        if col in label_encoders:
            le = label_encoders[col]
            classes_set = set([str(x) for x in le.classes_])
            df[col] = df[col].astype(str).apply(lambda x: x if x in classes_set else "<unknown>")
            try:
                df[col] = le.transform(df[col])
            except Exception:
                mapping = {c:i for i,c in enumerate(le.classes_)}
                df[col] = df[col].apply(lambda x: mapping.get(x, mapping.get("<unknown>", 0)))
        else:
            if col not in df.columns:
                df[col] = 0

    # Scaling numérico (con fallback)
    cols_to_scale = [c for c in NUM_COLS if c in df.columns]
    if scaler is not None and len(cols_to_scale) > 0:
        try:
            df[cols_to_scale] = scaler.transform(df[cols_to_scale])
        except Exception:
            try:
                if hasattr(scaler, "mean_") and hasattr(scaler, "scale_"):
                    for i,c in enumerate(cols_to_scale):
                        mean_i = scaler.mean_[i] if i < len(scaler.mean_) else 0
                        scale_i = scaler.scale_[i] if i < len(scaler.scale_) else 1
                        df[c] = (df[c] - mean_i) / (scale_i + 1e-12)
            except Exception:
                pass

    # Asegurar columnas finales
    for col in FINAL_FEATURE_ORDER:
        if col not in df.columns:
            df[col] = 0

    X = df[FINAL_FEATURE_ORDER].copy()
    return X


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

# Período de llegada
if "PERIODO_LLEGADA" in flights.columns:
    periodos_disp = ["Madrugada", "Mañana", "Tarde", "Noche"]
    periodos_disp = [p for p in periodos_disp if p in flights["PERIODO_LLEGADA"].unique()]
    periodos_sel = st.sidebar.multiselect(
        "Período de llegada",
        options=periodos_disp,
        default=periodos_disp
    )
else:
    periodos_sel = None
    
# Construir el diccionario Aerolineas
AIRLINES_FULL = (
    flights[["AIRLINE", "AIRLINE_NAME"]]
    .drop_duplicates()                      # por si hay muchas filas por aerolínea
    .set_index("AIRLINE")["AIRLINE_NAME"]  # índice = código, valor = nombre
    .to_dict()
)    


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
    df = df[df["PERIODO_LLEGADA"].isin(periodos_sel)]

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
# Helper para convertir minutos del día (0–1439) a cadena "HH:MM"
def minutos_a_hora_str(m):
    try:
        m = int(m)
        h = m // 60
        mi = m % 60
        return f"{h:02d}:{mi:02d}"
    except Exception:
        return "00:00"


# ============================
# TABS PRINCIPALES
# ============================

tab_resumen, tab_aerolineas, tab_aeropuertos, tab_tiempo, tab_causas, tab_prediccion = st.tabs(
    ["Resumen", "Aerolíneas", "Aeropuertos", "Tiempo", "Causas de retraso", "Predicción de retrasos"]
)

# ============================
# TAB 1 - RESUMEN
# ============================
with tab_resumen:
    st.subheader("Resumen Ejecutivo")

    col1, col2, col3, col4, col5 = st.columns(5)

    total_vuelos = len(df)
    total_vuelos_retrasos_sal = df[
        ((df["DEPARTURE_DELAY"]>15) == 1)  
    ]
    total_vuelos_retrasos_sal = len(total_vuelos_retrasos_sal) if analizar_salida else 0.0
    
    total_vuelos_retrasos_lle = df[
        ((df["ARRIVAL_DELAY"]>15) == 1)  
    ]
    total_vuelos_retrasos_lle = len(total_vuelos_retrasos_lle) if analizar_llegada else 0.0
    
    # % retraso llegada y salida
    porc_retraso_lleg = calcular_porcentaje_retraso(df["RETRASADO_LLEGADA"]) if analizar_llegada else 0.0
    porc_retraso_sal = calcular_porcentaje_retraso(df["RETRASADO_SALIDA"]) if analizar_salida else 0.0

    # Retraso promedio llegada (en minutos)
    retraso_prom_lleg = df["ARRIVAL_DELAY"].dropna()
    retraso_prom_lleg = retraso_prom_lleg[retraso_prom_lleg > 15]
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
    flights_delay_d = df[df["DEPARTURE_DELAY"] > 15]

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
    flights_delay = df[df["ARRIVAL_DELAY"] > 15]

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
    # retraso_promedio = flights_delay["ARRIVAL_DELAY"].mean()
    
    retraso_promedio = flights.loc[flights["ARRIVAL_DELAY"] > 15, "ARRIVAL_DELAY"].mean()
    flights_delay = flights[flights["ARRIVAL_DELAY"] > 0].copy()

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

    # ==========================
    # Gráficos de pastel de puntualidad
    # ==========================
    if "ARRIVAL_DELAY" in df.columns:
        # --- A) Clasificación general: Antes de tiempo / A tiempo / Retrasado ---
        if "ESTADO_LLEGADA_CAT" not in df.columns:
            def clasificar_estado_llegada(delay):
                if pd.isna(delay):
                    return "Sin dato"
                if delay < 0:
                    return "Antes de tiempo"
                if delay == 0:
                    return "A tiempo"
                return "Retrasado"

            df["ESTADO_LLEGADA_CAT"] = df["ARRIVAL_DELAY"].apply(clasificar_estado_llegada)

        resumen_puntualidad = (
            df
            .groupby("ESTADO_LLEGADA_CAT", dropna=False)
            .size()
            .reset_index(name="Cantidad")
        )

        orden_cat = ["Antes de tiempo", "A tiempo", "Retrasado", "Sin dato"]
        resumen_puntualidad["ESTADO_LLEGADA_CAT"] = pd.Categorical(
            resumen_puntualidad["ESTADO_LLEGADA_CAT"],
            categories=orden_cat,
            ordered=True
        )
        resumen_puntualidad = resumen_puntualidad.dropna(subset=["ESTADO_LLEGADA_CAT"])

        total_vuelos_cat = int(resumen_puntualidad["Cantidad"].sum())
        if total_vuelos_cat > 0:
            resumen_puntualidad["Porc_sobre_total"] = (
                resumen_puntualidad["Cantidad"] / total_vuelos_cat * 100
            )
        else:
            resumen_puntualidad["Porc_sobre_total"] = 0.0

        # --- B) Clasificación por umbral de 15 minutos ---
        def clasificar_15min(delay):
            if pd.isna(delay):
                return "Sin dato"
            # Consideramos <= 15 como a tiempo
            if delay <= 15:
                return "A tiempo (≤ 15 min)"
            return "Retrasado (> 15 min)"

        df["PUNTUALIDAD_15MIN"] = df["ARRIVAL_DELAY"].apply(clasificar_15min)

        resumen_15 = (
            df
            .groupby("PUNTUALIDAD_15MIN", dropna=False)
            .size()
            .reset_index(name="Cantidad")
        )

        orden_cat_15 = ["A tiempo (≤ 15 min)", "Retrasado (> 15 min)", "Sin dato"]
        resumen_15["PUNTUALIDAD_15MIN"] = pd.Categorical(
            resumen_15["PUNTUALIDAD_15MIN"],
            categories=orden_cat_15,
            ordered=True
        )
        resumen_15 = resumen_15.dropna(subset=["PUNTUALIDAD_15MIN"])

        total_vuelos_15 = int(resumen_15["Cantidad"].sum())
        if total_vuelos_15 > 0:
            resumen_15["Porc_sobre_total"] = (
                resumen_15["Cantidad"] / total_vuelos_15 * 100
            )
        else:
            resumen_15["Porc_sobre_total"] = 0.0

        # --- C) Dibujar los dos pasteles lado a lado ---
        st.markdown("### Análisis de puntualidad de llegadas")

        col_p1, col_p2 = st.columns(2)

        color_map_estado = {
            "Antes de tiempo": "#4CAF50",
            "A tiempo": "#2196F3",
            "Retrasado": "#F44336",
            "Sin dato": "#9E9E9E",
        }

        color_map_15 = {
            "A tiempo (≤ 15 min)": "#4CAF50",
            "Retrasado (> 15 min)": "#F44336",
            "Sin dato": "#9E9E9E",
        }

        # Pastel 1: distribución general
        with col_p1:
            st.markdown("#### Distribución general de puntualidad")
            if not resumen_puntualidad.empty:
                fig_pie_estado = px.pie(
                    resumen_puntualidad,
                    names="ESTADO_LLEGADA_CAT",
                    values="Cantidad",
                    title=f"Total de vuelos por estado de llegada (N = {total_vuelos_cat:,})",
                    color="ESTADO_LLEGADA_CAT",
                    color_discrete_map=color_map_estado,
                )
                fig_pie_estado.update_traces(
                    textposition="inside",
                    textinfo="label+percent",
                )
                st.plotly_chart(fig_pie_estado, use_container_width=True)
            else:
                st.info("No hay datos para la distribución general de puntualidad.")

        # Pastel 2: retrasos > 15 minutos
        with col_p2:
            st.markdown("#### Vuelos con retraso > 15 minutos")
            if not resumen_15.empty:
                fig_pie_15 = px.pie(
                    resumen_15,
                    names="PUNTUALIDAD_15MIN",
                    values="Cantidad",
                    title=f"Vuelos según umbral de 15 minutos (N = {total_vuelos_15:,})",
                    color="PUNTUALIDAD_15MIN",
                    color_discrete_map=color_map_15,
                )
                fig_pie_15.update_traces(
                    textposition="inside",
                    textinfo="label+percent",
                )
                st.plotly_chart(fig_pie_15, use_container_width=True)
            else:
                st.info("No hay datos para el análisis de umbral de 15 minutos.")

    else:
        st.info("El dataset no contiene la columna ARRIVAL_DELAY; no se pueden generar los gráficos de puntualidad.")

    st.markdown("---")


    # ==========================
    # Relación salida vs llegada + Histograma
    # ==========================
    st.markdown("### Análisis de retrasos ")

    col_scatter, col_hist = st.columns(2)

    # -------- Columna 1: Scatter salida vs llegada --------
    with col_scatter:
        st.markdown("#### Retraso salida vs llegada (umbral 15 min)")

        try:
            df_scatter = df[["DEPARTURE_DELAY", "ARRIVAL_DELAY"]].dropna().copy()

            if df_scatter.empty:
                st.info("No hay datos de retraso de llegada y llegada para graficar con los filtros actuales.")
            else:
                # Clasificación a tiempo / retrasado con umbral de 15 minutos
                def clasificar_15min(valor):
                    if pd.isna(valor):
                        return "Sin dato"
                    return "A tiempo (≤ 15 min)" if valor <= 15 else "Retrasado (> 15 min)"

                df_scatter["ESTADO_SALIDA_15"] = df_scatter["DEPARTURE_DELAY"].apply(clasificar_15min)
                df_scatter["ESTADO_LLEGADA_15"] = df_scatter["ARRIVAL_DELAY"].apply(clasificar_15min)

                # Categoría combinada para colorear el scatter
                def combinar_cat(row):
                    if row["ESTADO_SALIDA_15"] == "Sin dato" or row["ESTADO_LLEGADA_15"] == "Sin dato":
                        return "Sin dato"
                    if row["ESTADO_SALIDA_15"].startswith("A tiempo") and row["ESTADO_LLEGADA_15"].startswith("A tiempo"):
                        return "Salida y llegada a tiempo"
                    if row["ESTADO_SALIDA_15"].startswith("A tiempo") and row["ESTADO_LLEGADA_15"].startswith("Retrasado"):
                        return "Salida a tiempo, llegada retrasada"
                    if row["ESTADO_SALIDA_15"].startswith("Retrasado") and row["ESTADO_LLEGADA_15"].startswith("A tiempo"):
                        return "Salida retrasada, llegada a tiempo"
                    return "Salida y llegada retrasadas"

                df_scatter["CATEGORIA_15"] = df_scatter.apply(combinar_cat, axis=1)

                # Muestra aleatoria para no saturar el navegador
                max_puntos = 50000
                if len(df_scatter) > max_puntos:
                    df_scatter = df_scatter.sample(max_puntos, random_state=42)

                # Rango 1–99 percentil + aire
                x_min, x_max = df_scatter["DEPARTURE_DELAY"].quantile([0.01, 0.99])
                y_min, y_max = df_scatter["ARRIVAL_DELAY"].quantile([0.01, 0.99])

                x_min = float(x_min) - 5
                x_max = float(x_max) + 5
                y_min = float(y_min) - 5
                y_max = float(y_max) + 5

                color_map_cat = {
                    "Salida y llegada a tiempo": "#4CAF50",
                    "Salida a tiempo, llegada retrasada": "#FFC107",
                    "Salida retrasada, llegada a tiempo": "#03A9F4",
                    "Salida y llegada retrasadas": "#F44336",
                    "Sin dato": "#9E9E9E",
                }

                fig_rel = px.scatter(
                    df_scatter,
                    x="DEPARTURE_DELAY",
                    y="ARRIVAL_DELAY",
                    color="CATEGORIA_15",
                    color_discrete_map=color_map_cat,
                    labels={
                        "DEPARTURE_DELAY": "Retraso en salida (min)",
                        "ARRIVAL_DELAY": "Retraso en llegada (min)",
                        "CATEGORIA_15": "Clasificación (umbral 15 min)",
                    },
                    title="Retraso salida vs llegada (umbral 15 min)",
                    opacity=0.6,
                )

                # Líneas umbral 15 min
                linea_umbral_color = "#D218AD"
                fig_rel.add_hline(y=15, line_dash="dot", line_color=linea_umbral_color, line_width=2)
                fig_rel.add_vline(x=15, line_dash="dot", line_color=linea_umbral_color, line_width=2)

                # Rango, fondo y LEYENDA ABAJO
                fig_rel.update_layout(
                    xaxis=dict(range=[x_min, x_max]),
                    yaxis=dict(range=[y_min, y_max]),
                    plot_bgcolor="rgba(255, 248, 225, 0.6)",
                    legend_title="Categoría",
                    legend=dict(
                        orientation="h",
                        yanchor="top",
                        y=-0.2,          # debajo del gráfico
                        xanchor="center",
                        x=0.5
                    ),
                )

                st.plotly_chart(fig_rel, use_container_width=True)

        except KeyError as e:
            st.warning(f"No se pudo generar el gráfico de relación salida vs llegada: {e}")

    # -------- Columna 2: Histograma ARRIVAL_DELAY --------
    with col_hist:
        st.markdown("#### Histograma de retraso en llegada")

        if "ARRIVAL_DELAY" not in df.columns:
            st.warning("El dataset no contiene la columna ARRIVAL_DELAY.")
        else:
            delay_min, delay_max = -20, 300
            df_hist = (
                df[df["ARRIVAL_DELAY"].between(delay_min, delay_max)]
                [["ARRIVAL_DELAY"]]
                .dropna()
                .copy()
            )

            if df_hist.empty:
                st.info("No hay datos de retraso en llegada en el rango seleccionado.")
            else:
                total_vuelos_hist = len(df_hist)

                rango_inf, rango_sup = -15, 15
                en_rango = df_hist[
                    df_hist["ARRIVAL_DELAY"].between(rango_inf, rango_sup)
                ].shape[0]
                porc_en_rango = en_rango / total_vuelos_hist * 100

                fig_hist = px.histogram(
                    df_hist,
                    x="ARRIVAL_DELAY",
                    nbins=50,
                    title=f"Histograma de retraso en llegada (entre {delay_min} y {delay_max} min)",
                    labels={
                        "ARRIVAL_DELAY": "Retraso en llegada (min)",
                        "count": "Número de vuelos",
                    },
                )

                linea_umbral_color = "#455A64"
                fig_hist.add_vline(
                    x=0,
                    line_dash="dot",
                    line_color="#9E9E9E",
                    line_width=1,
                    annotation_text="0 min",
                    annotation_position="top left",
                )
                fig_hist.add_vline(
                    x=15,
                    line_dash="dot",
                    line_color=linea_umbral_color,
                    line_width=2,
                    annotation_text="+15 min",
                    annotation_position="top right",
                )

                fig_hist.update_layout(
                    plot_bgcolor="rgba(255, 248, 225, 0.6)",
                    xaxis_title="Retraso en llegada (min)",
                    yaxis_title="Número de vuelos",
                )

                st.plotly_chart(fig_hist, use_container_width=True)

                st.caption(
                    f"En el rango [{rango_inf} min, {rango_sup} min] se encuentran "
                    f"**{en_rango:,} vuelos**, lo que representa aproximadamente "
                    f"**{porc_en_rango:.1f}%** de los vuelos considerados en este histograma."
                )
        
    st.markdown("---")

    # ==========================
    # Tendencia mensual: Total de vuelos vs % de retrasos (> 15 min)
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

    # ---------------------------------
    # NUEVO: retraso definido por ARRIVAL_DELAY > 15
    # ---------------------------------
    if "ARRIVAL_DELAY" not in df.columns:
        st.error("No se encontró la columna ARRIVAL_DELAY para calcular los retrasos > 15 min.")
    else:
        df["ARRIVAL_DELAY"] = pd.to_numeric(df["ARRIVAL_DELAY"], errors="coerce")
        df["RETRASADO_15"] = (df["ARRIVAL_DELAY"] > 15).astype(int)

        # Resumen mensual: totales y # de retrasados (> 15 min)
        resumen_mes = (
            df.groupby("MONTH", observed=True)["RETRASADO_15"]
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

            fig = go.Figure()

            # ==========================
            # BARRAS = Total de vuelos
            # ==========================
            fig.add_trace(go.Bar(
                x=resumen_mes["MES_NOMBRE"],
                y=resumen_mes["Total"],
                name="Total de vuelos",
                marker_color="#1976D2",  # azul vivo
                text=resumen_mes["Total"].apply(lambda x: f"{x:,}"),
                textposition="inside",
                textfont=dict(color="white", size=11),
                yaxis="y1",
                hovertemplate="<b>%{x}</b><br>Total de vuelos: %{y:,}<extra></extra>"
            ))

            # ==========================
            # LÍNEA = % de retrasos (> 15 min)
            # ==========================
            fig.add_trace(go.Scatter(
                x=resumen_mes["MES_NOMBRE"],
                y=resumen_mes["Porc_Retrasado"],
                name="% Retrasos (> 15 min)",
                mode="lines+markers+text",
                line=dict(color="#F57C00", width=3),      # naranja intenso
                marker=dict(
                    size=10,
                    color=colors,
                    line=dict(color="#BF360C", width=1.5)
                ),
                text=[f"{v:.2f}%" for v in resumen_mes["Porc_Retrasado"]],
                textposition="bottom center",
                textfont=dict(color="white", size=12),
                yaxis="y2",
                hovertemplate="<b>%{x}</b><br>% Retrasos (> 15 min): %{y:.2f}%<extra></extra>"
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
                        line=dict(color="#090CE8", width=4, dash="dot")
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
                    "Tendencia mensual: Total de vuelos vs % de retrasos (> 15 min)<br>"
                    f"<sup>✈️ Total periodo filtrado: {total_vuelos_anual:,} vuelos | "
                    f"Promedio de retrasos (> 15 min): {promedio_retrasos_anual:.2f}%</sup>"
                ),
                xaxis=dict(title="Mes"),
                yaxis=dict(title="Total de vuelos", side="left", showgrid=False),
                yaxis2=dict(
                    title="% Retrasos (> 15 min)",
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


    # ======================================================
    # Vuelos a tiempo vs retrasados (> 15 min) por aerolínea
    # ======================================================
    st.markdown("### Vuelos a tiempo vs retrasados (> 15 min) por aerolínea")

    # Verificamos columnas necesarias
    if "ARRIVAL_DELAY" not in df.columns or "AIRLINE" not in df.columns:
        st.warning("No se encontraron las columnas necesarias (ARRIVAL_DELAY, AIRLINE) para este gráfico.")
    else:
        # Trabajamos sobre el df filtrado actual
        v_retrasos = df.copy()

        # Asegurar que ARRIVAL_DELAY es numérico
        v_retrasos["ARRIVAL_DELAY"] = pd.to_numeric(
            v_retrasos["ARRIVAL_DELAY"], errors="coerce"
        )

        # Retrasado si ARRIVAL_DELAY > 15 min
        v_retrasos["RETRASADO_15"] = (v_retrasos["ARRIVAL_DELAY"] > 15).astype(int)

        # === Agrupación por aerolínea ===
        # Si existe AIRLINE_NAME la usamos, si no solo AIRLINE
        if "AIRLINE_NAME" in v_retrasos.columns:
            group_cols = ["AIRLINE", "AIRLINE_NAME"]
        else:
            group_cols = ["AIRLINE"]

        resumen_retrasos = (
            v_retrasos
            .groupby(group_cols, observed=True)["RETRASADO_15"]
            .agg(
                Total="size",
                Retrasados="sum",
            )
            .reset_index()
        )

        # Si no hay datos con los filtros actuales, salir
        if resumen_retrasos.empty:
            st.info("No hay datos suficientes para generar el gráfico de aerolíneas con los filtros actuales.")
        else:
            # Vuelos a tiempo y porcentaje
            resumen_retrasos["A_tiempo"] = resumen_retrasos["Total"] - resumen_retrasos["Retrasados"]
            resumen_retrasos["Porcentaje_Retrasos"] = (
                resumen_retrasos["Retrasados"] / resumen_retrasos["Total"] * 100
            )

            # Nombre completo de aerolínea
            if "AIRLINE_NAME" in resumen_retrasos.columns:
                resumen_retrasos["AEROLINEA_FULL"] = (
                    resumen_retrasos["AIRLINE"] + " - " + resumen_retrasos["AIRLINE_NAME"]
                )
            else:
                resumen_retrasos["AEROLINEA_FULL"] = resumen_retrasos["AIRLINE"]

            # (Opcional) filtrar aerolíneas con pocos vuelos
            # resumen_retrasos = resumen_retrasos[resumen_retrasos["Total"] >= 100]

            # Ordenar por TOTAL de vuelos, de mayor a menor (más vuelos a la izquierda)
            resumen_retrasos = resumen_retrasos.sort_values("Total", ascending=False)

            # ------- Formato largo para barras apiladas -------
            df_long = resumen_retrasos.melt(
                id_vars=["AEROLINEA_FULL", "Total", "Porcentaje_Retrasos"],
                value_vars=["A_tiempo", "Retrasados"],
                var_name="Estado",
                value_name="Cantidad"
            )

            estado_map = {
                "A_tiempo": "A tiempo (≤ 15 min)",
                "Retrasados": "Retrasados (> 15 min)"
            }
            df_long["Estado"] = df_long["Estado"].map(estado_map)

            # Texto dentro de cada segmento: solo la cantidad
            df_long["Texto_cant"] = df_long["Cantidad"].map(lambda x: f"{x:,}")

            # Orden de categorías en X (según Total desc)
            orden_aerolineas = resumen_retrasos["AEROLINEA_FULL"].tolist()

            color_estado = {
                "A tiempo (≤ 15 min)": "#A5D6A7",   # verde pastel
                "Retrasados (> 15 min)": "#EF9A9A", # rojo pastel
            }

            fig_aero = px.bar(
                df_long,
                x="AEROLINEA_FULL",
                y="Cantidad",
                color="Estado",
                text="Texto_cant",
                title="Vuelos a tiempo vs retrasados (> 15 min) por aerolínea",
                labels={
                    "AEROLINEA_FULL": "Aerolínea",
                    "Cantidad": "Número de vuelos",
                    "Estado": "Estado de llegada",
                },
                color_discrete_map=color_estado,
                hover_data={
                    "Total": ":,",
                    "Porcentaje_Retrasos": ":.1f",
                    "Cantidad": ":,",
                },
                category_orders={"AEROLINEA_FULL": orden_aerolineas},
            )

            fig_aero.update_traces(
                textposition="inside",
                textfont=dict(size=11),
            )

            fig_aero.update_layout(
                barmode="stack",
                xaxis_tickangle=-45,
                title_x=0.5,
                plot_bgcolor="rgba(255, 248, 225, 0.6)",
                height=650,
                legend_title="Estado de llegada",
            )

            fig_aero.update_xaxes(
                categoryorder="array",
                categoryarray=orden_aerolineas,
            )

            # --------- % de retrasos encima de cada barra ---------
            fig_aero.add_scatter(
                x=resumen_retrasos["AEROLINEA_FULL"],
                y=resumen_retrasos["Total"] * 1.01,   # un poco por encima de la barra
                mode="text",
                text=resumen_retrasos["Porcentaje_Retrasos"].map(lambda x: f"{x:.1f}%"),
                textposition="top center",
                textfont=dict(size=11),
                showlegend=False,
            )

            st.plotly_chart(fig_aero, use_container_width=True)

    
    st.markdown("---")
    
    #==========================
    ### Distribución de vuelos retrasados (> 15 min) por Mes y Día de la semana
    #==========================

    st.markdown("### Distribución de vuelos retrasados (> 15 min) por Mes y Día de la semana")

    # Trabajamos sobre el df filtrado actual
    df_md = df.copy()

    # Asegurar que ARRIVAL_DELAY es numérico
    if "ARRIVAL_DELAY" not in df_md.columns:
        st.warning("No se encontró la columna ARRIVAL_DELAY para calcular los retrasos.")
    else:
        df_md["ARRIVAL_DELAY"] = pd.to_numeric(df_md["ARRIVAL_DELAY"], errors="coerce")

        # Filtrar SOLO vuelos retrasados > 15 minutos
        df_md = df_md[df_md["ARRIVAL_DELAY"] > 15].copy()

        if df_md.empty:
            st.info("Con los filtros actuales no hay vuelos retrasados (> 15 min) para mostrar en el treemap.")
        else:
            # ==========================
            # Preparar MES y DÍA
            # ==========================
            # Mes en texto
            meses = [
                "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
                "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"
            ]

            if "MONTH" in df_md.columns:
                df_md["MES"] = df_md["MONTH"].apply(
                    lambda m: meses[int(m) - 1] if pd.notna(m) else "Desconocido"
                )
            else:
                df_md["MES"] = "N/D"

            # Día de la semana
            if "DAY_OF_WEEK" in df_md.columns:
                dias_sem = {
                    1: "Lunes", 2: "Martes", 3: "Miércoles",
                    4: "Jueves", 5: "Viernes", 6: "Sábado", 7: "Domingo"
                }
                df_md["DIA_LABEL"] = df_md["DAY_OF_WEEK"].map(dias_sem)
            elif "DAY" in df_md.columns:
                df_md["DIA_LABEL"] = df_md["DAY"].astype(str)
            else:
                df_md["DIA_LABEL"] = "N/D"

            # Conteo (cada fila = 1 vuelo retrasado)
            df_md["COUNT"] = 1

            # Agrupación por Mes y Día
            df_mes_dia = (
                df_md.groupby(["MES", "DIA_LABEL"], observed=True)["COUNT"]
                .sum()
                .reset_index(name="TOTAL_RETRASADOS")
            )

            # Treemap Mes → Día (solo retrasados)
            fig_md = px.treemap(
                df_mes_dia,
                path=["MES", "DIA_LABEL"],          # Nivel 1: Mes, Nivel 2: Día
                values="TOTAL_RETRASADOS",
                color="TOTAL_RETRASADOS",
                color_continuous_scale="Greens",    # tono más fuerte = más retrasos
                hover_data={"TOTAL_RETRASADOS": ":,"},
                title="Vuelos retrasados (> 15 min) por Mes y Día de la semana",
            )

            # Mostrar etiqueta, valor y porcentaje dentro de cada mes
            fig_md.update_traces(
                textinfo="label+value+percent parent"
            )

            fig_md.update_layout(
                title_x=0.5,
                margin=dict(l=0, r=0, t=40, b=0),
            )

            st.plotly_chart(fig_md, use_container_width=True)
            # st.caption(
            #     "Cada rectángulo representa vuelos **retrasados más de 15 minutos**. "
            #     "El tamaño y el tono de verde indican cuántos retrasos hay por Mes y Día."
            # )

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

    col1, col2, col3 = st.columns(3)

    total_vuelos_aero = len(df_aero)
    porc_retraso_lleg_aero = calcular_porcentaje_retraso(df_aero["RETRASADO_LLEGADA"])
    # retraso_prom_lleg_aero = df_aero["ARRIVAL_DELAY"].dropna()
    # retraso_prom_lleg_aero = float(retraso_prom_lleg_aero.mean()) if not retraso_prom_lleg_aero.empty else 0.0
    
    # Promedio de retraso en llegada considerando solo vuelos con ARRIVAL_DELAY > 15
    retraso_prom_lleg_aero = pd.to_numeric(df_aero["ARRIVAL_DELAY"], errors="coerce")

    retraso_prom_lleg_aero = retraso_prom_lleg_aero[retraso_prom_lleg_aero > 15]

    retraso_prom_lleg_aero = (
        float(retraso_prom_lleg_aero.mean())
        if not retraso_prom_lleg_aero.empty
        else 0.0
    )


    col1.metric("Vuelos de la aerolínea", f"{total_vuelos_aero:,}".replace(",", "."))
    col2.metric("% retrasos llegada (aerolínea)", f"{porc_retraso_lleg_aero:.2f}%")
    col3.metric("Retraso prom. llegada (min)", f"{retraso_prom_lleg_aero:.2f}")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)

    # Top 5 aeropuertos de origen de esa aerolínea
    top_origen_aero = (
        df_aero["ORIGEN_AEROPUERTO"]
        .value_counts()
        .head(5)
        .reset_index()
    )

    # Top 5 aeropuertos de destino de esa aerolínea
    top_des_aero = (
        df_aero["DEST_AEROPUERTO"]
        .value_counts()
        .head(5)
        .reset_index()
    )

    # Renombrar columnas de forma explícita para evitar duplicados
    top_origen_aero.columns = ["ORIGEN_AEROPUERTO", "TOTAL_VUELOS"]

    col1.write("Top 5 aeropuertos de origen")
    col1.dataframe(top_origen_aero)
    
    # Renombrar columnas de forma explícita para evitar duplicados
    top_des_aero.columns = ["DEST_AEROPUERTO", "TOTAL_VUELOS"]

    col2.write("Top 5 aeropuertos de destino")
    col2.dataframe(top_des_aero)


    st.markdown("---")

    # ================================
    # Ranking de aerolíneas (ARRIVAL_DELAY > 15)
    # ================================

    # Asegurar que ARRIVAL_DELAY sea numérico
    df_aero_all = df.copy()
    df_aero_all["ARRIVAL_DELAY"] = pd.to_numeric(
        df_aero_all["ARRIVAL_DELAY"], errors="coerce"
    )

    # -------------------------------------------------
    # 1) % de vuelos con retraso > 15 min por aerolínea
    # -------------------------------------------------
    st.markdown("### % de vuelos con retraso (> 15 min) en llegada por aerolínea")

    # Variable binaria: retrasado si ARRIVAL_DELAY > 15
    df_aero_all["RETRASADO_15"] = (df_aero_all["ARRIVAL_DELAY"] > 15).astype(int)

    pct_retraso_15 = (
        df_aero_all
        .groupby("AIRLINE_NAME", observed=True)["RETRASADO_15"]
        .mean()
        .reset_index()
        .rename(columns={"RETRASADO_15": "porc_retraso_15"})
    )

    pct_retraso_15["porc_retraso_15"] = pct_retraso_15["porc_retraso_15"] * 100
    pct_retraso_15 = pct_retraso_15.sort_values("porc_retraso_15", ascending=False)

    fig_aero_lleg_15 = px.bar(
        pct_retraso_15,
        x="porc_retraso_15",
        y="AIRLINE_NAME",
        orientation="h",
        labels={
            "porc_retraso_15": "% vuelos con retraso > 15 min",
            "AIRLINE_NAME": "Aerolínea",
        },
        title="% de vuelos con retraso (> 15 min) en llegada por aerolínea",
        text=pct_retraso_15["porc_retraso_15"].round(1).astype(str) + "%",
        color="porc_retraso_15",                 # <-- degradado según el valor
        color_continuous_scale="Blues",          # escala en tonos azules
    )

    fig_aero_lleg_15.update_layout(
        yaxis={"categoryorder": "total ascending"},
        coloraxis_showscale=False,               # ocultar barra de color
        plot_bgcolor="rgba(255, 248, 225, 0.6)", # fondo similar a otros gráficos
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig_aero_lleg_15.update_traces(
        textposition="inside",
        textfont=dict(size=11, color="black"),
        marker_line_color="rgba(0,0,0,0.25)",
        marker_line_width=0.8,
    )

    st.plotly_chart(fig_aero_lleg_15, use_container_width=True)

    # -------------------------------------------------
    # 2) Retraso promedio de los vuelos retrasados (> 15 min)
    # -------------------------------------------------
    st.markdown("### Retraso promedio (solo vuelos retrasados > 15 min) por aerolínea")

    df_delay_15 = df_aero_all[df_aero_all["ARRIVAL_DELAY"] > 15].copy()

    if df_delay_15.empty:
        st.info("No hay vuelos con ARRIVAL_DELAY > 15 min para calcular retraso promedio por aerolínea.")
    else:
        df_avg_delay_15 = (
            df_delay_15
            .groupby("AIRLINE_NAME", observed=True)["ARRIVAL_DELAY"]
            .mean()
            .reset_index()
            .rename(columns={"ARRIVAL_DELAY": "retraso_prom_15"})
        )

        df_avg_delay_15 = df_avg_delay_15.sort_values("retraso_prom_15", ascending=False)

        fig_aero_delay_15 = px.bar(
            df_avg_delay_15,
            x="retraso_prom_15",
            y="AIRLINE_NAME",
            orientation="h",
            labels={
                "retraso_prom_15": "Retraso prom. llegada (> 15 min) [min]",
                "AIRLINE_NAME": "Aerolínea",
            },
            title="Retraso promedio en llegada por aerolínea (solo vuelos > 15 min)",
            text=df_avg_delay_15["retraso_prom_15"].round(1).astype(str),
            color="retraso_prom_15",              # <-- degradado según el retraso promedio
            color_continuous_scale="OrRd",        # escala naranja/rojo suave
        )

        fig_aero_delay_15.update_layout(
            yaxis={"categoryorder": "total ascending"},
            coloraxis_showscale=False,
            plot_bgcolor="rgba(255, 248, 225, 0.6)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        fig_aero_delay_15.update_traces(
            textposition="inside",
            textfont=dict(size=11, color="black"),
            marker_line_color="rgba(0,0,0,0.25)",
            marker_line_width=0.8,
        )

        st.plotly_chart(fig_aero_delay_15, use_container_width=True)

# ============================
# TAB 3 - AEROPUERTOS
# ============================
with tab_aeropuertos:

    st.subheader("Análisis por Aeropuertos")

    # -------------------------------------------------
    # 0) Preparar data y columna de retraso > 15 minutoS
    # -------------------------------------------------
    if "ARRIVAL_DELAY" not in df.columns:
        st.warning("No se encontró la columna ARRIVAL_DELAY en el dataset.")
        st.stop()

    df_aero = df.copy()
    df_aero["ARRIVAL_DELAY"] = pd.to_numeric(df_aero["ARRIVAL_DELAY"], errors="coerce")

    # Retrasado si ARRIVAL_DELAY > 15 minutoSuto
    df_aero["RETRASADO_1"] = (df_aero["ARRIVAL_DELAY"] > 15).astype("int8")

    # =================================================
    # 1) Top 10 aeropuertos por volumen (origen / destino)
    # =================================================
    col1, col2 = st.columns(2)

    # ---------- Origen ----------
    top_origen = (
        df_aero["ORIGEN_AEROPUERTO"]
        .value_counts()
        .nlargest(10)
        .reset_index()
    )
    top_origen.columns = ["ORIGEN_AEROPUERTO", "TOTAL_VUELOS"]
    top_origen = top_origen.sort_values("TOTAL_VUELOS", ascending=True)

    fig_top_origen = px.bar(
        top_origen,
        x="TOTAL_VUELOS",
        y="ORIGEN_AEROPUERTO",
        orientation="h",
        color="TOTAL_VUELOS",
        color_continuous_scale="Blues",
        labels={
            "ORIGEN_AEROPUERTO": "Aeropuerto origen",
            "TOTAL_VUELOS": "Número de vuelos",
        },
        title="Top 10 aeropuertos de origen por número de vuelos",
        text=top_origen["TOTAL_VUELOS"].apply(lambda x: f"{x:,}"),
    )
    fig_top_origen.update_traces(
        textposition="outside",
        marker=dict(line=dict(color="rgba(0,0,0,0.4)", width=0.6)),
    )
    fig_top_origen.update_layout(
        title_x=0.02,
        coloraxis_showscale=False,
        xaxis_title="Número de vuelos",
        yaxis_title="",
        plot_bgcolor="rgba(255,248,225,0.6)",
    )
    col1.plotly_chart(fig_top_origen, use_container_width=True)

    # ---------- Destino ----------
    top_dest = (
        df_aero["DEST_AEROPUERTO"]
        .value_counts()
        .nlargest(10)
        .reset_index()
    )
    top_dest.columns = ["DEST_AEROPUERTO", "TOTAL_VUELOS"]
    top_dest = top_dest.sort_values("TOTAL_VUELOS", ascending=True)

    fig_top_dest = px.bar(
        top_dest,
        x="TOTAL_VUELOS",
        y="DEST_AEROPUERTO",
        orientation="h",
        color="TOTAL_VUELOS",
        color_continuous_scale="Blues",
        labels={
            "DEST_AEROPUERTO": "Aeropuerto destino",
            "TOTAL_VUELOS": "Número de vuelos",
        },
        title="Top 10 aeropuertos de destino por número de vuelos",
        text=top_dest["TOTAL_VUELOS"].apply(lambda x: f"{x:,}"),
    )
    fig_top_dest.update_traces(
        textposition="outside",
        marker=dict(line=dict(color="rgba(0,0,0,0.4)", width=0.6)),
    )
    fig_top_dest.update_layout(
        title_x=0.02,
        coloraxis_showscale=False,
        xaxis_title="Número de vuelos",
        yaxis_title="",
        plot_bgcolor="rgba(255,248,225,0.6)",
    )
    col2.plotly_chart(fig_top_dest, use_container_width=True)

    st.markdown("---")

    # =================================================
    # 2) % de retrasos en llegada (> 15 minutoS) por aeropuerto
    #    (origen / destino)
    # =================================================
    col3, col4 = st.columns(2)

    # Umbral mínimo de vuelos por aeropuerto para evitar ruido
    MIN_VUELOS = 500

    # ---------- % retrasos por aeropuerto ORIGEN ----------
    resumen_origen = (
        df_aero
        .groupby("ORIGEN_AEROPUERTO", observed=True)["RETRASADO_1"]
        .agg(Total="size", Retrasados="sum")
        .reset_index()
    )
    resumen_origen["Porc_Retrasados"] = (
        resumen_origen["Retrasados"] / resumen_origen["Total"] * 100
    )
    resumen_origen = resumen_origen[resumen_origen["Total"] >= MIN_VUELOS]
    resumen_origen = resumen_origen.sort_values("Porc_Retrasados", ascending=False).head(10)
    resumen_origen = resumen_origen.sort_values("Porc_Retrasados", ascending=True)

    fig_retraso_origen = px.bar(
        resumen_origen,
        x="Porc_Retrasados",
        y="ORIGEN_AEROPUERTO",
        orientation="h",
        color="Porc_Retrasados",
        color_continuous_scale="RdYlGn_r",      # rojo = peor porcentaje
        labels={
            "ORIGEN_AEROPUERTO": "Aeropuerto origen",
            "Porc_Retrasados": "% vuelos retrasados (> 15 minutoS)",
        },
        title="% de vuelos con retraso en llegada (> 15 minutoS) por aeropuerto origen",
        text=resumen_origen["Porc_Retrasados"].round(2).astype(str) + "%",
    )
    fig_retraso_origen.update_traces(
        textposition="outside",
        marker=dict(line=dict(color="rgba(0,0,0,0.4)", width=0.6)),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Total vuelos: %{customdata[0]:,}<br>"
            "Retrasados (>15 minutoS): %{customdata[1]:,}<br>"
            "% retrasados: %{x:.2f}%<extra></extra>"
        ),
        customdata=resumen_origen[["Total", "Retrasados"]].to_numpy(),
    )
    fig_retraso_origen.update_layout(
        title_x=0.02,
        xaxis_title="% de vuelos retrasados (> 15 minutoS)",
        yaxis_title="",
        plot_bgcolor="rgba(255,248,225,0.6)",
        coloraxis_colorbar=dict(title="%"),
    )
    col3.plotly_chart(fig_retraso_origen, use_container_width=True)

    # ---------- % retrasos por aeropuerto DESTINO ----------
    resumen_dest = (
        df_aero
        .groupby("DEST_AEROPUERTO", observed=True)["RETRASADO_1"]
        .agg(Total="size", Retrasados="sum")
        .reset_index()
    )
    resumen_dest["Porc_Retrasados"] = (
        resumen_dest["Retrasados"] / resumen_dest["Total"] * 100
    )
    resumen_dest = resumen_dest[resumen_dest["Total"] >= MIN_VUELOS]
    resumen_dest = resumen_dest.sort_values("Porc_Retrasados", ascending=False).head(10)
    resumen_dest = resumen_dest.sort_values("Porc_Retrasados", ascending=True)

    fig_retraso_dest = px.bar(
        resumen_dest,
        x="Porc_Retrasados",
        y="DEST_AEROPUERTO",
        orientation="h",
        color="Porc_Retrasados",
        color_continuous_scale="RdYlGn_r",
        labels={
            "DEST_AEROPUERTO": "Aeropuerto destino",
            "Porc_Retrasados": "% vuelos retrasados (> 15 minutoS)",
        },
        title="% de vuelos con retraso en llegada (> 15 minutoS) por aeropuerto destino",
        text=resumen_dest["Porc_Retrasados"].round(2).astype(str) + "%",
    )
    fig_retraso_dest.update_traces(
        textposition="outside",
        marker=dict(line=dict(color="rgba(0,0,0,0.4)", width=0.6)),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Total vuelos: %{customdata[0]:,}<br>"
            "Retrasados (>15 minutoS): %{customdata[1]:,}<br>"
            "% retrasados: %{x:.2f}%<extra></extra>"
        ),
        customdata=resumen_dest[["Total", "Retrasados"]].to_numpy(),
    )
    fig_retraso_dest.update_layout(
        title_x=0.02,
        xaxis_title="% de vuelos retrasados (> 15 minutoS)",
        yaxis_title="",
        plot_bgcolor="rgba(255,248,225,0.6)",
        coloraxis_colorbar=dict(title="%"),
    )
    col4.plotly_chart(fig_retraso_dest, use_container_width=True)

    st.markdown("---")


     # ================================
    # Mapa de destinos (ARRIVAL_DELAY > 15)
    # ================================
    st.markdown("### Destinos con mayor % de retrasos en llegada (> 15 min)")

    cols_geo = ["DESTINATION_AIRPORT", "DEST_AEROPUERTO", "DEST_CIUDAD",
                "DEST_ESTADO", "DEST_LAT", "DEST_LON", "ARRIVAL_DELAY"]

    # Verificaciones básicas
    faltan = [c for c in cols_geo if c not in df.columns]
    if faltan:
        st.warning(f"No se pueden generar los mapas. Faltan columnas: {', '.join(faltan)}")
    else:
        df_geo = df[cols_geo].copy()
        df_geo["ARRIVAL_DELAY"] = pd.to_numeric(df_geo["ARRIVAL_DELAY"], errors="coerce")

        # Retraso si ARRIVAL_DELAY > 15 min
        df_geo["RETRASADO_15"] = (df_geo["ARRIVAL_DELAY"] > 15).astype("int8")

        # Agregación por aeropuerto destino
        geo_dest = (
            df_geo
            .groupby(
                ["DESTINATION_AIRPORT", "DEST_AEROPUERTO",
                "DEST_CIUDAD", "DEST_ESTADO", "DEST_LAT", "DEST_LON"],
                observed=True
            )["RETRASADO_15"]
            .agg(
                TOTAL="size",
                PORC_RETRASADOS="mean"      # fracción de vuelos retrasados (>15 min)
            )
            .reset_index()
        )

        # Limpieza: quitar sin coordenadas o sin vuelos
        geo_dest = geo_dest.replace([np.inf, -np.inf], np.nan).dropna(subset=["DEST_LAT", "DEST_LON"])
        geo_dest = geo_dest[geo_dest["TOTAL"] > 0]

        if geo_dest.empty:
            st.info("No hay datos suficientes para el mapa de destinos con los filtros actuales.")
        else:
            # Umbral de volumen para evitar ruido visual
            umbral = 500
            geo_dest_top = geo_dest[geo_dest["TOTAL"] >= umbral].copy()

            if geo_dest_top.empty:
                st.info(
                    f"No hay aeropuertos destino con al menos {umbral:,} vuelos para mostrar en el mapa."
                )
            else:
                # % de retrasos en lugar de fracción
                geo_dest_top["PORC_RETRASADOS_PCT"] = geo_dest_top["PORC_RETRASADOS"] * 100

                fig_dest = px.scatter_geo(
                    geo_dest_top,
                    lat="DEST_LAT",
                    lon="DEST_LON",
                    size="TOTAL",
                    size_max=28,
                    color="PORC_RETRASADOS_PCT",
                    color_continuous_scale="RdYlGn_r",  # rojo = peor %, verde = mejor
                    hover_name="DEST_AEROPUERTO",
                    hover_data={
                        "DEST_CIUDAD": True,
                        "DEST_ESTADO": True,
                        "TOTAL": ":,",
                        "PORC_RETRASADOS_PCT": ":.2f",
                        "DEST_LAT": False,
                        "DEST_LON": False,
                    },
                )

                # Estilo tipo "mapa claro" como en la versión previa
                fig_dest.update_traces(
                    marker=dict(
                        opacity=0.85,
                        line=dict(width=0.4, color="rgba(80,80,80,0.6)")
                    )
                )

                fig_dest.update_layout(
                    title=(
                        f"Destinos con mayor % de retrasos en llegada (> 15 min) "
                        f"(tamaño = volumen, umbral ≥ {umbral:,} vuelos)"
                    ),
                    template="plotly_white",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=0, r=0, t=60, b=0),
                    coloraxis_colorbar=dict(
                        title="% retrasos",
                        ticks="outside",
                        tickformat=".1f"
                    ),
                    geo=dict(
                        scope="north america",
                        projection_type="natural earth",
                        showland=True,
                        landcolor="rgb(240,248,255)",
                        showcountries=True,
                        countrycolor="rgb(180,180,180)",
                        showcoastlines=True,
                        coastlinecolor="rgb(150,150,150)",
                        # un poco de zoom a EE.UU.
                        lataxis_range=[15, 60],
                        lonaxis_range=[-130, -60],
                    ),
                )

                st.plotly_chart(fig_dest, use_container_width=True)
    st.markdown("---")
    st.markdown("---")
    st.markdown("### % de retrasos (> 15 min) por hora de llegada y aeropuerto destino")

    # Necesitamos estas columnas para el gráfico
    columnas_necesarias = ["DEST_AEROPUERTO", "ARRIVAL_DELAY"]
    if not all(col in df.columns for col in columnas_necesarias):
        st.info("No se encontraron las columnas necesarias para este gráfico "
                "(DEST_AEROPUERTO y ARRIVAL_DELAY).")
    else:
        df_hora = df.copy()

        # -------------------------------------------------
        # 1) Obtener HORA_LLEGADA (0–23) de la mejor fuente
        # -------------------------------------------------
        if "MINUTO_DIA_LLEGADA" in df_hora.columns:
            # Si ya tienes los minutos del día, es la fuente preferida
            df_hora = df_hora[df_hora["MINUTO_DIA_LLEGADA"].notna()].copy()
            df_hora["HORA_LLEGADA"] = (df_hora["MINUTO_DIA_LLEGADA"] // 60).astype(int)
        elif "SCHEDULED_ARRIVAL" in df_hora.columns:
            # Si no hay MINUTO_DIA_LLEGADA, usamos SCHEDULED_ARRIVAL en formato HHMM
            df_hora["SCHEDULED_ARRIVAL"] = pd.to_numeric(
                df_hora["SCHEDULED_ARRIVAL"], errors="coerce"
            )
            df_hora = df_hora[df_hora["SCHEDULED_ARRIVAL"].notna()].copy()
            df_hora["HORA_LLEGADA"] = (
                (df_hora["SCHEDULED_ARRIVAL"] // 100)
                .clip(0, 23)
                .astype(int)
            )
        else:
            st.info(
                "No se encontró ninguna columna horaria (MINUTO_DIA_LLEGADA o "
                "SCHEDULED_ARRIVAL) para calcular la hora de llegada."
            )
            df_hora = None

        if df_hora is not None and not df_hora.empty:
            # -------------------------------------------------
            # 2) Etiqueta binaria de retraso ARRIVAL_DELAY > 15
            # -------------------------------------------------
            df_hora["ARRIVAL_DELAY"] = pd.to_numeric(
                df_hora["ARRIVAL_DELAY"], errors="coerce"
            )
            df_hora = df_hora[df_hora["ARRIVAL_DELAY"].notna()].copy()
            df_hora["RETRASADO_15"] = (df_hora["ARRIVAL_DELAY"] > 15).astype(int)

            # -------------------------------------------------
            # 3) Para que el heatmap sea legible, usamos solo
            #    los destinos con más volumen (top 15)
            # -------------------------------------------------
            top_destinos = (
                df_hora.groupby("DEST_AEROPUERTO")["RETRASADO_15"]
                .size()
                .nlargest(15)
                .index
            )
            df_hora_top = df_hora[df_hora["DEST_AEROPUERTO"].isin(top_destinos)].copy()

            if df_hora_top.empty:
                st.info("No hay datos suficientes para mostrar el mapa de calor.")
            else:
                # -----------------------------------------
                # 4) Agregación: % de retrasados por
                #    DEST_AEROPUERTO x HORA_LLEGADA
                # -----------------------------------------
                resumen = (
                    df_hora_top
                    .groupby(["DEST_AEROPUERTO", "HORA_LLEGADA"], observed=True)["RETRASADO_15"]
                    .agg(Total="size", Retrasados="sum")
                    .reset_index()
                )
                resumen["Porc_Retrasados"] = (
                    resumen["Retrasados"] / resumen["Total"] * 100
                ).round(1)

                # Aseguramos rango de horas 0–23
                resumen = resumen[resumen["HORA_LLEGADA"].between(0, 23)]
                resumen["HORA_LLEGADA"] = resumen["HORA_LLEGADA"].astype(int)

                # Pivot para heatmap: filas = destino, columnas = hora
                tabla_heat = (
                    resumen
                    .pivot(index="DEST_AEROPUERTO",
                           columns="HORA_LLEGADA",
                           values="Porc_Retrasados")
                    .fillna(0)
                )
                # Ordenar destinos y horas
                tabla_heat = tabla_heat.sort_index()
                tabla_heat = tabla_heat.reindex(
                    columns=sorted(tabla_heat.columns)
                )

                # -----------------------------------------
                # 5) Mapa de calor con Plotly
                # -----------------------------------------
                fig_heat = px.imshow(
                    tabla_heat,
                    aspect="auto",
                    color_continuous_scale="RdYlGn_r",
                    labels={
                        "x": "Hora de llegada (0–23)",
                        "y": "Aeropuerto destino",
                        "color": "% retrasos (> 15 min)"
                    },
                    title="% de vuelos retrasados (> 15 min) por hora de llegada "
                          "y aeropuerto destino (Top 15 por volumen)"
                )

                # Etiquetas de 0–23 como HH:00
                fig_heat.update_xaxes(
                    tickmode="array",
                    tickvals=list(range(len(tabla_heat.columns))),
                    ticktext=[f"{h:02d}:00" for h in tabla_heat.columns],
                )

                fig_heat.update_layout(
                    title_x=0.5,
                    height=520,
                    plot_bgcolor="rgba(255,248,225,0.6)",
                    margin=dict(l=80, r=40, t=80, b=60),
                    coloraxis_colorbar=dict(title="% retrasos")
                )

                st.plotly_chart(fig_heat, use_container_width=True)

# ============================
# TAB 4 - TIEMPO Y PATRONES
# ============================
with tab_tiempo:
    st.subheader("Patrones Temporales de Retrasos")

    st.markdown("---")
    # ====================================================
    # Análisis por día de la semana (ARRIVAL_DELAY > 15)
    # ====================================================
    st.markdown("### Análisis por día de la semana")

    if "DAY_OF_WEEK" not in df.columns or "DAY_OF_WEEK_NOMBRE" not in df.columns or "ARRIVAL_DELAY" not in df.columns:
        st.warning("No se encontraron las columnas DAY_OF_WEEK, DAY_OF_WEEK_NOMBRE y/o ARRIVAL_DELAY.")
    else:
        df_dia = df.copy()
        df_dia["ARRIVAL_DELAY"] = pd.to_numeric(df_dia["ARRIVAL_DELAY"], errors="coerce")

        # Retrasado si ARRIVAL_DELAY > 15 min
        df_dia["RETRASADO_15"] = (df_dia["ARRIVAL_DELAY"] > 15).astype(int)

        # Resumen por día: Total y Retrasados (>15)
        resumen_dia = (
            df_dia.groupby(["DAY_OF_WEEK", "DAY_OF_WEEK_NOMBRE"], observed=True)["RETRASADO_15"]
                  .agg(Total="size", Retrasados="sum")
                  .reset_index()
        )

        if resumen_dia.empty:
            st.info("No hay datos suficientes para los gráficos por día de la semana con los filtros actuales.")
        else:
            # Orden correcto de los días (1–7)
            resumen_dia = resumen_dia.sort_values("DAY_OF_WEEK")

            # Columna 1 y 2
            col_d1, col_d2 = st.columns(2)

            # ====================================================
            # COL 1: Vuelos a tiempo vs retrasados (>15 min)
            # ====================================================
            with col_d1:
                #st.markdown("#### Vuelos a tiempo vs retrasados (> 15 min) por día de la semana")

                resumen_dia["A_Tiempo"] = resumen_dia["Total"] - resumen_dia["Retrasados"]

                # Formato largo
                resumen_long_dia = resumen_dia.melt(
                    id_vars=["DAY_OF_WEEK", "DAY_OF_WEEK_NOMBRE", "Total"],
                    value_vars=["A_Tiempo", "Retrasados"],
                    var_name="Estado",
                    value_name="Cantidad"
                )

                estado_map_dia = {
                    "A_Tiempo": "A tiempo (≤ 15 min)",
                    "Retrasados": "Retrasados (> 15 min)",
                }
                resumen_long_dia["Estado"] = resumen_long_dia["Estado"].map(estado_map_dia)

                # % distribución dentro de cada día
                resumen_long_dia["Porcentaje"] = (
                    resumen_long_dia["Cantidad"] / resumen_long_dia["Total"] * 100
                )

                # Texto: cantidad + porcentaje
                resumen_long_dia["Texto"] = resumen_long_dia.apply(
                    lambda r: f"{r['Cantidad']:,}\n({r['Porcentaje']:.1f}%)",
                    axis=1
                )

                color_estado_dia = {
                    "A tiempo (≤ 15 min)": "#A5D6A7",   # verde pastel
                    "Retrasados (> 15 min)": "#EF9A9A", # rojo pastel
                }

                fig_d1 = px.bar(
                    resumen_long_dia,
                    x="DAY_OF_WEEK_NOMBRE",
                    y="Cantidad",
                    color="Estado",
                    text="Texto",
                    title="Vuelos a tiempo vs retrasados (> 15 min) por día de la semana",
                    labels={
                        "DAY_OF_WEEK_NOMBRE": "Día de la semana",
                        "Cantidad": "Cantidad de vuelos",
                        "Estado": "Estado de llegada",
                    },
                    color_discrete_map=color_estado_dia,
                )

                fig_d1.update_traces(
                    textposition="inside",
                    textfont=dict(size=11),
                )

                fig_d1.update_layout(
                    barmode="stack",
                    yaxis_title="Cantidad de vuelos",
                    xaxis_title="Día de la semana",
                    plot_bgcolor="rgba(255, 248, 225, 0.6)",
                    legend=dict(
                        orientation="h",
                        yanchor="top",
                        y=-0.20,        # leyenda debajo del eje X
                        xanchor="center",
                        x=0.5,
                    ),
                    margin=dict(l=40, r=40, t=80, b=100),
                )

                st.plotly_chart(fig_d1, use_container_width=True)

            # ====================================================
            # COL 2: % de vuelos retrasados (>15 min)
            # ====================================================
            with col_d2:
                # st.markdown("#### % de retrasos (> 15 min) por día de la semana")

                resumen_dia["Porc_Retrasado"] = (
                    resumen_dia["Retrasados"] / resumen_dia["Total"] * 100
                )

                fig_d2 = px.bar(
                    resumen_dia,
                    x="DAY_OF_WEEK_NOMBRE",
                    y="Porc_Retrasado",
                    labels={
                        "DAY_OF_WEEK_NOMBRE": "Día de la semana",
                        "Porc_Retrasado": "% retrasos (> 15 min)",
                    },
                    title="% de vuelos retrasados (> 15 min) por día de la semana",
                    text=resumen_dia["Porc_Retrasado"].round(1).astype(str) + "%",
                    color="Porc_Retrasado",
                    color_continuous_scale="Oranges",
                )

                fig_d2.update_traces(
                    textposition="outside",
                    textfont=dict(size=11),
                )

                fig_d2.update_layout(
                    plot_bgcolor="rgba(255, 248, 225, 0.6)",
                    xaxis_title="Día de la semana",
                    yaxis_title="% retrasos (> 15 min)",
                    coloraxis_showscale=False,   # ocultar barra de color
                    margin=dict(l=40, r=40, t=80, b=80),
                )

                st.plotly_chart(fig_d2, use_container_width=True)

    st.markdown("---")
    # ====================================================
    # Vuelos a tiempo vs retrasados (> 15 min) y
    # Total de vuelos vs % de retrasos (> 15 min)
    # por PERIODO_LLEGADA (en dos columnas)
    # ====================================================
    st.markdown("### Análisis por período de llegada")

    if "PERIODO_LLEGADA" not in df.columns or "ARRIVAL_DELAY" not in df.columns:
        st.warning("No se encontraron las columnas PERIODO_LLEGADA y/o ARRIVAL_DELAY.")
    else:
        # Copia del dataframe filtrado y cálculo de retrasos > 15 min
        df_per = df.copy()
        df_per["ARRIVAL_DELAY"] = pd.to_numeric(df_per["ARRIVAL_DELAY"], errors="coerce")
        df_per["RETRASADO_15"] = (df_per["ARRIVAL_DELAY"] > 15).astype(int)

        # Resumen base por período
        resumen_base = (
            df_per.groupby("PERIODO_LLEGADA", observed=True)["RETRASADO_15"]
                  .agg(Total="size", Retrasados="sum")
                  .reset_index()
        )

        if resumen_base.empty:
            st.info("No hay datos suficientes para estos gráficos con los filtros actuales.")
        else:
            # Orden lógico de los períodos
            orden_periodos = ["Madrugada", "Mañana", "Tarde", "Noche"]
            resumen_base["PERIODO_LLEGADA"] = pd.Categorical(
                resumen_base["PERIODO_LLEGADA"],
                categories=[p for p in orden_periodos
                            if p in resumen_base["PERIODO_LLEGADA"].unique()],
                ordered=True
            )
            resumen_base = resumen_base.sort_values("PERIODO_LLEGADA")

            # Columnas para mostrar los dos gráficos
            col_per1, col_per2 = st.columns(2)

            # ====================================================
            # COL 1: Barras apiladas A tiempo vs Retrasados
            # ====================================================
            with col_per1:
                # st.markdown("#### Vuelos a tiempo vs retrasados (> 15 min) por período de llegada")

                resumen_stack = resumen_base.copy()
                resumen_stack["A_Tiempo"] = resumen_stack["Total"] - resumen_stack["Retrasados"]

                # Formato largo
                resumen_long = resumen_stack.melt(
                    id_vars=["PERIODO_LLEGADA", "Total"],
                    value_vars=["A_Tiempo", "Retrasados"],
                    var_name="Estado",
                    value_name="Cantidad"
                )

                estado_map = {
                    "A_Tiempo": "A tiempo (≤ 15 min)",
                    "Retrasados": "Retrasados (> 15 min)",
                }
                resumen_long["Estado"] = resumen_long["Estado"].map(estado_map)

                # % de distribución dentro de cada período
                resumen_long["Porcentaje"] = (
                    resumen_long["Cantidad"] / resumen_long["Total"] * 100
                )

                # Texto: cantidad + porcentaje
                resumen_long["Texto"] = resumen_long.apply(
                    lambda r: f"{r['Cantidad']:,}\n({r['Porcentaje']:.1f}%)",
                    axis=1
                )

                color_estado = {
                    "A tiempo (≤ 15 min)": "#A5D6A7",   # verde pastel
                    "Retrasados (> 15 min)": "#EF9A9A", # rojo pastel
                }

                fig2 = px.bar(
                    resumen_long,
                    x="PERIODO_LLEGADA",
                    y="Cantidad",
                    color="Estado",
                    text="Texto",
                    title="Vuelos a tiempo vs retrasados (> 15 min) por período de llegada",
                    labels={
                        "PERIODO_LLEGADA": "Período de llegada",
                        "Cantidad": "Cantidad de vuelos",
                        "Estado": "Estado de llegada",
                    },
                    color_discrete_map=color_estado,
                )

                fig2.update_traces(
                    textposition="inside",
                    textfont=dict(size=11),
                )

                fig2.update_layout(
                    barmode="stack",
                    yaxis_title="Cantidad de vuelos",
                    xaxis_title="Período de llegada",
                    plot_bgcolor="rgba(255, 248, 225, 0.6)",  # mismo fondo suave
                    # título alineado a la derecha
                    title=dict(
                        text="Vuelos a tiempo vs retrasados (> 15 min) por período de llegada",
                        x=1.0,
                        xanchor="right"
                    ),
                    legend=dict(
                        orientation="h",
                        yanchor="top",
                        y=-0.20,          # leyenda debajo del eje X
                        xanchor="center",
                        x=0.5,
                    ),
                    margin=dict(l=40, r=40, t=80, b=100),
                )

                st.plotly_chart(fig2, use_container_width=True)

            # ====================================================
            # COL 2: Total vs % de retrasos (> 15 min)
            # ====================================================
            with col_per2:
                # st.markdown("#### Total de vuelos vs % de retrasos (> 15 min) por período de llegada")

                resumen_stack = resumen_base.copy()
                resumen_stack["Porc_Retrasado"] = (
                    resumen_stack["Retrasados"] / resumen_stack["Total"] * 100
                )

                # Cálculos globales
                total_vuelos = int(resumen_stack["Total"].sum())
                promedio_retrasos = (
                    resumen_stack["Retrasados"].sum() / total_vuelos * 100
                    if total_vuelos > 0 else 0.0
                )

                pct = resumen_stack["Porc_Retrasado"].values
                colors = np.where(
                    pct < 15,
                    "#4CAF50",                          # verde
                    np.where(pct <= 25, "#FFC107", "#F44336")  # amarillo / rojo
                )

                y2_max = max(
                    30,
                    float(resumen_stack["Porc_Retrasado"].max() * 1.4),
                    float(promedio_retrasos * 1.4),
                )

                fig_per = go.Figure()

                # Barras: total de vuelos
                fig_per.add_trace(go.Bar(
                    x=resumen_stack["PERIODO_LLEGADA"],
                    y=resumen_stack["Total"],
                    name="Total de vuelos",
                    marker_color="#1976D2",
                    text=resumen_stack["Total"].apply(lambda x: f"{x:,}"),
                    textposition="inside",
                    textfont=dict(color="white", size=11),
                    yaxis="y1",
                    hovertemplate="<b>%{x}</b><br>Total de vuelos: %{y:,}<extra></extra>"
                ))

                # Línea + puntos: % retrasos > 15
                fig_per.add_trace(go.Scatter(
                    x=resumen_stack["PERIODO_LLEGADA"],
                    y=resumen_stack["Porc_Retrasado"],
                    name="% Retrasos (> 15 min)",
                    mode="lines+markers+text",
                    line=dict(color="#F57C00", width=3),
                    marker=dict(
                        size=10,
                        color=colors,
                        line=dict(color="#BF360C", width=1.5)
                    ),
                    text=[f"{v:.2f}%" for v in resumen_stack["Porc_Retrasado"]],
                    textposition="bottom center",
                    textfont=dict(color="white", size=11),
                    yaxis="y2",
                    hovertemplate="<b>%{x}</b><br>% Retrasos (> 15 min): %{y:.2f}%<extra></extra>"
                ))

                # Bandas de color + línea promedio
                fig_per.update_layout(
                    shapes=[
                        dict(type="rect", xref="paper", x0=0, x1=1, yref="y2",
                             y0=0, y1=15, fillcolor="rgba(129,199,132,0.5)",
                             line_width=0, layer="below"),
                        dict(type="rect", xref="paper", x0=0, x1=1, yref="y2",
                             y0=15, y1=25, fillcolor="rgba(255,241,118,0.6)",
                             line_width=0, layer="below"),
                        dict(type="rect", xref="paper", x0=0, x1=1, yref="y2",
                             y0=25, y1=y2_max, fillcolor="rgba(239,154,154,0.6)",
                             line_width=0, layer="below"),
                        dict(
                            type="line", xref="paper", x0=0, x1=1, yref="y2",
                            y0=promedio_retrasos, y1=promedio_retrasos,
                            line=dict(color="#00838F", width=3, dash="dot")
                        ),
                    ]
                )

                # Leyenda de umbrales + promedio, abajo
                fig_per.add_trace(go.Scatter(
                    x=[None], y=[None], mode="markers",
                    marker=dict(size=10, color="#4CAF50"),
                    name="< 15% retrasos"
                ))
                fig_per.add_trace(go.Scatter(
                    x=[None], y=[None], mode="markers",
                    marker=dict(size=10, color="#FFC107"),
                    name="15% – 25% retrasos"
                ))
                fig_per.add_trace(go.Scatter(
                    x=[None], y=[None], mode="markers",
                    marker=dict(size=10, color="#F44336"),
                    name="> 25% retrasos"
                ))
                fig_per.add_trace(go.Scatter(
                    x=[None], y=[None], mode="lines",
                    line=dict(color="#00838F", width=2, dash="dot"),
                    name=f"Promedio global ({promedio_retrasos:.2f}%)"
                ))

                fig_per.update_layout(
                    title=(
                        "Total de vuelos vs % de retrasos (> 15 min) por período de llegada<br>"
                        f"<sup>✈️ Total analizado: {total_vuelos:,} vuelos | "
                        f"Promedio global de retrasos (> 15 min): {promedio_retrasos:.2f}%</sup>"
                    ),
                    xaxis=dict(title="Período de llegada"),
                    yaxis=dict(title="Total de vuelos", side="left", showgrid=False),
                    yaxis2=dict(
                        title="% Retrasos (> 15 min)",
                        side="right",
                        overlaying="y",
                        showgrid=False,
                        range=[0, y2_max],
                    ),
                    legend=dict(
                        orientation="h",
                        yanchor="top",
                        y=-0.25,          # leyenda bien debajo del eje X
                        xanchor="center",
                        x=0.5,
                    ),
                    bargap=0.3,
                    template="plotly_white",
                    margin=dict(l=60, r=60, t=90, b=120),
                )

                st.plotly_chart(fig_per, use_container_width=True)

    st.markdown("---")
    # ======================================================
    # % de vuelos retrasados (> 15 min) según hora de LLEGADA
    # ======================================================
    st.markdown("---")
    st.markdown("### % de vuelos retrasados (> 15 min) según hora de llegada (bloques de 30 min)")

    # Verificamos columnas necesarias
    if "ARRIVAL_DELAY" not in df.columns or "MINUTO_DIA_LLEGADA" not in df.columns:
        st.info(
            "No se encontraron las columnas MINUTO_DIA_LLEGADA y/o ARRIVAL_DELAY "
            "para construir este gráfico."
        )
    else:
        # Trabajar sobre df filtrado por los controles del sidebar
        v = df[["ARRIVAL_DELAY", "MINUTO_DIA_LLEGADA"]].dropna().copy()
        v["ARRIVAL_DELAY"] = pd.to_numeric(v["ARRIVAL_DELAY"], errors="coerce")

        # Variable binaria: retrasado en llegada si ARRIVAL_DELAY > 15
        v["RETRASADO_LLEGADA"] = (v["ARRIVAL_DELAY"] > 15).astype("int8")

        # Asegurar tipo entero en minutos de llegada
        v["MINUTO_DIA_LLEGADA"] = v["MINUTO_DIA_LLEGADA"].astype("int32")

        # Bloques de 30 minutos
        v["BUCKET_30_LLEGADA"] = (v["MINUTO_DIA_LLEGADA"] // 30) * 30

        # Agregación: tamaño y % retrasados por bloque
        dash_lleg = (
            v.groupby("BUCKET_30_LLEGADA", observed=True)["RETRASADO_LLEGADA"]
             .agg(Total="size", Porc_Retrasados="mean")
             .reset_index()
        )

        if dash_lleg.empty:
            st.info("No hay datos suficientes para generar el gráfico por hora de llegada.")
        else:
            # % retrasados y etiqueta HH:MM
            dash_lleg["Porc_Retrasados"] = dash_lleg["Porc_Retrasados"] * 100
            dash_lleg["HORA_LABEL"] = dash_lleg["BUCKET_30_LLEGADA"].apply(minutos_a_hora_str)

            # customdata: [Hora HH:MM, Total vuelos en ese bloque]
            custom_lleg = np.stack(
                [dash_lleg["HORA_LABEL"], dash_lleg["Total"]],
                axis=-1
            )

            # Eje X en minutos, ticks cada 2 horas
            xticks = np.arange(0, 24 * 60 + 1, 120)
            xtick_labels = [minutos_a_hora_str(x) for x in xticks]

            # Figura de líneas
            fig_lleg = go.Figure()

            fig_lleg.add_trace(
                go.Scatter(
                    x=dash_lleg["BUCKET_30_LLEGADA"],
                    y=dash_lleg["Porc_Retrasados"],
                    mode="lines+markers",  # <-- sin texto impreso
                    line=dict(color="#F57C00", width=2),
                    marker=dict(size=6),
                    customdata=custom_lleg,
                    hovertemplate=(
                        "<b>Llegada %{customdata[0]}</b><br>"
                        "% retrasados (> 15 min): %{y:.2f}%<br>"
                        "n: %{customdata[1]:,}"
                        "<extra></extra>"
                    ),
                    name="% retrasos llegada",
                )
            )

            fig_lleg.update_xaxes(
                tickmode="array",
                tickvals=xticks,
                ticktext=xtick_labels,
                title_text="Hora del día (llegada, bloques de 30 min)",
            )

            fig_lleg.update_yaxes(
                title_text="% de vuelos retrasados (> 15 min)",
            )

            fig_lleg.update_layout(
                title=(
                    "Porcentaje de vuelos retrasados (> 15 min) según hora de llegada<br>"
                    "<sup>Bloques de 30 minutos, usando ARRIVAL_DELAY > 15 min</sup>"
                ),
                template="plotly_white",
                plot_bgcolor="rgba(255, 248, 225, 0.6)",  # mismo estilo suave del dashboard
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=60, r=40, t=80, b=60),
                showlegend=False,
            )

            st.plotly_chart(fig_lleg, use_container_width=True)

    st.markdown("---")     
        # ====================================================
    # Mapa de calor: % de retrasos (> 15 min) por Mes y Período de llegada
    # ====================================================
    st.markdown("---")
    st.markdown("### Mapa de calor: % de vuelos retrasados (> 15 min) por mes y período de llegada")

    if "ARRIVAL_DELAY" not in df.columns or "PERIODO_LLEGADA" not in df.columns or "MONTH" not in df.columns:
        st.info("No se encontraron las columnas ARRIVAL_DELAY, PERIODO_LLEGADA y/o MONTH para este gráfico.")
    else:
        df_mp = df.copy()
        df_mp["ARRIVAL_DELAY"] = pd.to_numeric(df_mp["ARRIVAL_DELAY"], errors="coerce")

        # Retrasado si ARRIVAL_DELAY > 15 min
        df_mp["RETRASADO_15"] = (df_mp["ARRIVAL_DELAY"] > 15).astype(int)

        # Nombres de meses
        meses_nombres = [
            "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
            "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"
        ]

        df_mp["MONTH"] = pd.to_numeric(df_mp["MONTH"], errors="coerce")
        df_mp = df_mp[df_mp["MONTH"].between(1, 12)]

        if df_mp.empty:
            st.info("No hay datos suficientes para generar el mapa de calor por mes y período.")
        else:
            df_mp["MES_NOMBRE"] = df_mp["MONTH"].apply(lambda m: meses_nombres[int(m) - 1])

            # Agrupación por Mes y Período de llegada
            resumen_mes_per = (
                df_mp.groupby(["MES_NOMBRE", "PERIODO_LLEGADA"], observed=True)["RETRASADO_15"]
                     .mean()
                     .reset_index()
                     .rename(columns={"RETRASADO_15": "Porc_Retrasado"})
            )

            resumen_mes_per["Porc_Retrasado"] = resumen_mes_per["Porc_Retrasado"] * 100

            # Orden lógico de ejes
            orden_mes = meses_nombres
            orden_periodos = ["Madrugada", "Mañana", "Tarde", "Noche"]

            resumen_mes_per["MES_NOMBRE"] = pd.Categorical(
                resumen_mes_per["MES_NOMBRE"],
                categories=[m for m in orden_mes if m in resumen_mes_per["MES_NOMBRE"].unique()],
                ordered=True
            )
            resumen_mes_per["PERIODO_LLEGADA"] = pd.Categorical(
                resumen_mes_per["PERIODO_LLEGADA"],
                categories=[p for p in orden_periodos if p in resumen_mes_per["PERIODO_LLEGADA"].unique()],
                ordered=True
            )
            resumen_mes_per = resumen_mes_per.sort_values(["PERIODO_LLEGADA", "MES_NOMBRE"])

            fig_mes_per = px.density_heatmap(
                resumen_mes_per,
                x="MES_NOMBRE",
                y="PERIODO_LLEGADA",
                z="Porc_Retrasado",
                color_continuous_scale="Reds",
                labels={
                    "MES_NOMBRE": "Mes",
                    "PERIODO_LLEGADA": "Período de llegada",
                    "Porc_Retrasado": "% retrasos (> 15 min)",
                },
                title="% de vuelos retrasados (> 15 min) por mes y período de llegada",
            )

            fig_mes_per.update_layout(
                title_x=0.5,
                plot_bgcolor="rgba(255, 248, 225, 0.6)",
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=60, r=40, t=70, b=60),
            )

            st.plotly_chart(fig_mes_per, use_container_width=True)

# ============================
# TAB 5 - CAUSAS DE RETRASO
# ============================
with tab_causas:
    st.subheader("Causas de retraso (> 15 min en llegada)")

    if "MOTIVO_RETRASO" not in df.columns or "ARRIVAL_DELAY" not in df.columns:
        st.info("El dataset no contiene las columnas necesarias (MOTIVO_RETRASO y/o ARRIVAL_DELAY).")
    else:
        # Trabajamos sobre el df ya filtrado por los controles del sidebar
        df_causas = df.copy()
        df_causas["ARRIVAL_DELAY"] = pd.to_numeric(
            df_causas["ARRIVAL_DELAY"], errors="coerce"
        )

        # Solo vuelos con retraso en llegada > 15 minutos
        df_causas = df_causas[df_causas["ARRIVAL_DELAY"] > 15]

        # Motivo no nulo
        df_causas = df_causas[df_causas["MOTIVO_RETRASO"].notna()]

        if df_causas.empty:
            st.info("No hay vuelos retrasados (> 15 min) con motivo de retraso para analizar.")
        else:
            # Limpiar nombre de motivo (quitar espacios repetidos)
            df_causas["MOTIVO_RETRASO_CLEAN"] = (
                df_causas["MOTIVO_RETRASO"]
                .astype(str)
                .str.strip()
                .str.replace(r"\s+", " ", regex=True)
            )

            st.markdown("---")

            # --------------------------------------------------
            # 2) Promedio de retraso por motivo (para tabla + dona)
            # --------------------------------------------------
            df_prom = (
                df_causas
                .groupby("MOTIVO_RETRASO_CLEAN")["ARRIVAL_DELAY"]
                .agg(promedio_retraso="mean", vuelos="size")
                .reset_index()
            )

            # Redondeos para mostrar
            df_prom["promedio_retraso"] = df_prom["promedio_retraso"].round(2)

            # Tabla ordenada por nombre de motivo
            st.markdown("### Promedio de retraso en llegada (> 15 min) por motivo")

            df_prom_tab = df_prom.sort_values("MOTIVO_RETRASO_CLEAN")
            st.dataframe(
                df_prom_tab.rename(
                    columns={
                        "MOTIVO_RETRASO_CLEAN": "Motivo retraso",
                        "promedio_retraso": "Promedio retraso (min)",
                        "vuelos": "Vuelos",
                    }
                ),
                use_container_width=True,
            )

            st.markdown("---")

            # --------------------------------------------------
            # 3) Gráfico tipo dona con TODAS las causas
            # --------------------------------------------------
            st.markdown("### Distribución de vuelos retrasados (> 15 min) por motivo")

            if df_prom.empty:
                st.info("No hay datos suficientes para construir la dona.")
            else:
                df_prom_graf = df_prom.copy()

                labels = df_prom_graf["MOTIVO_RETRASO_CLEAN"]
                values = df_prom_graf["vuelos"]

                # customdata: [vuelos, promedio_retraso] para el hover
                customdata = np.stack(
                    [df_prom_graf["vuelos"], df_prom_graf["promedio_retraso"]],
                    axis=-1
                )

                # Paleta secuencial suave
                base_palette = px.colors.sequential.YlOrRd
                colors = [
                    base_palette[i % len(base_palette)]
                    for i in range(len(labels))
                ]

                fig_prom = go.Figure(
                    data=[
                        go.Pie(
                            labels=labels,
                            values=values,
                            hole=0.55,                 # dona
                            sort=False,
                            direction="clockwise",
                            textposition="outside",    # etiquetas fuera
                            textinfo="label+percent",  # nombre + %
                            textfont=dict(size=11),
                            marker=dict(
                                colors=colors,
                                line=dict(color="black", width=0.4),
                            ),
                            pull=[0.03] * len(labels),  # ligero efecto de separación
                            customdata=customdata,
                            hovertemplate=(
                                "<b>%{label}</b><br>"
                                "Vuelos: %{customdata[0]:,}<br>"
                                "Promedio retraso: %{customdata[1]:.2f} min"
                                "<extra></extra>"
                            ),
                        )
                    ]
                )

                fig_prom.update_layout(
                    title=dict(
                        text=(
                            "Promedio de retraso en llegada (> 15 min) "
                            "por motivo (todas las causas)"
                        ),
                        x=0.5,
                    ),
                    showlegend=False,
                    margin=dict(l=40, r=40, t=80, b=40),
                )

                st.plotly_chart(fig_prom, use_container_width=True)
# -------------------------
# TAB 6 - Predicción interactiva
# -------------------------
with tab_prediccion:

        st.header("Simulador de Vuelos (Modelo de Planificación)")
        st.markdown("Introduce los detalles que conoce el pasajero: aerolínea, origen, destino, mes, día y hora de salida (HH:MM).")
        st.markdown("La hora de llegada no se ingresa; se usa la mediana histórica. Tiempo y distancia se obtienen desde tabla_rutas.")

        # if flights is None or flights.empty:
        #     st.warning("No hay datos históricos cargados. Revisa la ruta al CSV en la barra lateral.")
        # else:
        #     origen_df = flights[["ORIGIN_AIRPORT","ORIGEN_CIUDAD"]].drop_duplicates().rename(columns={"ORIGIN_AIRPORT":"code","ORIGEN_CIUDAD":"name"})
        #     destino_df = flights[["DESTINATION_AIRPORT","DEST_CIUDAD"]].drop_duplicates().rename(columns={"DESTINATION_AIRPORT":"code","DEST_CIUDAD":"name"})

        #     origen_options = {}
        #     for _, r in origen_df.iterrows():
        #         label = f"{r['code']} — {r['name']}" if pd.notna(r['name']) else f"{r['code']}"
        #         origen_options[label] = r['code']

        #     destino_options_full = {}
        #     for _, r in destino_df.iterrows():
        #         label = f"{r['code']} — {r['name']}" if pd.notna(r['name']) else f"{r['code']}"
        #         destino_options_full[label] = r['code']

        #     airline_codes = sorted(flights["AIRLINE"].dropna().unique().tolist())
        #     airline_options = [f"{c} — {AIRLINES_FULL.get(c, c)}" for c in airline_codes]

        #     col1, col2, col3 = st.columns(3)
        #     with col1:
        #         airline_sel = st.selectbox("Aerolínea", options=airline_options, key="airline_pred")
        #         airline_code = str(airline_sel).split(" — ")[0].strip()

        #         origen_sel = st.selectbox("Aeropuerto Origen", options=list(origen_options.keys()), key="origen_pred")
        #         origin_code = origen_options[origen_sel]
        
        # if flights is None or flights.empty:
        #     st.warning("No hay datos históricos cargados. Revisa la ruta al CSV en la barra lateral.")
        # else:
        #     # Dataframes base (todos los orígenes y destinos)
        #     origen_df = (
        #         flights[["ORIGIN_AIRPORT", "ORIGEN_CIUDAD"]]
        #         .drop_duplicates()
        #         .rename(columns={"ORIGIN_AIRPORT": "code", "ORIGEN_CIUDAD": "name"})
        #     )
        #     destino_df = (
        #         flights[["DESTINATION_AIRPORT", "DEST_CIUDAD"]]
        #         .drop_duplicates()
        #         .rename(columns={"DESTINATION_AIRPORT": "code", "DEST_CIUDAD": "name"})
        #     )

        #     # Destinos completos (los usamos más abajo, pero NO los filtramos aquí)
        #     destino_options_full = {}
        #     for _, r in destino_df.iterrows():
        #         label = f"{r['code']} — {r['name']}" if pd.notna(r['name']) else f"{r['code']}"
        #         destino_options_full[label] = r['code']

        #     # Aerolíneas: "CODE — Nombre"
        #     airline_codes = sorted(flights["AIRLINE"].dropna().unique().tolist())
        #     airline_options = [f"{c} — {AIRLINES_FULL.get(c, c)}" for c in airline_codes]

        #     col1, col2, col3 = st.columns(3)

        #     with col1:
        #         # 1️⃣ Seleccionas primero la aerolínea
        #         airline_sel = st.selectbox(
        #             "Aerolínea", options=airline_options, key="airline_pred"
        #         )
        #         airline_code = str(airline_sel).split(" — ")[0].strip()

        #         # 2️⃣ Filtras los ORÍGENES que tiene esa aerolínea
        #         origen_df_filtrado = (
        #             flights[flights["AIRLINE"] == airline_code]
        #             [["ORIGIN_AIRPORT", "ORIGEN_CIUDAD"]]
        #             .drop_duplicates()
        #             .rename(columns={"ORIGIN_AIRPORT": "code", "ORIGEN_CIUDAD": "name"})
        #         )

        #         # Si por alguna razón no hay rutas para esa aerolínea,
        #         # se muestra el listado completo como respaldo.
        #         if origen_df_filtrado.empty:
        #             origen_df_filtrado = origen_df

        #         origen_options = {}
        #         for _, r in origen_df_filtrado.iterrows():
        #             label = (
        #                 f"{r['code']} — {r['name']}"
        #                 if pd.notna(r['name']) and str(r['name']).strip() != ""
        #                 else f"{r['code']}"
        #             )
        #             origen_options[label] = r["code"]

        #         origen_sel = st.selectbox(
        #             "Aeropuerto Origen",
        #             options=list(origen_options.keys()),
        #             key="origen_pred",
        #         )
        #         origin_code = origen_options[origen_sel]

        if flights is None or flights.empty:
            st.warning("No hay datos históricos cargados. Revisa la ruta al CSV en la barra lateral.")
        else:
            # 🔁 Usar catálogos cacheados (muy rápidos)
            origen_df = get_origen_df(flights)
            destino_df = get_destino_df(flights)
            airline_options = get_airline_options(flights)
            origen_por_aerolinea = get_origen_por_aerolinea(flights)

            # Destinos completos (los usamos más abajo, pero NO los filtramos aquí)
            destino_options_full = {}
            for _, r in destino_df.iterrows():
                label = (
                    f"{r['code']} — {r['name']}"
                    if pd.notna(r['name']) and str(r['name']).strip() != ""
                    else f"{r['code']}"
                )
                destino_options_full[label] = r["code"]

            col1, col2, col3 = st.columns(3)

            with col1:
                # 1️⃣ Seleccionas primero la aerolínea
                airline_sel = st.selectbox(
                    "Aerolínea", options=airline_options, key="airline_pred"
                )
                airline_code = str(airline_sel).split(" — ")[0].strip()

                # 2️⃣ Filtras los ORÍGENES que tiene esa aerolínea,
                #    pero AHORA desde el catálogo cacheado, NO desde flights completo
                origen_df_filtrado = (
                    origen_por_aerolinea[
                        origen_por_aerolinea["AIRLINE"] == airline_code
                    ][["ORIGIN_AIRPORT", "ORIGEN_CIUDAD"]]
                    .drop_duplicates()
                    .rename(columns={"ORIGIN_AIRPORT": "code", "ORIGEN_CIUDAD": "name"})
                )

                # Si por alguna razón no hay rutas para esa aerolínea,
                # se muestra el listado completo como respaldo.
                if origen_df_filtrado.empty:
                    origen_df_filtrado = origen_df

                origen_options = {}
                for _, r in origen_df_filtrado.iterrows():
                    label = (
                        f"{r['code']} — {r['name']}"
                        if pd.notna(r['name']) and str(r['name']).strip() != ""
                        else f"{r['code']}"
                    )
                    origen_options[label] = r["code"]

                origen_sel = st.selectbox(
                    "Aeropuerto Origen",
                    options=list(origen_options.keys()),
                    key="origen_pred",
                )
                origin_code = origen_options[origen_sel]

            # ----- FILTRADO ESTRICTO: destinos para AIRLINE + ORIGIN -----
            valid_dest_df = flights[
                (flights["AIRLINE"] == airline_code) &
                (flights["ORIGIN_AIRPORT"] == origin_code)
            ][["DESTINATION_AIRPORT","DEST_CIUDAD"]].drop_duplicates()

            # Excluir mismo origen (por si aparece)
            valid_dest_df = valid_dest_df[valid_dest_df["DESTINATION_AIRPORT"] != origin_code]

            # Construir labels (codigo — ciudad)
            destino_options_filtrados = {}
            for _, row in valid_dest_df.iterrows():
                code = row["DESTINATION_AIRPORT"]
                name = row.get("DEST_CIUDAD", "")
                label = f"{code} — {name}" if pd.notna(name) and str(name).strip()!="" else f"{code}"
                destino_options_filtrados[label] = code

            with col2:
                if not destino_options_filtrados:
                    # Sin rutas para aerolinea+origen: mostrar indicador y bloquear predicción
                    destino_sel = st.selectbox("Aeropuerto Destino", options=["-- No hay rutas históricas para esta aerolínea desde el origen seleccionado --"], key="destino_none")
                    dest_code = None
                else:
                    destino_sel = st.selectbox("Aeropuerto Destino", options=list(destino_options_filtrados.keys()), key="destino_pred")
                    dest_code = destino_options_filtrados[destino_sel]

                # Mes y día con keys
                MONTHS = ["Enero","Febrero","Marzo","Abril","Mayo","Junio","Julio","Agosto","Septiembre","Octubre","Noviembre","Diciembre"]
                month_idx = st.selectbox("Mes", options=list(range(1,13)), format_func=lambda x: MONTHS[x-1], index=4, key="mes_pred")

                DAYS = {1:"Lunes",2:"Martes",3:"Miércoles",4:"Jueves",5:"Viernes",6:"Sábado",7:"Domingo"}
                day_of_week = st.selectbox("Día de la semana", options=list(DAYS.keys()), format_func=lambda x: DAYS[x], key="dia_pred")

            with col3:
                st.write("Hora de salida (HH:MM)")
                hora = st.selectbox("Hora (0–23)", list(range(24)), key="hora_pred")
                minuto = st.selectbox("Minuto (00–59)", ["{:02d}".format(i) for i in range(60)], key="minuto_pred")

            sched_dep = int(hora) * 100 + int(minuto)

            # Si no hay destino válido, no permite predecir
            # if dest_code is None:
            #     st.warning("No hay destinos válidos para la aerolínea y origen seleccionados. Cambia aerolínea u origen.")
            #     distancia = None
            #     sched_time = None
            #     sched_arr = None
            # else:
            #     ruta = tabla_rutas[
            #         (tabla_rutas["AIRLINE"] == airline_code) &
            #         (tabla_rutas["ORIGIN_AIRPORT"] == origin_code) &
            #         (tabla_rutas["DESTINATION_AIRPORT"] == dest_code)
            #     ]
            #     if ruta.empty:
            #         ruta = tabla_rutas[
            #             (tabla_rutas["ORIGIN_AIRPORT"] == origin_code) &
            #             (tabla_rutas["DESTINATION_AIRPORT"] == dest_code)
            #         ]

            #     if ruta.empty:
            #         st.warning("No se encontraron datos históricos para esta ruta.")
            #         distancia = None
            #         sched_time = None
            #         sched_arr = None
            #     else:
            #         distancia = float(ruta["DISTANCIA_HAV"].iloc[0]) if not pd.isna(ruta["DISTANCIA_HAV"].iloc[0]) else None
            #         sched_time = float(ruta["SCHEDULED_TIME"].iloc[0]) if not pd.isna(ruta["SCHEDULED_TIME"].iloc[0]) else None
            #         sched_arr = int(ruta["SCHEDULED_ARRIVAL"].iloc[0]) if not pd.isna(ruta["SCHEDULED_ARRIVAL"].iloc[0]) else None

            #         st.metric("Tiempo estimado (minutos, mediana histórica)", f"{int(sched_time):d}" if sched_time is not None else "N/A")
            #         st.metric("Distancia (millas, media histórica)", f"{distancia:.1f}" if distancia is not None else "N/A")
            #         if sched_arr is not None:
            #             st.info(f"Hora llegada estimada (mediana histórica): **{hhmm_to_hhmmss(sched_arr)}**")
            # if dest_code is None:
            #     st.warning(
            #         "No hay destinos válidos para la aerolínea y origen seleccionados. "
            #         "Cambia aerolínea u origen."
            #     )
            #     distancia = None
            #     sched_time = None
            #     sched_arr = None
            # else:
            #     # 1️⃣ Usar el dataframe filtrado del dashboard si existe,
            #     #    si no, usar flights completo como respaldo
            #     base_df = df if "df" in globals() else flights

            #     # 2️⃣ Filtrar por aerolínea, origen y destino
            #     df_ruta = base_df[
            #         (base_df["AIRLINE"] == airline_code) &
            #         (base_df["ORIGIN_AIRPORT"] == origin_code) &
            #         (base_df["DESTINATION_AIRPORT"] == dest_code)
            #     ]

            #     # 3️⃣ Opcional: afinar aún más por MES y DÍA seleccionados en el simulador
            #     #     (si quieres que dependa de esos selects también)
            #     df_ruta = df_ruta[
            #         (df_ruta["MONTH"] == month_idx) &
            #         (df_ruta["DAY_OF_WEEK"] == day_of_week)
            #     ]

            #     if df_ruta.empty:
            #         # 🔁 Respaldo: si con filtros no hay datos, usamos tabla_rutas global
            #         ruta = tabla_rutas[
            #             (tabla_rutas["AIRLINE"] == airline_code) &
            #             (tabla_rutas["ORIGIN_AIRPORT"] == origin_code) &
            #             (tabla_rutas["DESTINATION_AIRPORT"] == dest_code)
            #         ]

            #         if ruta.empty:
            #             st.warning(
            #                 "No se encontraron datos históricos para esta ruta con los filtros seleccionados."
            #             )
            #             distancia = None
            #             sched_time = None
            #             sched_arr = None
            #         else:
            #             distancia = (
            #                 float(ruta["DISTANCIA_HAV"].iloc[0])
            #                 if not pd.isna(ruta["DISTANCIA_HAV"].iloc[0])
            #                 else None
            #             )
            #             sched_time = (
            #                 float(ruta["SCHEDULED_TIME"].iloc[0])
            #                 if not pd.isna(ruta["SCHEDULED_TIME"].iloc[0])
            #                 else None
            #             )
            #             sched_arr = (
            #                 int(ruta["SCHEDULED_ARRIVAL"].iloc[0])
            #                 if not pd.isna(ruta["SCHEDULED_ARRIVAL"].iloc[0])
            #                 else None
            #             )
            #     else:
            #         # ✅ Aquí sí depende 100% de los datos seleccionados
            #         distancia = float(df_ruta["DISTANCE"].mean()) if not df_ruta["DISTANCE"].isna().all() else None
            #         sched_time = float(df_ruta["SCHEDULED_TIME"].median()) if not df_ruta["SCHEDULED_TIME"].isna().all() else None
            #         sched_arr = (
            #             int(df_ruta["SCHEDULED_ARRIVAL"].median())
            #             if not df_ruta["SCHEDULED_ARRIVAL"].isna().all()
            #             else None
            #         )

            #     # 4️⃣ Mostrar métricas (si se logró calcular algo)
            #     if sched_time is not None:
            #         st.metric(
            #             "Tiempo estimado (minutos, mediana histórica)",
            #             f"{int(sched_time):d}",
            #         )
            #     else:
            #         st.metric(
            #             "Tiempo estimado (minutos, mediana histórica)",
            #             "N/A",
            #         )

            #     if distancia is not None:
            #         st.metric(
            #             "Distancia (millas, media histórica)",
            #             f"{distancia:.1f}",
            #         )
            #     else:
            #         st.metric(
            #             "Distancia (millas, media histórica)",
            #             "N/A",
            #         )

            #     if sched_arr is not None:
            #         st.info(
            #             f"Hora llegada estimada (mediana histórica): "
            #             f"**{hhmm_to_hhmmss(sched_arr)}**"
                    # )
            if dest_code is None:
                st.warning(
                    "No hay destinos válidos para la aerolínea y origen seleccionados. "
                    "Cambia aerolínea u origen."
                )
                distancia = None
                sched_time = None
                sched_arr = None
            else:
                # 1️⃣ Usar el dataframe filtrado del dashboard si existe,
                #    si no, usar flights completo como respaldo
                base_df = df if "df" in globals() else flights

                # 2️⃣ Filtrar por aerolínea, origen y destino
                df_ruta = base_df[
                    (base_df["AIRLINE"] == airline_code) &
                    (base_df["ORIGIN_AIRPORT"] == origin_code) &
                    (base_df["DESTINATION_AIRPORT"] == dest_code)
                ]

                # 3️⃣ Afinar aún más por MES y DÍA seleccionados en el simulador
                df_ruta = df_ruta[
                    (df_ruta["MONTH"] == month_idx) &
                    (df_ruta["DAY_OF_WEEK"] == day_of_week)
                ]

                if df_ruta.empty:
                    # 🔁 Respaldo: si con filtros no hay datos, usamos tabla_rutas global
                    ruta = tabla_rutas[
                        (tabla_rutas["AIRLINE"] == airline_code) &
                        (tabla_rutas["ORIGIN_AIRPORT"] == origin_code) &
                        (tabla_rutas["DESTINATION_AIRPORT"] == dest_code)
                    ]

                    if ruta.empty:
                        st.warning(
                            "No se encontraron datos históricos para esta ruta con los filtros seleccionados."
                        )
                        distancia = None
                        sched_time = None
                        sched_arr = None
                    else:
                        distancia = (
                            float(ruta["DISTANCIA_HAV"].iloc[0])
                            if not pd.isna(ruta["DISTANCIA_HAV"].iloc[0])
                            else None
                        )
                        sched_time = (
                            float(ruta["SCHEDULED_TIME"].iloc[0])
                            if not pd.isna(ruta["SCHEDULED_TIME"].iloc[0])
                            else None
                        )
                else:
                    # ✅ Aquí sí depende 100% de los datos seleccionados
                    distancia = (
                        float(df_ruta["DISTANCE"].mean())
                        if not df_ruta["DISTANCE"].isna().all()
                        else None
                    )
                    sched_time = (
                        float(df_ruta["SCHEDULED_TIME"].median())
                        if not df_ruta["SCHEDULED_TIME"].isna().all()
                        else None
                    )

                # 4️⃣ Calcular hora de llegada a partir de:
                #    hora de salida seleccionada + duración histórica (sched_time)
                if sched_time is not None:
                    # hora es int (0–23), minuto es string "00"–"59"
                    dep_min = int(hora) * 60 + int(minuto)          # minutos desde medianoche
                    arr_min = dep_min + int(round(sched_time))      # sumar duración en minutos

                    # convertir de nuevo a HHMM (24h, manejando cruce de día)
                    arr_hour = (arr_min // 60) % 24
                    arr_minute = arr_min % 60
                    sched_arr = arr_hour * 100 + arr_minute
                else:
                    sched_arr = None

                # 5️⃣ Mostrar métricas
                if sched_time is not None:
                    st.metric(
                        "Tiempo estimado (minutos, mediana histórica)",
                        f"{int(sched_time):d}",
                    )
                else:
                    st.metric(
                        "Tiempo estimado (minutos, mediana histórica)",
                        "N/A",
                    )

                if distancia is not None:
                    st.metric(
                        "Distancia (millas, media histórica)",
                        f"{distancia:.1f}",
                    )
                else:
                    st.metric(
                        "Distancia (millas, media histórica)",
                        "N/A",
                    )

                if sched_arr is not None:
                    st.info(
                        f"Hora llegada estimada (mediana histórica): "
                        f"**{hhmm_to_hhmmss(sched_arr)}**"
                    )

            st.markdown("---")

            # Botón PREDECIR
            col_btn, col_sp = st.columns([1, 5])
            with col_btn:
                predict_click = st.button("Predecir Retraso", type="primary", key="predict_btn")

            if predict_click:
                if artifacts is None:
                    st.error("Artefactos del modelo no cargados. Coloca los .joblib en /models/.")
                elif dest_code is None:
                    st.error("No hay destino válido seleccionado. Elige una aerolínea/origen con rutas históricas.")
                elif distancia is None or sched_time is None:
                    st.error("Faltan datos históricos (distancia o tiempo programado) para esta ruta.")
                else:
                    input_data = {
                        "MONTH": int(month_idx),
                        "DAY_OF_WEEK": int(day_of_week),
                        "AIRLINE": airline_code,
                        "ORIGIN_AIRPORT": origin_code,
                        "DESTINATION_AIRPORT": dest_code,
                        "SCHEDULED_DEPARTURE": int(sched_dep),
                        "SCHEDULED_ARRIVAL": int(sched_arr) if sched_arr is not None else 0,
                        "SCHEDULED_TIME": float(sched_time),
                        "DISTANCE": float(distancia)
                    }
                    df_input = pd.DataFrame([input_data])
                    
                    # # === DEBUG opcional: comparar columnas ===
                    # st.write("🔎 Columnas de df_input (antes de preprocesar):")
                    # st.write(df_input.columns.tolist())
                    
                    try:
                        Xp = preprocess_data_for_api(df_input, artifacts)
                        
                        #--
                        # # DEBUG: comprobar que las columnas finales son las esperadas
                        # st.write("✅ Columnas que se envían al modelo (Xp.columns):")
                        # st.write(Xp.columns.tolist())
                        # st.write(f"Shape de Xp: {Xp.shape}")

                        # # (opcional) Validación fuerte
                        # missing = set(artifacts["feature_order"]) - set(Xp.columns)
                        # extra   = set(Xp.columns) - set(artifacts["feature_order"])
                        # if missing:
                        #     st.error(f"Faltan columnas en Xp respecto a feature_order: {missing}")
                        # if extra:
                        #     st.warning(f"Hay columnas extra en Xp que el modelo no esperaba: {extra}")
                        
                        # #--
                        
                        
                        model = artifacts["model"]
                        probs = model.predict_proba(Xp)[0]
                        prob_delay = float(probs[1])

                        col_r1, col_r2 = st.columns([2,1])
                        col_r1.metric("Probabilidad de llegar >15 min tarde", f"{prob_delay*100:.2f}%")
                        col_r2.progress(prob_delay)
                        # Definimos el umbral adecuado para el modelo de planificación
                        UMBRAL_OPTIMO = 0.423

                        if prob_delay > UMBRAL_OPTIMO:
                            st.error("⚠️ Retraso probable (por encima del umbral óptimo).")
                        else:
                            st.success("✅ Probablemente a tiempo (por debajo del umbral).")

                        origin_name = origen_sel
                        dest_name = destino_sel if dest_code is not None else ""
                        log_entry = {
                            "timestamp_utc": datetime.utcnow().isoformat(),
                            "airline": airline_code,
                            "origin_code": origin_code,
                            "origin_name": origin_name,
                            "dest_code": dest_code,
                            "dest_name": dest_name,
                            "month": month_idx,
                            "day_of_week": day_of_week,
                            "scheduled_dep": sched_dep,
                            "scheduled_arr": int(sched_arr) if sched_arr is not None else 0,
                            "scheduled_time": sched_time,
                            "distance": distancia,
                            "prob_delay": prob_delay
                        }
                        append_log(log_entry)

                    except Exception as e:
                        st.error(f"Error en la predicción: {e}")
                        if "Booster' object has no attribute 'handle" in str(e) or "lib_lightgbm.dll" in str(e) or "Could not find module" in str(e):
                            st.info("Error típico de incompatibilidad LightGBM/DLL. Reinstala la versión de lightgbm con la que entrenaste (ej. pip install lightgbm==3.3.5).")
                        st.exception(e)    
        st.markdown("---")
            
            # Descargar log si existe
        if PREDICTIONS_LOG.exists():
            csv_bytes = get_log_bytes()
            if csv_bytes:
                st.download_button("⬇️ Descargar log de predicciones", data=csv_bytes, file_name="predictions_log.csv", mime="text/csv")
        else:
            st.info("Aún no hay predicciones registradas.")

        # Mensaje artefactos
        if artifacts is None:
            st.warning("Artefactos del modelo no cargados. Predicción deshabilitada.")
        else:
            st.success("Artefactos cargados ✓")
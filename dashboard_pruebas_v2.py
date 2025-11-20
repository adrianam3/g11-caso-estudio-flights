import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime
import plotly.express as px

# -------------------------
# Configuracion y rutas
# -------------------------
st.set_page_config(page_title="Predicción de Vuelos", layout="wide")
PROJECT_ROOT = Path(__file__).resolve().parent

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "flight_delay_lgbm.joblib"
PREPROCESSOR_PATH = MODELS_DIR / "flight_delay_preprocessors.joblib"

# Ruta CSV (usuario especificada)
DATA_PATH = Path(r"D:\OneDrive\DOCUMENTOS\Personales\2024\uniandes\8 S\seminario\g11-caso-estudio-flights\data\processed\flights_clean.csv")

# Log de predicciones
PREDICTIONS_LOG = PROJECT_ROOT / "predictions_log.csv"

# Diccionario nombres aerolíneas 
AIRLINES_FULL = {
    "AA": "American Airlines",
    "DL": "Delta Air Lines",
    "UA": "United Airlines",
    "AS": "Alaska Airlines",
    "WN": "Southwest Airlines",
    "B6": "JetBlue",
    "NK": "Spirit Airlines",
    "F9": "Frontier Airlines",
}

# -------------------------
# Datos utiles
# -------------------------
def safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception as e:
        st.error(f"Error leyendo CSV {path}: {e}")
        return pd.DataFrame()

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
# Carga artefactos - modelo y preprocesador
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

# -------------------------
# Carga dataset y tabla de rutas
# -------------------------
@st.cache_data
def cargar_dataset_y_tabla_rutas(csv_path: Path):
    df = safe_read_csv(csv_path)
    if df.empty:
        return df, pd.DataFrame()

    df.columns = [c.upper() for c in df.columns]

    expected = [
        "AIRLINE","ORIGIN_AIRPORT","DESTINATION_AIRPORT","DISTANCE","SCHEDULED_TIME",
        "SCHEDULED_ARRIVAL","MONTH","DAY_OF_WEEK","ARRIVAL_DELAY",
        "ORIGEN_CIUDAD","DEST_CIUDAD","MOTIVO_RETRASO"
    ]
    for c in expected:
        if c not in df.columns:
            df[c] = np.nan

    dias = {1:"Lunes",2:"Martes",3:"Miércoles",4:"Jueves",5:"Viernes",6:"Sábado",7:"Domingo"}
    try:
        df["DAY_OF_WEEK"] = df["DAY_OF_WEEK"].astype(int)
        df["DIA_NOMBRE"] = df["DAY_OF_WEEK"].map(dias)
    except Exception:
        df["DIA_NOMBRE"] = df["DAY_OF_WEEK"].astype(str)

    tabla = (
        df.groupby(["AIRLINE","ORIGIN_AIRPORT","DESTINATION_AIRPORT"], dropna=False)
          .agg(DISTANCIA_HAV=("DISTANCE","mean"),
               SCHEDULED_TIME=("SCHEDULED_TIME","median"),
               SCHEDULED_ARRIVAL=("SCHEDULED_ARRIVAL","median"))
          .reset_index()
    )

    for c in ["DISTANCIA_HAV","SCHEDULED_TIME","SCHEDULED_ARRIVAL"]:
        if c in tabla.columns:
            tabla[c] = pd.to_numeric(tabla[c], errors="coerce")

    return df, tabla

flights_clean, tabla_rutas = cargar_dataset_y_tabla_rutas(DATA_PATH)

# -------------------------
# Preprocess para inferencia
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

    df["RUTA"] = df["ORIGIN_AIRPORT"].astype(str) + "_" + df["DESTINATION_AIRPORT"].astype(str)

    sched_dep = pd.to_numeric(df["SCHEDULED_DEPARTURE"], errors="coerce").fillna(0).astype(int)
    hs = (sched_dep // 100).clip(0,23)
    ms = (sched_dep % 100).clip(0,59)
    minuto_dia_salida = hs * 60 + ms
    df["SALIDA_SIN"] = np.sin(2 * np.pi * minuto_dia_salida / (24*60))
    df["SALIDA_COS"] = np.cos(2 * np.pi * minuto_dia_salida / (24*60))

    sched_arr = pd.to_numeric(df.get("SCHEDULED_ARRIVAL", 0), errors="coerce").fillna(0).astype(int)
    hl = (sched_arr // 100).clip(0,23)
    ml = (sched_arr % 100).clip(0,59)
    minuto_dia_llegada = hl * 60 + ml
    df["LLEGADA_SIN"] = np.sin(2 * np.pi * minuto_dia_llegada / (24*60))
    df["LLEGADA_COS"] = np.cos(2 * np.pi * minuto_dia_llegada / (24*60))

    df["MONTH_SIN"] = np.sin(2 * np.pi * df["MONTH"].astype(float) / 12)
    df["MONTH_COS"] = np.cos(2 * np.pi * df["MONTH"].astype(float) / 12)

    if "DISTANCE" in df.columns and "DISTANCIA_HAV" not in df.columns:
        df["DISTANCIA_HAV"] = df["DISTANCE"]

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

    for col in FINAL_FEATURE_ORDER:
        if col not in df.columns:
            df[col] = 0

    X = df[FINAL_FEATURE_ORDER].copy()
    return X

# -------------------------
# UI principal: sidebar + tabs
# -------------------------
def main(artifacts):
    st.sidebar.title("Navegación")
    st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c2/LightGBM_logo_black_text.svg/1280px-LightGBM_logo_black_text.svg.png", use_container_width=True)
    st.sidebar.info("Este dashboard usa un modelo LGBM para predecir la probabilidad de retrasos de vuelos.")
    st.sidebar.markdown("---")

    st.sidebar.header("Ruta CSV (si quieres cambiarla)")
    csv_input = st.sidebar.text_input("Ruta al CSV histórico (absoluta o relativa):", value=str(DATA_PATH))
    st.sidebar.markdown("---")

    if PREDICTIONS_LOG.exists():
        csv_bytes = get_log_bytes()
        if csv_bytes:
            st.sidebar.download_button("⬇️ Descargar log de predicciones", data=csv_bytes, file_name="predictions_log.csv", mime="text/csv")
    else:
        st.sidebar.info("Aún no hay predicciones registradas.")

    if artifacts is None:
        st.sidebar.warning("Artefactos del modelo no cargados. Predicción deshabilitada.")
    else:
        st.sidebar.success("Artefactos cargados ✓")

    global flights_clean, tabla_rutas
    if csv_input and Path(csv_input).exists() and Path(csv_input) != DATA_PATH:
        flights_clean, tabla_rutas = cargar_dataset_y_tabla_rutas(Path(csv_input))

    tab1, tab2 = st.tabs(["✈️ Predicción Interactiva", "📊 Análisis de Datos Históricos"])

    # -------------------------
    # PESTAÑA 1: Predicción interactiva -VALIDACIÓN ESTRICTA
    # -------------------------
    with tab1:
        st.header("Simulador de Vuelos (Modelo de Planificación)")
        st.markdown("Introduce los detalles que conoce el pasajero: aerolínea, origen, destino, mes, día y hora de salida (HH:MM).")
        st.markdown("La hora de llegada no se ingresa; se usa la mediana histórica. Tiempo y distancia se obtienen desde tabla_rutas.")

        if flights_clean is None or flights_clean.empty:
            st.warning("No hay datos históricos cargados. Revisa la ruta al CSV en la barra lateral.")
        else:
            origen_df = flights_clean[["ORIGIN_AIRPORT","ORIGEN_CIUDAD"]].drop_duplicates().rename(columns={"ORIGIN_AIRPORT":"code","ORIGEN_CIUDAD":"name"})
            destino_df = flights_clean[["DESTINATION_AIRPORT","DEST_CIUDAD"]].drop_duplicates().rename(columns={"DESTINATION_AIRPORT":"code","DEST_CIUDAD":"name"})

            origen_options = {}
            for _, r in origen_df.iterrows():
                label = f"{r['code']} — {r['name']}" if pd.notna(r['name']) else f"{r['code']}"
                origen_options[label] = r['code']

            destino_options_full = {}
            for _, r in destino_df.iterrows():
                label = f"{r['code']} — {r['name']}" if pd.notna(r['name']) else f"{r['code']}"
                destino_options_full[label] = r['code']

            airline_codes = sorted(flights_clean["AIRLINE"].dropna().unique().tolist())
            airline_options = [f"{c} — {AIRLINES_FULL.get(c, c)}" for c in airline_codes]

            col1, col2, col3 = st.columns(3)
            with col1:
                airline_sel = st.selectbox("Aerolínea", options=airline_options, key="airline_pred")
                airline_code = str(airline_sel).split(" — ")[0].strip()

                origen_sel = st.selectbox("Aeropuerto Origen", options=list(origen_options.keys()), key="origen_pred")
                origin_code = origen_options[origen_sel]

            # ----- FILTRADO ESTRICTO: destinos para AIRLINE + ORIGIN -----
            valid_dest_df = flights_clean[
                (flights_clean["AIRLINE"] == airline_code) &
                (flights_clean["ORIGIN_AIRPORT"] == origin_code)
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
            if dest_code is None:
                st.warning("No hay destinos válidos para la aerolínea y origen seleccionados. Cambia aerolínea u origen.")
                distancia = None
                sched_time = None
                sched_arr = None
            else:
                ruta = tabla_rutas[
                    (tabla_rutas["AIRLINE"] == airline_code) &
                    (tabla_rutas["ORIGIN_AIRPORT"] == origin_code) &
                    (tabla_rutas["DESTINATION_AIRPORT"] == dest_code)
                ]
                if ruta.empty:
                    ruta = tabla_rutas[
                        (tabla_rutas["ORIGIN_AIRPORT"] == origin_code) &
                        (tabla_rutas["DESTINATION_AIRPORT"] == dest_code)
                    ]

                if ruta.empty:
                    st.warning("No se encontraron datos históricos para esta ruta.")
                    distancia = None
                    sched_time = None
                    sched_arr = None
                else:
                    distancia = float(ruta["DISTANCIA_HAV"].iloc[0]) if not pd.isna(ruta["DISTANCIA_HAV"].iloc[0]) else None
                    sched_time = float(ruta["SCHEDULED_TIME"].iloc[0]) if not pd.isna(ruta["SCHEDULED_TIME"].iloc[0]) else None
                    sched_arr = int(ruta["SCHEDULED_ARRIVAL"].iloc[0]) if not pd.isna(ruta["SCHEDULED_ARRIVAL"].iloc[0]) else None

                    st.metric("Tiempo estimado (minutos, mediana histórica)", f"{int(sched_time):d}" if sched_time is not None else "N/A")
                    st.metric("Distancia (millas, media histórica)", f"{distancia:.1f}" if distancia is not None else "N/A")
                    if sched_arr is not None:
                        st.info(f"Hora llegada estimada (mediana histórica): **{hhmm_to_hhmmss(sched_arr)}**")

            st.markdown("---")

            # Botón PREDECIR
            col_btn, col_sp = st.columns([1, 5])
            with col_btfn:
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
                    try:
                        Xp = preprocess_data_for_api(df_input, artifacts)
                        model = artifacts["model"]
                        probs = model.predict_proba(Xp)[0]
                        prob_delay = float(probs[1])

                        col_r1, col_r2 = st.columns([2,1])
                        col_r1.metric("Probabilidad de llegar >15 min tarde", f"{prob_delay*100:.2f}%")
                        col_r2.progress(prob_delay)

                        if prob_delay > 0.708:
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

    # -------------------------
    # PESTAÑA 2: Análisis de Datos Históricos
    # -------------------------
    with tab2:
        st.header("Análisis de Datos Históricos")
        st.markdown("Estos gráficos se basan en tus datos históricos y muestran porcentajes sobre el subconjunto filtrado.")

        if flights_clean is None or flights_clean.empty:
            st.warning("No hay datos históricos cargados.")
        else:
            colf1, colf2 = st.columns([2,1])
            with colf1:
                filtro_airline = st.selectbox("Filtrar por Aerolínea (Todas)", options=["Todas"] + sorted(flights_clean["AIRLINE"].dropna().unique().tolist()), key="filtro_air")
            with colf2:
                only_retrasos = st.checkbox("Mostrar solo vuelos retrasados (ARRIVAL_DELAY > 0)", value=True, key="chk_retrasos")

            df_fil = flights_clean.copy()
            if filtro_airline != "Todas":
                df_fil = df_fil[df_fil["AIRLINE"] == filtro_airline]
            if only_retrasos:
                df_fil = df_fil[pd.to_numeric(df_fil["ARRIVAL_DELAY"], errors="coerce") > 0]

            st.subheader("KPIs")
            k1, k2, k3 = st.columns(3)
            k1.metric("Registros mostrados", f"{len(df_fil):,}")
            k2.metric("Registros retrasados (filtrados)", f"{len(df_fil):,}")
            pct = (len(df_fil) / (len(flights_clean) if len(flights_clean)>0 else 1) * 100)
            k3.metric("% sobre histórico", f"{pct:.2f}%")

            st.markdown("---")
            df_pct_dias = df_fil["DIA_NOMBRE"].value_counts(normalize=True).mul(100).round(2).reset_index()
            df_pct_dias.columns = ["DIA_NOMBRE","PORCENTAJE"]
            order = ["Lunes","Martes","Miércoles","Jueves","Viernes","Sábado","Domingo"]
            df_pct_dias["ORD"] = df_pct_dias["DIA_NOMBRE"].apply(lambda x: order.index(x) if x in order else 99)
            df_pct_dias = df_pct_dias.sort_values("ORD").drop(columns="ORD")
            st.dataframe(df_pct_dias)
            if not df_pct_dias.empty:
                fig = px.bar(df_pct_dias, x="DIA_NOMBRE", y="PORCENTAJE", text="PORCENTAJE")
                fig.update_traces(texttemplate="%{text:.2f}%")
                st.plotly_chart(fig, width="stretch")

            st.markdown("---")
            df_origen = df_fil["ORIGIN_AIRPORT"].value_counts().nlargest(15).reset_index()
            df_origen.columns = ["ORIGIN_AIRPORT","CANTIDAD"]
            if not df_origen.empty:
                df_origen["PORCENTAJE"] = (df_origen["CANTIDAD"] / df_origen["CANTIDAD"].sum() * 100).round(2)
                st.dataframe(df_origen)
                fig2 = px.bar(df_origen, x="ORIGIN_AIRPORT", y="PORCENTAJE", text="PORCENTAJE")
                fig2.update_traces(texttemplate="%{text:.2f}%")
                fig2.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig2, width="stretch")

            st.markdown("---")
            if "MOTIVO_RETRASO" in df_fil.columns:
                df_mot = df_fil[~df_fil["MOTIVO_RETRASO"].astype(str).str.lower().isin(["sin retraso","sin dato","nan",""])]
                df_mot_top = df_mot["MOTIVO_RETRASO"].value_counts().nlargest(15).reset_index()
                df_mot_top.columns = ["MOTIVO_RETRASO","CANTIDAD"]
                if not df_mot_top.empty:
                    df_mot_top["PORCENTAJE"] = (df_mot_top["CANTIDAD"] / df_mot_top["CANTIDAD"].sum() * 100).round(2)
                    st.dataframe(df_mot_top)
                    fig3 = px.bar(df_mot_top, x="MOTIVO_RETRASO", y="PORCENTAJE", text="PORCENTAJE")
                    fig3.update_traces(texttemplate="%{text:.2f}%")
                    fig3.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig3, width="stretch")

            st.markdown("---")
            if "MONTH" in df_fil.columns:
                meses = df_fil.groupby("MONTH").size().reset_index(name="CANTIDAD")
                meses["PORCENTAJE"] = (meses["CANTIDAD"] / meses["CANTIDAD"].sum() * 100).round(2)
                meses["MES_NOMBRE"] = meses["MONTH"].apply(lambda x: ["Enero","Febrero","Marzo","Abril","Mayo","Junio","Julio","Agosto","Septiembre","Octubre","Noviembre","Diciembre"][int(x)-1] if pd.notna(x) and 1<=int(x)<=12 else str(x))
                st.dataframe(meses[["MES_NOMBRE","CANTIDAD","PORCENTAJE"]])
                fig4 = px.line(meses.sort_values("MONTH"), x="MES_NOMBRE", y="PORCENTAJE", markers=True, text="PORCENTAJE")
                fig4.update_traces(texttemplate="%{text:.2f}%")
                st.plotly_chart(fig4, width="stretch")
  
# Ejecutar
if __name__ == "__main__":
    main(artifacts)

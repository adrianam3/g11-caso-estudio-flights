import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import os
import math

# ==============================================================================
# 1. Cargar Artefactos (Modelo y Preprocesadores)
# ==============================================================================
# Esto se ejecuta UNA SOLA VEZ al iniciar la API y se cachea

@st.cache_resource
def load_artifacts():
    """
    Carga el modelo, el scaler y los label encoders desde la carpeta /models.
    """
    print("Cargando artefactos del modelo...")
    PROJECT_ROOT = Path(__file__).resolve().parent
    MODELS_DIR = PROJECT_ROOT / "models"
    MODEL_PATH = MODELS_DIR / "flight_delay_lgbm.joblib"
    PREPROCESSOR_PATH = MODELS_DIR / "flight_delay_preprocessors.joblib"

    try:
        model = joblib.load(MODEL_PATH)
        print(f"✓ Modelo cargado desde {MODEL_PATH}")
        
        preprocessors = joblib.load(PREPROCESSOR_PATH)
        print(f"✓ Preprocesadores cargados desde {PREPROCESSOR_PATH}")

        artifacts = {
            "model": model,
            "label_encoders": preprocessors["label_encoders"],
            "scaler": preprocessors["scaler"],
            "cat_cols": preprocessors["cat_features_names"],
            "num_cols": preprocessors["num_features_names"]
        }
        # Crear el orden final de features con el que se entrenó
        artifacts["feature_order"] = artifacts["cat_cols"] + artifacts["num_cols"]
        
        print("✓ Artefactos cargados exitosamente.")
        return artifacts
    
    except FileNotFoundError as e:
        print(f"Error: No se encontraron los archivos del modelo en {MODELS_DIR}")
        print(f"Detalle: {e}")
        print("Por favor, ejecuta el script 'train.py' (v21) para generar los artefactos.")
        return None
    except Exception as e:
        print(f"Error desconocido al cargar artefactos: {e}")
        return None

# ==============================================================================
# 2. Función de Preprocesamiento para Inferencia (API/Dashboard)
# ==============================================================================
# Esta función replica EXACTAMENTE los pasos de model_preprocessing.py

def preprocess_data_for_api(df: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    """
    Aplica la transformación completa a los datos nuevos (1 fila).
    """
    # Extraer artefactos
    label_encoders = artifacts["label_encoders"]
    scaler = artifacts["scaler"]
    CAT_COLS = artifacts["cat_cols"]
    NUM_COLS = artifacts["num_cols"]
    FINAL_FEATURE_ORDER = artifacts["feature_order"]

    # 1. Derivar RUTA
    df["RUTA"] = df["ORIGIN_AIRPORT"].astype(str) + "_" + df["DESTINATION_AIRPORT"].astype(str)
    
    # 2. Derivar Cíclicas (Salida)
    hs = (df["SCHEDULED_DEPARTURE"] // 100).clip(0, 23)
    ms = (df["SCHEDULED_DEPARTURE"] % 100).clip(0, 59)
    minuto_dia_salida = (hs * 60 + ms)
    df["SALIDA_SIN"] = np.sin(2 * np.pi * minuto_dia_salida / (24*60))
    df["SALIDA_COS"] = np.cos(2 * np.pi * minuto_dia_salida / (24*60))

    # 3. Derivar Cíclicas (Llegada)
    hl = (df["SCHEDULED_ARRIVAL"] // 100).clip(0, 23)
    ml = (df["SCHEDULED_ARRIVAL"] % 100).clip(0, 59)
    minuto_dia_llegada = (hl * 60 + ml)
    df["LLEGADA_SIN"] = np.sin(2 * np.pi * minuto_dia_llegada / (24*60))
    df["LLEGADA_COS"] = np.cos(2 * np.pi * minuto_dia_llegada / (24*60))
    
    # 4. Derivar Cíclicas (Mes)
    df["MONTH_SIN"] = np.sin(2 * np.pi * df["MONTH"] / 12)
    df["MONTH_COS"] = np.cos(2 * np.pi * df["MONTH"] / 12)

    # (Renombrar DISTANCE a DISTANCIA_HAV si es necesario)
    if "DISTANCE" in df.columns and "DISTANCIA_HAV" not in df.columns:
        df["DISTANCIA_HAV"] = df["DISTANCE"]

    # 5. Aplicar LabelEncoders (con manejo de <unknown>)
    for col in CAT_COLS:
        if col in label_encoders:
            le = label_encoders[col]
            le_classes = set(le.classes_)
            df[col] = df[col].astype(str).apply(lambda x: x if x in le_classes else '<unknown>')
            df[col] = le.transform(df[col])
        else:
            print(f"Advertencia: No se encontró LabelEncoder para {col}")
    
    # 6. Aplicar StandardScaler
    # Asegurarse de que solo escalamos las columnas que existen en el scaler
    cols_to_scale = [col for col in NUM_COLS if col in df.columns]
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])
    
    # 7. Asegurar Orden Final
    # Asegurarse de que todas las columnas de feature_order existen, rellenando si falta alguna
    for col in FINAL_FEATURE_ORDER:
        if col not in df.columns:
            df[col] = 0 # Placeholder por si falta alguna derivada (ej. LLEGADA_SIN)
            
    df = df[FINAL_FEATURE_ORDER]
    
    return df

# ==============================================================================
# 3. Función Principal del Dashboard (AHORA RECIBE 'artifacts')
# ==============================================================================

def main(artifacts):
    st.set_page_config(page_title="Predicción de Vuelos", layout="wide")
    
    # --- artifacts ya están cargados ---
    if artifacts is None:
        st.error("Error al cargar los artefactos del modelo. Asegúrate de que los archivos .joblib existan en la carpeta /models.")
        return

    model = artifacts["model"]

    # --- Sidebar ---
    st.sidebar.title("Navegación")
    st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c2/LightGBM_logo_black_text.svg/1280px-LightGBM_logo_black_text.svg.png", use_column_width=True)
    st.sidebar.info("Este dashboard usa un modelo LGBM para predecir la probabilidad de retrasos de vuelos.")
    
    # Definir Pestañas
    tab1, tab2, tab3 = st.tabs([
        "✈️ Predicción Interactiva", 
        "📊 Análisis de Datos Históricos", 
        "📈 Métricas del Modelo"
    ])

    # ==========================
    # Pestaña 1: Predicción Interactiva
    # ==========================
    with tab1:
        st.header("Simulador de Vuelos (Modelo de Planificación)")
        st.markdown("Introduce los detalles de un vuelo **antes** de que despegue para predecir su probabilidad de retraso en la llegada (>15 min).")
        st.markdown("*(Este modelo NO usa el retraso en la salida, por eso su precisión es limitada)*")
        
        with st.container(border=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("Ruta")
                # Obtener listas de ejemplo (idealmente cargar desde un archivo)
                # Usaremos los encoders para obtener las clases conocidas
                le_airline = artifacts["label_encoders"]["AIRLINE"]
                le_origin = artifacts["label_encoders"]["ORIGIN_AIRPORT"]
                le_dest = artifacts["label_encoders"]["DESTINATION_AIRPORT"]
                
                # Excluir <unknown> si existe
                air_options = [c for c in le_airline.classes_ if c != '<unknown>']
                ori_options = [c for c in le_origin.classes_ if c != '<unknown>']
                des_options = [c for c in le_dest.classes_ if c != '<unknown>']

                airline = st.selectbox("Aerolínea (ej: AA, DL, UA)", options=air_options, index=0)
                origin = st.selectbox("Aeropuerto Origen (ej: JFK, LAX)", options=ori_options, index=ori_options.index("JFK") if "JK" in ori_options else 0)
                dest = st.selectbox("Aeropuerto Destino (ej: LAX, SFO)", options=des_options, index=des_options.index("LAX") if "LAX" in des_options else 0)

            with col2:
                st.subheader("Tiempo Programado")
                month = st.slider("Mes", 1, 12, 1)
                day_of_week = st.slider("Día de la Semana (1=Lun, 7=Dom)", 1, 7, 4)
                sched_dep = st.number_input("Hora Salida (HHMM, ej: 1430)", min_value=0, max_value=2359, value=1430, step=5)
                sched_arr = st.number_input("Hora Llegada (HHMM, ej: 1700)", min_value=0, max_value=2359, value=1700, step=5)

            with col3:
                st.subheader("Detalles del Vuelo")
                sched_time = st.number_input("Tiempo Programado (minutos)", min_value=30.0, max_value=1000.0, value=210.0, step=10.0)
                distance = st.number_input("Distancia (millas)", min_value=50.0, max_value=5000.0, value=2475.0, step=50.0)
                st.info("Nota: DEPARTURE_DELAY (retraso en salida) no se usa en este modelo de planificación.")


            if st.button("Predecir Retraso", type="primary", use_container_width=True):
                # 1. Crear DataFrame de entrada
                input_data = {
                    "MONTH": month,
                    "DAY_OF_WEEK": day_of_week,
                    "AIRLINE": airline,
                    "ORIGIN_AIRPORT": origin,
                    "DESTINATION_AIRPORT": dest,
                    "SCHEDULED_DEPARTURE": sched_dep,
                    "SCHEDULED_ARRIVAL": sched_arr,
                    "SCHEDULED_TIME": sched_time,
                    "DISTANCE": distance,
                    # DEPARTURE_DELAY no se incluye
                }
                df_input = pd.DataFrame([input_data])
                
                try:
                    # 2. Preprocesar los datos
                    X_processed = preprocess_data_for_api(df_input, artifacts)
                    
                    # 3. Realizar predicción de probabilidad
                    probabilidad = model.predict_proba(X_processed)[0]
                    prob_retraso = float(probabilidad[1]) # Probabilidad de la clase 1 (Retrasado)
                    
                    st.divider()
                    st.subheader("Resultado de la Predicción")
                    
                    col_res1, col_res2 = st.columns(2)
                    
                    with col_res1:
                        # (Este es el AUC del modelo de planificación, ~0.62)
                        MODEL_AUC = 0.621 
                        # (Este es el umbral F1 del Exp 2, LGBM, de tus logs)
                        UMBRAL_OPTIMO = 0.708 
                        
                        st.metric("Probabilidad de Retraso (>15 min)", f"{prob_retraso*100:.1f}%")
                        st.progress(prob_retraso)
                        
                        if prob_retraso > UMBRAL_OPTIMO:
                            st.error(f"⚠️ **Retraso Probable.** (Probabilidad > {UMBRAL_OPTIMO*100:.0f}% umbral F1)")
                        else:
                            st.success(f"✅ **A Tiempo.** (Probabilidad < {UMBRAL_OPTIMO*100:.0f}% umbral F1)")
                    
                    with col_res2:
                        st.info(f"**¿Cómo leer esto?**\n"
                                f"El modelo (AUC {MODEL_AUC:.3f}) estima una probabilidad de **{prob_retraso*100:.1f}%** de que este vuelo llegue con más de 15 minutos de retraso, basándose únicamente en su horario programado.")

                except Exception as e:
                    st.error(f"Error durante la predicción: {e}")
                    st.exception(e)

    # ==========================
    # Pestaña 2: Análisis de Datos Históricos
    # ==========================
    with tab2:
        st.header("Análisis de Datos Históricos")
        st.markdown("Estos gráficos se basan en los datos históricos (de tu `visualizacion_vuelos.ipynb`) y ayudan a entender los patrones de retraso.")
        
        # --- Asignación de Imágenes ---
        # DEBES CREAR UNA CARPETA 'images' y guardar tus JPEGs allí
        PROJECT_ROOT_IMG = Path(__file__).resolve().parent
        img_folder = PROJECT_ROOT_IMG / "images" 
        
        # Nombres de archivo esperados (basados en tus JPEGs)
        img_files = {
            "pie_retrasos": img_folder / "WhatsApp Image 2025-11-16 at 00.47.42.jpeg",
            "bar_aerolinea": img_folder / "WhatsApp Image 2025-11-16 at 00.47.43.jpeg",
            "bar_mes": img_folder / "WhatsApp Image 2025-11-16 at 00.47.43 (1).jpeg",
            "bar_dia": img_folder / "WhatsApp Image 2025-11-16 at 00.47.43 (2).jpeg",
            "bar_periodo": img_folder / "WhatsApp Image 2025-11-16 at 00.47.43 (3).jpeg",
            "bar_origen": img_folder / "WhatsApp Image 2025-11-16 at 00.47.44.jpeg",
            "bar_destino": img_folder / "WhatsApp Image 2025-11-16 at 00.47.44 (1).jpeg",
            "bar_ruta": img_folder / "WhatsApp Image 2025-11-16 at 00.47.45 (1).jpeg"
        }

        # --- Fila 1: KPI y Gráficos Principales ---
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            st.metric("Tasa General de Retraso", "18.5%") # De tu gráfico pie
            if img_files["pie_retrasos"].exists():
                st.image(str(img_files["pie_retrasos"]), caption="Proporción de Retrasos")
            else:
                st.warning(f"No se encontró: {img_files['pie_retrasos'].name}")
        
        with col2:
            if img_files["bar_aerolinea"].exists():
                st.image(str(img_files["bar_aerolinea"]), caption="Tasa de Retraso por Aerolínea")
            else:
                st.warning(f"No se encontró: {img_files['bar_aerolinea'].name}")

        with col3:
            if img_files["bar_mes"].exists():
                st.image(str(img_files["bar_mes"]), caption="Retrasos por Mes")
            else:
                st.warning(f"No se encontró: {img_files['bar_mes'].name}")

        st.divider()

        # --- Fila 2: Gráficos Temporales y de Aeropuertos ---
        col4, col5, col6 = st.columns(3)
        with col4:
            if img_files["bar_dia"].exists():
                st.image(str(img_files["bar_dia"]), caption="Retrasos por Día de la Semana")
            else:
                st.warning(f"No se encontró: {img_files['bar_dia'].name}")
        with col5:
            if img_files["bar_periodo"].exists():
                st.image(str(img_files["bar_periodo"]), caption="Retrasos por Periodo del Día")
            else:
                st.warning(f"No se encontró: {img_files['bar_periodo'].name}")
        with col6:
            if img_files["bar_origen"].exists():
                st.image(str(img_files["bar_origen"]), caption="Top 10 Aeropuertos Origen (Retrasos)")
            else:
                st.warning(f"No se encontró: {img_files['bar_origen'].name}")
        
        st.divider()
        
        # --- Fila 3: Rutas ---
        col7, col8 = st.columns(2)
        with col7:
            if img_files["bar_destino"].exists():
                st.image(str(img_files["bar_destino"]), caption="Top 10 Aeropuertos Destino (Retrasos)")
            else:
                st.warning(f"No se encontró: {img_files['bar_destino'].name}")
        with col8:
            if img_files["bar_ruta"].exists():
                st.image(str(img_files["bar_ruta"]), caption="Top 10 Rutas (Retrasos)")
            else:
                st.warning(f"No se encontró: {img_files['bar_ruta'].name}")


    # ==========================
    # Pestaña 3: Métricas del Modelo
    # ==========================
    with tab3:
        st.header("Rendimiento del Modelo (Modelo de Planificación)")
        st.warning("Importante: Estos son los resultados del modelo de **planificación** (AUC ~0.62), "
                   "que NO conoce el retraso en la salida. Es menos preciso, pero es el correcto para "
                   "el escenario de 'comprar un billete'.")
        
        # --- KPIs ---
        # (Estos valores vienen de los resultados de tu notebook Exp 2, LGBM)
        st.subheader("KPIs de Validación (Meses 10-12)")
        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
        kpi_col1.metric("ROC-AUC", "0.6212")
        kpi_col2.metric("PR-AUC", "0.2421")
        kpi_col3.metric("Mejor F1-Score", "0.3316")
        kpi_col4.metric("Umbral F1 Óptimo", "0.708") # (El umbral donde se da el F1 de 0.3316)
        
        st.divider()

        # --- Gráficos de Evaluación ---
        st.subheader("Gráficos de Evaluación")
        st.info("Necesitas generar estos gráficos desde tu notebook de entrenamiento (el 'Modelo de Planificación') "
                "y guardarlos en la carpeta 'images' como 'planning_confusion_matrix.png' y 'planning_feature_importance.png'.")

        img_cm = PROJECT_ROOT_IMG / "images" / "planning_confusion_matrix.png"
        img_fi = PROJECT_ROOT_IMG / "images" / "planning_feature_importance.png"

        plot_col1, plot_col2 = st.columns(2)
        with plot_col1:
            if img_cm.exists():
                st.image(str(img_cm), caption="Matriz de Confusión (Modelo Planificación)")
            else:
                st.warning(f"No se encontró 'planning_confusion_matrix.png' en la carpeta /images.")
        
        with plot_col2:
            if img_fi.exists():
                st.image(str(img_fi), caption="Importancia de Features (Modelo Planificación)")
            else:
                st.warning(f"No se encontró 'planning_feature_importance.png' en la carpeta /images.")

# ==============================================================================
# 4. Ejecutar la Aplicación
# ==============================================================================

if __name__ == "__main__":
    # *** FIX v22: Cargar artifacts aquí ***
    artifacts = load_artifacts() 
    
    if artifacts is None:
        print("Error fatal: No se pudieron cargar los artefactos del modelo.")
        # Mostrar error en la app si falla al cargar
        st.error("Error fatal: No se pudieron cargar los artefactos del modelo. Revisa la consola/terminal.")
    else:
        # Pasar los artifacts a main()
        main(artifacts)
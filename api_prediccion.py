import os
from pathlib import Path
from datetime import datetime
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ============================================================
# CONFIGURACIÓN DE RUTAS (igual que en tu dashboard)
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "flight_delay_lgbm.joblib"
PREPROCESSOR_PATH = MODELS_DIR / "flight_delay_preprocessors.joblib"

PREDICTIONS_LOG = PROJECT_ROOT / "predictions_log_api.csv"

# ============================================================
# MODELOS Pydantic (entrada / salida)
# ============================================================

class FlightPredictRequest(BaseModel):
    """
    Parámetros mínimos que necesita tu modelo
    (coinciden con las columnas que construyes en df_input del dashboard).
    """
    month: int = Field(..., ge=1, le=12, description="Mes (1-12)")
    day_of_week: int = Field(..., ge=1, le=7, description="Día de la semana (1=Lunes, 7=Domingo)")
    airline: str = Field(..., description="Código de aerolínea, p.ej.: AA, DL, UA")
    origin_airport: str = Field(..., description="Código IATA/FAA del aeropuerto de origen")
    destination_airport: str = Field(..., description="Código IATA/FAA del aeropuerto destino")

    # Hora de salida en formato HHMM (ej: 1430 => 14:30)
    scheduled_departure: int = Field(..., ge=0, le=2359, description="Hora salida HHMM")

    # Hora de llegada en HHMM (si no se conoce, el llamador puede enviar 0
    # o una estimación calculada a partir de tu tabla histórica)
    scheduled_arrival: Optional[int] = Field(0, ge=0, le=2359, description="Hora llegada HHMM")

    # Tiempo programado de vuelo en minutos (duración estimada)
    scheduled_time: float = Field(..., gt=0, description="Duración programada en minutos")

    # Distancia en millas
    distance: float = Field(..., gt=0, description="Distancia de la ruta en millas")


class FlightPredictResponse(BaseModel):
    prob_delay: float               # Probabilidad de llegar con retraso > 15 min
    prob_on_time: float             # Probabilidad de NO retrasarse
    delayed: bool                   # Clasificación binaria con el umbral
    threshold_used: float           # Umbral que se usó para clasificar
    message: str                    # Mensaje amigable
    model_version: Optional[str] = None  # Opcional, si quieres agregar metadata


# ============================================================
# UTILIDADES: LOG DE PREDICCIONES
# ============================================================

def ensure_log():
    """Crea el CSV de log si no existe."""
    if not PREDICTIONS_LOG.exists():
        cols = [
            "timestamp_utc", "airline", "origin_code", "dest_code",
            "month", "day_of_week", "scheduled_dep", "scheduled_arr",
            "scheduled_time", "distance", "prob_delay"
        ]
        pd.DataFrame(columns=cols).to_csv(PREDICTIONS_LOG, index=False)


def append_log(record: dict):
    """Append de una predicción al CSV de log."""
    try:
        ensure_log()
        df_old = pd.read_csv(PREDICTIONS_LOG)
        df_new = pd.concat([df_old, pd.DataFrame([record])], ignore_index=True)
        df_new.to_csv(PREDICTIONS_LOG, index=False)
    except Exception as e:
        # Aquí no lanzamos excepción al cliente, solo dejamos constancia en servidor
        print(f"[WARN] No se pudo guardar el log de predicción: {e}")


# ============================================================
# CARGA DE ARTEFACTOS (modelo + preprocessors)
# ============================================================

artifacts = None   # se cargan al iniciar la app

def load_artifacts():
    """
    Carga modelo y preprocesadores desde /models.
    La estructura coincide con la que usas en el dashboard.
    """
    global artifacts

    preprocessors = None
    model = None

    if PREPROCESSOR_PATH.exists():
        preprocessors = joblib.load(PREPROCESSOR_PATH)
    else:
        raise RuntimeError(f"No se encontró {PREPROCESSOR_PATH.name} en /models/")

    if MODEL_PATH.exists():
        try:
            model = joblib.load(MODEL_PATH)
        except Exception as e:
            # Error típico de LightGBM/dll
            raise RuntimeError(
                f"No se pudo cargar el modelo: {e}. "
                "Reinstala la misma versión de lightgbm con la que entrenaste."
            )
    else:
        raise RuntimeError(f"No se encontró {MODEL_PATH.name} en /models/")

    if model is None or preprocessors is None:
        raise RuntimeError("Modelo o preprocesadores no cargados correctamente.")

    artifacts = {
        "model": model,
        "label_encoders": preprocessors.get("label_encoders", {}),
        "scaler": preprocessors.get("scaler", None),
        "cat_cols": preprocessors.get("cat_features_names", []),
        "num_cols": preprocessors.get("num_features_names", []),
    }
    artifacts["feature_order"] = artifacts["cat_cols"] + artifacts["num_cols"]


# ============================================================
# PREPROCESADO (COPIA DE DASHBOARD, ADAPTADO A API)
# ============================================================

def preprocess_data_for_api(df: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    """
    Replica la lógica que usas en dashboard_vuelos_app.py
    para construir las features antes de llamar al modelo.
    """
    df = df.copy()
    if artifacts is None:
        raise ValueError("Artifacts is None. Modelo no cargado.")

    label_encoders = artifacts["label_encoders"]
    scaler = artifacts["scaler"]
    CAT_COLS = artifacts["cat_cols"]
    NUM_COLS = artifacts["num_cols"]
    FINAL_FEATURE_ORDER = artifacts["feature_order"]

    # --- Feature RUTA ---
    df["RUTA"] = df["ORIGIN_AIRPORT"].astype(str) + "_" + df["DESTINATION_AIRPORT"].astype(str)

    # --- Features cíclicas de hora de salida ---
    sched_dep = pd.to_numeric(df["SCHEDULED_DEPARTURE"], errors="coerce").fillna(0).astype(int)
    hs = (sched_dep // 100).clip(0, 23)
    ms = (sched_dep % 100).clip(0, 59)
    minuto_dia_salida = hs * 60 + ms
    df["SALIDA_SIN"] = np.sin(2 * np.pi * minuto_dia_salida / (24 * 60))
    df["SALIDA_COS"] = np.cos(2 * np.pi * minuto_dia_salida / (24 * 60))

    # --- Features cíclicas de hora de llegada ---
    sched_arr = pd.to_numeric(df.get("SCHEDULED_ARRIVAL", 0), errors="coerce").fillna(0).astype(int)
    hl = (sched_arr // 100).clip(0, 23)
    ml = (sched_arr % 100).clip(0, 59)
    minuto_dia_llegada = hl * 60 + ml
    df["LLEGADA_SIN"] = np.sin(2 * np.pi * minuto_dia_llegada / (24 * 60))
    df["LLEGADA_COS"] = np.cos(2 * np.pi * minuto_dia_llegada / (24 * 60))

    # --- Mes cíclico ---
    df["MONTH_SIN"] = np.sin(2 * np.pi * df["MONTH"].astype(float) / 12)
    df["MONTH_COS"] = np.cos(2 * np.pi * df["MONTH"].astype(float) / 12)

    # --- Distancia haversine (si no existe) ---
    if "DISTANCE" in df.columns and "DISTANCIA_HAV" not in df.columns:
        df["DISTANCIA_HAV"] = df["DISTANCE"]

    # --- Label encoding con manejo de <unknown> ---
    for col in CAT_COLS:
        if col in label_encoders:
            le = label_encoders[col]
            classes_set = set([str(x) for x in le.classes_])
            df[col] = df[col].astype(str).apply(
                lambda x: x if x in classes_set else "<unknown>"
            )
            try:
                df[col] = le.transform(df[col])
            except Exception:
                mapping = {c: i for i, c in enumerate(le.classes_)}
                df[col] = df[col].apply(lambda x: mapping.get(x, mapping.get("<unknown>", 0)))
        else:
            if col not in df.columns:
                df[col] = 0

    # --- Scaling numérico ---
    cols_to_scale = [c for c in NUM_COLS if c in df.columns]
    if scaler is not None and len(cols_to_scale) > 0:
        try:
            df[cols_to_scale] = scaler.transform(df[cols_to_scale])
        except Exception:
            try:
                if hasattr(scaler, "mean_") and hasattr(scaler, "scale_"):
                    for i, c in enumerate(cols_to_scale):
                        mean_i = scaler.mean_[i] if i < len(scaler.mean_) else 0
                        scale_i = scaler.scale_[i] if i < len(scaler.scale_) else 1
                        df[c] = (df[c] - mean_i) / (scale_i + 1e-12)
            except Exception:
                pass

    # --- Asegurar todas las columnas finales ---
    for col in FINAL_FEATURE_ORDER:
        if col not in df.columns:
            df[col] = 0

    X = df[FINAL_FEATURE_ORDER].copy()
    return X


# ============================================================
# INSTANCIA DE FastAPI
# ============================================================

app = FastAPI(
    title="API Predicción Retrasos de Vuelos",
    description="API para predecir retrasos >15 minutos usando LightGBM y preprocesadores entrenados.",
    version="1.0.0",
)

# Cargar artefactos al iniciar el servidor
@app.on_event("startup")
def startup_event():
    print("Cargando artefactos del modelo...")
    load_artifacts()
    print("Artefactos cargados correctamente.")


# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/health")
def health_check():
    """Endpoint simple para verificar que la API está viva."""
    return {"status": "ok", "model_loaded": artifacts is not None}


@app.post("/flights/predict-delay", response_model=FlightPredictResponse)
def predict_delay(req: FlightPredictRequest):
    """
    Endpoint principal de predicción.

    Recibe los parámetros del vuelo y devuelve la probabilidad
    de retraso > 15 minutos.
    """
    if artifacts is None:
        raise HTTPException(
            status_code=500,
            detail="Artefactos del modelo no cargados. Revisa logs del servidor.",
        )

    # Construir DataFrame con las columnas que espera tu pipeline
    input_data = {
        "MONTH": int(req.month),
        "DAY_OF_WEEK": int(req.day_of_week),
        "AIRLINE": req.airline,
        "ORIGIN_AIRPORT": req.origin_airport,
        "DESTINATION_AIRPORT": req.destination_airport,
        "SCHEDULED_DEPARTURE": int(req.scheduled_departure),
        "SCHEDULED_ARRIVAL": int(req.scheduled_arrival or 0),
        "SCHEDULED_TIME": float(req.scheduled_time),
        "DISTANCE": float(req.distance),
    }
    df_input = pd.DataFrame([input_data])

    try:
        Xp = preprocess_data_for_api(df_input, artifacts)
        model = artifacts["model"]
        probs = model.predict_proba(Xp)[0]
        prob_on_time = float(probs[0])
        prob_delay = float(probs[1])

    except Exception as e:
        msg = str(e)
        if "Booster' object has no attribute 'handle" in msg or "lib_lightgbm" in msg:
            msg += " (Posible incompatibilidad de LightGBM/DLL. Reinstala la misma versión usada en el entrenamiento.)"
        raise HTTPException(status_code=500, detail=f"Error interno al predecir: {msg}")

    # Umbral que usas en tu dashboard de planificación
    THRESHOLD = 0.423
    delayed = prob_delay > THRESHOLD

    # Log de la predicción
    log_entry = {
        "timestamp_utc": datetime.utcnow().isoformat(),
        "airline": req.airline,
        "origin_code": req.origin_airport,
        "dest_code": req.destination_airport,
        "month": req.month,
        "day_of_week": req.day_of_week,
        "scheduled_dep": req.scheduled_departure,
        "scheduled_arr": req.scheduled_arrival or 0,
        "scheduled_time": req.scheduled_time,
        "distance": req.distance,
        "prob_delay": prob_delay,
    }
    append_log(log_entry)

    # Mensaje amigable
    if delayed:
        msg = "⚠️ Retraso probable (probabilidad por encima del umbral)."
    else:
        msg = "✅ Probablemente llegará a tiempo (por debajo del umbral)."

    return FlightPredictResponse(
        prob_delay=prob_delay,
        prob_on_time=prob_on_time,
        delayed=delayed,
        threshold_used=THRESHOLD,
        message=msg,
        model_version=None,
    )

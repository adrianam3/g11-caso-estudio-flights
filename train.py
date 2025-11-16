import os
import pandas as pd
from pathlib import Path
from scripts.model_preprocessing import load_data, preprocess_data
from scripts.model_training import train_model
from scripts.model_saving import save_model_and_preprocessors

# --- 1. Definir Rutas (Estructura de tu template) ---
PROJECT_ROOT = Path(os.path.abspath("")).resolve()
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "flights_clean.csv" 
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODELS_DIR / "flight_delay_lgbm.joblib"
PREPROCESSOR_PATH = MODELS_DIR / "flight_delay_preprocessors.joblib"

# Sobrescribir DATA_PATH si es necesario (ajusta tu ruta aquí)
# DATA_PATH = r"D:\OneDrive\...\flights_clean.csv"

def main():
    """
    Orquesta el pipeline completo de entrenamiento.
    """
    print("--- [1/4] Iniciando carga de datos ---")
    try:
        df = load_data(DATA_PATH)
        print(f"Datos cargados: {df.shape}")
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo en {DATA_PATH}")
        print("Por favor, ajusta la variable DATA_PATH en train.py")
        return

    print("\n--- [2/4] Iniciando preprocesamiento y split ---")
    # Preprocesa los datos y aplica el split temporal (1-9 vs 10-12)
    processed_data = preprocess_data(df)
    
    # Desempaquetar los resultados
    X_train = processed_data["X_train"]
    X_valid = processed_data["X_valid"]
    y_train = processed_data["y_train"]
    y_valid = processed_data["y_valid"]
    label_encoders = processed_data["label_encoders"]
    scaler = processed_data["scaler"]
    cat_features_names = processed_data["cat_features_names"]
    
    print(f"Split completado. Train: {X_train.shape}, Valid: {X_valid.shape}")

    print("\n--- [3/4] Iniciando entrenamiento del modelo (LGBM) ---")
    model = train_model(
        X_train, y_train, 
        X_valid, y_valid, 
        categorical_features=cat_features_names
    )
    print(f"Modelo entrenado: {type(model)}")

    print("\n--- [4/4] Guardando modelo y preprocesadores ---")
    # Empaquetar todos los artefactos de preprocesamiento
    preprocessors = {
        "label_encoders": label_encoders,
        "scaler": scaler,
        "cat_features_names": processed_data["cat_features_names"],
        "num_features_names": processed_data["num_features_names"]
    }
    
    # Guardar ambos artefactos
    save_model_and_preprocessors(
        model=model,
        preprocessors=preprocessors,
        model_path=MODEL_PATH,
        preprocessor_path=PREPROCESSOR_PATH
    )
    print("--- Pipeline completado exitosamente ---")

if __name__ == "__main__":
    main()
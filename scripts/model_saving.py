import joblib
from pathlib import Path

def save_model_and_preprocessors(model, preprocessors, model_path: Path, preprocessor_path: Path):
    """
    Guarda el modelo entrenado y el diccionario de preprocesadores 
    (LabelEncoders y StandardScaler) en archivos .joblib.
    """
    
    # Guardar el Modelo
    print(f"Guardando modelo en: {model_path}")
    joblib.dump(model, model_path)
    
    # Guardar los Preprocesadores (LabelEncoders y Scaler)
    print(f"Guardando preprocesadores en: {preprocessor_path}")
    joblib.dump(preprocessors, preprocessor_path)
    
    print("Artefactos guardados.")
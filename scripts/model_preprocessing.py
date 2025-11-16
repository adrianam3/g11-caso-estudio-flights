# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.model_selection import train_test_split

# def load_data(data_path: str) -> pd.DataFrame:
#     """
#     Carga el archivo flights_clean.csv desde la ruta especificada.
#     """
#     # Definir tipos de datos para las columnas que usaremos
#     dtype_map = {
#         "MONTH": "int8", "DAY_OF_WEEK": "int8",
#         "AIRLINE": "category", "ORIGIN_AIRPORT": "category", "DESTINATION_AIRPORT": "category",
#         "SCHEDULED_DEPARTURE": "int32", "SCHEDULED_ARRIVAL": "int32",
#         "SCHEDULED_TIME": "float32", "DISTANCE": "float32",
#         "DEPARTURE_DELAY": "float32",
#         "RETRASADO_LLEGADA": "int8"
#     }
#     # Solo cargar las columnas que realmente necesitamos
#     use_cols = list(dtype_map.keys())
    
#     df = pd.read_csv(data_path, usecols=use_cols, dtype=dtype_map, low_memory=False)
#     return df

# def _derive_features(df: pd.DataFrame) -> pd.DataFrame:
#     """Añade features cíclicas, de ruta y maneja NaNs."""
    
#     # 1. Manejar NaNs (solo en DEPARTURE_DELAY, que es la única feature numérica que lo permite)
#     df["DEPARTURE_DELAY"] = df["DEPARTURE_DELAY"].fillna(0)

#     # 2. Crear RUTA
#     df["RUTA"] = df["ORIGIN_AIRPORT"].astype(str) + "_" + df["DESTINATION_AIRPORT"].astype(str)
    
#     # 3. Features Cíclicas (Salida)
#     hs = (df["SCHEDULED_DEPARTURE"] // 100).clip(0, 23)
#     ms = (df["SCHEDULED_DEPARTURE"] % 100).clip(0, 59)
#     minuto_dia_salida = (hs * 60 + ms).astype("int16")
#     df["SALIDA_SIN"] = np.sin(2 * np.pi * minuto_dia_salida / (24*60)).astype("float32")
#     df["SALIDA_COS"] = np.cos(2 * np.pi * minuto_dia_salida / (24*60)).astype("float32")

#     # 4. Features Cíclicas (Llegada)
#     hl = (df["SCHEDULED_ARRIVAL"] // 100).clip(0, 23)
#     ml = (df["SCHEDULED_ARRIVAL"] % 100).clip(0, 59)
#     minuto_dia_llegada = (hl * 60 + ml).astype("int16")
#     df["LLEGADA_SIN"] = np.sin(2 * np.pi * minuto_dia_llegada / (24*60)).astype("float32")
#     df["LLEGADA_COS"] = np.cos(2 * np.pi * minuto_dia_llegada / (24*60)).astype("float32")
    
#     # 5. Features Cíclicas (Mes)
#     df["MONTH_SIN"] = np.sin(2 * np.pi * df["MONTH"] / 12).astype("float32")
#     df["MONTH_COS"] = np.cos(2 * np.pi * df["MONTH"] / 12).astype("float32")

#     # (Usamos DISTANCE del CSV, no recalculamos Haversine por eficiencia)
    
#     return df

# def _apply_label_encoding(X_train, X_valid, cat_cols):
#     """Aplica LabelEncoder y maneja categorías desconocidas."""
#     X_train_le = X_train.copy()
#     X_valid_le = X_valid.copy()
#     encoders = {}
    
#     print("Aplicando LabelEncoder a:", cat_cols)
#     for col in cat_cols: 
#         le = LabelEncoder()
#         X_train_le[col] = le.fit_transform(X_train_le[col].astype(str))
        
#         # Manejar categorías no vistas en validación
#         le_classes = set(le.classes_)
#         X_valid_le[col] = X_valid_le[col].astype(str).apply(lambda x: x if x in le_classes else '<unknown>')
#         if '<unknown>' not in le_classes:
#             le.classes_ = np.append(le.classes_, '<unknown>')
        
#         X_valid_le[col] = le.transform(X_valid_le[col])
#         encoders[col] = le
            
#     return X_train_le, X_valid_le, encoders

# def _apply_scaling(X_train, X_valid, num_cols):
#     """Aplica StandardScaler a las features numéricas."""
#     print("Aplicando StandardScaler a:", num_cols)
#     scaler = StandardScaler()
    
#     X_train_scaled = X_train.copy()
#     X_valid_scaled = X_valid.copy()
    
#     X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
#     X_valid_scaled[num_cols] = scaler.transform(X_valid[num_cols])
    
#     return X_train_scaled, X_valid_scaled, scaler


# def preprocess_data(df: pd.DataFrame) -> dict:
#     """
#     Pipeline completo de preprocesamiento:
#     1. Deriva features (Ruta, Cíclicas).
#     2. Define listas de features.
#     3. Aplica Split Temporal (Train 1-9, Valid 10-12).
#     4. Aplica LabelEncoder a categóricas.
#     5. Aplica StandardScaler a numéricas.
#     6. Retorna un diccionario con todos los artefactos.
#     """
    
#     # 1. Derivar Features
#     df_feat = _derive_features(df)
    
#     # 2. Definir listas de features
#     TARGET_COL = "RETRASADO_LLEGADA"
    
#     # Features Categóricas (para LabelEncoder)
#     CAT_COLS = ["AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "RUTA"]
    
#     # Features Numéricas (para StandardScaler)
#     NUM_COLS = [
#         "MONTH", "DAY_OF_WEEK", 
#         "SALIDA_SIN", "SALIDA_COS", 
#         "LLEGADA_SIN", "LLEGADA_COS",
#         "MONTH_SIN", "MONTH_COS",
#         "SCHEDULED_TIME", "DISTANCE",
#         "DEPARTURE_DELAY" # La feature clave
#     ]
    
#     FEATURES = CAT_COLS + NUM_COLS

#     # 3. Split Temporal (Train 1-9, Valid 10-12)
#     print("Realizando split temporal (Train 1-9, Valid 10-12)...")
#     train_mask = df_feat["MONTH"].between(1, 9)
#     valid_mask = df_feat["MONTH"].between(10, 12)
    
#     # Dividir X e y
#     X_train = df_feat.loc[train_mask, FEATURES].copy()
#     y_train = df_feat.loc[train_mask, TARGET_COL].astype("int8").copy()
#     X_valid = df_feat.loc[valid_mask, FEATURES].copy()
#     y_valid = df_feat.loc[valid_mask, TARGET_COL].astype("int8").copy()

#     # 4. Aplicar LabelEncoder a categóricas
#     X_train, X_valid, label_encoders = _apply_label_encoding(X_train, X_valid, CAT_COLS)

#     # 5. Aplicar StandardScaler a numéricas
#     X_train, X_valid, scaler = _apply_scaling(X_train, X_valid, NUM_COLS)
    
#     # 6. Retornar todo en un diccionario
#     return {
#         "X_train": X_train,
#         "X_valid": X_valid,
#         "y_train": y_train,
#         "y_valid": y_valid,
#         "label_encoders": label_encoders,
#         "scaler": scaler,
#         "cat_features_names": CAT_COLS, # Nombres de las columnas categóricas
#         "num_features_names": NUM_COLS
#     }

## nuevo

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def load_data(data_path: str) -> pd.DataFrame:
    """
    Carga el archivo flights_clean.csv (SIN DEPARTURE_DELAY).
    """
    # Definir tipos de datos
    dtype_map = {
        "MONTH": "int8", "DAY_OF_WEEK": "int8",
        "AIRLINE": "category", "ORIGIN_AIRPORT": "category", "DESTINATION_AIRPORT": "category",
        "SCHEDULED_DEPARTURE": "int32", "SCHEDULED_ARRIVAL": "int32",
        "SCHEDULED_TIME": "float32", "DISTANCE": "float32",
        # "DEPARTURE_DELAY": "float32", # <-- ELIMINADO
        "RETRASADO_LLEGADA": "int8"
    }
    # Solo cargar las columnas que realmente necesitamos
    use_cols = list(dtype_map.keys())
    
    df = pd.read_csv(data_path, usecols=use_cols, dtype=dtype_map, low_memory=False)
    return df

def _derive_features(df: pd.DataFrame) -> pd.DataFrame:
    """Añade features cíclicas, de ruta y maneja NaNs."""
    
    # 1. Manejar NaNs (DEPARTURE_DELAY ya no está)
    # df["DEPARTURE_DELAY"] = df["DEPARTURE_DELAY"].fillna(0) # <-- ELIMINADO

    # 2. Crear RUTA
    df["RUTA"] = df["ORIGIN_AIRPORT"].astype(str) + "_" + df["DESTINATION_AIRPORT"].astype(str)
    
    # 3. Features Cíclicas (Salida)
    hs = (df["SCHEDULED_DEPARTURE"] // 100).clip(0, 23)
    ms = (df["SCHEDULED_DEPARTURE"] % 100).clip(0, 59)
    minuto_dia_salida = (hs * 60 + ms).astype("int16")
    df["SALIDA_SIN"] = np.sin(2 * np.pi * minuto_dia_salida / (24*60)).astype("float32")
    df["SALIDA_COS"] = np.cos(2 * np.pi * minuto_dia_salida / (24*60)).astype("float32")

    # 4. Features Cíclicas (Llegada)
    hl = (df["SCHEDULED_ARRIVAL"] // 100).clip(0, 23)
    ml = (df["SCHEDULED_ARRIVAL"] % 100).clip(0, 59)
    minuto_dia_llegada = (hl * 60 + ml).astype("int16")
    df["LLEGADA_SIN"] = np.sin(2 * np.pi * minuto_dia_llegada / (24*60)).astype("float32")
    df["LLEGADA_COS"] = np.cos(2 * np.pi * minuto_dia_llegada / (24*60)).astype("float32")
    
    # 5. Features Cíclicas (Mes)
    df["MONTH_SIN"] = np.sin(2 * np.pi * df["MONTH"] / 12).astype("float32")
    df["MONTH_COS"] = np.cos(2 * np.pi * df["MONTH"] / 12).astype("float32")

    return df

def _apply_label_encoding(X_train, X_valid, cat_cols):
    """Aplica LabelEncoder y maneja categorías desconocidas."""
    X_train_le = X_train.copy()
    X_valid_le = X_valid.copy()
    encoders = {}
    
    print("Aplicando LabelEncoder a:", cat_cols)
    for col in cat_cols: 
        le = LabelEncoder()
        X_train_le[col] = le.fit_transform(X_train_le[col].astype(str))
        
        le_classes = set(le.classes_)
        X_valid_le[col] = X_valid_le[col].astype(str).apply(lambda x: x if x in le_classes else '<unknown>')
        if '<unknown>' not in le_classes:
            le.classes_ = np.append(le.classes_, '<unknown>')
        
        X_valid_le[col] = le.transform(X_valid_le[col])
        encoders[col] = le
            
    return X_train_le, X_valid_le, encoders

def _apply_scaling(X_train, X_valid, num_cols):
    """Aplica StandardScaler a las features numéricas."""
    print("Aplicando StandardScaler a:", num_cols)
    scaler = StandardScaler()
    
    X_train_scaled = X_train.copy()
    X_valid_scaled = X_valid.copy()
    
    X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_valid_scaled[num_cols] = scaler.transform(X_valid[num_cols])
    
    return X_train_scaled, X_valid_scaled, scaler


def preprocess_data(df: pd.DataFrame) -> dict:
    """
    Pipeline completo de preprocesamiento (SIN DEPARTURE_DELAY).
    """
    
    # 1. Derivar Features
    df_feat = _derive_features(df)
    
    # 2. Definir listas de features
    TARGET_COL = "RETRASADO_LLEGADA"
    
    CAT_COLS = ["AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "RUTA"]
    
    NUM_COLS = [
        "MONTH", "DAY_OF_WEEK", 
        "SALIDA_SIN", "SALIDA_COS", 
        "LLEGADA_SIN", "LLEGADA_COS",
        "MONTH_SIN", "MONTH_COS",
        "SCHEDULED_TIME", "DISTANCE",
        # "DEPARTURE_DELAY" # <-- ELIMINADO
    ]
    
    FEATURES = CAT_COLS + NUM_COLS

    # 3. Split Temporal (Train 1-9, Valid 10-12)
    print("Realizando split temporal (Train 1-9, Valid 10-12)...")
    train_mask = df_feat["MONTH"].between(1, 9)
    valid_mask = df_feat["MONTH"].between(10, 12)
    
    X_train = df_feat.loc[train_mask, FEATURES].copy()
    y_train = df_feat.loc[train_mask, TARGET_COL].astype("int8").copy()
    X_valid = df_feat.loc[valid_mask, FEATURES].copy()
    y_valid = df_feat.loc[valid_mask, TARGET_COL].astype("int8").copy()

    # 4. Aplicar LabelEncoder a categóricas
    X_train, X_valid, label_encoders = _apply_label_encoding(X_train, X_valid, CAT_COLS)

    # 5. Aplicar StandardScaler a numéricas
    X_train, X_valid, scaler = _apply_scaling(X_train, X_valid, NUM_COLS)
    
    # 6. Retornar todo en un diccionario
    return {
        "X_train": X_train,
        "X_valid": X_valid,
        "y_train": y_train,
        "y_valid": y_valid,
        "label_encoders": label_encoders,
        "scaler": scaler,
        "cat_features_names": CAT_COLS, 
        "num_features_names": NUM_COLS
    }
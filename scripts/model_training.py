# import lightgbm as lgb
# from sklearn.metrics import roc_auc_score

# def train_model(X_train, y_train, X_valid, y_valid, categorical_features=None):
#     """
#     Entrena un modelo LightGBM Classifier usando los datos de train/valid.
#     """
    
#     # (Parámetros optimizados de Exp 1, usando class_weight para balancear)
#     params = {
#         'objective': 'binary',
#         'metric': 'auc',
#         'n_estimators': 1000, 
#         'learning_rate': 0.05,
#         'num_leaves': 127,
#         'class_weight': 'balanced', 
#         'n_jobs': -1,
#         'random_state': 42,
#         'colsample_bytree': 0.8,
#         'subsample': 0.8,
#         'min_child_samples': 200
#     }
    
#     model = lgb.LGBMClassifier(**params)
    
#     fit_params = {
#         "eval_set": [(X_valid, y_valid)],
#         "eval_metric": "auc",
#         "callbacks": [lgb.early_stopping(100), lgb.log_evaluation(200)]
#     }
    
#     # FIX: Forzar el Dtype 'category' para que LGBM los trate correctamente
#     if categorical_features:
#         fit_params["categorical_feature"] = categorical_features
#         for col in categorical_features:
#             X_train[col] = X_train[col].astype('category')
#             X_valid[col] = X_valid[col].astype('category')
            
#     print("Iniciando entrenamiento de LGBM...")
#     model.fit(X_train, y_train, **fit_params)
    
#     print(f"Entrenamiento finalizado. Best iteration: {model.best_iteration_}")
    
#     # Evaluar en validación
#     y_proba = model.predict_proba(X_valid)[:, 1]
#     auc_valid = roc_auc_score(y_valid, y_proba)
#     print(f"ROC-AUC en set de validación (Meses 10-12): {auc_valid:.4f}")
    
#     return model

import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import pandas as pd

def _print_parameters(params):
    """
    Imprime y explica los parámetros clave del modelo LGBM.
    """
    print("\n" + "="*60)
    print("Parámetros de LightGBM (LGBM) para el Entrenamiento")
    print("="*60)
    
    # Diccionario de explicaciones
    explanations = {
        'objective': "Define el problema. 'binary' = clasificación binaria (Retrasado/No Retrasado).",
        'metric': "La métrica a optimizar. 'auc' (Área Bajo la Curva ROC) es excelente para clasificación desbalanceada.",
        'n_estimators': f"El número MÁXIMO de árboles (iteraciones) que el modelo construirá ({params.get('n_estimators', 1000)}). 'early_stopping' lo frenará antes si no mejora.",
        'learning_rate': f"Controla la velocidad de aprendizaje ({params.get('learning_rate', 0.05)}). Valores más bajos suelen ser más precisos pero tardan más.",
        'num_leaves': f"El número máximo de hojas por árbol ({params.get('num_leaves', 127)}). Un valor alto permite capturar patrones complejos.",
        'class_weight': f"Maneja el desbalance de datos. '{params.get('class_weight', 'balanced')}' = asigna automáticamente más peso a la clase minoritaria (retrasos).",
        'n_jobs': "Cuántos núcleos de CPU usar. '-1' = usar todos los disponibles.",
        'random_state': f"Semilla para reproducibilidad ({params.get('random_state', 42)}). Asegura que si se corre de nuevo, dé el mismo resultado.",
        'colsample_bytree': f"Fracción de features (columnas) a usar por árbol ({params.get('colsample_bytree', 0.8)} = 80%). Ayuda a prevenir el sobreajuste.",
        'subsample': f"Fracción de datos (filas) a usar por árbol ({params.get('subsample', 0.8)} = 80%). También previene el sobreajuste (bagging).",
        'min_child_samples': f"Número mínimo de muestras requeridas en una hoja ({params.get('min_child_samples', 200)}). Previene que el modelo cree hojas para patrones muy específicos (ruido)."
    }
    
    # Imprimir solo los parámetros que definimos
    for key in [
        'objective', 'metric', 'n_estimators', 'learning_rate', 
        'num_leaves', 'class_weight', 'n_jobs', 'random_state', 
        'colsample_bytree', 'subsample', 'min_child_samples'
    ]:
        if key in params:
            print(f"  - {key}: {params[key]}")
            print(f"    └ {explanations[key]}")
    
    print("="*60 + "\n")

def train_model(X_train, y_train, X_valid, y_valid, categorical_features=None):
    """
    Entrena un modelo LightGBM Classifier usando los datos de train/valid.
    """
    
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'n_estimators': 1000, 
        'learning_rate': 0.05,
        'num_leaves': 127,
        'class_weight': 'balanced', 
        'n_jobs': -1,
        'random_state': 42,
        'colsample_bytree': 0.8,
        'subsample': 0.8,
        'min_child_samples': 200
    }
    
    _print_parameters(params)
    
    model = lgb.LGBMClassifier(**params)
    
    fit_params = {
        "eval_set": [(X_valid, y_valid)],
        "eval_metric": "auc",
        "callbacks": [lgb.early_stopping(100), lgb.log_evaluation(200)]
    }
    
    # *** FIX v21 (LGBM) ***
    # Si estamos usando LabelEncoder (que crea ints), solo pasamos la LISTA DE NOMBRES.
    # NO convertimos los DataFrames a dtype 'category'
    if categorical_features:
        print(f"Tratando como categóricas (LabelEncoded): {categorical_features}")
        fit_params["categorical_feature"] = categorical_features
        
    print("Iniciando entrenamiento de LGBM...")
    model.fit(X_train, y_train, **fit_params)
    
    print(f"Entrenamiento finalizado. Best iteration: {model.best_iteration_}")
    
    # Evaluar en validación
    y_proba = model.predict_proba(X_valid)[:, 1]
    auc_valid = roc_auc_score(y_valid, y_proba)
    print(f"ROC-AUC en set de validación (Meses 10-12): {auc_valid:.4f}")
    
    return model
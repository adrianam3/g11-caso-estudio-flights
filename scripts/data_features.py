import pandas as pd
import numpy as np

# =========================
# CONFIGURACIÓN DE CAUSAS
# =========================

CAUSAS_COLS = [
    "AIR_SYSTEM_DELAY",
    "SECURITY_DELAY",
    "AIRLINE_DELAY",
    "LATE_AIRCRAFT_DELAY",
    "WEATHER_DELAY"
]

MAP_CAUSAS_ES = {
    "AIR_SYSTEM_DELAY":      "Sistema aéreo (NAS)",
    "SECURITY_DELAY":        "Seguridad",
    "AIRLINE_DELAY":         "Aerolínea",
    "LATE_AIRCRAFT_DELAY":   "Aeronave llegada tardía",
    "WEATHER_DELAY":         "Clima"
}

PRIORIDAD_ES = [
    "Aerolínea",
    "Aeronave llegada tardía",
    "Clima",
    "Seguridad",
    "Sistema aéreo (NAS)",
]
PRIO_IDX = {nombre: i for i, nombre in enumerate(PRIORIDAD_ES)}


def motivo_retraso_concat_ordenado(row):
    """Determina el motivo principal del retraso, ordenado por prioridad."""
    arr = row["ARRIVAL_DELAY"]
    if pd.isna(arr):
        return "Desconocido"
    if arr <= 0:
        return "Sin retraso"

    presentes = [
        MAP_CAUSAS_ES[c]
        for c in CAUSAS_COLS
        if pd.notna(row.get(c)) and row.get(c) > 0
    ]

    if not presentes:
        return "Retraso sin causa reportada"

    presentes_orden = sorted(set(presentes), key=lambda n: PRIO_IDX.get(n, 999))
    return " - ".join(presentes_orden)


# =========================
# GENERACIÓN DE NUEVAS COLUMNAS
# =========================
def generar_nuevas_columnas(v):
    """Genera columnas derivadas útiles para análisis y predicción."""
    print("🧠 Generando nuevas columnas...")

    # --- Motivo del retraso ---
    v["MOTIVO_RETRASO"] = v.apply(motivo_retraso_concat_ordenado, axis=1)

    # --- Cantidad de causas ---
    v["CANTIDAD_CAUSAS"] = v[CAUSAS_COLS].gt(0).sum(axis=1).astype(int)
    v.loc[v["ARRIVAL_DELAY"] <= 0, "CANTIDAD_CAUSAS"] = 0

    # --- Variables binarias de retraso ---
    v["RETRASADO_LLEGADA"] = (v["ARRIVAL_DELAY"] > 15).astype(int)
    v["RETRASADO_SALIDA"] = (v["DEPARTURE_DELAY"] > 15).astype(int)

    # --- Hora programada ---
    v["HORA_SALIDA"] = (v["SCHEDULED_DEPARTURE"] // 100).clip(0, 23)
    v["HORA_LLEGADA"] = (v["SCHEDULED_ARRIVAL"] // 100).clip(0, 23)

    # --- Minuto programado ---
    v["MIN_SALIDA"] = (v["SCHEDULED_DEPARTURE"] % 100).clip(0, 59)
    v["MIN_LLEGADA"] = (v["SCHEDULED_ARRIVAL"] % 100).clip(0, 59)

    # --- Minuto del día ---
    v["MINUTO_DIA_SALIDA"] = v["HORA_SALIDA"] * 60 + v["MIN_SALIDA"]
    v["MINUTO_DIA_LLEGADA"] = v["HORA_LLEGADA"] * 60 + v["MIN_LLEGADA"]

    # --- Codificación cíclica ---
    v["SALIDA_SIN"] = np.sin(2 * np.pi * v["MINUTO_DIA_SALIDA"] / (24 * 60))
    v["SALIDA_COS"] = np.cos(2 * np.pi * v["MINUTO_DIA_SALIDA"] / (24 * 60))
    v["LLEGADA_SIN"] = np.sin(2 * np.pi * v["MINUTO_DIA_LLEGADA"] / (24 * 60))
    v["LLEGADA_COS"] = np.cos(2 * np.pi * v["MINUTO_DIA_LLEGADA"] / (24 * 60))

    # --- Período del día ---
    v["PERIODO_SALIDA"] = pd.cut(
        v["HORA_SALIDA"],
        bins=[0, 6, 12, 18, 24],
        labels=["Madrugada", "Mañana", "Tarde", "Noche"],
        right=False
    )
    v["PERIODO_LLEGADA"] = pd.cut(
        v["HORA_LLEGADA"],
        bins=[0, 6, 12, 18, 24],
        labels=["Madrugada", "Mañana", "Tarde", "Noche"],
        right=False
    )

    # --- Ruta (origen-destino) ---
    # v["RUTA"] = v["ORIGIN_AIRPORT"].astype(str) + "_" + v["DESTINATION_AIRPORT"].astype(str)

    print("✅ Nuevas columnas generadas correctamente.")
    return v


# def resumen_causas(v):
#     """Muestra resumen estadístico de las causas de retraso."""
#     mask_pos = v["ARRIVAL_DELAY"] > 0
#     con_retraso = mask_pos.sum()
#     cero = (mask_pos & (v["CANTIDAD_CAUSAS"] == 0)).sum()
#     una = (mask_pos & (v["CANTIDAD_CAUSAS"] == 1)).sum()
#     multi = (mask_pos & (v["CANTIDAD_CAUSAS"] >= 2)).sum()

#     print("\n📊 Resumen de causas (solo ARRIVAL_DELAY>0):")
#     print(f"  Vuelos con retraso:          {con_retraso:,}")
#     print(f"  0 causas reportadas:         {cero:,}   ({cero/con_retraso*100:.2f}%)")
#     print(f"  1 causa reportada:           {una:,}    ({una/con_retraso*100:.2f}%)")
#     print(f"  ≥2 causas (multifactorial):  {multi:,}  ({multi/con_retraso*100:.2f}%)")

#     dist_causas = (
#         v.loc[mask_pos, "CANTIDAD_CAUSAS"]
#         .value_counts().sort_index().to_frame("conteo")
#     )
#     dist_causas["porcentaje"] = (dist_causas["conteo"] / con_retraso * 100).round(2)
#     return dist_causas


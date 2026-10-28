"""
Ingeniería de características: KPI deterministas y variables para el pronóstico (sin fuga temporal).
Las métricas financieras se calculan SIEMPRE aquí, en Python, y nunca por un modelo.
"""
from __future__ import annotations

import unicodedata

import numpy as np
import pandas as pd

from src.rules.reglas_negocio import estado_por_regla

BASE_FEATURES = ["gasto_usd", "impresiones", "clics", "conversiones", "ingresos_usd",
                 "ctr", "roas", "cpa_usd"]
NUEVAS_FEATURES = ["cvr", "profit_usd", "cpm_usd"]
FEATURES = BASE_FEATURES + NUEVAS_FEATURES
ORDEN_CLASES = ["Crítico", "Alerta", "Óptimo"]


def _div(a, b, relleno=0.0):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    out = np.full_like(a, relleno, dtype=float)
    np.divide(a, b, out=out, where=b > 0)
    return out


def calcular_kpis(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ctr"] = _div(df["clics"], df["impresiones"]) * 100            # %
    df["cvr"] = _div(df["conversiones"], df["clics"]) * 100           # %
    df["roas"] = _div(df["ingresos_usd"], df["gasto_usd"])
    df["cpa_usd"] = np.where(df["conversiones"] > 0, _div(df["gasto_usd"], df["conversiones"]),
                             df["gasto_usd"])
    df["profit_usd"] = df["ingresos_usd"] - df["gasto_usd"]
    df["cpm_usd"] = _div(df["gasto_usd"], df["impresiones"]) * 1000
    return df


def _normalizar_etiqueta(x) -> str:
    t = unicodedata.normalize("NFKD", str(x)).encode("ascii", "ignore").decode().strip().lower()
    mapa = {"critico": "Crítico", "alerta": "Alerta", "optimo": "Óptimo"}
    return mapa.get(t, str(x))


def preparar_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """KPI + estado por regla de negocio. Si el dataset trae una etiqueta, se mide su concordancia."""
    df = calcular_kpis(df)
    origen = "regla de negocio (ROAS < 1,8 Crítico; ≥ 2,8 Óptimo)"
    if "estado_rendimiento" in df.columns and df["estado_rendimiento"].notna().mean() > 0.95:
        df["estado_etiqueta_original"] = df["estado_rendimiento"].map(_normalizar_etiqueta)
    df["estado_rendimiento"] = estado_por_regla(df["roas"])
    if "estado_etiqueta_original" in df.columns:
        conc = (df["estado_etiqueta_original"] == df["estado_rendimiento"]).mean()
        origen += f"; concordancia con la etiqueta original: {conc:.1%}"
    df = df.sort_values(["cliente", "fecha"]).reset_index(drop=True)
    return df, origen


def segmento_por_escala(df: pd.DataFrame) -> pd.Series:
    """Segmento presupuestario por terciles de gasto medio del cliente (auditoría de sesgo)."""
    gasto_medio = df.groupby("cliente")["gasto_usd"].transform("mean")
    try:
        return pd.qcut(gasto_medio.rank(method="dense"), 3, labels=["Pyme", "Mediana", "Grande"])
    except ValueError:
        return pd.Series(np.where(gasto_medio >= gasto_medio.median(), "Grande", "Pyme"), index=df.index)


def comparativo_historico(sub: pd.DataFrame, pos: int) -> dict:
    """Comparación determinista del periodo `pos` con el pasado de la misma cuenta (sub ordenado por fecha)."""
    act = sub.iloc[pos]

    def var(a, b):
        return None if b is None or b == 0 else 100 * (a - b) / abs(b)
    prev = sub.iloc[pos - 1] if pos >= 1 else None
    hace12 = sub.iloc[pos - 12] if pos >= 12 else None
    ult12 = sub.iloc[max(0, pos - 11):pos + 1]
    return {
        "roas_mes_anterior": None if prev is None else float(prev["roas"]),
        "var_roas": None if prev is None else var(act["roas"], prev["roas"]),
        "roas_hace_12m": None if hace12 is None else float(hace12["roas"]),
        "var_roas_anual": None if hace12 is None else var(act["roas"], hace12["roas"]),
        "roas_promedio_12m": float(ult12["ingresos_usd"].sum() / ult12["gasto_usd"].sum()),
        "inversion_12m": float(ult12["gasto_usd"].sum()),
        "ingresos_12m": float(ult12["ingresos_usd"].sum()),
        "cpa_mes_anterior": None if prev is None else float(prev["cpa_usd"]),
        "var_cpa": None if prev is None else var(act["cpa_usd"], prev["cpa_usd"]),
        "var_inversion": None if prev is None else var(act["gasto_usd"], prev["gasto_usd"]),
        "meses_historia": int(pos + 1),
    }

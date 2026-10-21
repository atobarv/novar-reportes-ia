"""
Ingeniería de características — KPIs de negocio deterministas y etiquetado.
Las métricas financieras (ROAS, CPA, CTR, CVR, CPM, beneficio) se calculan SIEMPRE aquí,
en Python, y nunca por el LLM (guardrail aritmético).
"""
from __future__ import annotations

import unicodedata

import numpy as np
import pandas as pd

BASE_FEATURES = ["gasto_usd", "impresiones", "clics", "conversiones", "ingresos_usd",
                 "ctr", "roas", "cpa_usd"]
NUEVAS_FEATURES = ["cvr", "profit_usd", "cpm_usd"]          # 3 variables de negocio derivadas
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
    # Sin conversiones, el CPA se iguala al gasto total (peor caso conservador)
    df["cpa_usd"] = np.where(df["conversiones"] > 0, _div(df["gasto_usd"], df["conversiones"]),
                             df["gasto_usd"])
    df["profit_usd"] = df["ingresos_usd"] - df["gasto_usd"]
    df["cpm_usd"] = _div(df["gasto_usd"], df["impresiones"]) * 1000
    return df


def etiquetar_por_regla(df: pd.DataFrame, roas_critico: float = 1.2, roas_optimo: float = 2.8,
                        profit_min: float = 50.0) -> pd.Series:
    """Regla experta usada solo cuando el dataset no trae etiqueta."""
    estado = np.where((df["roas"] < roas_critico) | (df["conversiones"] == 0), "Crítico",
                      np.where((df["roas"] >= roas_optimo) & (df["profit_usd"] >= profit_min),
                               "Óptimo", "Alerta"))
    return pd.Series(estado, index=df.index, name="estado_rendimiento")


def _normalizar_etiqueta(x) -> str:
    t = unicodedata.normalize("NFKD", str(x)).encode("ascii", "ignore").decode().strip().lower()
    mapa = {"critico": "Crítico", "bajo": "Crítico", "alerta": "Alerta", "medio": "Alerta",
            "optimo": "Óptimo", "alto": "Óptimo", "0": "Crítico", "1": "Alerta", "2": "Óptimo"}
    return mapa.get(t, str(x))


def preparar_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Devuelve el dataset con KPIs y la columna objetivo, e indica el origen de la etiqueta."""
    df = calcular_kpis(df)
    if "estado_rendimiento" in df.columns and df["estado_rendimiento"].notna().mean() > 0.95:
        df["estado_rendimiento"] = df["estado_rendimiento"].map(_normalizar_etiqueta)
        origen = "etiqueta del dataset (criterio experto)"
    else:
        df["estado_rendimiento"] = etiquetar_por_regla(df)
        origen = "regla experta (ROAS / beneficio / conversiones)"
    return df, origen


def segmento_por_escala(df: pd.DataFrame) -> pd.Series:
    """Segmento presupuestario por terciles de gasto medio del cliente (auditoría de sesgo)."""
    gasto_medio = df.groupby("cliente")["gasto_usd"].transform("mean")
    try:
        return pd.qcut(gasto_medio.rank(method="dense"), 3, labels=["Pyme", "Mediana", "Grande"])
    except ValueError:  # muy pocos clientes para terciles
        return pd.Series(np.where(gasto_medio >= gasto_medio.median(), "Grande", "Pyme"),
                         index=df.index)

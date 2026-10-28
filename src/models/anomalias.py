"""
Detección de patrones anómalos (aprendizaje no supervisado).

Cada mes de cada cuenta se describe con desviaciones robustas (puntaje z con mediana y MAD) de seis
métricas respecto de los meses ANTERIORES de la misma cuenta (sin fuga temporal). Un Isolation Forest
global aprende cómo se ven los meses normales y marca las combinaciones inusuales, por ejemplo: CTR muy
alto con conversión muy baja (tráfico inválido) o inversión muy alta sin aumento de ingresos (gasto duplicado).
La referencia determinista es una regla de umbral sobre el ROAS (|z| > 3).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold

METRICAS = ["roas", "cpa_usd", "ctr", "cvr", "cpm_usd", "gasto_usd"]
NOMBRES = {"roas": "ROAS", "cpa_usd": "CPA", "ctr": "CTR", "cvr": "tasa de conversión",
           "cpm_usd": "CPM", "gasto_usd": "inversión"}
Z_COLS = [f"z_{m}" for m in METRICAS]


def caracteristicas(df: pd.DataFrame, ventana: int = 12, min_hist: int = 12) -> pd.DataFrame:
    """Puntajes z robustos de cada mes frente a los `ventana` meses previos de la cuenta."""
    salidas = []
    for _, g in df.sort_values(["cliente", "fecha"]).groupby("cliente", sort=False):
        g = g.copy()
        for m in METRICAS:
            v = np.log(g[m].astype(float).clip(lower=1e-3))
            previo = v.shift(1).rolling(ventana, min_periods=min_hist)
            med = previo.median()
            mad = previo.apply(lambda w: np.median(np.abs(w - np.median(w))), raw=True)
            g[f"z_{m}"] = (v - med) / (1.4826 * mad.clip(lower=0.03))
        salidas.append(g)
    out = pd.concat(salidas)
    out["evaluable"] = out[Z_COLS].notna().all(axis=1)
    return out


class DetectorAnomalias:
    def __init__(self, tasa: float = 0.06, seed: int = 42):
        self.tasa, self.seed = tasa, seed

    def ajustar(self, F: pd.DataFrame):
        X = F.loc[F["evaluable"], Z_COLS].clip(-15, 15)
        self.modelo = IsolationForest(n_estimators=400, contamination="auto", random_state=self.seed).fit(X)
        puntajes = -self.modelo.score_samples(X)
        self.umbral = float(np.quantile(puntajes, 1 - self.tasa))
        return self

    def puntuar(self, F: pd.DataFrame) -> pd.DataFrame:
        F = F.copy()
        F["puntaje_anomalia"] = np.nan
        ok = F["evaluable"]
        F.loc[ok, "puntaje_anomalia"] = -self.modelo.score_samples(F.loc[ok, Z_COLS].clip(-15, 15))
        F["es_anomalia"] = F["puntaje_anomalia"] > self.umbral
        F["metrica_principal"] = F[Z_COLS].abs().fillna(0).idxmax(axis=1).str[2:].map(NOMBRES).where(ok)
        return F


def regla_umbral(F: pd.DataFrame, z: float = 3.0) -> pd.Series:
    """Referencia determinista: el ROAS se aleja más de `z` desviaciones robustas de su historia."""
    return F["evaluable"] & (F["z_roas"].abs() > z)


def evaluar(F: pd.DataFrame, verdad: pd.Series, tasa: float = 0.06, seed: int = 42, corte=None) -> pd.DataFrame:
    """Precisión, recall, F1 y AP del detector frente a la verdad de referencia.
    Esquemas: por cliente (GroupKFold) y temporal (ajuste con meses ≤ corte, evaluación posterior)."""
    F = F.copy()
    F["verdad"] = verdad.values
    E = F[F["evaluable"]]
    filas = []

    def registrar(nombre, esquema, y, pred, puntaje=None):
        tp = int((pred & y).sum()); fp = int((pred & ~y).sum()); fn = int((~pred & y).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        filas.append({"metodo": nombre, "esquema": esquema, "meses_evaluados": int(len(y)),
                      "anomalias_reales": int(y.sum()), "marcadas": int(pred.sum()), "precision": p,
                      "recall": r, "F1": 2 * p * r / (p + r) if p + r else 0.0,
                      "AP": float(average_precision_score(y, puntaje)) if puntaje is not None and y.any() else np.nan})

    # por cliente
    pred_if = pd.Series(False, index=E.index); punt = pd.Series(np.nan, index=E.index)
    for tr, te in GroupKFold(n_splits=7).split(E, groups=E["cliente"]):
        d = DetectorAnomalias(tasa, seed).ajustar(E.iloc[tr])
        s = d.puntuar(E.iloc[te])
        pred_if.iloc[te] = s["es_anomalia"].values; punt.iloc[te] = s["puntaje_anomalia"].values
    registrar("Isolation Forest (ML)", "por cliente", E["verdad"].values, pred_if.values, punt.values)
    registrar("Regla |z ROAS| > 3", "por cliente", E["verdad"].values, regla_umbral(E).values, E["z_roas"].abs().values)
    # temporal
    if corte is not None:
        fe = pd.to_datetime(E["fecha"])
        tr, te = E[fe <= corte], E[fe > corte]
        s = DetectorAnomalias(tasa, seed).ajustar(tr).puntuar(te)
        registrar("Isolation Forest (ML)", "temporal", te["verdad"].values, s["es_anomalia"].values,
                  s["puntaje_anomalia"].values)
        registrar("Regla |z ROAS| > 3", "temporal", te["verdad"].values, regla_umbral(te).values,
                  te["z_roas"].abs().values)
    return pd.DataFrame(filas)


def imputar_anomalias(df: pd.DataFrame, marcas: pd.Series, columnas=("roas", "gasto_usd", "ingresos_usd")) -> pd.DataFrame:
    """Reemplaza los meses marcados por la mediana de los 6 meses previos de la cuenta, para que una falla de
    medición no distorsione el pronóstico. Los datos originales se conservan para el reporte."""
    out = df.sort_values(["cliente", "fecha"]).copy()
    m = marcas.reindex(out.index).fillna(False).astype(bool)
    for col in columnas:
        if col not in out:
            continue
        med = out.groupby("cliente")[col].transform(lambda v: v.shift(1).rolling(6, min_periods=1).median())
        out.loc[m, col] = med[m]
    return out

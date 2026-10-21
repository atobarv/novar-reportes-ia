"""
Pruebas de robustez del clasificador.
Simulan dos fuentes de error reales de la operación de una agencia:
  * Ruido de etiqueta: inconsistencia del criterio humano al etiquetar el estado de una cuenta.
  * Ruido de atribución: discrepancia entre plataformas (p. ej., Meta Ads vs GA4) en el gasto y los
    ingresos atribuidos; se aplica en evaluación y se recalculan todos los KPIs derivados.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import export_text

from src.features.feature_engineering import calcular_kpis


def _ruido_atribucion(df: pd.DataFrame, sigma: float, rng) -> pd.DataFrame:
    d = df.copy()
    d["ingresos_usd"] = d["ingresos_usd"] * rng.lognormal(0, sigma, len(d))
    d["gasto_usd"] = d["gasto_usd"] * rng.lognormal(0, sigma, len(d))
    return calcular_kpis(d)


def evaluar_robustez(modelos: dict, df: pd.DataFrame, y: pd.Series,
                     ruidos_etiqueta=(0, 0.05, 0.10, 0.15, 0.20),
                     ruidos_atribucion=(0, 0.05, 0.10, 0.20), repeticiones: int = 5,
                     seed: int = 42) -> pd.DataFrame:
    y = np.asarray(y)
    clases = np.unique(y)
    filas = []
    for nombre, (pipe, feats) in modelos.items():
        for i_e, pe in enumerate(ruidos_etiqueta):
            for i_a, sa in enumerate(ruidos_atribucion):
                accs, f1s = [], []
                for rep in range(repeticiones):
                    # Números aleatorios comunes: todos los modelos ven exactamente el mismo ruido
                    rng = np.random.default_rng([seed, i_e, i_a, rep])
                    cv = StratifiedKFold(5, shuffle=True, random_state=seed + rep)
                    for tr, te in cv.split(df, y):
                        ytr = y[tr].copy()
                        m = rng.random(len(tr)) < pe
                        ytr[m] = [rng.choice(clases[clases != v]) for v in ytr[m]]
                        mdl = clone(pipe).fit(df.iloc[tr][feats], ytr)
                        X_te = (_ruido_atribucion(df.iloc[te], sa, rng) if sa else df.iloc[te])[feats]
                        pred = mdl.predict(X_te)
                        accs.append(accuracy_score(y[te], pred))
                        f1s.append(f1_score(y[te], pred, average="weighted"))
                filas.append({"modelo": nombre, "ruido_etiqueta": pe, "ruido_atribucion": sa,
                              "accuracy": float(np.mean(accs)), "accuracy_std": float(np.std(accs)),
                              "f1_weighted": float(np.mean(f1s))})
    return pd.DataFrame(filas)


def resumen_robustez(tabla: pd.DataFrame) -> pd.DataFrame:
    """Índice de robustez = accuracy media en todos los escenarios con algún tipo de ruido."""
    limpio = tabla[(tabla["ruido_etiqueta"] == 0) & (tabla["ruido_atribucion"] == 0)]
    ruidoso = tabla[(tabla["ruido_etiqueta"] > 0) | (tabla["ruido_atribucion"] > 0)]
    severo = tabla[(tabla["ruido_etiqueta"] >= 0.10) & (tabla["ruido_atribucion"] >= 0.05)]
    r = pd.DataFrame({
        "accuracy_datos_limpios": limpio.set_index("modelo")["accuracy"],
        "indice_robustez": ruidoso.groupby("modelo")["accuracy"].mean(),
        "accuracy_ruido_severo": severo.groupby("modelo")["accuracy"].mean(),
        "peor_caso": tabla.groupby("modelo")["accuracy"].min(),
    })
    r["caida_vs_limpio"] = r["accuracy_datos_limpios"] - r["indice_robustez"]
    return r.sort_values("indice_robustez", ascending=False)


def reglas_arbol(pipeline, features: list[str]) -> str:
    """Reglas del árbol de decisión en texto (evidencia de interpretabilidad)."""
    return export_text(pipeline.named_steps["clf"], feature_names=list(features), decimals=2)

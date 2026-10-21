"""
Interpretabilidad: TreeSHAP (global y local) y LIME (local, independiente del modelo).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _partes(pipeline):
    scaler = pipeline.named_steps.get("scaler")
    clf = pipeline.named_steps["clf"]
    return scaler, clf


def _transformar(pipeline, X: pd.DataFrame) -> pd.DataFrame:
    scaler, _ = _partes(pipeline)
    Xt = scaler.transform(X) if scaler is not None else X.values
    return pd.DataFrame(Xt, columns=X.columns, index=X.index)


def shap_valores(pipeline, X: pd.DataFrame):
    """Devuelve (array n x f x c, explainer, X_transformado)."""
    import shap
    _, clf = _partes(pipeline)
    Xt = _transformar(pipeline, X)
    explainer = shap.TreeExplainer(clf)
    sv = explainer.shap_values(Xt)
    if isinstance(sv, list):
        sv = np.stack(sv, axis=-1)
    sv = np.asarray(sv)
    if sv.ndim == 2:
        sv = sv[..., None]
    return sv, explainer, Xt


def importancia_global(sv: np.ndarray, features: list[str]) -> pd.DataFrame:
    media_abs = np.abs(sv).mean(axis=(0, 2))
    tabla = pd.DataFrame({"feature": features, "media_abs_shap": media_abs})
    tabla["peso_pct"] = 100 * tabla["media_abs_shap"] / tabla["media_abs_shap"].sum()
    return tabla.sort_values("peso_pct", ascending=False).reset_index(drop=True)


def contribuciones_locales(pipeline, fila: pd.DataFrame, clases: list[str]) -> tuple[str, pd.DataFrame]:
    """Contribuciones SHAP de una fila hacia la clase predicha."""
    sv, _, _ = shap_valores(pipeline, fila)
    pred = pipeline.predict(fila)[0]
    idx = list(pipeline.classes_).index(pred)
    contrib = pd.DataFrame({"feature": fila.columns, "valor": fila.iloc[0].values,
                            "shap": sv[0, :, idx]})
    contrib["abs"] = contrib["shap"].abs()
    return pred, contrib.sort_values("abs", ascending=False).drop(columns="abs").reset_index(drop=True)


def explicar_lime(pipeline, X_train: pd.DataFrame, fila: pd.DataFrame, num_features: int = 6,
                  seed: int = 42):
    from lime.lime_tabular import LimeTabularExplainer
    explainer = LimeTabularExplainer(
        X_train.values, feature_names=list(X_train.columns),
        class_names=list(pipeline.classes_), mode="classification",
        discretize_continuous=True, random_state=seed)
    fn = lambda arr: pipeline.predict_proba(pd.DataFrame(arr, columns=X_train.columns))
    pred = pipeline.predict(fila)[0]
    idx = list(pipeline.classes_).index(pred)
    exp = explainer.explain_instance(fila.iloc[0].values, fn, num_features=num_features,
                                     labels=(idx,))
    return pred, exp, idx

"""
Clasificador tabular de rendimiento publicitario (Crítico / Alerta / Óptimo).
Baselines: Regresión Logística y Árbol de Decisión (profundidad 3) sobre variables base.
Modelo ajustado: RobustScaler + RandomForest (50 árboles, profundidad 4, semilla 42); la profundidad 4
se ratificó en v1.1 por ser la configuración más robusta ante ruido de etiqueta y de atribución
sobre variables base + 3 variables de negocio (cvr, profit_usd, cpm_usd).
Validación: 5-Fold Stratified Cross-Validation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.features.feature_engineering import BASE_FEATURES, FEATURES

METRICAS = {"accuracy": "accuracy", "f1_weighted": "f1_weighted",
            "precision_weighted": "precision_weighted", "recall_weighted": "recall_weighted"}


def construir_modelos(seed: int = 42) -> dict:
    """Devuelve {nombre: (pipeline, lista_de_features)}."""
    return {
        "Regresión Logística (baseline)": (
            Pipeline([("scaler", StandardScaler()),
                      ("clf", LogisticRegression(max_iter=3000, random_state=seed))]),
            BASE_FEATURES),
        "Árbol de Decisión d=3 (baseline)": (
            Pipeline([("clf", DecisionTreeClassifier(max_depth=3, random_state=seed))]),
            BASE_FEATURES),
        "Random Forest + RobustScaler (ajustado)": (
            Pipeline([("scaler", RobustScaler()),
                      ("clf", RandomForestClassifier(n_estimators=50, max_depth=4,
                                                     random_state=seed, n_jobs=-1))]),
            FEATURES),
    }


def evaluar_cv(pipeline, X: pd.DataFrame, y: pd.Series, n_splits: int = 5, seed: int = 42):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    res = cross_validate(pipeline, X, y, cv=cv, scoring=METRICAS, return_train_score=True)
    resumen = {}
    for m in METRICAS:
        resumen[f"train_{m}"] = float(np.mean(res[f"train_{m}"]))
        resumen[f"val_{m}"] = float(np.mean(res[f"test_{m}"]))
        resumen[f"val_{m}_std"] = float(np.std(res[f"test_{m}"]))
    resumen["brecha_train_val_accuracy"] = resumen["train_accuracy"] - resumen["val_accuracy"]
    resumen["fit_time_s"] = float(np.mean(res["fit_time"]))
    y_pred = cross_val_predict(pipeline, X, y, cv=cv)
    return resumen, y_pred


def busqueda_hiperparametros(X: pd.DataFrame, y: pd.Series, seed: int = 42) -> dict:
    """Justifica la configuración elegida: compara profundidad y número de árboles."""
    pipe = Pipeline([("scaler", RobustScaler()),
                     ("clf", RandomForestClassifier(random_state=seed, n_jobs=-1))])
    grid = {"clf__n_estimators": [50, 100, 200], "clf__max_depth": [3, 4, 6, None]}
    gs = GridSearchCV(pipe, grid, cv=StratifiedKFold(5, shuffle=True, random_state=seed),
                      scoring="f1_weighted", return_train_score=True, n_jobs=-1)
    gs.fit(X, y)
    tabla = pd.DataFrame(gs.cv_results_)[["param_clf__n_estimators", "param_clf__max_depth",
                                         "mean_train_score", "mean_test_score", "std_test_score"]]
    tabla.columns = ["n_estimators", "max_depth", "f1_train", "f1_val", "f1_val_std"]
    tabla["max_depth"] = tabla["max_depth"].astype(str)
    tabla["brecha"] = tabla["f1_train"] - tabla["f1_val"]
    return {"mejor": {k.replace("clf__", ""): (None if v is None else int(v))
                      for k, v in gs.best_params_.items()},
            "mejor_f1_val": float(gs.best_score_),
            "tabla": tabla.sort_values("f1_val", ascending=False).reset_index(drop=True)}


def entrenar_final(pipeline, X: pd.DataFrame, y: pd.Series):
    return pipeline.fit(X, y)

"""
Pronóstico del desempeño a 1–12 meses con modelos globales (una sola red de datos para todas las cuentas).

Estrategia directa multi-horizonte: para cada mes de origen t y cada horizonte h se construye un par
(variables con información hasta t, objetivo en t+h). Todas las variables usan SOLO datos de meses ≤ t
(sin fuga temporal) y se expresan en escala logarítmica relativa al nivel de 12 meses de la propia cuenta,
lo que permite generalizar a cuentas no vistas.

Validación (retroalimentación del tutor):
  * temporal          → entrenar con objetivos hasta el corte y evaluar pronósticos emitidos después;
  * por cliente       → GroupKFold: entrenar con unas cuentas y evaluar con otras distintas;
  * cliente + temporal → la combinación más estricta: cuentas y meses nunca vistos.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.rules.reglas_negocio import RANGO, estado_por_regla

H_MAX = 12
MIN_HIST = 12
HORIZONTES = (1, 3, 6, 12)
VERTICALES = ["Automotriz", "B2B_Servicios", "Ecommerce", "Educacion", "Gastronomia", "Inmobiliario",
              "Retail", "Salud_Clinica"]
VARS = (["nivel12", "rel_t", "rel_t1", "rel_t2", "rel_media3", "rel_media6", "rel_mediana12", "std6",
         "pendiente12", "pendiente24", "rel_estacional", "perfil_vertical", "perfil_cuenta", "h", "mes_sin",
         "mes_cos", "gasto_rel3"]
        + [f"v_{v}" for v in VERTICALES])
FEATS_ARBOL = ["rel_t", "rel_media3", "rel_media6", "pendiente24", "perfil_vertical", "perfil_cuenta",
               "rel_estacional", "h", "std6", "gasto_rel3"]
FEATS_LINEAL = FEATS_ARBOL + ["rel_t_h", "rel_media3_h", "rel_media6_h", "tendencia_h"]
NOMBRES_VAR = {
    "nivel12": "nivel de 12 meses", "rel_t": "último mes", "rel_t1": "mes t−1", "rel_t2": "mes t−2",
    "rel_media3": "promedio 3 meses", "rel_media6": "promedio 6 meses", "rel_mediana12": "mediana 12 meses",
    "std6": "volatilidad 6 meses", "pendiente12": "tendencia 12 meses",
    "rel_estacional": "mismo mes del año anterior", "pendiente24": "tendencia 24 meses",
    "perfil_vertical": "estacionalidad de la vertical", "perfil_cuenta": "estacionalidad de la cuenta",
    "h": "horizonte", "rel_t_h": "último mes ÷ horizonte", "rel_media3_h": "promedio 3 meses ÷ horizonte",
    "rel_media6_h": "promedio 6 meses ÷ horizonte", "tendencia_h": "tendencia × horizonte", "mes_sin": "mes del año (sen)",
    "mes_cos": "mes del año (cos)", "gasto_rel3": "inversión reciente"}


def _pendiente(v: np.ndarray) -> float:
    x = np.arange(len(v))
    return float(np.polyfit(x, v, 1)[0]) if len(v) > 2 else 0.0


def _perfiles(df: pd.DataFrame, objetivo: str) -> pd.DataFrame:
    """Desviación estacional de cada observación: log(valor) − promedio de los 12 meses previos de la cuenta."""
    filas = []
    for cuenta, g in df.groupby("cliente", sort=True):
        g = g.sort_values("fecha").reset_index(drop=True)
        y = np.log(g[objetivo].astype(float).clip(lower=0.05)).values
        f = pd.to_datetime(g["fecha"])
        for s in range(12, len(g)):
            filas.append({"cliente": cuenta, "vertical": str(g["vertical"].iloc[0]) if "vertical" in g else "",
                          "fecha": f.iloc[s], "mes": f.iloc[s].month, "desv": y[s] - y[s - 12:s].mean()})
    return pd.DataFrame(filas)


def _perfil_hasta(arr: dict, filtro: np.ndarray, mes: int, hasta) -> float:
    m = filtro & (arr["mes"] == mes) & (arr["fecha"] <= np.datetime64(hasta))
    return float(arr["desv"][m].mean()) if m.any() else 0.0


def construir_pares(df: pd.DataFrame, objetivo: str = "roas", h_max: int = H_MAX,
                    min_hist: int = MIN_HIST, incluir_futuro: bool = False, cuentas=None,
                    solo_ultimo_origen: bool = False) -> pd.DataFrame:
    """Pares (origen t, horizonte h). Con incluir_futuro=True agrega los pares cuyo objetivo aún no existe.
    Los perfiles estacionales usan solo observaciones con fecha ≤ origen (sin fuga temporal)."""
    perf = _perfiles(df, objetivo)
    arr = {"mes": perf["mes"].to_numpy(), "fecha": perf["fecha"].to_numpy(dtype="datetime64[ns]"),
           "desv": perf["desv"].to_numpy(), "vertical": perf["vertical"].to_numpy(),
           "cliente": perf["cliente"].to_numpy()} if len(perf) else None
    cache = {}
    filas = []
    for cuenta, g in df.groupby("cliente", sort=True):
        if cuentas is not None and cuenta not in cuentas:
            continue
        g = g.sort_values("fecha").reset_index(drop=True)
        y = np.log(g[objetivo].astype(float).clip(lower=0.05)).values
        lg = np.log(g["gasto_usd"].astype(float).clip(lower=1.0)).values
        fechas = pd.to_datetime(g["fecha"])
        vertical = str(g["vertical"].iloc[0]) if "vertical" in g else ""
        f_vert = arr["vertical"] == vertical if arr is not None else None
        f_cta = arr["cliente"] == cuenta if arr is not None else None
        n = len(g)
        for t in range(max(min_hist - 1, n - 1 if solo_ultimo_origen else 0), n):
            ult12 = y[t - 11:t + 1]
            nivel = float(ult12.mean())
            ult24 = y[max(0, t - 23):t + 1]
            base = {
                "cliente": cuenta, "vertical": vertical, "t": t, "fecha_origen": fechas.iloc[t],
                "nivel12": nivel, "rel_t": y[t] - nivel, "rel_t1": y[t - 1] - nivel, "rel_t2": y[t - 2] - nivel,
                "rel_media3": y[t - 2:t + 1].mean() - nivel, "rel_media6": y[t - 5:t + 1].mean() - nivel,
                "rel_mediana12": float(np.median(ult12)) - nivel, "std6": float(y[t - 5:t + 1].std()),
                "pendiente12": _pendiente(ult12), "pendiente24": _pendiente(ult24),
                "gasto_rel3": float(lg[t - 2:t + 1].mean() - lg[t - 11:t + 1].mean()),
                "y_t": y[t],
            }
            for v in VERTICALES:
                base[f"v_{v}"] = 1.0 if vertical == v else 0.0
            for h in range(1, h_max + 1):
                if t + h >= n and not incluir_futuro:
                    break
                f_obj = fechas.iloc[t] + pd.DateOffset(months=h)
                clave = (cuenta, t, f_obj.month)
                if clave not in cache:
                    cache[clave] = ((_perfil_hasta(arr, f_vert, f_obj.month, fechas.iloc[t]),
                                     _perfil_hasta(arr, f_cta, f_obj.month, fechas.iloc[t])) if arr is not None
                                    else (0.0, 0.0))
                pv, pc = cache[clave]
                fila = dict(base)
                fila.update({
                    "h": h, "fecha_objetivo": f_obj, "t_objetivo": t + h,
                    "mes_sin": np.sin(2 * np.pi * f_obj.month / 12), "mes_cos": np.cos(2 * np.pi * f_obj.month / 12),
                    "y_estacional": y[t + h - 12], "rel_estacional": y[t + h - 12] - nivel,
                    "perfil_vertical": pv, "perfil_cuenta": pc,
                    "y_real": y[t + h] if t + h < n else np.nan,
                })
                fila["rel_objetivo"] = fila["y_real"] - nivel
                fila["rel_t_h"] = fila["rel_t"] / h
                fila["rel_media3_h"] = fila["rel_media3"] / h
                fila["rel_media6_h"] = fila["rel_media6"] / h
                fila["tendencia_h"] = fila["pendiente24"] * (h + 5.5)
                filas.append(fila)
    return pd.DataFrame(filas)


# ----------------------------------------------------------------------------- modelos
class EnsamblePromedio(RegressorMixin, BaseEstimator):
    """Promedio de varios modelos, cada uno con su propio conjunto de variables."""

    def __init__(self, miembros=None):
        self.miembros = miembros          # lista de (estimador, variables)

    def fit(self, X, y):
        from sklearn.base import clone
        self.ajustados_ = [(clone(e).fit(X[f], y), f) for e, f in self.miembros]
        return self

    def predict(self, X):
        return np.mean([e.predict(X[f]) for e, f in self.ajustados_], axis=0)


def modelos_ml(seed: int = 42) -> dict:
    """{nombre: (estimador, variables)}. Todos predicen el log-valor relativo al nivel de 12 meses."""
    return {
        "Ridge con interacciones (global)": (
            make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20))), FEATS_LINEAL),
        "Gradient Boosting (global)": (
            GradientBoostingRegressor(loss="huber", n_estimators=200, max_depth=2, learning_rate=0.03,
                                      subsample=0.8, min_samples_leaf=20, random_state=seed), FEATS_ARBOL),
        "Random Forest (global)": (
            RandomForestRegressor(n_estimators=300, min_samples_leaf=20, max_features=0.5, random_state=seed,
                                  n_jobs=-1), FEATS_ARBOL),
        "Ensamble ML (Ridge + Gradient Boosting)": (
            EnsamblePromedio([
                (make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20))), FEATS_LINEAL),
                (GradientBoostingRegressor(loss="huber", n_estimators=200, max_depth=2, learning_rate=0.03,
                                           subsample=0.8, min_samples_leaf=20, random_state=seed), FEATS_ARBOL)]),
            FEATS_LINEAL),
    }


REFERENCIAS = {
    "Ingenuo (último mes)": lambda P: P["y_t"].values,
    "Promedio 12 meses": lambda P: P["nivel12"].values,
    "Estacional ingenuo (mismo mes año anterior)": lambda P: P["y_estacional"].values,
    "Perfil estacional de la vertical": lambda P: (P["nivel12"] + P["perfil_vertical"]).values,
}


def _ajustar_predecir(modelo, entreno: pd.DataFrame, prueba: pd.DataFrame) -> np.ndarray:
    from sklearn.base import clone
    est, feats = modelo
    m = clone(est).fit(entreno[feats], entreno["rel_objetivo"])
    return prueba["nivel12"].values + m.predict(prueba[feats])


def _particiones(P: pd.DataFrame, esquema: str, corte: int, n_grupos: int = 7):
    """Genera (índices de entrenamiento, índices de prueba) según el esquema de validación."""
    obs = P["y_real"].notna()
    if esquema == "temporal":
        yield P.index[obs & (P["t_objetivo"] <= corte)], P.index[obs & (P["t"] >= corte)]
        return
    gkf = GroupKFold(n_splits=n_grupos)
    Q = P[obs]
    for tr, te in gkf.split(Q, groups=Q["cliente"]):
        tr_idx, te_idx = Q.index[tr], Q.index[te]
        if esquema == "cliente":
            yield tr_idx, te_idx
        elif esquema == "cliente+temporal":
            yield (tr_idx[P.loc[tr_idx, "t_objetivo"] <= corte], te_idx[P.loc[te_idx, "t"] >= corte])


def predecir_esquema(P: pd.DataFrame, esquema: str, corte: int, seed: int = 42) -> pd.DataFrame:
    """Predicciones fuera de muestra de todos los modelos para un esquema de validación."""
    salidas = []
    for tr, te in _particiones(P, esquema, corte):
        prueba = P.loc[te].copy()
        for nombre, f in REFERENCIAS.items():
            prueba[nombre] = f(prueba)
        for nombre, m in modelos_ml(seed).items():
            prueba[nombre] = _ajustar_predecir(m, P.loc[tr], prueba)
        salidas.append(prueba)
    R = pd.concat(salidas)
    R["esquema"] = esquema
    return R


def seleccionar_modelo(P: pd.DataFrame, corte: int, seed: int = 42) -> tuple[str, pd.DataFrame]:
    """Elige el modelo de ML usando SOLO el periodo de entrenamiento (validación por cliente)."""
    E = P[P["y_real"].notna() & (P["t_objetivo"] <= corte)]
    R = predecir_esquema(E, "cliente", corte, seed)
    M = metricas(R)
    ml = M[M["modelo"].isin(list(modelos_ml(seed)))]
    tabla = ml.groupby("modelo")["MAE"].mean().sort_values().rename("MAE_medio_validacion").reset_index()
    return tabla["modelo"].iloc[0], tabla


def nombres_modelos() -> list[str]:
    return list(REFERENCIAS) + list(modelos_ml())


def metricas(R: pd.DataFrame, horizontes=HORIZONTES) -> pd.DataFrame:
    """MAE y MAPE en escala original del ROAS, y mejora frente al pronóstico ingenuo."""
    filas = []
    for h in horizontes:
        S = R[R["h"] == h]
        if S.empty:
            continue
        real = np.exp(S["y_real"].values)
        mae_ing = np.mean(np.abs(np.exp(S["Ingenuo (último mes)"].values) - real))
        for m in nombres_modelos():
            pred = np.exp(S[m].values)
            mae = np.mean(np.abs(pred - real))
            filas.append({"horizonte": h, "modelo": m, "n": len(S), "MAE": mae,
                          "MAPE_%": 100 * np.mean(np.abs(pred - real) / real),
                          "RMSE": float(np.sqrt(np.mean((pred - real) ** 2))),
                          "mejora_vs_ingenuo_%": 100 * (1 - mae / mae_ing)})
    return pd.DataFrame(filas)


def evaluar_deterioro(R: pd.DataFrame, modelo: str, h: int = 3) -> pd.DataFrame:
    """¿Anticipa el modelo el paso a un estado peor? Comparación con referencias."""
    S = R[R["h"] == h]
    actual = estado_por_regla(np.exp(S["y_t"].values))
    futuro = estado_por_regla(np.exp(S["y_real"].values))
    real_det = np.vectorize(RANGO.get)(futuro) < np.vectorize(RANGO.get)(actual)
    filas = []
    for m in [modelo, "Perfil estacional de la vertical", "Promedio 12 meses", "Ingenuo (último mes)"]:
        proy = estado_por_regla(np.exp(S[m].values))
        pred_det = np.vectorize(RANGO.get)(proy) < np.vectorize(RANGO.get)(actual)
        tp = int((pred_det & real_det).sum()); fp = int((pred_det & ~real_det).sum())
        fn = int((~pred_det & real_det).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        filas.append({"modelo": m, "horizonte": h, "deterioros_reales": int(real_det.sum()),
                      "alertas_emitidas": int(pred_det.sum()), "precision": prec, "recall": rec,
                      "F1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
                      "acierto_estado_futuro": float((proy == futuro).mean())})
    return pd.DataFrame(filas)


# ------------------------------------------------------------------- modelo final + intervalos
class Pronosticador:
    """Modelo global final con intervalos de predicción conformes por horizonte."""

    def __init__(self, nombre: str = "Ridge con interacciones (global)", objetivo: str = "roas", cobertura: float = 0.8,
                 seed: int = 42):
        self.nombre, self.objetivo, self.cobertura, self.seed = nombre, objetivo, cobertura, seed
        self.modelo = None
        self.cuantiles = {}

    def ajustar(self, df: pd.DataFrame):
        P = construir_pares(df, self.objetivo)
        P = P[P["y_real"].notna()].reset_index(drop=True)
        base = modelos_ml(self.seed)[self.nombre]
        # residuos fuera de muestra (por cliente) para calibrar los intervalos conformes
        oof = np.full(len(P), np.nan)
        for tr, te in GroupKFold(n_splits=7).split(P, groups=P["cliente"]):
            oof[te] = _ajustar_predecir(base, P.iloc[tr], P.iloc[te])
        res = P["y_real"].values - oof
        a = (1 - self.cobertura) / 2
        self.residuos = {}
        for h in range(1, H_MAX + 1):
            r = res[P["h"].values == h]
            self.residuos[h] = r
            self.cuantiles[h] = (float(np.quantile(r, a)), float(np.quantile(r, 1 - a)))
        from sklearn.base import clone
        est, self.feats = base
        self.modelo = clone(est).fit(P[self.feats], P["rel_objetivo"])
        self.pares_entreno = P
        return self

    def proyectar(self, df: pd.DataFrame, horizontes=HORIZONTES, cuentas=None) -> pd.DataFrame:
        """Proyección desde el último mes disponible de cada cuenta."""
        P = construir_pares(df, self.objetivo, incluir_futuro=True, solo_ultimo_origen=True, cuentas=cuentas)
        P = P[P["h"].isin(horizontes)].copy()
        P["pred_log"] = P["nivel12"] + self.modelo.predict(P[self.feats])
        P["inf_log"] = P["pred_log"] + P["h"].map(lambda h: self.cuantiles[h][0])
        P["sup_log"] = P["pred_log"] + P["h"].map(lambda h: self.cuantiles[h][1])
        out = P[["cliente", "vertical", "fecha_origen", "fecha_objetivo", "h"]].copy()
        out["proyeccion"] = np.exp(P["pred_log"].values)
        out["limite_inferior"] = np.exp(P["inf_log"].values)
        out["limite_superior"] = np.exp(P["sup_log"].values)
        out["valor_actual"] = np.exp(P["y_t"].values)
        if self.objetivo == "roas":
            out["estado_proyectado"] = estado_por_regla(out["proyeccion"].values)
            out["prob_deterioro"] = [float(prob_deterioro(np.array([pl]), np.array([va]), self.residuos[h])[0])
                                     for pl, va, h in zip(P["pred_log"], out["valor_actual"], out["h"])]
        # factores interpretables de la proyección (en % sobre el nivel de 12 meses)
        out["nivel_12m"] = np.exp(P["nivel12"].values)
        out["efecto_estacional_%"] = 100 * (np.exp(P["perfil_vertical"].values) - 1)
        out["tendencia_anual_%"] = 100 * (np.exp(12 * P["pendiente24"].values) - 1)
        return out.reset_index(drop=True)

    def proyectar_cuenta(self, df: pd.DataFrame, cuenta: str, horizontes=HORIZONTES, hasta=None) -> pd.DataFrame:
        """Proyección de una cuenta desde su último mes o desde el mes `hasta` (útil para contrastar con lo real)."""
        sub = df[df["cliente"] == cuenta]
        if hasta is not None:
            sub = sub[pd.to_datetime(sub["fecha"]) <= pd.to_datetime(hasta)]
        if len(sub) < MIN_HIST:
            return pd.DataFrame()
        # los perfiles estacionales de la vertical usan todas las cuentas hasta la misma fecha
        contexto = df[pd.to_datetime(df["fecha"]) <= pd.to_datetime(sub["fecha"]).max()]
        return self.proyectar(contexto, horizontes, cuentas={cuenta}).reset_index(drop=True)

    def importancia(self, n_repeticiones: int = 5) -> pd.DataFrame:
        """Importancia por permutación (aumento del error al desordenar cada variable), independiente del modelo."""
        from sklearn.inspection import permutation_importance
        P = self.pares_entreno
        r = permutation_importance(self.modelo, P[self.feats], P["rel_objetivo"], n_repeats=n_repeticiones,
                                   random_state=self.seed, scoring="neg_mean_absolute_error")
        t = pd.DataFrame({"variable": self.feats, "importancia": r.importances_mean})
        t["descripcion"] = t["variable"].map(NOMBRES_VAR).fillna(t["variable"])
        t["peso_%"] = 100 * t["importancia"].clip(lower=0) / t["importancia"].clip(lower=0).sum()
        return t.sort_values("importancia", ascending=False).reset_index(drop=True)


def evaluar_intervalos(P: pd.DataFrame, nombre: str, corte: int, cobertura: float = 0.8,
                       seed: int = 42) -> pd.DataFrame:
    """Cobertura de intervalos conformes en el esquema temporal: se calibran con residuos fuera de muestra
    del periodo de entrenamiento y se evalúan en los pronósticos posteriores al corte."""
    from sklearn.base import clone
    base = modelos_ml(seed)[nombre]
    obs = P["y_real"].notna()
    E = P[obs & (P["t_objetivo"] <= corte)].reset_index(drop=True)
    T = P[obs & (P["t"] >= corte)].reset_index(drop=True)
    oof = np.full(len(E), np.nan)
    for tr, te in GroupKFold(n_splits=7).split(E, groups=E["cliente"]):
        oof[te] = _ajustar_predecir(base, E.iloc[tr], E.iloc[te])
    res = E["y_real"].values - oof
    est, feats = base
    pred = T["nivel12"].values + clone(est).fit(E[feats], E["rel_objetivo"]).predict(T[feats])
    a = (1 - cobertura) / 2
    filas = []
    for h in HORIZONTES:
        r = res[E["h"].values == h]
        lo, hi = np.quantile(r, a), np.quantile(r, 1 - a)
        m = T["h"].values == h
        y = T["y_real"].values[m]
        dentro = ((y >= pred[m] + lo) & (y <= pred[m] + hi)).mean()
        filas.append({"horizonte": h, "n": int(m.sum()), "cobertura_objetivo": cobertura,
                      "cobertura_observada": float(dentro),
                      "amplitud_media_%": float(100 * (np.exp(hi) - np.exp(lo)))})
    return pd.DataFrame(filas)


def _umbral_inferior(estado_actual: np.ndarray) -> np.ndarray:
    from src.rules.reglas_negocio import UMBRAL_CRITICO, UMBRAL_OPTIMO
    return np.where(estado_actual == "Óptimo", UMBRAL_OPTIMO,
                    np.where(estado_actual == "Alerta", UMBRAL_CRITICO, np.nan))


def prob_deterioro(pred_log: np.ndarray, roas_actual: np.ndarray, residuos: np.ndarray) -> np.ndarray:
    """P(el ROAS futuro cae por debajo del umbral del estado actual), con la distribución empírica de residuos."""
    thr = _umbral_inferior(estado_por_regla(roas_actual))
    out = np.zeros(len(pred_log))
    ok = ~np.isnan(thr)
    out[ok] = [(p + residuos < np.log(t)).mean() for p, t in zip(pred_log[ok], thr[ok])]
    return out


def evaluar_alertas(P: pd.DataFrame, nombre: str, corte: int, horizontes=(3, 6), umbral_prob: float = 0.35,
                    esquema: str = "cliente+temporal", seed: int = 42) -> pd.DataFrame:
    """Alerta temprana de deterioro con probabilidad (modelo + residuos fuera de muestra) frente a referencias
    deterministas (cruce del umbral por el pronóstico puntual)."""
    from sklearn.base import clone
    from sklearn.metrics import average_precision_score
    base = modelos_ml(seed)[nombre]
    obs = P["y_real"].notna()
    filas = []
    for tr, te in _particiones(P, esquema, corte):
        E = P.loc[tr].reset_index(drop=True)
        T = P.loc[te].reset_index(drop=True)
        oof = np.full(len(E), np.nan)
        for a, b in GroupKFold(n_splits=min(6, E["cliente"].nunique())).split(E, groups=E["cliente"]):
            oof[b] = _ajustar_predecir(base, E.iloc[a], E.iloc[b])
        res_h = {h: (E["y_real"].values - oof)[E["h"].values == h] for h in horizontes}
        est, feats = base
        pred = T["nivel12"].values + clone(est).fit(E[feats], E["rel_objetivo"]).predict(T[feats])
        for h in horizontes:
            m = T["h"].values == h
            S = T[m]
            actual = estado_por_regla(np.exp(S["y_t"].values))
            futuro = estado_por_regla(np.exp(S["y_real"].values))
            real = np.vectorize(RANGO.get)(futuro) < np.vectorize(RANGO.get)(actual)
            prob = prob_deterioro(pred[m], np.exp(S["y_t"].values), res_h[h])
            filas.append(pd.DataFrame({"h": h, "real": real, "prob": prob,
                                       "ref_promedio12": np.vectorize(RANGO.get)(estado_por_regla(np.exp(S["nivel12"].values))) < np.vectorize(RANGO.get)(actual),
                                       "ref_perfil": np.vectorize(RANGO.get)(estado_por_regla(np.exp((S["nivel12"] + S["perfil_vertical"]).values))) < np.vectorize(RANGO.get)(actual)}))
    D = pd.concat(filas)
    salida = []

    def fila(metodo, h, real, pred, puntaje=None):
        tp = int((pred & real).sum()); fp = int((pred & ~real).sum()); fn = int((~pred & real).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        salida.append({"metodo": metodo, "horizonte": h, "casos": int(len(real)), "deterioros_reales": int(real.sum()),
                       "alertas": int(pred.sum()), "precision": p, "recall": r,
                       "F1": 2 * p * r / (p + r) if p + r else 0.0,
                       "AP": float(average_precision_score(real, puntaje)) if puntaje is not None and real.any() else np.nan})
    for h in horizontes:
        S = D[D["h"] == h]
        real = S["real"].values
        fila(f"{nombre}: probabilidad ≥ {umbral_prob:.0%}", h, real, S["prob"].values >= umbral_prob, S["prob"].values)
        fila("Perfil estacional de la vertical (cruce de umbral)", h, real, S["ref_perfil"].values)
        fila("Promedio 12 meses (cruce de umbral)", h, real, S["ref_promedio12"].values)
        fila("Ingenuo (sin cambio de estado)", h, real, np.zeros(len(real), bool))
    return pd.DataFrame(salida)
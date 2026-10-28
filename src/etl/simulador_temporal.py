"""
Simulador del dataset temporal v2 (48 meses por cuenta: octubre de 2022 a septiembre de 2026).

Motivo: el dataset v1 (13 meses por cuenta) no tiene estructura temporal (autocorrelación del ROAS ≈ −0,11,
sin tendencia ni estacionalidad) y es demasiado corto para entrenar y validar proyecciones a 6 y 12 meses.
El v2 conserva las 14 cuentas, sus verticales y sus niveles de ROAS, inversión, CTR y CPA del v1, y agrega
dinámicas documentadas propias de la operación publicitaria:

  * estacionalidad por vertical (p. ej., Black Friday y Navidad en e-commerce y retail),
  * tendencia por cuenta (mejora o deterioro gradual, con el signo observado en el v1),
  * persistencia mensual (proceso AR(1) sobre el log-ROAS),
  * rendimientos decrecientes de la inversión (más gasto, menor ROAS marginal),
  * inflación anual del CPM y un cambio estructural (relanzamiento) en una cuenta,
  * anomalías operativas inyectadas (falla de tracking, gasto duplicado, pico de CPM, tráfico inválido).

Las anomalías se guardan en un archivo aparte (verdad de referencia) que NUNCA se usa como variable de
entrada: solo sirve para evaluar el detector de anomalías.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Calibración tomada del dataset v1 (medias y medianas por cuenta)
CUENTAS = {
    #            vertical         ROAS   gasto  CTR    CPA    tendencia anual del ROAS
    "Cliente_01": ("Ecommerce",     2.07, 1875, 0.020, 54.0, 0.00),
    "Cliente_02": ("B2B_Servicios", 2.31, 1709, 0.025, 39.1, +0.10),
    "Cliente_03": ("Ecommerce",     2.45, 2039, 0.023, 40.6, 0.00),
    "Cliente_04": ("Salud_Clinica", 4.37, 1910, 0.024, 17.3, -0.16),
    "Cliente_05": ("Inmobiliario",  1.22, 1804, 0.021, 64.1, 0.00),   # relanzamiento en el mes 22
    "Cliente_06": ("Ecommerce",     4.16, 1621, 0.024, 25.9, -0.04),
    "Cliente_07": ("Educacion",     2.33, 1624, 0.024, 47.1, +0.04),
    "Cliente_08": ("Automotriz",    1.13, 1739, 0.024, 90.7, -0.03),
    "Cliente_09": ("B2B_Servicios", 2.27, 1802, 0.023, 45.8, -0.06),
    "Cliente_10": ("Ecommerce",     4.12, 1519, 0.020, 25.4, -0.14),
    "Cliente_11": ("Retail",        4.49, 1923, 0.022, 21.4, +0.03),
    "Cliente_12": ("Salud_Clinica", 2.35, 2030, 0.016, 46.8, +0.13),
    "Cliente_13": ("Gastronomia",   2.46, 1548, 0.018, 41.8, -0.05),
    "Cliente_14": ("Ecommerce",     3.97, 1704, 0.026, 22.0, +0.02),
}

# Estacionalidad multiplicativa por vertical (mes 1..12): (efecto en ROAS, efecto en inversión)
_EST = {
    "Ecommerce":     ({11: .30, 12: .26, 1: -.18, 2: -.08, 7: .06}, {11: .45, 12: .50, 1: -.20}),
    "Retail":        ({11: .22, 12: .32, 1: -.16, 5: .10},          {11: .35, 12: .55, 1: -.20}),
    "Educacion":     ({1: .20, 2: .16, 8: .24, 9: .14, 6: -.16, 7: -.20}, {1: .30, 8: .40, 9: .20, 7: -.25}),
    "Salud_Clinica": ({1: .14, 3: .08, 12: -.16, 8: -.08},          {1: .20, 12: -.15}),
    "Inmobiliario":  ({3: .12, 4: .08, 12: -.14, 8: -.06},          {3: .15, 12: -.20}),
    "Automotriz":    ({3: .16, 4: .10, 12: -.14, 1: -.08},          {3: .25, 12: -.10}),
    "B2B_Servicios": ({9: .10, 10: .12, 8: -.14, 12: -.18},         {10: .15, 8: -.15, 12: -.25}),
    "Gastronomia":   ({12: .22, 5: .12, 2: -.14, 9: -.08},          {12: .30, 2: -.15}),
}

TIPOS_ANOMALIA = ["falla_tracking", "gasto_duplicado", "pico_cpm", "trafico_invalido"]


def _efecto(vertical: str, mes: int, cual: int) -> float:
    return _EST[vertical][cual].get(mes, 0.0)


def generar_dataset_temporal(inicio: str = "2022-10-01", meses: int = 48, seed: int = 42,
                             n_anomalias: int = 36) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Devuelve (dataset de exportación mensual, tabla de anomalías inyectadas)."""
    rng = np.random.default_rng(seed)
    fechas = pd.date_range(inicio, periods=meses, freq="MS")
    filas, anomalias = [], []
    # posiciones de las anomalías: ni en los 12 primeros meses (historia mínima del detector) ni en los 2 últimos
    elegibles = [(c, t) for c in CUENTAS for t in range(12, meses - 2)]
    sel = rng.choice(len(elegibles), size=n_anomalias, replace=False)
    plan = {elegibles[i]: TIPOS_ANOMALIA[j % len(TIPOS_ANOMALIA)] for j, i in enumerate(sel)}
    for k, (cuenta, (vertical, roas0, gasto0, ctr0, cpa0, tend)) in enumerate(CUENTAS.items()):
        ar = 0.0
        presupuesto = 0.0
        ticket = roas0 * cpa0                      # ingreso medio por conversión
        cpm0 = rng.uniform(5.5, 8.5)
        for t, f in enumerate(fechas):
            anios = t / 12
            # inversión: estacionalidad + plan de presupuesto persistente + ruido
            presupuesto = 0.7 * presupuesto + rng.normal(0, 0.12)
            gasto = gasto0 * np.exp(_efecto(vertical, f.month, 1) + presupuesto + rng.normal(0, 0.10))
            # log-ROAS: nivel + tendencia + estacionalidad + AR(1) + rendimientos decrecientes + ruido
            ar = 0.60 * ar + rng.normal(0, 0.06)
            cambio = 0.0
            if cuenta == "Cliente_05" and t >= 22:   # relanzamiento de la oferta inmobiliaria
                cambio = 0.32
            log_roas = (np.log(roas0) + tend * (anios - 1.5) + _efecto(vertical, f.month, 0) + ar + cambio
                        - 0.12 * np.log(gasto / gasto0) + rng.normal(0, 0.04))
            roas = float(np.exp(log_roas))
            ingresos = gasto * roas
            cpm = cpm0 * (1.06 ** anios) * (1 + 0.6 * max(_efecto(vertical, f.month, 1), 0)) * rng.lognormal(0, 0.08)
            impresiones = gasto / cpm * 1000
            ctr = ctr0 * rng.lognormal(0, 0.10)
            clics = impresiones * ctr
            conversiones = max(1, rng.poisson(ingresos / ticket))
            fila = {"periodo": f.strftime("%Y-%m"), "cliente_id": cuenta, "vertical": vertical,
                    "gasto_usd": gasto, "impresiones": impresiones, "clics": clics,
                    "conversiones": conversiones, "ingresos_usd": ingresos}
            # anomalías operativas inyectadas (verdad de referencia guardada aparte)
            tipo = plan.get((cuenta, t))
            if tipo:
                if tipo == "falla_tracking":
                    fila["conversiones"] = max(0, int(fila["conversiones"] * rng.uniform(0.03, 0.15)))
                    fila["ingresos_usd"] *= rng.uniform(0.03, 0.15)
                elif tipo == "gasto_duplicado":
                    fila["gasto_usd"] *= rng.uniform(1.9, 2.4)
                    fila["impresiones"] *= rng.uniform(1.0, 1.1)
                elif tipo == "pico_cpm":
                    fila["impresiones"] /= rng.uniform(2.2, 3.0)
                    fila["clics"] /= rng.uniform(1.8, 2.4)
                elif tipo == "trafico_invalido":
                    fila["clics"] *= rng.uniform(3.5, 5.0)
                anomalias.append({"periodo": fila["periodo"], "cliente_id": cuenta, "anomalia": tipo})
            filas.append(fila)
    df = pd.DataFrame(filas)
    df["gasto_usd"] = df["gasto_usd"].round(2)
    df["ingresos_usd"] = df["ingresos_usd"].round(2)
    df["impresiones"] = df["impresiones"].round().astype(int)
    df["clics"] = df["clics"].round().astype(int)
    df["roas"] = (df["ingresos_usd"] / df["gasto_usd"]).round(2)
    return df, pd.DataFrame(anomalias, columns=["periodo", "cliente_id", "anomalia"])


def diagnostico_temporal(df: pd.DataFrame, col_cliente: str = "cliente", col_fecha: str = "fecha",
                         col: str = "roas") -> dict:
    """Persistencia (autocorrelación a 1 mes), estacionalidad (autocorrelación a 12 meses) y tendencia del log-ROAS."""
    acf1, acf12, pend, n = [], [], [], []
    for _, g in df.sort_values([col_cliente, col_fecha]).groupby(col_cliente):
        y = np.log(g[col].clip(lower=0.05).values)
        y = y - y.mean()
        n.append(len(y))
        if len(y) > 3:
            acf1.append(np.corrcoef(y[:-1], y[1:])[0, 1])
            pend.append(np.polyfit(np.arange(len(y)), y, 1)[0] * 12)
        if len(y) >= 24:
            acf12.append(np.corrcoef(y[:-12], y[12:])[0, 1])
    return {"meses_por_cuenta": int(np.median(n)), "acf1_media": float(np.mean(acf1)),
            "acf12_media": float(np.mean(acf12)) if acf12 else None,
            "pendiente_anual_abs_media": float(np.mean(np.abs(pend))),
            "valida_proyeccion_12m": bool(np.median(n) >= 36)}

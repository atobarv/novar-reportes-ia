"""
Capa determinista: todo lo que se resuelve con aritmética o con reglas de negocio explícitas.

El ROAS es un cálculo (ingresos / inversión), no una predicción. Por eso el estado ACTUAL de una cuenta
(Crítico / Alerta / Óptimo) se obtiene con la regla comercial de Novar Agencia y no con un modelo de ML.
El aprendizaje automático se reserva para las tareas que requieren inferir algo que no se puede calcular
(proyectar el desempeño futuro y detectar patrones anómalos); ver TAREAS.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

UMBRAL_CRITICO = 1.8     # ROAS < 1,8  -> Crítico (la inversión no se recupera con margen)
UMBRAL_OPTIMO = 2.8      # ROAS ≥ 2,8  -> Óptimo
ESTADOS = ["Crítico", "Alerta", "Óptimo"]
RANGO = {e: i for i, e in enumerate(ESTADOS)}

TAREAS = [
    # (tarea, enfoque, justificación)
    ("Consolidar exportaciones y calcular KPI (ROAS, CPA, CTR, CVR, CPM, beneficio)", "Determinista",
     "Son operaciones aritméticas exactas; un modelo solo agregaría error."),
    ("Clasificar el estado actual de la cuenta", "Determinista (regla de negocio)",
     "La agencia define el estado con umbrales fijos de ROAS (< 1,8 Crítico; ≥ 2,8 Óptimo)."),
    ("Comparar con el pasado (mes anterior, mismo mes del año anterior, promedio 12 meses)", "Determinista",
     "Son diferencias y promedios sobre datos observados."),
    ("Recomendaciones de acción", "Determinista (reglas de negocio)",
     "Se derivan del estado, de las variaciones y de las alertas con reglas auditables."),
    ("Proyectar el ROAS a 3, 6 y 12 meses", "Aprendizaje automático",
     "El futuro no se puede calcular: requiere aprender tendencia, estacionalidad y persistencia de datos históricos."),
    ("Anticipar deterioros (paso a un estado peor)", "Aprendizaje automático + regla",
     "Se aplica la regla de negocio sobre el ROAS proyectado por el modelo."),
    ("Detectar meses con patrones anómalos (fallas de tracking, gasto duplicado, tráfico inválido)", "Aprendizaje automático no supervisado",
     "Las anomalías combinan varias métricas y no tienen un umbral único conocido."),
    ("Redactar el resumen ejecutivo", "Modelo de lenguaje con verificación determinista",
     "La redacción requiere lenguaje natural; las cifras se verifican contra los datos calculados."),
]


def estado_por_regla(roas) -> np.ndarray | str:
    r = np.asarray(roas, dtype=float)
    est = np.where(r < UMBRAL_CRITICO, "Crítico", np.where(r >= UMBRAL_OPTIMO, "Óptimo", "Alerta"))
    return est.item() if est.ndim == 0 else est


def es_deterioro(estado_actual, estado_futuro) -> np.ndarray:
    a = np.vectorize(RANGO.get)(np.asarray(estado_actual))
    f = np.vectorize(RANGO.get)(np.asarray(estado_futuro))
    return f < a


def recomendaciones(h: dict, estado: str, proyecciones: dict | None = None, anomalia: dict | None = None) -> list[str]:
    """Reglas de acción auditables. h = hechos del periodo; proyecciones = {meses: dict}; anomalia = dict."""
    r = []
    if h.get("dato_imputado"):
        r.append("completar o verificar los datos del mes antes de enviar, porque contienen valores imputados")
    if anomalia and anomalia.get("es_anomalia"):
        r.append(f"verificar la medición del periodo antes de decidir, porque se detectó un patrón anómalo en "
                 f"{anomalia.get('metrica_principal', 'las métricas')}")
    if estado == "Crítico":
        r.append("pausar o reestructurar los conjuntos de anuncios con retorno inferior a la inversión")
    if proyecciones:
        h3 = proyecciones.get(3)
        if h3 and RANGO[h3["estado"]] < RANGO[estado]:
            r.append(f"actuar de forma preventiva, porque se proyecta un deterioro a {h3['estado']} en 3 meses")
        elif h3 and RANGO[h3["estado"]] > RANGO[estado]:
            if estado == "Crítico" or (anomalia and anomalia.get("es_anomalia")):
                r.append("confirmar la recuperación proyectada a 3 meses antes de cambiar el presupuesto")
            else:
                r.append("mantener la estrategia, porque la proyección a 3 meses indica una mejora del estado")
    if h.get("var_cpa") is not None and h["var_cpa"] > 15:
        r.append("revisar segmentación y creatividades, porque el CPA aumentó frente al mes anterior")
    if h.get("cvr", 99) < 2.0 and h.get("clics", 0) > 0:
        r.append("optimizar la página de destino y el formulario para mejorar la tasa de conversión")
    if estado == "Óptimo" and not any("deterioro" in x for x in r):
        r.append("escalar el presupuesto de forma gradual vigilando el ROAS marginal")
    if not r:
        r.append("mantener la estrategia actual y monitorear la tendencia del ROAS")
    return r[:3]


def tabla_tareas() -> pd.DataFrame:
    return pd.DataFrame(TAREAS, columns=["Tarea", "Enfoque", "Justificación"])

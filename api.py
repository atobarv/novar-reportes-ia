"""
Endpoint REST ligero (FastAPI) · Sistema de Generación Inteligente de Reportes de Marketing · v2.1

Ejecutar:   uvicorn api:app --port 8000
Documentación interactiva (OpenAPI): http://localhost:8000/docs

Flujo de uso:  acción (POST /reporte con las métricas mensuales de una o más cuentas)
            -> procesamiento (ETL, KPI, regla de estado, anomalías, proyección 3/6/12, agentes, guardrail)
            -> resultado (JSON con estado, comparativo, proyecciones, alertas, recomendaciones y texto verificado).
El texto devuelto es un BORRADOR: requiere la aprobación del Director de Cuenta antes de enviarse al cliente.
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))

from src.agents.llm_narrative_generator import (AgenteAnalista, AgenteExtractor, AgentePronostico,  # noqa: E402
                                                GuardrailNumerico, bloque_datos, plantilla_determinista)
from src.etl.ingesta_marketing import anonimizar, limpiar  # noqa: E402
from src.features.feature_engineering import preparar_dataset  # noqa: E402
from src.models.anomalias import caracteristicas, imputar_anomalias  # noqa: E402

VERSION = "2.1"
app = FastAPI(title="Novar · Reportes inteligentes", version=VERSION,
              description="Borrador de reporte mensual con comparativo histórico y proyección del ROAS.")
_ARTEFACTOS = {}


def artefactos():
    if not _ARTEFACTOS:
        _ARTEFACTOS["pron"] = joblib.load(RAIZ / "models" / "pronosticador_roas.joblib")
        _ARTEFACTOS["det"] = joblib.load(RAIZ / "models" / "detector_anomalias.joblib")
    return _ARTEFACTOS["pron"], _ARTEFACTOS["det"]


class Registro(BaseModel):
    fecha: str = Field(..., examples=["2026-09-01"], description="Mes (AAAA-MM o AAAA-MM-DD)")
    cliente: str = Field(..., examples=["CLI-8eb413e3"])
    vertical: str = Field("", examples=["B2B_Servicios"])
    gasto_usd: float | str | None = Field(..., examples=[2558.86])
    impresiones: float | str | None = Field(..., examples=[310000])
    clics: float | str | None = Field(..., examples=[5200])
    conversiones: float | str | None = Field(..., examples=[45])
    ingresos_usd: float | str | None = Field(..., examples=[6026.08])
    plataforma: str | None = Field(None, examples=["Meta Ads"])


class Solicitud(BaseModel):
    registros: list[Registro] = Field(..., description="Historia mensual (ideal: todas las cuentas y ≥ 12 meses)")
    cliente: str = Field(..., description="Cuenta a reportar")
    mes: str | None = Field(None, description="Mes a reportar (AAAA-MM); por defecto, el último")
    sal: str | None = Field(None, description="Si se envía, las cuentas se seudonimizan con SHA-256 + sal")


@app.get("/salud")
def salud():
    return {"estado": "ok", "version": VERSION}


@app.post("/reporte")
def reporte(sol: Solicitud):
    try:
        crudo = pd.DataFrame([r.model_dump() for r in sol.registros])
        df, calidad = limpiar(crudo)
        cuenta = sol.cliente
        if sol.sal:
            df = anonimizar(df, sol.sal)
            import hashlib
            cuenta = "CLI-" + hashlib.sha256((sol.sal + sol.cliente).encode()).hexdigest()[:8]
        df, _ = preparar_dataset(df)
        if cuenta not in set(df["cliente"]):
            raise ValueError(f"La cuenta «{sol.cliente}» no tiene meses con actividad en los registros enviados.")
        pron, det = artefactos()
        F = det.puntuar(caracteristicas(df))
        dm = imputar_anomalias(df, F["es_anomalia"])
        fechas = pd.to_datetime(df.loc[df["cliente"] == cuenta, "fecha"])
        fecha = fechas.max() if sol.mes is None else pd.Timestamp(sol.mes + ("-01" if len(sol.mes) == 7 else ""))
        if fecha not in set(fechas):
            raise ValueError(f"No hay datos de {sol.mes} para la cuenta.")
        h = AgenteExtractor(df).consolidar(cuenta, fecha)
        p = AgentePronostico(pron, dm).proyectar(cuenta, fecha)
        d = AgenteAnalista(F).diagnosticar(h, p)
        texto = plantilla_determinista(h, d, p)
        ok, fallas = GuardrailNumerico(bloque_datos(h, d, p), h, p).verificar(texto)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return {
        "version": VERSION, "cuenta": cuenta, "mes": h["periodo"], "estado": d["estado"],
        "regla_estado": "ROAS < 1,8 Crítico; 1,8 ≤ ROAS < 2,8 Alerta; ROAS ≥ 2,8 Óptimo",
        "kpi": {k: round(h[k], 4) for k in ["inversion", "ingresos", "beneficio", "roas", "cpa", "ctr", "cvr"]},
        "comparativo": {k: (None if h.get(k) is None else round(h[k], 4))
                        for k in ["roas_mes_anterior", "var_roas", "roas_hace_12m", "var_roas_anual", "roas_promedio_12m"]},
        "proyecciones": {f"{k}_meses": {"mes": v["periodo"], "roas": round(v["roas"], 3), "rango_80": [round(v["inf"], 3), round(v["sup"], 3)],
                                        "estado": v["estado"], "prob_deterioro": round(v["prob_deterioro"], 3)} for k, v in p.items()},
        "proyeccion_disponible": bool(p),
        "anomalia": d["anomalia"], "dato_imputado": h.get("dato_imputado", False),
        "recomendaciones": d["recomendaciones"], "texto_borrador": texto,
        "guardrail": {"aprobado": ok, "fallas": fallas},
        "advertencias_calidad": calidad.get("advertencias", []),
        "requiere_aprobacion_humana": True,
    }

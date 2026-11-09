"""
Ejecución del pipeline completo por línea de comandos (sin notebook) · v2.2

Uso:
    python scripts/run_pipeline.py --datos data/raw/dataset_marketing_novar_temporal_v2.csv
    python scripts/run_pipeline.py --datos <csv> --llm                 # redacción con Qwen2.5 (GPU recomendada)
    python scripts/run_pipeline.py --datos <csv> --cliente CLI-xxxx --mes 2026-03

Pasos: ETL -> KPI y estado por regla -> detector de anomalías -> proyección 3/6/12 meses -> agentes -> PDF.
Si existen models/pronosticador_roas.joblib y models/detector_anomalias.joblib se reutilizan; si no, se entrenan.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src.agents.llm_narrative_generator import (AgenteAnalista, AgenteExtractor, AgentePronostico,  # noqa: E402
                                                AgenteRedactor)
from src.etl.ingesta_marketing import anonimizar, cargar_archivo, limpiar  # noqa: E402
from src.features.feature_engineering import preparar_dataset  # noqa: E402
from src.models.anomalias import DetectorAnomalias, caracteristicas, imputar_anomalias  # noqa: E402
from src.models.pronostico import Pronosticador  # noqa: E402
from src.reports.exportador import exportar, generar_html  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Pipeline de reportes inteligentes · Novar Agencia v2.2")
    ap.add_argument("--datos", required=True, help="CSV o Excel con métricas mensuales por cuenta")
    ap.add_argument("--sal", default=None, help="Sal para seudonimizar (si se omite, no se seudonimiza)")
    ap.add_argument("--llm", action="store_true", help="Redactar con el LLM (si no, plantilla determinista)")
    ap.add_argument("--modelo-llm", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--cliente", default=None)
    ap.add_argument("--mes", default=None, help="Mes a analizar (AAAA-MM); por defecto, el último")
    ap.add_argument("--salida", default=str(RAIZ / "outputs" / "reports"))
    a = ap.parse_args()

    df, _ = limpiar(cargar_archivo(a.datos))
    if a.sal:
        df = anonimizar(df, a.sal)
    df, origen = preparar_dataset(df)
    print(f"[ETL] {len(df)} registros · {df['cliente'].nunique()} cuentas · estado: {origen}")

    ruta_det, ruta_pr = RAIZ / "models" / "detector_anomalias.joblib", RAIZ / "models" / "pronosticador_roas.joblib"
    F = caracteristicas(df)
    det = joblib.load(ruta_det) if ruta_det.exists() else DetectorAnomalias().ajustar(F)
    FA = det.puntuar(F)
    df_modelo = imputar_anomalias(df, FA["es_anomalia"])
    print(f"[Anomalías] {int(FA['es_anomalia'].sum())} meses marcados")
    pr = joblib.load(ruta_pr) if ruta_pr.exists() else Pronosticador().ajustar(df_modelo)
    print(f"[Proyección] modelo: {pr.nombre}")

    ext, apr, ana = AgenteExtractor(df), AgentePronostico(pr, df_modelo), AgenteAnalista(FA)
    red = AgenteRedactor(a.modelo_llm, usar_llm=a.llm)
    cuentas = [a.cliente] if a.cliente else sorted(df["cliente"].unique())
    mes = pd.Timestamp(a.mes + "-01") if a.mes else None
    for c in cuentas:
        h = ext.consolidar(c, mes)
        p = apr.proyectar(c, mes)
        d = ana.diagnosticar(h, p)
        r = red.generar(h, d, p)
        ruta = exportar(generar_html(h, d, p, r["texto"], r["fuente"], ext.historia(c, mes, 24)),
                        Path(a.salida) / f"reporte_{c}_{h['fecha']:%Y-%m}")
        p12 = p.get(12, {}).get("roas")
        print(f"[Reporte] {c} · {h['periodo']}: {d['estado']} · ROAS {h['roas']:.2f}"
              + (f" → 12 m {p12:.2f}" if p12 else "") + f" · {r['fuente']} → {ruta.name}")


if __name__ == "__main__":
    main()

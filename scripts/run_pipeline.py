"""
Ejecución del pipeline completo por línea de comandos (sin notebook).

Uso:
    python scripts/run_pipeline.py --datos data/raw/dataset_marketing_novar_agencia.csv
    python scripts/run_pipeline.py --datos <csv> --llm            # redacción con Qwen2.5 (GPU recomendada)
    python scripts/run_pipeline.py --datos <csv> --cliente CLI-xxxx

Pasos: ETL -> KPIs -> entrenamiento y validación 5-Fold -> modelo .joblib -> reportes PDF/HTML.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import joblib

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src.agents.llm_narrative_generator import AgenteAnalista, AgenteExtractor, AgenteRedactor  # noqa: E402
from src.etl.ingesta_marketing import anonimizar, cargar_archivo, limpiar  # noqa: E402
from src.features.feature_engineering import FEATURES, preparar_dataset  # noqa: E402
from src.models.baseline_models import construir_modelos, evaluar_cv  # noqa: E402
from src.reports.exportador import exportar, generar_html  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Pipeline de reportes inteligentes · Novar Agencia")
    ap.add_argument("--datos", required=True, help="CSV o Excel con métricas publicitarias")
    ap.add_argument("--sal", default="novar-demo", help="Sal para seudonimizar cuentas")
    ap.add_argument("--llm", action="store_true", help="Redactar con el LLM (si no, plantilla determinista)")
    ap.add_argument("--modelo-llm", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--cliente", default=None, help="Generar solo el reporte de esta cuenta")
    ap.add_argument("--salida", default=str(RAIZ / "outputs"))
    a = ap.parse_args()

    salida = Path(a.salida)
    df, calidad = limpiar(cargar_archivo(a.datos))
    df = anonimizar(df, a.sal)
    df, origen = preparar_dataset(df)
    y = df["estado_rendimiento"]
    print(f"[ETL] {len(df)} registros · {df['cliente'].nunique()} cuentas · etiqueta: {origen}")

    pipe, feats = construir_modelos()["Random Forest + RobustScaler (ajustado)"]
    metricas, _ = evaluar_cv(pipe, df[feats], y)
    print(f"[Modelo] Accuracy 5-Fold: {metricas['val_accuracy']:.2%} ± {metricas['val_accuracy_std']:.2%} · "
          f"F1: {metricas['val_f1_weighted']:.2%}")
    pipe.fit(df[FEATURES], y)
    (RAIZ / "models").mkdir(exist_ok=True)
    joblib.dump({"pipeline": pipe, "features": FEATURES, "clases": list(pipe.classes_),
                 "entrenado": datetime.now().isoformat(timespec="seconds")},
                RAIZ / "models" / "modelo_rendimiento.joblib")

    extractor, analista = AgenteExtractor(df), AgenteAnalista(pipe)
    redactor = AgenteRedactor(a.modelo_llm, usar_llm=a.llm)
    cuentas = [a.cliente] if a.cliente else sorted(df["cliente"].unique())
    for c in cuentas:
        h, fila = extractor.consolidar(c)
        d = analista.diagnosticar(h, fila)
        r = redactor.generar(h, d)
        ruta = exportar(generar_html(h, d, r["texto"], r["fuente"]), salida / "reports" / f"reporte_{c}")
        print(f"[Reporte] {c}: {d['estado']} ({d['confianza']:.1f} %) · {r['fuente']} → {ruta.name}")
    (salida / "metrics").mkdir(parents=True, exist_ok=True)
    (salida / "metrics" / "ultima_ejecucion_cli.json").write_text(
        json.dumps({"calidad": calidad, "metricas_cv": metricas}, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")


if __name__ == "__main__":
    main()

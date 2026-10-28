"""
Exportación del reporte ejecutivo (HTML -> PDF con WeasyPrint) con gráfico de histórico y proyección,
comparativo con el pasado y bloque de validación humana (HITL).
Identidad visual Novar Agencia: negro, rojo #CC1417 y blanco.
"""
from __future__ import annotations

import base64
import html
import io
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.agents.llm_narrative_generator import pct, usd
from src.rules.reglas_negocio import UMBRAL_CRITICO, UMBRAL_OPTIMO

COLOR_ESTADO = {"Óptimo": "#1E7D3A", "Alerta": "#C98A00", "Crítico": "#CC1417"}


def grafico_proyeccion(hist: pd.DataFrame, proy: dict, reales_posteriores: pd.DataFrame | None = None,
                       ancho: float = 7.2, alto: float = 2.4):
    """Figura matplotlib: ROAS histórico, proyección a 3/6/12 meses con rango del 80 % y umbrales de negocio."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(ancho, alto))
    ax.axhspan(0, UMBRAL_CRITICO, color="#CC1417", alpha=0.06)
    ax.axhspan(UMBRAL_CRITICO, UMBRAL_OPTIMO, color="#E0A100", alpha=0.06)
    ax.axhline(UMBRAL_CRITICO, color="#CC1417", lw=0.8, ls=":")
    ax.axhline(UMBRAL_OPTIMO, color="#1E7D3A", lw=0.8, ls=":")
    f = pd.to_datetime(hist["fecha"])
    ax.plot(f, hist["roas"], color="#111111", lw=1.8, marker="o", ms=3, label="ROAS observado")
    if proy:
        ks = sorted(proy)
        fx = [f.iloc[-1]] + [pd.to_datetime(proy[k]["fecha"]) for k in ks]
        y = [hist["roas"].iloc[-1]] + [proy[k]["roas"] for k in ks]
        lo = [hist["roas"].iloc[-1]] + [proy[k]["inf"] for k in ks]
        hi = [hist["roas"].iloc[-1]] + [proy[k]["sup"] for k in ks]
        ax.fill_between(fx, lo, hi, color="#CC1417", alpha=0.15, label="Rango 80 %")
        ax.plot(fx, y, color="#CC1417", lw=1.8, ls="--", marker="s", ms=4, label="Proyección (ML)")
        for k in ks:
            ax.annotate(f"{k}m: {proy[k]['roas']:.2f}", (pd.to_datetime(proy[k]["fecha"]), proy[k]["roas"]),
                        textcoords="offset points", xytext=(0, 7), ha="center", fontsize=7, color="#CC1417")
    if reales_posteriores is not None and len(reales_posteriores):
        ax.plot(pd.to_datetime(reales_posteriores["fecha"]), reales_posteriores["roas"], color="#8A8A8A", lw=1.2,
                marker="o", ms=2.5, label="Lo que ocurrió después")
    vals = list(hist["roas"]) + ([proy[k]["inf"] for k in proy] + [proy[k]["sup"] for k in proy] if proy else [])
    ax.set_ylim(max(0, min(vals + [UMBRAL_CRITICO]) - 0.3), max(vals + [UMBRAL_OPTIMO]) + 0.45)
    ax.set_ylabel("ROAS")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=7, loc="upper left", ncol=4, frameon=False)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    return fig


def _png_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=170, bbox_inches="tight", facecolor="white")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def generar_html(h: dict, d: dict, proy: dict, narrativa: str, fuente: str, hist: pd.DataFrame | None = None,
                 validador: str = "", aprobado: bool = False, correcciones: int | None = None) -> str:
    color = COLOR_ESTADO.get(d["estado"], "#333")
    kpis = [("Inversión", usd(h["inversion"])), ("Ingresos atribuidos", usd(h["ingresos"])),
            ("Beneficio neto", usd(h["beneficio"])), ("ROAS", f"{h['roas']:.2f}"), ("CPA", usd(h["cpa"])),
            ("Conversiones", f"{h['conversiones']:,}"), ("CTR", pct(h["ctr"])), ("Tasa de conversión", pct(h["cvr"]))]
    t_kpi = "".join(f"<tr><td>{k}</td><td class='v'>{v}</td></tr>" for k, v in kpis)
    comp = [("Mes anterior", h.get("roas_mes_anterior"), h.get("var_roas")),
            ("Mismo mes del año anterior", h.get("roas_hace_12m"), h.get("var_roas_anual")),
            ("Promedio últimos 12 meses", h.get("roas_promedio_12m"), None)]
    t_comp = "".join(f"<tr><td>{k}</td><td class='v'>{'' if v is None else f'{v:.2f}'}</td>"
                     f"<td class='v'>{'' if var is None else pct(var, True)}</td></tr>" for k, v, var in comp)
    t_proy = "".join(f"<tr><td>{k} meses ({html.escape(p['periodo'])})</td><td class='v'>{p['roas']:.2f}</td>"
                     f"<td class='v'>{p['inf']:.2f} – {p['sup']:.2f}</td><td class='v'>{p['estado']}</td>"
                     f"<td class='v'>{100 * p['prob_deterioro']:.0f} %</td></tr>" for k, p in sorted(proy.items()))
    recs = "".join(f"<li>{html.escape(r[0].upper() + r[1:])}.</li>" for r in d["recomendaciones"])
    img = ""
    if hist is not None and len(hist):
        img = f"<img src='data:image/png;base64,{_png_base64(grafico_proyeccion(hist, proy))}' style='width:100%'>"
    an = d["anomalia"]
    nota_an = (f"<p class='alerta'>⚠ Patrón anómalo detectado en {html.escape(str(an.get('metrica_principal')))}: "
               "verifique la medición antes de enviar.</p>") if an.get("es_anomalia") else ""
    estado_val = ("APROBADO por " + html.escape(validador) + (f" · {correcciones} palabras corregidas"
                  if correcciones is not None else "")) if aprobado and validador else "PENDIENTE DE VALIDACIÓN HUMANA"
    return f"""<!doctype html><html lang="es"><head><meta charset="utf-8">
<style>
 @page {{ size: A4; margin: 11mm 13mm; }}
 body {{ font-family: Arial, Helvetica, sans-serif; color:#111; font-size:9.2pt; }}
 .top {{ background:#000; color:#fff; padding:12px 16px; border-bottom:5px solid #CC1417; }}
 .top h1 {{ margin:0; font-size:15.5pt; }} .top p {{ margin:4px 0 0; font-size:8.5pt; color:#ddd; }}
 .badge {{ display:inline-block; padding:4px 11px; border-radius:4px; color:#fff; background:{color}; font-weight:bold; }}
 table {{ width:100%; border-collapse:collapse; margin:6px 0; }}
 th {{ text-align:left; font-size:8.5pt; color:#555; border-bottom:1px solid #999; padding:4px; }}
 td {{ border-bottom:1px solid #e5e5e5; padding:2.5px 4px; }} td.v {{ text-align:right; font-weight:bold; }}
 h2 {{ font-size:10.5pt; border-left:4px solid #CC1417; padding-left:8px; margin:9px 0 4px; }}
 .dos {{ display:flex; gap:14px; }} .dos > div {{ flex:1; }}
 .nota {{ font-size:7.8pt; color:#555; }} .alerta {{ color:#CC1417; font-weight:bold; }}
 .hitl {{ margin-top:9px; border:1px dashed #999; padding:9px; font-size:8.8pt; }}
</style></head><body>
<div class="top"><h1>Reporte ejecutivo de desempeño publicitario</h1>
<p>Novar Agencia · Cuenta {html.escape(h['cliente'])} · {html.escape(h['vertical'])} · {html.escape(h['periodo'])}</p></div>
<h2>Estado del mes</h2>
<p><span class="badge">{d['estado']}</span> &nbsp; Regla de negocio: ROAS &lt; {UMBRAL_CRITICO} Crítico · ROAS ≥ {UMBRAL_OPTIMO} Óptimo</p>
{nota_an}
<div class="dos"><div><h2>Indicadores del mes</h2><table>{t_kpi}</table></div>
<div><h2>Comparativo con el pasado (ROAS)</h2><table><tr><th>Referencia</th><th>ROAS</th><th>Variación actual</th></tr>{t_comp}</table></div></div>
<h2>Proyección del ROAS (modelo de aprendizaje automático)</h2>
{img}
<table><tr><th>Horizonte</th><th>ROAS proyectado</th><th>Rango 80 %</th><th>Estado proyectado</th><th>Prob. de deterioro</th></tr>{t_proy}</table>
<p class="nota">Las proyecciones son estimaciones con incertidumbre; el rango indica dónde se espera el ROAS en 8 de cada 10 casos.</p>
<h2>Resumen ejecutivo</h2><p>{html.escape(narrativa)}</p>
<h2>Recomendaciones</h2><ul>{recs}</ul>
<p class="nota">Fuente del texto: {html.escape(fuente)} · Cifras calculadas y verificadas por el sistema · Generado: {datetime.now():%d/%m/%Y %H:%M}</p>
<div class="hitl"><b>Validación Human-in-the-Loop:</b> {estado_val}<br>
Este reporte fue generado por un sistema de IA que actúa como copiloto. Las cifras se calculan de forma determinista,
las proyecciones provienen de un modelo validado y el texto requiere revisión y firma del Director de Cuenta.<br><br>
Firma: ______________________________ &nbsp; Fecha: ____/____/______</div>
</body></html>"""


def exportar(html_txt: str, ruta_sin_ext: str | Path) -> Path:
    ruta_sin_ext = Path(ruta_sin_ext)
    ruta_sin_ext.parent.mkdir(parents=True, exist_ok=True)
    ruta_html = ruta_sin_ext.with_suffix(".html")
    ruta_html.write_text(html_txt, encoding="utf-8")
    try:
        from weasyprint import HTML
        ruta_pdf = ruta_sin_ext.with_suffix(".pdf")
        HTML(string=html_txt).write_pdf(str(ruta_pdf))
        return ruta_pdf
    except Exception as e:
        print(f"[aviso] PDF no generado ({e.__class__.__name__}); se conserva {ruta_html.name}")
        return ruta_html


def pdf_bytes(html_txt: str) -> bytes | None:
    try:
        from weasyprint import HTML
        return HTML(string=html_txt).write_pdf()
    except Exception:
        return None

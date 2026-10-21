"""
Exportación del reporte ejecutivo (HTML -> PDF con WeasyPrint) con bloque de validación HITL.
Identidad visual Novar Agencia: negro, rojo #CC1417 y blanco.
"""
from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path

from src.agents.llm_narrative_generator import pct, usd

COLOR_ESTADO = {"Óptimo": "#1E7D3A", "Alerta": "#C98A00", "Crítico": "#CC1417"}


def generar_html(h: dict, d: dict, narrativa: str, fuente: str, validador: str = "",
                 aprobado: bool = False) -> str:
    color = COLOR_ESTADO.get(d["estado"], "#333")
    filas = [("Inversión", usd(h["inversion"])), ("Ingresos atribuidos", usd(h["ingresos"])),
             ("Beneficio neto", usd(h["beneficio"])), ("ROAS", f"{h['roas']:.2f}"),
             ("CPA", usd(h["cpa"])), ("Conversiones", f"{h['conversiones']:,}"),
             ("CTR", pct(h["ctr"])), ("Tasa de conversión", pct(h["cvr"])),
             ("CPM", usd(h["cpm"]))]
    tabla = "".join(f"<tr><td>{k}</td><td class='v'>{v}</td></tr>" for k, v in filas)
    recs = "".join(f"<li>{html.escape(r[0].upper() + r[1:])}.</li>" for r in d["recomendaciones"])
    estado_val = ("APROBADO por " + html.escape(validador)) if aprobado and validador else "PENDIENTE DE VALIDACIÓN HUMANA"
    return f"""<!doctype html><html lang="es"><head><meta charset="utf-8">
<style>
 @page {{ size: A4; margin: 18mm 16mm; }}
 body {{ font-family: Arial, Helvetica, sans-serif; color:#111; font-size:10.5pt; }}
 .top {{ background:#000; color:#fff; padding:14px 18px; border-bottom:5px solid #CC1417; }}
 .top h1 {{ margin:0; font-size:17pt; }} .top p {{ margin:4px 0 0; font-size:9pt; color:#ddd; }}
 .badge {{ display:inline-block; padding:5px 12px; border-radius:4px; color:#fff;
          background:{color}; font-weight:bold; }}
 table {{ width:100%; border-collapse:collapse; margin:10px 0; }}
 td {{ border-bottom:1px solid #e5e5e5; padding:6px 4px; }} td.v {{ text-align:right; font-weight:bold; }}
 h2 {{ font-size:12pt; border-left:4px solid #CC1417; padding-left:8px; margin-top:18px; }}
 .nota {{ font-size:8.5pt; color:#555; }}
 .hitl {{ margin-top:22px; border:1px dashed #999; padding:10px; font-size:9.5pt; }}
</style></head><body>
<div class="top"><h1>Reporte ejecutivo de desempeño publicitario</h1>
<p>Novar Agencia · Cuenta {html.escape(h['cliente'])} · {html.escape(h['periodo'])} · {html.escape(h['plataforma'])}</p></div>
<h2>Diagnóstico</h2>
<p><span class="badge">{d['estado']}</span> &nbsp; Confianza del modelo: {d['confianza']:.1f} %</p>
<h2>Indicadores clave (calculados por el sistema)</h2><table>{tabla}</table>
<h2>Resumen ejecutivo</h2><p>{html.escape(narrativa)}</p>
<h2>Recomendaciones</h2><ul>{recs}</ul>
<p class="nota">Factores determinantes (SHAP): {html.escape(', '.join(d['drivers']) or 'n/d')} ·
Fuente del texto: {html.escape(fuente)} · Generado: {datetime.now():%d/%m/%Y %H:%M}</p>
<div class="hitl"><b>Validación Human-in-the-Loop:</b> {estado_val}<br>
Este reporte fue generado por un sistema de IA que actúa como copiloto. Las cifras se calculan de
forma determinista y el texto requiere revisión y firma del Director de Cuenta antes de su envío.<br><br>
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
    except Exception as e:  # WeasyPrint sin librerías del sistema -> se conserva el HTML
        print(f"[aviso] PDF no generado ({e.__class__.__name__}); se conserva {ruta_html.name}")
        return ruta_html


def pdf_bytes(html_txt: str) -> bytes | None:
    try:
        from weasyprint import HTML
        return HTML(string=html_txt).write_pdf()
    except Exception:
        return None

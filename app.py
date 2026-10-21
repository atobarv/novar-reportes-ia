"""
Dashboard Streamlit — Sistema de Generación Inteligente de Reportes de Marketing (Novar Agencia).
Ejecutar:  streamlit run app.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))

from src.agents.llm_narrative_generator import (AgenteAnalista, AgenteExtractor, AgenteRedactor,
                                                plantilla_determinista, usd)
from src.explain.interpretabilidad import contribuciones_locales
from src.features.feature_engineering import FEATURES
from src.reports.exportador import generar_html, pdf_bytes

st.set_page_config(page_title="Novar · Reportes inteligentes", page_icon="📊", layout="wide")
st.markdown("""<style>
.block-container {padding-top: 1.2rem;}
h1, h2, h3 {color: #111;}
div[data-testid="stMetricValue"] {font-size: 1.4rem;}
.barra {background:#000;color:#fff;padding:12px 18px;border-bottom:5px solid #CC1417;border-radius:4px;}
</style>""", unsafe_allow_html=True)
st.markdown("<div class='barra'><b>NOVAR AGENCIA</b> · Sistema de Generación Inteligente de "
            "Reportes de Marketing (MVP · UDLA MIA)</div>", unsafe_allow_html=True)


@st.cache_resource
def cargar_artefactos():
    paquete = joblib.load(RAIZ / "models" / "modelo_rendimiento.joblib")
    datos = pd.read_csv(RAIZ / "data" / "processed" / "dataset_procesado.csv", parse_dates=["fecha"])
    return paquete, datos


@st.cache_resource
def cargar_redactor(usar_llm: bool):
    modelo = os.environ.get("NOVAR_LLM", "Qwen/Qwen2.5-1.5B-Instruct")
    return AgenteRedactor(model_id=modelo, usar_llm=usar_llm)


paquete, datos = cargar_artefactos()
pipeline = paquete["pipeline"]

with st.sidebar:
    st.header("Parámetros")
    cliente = st.selectbox("Cuenta (seudonimizada)", sorted(datos["cliente"].unique()))
    sub = datos[datos["cliente"] == cliente].sort_values("fecha")
    etiquetas = [f"{f:%d/%m/%Y} · {p}" if pd.notna(f) else f"Registro {i}"
                 for i, (f, p) in enumerate(zip(sub["fecha"], sub["plataforma"]))]
    sel = st.selectbox("Periodo", range(len(sub)), index=len(sub) - 1, format_func=lambda i: etiquetas[i])
    st.subheader("Simulación de escenarios")
    f_gasto = st.slider("Variación de inversión (%)", -50, 100, 0, 5)
    f_ingresos = st.slider("Variación de ingresos (%)", -50, 100, 0, 5)
    f_conv = st.slider("Variación de conversiones (%)", -50, 100, 0, 5)
    usar_llm = st.toggle("Redactar con LLM (requiere GPU)", value=False)

datos_sim = datos.copy()
idx_real = sub.index[sel]
datos_sim.loc[idx_real, "gasto_usd"] *= 1 + f_gasto / 100
datos_sim.loc[idx_real, "ingresos_usd"] *= 1 + f_ingresos / 100
datos_sim.loc[idx_real, "conversiones"] = round(datos_sim.loc[idx_real, "conversiones"] * (1 + f_conv / 100))

extractor = AgenteExtractor(datos_sim)
hechos, fila = extractor.consolidar(cliente, idx_real)
diag = AgenteAnalista(pipeline).diagnosticar(hechos, fila)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Inversión", usd(hechos["inversion"]))
c2.metric("Ingresos", usd(hechos["ingresos"]))
c3.metric("ROAS", f"{hechos['roas']:.2f}",
          None if hechos["var_roas"] is None else f"{hechos['var_roas']:+.1f} %")
c4.metric("CPA", usd(hechos["cpa"]),
          None if hechos["var_cpa"] is None else f"{hechos['var_cpa']:+.1f} %", delta_color="inverse")
c5.metric("Conversiones", f"{hechos['conversiones']:,}")

color = {"Óptimo": "#1E7D3A", "Alerta": "#C98A00", "Crítico": "#CC1417"}[diag["estado"]]
st.markdown(f"### Estado: <span style='color:{color}'>{diag['estado']}</span> "
            f"<small>(confianza {diag['confianza']:.1f} %)</small>", unsafe_allow_html=True)

izq, der = st.columns([1.1, 1])
with izq:
    st.subheader("¿Por qué? Contribuciones SHAP")
    _, contrib = contribuciones_locales(pipeline, fila[FEATURES], list(pipeline.classes_))
    st.bar_chart(contrib.head(6).set_index("feature")["shap"], color="#CC1417")
    st.caption("Valores positivos empujan hacia el estado predicho; negativos lo alejan.")
with der:
    st.subheader("Tendencia del ROAS")
    hist = extractor.df[extractor.df["cliente"] == cliente].set_index("fecha")["roas"]
    st.line_chart(hist, color="#111111")

st.subheader("Resumen ejecutivo")
if st.button("Generar reporte", type="primary"):
    with st.spinner("Redactando y verificando cifras…"):
        res = cargar_redactor(usar_llm).generar(hechos, diag)
    st.session_state["reporte"] = res
res = st.session_state.get("reporte")
if res:
    st.write(res["texto"])
    st.caption(f"Fuente: {res['fuente']} · Latencia LLM: {res['latencia_s']:.2f} s · "
               f"Guardrail numérico: {'superado' if res['fiel_crudo'] in (True, None) else 'corrigió el texto'}")
    for r in diag["recomendaciones"]:
        st.markdown(f"- {r[0].upper() + r[1:]}.")
    st.divider()
    st.subheader("Validación humana (HITL)")
    validador = st.text_input("Director de Cuenta que valida")
    aprobado = st.checkbox("Revisé cifras y redacción; apruebo el envío al cliente.")
    html_txt = generar_html(hechos, diag, res["texto"], res["fuente"], validador, aprobado)
    if aprobado and validador.strip():
        pdf = pdf_bytes(html_txt)
        if pdf:
            st.download_button("Descargar PDF", pdf, file_name=f"reporte_{cliente}.pdf",
                               mime="application/pdf")
        st.download_button("Descargar HTML", html_txt, file_name=f"reporte_{cliente}.html",
                           mime="text/html")
    else:
        st.info("La descarga se habilita tras la validación del Director de Cuenta.")
else:
    st.caption("Vista previa determinista: " + plantilla_determinista(hechos, diag))

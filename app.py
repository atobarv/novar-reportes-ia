"""
Dashboard Streamlit — Sistema de Generación Inteligente de Reportes de Marketing (Novar Agencia) · v2.2
Ejecutar:  streamlit run app.py

Pestañas: estado y comparativo (determinista) · proyección 3/6/12 meses (ML) · anomalías (ML) ·
reporte con validación humana y registro de la prueba operativa (tiempos, correcciones y utilidad).
"""
from __future__ import annotations

import difflib
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))

from src.agents.llm_narrative_generator import (AgenteAnalista, AgenteExtractor, AgentePronostico,  # noqa: E402
                                                AgenteRedactor, nombre_mes, pct, usd)
from src.etl.ingesta_marketing import anonimizar, cargar_archivo, limpiar, sal_configurada  # noqa: E402
from src.features.feature_engineering import preparar_dataset  # noqa: E402
from src.models.anomalias import caracteristicas, imputar_anomalias  # noqa: E402
from src.models.pronostico import MIN_HIST  # noqa: E402
from src.reports.exportador import generar_html, grafico_proyeccion, pdf_bytes  # noqa: E402
from src.rules.reglas_negocio import UMBRAL_CRITICO, UMBRAL_OPTIMO  # noqa: E402

REGISTRO = Path(os.environ.get("NOVAR_REGISTRO", RAIZ / "data" / "prueba_operativa" / "registro_prueba_operativa.csv"))
COLOR = {"Óptimo": "#1E7D3A", "Alerta": "#C98A00", "Crítico": "#CC1417"}

st.set_page_config(page_title="Novar · Reportes inteligentes", page_icon="📊", layout="wide")
st.markdown("""<style>
.block-container {padding-top: 1.1rem;}
div[data-testid="stMetricValue"] {font-size: 1.35rem;}
.barra {background:#000;color:#fff;padding:11px 18px;border-bottom:5px solid #CC1417;border-radius:4px;}
.badge {color:#fff;padding:3px 10px;border-radius:4px;font-weight:bold;}
</style>""", unsafe_allow_html=True)
st.markdown("<div class='barra'><b>NOVAR AGENCIA</b> · Sistema de Generación Inteligente de Reportes de Marketing "
            "· v2.2 (UDLA MIA)</div>", unsafe_allow_html=True)


# --------------------------------------------------------------------------------------------- carga
@st.cache_resource
def artefactos():
    return (joblib.load(RAIZ / "models" / "pronosticador_roas.joblib"),
            joblib.load(RAIZ / "models" / "detector_anomalias.joblib"))


@st.cache_data
def datos_proyecto():
    return pd.read_csv(RAIZ / "data" / "processed" / "dataset_procesado.csv", parse_dates=["fecha"])


@st.cache_resource
def redactor(usar_llm: bool):
    return AgenteRedactor(os.environ.get("NOVAR_LLM", "Qwen/Qwen2.5-1.5B-Instruct"), usar_llm=usar_llm)


def preparar(df: pd.DataFrame, detector):
    F = detector.puntuar(caracteristicas(df))
    df_modelo = imputar_anomalias(df, F["es_anomalia"])
    return F, df_modelo


pronosticador, detector = artefactos()

with st.sidebar:
    st.header("Datos")
    fuente = st.radio("Origen", ["Dataset del proyecto", "Cargar exportación (CSV/Excel)"])
    if fuente.startswith("Cargar"):
        archivo = st.file_uploader("Exportación mensual por cuenta", type=["csv", "xlsx", "xls"])
        st.caption("Columnas mínimas: fecha/periodo, cliente/cuenta, gasto, impresiones, clics, conversiones, "
                   f"ingresos. Para proyectar se requieren al menos {MIN_HIST} meses por cuenta.")
        if archivo is None:
            st.info("Sube un archivo para continuar.")
            st.stop()
        if "t_carga" not in st.session_state or st.session_state.get("archivo") != archivo.name:
            st.session_state.update(t_carga=time.time(), archivo=archivo.name)
        tmp = RAIZ / "data" / "cargas" / archivo.name
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_bytes(archivo.getvalue())
        try:
            crudo, calidad = limpiar(cargar_archivo(tmp))
        except Exception as e:  # error controlado: mensaje claro en lugar de un fallo de la aplicación
            st.error(f"No se pudo procesar el archivo: {e}")
            st.stop()
        for aviso in calidad.get("advertencias", []):
            st.warning(aviso)
        if st.checkbox("Seudonimizar cuentas (LOPDP)", value=True):
            try:
                crudo = anonimizar(crudo, sal_configurada())
            except ValueError as e:      # sin sal secreta no se seudonimiza con una sal pública
                st.error(str(e))
                st.stop()
        df, _ = preparar_dataset(crudo)
        st.success(f"{len(df)} registros · {df['cliente'].nunique()} cuentas")
    else:
        df = datos_proyecto()
    F, df_modelo = preparar(df, detector)
    cuenta = st.selectbox("Cuenta", sorted(df["cliente"].unique()))
    fechas = sorted(pd.to_datetime(df[df["cliente"] == cuenta]["fecha"]))
    fecha = st.selectbox("Mes analizado", fechas, index=len(fechas) - 1, format_func=nombre_mes)
    st.divider()
    usar_llm = st.toggle("Redactar con LLM (requiere GPU)", value=False)
    director = st.text_input("Director de Cuenta (para la prueba operativa)")

extractor = AgenteExtractor(df)
h = extractor.consolidar(cuenta, fecha)
proy = AgentePronostico(pronosticador, df_modelo).proyectar(cuenta, fecha)
diag = AgenteAnalista(F).diagnosticar(h, proy)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Estado y comparativo", "Proyección 3 · 6 · 12 meses (ML)",
                                        "Anomalías (ML)", "Reporte y validación", "Registro de la prueba operativa"])

# --------------------------------------------------------------------------------------------- 1
with tab1:
    st.markdown(f"#### {cuenta} · {h['periodo']} &nbsp; <span class='badge' style='background:{COLOR[diag['estado']]}'>"
                f"{diag['estado']}{' (provisional)' if diag.get('provisional') else ''}</span>", unsafe_allow_html=True)
    if h.get("dato_imputado"):
        st.warning("El mes contiene valores faltantes imputados: el estado es provisional hasta completar los datos.")
    st.caption(f"Estado calculado con la regla de negocio (sin ML): ROAS < {UMBRAL_CRITICO} Crítico · "
               f"ROAS ≥ {UMBRAL_OPTIMO} Óptimo.")
    c = st.columns(5)
    c[0].metric("Inversión", usd(h["inversion"]))
    c[1].metric("Ingresos", usd(h["ingresos"]))
    c[2].metric("ROAS", f"{h['roas']:.2f}", None if h["var_roas"] is None else f"{h['var_roas']:+.1f} % vs mes ant.")
    c[3].metric("CPA", usd(h["cpa"]), None if h["var_cpa"] is None else f"{h['var_cpa']:+.1f} %", delta_color="inverse")
    c[4].metric("Conversiones", f"{h['conversiones']:,}")
    st.subheader("Comparativo con el pasado")
    comp = pd.DataFrame([
        ["Mes anterior", h["roas_mes_anterior"], h["var_roas"]],
        ["Mismo mes del año anterior", h["roas_hace_12m"], h["var_roas_anual"]],
        ["Promedio de los últimos 12 meses", h["roas_promedio_12m"], float("nan")]],
        columns=["Referencia", "ROAS", "Variación del mes actual (%)"])
    st.dataframe(comp.style.format({"ROAS": "{:.2f}", "Variación del mes actual (%)": "{:+.1f}"}, na_rep="—"),
                 hide_index=True, width="stretch")
    hist = extractor.historia(cuenta, fecha, 24).set_index("fecha")[["roas"]].rename(columns={"roas": "ROAS"})
    st.line_chart(hist, color="#111111")
    if diag["anomalia"].get("es_anomalia"):
        st.warning(f"Patrón anómalo detectado en {diag['anomalia'].get('metrica_principal')}: verifique la medición.")

# --------------------------------------------------------------------------------------------- 2
with tab2:
    if not proy:
        st.info(f"Se necesitan al menos {MIN_HIST} meses de historia para proyectar.")
    else:
        posteriores = extractor.df[(extractor.df["cliente"] == cuenta) &
                                   (pd.to_datetime(extractor.df["fecha"]) > pd.to_datetime(fecha))]
        ver = st.checkbox("Mostrar lo que ocurrió después (validación visual)", value=len(posteriores) > 0,
                          disabled=len(posteriores) == 0)
        fig = grafico_proyeccion(extractor.historia(cuenta, fecha, 24), proy,
                                 posteriores.head(12) if ver else None, ancho=10, alto=3.6)
        st.pyplot(fig)
        tabla = pd.DataFrame([{"Horizonte": f"{k} meses", "Mes": p["periodo"], "ROAS proyectado": p["roas"],
                               "Rango 80 %": f"{p['inf']:.2f} – {p['sup']:.2f}", "Estado proyectado": p["estado"],
                               "Prob. de deterioro": f"{100 * p['prob_deterioro']:.0f} %",
                               "Efecto estacional histórico": f"{p['efecto_estacional']:+.1f} %",
                               "Tendencia anual": f"{p['tendencia_anual']:+.1f} %"} for k, p in sorted(proy.items())])
        st.dataframe(tabla.style.format({"ROAS proyectado": "{:.2f}"}), hide_index=True, width="stretch")
        st.caption(f"Modelo: {pronosticador.nombre}. Validado con cuentas y meses no vistos. El rango cubre el 80 % "
                   "de los casos según los errores fuera de muestra.")

# --------------------------------------------------------------------------------------------- 3
with tab3:
    Fa = F[F["cliente"] == cuenta][["fecha", "puntaje_anomalia", "es_anomalia", "metrica_principal"]].copy()
    st.write(f"Meses marcados como anómalos en la cuenta: **{int(Fa['es_anomalia'].sum())}** "
             f"(umbral del modelo: {detector.umbral:.3f}).")
    st.line_chart(Fa.set_index("fecha")[["puntaje_anomalia"]], color="#CC1417")
    st.dataframe(Fa[Fa["es_anomalia"]].assign(fecha=lambda x: x["fecha"].map(nombre_mes))
                 .rename(columns={"fecha": "Mes", "puntaje_anomalia": "Puntaje", "metrica_principal": "Métrica principal"})
                 .drop(columns="es_anomalia"), hide_index=True, width="stretch")

# --------------------------------------------------------------------------------------------- 4
def _palabras(t):
    return re.findall(r"\S+", t)


def contar_correcciones(original: str, final: str) -> dict:
    a, b = _palabras(original), _palabras(final)
    cambios = sum(max(i2 - i1, j2 - j1) for op, i1, i2, j1, j2 in
                  difflib.SequenceMatcher(None, a, b).get_opcodes() if op != "equal")
    num = lambda t: re.findall(r"[-+]?\d[\d.,]*", t)  # noqa: E731
    cifras = len(set(num(original)) ^ set(num(final)))
    ora = lambda t: {s.strip() for s in re.split(r"(?<=[.!?])\s+", t) if s.strip()}  # noqa: E731
    return {"palabras_cambiadas": cambios, "cifras_cambiadas": cifras,
            "oraciones_editadas": len(ora(final) - ora(original))}


with tab4:
    st.caption("Prueba operativa: el cronómetro inicia al pulsar «Iniciar preparación» (o al cargar un archivo) "
               "y termina con la aprobación del Director de Cuenta.")
    c1, c2 = st.columns([1, 2])
    min_export = c1.number_input("Minutos para exportar datos de las plataformas (fuera del sistema)",
                                 min_value=0.0, step=1.0, value=0.0)
    if c2.button("Iniciar preparación (cronómetro)"):
        st.session_state["t_inicio"] = time.time()
    t_inicio = st.session_state.get("t_inicio", st.session_state.get("t_carga"))
    if t_inicio:
        c2.write(f"⏱️ Cronómetro iniciado a las {datetime.fromtimestamp(t_inicio):%H:%M:%S}")
    if st.button("Generar reporte", type="primary"):
        t0 = time.time()
        with st.spinner("Calculando, proyectando y redactando…"):
            res = redactor(usar_llm).generar(h, diag, proy)
        st.session_state.update(reporte=res, t_gen_ini=t0, t_gen_fin=time.time(), clave=(cuenta, str(fecha)))
    res = st.session_state.get("reporte")
    if res and st.session_state.get("clave") == (cuenta, str(fecha)):
        st.caption(f"Fuente del texto: {res['fuente']} · Guardrail: "
                   f"{'sin observaciones' if not res['fallas'] else '; '.join(res['fallas'])} · "
                   f"generación: {st.session_state['t_gen_fin'] - st.session_state['t_gen_ini']:.1f} s")
        texto_final = st.text_area("Resumen ejecutivo (el Director de Cuenta puede corregirlo)", res["texto"],
                                   height=170)
        st.markdown("**Recomendaciones generadas: ¿son útiles?**")
        utiles = [st.radio(r[0].upper() + r[1:], ["Útil", "No útil"], horizontal=True, key=f"rec{i}")
                  for i, r in enumerate(diag["recomendaciones"])]
        utilidad = st.slider("Utilidad general del reporte (1 = nada útil, 5 = muy útil)", 1, 5, 4)
        comentarios = st.text_input("Comentarios del Director de Cuenta")
        corr = contar_correcciones(res["texto"], texto_final)
        st.write(f"Correcciones detectadas: **{corr['palabras_cambiadas']}** palabras · "
                 f"**{corr['cifras_cambiadas']}** cifras · **{corr['oraciones_editadas']}** oraciones.")
        aprobado = st.checkbox("Revisé cifras, proyección y redacción; apruebo el envío al cliente.")
        if aprobado and director.strip():
            html_txt = generar_html(h, diag, proy, texto_final, res["fuente"], extractor.historia(cuenta, fecha, 24),
                                    director, True, corr["palabras_cambiadas"])
            if st.button("Registrar aprobación en la prueba operativa"):
                t_fin = time.time()
                ini = t_inicio or st.session_state["t_gen_ini"]
                fila = {"fecha_registro": datetime.now().isoformat(timespec="seconds"), "director": director,
                        "cuenta": cuenta, "mes_analizado": nombre_mes(fecha), "fuente_texto": res["fuente"],
                        "min_exportacion": min_export,
                        "min_preparacion_en_sistema": (st.session_state["t_gen_ini"] - ini) / 60,
                        "seg_generacion": st.session_state["t_gen_fin"] - st.session_state["t_gen_ini"],
                        "min_revision": (t_fin - st.session_state["t_gen_fin"]) / 60,
                        "min_total": min_export + (t_fin - ini) / 60, **corr,
                        "recomendaciones_utiles": utiles.count("Útil"), "recomendaciones_total": len(utiles),
                        "utilidad_1a5": utilidad, "aprobado": True, "comentarios": comentarios,
                        "texto_generado": res["texto"], "texto_aprobado": texto_final}
                REGISTRO.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame([fila]).to_csv(REGISTRO, mode="a", header=not REGISTRO.exists(), index=False,
                                            encoding="utf-8")
                st.success(f"Registrado: {fila['min_total']:.1f} minutos en total · "
                           f"{corr['palabras_cambiadas']} palabras corregidas.")
                st.session_state.pop("t_inicio", None)
            pdf = pdf_bytes(html_txt)
            if pdf:
                st.download_button("Descargar PDF aprobado", pdf, file_name=f"reporte_{cuenta}_{fecha:%Y-%m}.pdf",
                                   mime="application/pdf")
            st.download_button("Descargar HTML", html_txt, file_name=f"reporte_{cuenta}_{fecha:%Y-%m}.html",
                               mime="text/html")
        elif aprobado:
            st.info("Escribe el nombre del Director de Cuenta en la barra lateral para registrar la aprobación.")

# --------------------------------------------------------------------------------------------- 5
with tab5:
    if REGISTRO.exists():
        R = pd.read_csv(REGISTRO)
        c = st.columns(5)
        c[0].metric("Reportes aprobados", len(R))
        c[1].metric("Minutos totales (media)", f"{R['min_total'].mean():.1f}")
        c[2].metric("Palabras corregidas (media)", f"{R['palabras_cambiadas'].mean():.1f}")
        c[3].metric("Recomendaciones útiles", f"{100 * R['recomendaciones_utiles'].sum() / max(1, R['recomendaciones_total'].sum()):.0f} %")
        c[4].metric("Utilidad media (1–5)", f"{R['utilidad_1a5'].mean():.1f}")
        st.dataframe(R.drop(columns=["texto_generado", "texto_aprobado"]), hide_index=True, width="stretch")
        st.download_button("Descargar registro (CSV)", REGISTRO.read_bytes(), file_name=REGISTRO.name, mime="text/csv")
    else:
        st.info("Aún no hay reportes registrados. Genera, revisa y aprueba reportes en la pestaña anterior.")

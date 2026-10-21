"""
Módulo ETL — Ingesta, normalización, validación y anonimización.
Proyecto: Sistema de Generación Inteligente de Reportes de Marketing (Novar Agencia · UDLA MIA).
Cumplimiento: Ley Orgánica de Protección de Datos Personales (LOPDP, Ecuador) — los
identificadores de clientes se seudonimizan con SHA-256 + sal antes de cualquier modelado.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Esquema canónico y alias aceptados (exportaciones de Meta Ads, Google Ads,
# GA4 y CRM suelen traer nombres distintos para la misma métrica).
# ---------------------------------------------------------------------------
ALIAS = {
    "fecha": ["fecha", "date", "periodo", "mes", "semana", "day", "reporting_starts"],
    "cliente": ["cliente", "client", "cuenta", "account", "account_name", "cliente_id", "marca"],
    "plataforma": ["plataforma", "platform", "canal", "fuente", "source"],
    "gasto_usd": ["gasto_usd", "gasto", "spend", "amount_spent", "amount_spent_usd", "inversion",
                  "costo", "cost", "importe_gastado", "importe_gastado_usd"],
    "impresiones": ["impresiones", "impressions", "impr"],
    "clics": ["clics", "clicks", "clicks_all", "link_clicks", "clic", "clics_en_el_enlace"],
    "conversiones": ["conversiones", "conversions", "results", "resultados", "leads", "compras",
                     "purchases"],
    "ingresos_usd": ["ingresos_usd", "ingresos", "revenue", "conversion_value", "valor_conversion",
                     "ventas", "purchase_value", "valor_de_conversion"],
    "estado_rendimiento": ["estado_rendimiento", "rendimiento", "estado", "label", "target", "clase",
                           "status"],
    "vertical": ["vertical", "industria", "sector", "rubro", "categoria_negocio"],
    "referencia_experta": ["diagnostico_historico_referencia", "diagnostico_historico", "diagnostico",
                           "referencia", "resumen_referencia"],
}
# Columnas que codifican la etiqueta y producirían fuga de información (data leakage)
FUGA = {"status_code", "codigo_estado", "label_id", "target_code", "estado_codigo"}
NUMERICAS = ["gasto_usd", "impresiones", "clics", "conversiones", "ingresos_usd"]
OBLIGATORIAS = ["cliente"] + NUMERICAS


def _norm(texto: str) -> str:
    texto = unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", texto.lower()).strip("_")


def cargar_archivo(ruta: str | Path) -> pd.DataFrame:
    """Lee CSV (coma o punto y coma) o Excel."""
    ruta = Path(ruta)
    if ruta.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(ruta)
    df = pd.read_csv(ruta, sep=None, engine="python", encoding="utf-8-sig")
    return df


def normalizar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    inverso = {_norm(a): canon for canon, alias in ALIAS.items() for a in alias}
    renombre, usados = {}, set()
    for col in df.columns:
        canon = inverso.get(_norm(col))
        if canon and canon not in usados:
            renombre[col] = canon
            usados.add(canon)
    return df.rename(columns=renombre)


def _a_numero(serie: pd.Series) -> pd.Series:
    if serie.dtype.kind in "if":
        return serie.astype(float)
    s = serie.astype(str).str.replace(r"[^\d,.\-]", "", regex=True)
    # "1.234,56" -> 1234.56 ; "1,234.56" -> 1234.56 ; "12,5" -> 12.5
    def conv(x: str):
        if not x or x in {"-", ".", ","}:
            return np.nan
        if "," in x and "." in x:
            x = x.replace(".", "").replace(",", ".") if x.rfind(",") > x.rfind(".") else x.replace(",", "")
        elif "," in x:
            partes = x.split(",")
            x = x.replace(",", "") if (len(partes) > 2 or len(partes[-1]) == 3) else x.replace(",", ".")
        try:
            return float(x)
        except ValueError:
            return np.nan
    return s.map(conv)


def limpiar(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Normaliza tipos, elimina duplicados y registra un reporte de calidad."""
    df = normalizar_columnas(df.copy())
    faltantes = [c for c in OBLIGATORIAS if c not in df.columns]
    if faltantes:
        raise ValueError(f"Columnas obligatorias ausentes tras normalizar: {faltantes}. "
                         f"Columnas recibidas: {list(df.columns)}")
    calidad = {"filas_entrada": int(len(df))}
    fugas = [c for c in df.columns if _norm(c) in FUGA]
    calidad["columnas_excluidas_por_fuga"] = fugas
    df = df.drop(columns=fugas)
    for c in NUMERICAS:
        df[c] = _a_numero(df[c])
    calidad["nulos_numericos"] = {c: int(df[c].isna().sum()) for c in NUMERICAS}
    calidad["valores_negativos"] = {c: int((df[c] < 0).sum()) for c in NUMERICAS}
    df[NUMERICAS] = df[NUMERICAS].clip(lower=0).fillna(0)
    if "fecha" in df.columns:
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce", dayfirst=False)
    else:
        df["fecha"] = pd.NaT
    if "plataforma" not in df.columns:
        df["plataforma"] = "No especificada"
    df["cliente"] = df["cliente"].astype(str).str.strip()
    antes = len(df)
    df = df.drop_duplicates()
    calidad["duplicados_eliminados"] = int(antes - len(df))
    sin_actividad = (df["gasto_usd"] <= 0) | (df["impresiones"] <= 0)
    calidad["filas_sin_actividad_eliminadas"] = int(sin_actividad.sum())
    df = df.loc[~sin_actividad].reset_index(drop=True)
    calidad["filas_salida"] = int(len(df))
    calidad["clientes"] = int(df["cliente"].nunique())
    calidad["plataformas"] = sorted(df["plataforma"].astype(str).unique().tolist())
    return df, calidad


def anonimizar(df: pd.DataFrame, sal: str, ruta_mapeo: str | Path | None = None) -> pd.DataFrame:
    """Seudonimiza el nombre del cliente (SHA-256 + sal). El mapeo se guarda fuera del repo."""
    df = df.copy()
    if df["cliente"].str.match(r"^CLI-[0-9a-f]{8}$").all():
        return df  # ya anonimizado
    mapeo = {n: "CLI-" + hashlib.sha256((sal + n).encode()).hexdigest()[:8]
             for n in sorted(df["cliente"].unique())}
    df["cliente"] = df["cliente"].map(mapeo)
    if ruta_mapeo:
        Path(ruta_mapeo).parent.mkdir(parents=True, exist_ok=True)
        Path(ruta_mapeo).write_text(json.dumps(mapeo, ensure_ascii=False, indent=2), encoding="utf-8")
    return df


def generar_dataset_sintetico(n_clientes: int = 7, n_periodos: int = 26, seed: int = 42) -> pd.DataFrame:
    """Dataset de respaldo (7 clientes x 26 semanas = 182 registros) con la misma
    estructura que las exportaciones reales. Solo se usa si no se encuentra el CSV real."""
    rng = np.random.default_rng(seed)
    segmentos = ["pyme", "pyme", "pyme", "mediana", "mediana", "grande", "grande"][:n_clientes]
    base_gasto = {"pyme": (60, 160), "mediana": (300, 750), "grande": (1200, 3000)}
    plat_param = {  # cpm (min,max), ctr (min,max)
        "Meta Ads": ((4, 9), (0.009, 0.022)),
        "Google Ads": ((10, 26), (0.03, 0.07)),
        "TikTok Ads": ((3, 6), (0.006, 0.015)),
    }
    filas = []
    inicio = pd.Timestamp("2026-01-05")
    for i, seg in enumerate(segmentos):
        nombre = f"Cliente_{i + 1:02d}"
        plataforma_pref = rng.choice(list(plat_param), p=[0.55, 0.3, 0.15])
        calidad = rng.uniform(0.45, 1.7)           # calidad comercial de la cuenta
        ticket = rng.uniform(25, 110)              # valor promedio por conversión
        g_min, g_max = base_gasto[seg]
        for p in range(n_periodos):
            plataforma = plataforma_pref if rng.random() < 0.75 else rng.choice(list(plat_param))
            (c0, c1), (t0, t1) = plat_param[plataforma]
            estacional = 2.4 if p in (20, 21) else (1.3 if p in (19, 22) else 1.0)  # picos tipo Black Friday
            gasto = rng.uniform(g_min, g_max) * estacional * rng.lognormal(0, 0.12)
            cpm = rng.uniform(c0, c1) * (1.25 if estacional > 1 else 1.0)
            impresiones = gasto / cpm * 1000
            clics = impresiones * rng.uniform(t0, t1)
            cvr = np.clip(rng.normal(0.016 * calidad, 0.006), 0.0, 0.2)
            conversiones = rng.binomial(int(clics), cvr) if clics >= 1 else 0
            ingresos = conversiones * ticket * rng.lognormal(0, 0.25)
            filas.append({
                "fecha": (inicio + pd.Timedelta(weeks=p)).date().isoformat(),
                "cliente": nombre, "segmento": seg, "plataforma": plataforma,
                "gasto_usd": round(gasto, 2), "impresiones": int(impresiones), "clics": int(clics),
                "conversiones": int(conversiones), "ingresos_usd": round(ingresos, 2),
            })
    df = pd.DataFrame(filas)
    # Etiqueta experta (regla del Director de Cuenta) + 2 % de ruido de criterio humano
    roas = df["ingresos_usd"] / df["gasto_usd"]
    profit = df["ingresos_usd"] - df["gasto_usd"]
    estado = np.where((roas < 1.2) | (df["conversiones"] == 0), "Crítico",
                      np.where((roas >= 2.8) & (profit >= 50), "Óptimo", "Alerta"))
    ruido = rng.random(len(df)) < 0.02
    estado[ruido] = rng.choice(["Crítico", "Alerta", "Óptimo"], ruido.sum())
    df["estado_rendimiento"] = estado
    return df


def huella_dataset(ruta: str | Path) -> dict:
    """Huella SHA-256 para trazabilidad/versionado de datos (complementa DVC)."""
    contenido = Path(ruta).read_bytes()
    return {"archivo": Path(ruta).name, "sha256": hashlib.sha256(contenido).hexdigest(),
            "bytes": len(contenido)}

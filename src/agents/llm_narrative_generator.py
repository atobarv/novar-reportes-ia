"""
Módulo generativo multi-agente con guardrails deterministas.

  Agente Extractor  -> consolida KPIs del cliente/periodo (aritmética en Python, inmutable)
  Agente Analista   -> clasifica el rendimiento (Random Forest) + factores SHAP + reglas de acción
  Agente Redactor   -> redacta el resumen ejecutivo con Qwen2.5-1.5B-Instruct (one-shot)
  Guardrail         -> (1) toda cifra del texto debe existir en los datos verificados y
                       (2) cada cifra debe estar atribuida a la métrica correcta y con su unidad.
                       Si falla, reintenta con instrucción estricta y, en último caso, usa una
                       plantilla determinista.
El sistema es un copiloto: todo reporte requiere validación humana (HITL) antes del envío.
"""
from __future__ import annotations

import re
import time

import numpy as np
import pandas as pd

from src.features.feature_engineering import FEATURES, calcular_kpis

NOMBRES_FEATURE = {
    "roas": "ROAS", "profit_usd": "beneficio neto", "cpa_usd": "CPA", "cvr": "tasa de conversión",
    "ctr": "CTR", "cpm_usd": "CPM", "gasto_usd": "inversión", "ingresos_usd": "ingresos",
    "conversiones": "conversiones", "clics": "clics", "impresiones": "impresiones",
}
MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre",
         "octubre", "noviembre", "diciembre"]


def usd(x: float) -> str:
    return f"${x:,.2f}"


def pct(x: float, signo: bool = False) -> str:
    return f"{x:+.1f} %" if signo else f"{x:.2f} %"


# ---------------------------------------------------------------------------
class AgenteExtractor:
    def __init__(self, df: pd.DataFrame):
        # Se conserva el índice original para poder ubicar cualquier registro del dataset
        self.df = calcular_kpis(df).sort_values(["cliente", "fecha"], kind="stable")
        fechas = pd.to_datetime(self.df["fecha"], errors="coerce")
        self.mensual = bool(fechas.notna().all() and (fechas.dt.day == 1).all())

    def _periodo(self, fecha) -> str:
        if pd.isna(fecha):
            return "el periodo actual"
        f = pd.to_datetime(fecha)
        return f"{MESES[f.month - 1]} de {f.year}" if self.mensual else f"la semana del {f:%d/%m/%Y}"

    def consolidar(self, cliente: str, indice: int | None = None) -> tuple[dict, pd.DataFrame]:
        sub = self.df[self.df["cliente"] == cliente].reset_index()
        if sub.empty:
            raise ValueError(f"Cliente no encontrado: {cliente}")
        pos = len(sub) - 1 if indice is None else int(np.where(sub["index"] == indice)[0][0])
        act = sub.iloc[pos]
        prev = sub.iloc[pos - 1] if pos > 0 else None

        def var(campo):
            if prev is None or prev[campo] == 0:
                return None
            return 100 * (act[campo] - prev[campo]) / abs(prev[campo])

        hechos = {
            "cliente": act["cliente"], "periodo": self._periodo(act["fecha"]),
            "unidad_periodo": "mes" if self.mensual else "periodo",
            "plataforma": str(act.get("plataforma", "No especificada")),
            "vertical": str(act["vertical"]) if "vertical" in act.index else None,
            "inversion": float(act["gasto_usd"]), "ingresos": float(act["ingresos_usd"]),
            "conversiones": int(act["conversiones"]), "clics": int(act["clics"]),
            "impresiones": int(act["impresiones"]), "roas": float(act["roas"]),
            "cpa": float(act["cpa_usd"]), "ctr": float(act["ctr"]), "cvr": float(act["cvr"]),
            "beneficio": float(act["profit_usd"]), "cpm": float(act["cpm_usd"]),
            "var_roas": var("roas"), "var_cpa": var("cpa_usd"), "var_conversiones": var("conversiones"),
            "var_inversion": var("gasto_usd"),
            "referencia_experta": (str(act["referencia_experta"]) if "referencia_experta" in act.index
                                   and pd.notna(act["referencia_experta"]) else None),
        }
        fila = pd.DataFrame([act[FEATURES].astype(float).values], columns=FEATURES)
        return hechos, fila


# ---------------------------------------------------------------------------
class AgenteAnalista:
    def __init__(self, pipeline, features: list[str] = FEATURES):
        self.pipeline = pipeline
        self.features = features

    def diagnosticar(self, hechos: dict, fila: pd.DataFrame, con_shap: bool = True) -> dict:
        proba = self.pipeline.predict_proba(fila[self.features])[0]
        clases = list(self.pipeline.classes_)
        estado = clases[int(np.argmax(proba))]
        drivers = []
        if con_shap:
            try:
                from src.explain.interpretabilidad import contribuciones_locales
                _, contrib = contribuciones_locales(self.pipeline, fila[self.features], clases)
                drivers = [NOMBRES_FEATURE.get(f, f) for f in contrib["feature"].head(3)]
            except Exception:
                drivers = []
        return {"estado": estado, "confianza": float(np.max(proba)) * 100,
                "probabilidades": dict(zip(clases, map(float, proba))),
                "drivers": drivers, "recomendaciones": self._recomendaciones(hechos, estado)}

    @staticmethod
    def _recomendaciones(h: dict, estado: str) -> list[str]:
        r = []
        if h["conversiones"] == 0 or h["roas"] < 1.2 or estado == "Crítico":
            r.append("pausar o reestructurar los conjuntos de anuncios con retorno inferior a la inversión")
        if h["var_cpa"] is not None and h["var_cpa"] > 15:
            r.append("revisar segmentación y creatividades, porque el CPA aumentó frente al periodo anterior")
        if h["ctr"] < 1.0:
            r.append("renovar creatividades y mensajes para recuperar el CTR")
        if h["cvr"] < 2.0 and h["clics"] > 0:
            r.append("optimizar la página de destino y el formulario para mejorar la tasa de conversión")
        if estado == "Óptimo":
            r.append("escalar el presupuesto de forma gradual manteniendo el control del CPA")
        if not r:
            r.append("mantener la estrategia actual y monitorear la tendencia del ROAS")
        return r[:3]


# ---------------------------------------------------------------------------
# Guardrail numérico
_NUM = re.compile(r"(?<![\w.\-])[-+]?\d[\d.,]*(?![\w])")
_LISTA = re.compile(r"(?m)^\s*\d+[.)]\s+")


def _candidatos(token: str) -> set[float]:
    t = token.strip(".,").lstrip("+-")
    if not t or not t[0].isdigit():
        return set()
    cands = set()

    def intentar(s):
        try:
            cands.add(abs(float(s)))
        except ValueError:
            pass
    if "," in t and "." in t:
        intentar(t.replace(".", "").replace(",", ".") if t.rfind(",") > t.rfind(".") else t.replace(",", ""))
    elif "," in t:
        partes = t.split(",")
        if len(partes) > 2:
            intentar(t.replace(",", ""))
        else:
            intentar(t.replace(",", "."))
            if len(partes[1]) == 3:
                intentar(t.replace(",", ""))
    elif "." in t:
        partes = t.split(".")
        if len(partes) > 2:
            intentar(t.replace(".", ""))
        else:
            intentar(t)
            if len(partes[1]) == 3:
                intentar(t.replace(".", ""))
    else:
        intentar(t)
    return cands


def extraer_numeros(texto: str) -> list[tuple[str, set[float]]]:
    texto = _LISTA.sub(" ", texto)
    return [(m.group(), _candidatos(m.group())) for m in _NUM.finditer(texto)]


def _coincide(cands: set[float], valores) -> bool:
    for v in cands:
        for a in valores:
            if a is None:
                continue
            a = abs(a)
            if (abs(v - a) <= 0.0051 or abs(v - round(a, 1)) < 1e-6
                    or (a >= 10 and abs(v - round(a)) < 1e-6)):
                return True
    return False


# Palabras clave de cada métrica (orden: las expresiones largas primero)
_CLAVES = [
    ("roas", r"retorno (?:de|sobre) (?:la )?inversi[oó]n(?: publicitaria)?|\broas\b"),
    ("cpa", r"cost[oe] por adquisici[oó]n|\bcpa\b"),
    ("cvr", r"tasa de conversi[oó]n"),
    ("ctr", r"tasa de clics|\bctr\b"),
    ("beneficio", r"beneficio(?: neto)?"),
    ("ingresos", r"ingresos?(?: atribuidos?)?"),
    ("inversion", r"inversi[oó]n|\bgasto\b"),
    ("confianza", r"confianza"),
]
_PATRON_CLAVES = re.compile("|".join(f"(?P<{k}>{p})" for k, p in _CLAVES), re.IGNORECASE)


def _esperados(metrica: str, h: dict, d: dict, unidad: str):
    """Valores admisibles para la primera cifra que sigue a la métrica, según su unidad."""
    tabla = {
        "roas": {"num": [h["roas"]], "%": [h["var_roas"]], "$": []},
        "cpa": {"num": [h["cpa"]], "$": [h["cpa"]], "%": [h["var_cpa"]]},
        "cvr": {"num": [h["cvr"]], "%": [h["cvr"]], "$": []},
        "ctr": {"num": [h["ctr"]], "%": [h["ctr"]], "$": []},
        "beneficio": {"num": [h["beneficio"]], "$": [h["beneficio"]], "%": []},
        "ingresos": {"num": [h["ingresos"]], "$": [h["ingresos"]], "%": []},
        "inversion": {"num": [h["inversion"]], "$": [h["inversion"]], "%": [h["var_inversion"]]},
        "confianza": {"num": [d["confianza"]], "%": [d["confianza"]], "$": []},
    }
    return [v for v in tabla[metrica][unidad] if v is not None]


def verificar_atribucion(texto: str, h: dict, d: dict) -> list[str]:
    """Comprueba que la cifra que sigue a cada métrica corresponda a esa métrica y unidad."""
    fallas = []
    marcas = list(_PATRON_CLAVES.finditer(texto))
    for i, m in enumerate(marcas):
        metrica = m.lastgroup
        fin = marcas[i + 1].start() if i + 1 < len(marcas) else len(texto)
        alcance = texto[m.end():fin]
        corte = re.search(r"\.(?:\s|$)|;", alcance)
        if corte:
            alcance = alcance[:corte.start()]
        alcance = alcance[:120]
        numeros = list(_NUM.finditer(alcance))
        for j, n in enumerate(numeros):
            antes = alcance[max(0, n.start() - 2):n.start()]
            despues = alcance[n.end():n.end() + 2]
            unidad = "$" if "$" in antes else "%" if "%" in despues else "num"
            if metrica == "roas" and unidad == "$":
                fallas.append(f"ROAS expresado en dólares: {n.group()}")
                continue
            if j > 0 and metrica != "roas":   # ROAS: se verifican todas sus cifras; resto: la primera
                continue
            cands = _candidatos(n.group())
            if cands and not _coincide(cands, _esperados(metrica, h, d, unidad)):
                fallas.append(f"{metrica}≠{n.group()}{'%' if unidad == '%' else ''}")
    return fallas


class GuardrailNumerico:
    """(1) Existencia: toda cifra del texto coincide con una cifra verificada.
       (2) Atribución: la cifra que sigue a cada métrica es la de esa métrica y con su unidad."""

    def __init__(self, bloque: str, hechos: dict | None = None, diagnostico: dict | None = None):
        base = set()
        for _, c in extraer_numeros(bloque):
            base |= c
        self.permitidos = set()
        for v in base:
            self.permitidos |= {v, round(v, 1), round(v, 2)} | ({round(v)} if v >= 10 else set())
        self.hechos, self.diagnostico = hechos, diagnostico

    def verificar_existencia(self, texto: str) -> list[str]:
        return [tok for tok, c in extraer_numeros(texto) if c and not _coincide(c, self.permitidos)]

    def verificar(self, texto: str) -> tuple[bool, list[str]]:
        fallas = self.verificar_existencia(texto)
        if self.hechos is not None:
            fallas += verificar_atribucion(texto, self.hechos, self.diagnostico)
        return (len(fallas) == 0), fallas


# ---------------------------------------------------------------------------
SISTEMA = (
    "Eres el Agente Redactor Ejecutivo de Novar Agencia (Quito, Ecuador). Redactas resúmenes "
    "ejecutivos de desempeño publicitario para gerentes de clientes. Reglas obligatorias: "
    "1) usa ÚNICAMENTE las cifras del bloque DATOS VERIFICADOS, copiadas exactamente y junto a la "
    "métrica a la que pertenecen; 2) no calcules, no redondees ni inventes cifras, porcentajes o fechas; "
    "3) el ROAS es un número sin símbolo $ ni %; 4) compara solo con el periodo anterior indicado en los "
    "datos; 5) escribe exactamente tres oraciones en un solo párrafo, sin títulos, etiquetas ni viñetas, "
    "con un máximo de 70 palabras: situación general, factor principal y la recomendación prioritaria."
)

EJEMPLO_DATOS = """- Cuenta: CLI-ejemplo | Periodo: marzo de 2025
- Inversión: $1,200.00
- Ingresos atribuidos: $3,000.00
- Beneficio neto: $1,800.00
- ROAS: 2.50
- CPA: $40.00
- Conversiones: 30
- Variación del ROAS vs mes anterior: +8.0 %
- Estado clasificado por el modelo: Alerta (confianza 85.0 %)
- Factores determinantes (SHAP): ROAS, beneficio neto, CPA
- Recomendación prioritaria: optimizar la página de destino para mejorar la tasa de conversión"""

EJEMPLO_RESPUESTA = (
    "En marzo de 2025, la cuenta CLI-ejemplo invirtió $1,200.00 y generó $3,000.00 en ingresos "
    "atribuidos, con un ROAS de 2.50 y un CPA de $40.00. El ROAS mejoró +8.0 % frente al mes anterior y "
    "el modelo clasifica el desempeño como Alerta (confianza 85.0 %), explicado sobre todo por el ROAS "
    "y el beneficio neto. Se recomienda optimizar la página de destino para mejorar la tasa de conversión.")


def bloque_datos(h: dict, d: dict) -> str:
    lineas = [
        f"- Cuenta: {h['cliente']} | Periodo: {h['periodo']}",
        f"- Inversión: {usd(h['inversion'])}",
        f"- Ingresos atribuidos: {usd(h['ingresos'])}",
        f"- Beneficio neto: {usd(h['beneficio'])}",
        f"- ROAS: {h['roas']:.2f}",
        f"- CPA: {usd(h['cpa'])}",
        f"- Conversiones: {h['conversiones']}",
        f"- CTR: {pct(h['ctr'])} | Tasa de conversión: {pct(h['cvr'])}",
    ]
    u = h.get("unidad_periodo", "periodo")
    if h["var_roas"] is not None:
        lineas.append(f"- Variación del ROAS vs {u} anterior: {pct(h['var_roas'], True)}")
    if h["var_cpa"] is not None:
        lineas.append(f"- Variación del CPA vs {u} anterior: {pct(h['var_cpa'], True)}")
    lineas.append(f"- Estado clasificado por el modelo: {d['estado']} (confianza {d['confianza']:.1f} %)")
    if d["drivers"]:
        lineas.append(f"- Factores determinantes (SHAP): {', '.join(d['drivers'])}")
    lineas.append(f"- Recomendación prioritaria: {d['recomendaciones'][0]}")
    return "\n".join(lineas)


def plantilla_determinista(h: dict, d: dict) -> str:
    drivers = ", ".join(d["drivers"]) if d["drivers"] else "el ROAS y el beneficio neto"
    texto = (f"En {h['periodo']}, la cuenta {h['cliente']} invirtió {usd(h['inversion'])} y generó "
             f"{usd(h['ingresos'])} en ingresos atribuidos, con un ROAS de {h['roas']:.2f} y un CPA de "
             f"{usd(h['cpa'])} sobre {h['conversiones']} conversiones. ")
    if h["var_roas"] is not None:
        texto += f"El ROAS varió {pct(h['var_roas'], True)} frente al {h.get('unidad_periodo', 'periodo')} anterior. "
    texto += (f"El modelo clasifica el desempeño como {d['estado']} (confianza {d['confianza']:.1f} %), "
              f"explicado principalmente por {drivers}. Se recomienda {d['recomendaciones'][0]}.")
    return texto


def _recortar_oracion(texto: str) -> str:
    texto = re.sub(r"\s+", " ", texto.replace("**", "")).strip()
    texto = re.sub(r"^(resumen ejecutivo|situaci[oó]n general)\s*:\s*", "", texto, flags=re.I)
    if texto and texto[-1] not in ".!?":
        cortes = [m.end() for m in re.finditer(r"\.(?=\s|$)", texto)]
        if cortes and cortes[-1] > 40:
            texto = texto[:cortes[-1]]
    return texto


class AgenteRedactor:
    def __init__(self, model_id: str = "Qwen/Qwen2.5-1.5B-Instruct", usar_llm: bool = True,
                 max_new_tokens: int = 140, temperature: float = 0.1, seed: int = 42):
        self.model_id, self.usar_llm = model_id, usar_llm
        self.max_new_tokens, self.temperature, self.seed = max_new_tokens, temperature, seed
        self.model = self.tok = None
        self.dispositivo = "cpu"
        if usar_llm:
            self._cargar()

    def _cargar(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        cuda = torch.cuda.is_available()
        self.dispositivo = torch.cuda.get_device_name(0) if cuda else "cpu"
        self.tok = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, torch_dtype=torch.float16 if cuda else torch.float32,
            device_map="auto" if cuda else None)
        self.model.eval()

    def _llm(self, datos: str, estricto: bool = False) -> tuple[str, int, float]:
        import torch
        torch.manual_seed(self.seed)
        usuario = f"DATOS VERIFICADOS (calculados por el sistema):\n{datos}\n\nRedacta el resumen ejecutivo."
        if estricto:
            usuario += (" IMPORTANTE: tu intento anterior contenía cifras no verificadas o asignadas a la "
                        "métrica equivocada. Copia cada cifra junto a su métrica exactamente como en DATOS.")
        mensajes = [{"role": "system", "content": SISTEMA},
                    {"role": "user", "content": f"DATOS VERIFICADOS (calculados por el sistema):\n"
                                                f"{EJEMPLO_DATOS}\n\nRedacta el resumen ejecutivo."},
                    {"role": "assistant", "content": EJEMPLO_RESPUESTA},
                    {"role": "user", "content": usuario}]
        prompt = self.tok.apply_chat_template(mensajes, tokenize=False, add_generation_prompt=True)
        entradas = self.tok(prompt, return_tensors="pt").to(self.model.device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            salida = self.model.generate(
                **entradas, max_new_tokens=self.max_new_tokens, do_sample=True,
                temperature=self.temperature, top_p=0.9, repetition_penalty=1.05,
                pad_token_id=self.tok.eos_token_id)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        nuevos = salida[0, entradas["input_ids"].shape[1]:]
        return self.tok.decode(nuevos, skip_special_tokens=True), int(len(nuevos)), dt

    def generar(self, hechos: dict, diagnostico: dict) -> dict:
        datos = bloque_datos(hechos, diagnostico)
        guard = GuardrailNumerico(datos, hechos, diagnostico)
        if self.model is None:
            texto = plantilla_determinista(hechos, diagnostico)
            return {"texto": texto, "texto_llm_crudo": None, "fuente": "plantilla",
                    "fiel_crudo": None, "fiel_existencia_crudo": None, "cifras_no_verificadas": [],
                    "latencia_s": 0.0, "tokens": 0, "tokens_por_s": 0.0, "intentos": 0,
                    "truncado": False}
        crudo, ntok, dt = self._llm(datos)
        crudo = _recortar_oracion(crudo)
        ok, fallas = guard.verificar(crudo)
        res = {"texto_llm_crudo": crudo, "fiel_crudo": ok,
               "fiel_existencia_crudo": len(guard.verificar_existencia(crudo)) == 0,
               "cifras_no_verificadas": fallas, "latencia_s": dt, "tokens": ntok,
               "tokens_por_s": ntok / dt if dt else 0.0, "intentos": 1,
               "truncado": ntok >= self.max_new_tokens}
        if ok:
            res.update(texto=crudo, fuente="llm")
            return res
        crudo2, ntok2, dt2 = self._llm(datos, estricto=True)
        crudo2 = _recortar_oracion(crudo2)
        ok2, _ = guard.verificar(crudo2)
        res["intentos"] = 2
        res["latencia_total_s"] = dt + dt2
        if ok2:
            res.update(texto=crudo2, fuente="llm_reintento")
        else:
            res.update(texto=plantilla_determinista(hechos, diagnostico), fuente="plantilla_fallback")
        return res


def verificar_fidelidad_final(texto: str, hechos: dict, diagnostico: dict) -> bool:
    ok, _ = GuardrailNumerico(bloque_datos(hechos, diagnostico), hechos, diagnostico).verificar(texto)
    return ok

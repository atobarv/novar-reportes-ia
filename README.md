# Sistema de Generación Inteligente de Reportes de Marketing · Novar Agencia · v2.0

Prototipo del proyecto de titulación de la **Maestría en Inteligencia Artificial Aplicada (UDLA – FICA)**.
Autor: Julio Alejandro Tobar Valencia · Tutor: MSc. Paulo Guerra Terán · Sponsor: Novar Agencia (Quito, Ecuador).

**Notebook (Colab):** https://drive.google.com/file/d/1AlZ0KEUty1ZW0mb5Elhwya2I4XUGSPC2/view?usp=sharing
**Video demo:** https://drive.google.com/file/d/1DMhVk4nvT4OnvS_TD2CkKdzhl3Ulc-r3/view?usp=sharing

## 1. Problema
La consolidación mensual de métricas (Meta Ads, Google Ads, GA4, CRM) y la redacción del informe de cada cliente
tomaban unas **18 horas por cliente al mes**, con retrasos de hasta 7 días hábiles y riesgo de errores en ROAS y CPA.

## 2. Solución: reglas donde basta calcular, aprendizaje automático donde hay que inferir
| Tarea | Enfoque |
|---|---|
| Consolidar exportaciones y calcular KPI (ROAS, CPA, CTR, CVR, CPM, beneficio) | Determinista |
| Clasificar el estado actual de la cuenta | Determinista (regla de negocio) |
| Comparar con el pasado (mes anterior, mismo mes del año anterior, promedio 12 meses) | Determinista |
| Recomendaciones de acción | Determinista (reglas de negocio) |
| Proyectar ROAS e inversión a 3, 6 y 12 meses | Aprendizaje automático |
| Anticipar deterioros (paso a un estado peor) | Aprendizaje automático + regla |
| Detectar meses con patrones anómalos (fallas de tracking, gasto duplicado, tráfico inválido) | Aprendizaje automático no supervisado |
| Redactar el resumen ejecutivo | Modelo de lenguaje con verificación determinista |

El estado actual (Crítico/Alerta/Óptimo) se calcula con la regla de negocio porque el ROAS es un cálculo, no una
predicción: un experimento con validación aleatoria, por cliente y por fecha mostró que los clasificadores de ML solo
reproducen esa regla. El ML se usa para **proyectar el ROAS a 3, 6 y 12 meses**, **anticipar deterioros** y
**detectar anomalías**; un LLM local redacta el comparativo del pasado y la proyección, verificado cifra por cifra.

## 3. Pipeline
```
Exportaciones CSV/Excel ─► ETL (normalización, exclusión de fugas, seudonimización SHA-256)
  ─► KPI y estado por regla de negocio (determinista) ─► comparativo con el pasado (determinista)
  ─► Detector de anomalías (Isolation Forest) ─► imputación de meses anómalos
  ─► Proyección del ROAS a 3/6/12 meses (Ensamble ML (Ridge + Gradient Boosting)) + rango 80 % + probabilidad de deterioro
  ─► Agentes Extractor · Pronóstico · Analista · Redactor (Qwen2.5-1.5B) ─► Guardrail (existencia, atribución, dirección)
  ─► Tablero Streamlit + PDF ─► Revisión, corrección y aprobación del Director de Cuenta (registro de la prueba operativa)
```

## 4. Datos
- `data/raw/dataset_marketing_novar_temporal_v2.csv`: 14 cuentas × 48 meses (oct-2022 a sep-2026), **simulado** y
  calibrado con el dataset v1 (13 meses por cuenta, sin estructura temporal: autocorrelación -0.11).
  El generador (`src/etl/simulador_temporal.py`) documenta estacionalidad, tendencia, persistencia, rendimientos
  decrecientes y anomalías operativas.
- `data/eval/anomalias_inyectadas.csv`: verdad de referencia de las anomalías (solo para evaluar, nunca como entrada).

## 5. Validación sin fuga de información
Toda evaluación de ML usa **cuentas no vistas**, **meses posteriores** al corte (Sep-2025) o ambos.

**Proyección del ROAS (cuentas y meses no vistos)**

| Horizonte | MAE del ROAS | Mejora vs pronóstico ingenuo |
|---|---|---|
| 1 meses | 0.282 | +15.5 % |
| 3 meses | 0.305 | +25.7 % |
| 6 meses | 0.353 | +18.9 % |
| 12 meses | 0.331 | -4.3 % |

**Detección de anomalías**

| Método | Esquema | Precisión | Recall | F1 |
|---|---|---|---|---|
| Isolation Forest (ML) | por cliente | 93.9% | 86.1% | 89.9% |
| Regla |z ROAS| > 3 | por cliente | 39.5% | 47.2% | 43.0% |
| Isolation Forest (ML) | temporal | 100.0% | 90.9% | 95.2% |
| Regla |z ROAS| > 3 | temporal | 85.7% | 54.5% | 66.7% |

**Redacción (test ciego N=28)**: fidelidad final 100% · LLM sin corrección
68% · latencia media 9.70 s.

**Impacto:** 18 h → 0.05 h por cliente (medido en la prueba operativa),
reducción de 99.7%.

**Pruebas automatizadas:** 9 passed

## 6. Estructura
```
app.py                       Tablero Streamlit (estado, proyección, anomalías, reporte y prueba operativa)
src/etl/                     Ingesta, limpieza, seudonimización y simulador temporal v2
src/features/                KPI y comparativo histórico
src/rules/                   Regla de estado, recomendaciones y tabla determinista vs ML
src/models/                  Pronóstico (3/6/12 m), anomalías y experimento de verificación del clasificador
src/agents/                  Agentes y guardrail
src/reports/                 PDF con gráfico de proyección y bloque HITL
scripts/run_pipeline.py      Ejecución por consola
tests/                       Pruebas pytest
models/                      pronosticador_roas.joblib · detector_anomalias.joblib
outputs/                     Figuras, métricas, reportes PDF
notebooks/                   Notebook de Colab que reproduce todo
```

## 7. Ejecución
**Colab:** abrir `notebooks/Proyecto_Final_Capstone_Novar.ipynb`, GPU T4, *Ejecutar todas*.
**Local:**
```bash
pip install -r requirements.txt
python -m pytest tests -v
python scripts/run_pipeline.py --datos data/raw/dataset_marketing_novar_temporal_v2.csv          # sin LLM
python scripts/run_pipeline.py --datos data/raw/dataset_marketing_novar_temporal_v2.csv --llm    # con LLM
streamlit run app.py
docker compose up --build
```

## 8. Ética y limitaciones
Seudonimización (LOPDP), LLM local, aprobación humana obligatoria, auditoría de error por vertical y escala
(`docs/MODEL_CARD.md`). Datos simulados: validar con exportaciones reales antes del uso con clientes.

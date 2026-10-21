# Sistema de Generación Inteligente de Reportes de Marketing · Novar Agencia

Prototipo final (v1.1) del proyecto de titulación de la **Maestría en Inteligencia Artificial Aplicada
(UDLA – FICA)**. Autor: Julio Alejandro Tobar Valencia · Tutor: MSc. Paulo Guerra Terán ·
Sponsor: Novar Agencia (Quito, Ecuador).




## 1. Problema
En Novar Agencia la consolidación mensual de métricas (Meta Ads, Google Ads, GA4 y CRM) y la redacción
del informe ejecutivo de cada cliente se hacían a mano: unas **18 horas por cliente al mes**, retrasos de
hasta 7 días hábiles y un costo indirecto cercano a **USD 1 800 mensuales**, con riesgo de errores al
calcular ROAS y CPA.

## 2. Solución
Un copiloto de IA que (1) ingiere y normaliza las exportaciones publicitarias, (2) calcula los KPIs de
forma determinista, (3) clasifica el rendimiento de cada cuenta (Crítico / Alerta / Óptimo) con un modelo
explicable (Random Forest + SHAP/LIME), (4) redacta el resumen ejecutivo con un LLM local
(Qwen2.5-1.5B-Instruct) protegido por un **guardrail numérico** que verifica existencia y atribución de
cada cifra, y (5) exige validación humana (HITL) antes de enviar el PDF al cliente.

## 3. Pipeline
```
Exportaciones CSV/Excel ─► ETL (normalización, exclusión de fugas, seudonimización SHA-256)
   ─► KPIs deterministas (ROAS, CPA, CTR, CVR, CPM, beneficio)
   ─► Clasificador RobustScaler + RandomForest (5-Fold CV) ─► SHAP / LIME
   ─► Agentes: Extractor → Analista → Redactor (Qwen2.5-1.5B, one-shot) → Guardrail (reintento / plantilla)
   ─► Dashboard Streamlit + PDF (WeasyPrint) ─► Validación y firma del Director de Cuenta (HITL)
```

## 4. Estructura del repositorio
```
data/raw/            Dataset seudonimizado de entrada
data/processed/      Dataset con KPIs y etiqueta
notebooks/           Notebook de Colab que reproduce todo el proyecto
src/etl/             Ingesta, limpieza, seudonimización
src/features/        KPIs y etiquetado
src/models/          Baselines, Random Forest, validación cruzada, pruebas de robustez
src/explain/         TreeSHAP y LIME
src/agents/          Agentes y guardrail numérico
src/reports/         Exportación HTML/PDF con bloque HITL
scripts/             run_pipeline.py (ejecución por línea de comandos)
tests/               Pruebas automatizadas (pytest)
models/              Modelo entrenado (.joblib)
outputs/figures/     Figuras (EDA, modelos, SHAP, LIME, sesgos, robustez, diagramas)
outputs/metrics/     resultados_finales.json/.xlsx, tablas CSV, reporte de pytest
outputs/reports/     Reportes PDF de ejemplo por cuenta
docs/                MODEL_CARD.md
app.py               Dashboard Streamlit
Dockerfile, docker-compose.yml, requirements.txt, CHANGELOG.md
```

## 5. Requisitos
- Python 3.10 o superior. GPU NVIDIA (p. ej. T4 de Google Colab gratuito) recomendada para el LLM;
  sin GPU el sistema funciona con la plantilla determinista.
- Dependencias exactas en `requirements.txt` (scikit-learn, shap, lime, torch, transformers, streamlit,
  weasyprint, rouge-score, sacrebleu, pytest, entre otras).
- WeasyPrint requiere las librerías de sistema `libpango-1.0-0` y `libpangoft2-1.0-0` (incluidas en el Dockerfile).

## 6. Ejecución paso a paso
**Opción A · Google Colab (reproduce todo el proyecto)**
1. Abrir `notebooks/Proyecto_Final_Capstone_Novar.ipynb` en Colab y activar GPU T4.
2. Colocar el dataset en la carpeta de trabajo de Drive y ejecutar *Entorno de ejecución › Ejecutar todas*.
3. La celda 16 abre el dashboard.

**Opción B · Local**
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest tests -v                                # pruebas
python scripts/run_pipeline.py --datos data/raw/dataset_marketing_novar_agencia.csv          # sin LLM
python scripts/run_pipeline.py --datos data/raw/dataset_marketing_novar_agencia.csv --llm    # con LLM
streamlit run app.py                                     # dashboard en http://localhost:8501
```

**Opción C · Docker**
```bash
docker compose up --build        # http://localhost:8501
```

## 7. Resultados (ejecución 21/10/2026, semilla 42)
Datos: 182 registros mensuales · 14 cuentas ·
origen: Exportaciones de Novar Agencia (dataset_marketing_novar_agencia.csv) (simulado con la estructura de las exportaciones reales de la agencia).

**Clasificación · 5-Fold Stratified CV**

| Modelo | Accuracy | F1 ponderado | Brecha train–val |
|---|---|---|---|
| Regresión Logística (baseline) | 91.82% ± 5.39% | 91.81% | 2.41% |
| Árbol de Decisión d=3 (baseline) | 100.00% ± 0.00% | 100.00% | 0.00% |
| Random Forest + RobustScaler (ajustado) | 96.17% ± 3.72% | 95.96% | 3.69% |

**Robustez · accuracy media bajo ruido de etiqueta y de atribución**

| Modelo | Datos limpios | Índice de robustez | Ruido severo |
|---|---|---|---|
| Random Forest + RobustScaler (ajustado) | 96.2% | 83.8% | 80.9% |
| Random Forest d=6 (mejor F1 en datos limpios) | 98.1% | 83.5% | 79.7% |
| Árbol de Decisión d=3 (baseline) | 100.0% | 81.8% | 77.1% |
| Regresión Logística (baseline) | 90.6% | 79.1% | 75.8% |

**Módulo generativo · test ciego N=28 (GPU Tesla T4)**

| Métrica | Valor |
|---|---|
| Latencia media / p95 | 6.10 s / 6.22 s |
| Throughput | 22.94 tokens/s |
| Fidelidad del LLM sin corrección (existencia / existencia + atribución) | 96.4% / 96.4% |
| Fidelidad final con guardrail | 100.0% |
| ROUGE-L / BLEU vs diagnóstico de referencia | 0.229 / 0.049 |

**Impacto:** reducción estimada del tiempo de elaboración de **97.2%**
(18 h → 0.50 h por cliente,
incluida la revisión humana) y ahorro aproximado de USD 1,750 al mes.

**Pruebas automatizadas:** ============================== 7 passed in 3.08s ===============================

## 8. Ética, privacidad y limitaciones
Seudonimización SHA-256 con sal (LOPDP Ecuador), LLM local sin envío de datos a terceros, explicabilidad
SHAP/LIME, auditoría de sesgos por escala presupuestaria y vertical, y validación humana obligatoria
(ver `docs/MODEL_CARD.md`). Los datos son simulados y la etiqueta sigue umbrales de ROAS definidos por la
agencia; la validación con datos reales de clientes es trabajo futuro.

## 9. Control de versiones
Historial en `CHANGELOG.md`. Código versionado con Git (etiqueta `v1.1`) y datos con DVC y huella SHA-256.

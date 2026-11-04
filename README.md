# Sistema de Generación Inteligente de Reportes de Marketing · Novar Agencia · v2.1

Prototipo del proyecto de titulación de la **Maestría en Inteligencia Artificial Aplicada (UDLA – FICA)**.
Autor: Julio Alejandro Tobar Valencia · Tutor: MSc. Paulo Guerra Terán · Sponsor: Novar Agencia (Quito, Ecuador).
Licencia: MIT (ver `LICENSE`). Datos: simulados y seudonimizados.

**Notebook (Colab):** https://drive.google.com/file/d/1_-TPoD2bP1HBRXdfbJO5j9kZRhqMtru3/view?usp=sharing
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
| Proyectar el ROAS a 3, 6 y 12 meses | Aprendizaje automático |
| Anticipar deterioros (paso a un estado peor) | Aprendizaje automático + regla |
| Detectar meses con patrones anómalos (fallas de tracking, gasto duplicado, tráfico inválido) | Aprendizaje automático no supervisado |
| Redactar el resumen ejecutivo | Modelo de lenguaje con verificación determinista |

## 3. Flujo de uso (acción → procesamiento → resultado)
1. **Acción:** el Director de Cuenta carga la exportación mensual (tablero, consola o `POST /reporte`) y elige cuenta y mes.
2. **Procesamiento:** ETL (validación, imputación marcada de faltantes, consolidación por plataforma, seudonimización) →
   KPI, estado por regla y comparativo histórico → anomalías → proyección 3/6/12 meses → agentes → guardrail.
3. **Resultado:** borrador con estado, comparativo, proyección con rango del 80 %, alertas y recomendaciones; el
   Director lo corrige y aprueba (PDF y registro de la prueba operativa). **Nada se envía sin aprobación humana.**

### Entradas
| Columna canónica | Alias aceptados (ejemplos) | Obligatoria |
|---|---|---|
| `fecha` | date, periodo, mes, reporting_starts | Sí, para comparar y proyectar |
| `cliente` | cuenta, account, account_name | Sí |
| `gasto_usd` | spend, amount_spent, costo, inversion | Sí |
| `impresiones` | impressions | Sí |
| `clics` | clicks, link_clicks | Sí |
| `conversiones` | results, leads, purchases | Sí |
| `ingresos_usd` | revenue, conversion_value, purchase_value | Sí |
| `vertical`, `plataforma` | industria, sector · platform, canal | No |

Formatos numéricos «1.234,56», «1,234.56» y «$2.558,86» se interpretan automáticamente. Se requieren **12 meses** de
historia por cuenta para proyectar.

### Salidas
`outputs/reports/*.pdf` (reporte con firma), `outputs/metrics/*.csv|json|xlsx` (métricas), registro de la prueba
operativa (CSV) y respuesta JSON del endpoint.

## 4. Pipeline
```
Exportaciones CSV/Excel ─► ETL (validación, imputación marcada, consolidación, exclusión de fugas, SHA-256)
  ─► KPI y estado por regla de negocio ─► comparativo con el pasado (determinista)
  ─► Detector de anomalías (Isolation Forest + salvaguarda |z| ≥ 10) ─► imputación de meses anómalos
  ─► Proyección del ROAS a 3/6/12 meses (Ensamble ML (Ridge + Gradient Boosting)) + rango 80 % + probabilidad de deterioro
  ─► Agentes Extractor · Pronóstico · Analista · Redactor (Qwen2.5-1.5B, prompt P3)
  ─► Guardrail (existencia, atribución, dirección) ─► Tablero Streamlit / API REST / PDF ─► Aprobación del Director
```

## 5. Datos
- `data/raw/dataset_marketing_novar_temporal_v2.csv`: 14 cuentas × 48 meses (oct-2022 a sep-2026), **simulado** y
  calibrado con el dataset v1 (13 meses por cuenta, sin estructura temporal: autocorrelación -0.11).
- `data/eval/anomalias_inyectadas.csv`: verdad de referencia de las anomalías (solo para evaluar).

## 6. Resultados (cuentas y meses no vistos)
| Horizonte | MAE del ROAS | Mejora vs pronóstico ingenuo |
|---|---|---|
| 1 meses | 0.282 | +15.5 % |
| 3 meses | 0.305 | +25.7 % |
| 6 meses | 0.353 | +18.9 % |
| 12 meses | 0.331 | -4.3 % |

| Detección de anomalías | Esquema | Precisión | Recall | F1 |
|---|---|---|---|---|
| Isolation Forest (ML) | por cliente | 93.9% | 86.1% | 89.9% |
| Regla |z ROAS| > 3 | por cliente | 39.5% | 47.2% | 43.0% |
| Isolation Forest (ML) | temporal | 100.0% | 90.9% | 95.2% |
| Regla |z ROAS| > 3 | temporal | 85.7% | 54.5% | 66.7% |

**Redacción (test ciego N=28, prompt P3)**: fidelidad final 100% ·
texto del LLM sin corrección 86% · latencia media 10.80 s.

**Impacto:** 18 h → 0.50 h por cliente en el escenario conservador
(−97.2%); 0.05 h (medido en la prueba operativa).

**Pruebas automatizadas:** 18 passed

## 7. Robustez y límites de uso
| Métrica (deriva) | Cambio de la media | PSI | Nivel |
|---|---|---|---|
| ROAS | -4.4 % | 0.337 | alto |
| CPM (USD) | +10.5 % | 0.622 | alto |
| CPA (USD) | +29.3 % | 0.043 | bajo |
| CTR (%) | +1.5 % | 0.083 | bajo |
| Tasa de conversión (%) | +7.4 % | 0.058 | bajo |
| Inversión (USD) | +5.3 % | 0.194 | moderado |

Faltantes imputados y marcados (estado **provisional**), formatos y plataformas consolidados, errores con mensaje
claro, y pruebas adversariales del LLM (instrucciones inyectadas, proyección ausente). **No usar** con cuentas de
menos de 12 meses (no hay proyección), con meses anómalos o imputados sin verificar, ni para decidir presupuestos
sin revisión humana. Detalle en `docs/MODEL_CARD.md` y `outputs/metrics/robustez_*.csv`.

## 8. Estructura
```
app.py                       Tablero Streamlit (estado, proyección, anomalías, reporte y prueba operativa)
api.py                       Endpoint REST (FastAPI): GET /salud · POST /reporte
src/etl/                     Ingesta, validación, imputación, seudonimización y simulador temporal v2
src/features/                KPI y comparativo histórico
src/rules/                   Regla de estado, recomendaciones y tabla determinista vs ML
src/models/                  Pronóstico (3/6/12 m), anomalías y experimento de verificación del clasificador
src/agents/                  Agentes, versiones del prompt y guardrail
src/evaluation/              Pruebas de robustez, deriva y pruebas adversariales del LLM
src/reports/                 PDF con gráfico de proyección y bloque HITL
scripts/run_pipeline.py      Ejecución por consola
tests/                       Pruebas pytest
models/                      pronosticador_roas.joblib · detector_anomalias.joblib · config_redactor.json
outputs/                     Figuras, métricas, reportes PDF
notebooks/                   Notebook de Colab que reproduce todo
```

## 9. Ejecución
**Colab:** abrir `notebooks/Proyecto_Final_Capstone_Novar.ipynb`, GPU T4, *Ejecutar todas* (≈ 30 min).

**Local:**
```bash
pip install -r requirements.txt
python -m pytest tests -v
python scripts/run_pipeline.py --datos data/raw/dataset_marketing_novar_temporal_v2.csv --cliente CLI-1b0dc833
streamlit run app.py                      # tablero
uvicorn api:app --port 8000               # endpoint; documentación en http://localhost:8000/docs
docker compose up --build                 # tablero (8501) y API (8000)
```

Salida esperada de la consola:
```
[ETL] 672 registros · 14 cuentas · estado: regla de negocio (ROAS < 1,8 Crítico; ≥ 2,8 Óptimo)
[Anomalías] 31 meses marcados
[Proyección] modelo: Ensamble ML (Ridge + Gradient Boosting)
[Reporte] CLI-1b0dc833 · septiembre de 2026: Óptimo · ROAS 3.76 → 12 m 3.99 · plantilla → reporte_CLI-1b0dc833_2026-09.pdf
```

Ejemplo de solicitud y respuesta del endpoint (extracto):
```bash
curl -X POST http://localhost:8000/reporte -H "Content-Type: application/json" \
     -d '{"cliente": "CLI-1b0dc833", "registros": [{"fecha": "2026-09-01", "cliente": "CLI-1b0dc833", "gasto_usd": 2770.51, "impresiones": 391000, "clics": 9800, "conversiones": 118, "ingresos_usd": 10422.35}, "..."]}'
```
```json
{
 "version": "2.1",
 "cuenta": "CLI-1b0dc833",
 "mes": "septiembre de 2026",
 "estado": "Óptimo",
 "kpi": {
  "inversion": 2770.51,
  "ingresos": 10422.35,
  "beneficio": 7651.84,
  "roas": 3.7619,
  "cpa": 23.4789,
  "ctr": 2.5287,
  "cvr": 1.1997
 },
 "proyecciones": {
  "3_meses": {
   "mes": "diciembre de 2026",
   "roas": 4.369,
   "rango_80": [
    3.728,
    5.276
   ],
   "estado": "Óptimo",
   "prob_deterioro": 0.008
  },
  "6_meses": {
   "mes": "marzo de 2027",
   "roas": 4.12,
   "rango_80": [
    3.436,
    5.058
   ],
   "estado": "Óptimo",
   "prob_deterioro": 0.009
  },
  "12_meses": {
   "mes": "septiembre de 2027",
   "roas": 3.986,
   "rango_80": [
    3.175,
    5.168
   ],
   "estado": "Óptimo",
   "prob_deterioro": 0.031
  }
 },
 "recomendaciones": [
  "revisar segmentación y creatividades, porque el CPA aumentó frente al mes anterior",
  "optimizar la página de destino y el formulario para mejorar la tasa de conversión",
  "escalar el presupuesto de forma gradual vigilando el ROAS marginal"
 ],
 "guardrail": {
  "aprobado": true,
  "fallas": []
 },
 "requiere_aprobacion_humana": true
}
```
Entrada inválida → `422` con `{"detail": "La columna «ingresos_usd» no contiene valores numéricos reconocibles...."}`.

## 10. Versiones
`v1.0` pipeline inicial · `v1.1` guardrail de atribución, robustez del clasificador · `v2.0` reglas + ML para proyección,
anomalías y deterioro; validación por cliente y fecha; prueba operativa · `v2.1` robustez, versiones del prompt,
endpoint REST, validación de entradas y licencia. Ver `CHANGELOG.md`.

## 11. Ética
Seudonimización (LOPDP), LLM local, aprobación humana obligatoria, auditoría de error por vertical y escala, y
declaración de límites en `docs/MODEL_CARD.md`.

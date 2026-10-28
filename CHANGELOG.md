# Historial de versiones

## v2.0 — 28/10/2026 · Respuesta a la retroalimentación del tutor
- El estado actual se calcula con la **regla de negocio**; el clasificador de ML queda como experimento de verificación
  (validación aleatoria, por cliente y por fecha) que demuestra que solo reproduce la regla.
- Nueva separación **determinista vs ML** (`src/rules/reglas_negocio.py`).
- **Proyección del ROAS a 3, 6 y 12 meses** con modelo global, rango del 80 % y probabilidad de deterioro.
- **Detección de anomalías** con Isolation Forest e imputación antes de proyectar.
- **Validación por cliente, por fecha y combinada** para todos los modelos de ML.
- Dataset temporal v2 (48 meses) calibrado con el v1, con generador documentado y anomalías de referencia aparte.
- Guardrail con verificación de **dirección** de variaciones (signo y verbo).
- Reporte PDF con gráfico de histórico y proyección; tablero con **registro de la prueba operativa**
  (tiempos, correcciones y utilidad de las recomendaciones).
- Control de versiones sin reescribir el historial (clonado del repositorio y etiqueta `v2.0`).

## v1.1 — 21/10/2026
- Guardrail de atribución, prompt de una muestra, pruebas de robustez, pytest, CLI, README y Docker.

## v1.0 — 21/10/2026
- Pipeline integral en Colab: ETL, Random Forest, SHAP/LIME, LLM con guardrail y Streamlit.

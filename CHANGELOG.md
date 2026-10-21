# Historial de versiones

## v1.1 — 21/10/2026 · Prototipo final consolidado
- Guardrail numérico ampliado: además de verificar que cada cifra exista, verifica que esté atribuida a la
  métrica correcta y con su unidad (p. ej., detecta «ROAS de −586.23» o un ROAS expresado en dólares).
- Prompt one-shot y límite de 70 palabras en tres oraciones: elimina títulos y truncamientos del LLM.
- Pruebas de robustez (ruido de etiqueta y de atribución) para seleccionar el modelo final.
- Se ratifica el Random Forest de profundidad 4: aunque la profundidad 6 obtiene mejor F1 con datos limpios,
  la profundidad 4 es la más robusta ante ruido (números aleatorios comunes, 5 repeticiones).
- Reglas del árbol de decisión exportadas como evidencia de interpretabilidad.
- Exclusión automática de columnas que filtran la etiqueta (status_code) y soporte de periodos mensuales.
- Auditoría de sesgos por vertical de negocio; ROUGE/BLEU contra los diagnósticos de referencia del dataset.
- Script `scripts/run_pipeline.py`, pruebas `pytest`, README completo, Dockerfile y docker-compose.

## v1.0 — 21/10/2026 · Pipeline integral en Colab
- Repositorio modular `src/` (ETL, features, modelos, interpretabilidad, agentes, reportes) y dashboard Streamlit.
- 5-Fold Stratified CV, ablación de variables de negocio, búsqueda de hiperparámetros, SHAP, LIME, Model Card.
- Test ciego N=28 del LLM Qwen2.5-1.5B-Instruct en GPU T4 con guardrail de existencia de cifras.

## v0.x — agosto–septiembre 2026 · Prototipos de hitos 5–7
- Baselines (Regresión Logística, Árbol de Decisión) y primer Random Forest con RobustScaler.
- Primer módulo generativo con guardrails deterministas; laboratorio ético y Model Card inicial.

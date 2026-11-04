# Historial de versiones (versionado semántico)

## v2.1 — 04/11/2026 · Robustez, prompting responsable y despliegue ligero
- **Validación de entradas:** faltantes imputados con la mediana de 6 meses previos y marcados (estado provisional);
  consolidación de varias filas por cuenta y mes (plataformas o campañas); mensajes de error explícitos.
- **Detector de anomalías:** salvaguarda determinista para desviaciones extremas en una métrica (|z| ≥ 10).
- **Prompts versionados:** P1 (sin ejemplo), P2 (una muestra, v2.0) y P3 (magnitudes sin signo, defensa ante
  instrucciones en los datos); la versión se elige con meses de validación y se guarda en `models/config_redactor.json`.
- **Pruebas de robustez:** sensibilidad, estrés, error controlado, deriva de datos (PSI y KS) y pruebas adversariales
  ligeras del LLM (`src/evaluation/robustez.py`).
- **Endpoint REST** (`api.py`, FastAPI) con documentación OpenAPI; servicio adicional en `docker-compose.yml`.
- Sanitización de los campos de texto antes del prompt; recomendaciones más prudentes en meses Críticos o anómalos.
- `LICENSE` (MIT), requisitos congelados y README con flujo de uso, entradas, salidas, ejemplos y límites.

## v2.0 — 28/10/2026 · Respuesta a la retroalimentación del tutor
- Estado actual por **regla de negocio**; el clasificador queda como experimento de verificación.
- Separación **determinista vs ML**; proyección del ROAS a 3, 6 y 12 meses con rango del 80 % y probabilidad de
  deterioro; detección de anomalías con Isolation Forest; validación por cliente, por fecha y combinada.
- Dataset temporal v2 (48 meses); guardrail de dirección; PDF con gráfico; registro de la prueba operativa.

## v1.1 — 21/10/2026
- Guardrail de atribución, prompt de una muestra, pruebas de robustez, pytest, CLI, README y Docker.

## v1.0 — 21/10/2026
- Pipeline integral en Colab: ETL, Random Forest, SHAP/LIME, LLM con guardrail y Streamlit.

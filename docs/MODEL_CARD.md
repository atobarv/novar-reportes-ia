# Model Card · Clasificador de rendimiento publicitario (Novar Agencia)

**Versión:** 1.0 · **Fecha:** 21/10/2026 · **Responsable:** Julio Alejandro Tobar Valencia

## Uso previsto
Clasificar cada periodo de una cuenta publicitaria en *Crítico / Alerta / Óptimo* para priorizar la
atención del Director de Cuenta y alimentar la redacción del reporte ejecutivo. **No** decide
presupuestos ni se comunica con clientes sin validación humana (HITL).

## Datos
- Origen: Exportaciones de Novar Agencia (dataset_marketing_novar_agencia.csv) · 182 registros · 14 cuentas seudonimizadas
- Etiqueta: etiqueta del dataset (criterio experto)
- Huella SHA-256: `25ea0f0d411bbf88…`

## Modelo
RobustScaler + RandomForestClassifier (50 árboles, profundidad 4, semilla 42; ratificada por pruebas de robustez) con
11 variables (cvr, profit_usd, cpm_usd derivadas).

## Desempeño (5-Fold Stratified CV)
- Accuracy: 96.17% ± 3.72%
- F1 ponderado: 95.96% ± 3.95%
- Brecha entrenamiento–validación: 3.69%

## Explicabilidad
Variables con mayor peso (TreeSHAP): roas (50.4 %), profit_usd (20.0 %), cpa_usd (12.6 %). Concordancia SHAP–LIME (top-3): 88.0%.

## Equidad
Brecha de accuracy por escala presupuestaria: 9.2%; por vertical de negocio: 15.4%.
Riesgos: sesgo de escala (cuentas pequeñas con poco volumen) y de atribución entre plataformas.

## Limitaciones
Muestra pequeña; las etiquetas reflejan el criterio de un experto; la atribución de ingresos depende
de cada plataforma. Reentrenar cuando cambien los umbrales comerciales o ingresen nuevas cuentas.

## Principios éticos (Correa et al., 2023 — 200 guías globales)
- **Transparencia / explicabilidad** — Cumple: TreeSHAP global y local, LIME y factores SHAP en cada reporte
- **Confiabilidad y seguridad** — Cumple: Guardrail numérico: toda cifra se verifica contra datos calculados en Python
- **Justicia y no discriminación** — Cumple con monitoreo: Brecha de accuracy entre escalas presupuestarias: 9.2%; uso de variables relativas
- **Privacidad (LOPDP Ecuador)** — Cumple: Seudonimización SHA-256 con sal; mapeo fuera del repositorio; LLM local sin envío a terceros
- **Responsabilidad / rendición de cuentas** — Cumple: Protocolo HITL: firma obligatoria del Director de Cuenta antes del envío
- **Autonomía humana** — Cumple: El sistema es copiloto; no envía reportes ni modifica campañas
- **Beneficencia / sostenibilidad** — Cumple: LLM de 1.5B en fp16 (bajo consumo) frente a modelos de API de gran escala
- **Trazabilidad y reproducibilidad** — Cumple: Semilla 42, huella SHA-256 de datos y versionado DVC

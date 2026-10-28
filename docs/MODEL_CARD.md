# Model Card · Sistema de reportes inteligentes (Novar Agencia) · v2.0

**Fecha:** 28/10/2026 · **Responsable:** Julio Alejandro Tobar Valencia

## Componentes y uso previsto
- **Estado actual:** regla de negocio (ROAS < 1,8 Crítico; ≥ 2,8 Óptimo). Sin ML.
- **Proyección del ROAS a 3/6/12 meses:** Ensamble ML (Ridge + Gradient Boosting), con rango del 80 % y probabilidad de deterioro.
- **Detección de anomalías:** Isolation Forest sobre desviaciones robustas de 6 métricas.
- **Redacción:** Qwen/Qwen2.5-1.5B-Instruct con guardrail determinista. Todo reporte requiere aprobación humana.

## Datos
Dataset v2 simulado y calibrado con el v1: 672 meses-cuenta, 14 cuentas, 10/2022–09/2026.

## Desempeño (cuentas y meses no vistos)
- Horizonte 1 meses: MAE 0.282 · mejora vs ingenuo +15.5 %
- Horizonte 3 meses: MAE 0.305 · mejora vs ingenuo +25.7 %
- Horizonte 6 meses: MAE 0.353 · mejora vs ingenuo +18.9 %
- Horizonte 12 meses: MAE 0.331 · mejora vs ingenuo -4.3 %
- Anomalías (por cliente): F1 89.9% vs 43.0% de la regla de umbral.

## Limitaciones
Datos simulados; la proyección no anticipa cambios estructurales (relanzamientos, cambios de presupuesto);
los rangos se amplían con el horizonte. Validar con exportaciones reales antes del uso con clientes.

## Principios éticos (Correa et al., 2023)
- **Transparencia / explicabilidad** — Cumple: Estado por regla explícita; importancia de variables del pronóstico; factores (estacionalidad, tendencia) en cada proyección
- **Confiabilidad y seguridad** — Cumple: Guardrail de existencia, atribución y dirección; plantilla de respaldo; rangos del 80 % calibrados
- **Justicia y no discriminación** — Cumple con monitoreo: Brecha de error relativo a 3 meses: 7.7% entre verticales y 0.7% entre escalas
- **Privacidad (LOPDP Ecuador)** — Cumple: Seudonimización SHA-256 con sal; mapeo fuera del repositorio; LLM local
- **Responsabilidad / rendición de cuentas** — Cumple: Aprobación y registro del Director de Cuenta antes del envío
- **Autonomía humana** — Cumple: Copiloto: no envía reportes ni modifica campañas
- **Sostenibilidad** — Cumple: Modelos ligeros (CPU) y LLM de 1,5 mil millones de parámetros
- **Trazabilidad y reproducibilidad** — Cumple: Semilla 42, huella SHA-256, Git (v2.0) y DVC

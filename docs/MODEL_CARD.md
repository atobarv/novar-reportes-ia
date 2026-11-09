# Model Card · Sistema de reportes inteligentes (Novar Agencia) · v2.2

**Fecha:** 09/11/2026 · **Responsable:** Julio Alejandro Tobar Valencia

## Componentes y uso previsto
- **Estado actual:** regla de negocio (ROAS < 1,8 Crítico; ≥ 2,8 Óptimo). Sin ML.
- **Proyección del ROAS a 3/6/12 meses:** Ensamble ML (Ridge + Gradient Boosting), con rango del 80 % y probabilidad de deterioro.
- **Detección de anomalías:** Isolation Forest sobre desviaciones robustas de 6 métricas.
- **Redacción:** Qwen/Qwen2.5-1.5B-Instruct con guardrail determinista; versión del prompt P3 · una muestra, magnitudes sin signo y defensa ante instrucciones (v2.1)
  (elegida con datos de validación). Todo reporte requiere aprobación humana.

## Datos
Dataset v2 simulado y calibrado con el v1: 672 meses-cuenta, 14 cuentas, 10/2022–09/2026.

## Entorno y privacidad
Demostración en Google Colab (GPU T4) con archivos en Google Drive y datos simulados: el LLM corre en la sesión sin
API comercial, pero Colab y Drive son servicios de terceros. Despliegue propio previsto con Docker en un equipo de la
agencia. Seudonimización SHA-256 con sal secreta obligatoria (`NOVAR_SAL`, ≥ 16 caracteres, fuera del repositorio).

## Desempeño (cuentas y meses no vistos; detector y perfil de vertical ajustados dentro de cada partición)
- Horizonte 1 meses: MAE 0.288 · mejora vs ingenuo +17.0 %
- Horizonte 3 meses: MAE 0.314 · mejora vs ingenuo +24.5 %
- Horizonte 6 meses: MAE 0.365 · mejora vs ingenuo +18.3 %
- Horizonte 12 meses: MAE 0.344 · mejora vs ingenuo -8.4 %
- Anomalías (por cliente): F1 89.9% vs 43.0% de la regla de umbral.

## Robustez (celda 15b)
- Faltantes: se imputan con la mediana de 6 meses previos, se marcan y el estado del mes queda **provisional**.
- Formatos de exportación, división por plataformas y filas duplicadas: salida idéntica a la original.
- Ingresos ×10 en un mes: lo detecta la salvaguarda de desviación extrema del detector de anomalías.
- Entradas inválidas (archivo vacío, columnas o fechas faltantes): mensaje de error explícito, sin fallo de la aplicación.
- Deriva: se monitorea el PSI de ROAS, CPM, CPA, CTR, tasa de conversión e inversión; reentrenar si PSI ≥ 0,25.

## Limitaciones y cuándo NO usar el sistema
Datos simulados; la proyección no anticipa cambios estructurales (relanzamientos, cambios de presupuesto);
los rangos se amplían con el horizonte. No usar: con cuentas de menos de 12 meses (no hay proyección), con meses
marcados como anómalos o imputados sin verificar la fuente, ni para decidir presupuestos sin revisión humana.
Validar con exportaciones reales antes del uso con clientes.

## Principios éticos (Correa et al., 2023)
- **Transparencia / explicabilidad** — Cumple: Estado por regla explícita; importancia de variables del pronóstico; factores (estacionalidad, tendencia) en cada proyección
- **Confiabilidad y seguridad** — Cumple: Guardrail de existencia, atribución y dirección; plantilla de respaldo; rangos del 80 % calibrados
- **Justicia y no discriminación** — Cumple con monitoreo: Brecha de error relativo a 3 meses: 7.7% entre verticales y 1.1% entre escalas
- **Privacidad (LOPDP Ecuador)** — Cumple: Seudonimización SHA-256 con sal secreta obligatoria; mapeo fuera del repositorio; LLM sin API comercial (demostración en Colab/Drive; despliegue propio con Docker)
- **Responsabilidad / rendición de cuentas** — Cumple: Aprobación y registro del Director de Cuenta antes del envío
- **Autonomía humana** — Cumple: Copiloto: no envía reportes ni modifica campañas
- **Sostenibilidad** — Cumple: Modelos ligeros (CPU) y LLM de 1,5 mil millones de parámetros
- **Trazabilidad y reproducibilidad** — Cumple: Semilla 42, huella SHA-256, Git (v2.2), DVC y versión de prompt registrada
- **Robustez** — Cumple con monitoreo: Pruebas de sensibilidad, estrés, error controlado, deriva y adversariales del LLM (celdas 13b y 15b)

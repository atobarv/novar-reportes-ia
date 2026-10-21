"""
Pruebas automatizadas del prototipo (pytest).
Ejecutar desde la raíz del repositorio:  python -m pytest tests -v
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agents.llm_narrative_generator import (AgenteAnalista, AgenteExtractor, GuardrailNumerico,
                                                bloque_datos, plantilla_determinista)
from src.etl.ingesta_marketing import anonimizar, generar_dataset_sintetico, limpiar
from src.features.feature_engineering import FEATURES, calcular_kpis, preparar_dataset
from src.models.baseline_models import construir_modelos, evaluar_cv


@pytest.fixture(scope="module")
def datos():
    df, _ = limpiar(generar_dataset_sintetico(seed=42))
    df = anonimizar(df, "sal-prueba")
    df, _ = preparar_dataset(df)
    return df


@pytest.fixture(scope="module")
def modelo(datos):
    pipe, feats = construir_modelos()["Random Forest + RobustScaler (ajustado)"]
    return pipe.fit(datos[feats], datos["estado_rendimiento"])


def test_kpis_deterministas():
    d = pd.DataFrame({"gasto_usd": [100.0], "impresiones": [20000], "clics": [400],
                      "conversiones": [10], "ingresos_usd": [350.0]})
    k = calcular_kpis(d).iloc[0]
    assert k["roas"] == pytest.approx(3.5)
    assert k["cpa_usd"] == pytest.approx(10.0)
    assert k["ctr"] == pytest.approx(2.0)
    assert k["cvr"] == pytest.approx(2.5)
    assert k["profit_usd"] == pytest.approx(250.0)
    assert k["cpm_usd"] == pytest.approx(5.0)


def test_kpis_sin_conversiones_no_divide_por_cero():
    d = pd.DataFrame({"gasto_usd": [80.0], "impresiones": [1000], "clics": [0],
                      "conversiones": [0], "ingresos_usd": [0.0]})
    k = calcular_kpis(d).iloc[0]
    assert np.isfinite(k[["roas", "cpa_usd", "cvr", "ctr"]].astype(float)).all()
    assert k["cpa_usd"] == pytest.approx(80.0)


def test_limpieza_formatos_y_fuga():
    crudo = pd.DataFrame({"Cliente": ["A", "A"], "Amount Spent": ["$1.234,50", "$1,000.00"],
                          "Impressions": [1000, 2000], "Clicks": [10, 20], "Results": [1, 2],
                          "Revenue": ["2.000,00", "3000"], "status_code": [0, 1]})
    df, calidad = limpiar(crudo)
    assert list(df["gasto_usd"]) == [1234.5, 1000.0]
    assert "status_code" not in df.columns
    assert calidad["columnas_excluidas_por_fuga"] == ["status_code"]


def test_seudonimizacion_consistente_y_no_reversible():
    df = pd.DataFrame({"cliente": ["Clínica Uno", "Clínica Uno", "Tienda Dos"]})
    a = anonimizar(df, "s1")
    assert a["cliente"].nunique() == 2
    assert a["cliente"].str.match(r"^CLI-[0-9a-f]{8}$").all()
    assert not a["cliente"].str.contains("Clínica").any()
    assert set(anonimizar(df, "s2")["cliente"]) != set(a["cliente"])


def test_modelo_supera_umbral(datos):
    pipe, feats = construir_modelos()["Random Forest + RobustScaler (ajustado)"]
    r, _ = evaluar_cv(pipe, datos[feats], datos["estado_rendimiento"])
    assert r["val_accuracy"] >= 0.85
    assert r["val_f1_weighted"] >= 0.85


def test_guardrail_detecta_cifras_inventadas_y_mal_atribuidas(datos, modelo):
    ex, an = AgenteExtractor(datos), AgenteAnalista(modelo)
    h, fila = ex.consolidar(datos["cliente"].iloc[0])
    d = an.diagnosticar(h, fila, con_shap=False)
    g = GuardrailNumerico(bloque_datos(h, d), h, d)
    assert g.verificar(plantilla_determinista(h, d))[0]
    assert not g.verificar("El ROAS alcanzó 987.65 este periodo.")[0]           # cifra inexistente
    assert not g.verificar(f"El ROAS fue de ${h['inversion']:,.2f}.")[0]          # métrica y unidad erróneas
    assert not g.verificar(f"El CPA fue de ${h['ingresos']:,.2f}.")[0]            # cifra de otra métrica


def test_reporte_contiene_estado_y_recomendacion(datos, modelo):
    ex, an = AgenteExtractor(datos), AgenteAnalista(modelo)
    h, fila = ex.consolidar(datos["cliente"].iloc[-1])
    d = an.diagnosticar(h, fila, con_shap=False)
    texto = plantilla_determinista(h, d)
    assert d["estado"] in texto and "Se recomienda" in texto
    assert 0 < d["confianza"] <= 100

"""
Pruebas automatizadas del prototipo v2.0 (pytest).
Ejecutar desde la raíz del repositorio:  python -m pytest tests -v
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agents.llm_narrative_generator import (AgenteAnalista, AgenteExtractor, AgentePronostico,  # noqa: E402
                                                GuardrailNumerico, bloque_datos, plantilla_determinista)
from src.etl.ingesta_marketing import anonimizar, limpiar  # noqa: E402
from src.etl.simulador_temporal import generar_dataset_temporal  # noqa: E402
from src.features.feature_engineering import calcular_kpis, preparar_dataset  # noqa: E402
from src.models.anomalias import DetectorAnomalias, caracteristicas, imputar_anomalias  # noqa: E402
from src.models.pronostico import Pronosticador, _particiones, construir_pares  # noqa: E402
from src.rules.reglas_negocio import estado_por_regla, es_deterioro  # noqa: E402


@pytest.fixture(scope="module")
def datos():
    crudo, verdad = generar_dataset_temporal(seed=42)
    df, _ = limpiar(crudo)
    df, _ = preparar_dataset(df)
    return df, verdad


@pytest.fixture(scope="module")
def sistema(datos):
    df, _ = datos
    F = caracteristicas(df)
    det = DetectorAnomalias().ajustar(F)
    FA = det.puntuar(F)
    dm = imputar_anomalias(df, FA["es_anomalia"])
    pr = Pronosticador("Ridge con interacciones (global)").ajustar(dm)
    return df, FA, dm, pr


def test_kpis_deterministas():
    d = pd.DataFrame({"gasto_usd": [100.0], "impresiones": [20000], "clics": [400],
                      "conversiones": [10], "ingresos_usd": [350.0]})
    k = calcular_kpis(d).iloc[0]
    assert k["roas"] == pytest.approx(3.5) and k["cpa_usd"] == pytest.approx(10.0)
    assert k["ctr"] == pytest.approx(2.0) and k["cvr"] == pytest.approx(2.5)


def test_regla_de_estado_y_deterioro():
    assert list(estado_por_regla([1.79, 1.8, 2.79, 2.8])) == ["Crítico", "Alerta", "Alerta", "Óptimo"]
    assert list(es_deterioro(["Óptimo", "Alerta", "Crítico"], ["Alerta", "Óptimo", "Crítico"])) == [True, False, False]


def test_limpieza_formatos_y_fuga():
    crudo = pd.DataFrame({"Cliente": ["A", "A"], "Amount Spent": ["$1.234,50", "$1,000.00"],
                          "Impressions": [1000, 2000], "Clicks": [10, 20], "Results": [1, 2],
                          "Revenue": ["2.000,00", "3000"], "status_code": [0, 1]})
    df, calidad = limpiar(crudo)
    assert list(df["gasto_usd"]) == [1234.5, 1000.0]
    assert calidad["columnas_excluidas_por_fuga"] == ["status_code"]


def test_seudonimizacion():
    df = pd.DataFrame({"cliente": ["Clínica Uno", "Clínica Uno", "Tienda Dos"]})
    a = anonimizar(df, "s1")
    assert a["cliente"].nunique() == 2 and a["cliente"].str.match(r"^CLI-[0-9a-f]{8}$").all()


def test_variables_sin_fuga_temporal(datos):
    """Las variables de un origen t no cambian si se borran los meses posteriores a t."""
    df, _ = datos
    cta = df["cliente"].iloc[0]
    corte = pd.to_datetime(df["fecha"]).sort_values().unique()[30]
    completo = construir_pares(df, cuentas={cta})
    truncado = construir_pares(df[pd.to_datetime(df["fecha"]) <= corte], cuentas={cta}, incluir_futuro=True)
    cols = ["rel_t", "rel_media6", "pendiente24", "perfil_vertical", "perfil_cuenta", "rel_estacional"]
    a = completo[completo["fecha_origen"] == corte].set_index("h")[cols]
    b = truncado[truncado["fecha_origen"] == corte].set_index("h")[cols]
    pd.testing.assert_frame_equal(a, b.loc[a.index])


def test_particiones_por_cliente_y_fecha(datos):
    df, _ = datos
    P = construir_pares(df)
    for tr, te in _particiones(P, "cliente+temporal", corte=35):
        assert set(P.loc[tr, "cliente"]).isdisjoint(set(P.loc[te, "cliente"]))
        assert P.loc[tr, "t_objetivo"].max() <= 35 <= P.loc[te, "t"].min()


def test_detector_supera_regla(datos, sistema):
    df, verdad = datos
    _, FA, _, _ = sistema
    v = pd.Series(list(zip(FA["cliente"], pd.to_datetime(FA["fecha"]).dt.strftime("%Y-%m")))).isin(
        set(zip(verdad["cliente_id"], verdad["periodo"]))).values
    recall = (FA["es_anomalia"].values & v).sum() / v.sum()
    assert recall >= 0.6


def test_proyeccion_3_6_12_con_rango(sistema):
    df, _, dm, pr = sistema
    P = pr.proyectar(dm, (3, 6, 12))
    assert set(P["h"]) == {3, 6, 12} and len(P) == 3 * df["cliente"].nunique()
    assert (P["limite_inferior"] <= P["proyeccion"]).all() and (P["proyeccion"] <= P["limite_superior"]).all()
    assert P["prob_deterioro"].between(0, 1).all()


def test_guardrail_existencia_atribucion_direccion(sistema):
    df, FA, dm, pr = sistema
    cta = df["cliente"].iloc[0]
    h = AgenteExtractor(df).consolidar(cta)
    p = AgentePronostico(pr, dm).proyectar(cta)
    d = AgenteAnalista(FA).diagnosticar(h, p)
    g = GuardrailNumerico(bloque_datos(h, d, p), h, p)
    assert g.verificar(plantilla_determinista(h, d, p))[0]
    assert not g.verificar("El ROAS alcanzó 987.65 este mes.")[0]
    assert not g.verificar(f"El ROAS fue de ${h['inversion']:,.2f}.")[0]
    v = h["var_roas"]
    verbo = "cayó" if v >= 0 else "subió"
    assert not g.verificar(f"El ROAS {verbo} {v:+.1f} % frente al mes anterior.")[0]
    assert not g.verificar(f"El ROAS cambió {-v:+.1f} % frente al mes anterior.")[0]

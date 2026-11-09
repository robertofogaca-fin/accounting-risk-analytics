import numpy as np
import pandas as pd

from accounting_risk.avaliacao import comparar_metodos, fila_exclusiva_ml, recall_por_tipo
from accounting_risk.data_generator import gerar_razao
from accounting_risk.ml import FEATURES_BRUTAS, combinar, construir_features, pontuar_isolation_forest
from accounting_risk.scoring import aplicar_detectores


def _razao():
    return gerar_razao(n=5000, n_por_tipo=30, seed=7, desconhecidas=True)


def test_features_brutas_e_engenheiradas():
    df = _razao()
    brutas = construir_features(df, engenheiradas=False)
    eng = construir_features(df, engenheiradas=True)
    assert list(brutas.columns) == FEATURES_BRUTAS
    assert set(brutas.columns) < set(eng.columns)
    assert np.isfinite(eng.to_numpy(dtype=float)).all()


def test_isolation_forest_e_reprodutivel_e_nao_usa_rotulo():
    df = _razao()
    a = pontuar_isolation_forest(df, seed=1)
    b = pontuar_isolation_forest(df.drop(columns=["flag_anomalia", "tipo_anomalia"]), seed=1)
    assert a.equals(b)


def test_engenharia_de_atributos_melhora_o_isolation_forest():
    df = _razao()
    scores = {
        "bruto": pontuar_isolation_forest(df, engenheiradas=False),
        "eng": pontuar_isolation_forest(df, engenheiradas=True),
    }
    tabela = comparar_metodos(df, scores).set_index("metodo")
    assert tabela.loc["eng", "auc"] > tabela.loc["bruto", "auc"] + 0.1


def test_combinar_mantem_ordem_do_score_e_usa_ml_para_desempatar():
    regras = pd.Series([0.0, 0.0, 50.0])
    ml = pd.Series([0.9, 0.1, 0.2])
    c = combinar(regras, ml, peso_ml=0.5)
    assert c.idxmax() == 2 and c[0] > c[1]


def test_comparar_metodos_com_score_perfeito():
    df = _razao()
    perfeito = df["flag_anomalia"].astype(float)
    t = comparar_metodos(df, {"perfeito": perfeito}).iloc[0]
    assert t["auc"] == 1.0 and t["ap"] == 1.0
    r = recall_por_tipo(df, {"perfeito": perfeito})["perfeito"]
    assert (r == 1.0).all()


def test_fila_exclusiva_do_ml_acha_a_anomalia_desconhecida():
    df = _razao()
    p = aplicar_detectores(df)
    fila = fila_exclusiva_ml(df, p["score"], pontuar_isolation_forest(df), tamanhos=(30,)).iloc[0]
    assert fila["precision"] >= 0.3 and fila["lift"] >= 20
    assert fila["tipos_encontrados"].startswith("centro_custo_incomum")

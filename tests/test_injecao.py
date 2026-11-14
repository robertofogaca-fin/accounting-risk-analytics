import numpy as np
import pandas as pd
import pytest

from accounting_risk.enriquecimento import NIVEIS_ALCADA, atribuir_contrapartes, gerar_alcadas
from accounting_risk.injecao import TIPO, ParametrosFracionamento, injetar_fracionamento


@pytest.fixture
def base():
    rng = np.random.default_rng(0)
    n = 5_000
    contas = ["1.1.1.10.00-1", "2.1.1.10.00-2", "4.0.0.00.00-0"]
    return pd.DataFrame(
        {
            "id_lancamento": np.arange(1, n + 1),
            "data_lancamento": pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 365, n), unit="D"),
            "conta_cosif": rng.choice(contas, n),
            "centro_custo": rng.choice(["FIN", "TI", "RH"], n),
            "valor": np.round(rng.lognormal(8.5, 0.7, n), 2),
            "usuario": rng.choice([f"usuario{i:03d}" for i in range(20)], n),
            "historico": "Lancamento contabil rotineiro",
            "documento": [f"DOC{i:07d}" for i in range(1, n + 1)],
            "flag_anomalia": 0,
            "tipo_anomalia": "NORMAL",
        }
    )


@pytest.fixture
def params():
    return ParametrosFracionamento(eventos=40, seed=7)


@pytest.fixture
def resultado(base, params):
    alcadas = gerar_alcadas(base["usuario"])
    enriquecida = atribuir_contrapartes(base)
    com_fraude, gabarito = injetar_fracionamento(enriquecida, alcadas, params)
    return enriquecida, alcadas, com_fraude, gabarito


def test_contraparte_em_todas_as_linhas_e_reprodutivel(base):
    a, b = atribuir_contrapartes(base), atribuir_contrapartes(base)
    assert a["contraparte"].notna().all()
    assert a["contraparte"].equals(b["contraparte"])


def test_contrapartes_concentradas(base):
    share = atribuir_contrapartes(base)["contraparte"].value_counts(normalize=True)
    assert share.head(int(len(share) * 0.1)).sum() > 0.3  # poucos concentram o volume


def test_alcadas_cobrem_todos_os_usuarios(base):
    alcadas = gerar_alcadas(base["usuario"])
    assert set(alcadas["usuario"]) == set(base["usuario"])
    assert set(alcadas["nivel"]) <= set(NIVEIS_ALCADA)


def test_parcelas_abaixo_da_alcada_e_soma_acima(resultado, params):
    _, _, base, gab = resultado
    fraude = base[base["tipo_anomalia"] == TIPO].merge(gab[["documento", "evento", "limite_alcada"]], on="documento")
    assert (fraude["valor"] < fraude["limite_alcada"]).all()
    assert (fraude["valor"] >= params.faixa_parcela[0] * fraude["limite_alcada"] - 0.01).all()
    por_evento = fraude.groupby("evento")
    assert (por_evento["valor"].sum() > por_evento["limite_alcada"].first()).all()
    assert por_evento.size().between(params.parcelas_min, params.parcelas_max).all()


def test_evento_mantem_usuario_contraparte_conta_e_janela(resultado, params):
    _, _, base, gab = resultado
    fraude = base[base["tipo_anomalia"] == TIPO].merge(gab[["documento", "evento"]], on="documento")
    g = fraude.groupby("evento")
    for coluna in ["usuario", "contraparte", "conta_cosif", "centro_custo"]:
        assert (g[coluna].nunique() == 1).all(), coluna
    janela = (g["data_lancamento"].max() - g["data_lancamento"].min()).dt.days
    assert (janela <= params.janela_dias_max).all()


def test_sem_vazamento_por_coluna_ou_posicao(resultado):
    enriquecida, _, base, _ = resultado
    assert list(base.columns) == list(enriquecida.columns)  # nenhuma coluna exclusiva da fraude
    fraude = base[base["tipo_anomalia"] == TIPO]
    assert set(fraude["historico"]) <= set(enriquecida["historico"])
    assert base["documento"].is_unique  # documentos novos não colidem
    assert base["id_lancamento"].tolist() == list(range(1, len(base) + 1))
    # injetadas não ficam agrupadas no fim do arquivo
    posicoes = fraude.index.to_numpy() / len(base)
    assert posicoes.mean() == pytest.approx(0.5, abs=0.15)


def test_gabarito_bate_com_a_base(resultado):
    _, _, base, gab = resultado
    assert len(gab) == (base["tipo_anomalia"] == TIPO).sum()
    assert gab["id_lancamento"].notna().all()


def test_reprodutivel_com_a_mesma_seed(base, params):
    alcadas = gerar_alcadas(base["usuario"])
    enriquecida = atribuir_contrapartes(base)
    a, _ = injetar_fracionamento(enriquecida, alcadas, params)
    b, _ = injetar_fracionamento(enriquecida, alcadas, params)
    pd.testing.assert_frame_equal(a, b)

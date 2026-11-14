import pandas as pd
import pytest

from accounting_risk.fracionamento import score_fracionamento

ALCADAS = pd.DataFrame({"usuario": ["ana", "bia"], "limite_alcada": [10_000.0, 40_000.0]})


def _df(linhas):
    return pd.DataFrame(
        linhas, columns=["data_lancamento", "usuario_criacao", "contraparte", "conta_cosif", "valor"]
    ).assign(data_lancamento=lambda d: pd.to_datetime(d["data_lancamento"]))


def test_parcelas_proximas_do_limite_somando_acima_sao_sinalizadas():
    df = _df([
        ("2025-03-03", "ana", "CP1", "C1", 9_500),
        ("2025-03-05", "ana", "CP1", "C1", 9_200),
        ("2025-03-20", "ana", "CP1", "C1", 9_000),  # fora da janela de 7 dias
    ])
    s = score_fracionamento(df, ALCADAS)
    assert s.iloc[0] > 0 and s.iloc[1] > 0
    assert s.iloc[2] == 0
    assert s.iloc[0] == pytest.approx(1.87 * (1.87 / 2))


def test_soma_abaixo_da_alcada_nao_sinaliza():
    df = _df([("2025-03-03", "ana", "CP1", "C1", 3_000), ("2025-03-04", "ana", "CP1", "C1", 4_000)])
    assert (score_fracionamento(df, ALCADAS) == 0).all()


def test_lancamento_acima_da_alcada_fica_de_fora():
    # o de 12 mil já exige aprovação superior: não é fracionamento
    df = _df([("2025-03-03", "ana", "CP1", "C1", 12_000), ("2025-03-04", "ana", "CP1", "C1", 2_000)])
    assert (score_fracionamento(df, ALCADAS) == 0).all()


def test_chave_separa_usuario_contraparte_e_conta():
    df = _df([
        ("2025-03-03", "ana", "CP1", "C1", 9_000),
        ("2025-03-04", "ana", "CP2", "C1", 9_000),
        ("2025-03-04", "ana", "CP1", "C2", 9_000),
        ("2025-03-05", "bia", "CP1", "C1", 9_000),
    ])
    assert (score_fracionamento(df, ALCADAS) == 0).all()


def test_alcada_por_usuario():
    # 2 x 9 mil passa da alçada da ana (10 mil), mas não da bia (40 mil)
    df = _df([
        ("2025-03-03", "ana", "CP1", "C1", 9_000), ("2025-03-04", "ana", "CP1", "C1", 9_000),
        ("2025-03-03", "bia", "CP1", "C1", 9_000), ("2025-03-04", "bia", "CP1", "C1", 9_000),
    ])
    s = score_fracionamento(df, ALCADAS)
    assert (s.iloc[:2] > 0).all() and (s.iloc[2:] == 0).all()


def test_mais_proximo_do_limite_pontua_mais():
    df = _df([
        ("2025-03-03", "ana", "CP1", "C1", 9_800), ("2025-03-04", "ana", "CP1", "C1", 9_700),
        ("2025-03-03", "ana", "CP9", "C1", 6_000), ("2025-03-04", "ana", "CP9", "C1", 5_000),
    ])
    s = score_fracionamento(df, ALCADAS)
    assert s.iloc[0] > s.iloc[2] > 0


def test_usuario_sem_alcada_cadastrada_e_ignorado():
    df = _df([("2025-03-03", "zeca", "CP1", "C1", 9_000), ("2025-03-04", "zeca", "CP1", "C1", 9_000)])
    assert (score_fracionamento(df, ALCADAS) == 0).all()


def test_grupos_suspeitos_uma_linha_por_grupo():
    from accounting_risk.fracionamento import grupos_suspeitos

    df = _df([
        ("2025-03-03", "ana", "CP1", "C1", 9_500),
        ("2025-03-05", "ana", "CP1", "C1", 9_200),
        ("2025-03-06", "ana", "CP1", "C1", 9_100),
        ("2025-04-10", "ana", "CP1", "C1", 9_000),  # isolado: não forma grupo
    ])
    grupos, grupo_por_linha = grupos_suspeitos(df, ALCADAS)
    assert len(grupos) == 1
    linha = grupos.iloc[0]
    assert linha["lancamentos"] == 3 and linha["soma"] == pytest.approx(27_800)
    assert linha["excedente"] == pytest.approx(17_800)
    assert grupo_por_linha.dropna().size == 3

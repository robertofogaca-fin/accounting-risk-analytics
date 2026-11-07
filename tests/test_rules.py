import pandas as pd

from accounting_risk import rules
from accounting_risk.data_generator import TIPOS_CONHECIDOS, TIPOS_SUTIS, gerar_razao
from accounting_risk.plano_contas import CODIGOS
from accounting_risk.scoring import aplicar_regras, resumo_por_tipo


def _base(**kwargs):
    linha = {
        "data_lancamento": pd.Timestamp("2026-03-10"),  # terça-feira
        "conta_cosif": CODIGOS[0],
        "valor": 1234.56,
        "tipo": "D",
        "centro_custo": "CC100",
        "usuario_criacao": "usuario01",
        "usuario_aprovacao": "usuario02",
        "origem": "automatico",
        "historico": "Pagamento de fornecedor ABC",
        "flag_anomalia": 0,
        "tipo_anomalia": "NORMAL",
    }
    linha.update(kwargs)
    return pd.DataFrame([linha])


def test_fim_de_semana():
    assert rules.fim_de_semana(_base(data_lancamento=pd.Timestamp("2026-03-14"))).iloc[0]
    assert not rules.fim_de_semana(_base()).iloc[0]


def test_valor_redondo():
    assert rules.valor_redondo(_base(valor=5_000.0)).iloc[0]
    assert not rules.valor_redondo(_base(valor=5_000.37)).iloc[0]
    assert not rules.valor_redondo(_base(valor=500.0)).iloc[0]


def test_duplicidade():
    df = pd.concat([_base(), _base(), _base(valor=99.9)], ignore_index=True)
    assert rules.duplicidade(df).tolist() == [True, True, False]


def test_manual_alto():
    assert rules.manual_alto(_base(origem="manual", valor=200_000)).iloc[0]
    assert not rules.manual_alto(_base(origem="automatico", valor=200_000)).iloc[0]


def test_sem_segregacao():
    assert rules.sem_segregacao(_base(usuario_aprovacao="usuario01")).iloc[0]
    assert not rules.sem_segregacao(_base()).iloc[0]


def test_historico_vazio():
    assert rules.historico_vazio(_base(historico="")).iloc[0]
    assert not rules.historico_vazio(_base()).iloc[0]


def test_conta_invalida():
    assert rules.conta_invalida(_base(conta_cosif="1.1.1.00.00-3")).iloc[0]  # DV errado
    assert rules.conta_invalida(_base(conta_cosif="1.7.0.00.00-0")).iloc[0]  # fora do plano
    assert not rules.conta_invalida(_base()).iloc[0]


def test_lancamento_limpo_tem_score_zero():
    assert aplicar_regras(_base())["score"].iloc[0] == 0


def test_regras_pegam_os_tipos_conhecidos_mas_nao_os_sutis():
    df = gerar_razao(n=3000, n_por_tipo=15, seed=3)
    tabela = resumo_por_tipo(aplicar_regras(df))
    assert (tabela.loc[TIPOS_CONHECIDOS, "recall"] == 1.0).all()
    assert (tabela.loc[TIPOS_SUTIS, "recall"] < 0.2).all()

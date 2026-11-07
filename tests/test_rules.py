import pandas as pd

from accounting_risk import rules
from accounting_risk.data_generator import TIPOS_CONHECIDOS, TIPOS_SUTIS, gerar_razao
from accounting_risk.data_generator import CODIGOS as OPERACIONAIS
from accounting_risk.plano_contas import CODIGOS_SET, PLANO, formatar_codigo
from accounting_risk.scoring import aplicar_regras, resumo_por_tipo


def _base(**kwargs):
    linha = {
        "data_lancamento": pd.Timestamp("2026-03-10"),  # terça-feira
        "conta_cosif": OPERACIONAIS[0],
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


def test_dia_nao_util():
    assert rules.dia_nao_util(_base(data_lancamento=pd.Timestamp("2026-03-14"))).iloc[0]  # sábado
    assert rules.dia_nao_util(_base(data_lancamento=pd.Timestamp("2026-04-21"))).iloc[0]  # Tiradentes
    assert rules.dia_nao_util(_base(data_lancamento=pd.Timestamp("2026-02-17"))).iloc[0]  # Carnaval
    assert not rules.dia_nao_util(_base(data_lancamento=pd.Timestamp("2026-02-18"))).iloc[0]  # Cinzas: expediente parcial
    assert not rules.dia_nao_util(_base(data_lancamento=pd.Timestamp("2025-12-31"))).iloc[0]  # fechamento do ano
    assert not rules.dia_nao_util(_base()).iloc[0]


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
    fora_do_elenco = formatar_codigo("199999999")  # estrutura e DV válidos, mas inexistente
    assert fora_do_elenco not in CODIGOS_SET
    assert rules.conta_invalida(_base(conta_cosif=fora_do_elenco)).iloc[0]
    assert rules.conta_invalida(_base(conta_cosif="1.1.1.10.00.00-3")).iloc[0]  # DV errado
    assert not rules.conta_invalida(_base()).iloc[0]


def test_conta_fora_vigencia():
    encerrada = PLANO.dropna(subset=["vigencia_fim"]).iloc[0]
    futura = PLANO[PLANO["vigencia_ini"] >= "2027-01-01"].iloc[0]
    data = pd.Timestamp("2027-06-15") if encerrada["vigencia_fim"] < pd.Timestamp("2027-01-01") else pd.Timestamp("2026-03-10")
    assert rules.conta_fora_vigencia(_base(conta_cosif=encerrada["codigo"], data_lancamento=data)).iloc[0]
    assert rules.conta_fora_vigencia(_base(conta_cosif=futura["codigo"])).iloc[0]  # usada em 2026, vigente só em 2027
    assert not rules.conta_fora_vigencia(_base()).iloc[0]


def test_lancamento_limpo_tem_score_zero():
    assert aplicar_regras(_base())["score"].iloc[0] == 0


def test_regras_pegam_os_tipos_conhecidos_mas_nao_os_sutis():
    df = gerar_razao(n=3000, n_por_tipo=15, seed=3)
    tabela = resumo_por_tipo(aplicar_regras(df))
    assert (tabela.loc[TIPOS_CONHECIDOS, "recall"] == 1.0).all()
    assert (tabela.loc[TIPOS_SUTIS, "recall"] < 0.2).all()

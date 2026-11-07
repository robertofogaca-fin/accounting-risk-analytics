import pandas as pd

from accounting_risk.plano_contas import (
    CODIGOS, CODIGOS_SET, PLANO, calcular_dv, codigo_valido, formatar_codigo, so_digitos,
)

# Códigos oficiais da estrutura anterior (7 dígitos), usados para conferir o algoritmo do DV
LEGADO = [
    "1.0.0.00.00-7", "2.0.0.00.00-4", "3.0.0.00.00-1", "4.0.0.00.00-8", "6.0.0.00.00-2",
    "7.0.0.00.00-9", "8.0.0.00.00-6", "9.0.0.00.00-3", "3.0.1.00.00-4", "3.0.9.99.00-2",
    "9.0.1.00.00-6", "9.0.9.99.00-4", "9.0.7.75.00-8", "3.0.7.75.00-6", "3.0.7.99.00-6",
    "9.1.0.00.00-2", "9.9.0.00.00-4",
]


def test_dv_confere_com_codigos_oficiais_da_estrutura_anterior():
    for c in LEGADO:
        d = so_digitos(c)
        assert calcular_dv(d[:7]) == int(d[7]), c


def test_todas_as_contas_do_plano_tem_dv_valido():
    assert len(PLANO) == 4401
    assert all(codigo_valido(c) for c in CODIGOS)
    assert len(CODIGOS_SET) == len(CODIGOS)


def test_exemplo_de_conta_disponibilidades():
    assert codigo_valido("1.1.0.00.00.00-2")
    assert PLANO.set_index("codigo").loc["1.1.0.00.00.00-2", "descricao"] == "DISPONIBILIDADES"
    assert formatar_codigo("110000000") == "1.1.0.00.00.00-2"
    assert formatar_codigo("1100000002") == "1.1.0.00.00.00-2"


def test_dv_errado_e_formato_invalido():
    assert not codigo_valido("1.1.0.00.00.00-3")
    assert not codigo_valido("1.1.0.00.00-6")  # estrutura anterior (7 dígitos)
    assert not codigo_valido("abc")


def test_niveis_e_folhas():
    grupos = PLANO[PLANO["nivel"] == 1]
    assert len(grupos) == 8 and not grupos["folha"].any()
    assert PLANO.loc[PLANO["folha"], "nivel"].min() >= 4


def test_vigencia_fim_inclusiva_ate_o_fim_do_mes():
    fim = PLANO.dropna(subset=["vigencia_fim"]).iloc[0]["vigencia_fim"]
    assert fim == fim + pd.offsets.MonthEnd(0)

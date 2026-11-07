from accounting_risk.plano_contas import CODIGOS, calcular_dv, codigo_valido, formatar_codigo

# Códigos oficiais do Cosif (Res. BCB 92/2021 e críticas do BCB) usados para conferir o algoritmo do DV
OFICIAIS = [
    "1.0.0.00.00-7", "2.0.0.00.00-4", "3.0.0.00.00-1", "4.0.0.00.00-8", "6.0.0.00.00-2",
    "7.0.0.00.00-9", "8.0.0.00.00-6", "9.0.0.00.00-3", "3.0.1.00.00-4", "3.0.9.99.00-2",
    "9.0.1.00.00-6", "9.0.9.99.00-4", "9.0.7.75.00-8", "3.0.7.75.00-6", "3.0.7.99.00-6",
    "9.1.0.00.00-2", "9.9.0.00.00-4",
]


def test_dv_confere_com_codigos_oficiais():
    assert all(codigo_valido(c) for c in OFICIAIS)


def test_dv_errado_e_formato_invalido():
    assert not codigo_valido("3.0.0.00.00-2")
    assert not codigo_valido("30000001")
    assert not codigo_valido("1.1.1.10.00-1")


def test_formatar_codigo():
    assert formatar_codigo("3000000") == "3.0.0.00.00-1"
    assert calcular_dv("9077500") == 8


def test_plano_do_projeto_so_tem_codigos_validos():
    assert all(codigo_valido(c) for c in CODIGOS)
    assert len(set(CODIGOS)) == len(CODIGOS)

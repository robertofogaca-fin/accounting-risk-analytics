from accounting_risk.data_generator import TIPOS_CONHECIDOS, TIPOS_SUTIS, gerar_razao
from accounting_risk.plano_contas import CODIGOS


def test_gerador_e_reprodutivel():
    a = gerar_razao(n=1500, n_por_tipo=10, seed=1)
    b = gerar_razao(n=1500, n_por_tipo=10, seed=1)
    assert a.equals(b)


def test_gerador_injeta_todos_os_tipos_rotulados():
    df = gerar_razao(n=2000, n_por_tipo=10, seed=7)
    contagem = df[df["flag_anomalia"] == 1]["tipo_anomalia"].value_counts()
    assert set(contagem.index) == set(TIPOS_CONHECIDOS + TIPOS_SUTIS)
    assert (contagem == 10).all()
    assert df["id_lancamento"].is_unique


def test_contas_normais_seguem_o_plano_cosif():
    df = gerar_razao(n=2000, n_por_tipo=10, seed=7)
    normais = df[df["flag_anomalia"] == 0]
    assert normais["conta_cosif"].isin(CODIGOS).all()
    invalidas = df[df["tipo_anomalia"] == "conta_invalida"]
    assert (~invalidas["conta_cosif"].isin(CODIGOS)).all()

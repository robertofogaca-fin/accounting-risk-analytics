from accounting_risk.data_generator import TIPOS_CONHECIDOS, TIPOS_SUTIS, gerar_razao
from accounting_risk.plano_contas import CODIGOS_SET
from accounting_risk.rules import conta_fora_vigencia, dia_nao_util


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
    assert normais["conta_cosif"].isin(CODIGOS_SET).all()
    assert not conta_fora_vigencia(normais).any()
    invalidas = df[df["tipo_anomalia"] == "conta_invalida"]
    assert (~invalidas["conta_cosif"].isin(CODIGOS_SET)).all()
    fora = df[df["tipo_anomalia"] == "conta_fora_vigencia"]
    assert fora["conta_cosif"].isin(CODIGOS_SET).all() and conta_fora_vigencia(fora).all()


def test_normais_so_em_dias_uteis_e_anomalia_so_em_dia_nao_util():
    df = gerar_razao(n=2000, n_por_tipo=10, seed=7)
    normais = df[df["flag_anomalia"] == 0]
    assert not dia_nao_util(normais).any()
    anomalas = df[df["tipo_anomalia"] == "dia_nao_util"]
    assert dia_nao_util(anomalas).all()
    assert (anomalas["data_lancamento"].dt.day_of_week < 5).any()  # inclui feriados de dia de semana

import numpy as np
import pandas as pd

from accounting_risk import stats
from accounting_risk.data_generator import TIPOS_CONHECIDOS, TIPOS_SUTIS, gerar_razao
from accounting_risk.plano_contas import CODIGOS
from accounting_risk.scoring import aplicar_detectores, resumo_por_tipo


def _quadro(rng, n, usuario="usuario01", conta=None, datas=None, valores=None):
    conta = conta or CODIGOS[0]
    return pd.DataFrame(
        {
            "data_lancamento": datas if datas is not None else pd.Timestamp("2026-03-10"),
            "conta_cosif": conta,
            "valor": valores if valores is not None else np.round(900 * np.exp(0.9 * rng.standard_normal(n)), 2),
            "usuario_criacao": usuario,
        }
    )


def test_zscore_robusto_ignora_outlier_na_estimativa():
    s = pd.Series([10.0, 11, 9, 10, 12, 8, 1000])
    z = stats.zscore_robusto(s, pd.Series(["a"] * 7))
    assert z.iloc[-1] > 50 and z.iloc[:-1].abs().max() < 2


def test_valor_atipico_conta_pega_so_o_outlier():
    rng = np.random.default_rng(0)
    df = _quadro(rng, 500)
    df.loc[10, "valor"] = 900 * np.exp(0.9 * 6)
    sinal = stats.valor_atipico_conta(df)
    assert sinal.iloc[10] and sinal.sum() <= 2


def test_par_usuario_conta_raro():
    rng = np.random.default_rng(1)
    df = pd.concat([_quadro(rng, 300, conta=CODIGOS[0]), _quadro(rng, 1, conta=CODIGOS[3])], ignore_index=True)
    sinal = stats.par_usuario_conta_raro(df)
    assert sinal.iloc[-1] and sinal.sum() == 1


def test_pico_mensal_marca_so_o_mes_anomalo():
    rng = np.random.default_rng(2)
    partes = []
    for m in range(1, 13):
        n = 150 if m == 7 else int(rng.integers(45, 56))
        partes.append(_quadro(rng, n, datas=pd.Timestamp(2026, m, 10)))
    df = pd.concat(partes, ignore_index=True)
    sinal = stats.pico_mensal(df)
    assert set(df.loc[sinal, "data_lancamento"].dt.month) == {7}


def test_desvio_digitos_pega_excesso_do_digito_9():
    rng = np.random.default_rng(3)
    partes = [_quadro(rng, 1500, usuario=f"usuario{i:02d}") for i in range(1, 6)]
    suspeito = _quadro(rng, 150, usuario="usuario01", valores=rng.uniform(9500, 9999.99, 150))
    df = pd.concat(partes + [suspeito], ignore_index=True)
    sinal = stats.desvio_digitos_usuario(df)
    assert sinal.iloc[-150:].mean() > 0.95
    assert df.loc[sinal, "usuario_criacao"].eq("usuario01").all()


def test_benford_mad_distingue_dados_conformes():
    rng = np.random.default_rng(4)
    conforme = 10 ** rng.uniform(0, 5, 20_000)
    concentrado = rng.uniform(9000, 9999, 20_000)
    assert stats.benford_mad(conforme) < 0.006
    assert stats.benford_mad(concentrado) > 0.015


def test_estatistica_completa_pega_todos_os_tipos():
    df = gerar_razao(n=5000, n_por_tipo=30, seed=7)
    tabela = resumo_por_tipo(aplicar_detectores(df))
    assert (tabela.loc[TIPOS_CONHECIDOS + TIPOS_SUTIS, "recall"] >= 0.95).all()

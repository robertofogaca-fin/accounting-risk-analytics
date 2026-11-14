"""Detector de fracionamento abaixo da alçada (v0.6).

Regras fixadas em docs/v0.6_especificacao_fracionamento.md antes da avaliação:
  - só lançamentos abaixo da alçada do usuário;
  - grupo = usuário + contraparte + conta, em janela móvel de `janela_dias` dias corridos;
  - janela suspeita: 2 ou mais lançamentos e soma acima da alçada;
  - score = (soma / alçada) x (média de valor / alçada); cada lançamento fica com o maior score entre as
    janelas suspeitas que começam ou terminam nele.
"""
import numpy as np
import pandas as pd

CHAVE = ("contraparte", "conta_cosif")


def score_fracionamento(
    df: pd.DataFrame,
    alcadas: pd.DataFrame,
    janela_dias: int = 7,
    coluna_usuario: str = "usuario_criacao",
) -> pd.Series:
    """Score de fracionamento por lançamento (0 = sem janela suspeita). Índice igual ao de `df`."""
    limite = df[coluna_usuario].map(alcadas.set_index("usuario")["limite_alcada"])
    razao = (df["valor"] / limite).to_numpy()
    candidato = np.isfinite(razao) & (razao > 0) & (razao < 1)

    sub = pd.DataFrame(
        {
            "grupo": pd.MultiIndex.from_frame(df.loc[candidato, [coluna_usuario, *CHAVE]].astype(str)).factorize()[0]
            if candidato.any()
            else [],
            "dia": (pd.to_datetime(df.loc[candidato, "data_lancamento"]) - pd.Timestamp("1970-01-01")).dt.days.to_numpy(),
            "razao": razao[candidato],
        },
        index=df.index[candidato],
    ).sort_values(["grupo", "dia"], kind="stable")

    score = pd.Series(0.0, index=df.index)
    if sub.empty:
        return score

    deslocamento = int(sub["dia"].max() - sub["dia"].min()) + 2 * janela_dias + 1
    chave = sub["grupo"].to_numpy(np.int64) * deslocamento + (sub["dia"] - sub["dia"].min()).to_numpy(np.int64)
    acumulado = np.concatenate([[0.0], np.cumsum(sub["razao"].to_numpy())])

    melhor = np.zeros(len(sub))
    for inicio_off, fim_off in ((-(janela_dias - 1), 0), (0, janela_dias - 1)):  # termina / começa no lançamento
        lo = np.searchsorted(chave, chave + inicio_off, side="left")
        hi = np.searchsorted(chave, chave + fim_off, side="right")
        n = hi - lo
        soma = acumulado[hi] - acumulado[lo]
        suspeita = (n >= 2) & (soma > 1.0)
        s = np.where(suspeita, soma * (soma / np.maximum(n, 1)), 0.0)
        melhor = np.maximum(melhor, s)

    score.loc[sub.index] = melhor
    return score


def grupos_suspeitos(
    df: pd.DataFrame,
    alcadas: pd.DataFrame,
    janela_dias: int = 7,
    coluna_usuario: str = "usuario_criacao",
    score: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Junta os lançamentos sinalizados em grupos para revisão (uma linha por grupo).

    Dentro de cada usuário + contraparte + conta, os lançamentos sinalizados são agrupados em ordem de data;
    um novo grupo começa quando o lançamento cai a `janela_dias` dias ou mais do primeiro do grupo.
    Retorna (grupos ordenados por score, id do grupo por lançamento sinalizado).
    """
    score = score_fracionamento(df, alcadas, janela_dias, coluna_usuario) if score is None else score
    sinal = df.loc[score > 0, ["data_lancamento", coluna_usuario, *CHAVE, "valor"]].copy()
    if sinal.empty:
        return pd.DataFrame(), pd.Series(dtype=int)
    sinal["data_lancamento"] = pd.to_datetime(sinal["data_lancamento"])
    sinal = sinal.sort_values([coluna_usuario, *CHAVE, "data_lancamento"], kind="stable")

    chave = sinal[[coluna_usuario, *CHAVE]].astype(str).agg("|".join, axis=1).to_numpy()
    dias = (sinal["data_lancamento"] - pd.Timestamp("1970-01-01")).dt.days.to_numpy()
    grupo = np.empty(len(sinal), dtype=np.int64)
    atual, inicio = -1, None
    for i in range(len(sinal)):
        if i == 0 or chave[i] != chave[i - 1] or dias[i] - inicio >= janela_dias:
            atual, inicio = atual + 1, dias[i]
        grupo[i] = atual
    sinal["grupo"] = grupo

    limite = alcadas.set_index("usuario")["limite_alcada"]
    g = sinal.groupby("grupo").agg(
        usuario=(coluna_usuario, "first"),
        contraparte=("contraparte", "first"),
        conta_cosif=("conta_cosif", "first"),
        data_inicio=("data_lancamento", "min"),
        data_fim=("data_lancamento", "max"),
        lancamentos=("valor", "size"),
        soma=("valor", "sum"),
        maior_valor=("valor", "max"),
    )
    g["alcada"] = g["usuario"].map(limite)
    g = g[(g["lancamentos"] >= 2) & (g["soma"] > g["alcada"])]
    g["excedente"] = g["soma"] - g["alcada"]
    g["score"] = (g["soma"] / g["alcada"]) * (g["soma"] / g["lancamentos"] / g["alcada"])
    g = g.sort_values("score", ascending=False)
    return g, sinal["grupo"].where(sinal["grupo"].isin(g.index))

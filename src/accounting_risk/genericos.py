"""Detectores genéricos de raridade, para bases externas (v0.5).

Não dependem de plano de contas, de calendário nem de perfis conhecidos: olham frequências.
Ficam FORA da lista padrão (`scoring.TODOS`) para não mudar os resultados do cenário sintético.
Use `scoring.TODOS + genericos.GENERICOS`.
"""
import numpy as np
import pandas as pd

from .rules import Detector
from .stats import zscore_robusto


def documento_repetido(df: pd.DataFrame) -> pd.Series:
    """Segunda ocorrência em diante de um mesmo número de documento."""
    return df["documento"].duplicated(keep="first")


def centro_custo_raro(df: pd.DataFrame, share_max: float = 0.02, min_conta: int = 100) -> pd.Series:
    """Centro de custo que a conta quase nunca usa (participação dentro da conta abaixo de `share_max`)."""
    por_conta = df.groupby("conta_cosif")["valor"].transform("size")
    por_par = df.groupby(["conta_cosif", "centro_custo"])["valor"].transform("size")
    return (por_conta >= min_conta) & (por_par / por_conta < share_max)


def historico_raro(df: pd.DataFrame, share_max: float = 0.02, max_distintos: int = 50) -> pd.Series:
    """Texto de histórico raro, só quando o histórico é pouco variado (campo quase categórico)."""
    if df["historico"].nunique() > max_distintos:
        return pd.Series(False, index=df.index)
    share = df["historico"].map(df["historico"].value_counts(normalize=True))
    return share < share_max


def usuario_volume_atipico(df: pd.DataFrame, limiar: float = 4.0, min_usuarios: int = 10) -> pd.Series:
    """Usuário cujo volume de lançamentos destoa do volume dos demais (z-score robusto sobre as contagens)."""
    contagem = df["usuario_criacao"].value_counts()
    if len(contagem) < min_usuarios:
        return pd.Series(False, index=df.index)
    z = zscore_robusto(contagem.astype(float), pd.Series("todos", index=contagem.index))
    atipicos = set(z[z > limiar].index)
    return df["usuario_criacao"].isin(atipicos)


GENERICOS = [
    Detector("documento_repetido", "Documento já utilizado em lançamento anterior", 4, documento_repetido, "", ("documento",)),
    Detector("centro_custo_raro", "Centro de custo raro para a conta", 3, centro_custo_raro, "", ("conta_cosif", "centro_custo", "valor")),
    Detector("historico_raro", "Histórico raro em campo pouco variado", 2, historico_raro, "", ("historico",)),
    Detector("usuario_volume_atipico", "Usuário com volume de lançamentos fora do padrão", 3, usuario_volume_atipico, "", ("usuario_criacao",)),
]

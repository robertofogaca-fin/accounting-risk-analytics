"""Detectores baseados em regras determinísticas.

Cada detector recebe o DataFrame do razão e devolve uma Series booleana
(True = lançamento disparou o sinal). O peso reflete a gravidade relativa:
a soma dos pesos disparados vira o score (10 pontos = score 100).
"""
from dataclasses import dataclass
from typing import Callable

import pandas as pd

from .data_generator import LIMITE_MANUAL_ALTO
from .plano_contas import CODIGOS


@dataclass(frozen=True)
class Detector:
    nome: str
    descricao: str
    peso: int
    funcao: Callable[[pd.DataFrame], pd.Series]
    tipo_alvo: str = ""  # tipo de anomalia que o detector pretende pegar (para avaliação)


Regra = Detector  # compatibilidade com a v0.1


def fim_de_semana(df: pd.DataFrame) -> pd.Series:
    return df["data_lancamento"].dt.day_of_week >= 5


def valor_redondo(df: pd.DataFrame, minimo: float = 1_000.0) -> pd.Series:
    return (df["valor"] >= minimo) & ((df["valor"] % 1_000) == 0)


def duplicidade(df: pd.DataFrame) -> pd.Series:
    chave = ["data_lancamento", "conta_cosif", "valor", "tipo", "centro_custo"]
    return df.duplicated(subset=chave, keep=False)


def manual_alto(df: pd.DataFrame, limite: float = LIMITE_MANUAL_ALTO) -> pd.Series:
    return (df["origem"] == "manual") & (df["valor"] > limite)


def sem_segregacao(df: pd.DataFrame) -> pd.Series:
    return df["usuario_criacao"] == df["usuario_aprovacao"]


def historico_vazio(df: pd.DataFrame) -> pd.Series:
    return df["historico"].fillna("").str.strip().str.len() < 5


def conta_invalida(df: pd.DataFrame) -> pd.Series:
    """Conta que não existe no plano de contas COSIF da instituição (inclui DV errado)."""
    return ~df["conta_cosif"].isin(CODIGOS)


REGRAS = [
    Detector("fim_de_semana", "Lançamento em sábado ou domingo", 1, fim_de_semana, "fim_de_semana"),
    Detector("valor_redondo", "Valor redondo (múltiplo de 1.000)", 2, valor_redondo, "valor_redondo"),
    Detector("duplicidade", "Mesma data, conta, valor, tipo e centro de custo", 3, duplicidade, "duplicidade"),
    Detector("manual_alto", "Lançamento manual acima do limite", 4, manual_alto, "manual_alto"),
    Detector("sem_segregacao", "Mesmo usuário cria e aprova", 5, sem_segregacao, "sem_segregacao"),
    Detector("historico_vazio", "Histórico ausente ou muito curto", 1, historico_vazio, "historico_vazio"),
    Detector("conta_invalida", "Conta fora do plano COSIF (código inexistente ou DV errado)", 7, conta_invalida, "conta_invalida"),
]

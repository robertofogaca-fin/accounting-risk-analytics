"""Enriquecimento da base externa para o teste de fraudes que imitam o padrão normal (v0.6).

A base COSIF de 500 mil lançamentos não tem contraparte nem alçada. As duas são criadas aqui como
PREMISSAS do teste, com duas regras para não repetir o vazamento visto na v0.5 (anomalia que se entrega
por um campo que só ela tem):

1. A contraparte existe em TODAS as linhas, com a mesma lógica para lançamentos normais e fraudados.
2. A alçada é cadastro (tabela por usuário), não coluna do razão — como nos sistemas reais.
"""
import numpy as np
import pandas as pd

SEED = 2026

# Contraparte: cada conta tem uma carteira própria, com concentração tipo Zipf
# (poucas contrapartes concentram a maior parte dos lançamentos).
N_CONTRAPARTES_TOTAL = 600
CONTRAPARTES_POR_CONTA = 120
EXPOENTE_ZIPF = 1.1

# Alçada: limite de aprovação própria do usuário. Acima dele, o lançamento exige aprovação superior.
NIVEIS_ALCADA = {
    "ANALISTA": {"limite": 10_000.0, "participacao": 0.70},
    "COORDENADOR": {"limite": 20_000.0, "participacao": 0.25},
    "GERENTE": {"limite": 40_000.0, "participacao": 0.05},
}


def _pesos_zipf(n: int, s: float) -> np.ndarray:
    pesos = 1.0 / np.arange(1, n + 1) ** s
    return pesos / pesos.sum()


def carteiras_por_conta(contas, seed: int = SEED) -> dict:
    """Sorteia, para cada conta, uma carteira ordenada de contrapartes (a ordem define o peso Zipf)."""
    rng = np.random.default_rng(seed)
    universo = np.array([f"CP{i:05d}" for i in range(1, N_CONTRAPARTES_TOTAL + 1)])
    return {
        conta: rng.choice(universo, size=CONTRAPARTES_POR_CONTA, replace=False)
        for conta in sorted(pd.unique(pd.Series(contas)))
    }


def sortear_contrapartes(contas: pd.Series, carteiras: dict, rng: np.random.Generator) -> pd.Series:
    """Atribui uma contraparte a cada linha, a partir da carteira da conta (mesma regra para qualquer linha)."""
    pesos = _pesos_zipf(CONTRAPARTES_POR_CONTA, EXPOENTE_ZIPF)
    resultado = pd.Series(index=contas.index, dtype=object)
    for conta, idx in contas.groupby(contas).groups.items():
        resultado.loc[idx] = rng.choice(carteiras[conta], size=len(idx), p=pesos)
    return resultado


def atribuir_contrapartes(df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Devolve uma cópia da base com a coluna `contraparte` preenchida em todas as linhas."""
    rng = np.random.default_rng(seed + 1)
    saida = df.copy()
    saida["contraparte"] = sortear_contrapartes(saida["conta_cosif"], carteiras_por_conta(saida["conta_cosif"], seed), rng)
    return saida


def gerar_alcadas(usuarios, seed: int = SEED) -> pd.DataFrame:
    """Tabela de alçadas: um nível por usuário, sorteado nas participações de `NIVEIS_ALCADA`."""
    rng = np.random.default_rng(seed + 2)
    usuarios = sorted(pd.unique(pd.Series(usuarios)))
    niveis = list(NIVEIS_ALCADA)
    sorteio = rng.choice(niveis, size=len(usuarios), p=[NIVEIS_ALCADA[n]["participacao"] for n in niveis])
    return pd.DataFrame(
        {
            "usuario": usuarios,
            "nivel": sorteio,
            "limite_alcada": [NIVEIS_ALCADA[n]["limite"] for n in sorteio],
        }
    )

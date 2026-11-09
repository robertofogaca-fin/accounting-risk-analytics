"""Carrega uma base de lançamentos real ou externa para o formato que os detectores esperam.

Colunas obrigatórias: data_lancamento, conta_cosif, valor.
Colunas opcionais (detectores que dependem delas ficam desligados se faltarem):
  tipo (D/C), centro_custo, usuario_criacao, usuario_aprovacao, origem (manual/automatico),
  historico, documento. Para validação com rótulo: flag_anomalia (0/1) e tipo_anomalia.
"""
from pathlib import Path

import pandas as pd

OBRIGATORIAS = ["data_lancamento", "conta_cosif", "valor"]
OPCIONAIS = [
    "tipo", "centro_custo", "usuario_criacao", "usuario_aprovacao", "origem", "historico", "documento",
    "flag_anomalia", "tipo_anomalia",
]


def _separador(caminho: Path) -> str:
    with open(caminho, encoding="utf-8-sig") as f:
        primeira = f.readline()
    return ";" if primeira.count(";") > primeira.count(",") else ","


def carregar_razao(caminho, mapeamento: dict | None = None, decimal: str = ".", dayfirst: bool = False) -> pd.DataFrame:
    """Lê CSV ou Excel e renomeia colunas.

    mapeamento: {nome_no_arquivo: nome_padrao}, ex.: {"usuario": "usuario_criacao", "DT_LANC": "data_lancamento"}.
    """
    caminho = Path(caminho)
    if caminho.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(caminho)
    else:
        df = pd.read_csv(caminho, sep=_separador(caminho), encoding="utf-8-sig", decimal=decimal)
    df = df.rename(columns=str.strip).rename(columns=mapeamento or {})

    faltam = [c for c in OBRIGATORIAS if c not in df.columns]
    if faltam:
        raise ValueError(
            f"Colunas obrigatórias ausentes: {faltam}. Colunas encontradas: {list(df.columns)}. "
            "Use o mapeamento {nome_no_arquivo: nome_padrao}."
        )
    df["data_lancamento"] = pd.to_datetime(df["data_lancamento"], dayfirst=dayfirst)
    df["valor"] = pd.to_numeric(df["valor"], errors="raise")
    df["conta_cosif"] = df["conta_cosif"].astype(str).str.strip()
    ausentes = [c for c in OPCIONAIS if c not in df.columns]
    df.attrs["colunas_ausentes"] = ausentes
    return df

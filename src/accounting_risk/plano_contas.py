"""Plano de contas COSIF (Banco Central do Brasil), versão com 9 dígitos + DV.

Estrutura do código: X.X.X.XX.XX.XX-D  (ex.: 1.1.0.00.00.00-2 = DISPONIBILIDADES)
  - 1º dígito = grupo (1 Ativo Realizável, 2 Ativo Permanente, 3 Compensação Ativa,
    4 Passivo Exigível, 6 Patrimônio Líquido, 7 Resultado Credor, 8 Resultado Devedor,
    9 Compensação Passiva)
  - 6 níveis: grupo | subgrupo | desdobramento | 2 dígitos | 2 dígitos | 2 dígitos
  - D = dígito verificador: módulo 10, pesos 3-7-1 repetidos da direita para a esquerda sobre
    os dígitos da base (-soma mod 10). Conferido nas 4.401 contas do arquivo e em 17 códigos
    oficiais da estrutura anterior (7 dígitos).

O arquivo `dados/plano_contas_cosif.csv` traz código, nome, início/fim de vigência (AAAAMM) e
natureza (Normal, Retificadora, Duplo posicionamento). Confirme fonte e versão oficiais no BCB.
Premissa: o fim de vigência (DT_FIM_VIGEN) é inclusivo, isto é, a conta vale até o fim desse mês.
"""
import re
from pathlib import Path

import pandas as pd

GRUPOS = {
    "1": "Ativo Realizável",
    "2": "Ativo Permanente",
    "3": "Compensação Ativa",
    "4": "Passivo Exigível",
    "6": "Patrimônio Líquido",
    "7": "Resultado Credor",
    "8": "Resultado Devedor",
    "9": "Compensação Passiva",
}

ARQUIVO_PADRAO = Path(__file__).parent / "dados" / "plano_contas_cosif.csv"
PESOS_DV = (3, 7, 1)  # da direita para a esquerda
_CORTES = [1, 2, 3, 5, 7, 9]  # fim de cada nível dentro da base de 9 dígitos
PADRAO = re.compile(r"^\d\.\d\.\d\.\d{2}\.\d{2}\.\d{2}-\d$")


def calcular_dv(base: str) -> int:
    """DV de uma base numérica (9 dígitos na estrutura atual; 7 na anterior)."""
    soma = sum(int(d) * PESOS_DV[i % 3] for i, d in enumerate(reversed(base)))
    return (-soma) % 10


def so_digitos(codigo: str) -> str:
    return re.sub(r"\D", "", str(codigo))


def formatar_codigo(codigo: str) -> str:
    """Aceita a base (9 dígitos) ou o código com DV (10 dígitos) e devolve X.X.X.XX.XX.XX-D."""
    d = so_digitos(codigo)
    base = d[:9]
    dv = d[9] if len(d) == 10 else str(calcular_dv(base))
    return f"{base[0]}.{base[1]}.{base[2]}.{base[3:5]}.{base[5:7]}.{base[7:9]}-{dv}"


def codigo_valido(codigo: str) -> bool:
    """True se o código tem a estrutura COSIF atual (9 dígitos + DV) e o DV confere."""
    d = so_digitos(codigo)
    return len(d) == 10 and int(d[9]) == calcular_dv(d[:9])


def _nivel(base: str) -> int:
    nivel, anterior = 0, 0
    for i, corte in enumerate(_CORTES, 1):
        if set(base[anterior:corte]) != {"0"}:
            nivel = i
        anterior = corte
    return nivel


def carregar_plano(caminho=None) -> pd.DataFrame:
    """Carrega o plano e deriva: código formatado, grupo, nível, folha e vigência."""
    bruto = pd.read_csv(caminho or ARQUIVO_PADRAO, sep=";", encoding="utf-8-sig", dtype=str)
    bruto = bruto.rename(columns=str.strip)
    base = bruto["COD_CONTA"].str[:9]
    nivel = base.map(_nivel)

    contagem = {}
    for b in base:
        for corte in _CORTES:
            contagem[b[:corte]] = contagem.get(b[:corte], 0) + 1
    folha = [contagem[b[: _CORTES[n - 1]]] == 1 for b, n in zip(base, nivel)]

    return pd.DataFrame(
        {
            "codigo": bruto["COD_CONTA"].map(formatar_codigo),
            "descricao": bruto["NOME_CONTA"],
            "grupo": bruto["COD_CONTA"].str[0].map(GRUPOS),
            "natureza": bruto["NATUREZA"],
            "nivel": nivel,
            "folha": folha,
            "vigencia_ini": pd.to_datetime(bruto["DT_INI_VIGEN"], format="%Y%m"),
            # fim inclusivo: vale até o último dia do mês informado
            "vigencia_fim": pd.to_datetime(bruto["DT_FIM_VIGEN"], format="%Y%m") + pd.offsets.MonthEnd(0),
        }
    )


PLANO = carregar_plano()
CODIGOS = PLANO["codigo"].tolist()
CODIGOS_SET = frozenset(CODIGOS)
DESCRICAO = dict(zip(PLANO["codigo"], PLANO["descricao"]))
VIGENCIA_INI = dict(zip(PLANO["codigo"], PLANO["vigencia_ini"]))
VIGENCIA_FIM = dict(zip(PLANO["codigo"], PLANO["vigencia_fim"]))

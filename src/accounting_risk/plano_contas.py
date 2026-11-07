"""Plano de contas no padrão COSIF (Banco Central do Brasil).

Estrutura do código: X.X.X.XX.XX-D
  - 1º dígito = grupo (1 Ativo Realizável, 2 Ativo Permanente, 3 Compensação Ativa,
    4 Passivo Exigível, 6 Patrimônio Líquido, 7 Resultado Credor, 8 Resultado Devedor,
    9 Compensação Passiva)
  - D = dígito verificador (módulo 10, pesos 3-1-7 repetidos sobre os 7 dígitos da base)

O algoritmo do DV foi inferido e conferido contra 17 códigos oficiais (ver tests/test_plano_contas.py).
As contas abaixo são um subconjunto ILUSTRATIVO: valide códigos e títulos contra o elenco
oficial vigente do Cosif antes de qualquer uso real. O BCB está reestruturando o elenco
(Res. BCB 320), então confirme qual versão a sua instituição utiliza.
"""
import re

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

PESOS_DV = (3, 1, 7)
PADRAO = re.compile(r"^(\d)\.(\d)\.(\d)\.(\d{2})\.(\d{2})-(\d)$")


def calcular_dv(base: str) -> int:
    """DV a partir dos 7 dígitos da base (ex.: '3000000' -> 1)."""
    soma = sum(int(d) * PESOS_DV[i % 3] for i, d in enumerate(base))
    return (-soma) % 10


def formatar_codigo(base: str) -> str:
    return f"{base[0]}.{base[1]}.{base[2]}.{base[3:5]}.{base[5:7]}-{calcular_dv(base)}"


def codigo_valido(codigo: str) -> bool:
    """True se o código tem o formato COSIF e o DV confere."""
    m = PADRAO.match(str(codigo))
    if not m:
        return False
    base = "".join(m.groups()[:5])
    return int(m.group(6)) == calcular_dv(base)


# (base de 7 dígitos, título resumido)
_CONTAS_BASE = [
    ("1110000", "Caixa"),
    ("1600000", "Operações de Crédito"),
    ("1800000", "Outros Créditos"),
    ("4110000", "Depósitos à Vista"),
    ("4600000", "Obrigações por Empréstimos e Repasses"),
    ("7110000", "Rendas de Operações de Crédito"),
    ("8110000", "Despesas de Captação"),
    ("3000000", "Compensação Ativa"),
    ("9000000", "Compensação Passiva"),
]

PLANO = pd.DataFrame(
    {
        "codigo": [formatar_codigo(b) for b, _ in _CONTAS_BASE],
        "descricao": [d for _, d in _CONTAS_BASE],
        "grupo": [GRUPOS[b[0]] for b, _ in _CONTAS_BASE],
    }
)
CODIGOS = PLANO["codigo"].tolist()
DESCRICAO = dict(zip(PLANO["codigo"], PLANO["descricao"]))

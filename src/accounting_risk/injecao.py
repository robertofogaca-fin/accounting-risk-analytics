"""Injeção de fracionamento abaixo da alçada (v0.6).

Cenário: o usuário precisa registrar um pagamento acima da própria alçada e, para não depender de
aprovação superior, quebra o valor em 2 a 5 lançamentos, cada um logo abaixo do limite, para a mesma
contraparte e a mesma conta, em poucos dias.

Cada parcela, olhada sozinha, é um lançamento normal: valor dentro da faixa usual, conta e centro de custo
habituais do usuário, mesmo histórico dos lançamentos rotineiros. A fraude só aparece no agrupamento.

Os parâmetros abaixo foram fixados ANTES de escrever qualquer detector, e não devem ser ajustados depois
para melhorar resultado. Ver docs/v0.6_especificacao_fracionamento.md.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

TIPO = "FRACIONAMENTO_ALCADA"


@dataclass(frozen=True)
class ParametrosFracionamento:
    eventos: int = 300                 # pagamentos fracionados
    parcelas_min: int = 2
    parcelas_max: int = 5
    faixa_parcela: tuple = (0.80, 0.99)  # parcela como fração da alçada do usuário
    janela_dias_max: int = 4           # da primeira à última parcela
    seed: int = 2026


def _documentos_livres(documentos: pd.Series, quantidade: int, rng: np.random.Generator) -> list:
    """Números de documento no formato da base que ainda não foram usados (lacunas da numeração)."""
    usados = set(documentos)
    numeros = documentos.str.extract(r"^DOC(\d+)$")[0].dropna().astype(int)
    largura = int(documentos.str.len().mode()[0]) - 3
    livres = [n for n in range(1, int(numeros.max()) + 1) if f"DOC{n:0{largura}d}" not in usados]
    if len(livres) < quantidade:
        livres += list(range(int(numeros.max()) + 1, int(numeros.max()) + 1 + quantidade - len(livres)))
    escolhidos = rng.choice(livres, size=quantidade, replace=False)
    return [f"DOC{n:0{largura}d}" for n in escolhidos]


def injetar_fracionamento(
    df: pd.DataFrame,
    alcadas: pd.DataFrame,
    params: ParametrosFracionamento = ParametrosFracionamento(),
    coluna_usuario: str = "usuario",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Acrescenta os lançamentos fracionados à base.

    Usa como molde lançamentos NORMAIS reais do próprio usuário (data, conta, centro de custo, contraparte,
    histórico), para que nenhuma coluna diferencie a parcela de um lançamento comum.

    Retorna (base_com_fraudes, gabarito). O gabarito liga cada parcela ao seu evento e fica FORA da base.
    """
    rng = np.random.default_rng(params.seed)
    data_max = pd.to_datetime(df["data_lancamento"]).max()
    limite = alcadas.set_index("usuario")["limite_alcada"]
    normais = df[df["tipo_anomalia"] == "NORMAL"]
    normais = normais[normais[coluna_usuario].isin(limite.index)]

    moldes = normais.sample(n=params.eventos, random_state=params.seed)
    n_parcelas = rng.integers(params.parcelas_min, params.parcelas_max + 1, size=params.eventos)
    documentos = _documentos_livres(df["documento"].astype(str), int(n_parcelas.sum()), rng)

    linhas, gabarito, d = [], [], 0
    for evento, ((_, molde), n) in enumerate(zip(moldes.iterrows(), n_parcelas), start=1):
        alcada = float(limite[molde[coluna_usuario]])
        fracoes = rng.uniform(*params.faixa_parcela, size=n)
        valores = np.round(alcada * fracoes, 2)
        deslocamentos = np.sort(rng.integers(0, params.janela_dias_max + 1, size=n))
        deslocamentos -= deslocamentos[0]
        inicio = min(pd.Timestamp(molde["data_lancamento"]), data_max - pd.Timedelta(days=int(deslocamentos[-1])))
        for valor, dias in zip(valores, deslocamentos):
            nova = molde.copy()
            nova["data_lancamento"] = inicio + pd.Timedelta(days=int(dias))
            nova["valor"] = float(valor)
            nova["documento"] = documentos[d]
            nova["flag_anomalia"] = 1
            nova["tipo_anomalia"] = TIPO
            linhas.append(nova)
            gabarito.append(
                {
                    "documento": documentos[d],
                    "evento": evento,
                    "parcelas_no_evento": int(n),
                    "valor_total_evento": round(float(valores.sum()), 2),
                    "limite_alcada": alcada,
                }
            )
            d += 1

    injetadas = pd.DataFrame(linhas)
    # Parcelas em posições aleatórias, sem mudar a ordem das linhas originais (detectores que dependem da
    # ordem, como o de documento repetido, continuam vendo a base como antes). Depois, renumera o id:
    # nem a posição nem o id podem denunciar a linha injetada.
    ordem = np.concatenate([np.arange(len(df), dtype=float), rng.uniform(0, len(df), size=len(injetadas))])
    base = pd.concat([df, injetadas], ignore_index=True)
    base = base.iloc[np.argsort(ordem, kind="stable")].reset_index(drop=True)
    if "id_lancamento" in base.columns:
        base["id_lancamento"] = np.arange(1, len(base) + 1)
    gab = pd.DataFrame(gabarito).merge(base[["documento", "id_lancamento"]], on="documento", how="left")
    return base, gab

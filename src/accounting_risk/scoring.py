"""Aplica os detectores, calcula o score (0 a 100) e avalia contra os rótulos sintéticos."""
import numpy as np
import pandas as pd

from .rules import REGRAS, Detector
from .stats import ESTATISTICOS

TODOS = REGRAS + ESTATISTICOS
PONTOS_PARA_100 = 10  # soma de pesos que leva o score a 100


def aplicar_detectores(df: pd.DataFrame, detectores: list[Detector] = TODOS, desligar=(), plano_proprio: bool = False) -> pd.DataFrame:
    """Colunas novas: sinal_<nome>, score, nivel_risco, regras_disparadas.

    Detectores cujas colunas necessárias não existem, ou listados em `desligar`, ficam de fora.
    `plano_proprio=True` indica que a base usa contas que não estão no elenco COSIF do projeto: desliga
    `conta_invalida` e `conta_fora_vigencia` e deixa os detectores estatísticos rodarem em todas as contas.
    A lista dos que rodaram está em `resultado.attrs["detectores_ativos"]` e dos que ficaram de fora
    (com o motivo) em `resultado.attrs["detectores_desligados"]`.
    """
    if plano_proprio:
        # contas fora do elenco COSIF do projeto: as regras de plano/vigência marcariam tudo
        desligar = set(desligar) | {"conta_invalida", "conta_fora_vigencia"}
    ativos, desligados = [], {}
    for d in detectores:
        faltam = [c for c in d.requer if c not in df.columns]
        if d.nome in set(desligar):
            desligados[d.nome] = "desligado pelo usuário"
        elif faltam:
            desligados[d.nome] = "faltam colunas: " + ", ".join(faltam)
        else:
            ativos.append(d)
    detectores = ativos
    out = df.copy()
    out.attrs["plano_proprio"] = plano_proprio
    out.attrs["detectores_ativos"] = [d.nome for d in ativos]
    out.attrs["detectores_desligados"] = desligados
    for d in detectores:
        out[f"sinal_{d.nome}"] = d.funcao(out).astype(bool)

    flags = out[[f"sinal_{d.nome}" for d in detectores]].to_numpy()
    pesos = np.array([d.peso for d in detectores])
    nomes = np.array([d.nome for d in detectores])
    out["score"] = np.minimum(100.0, (flags @ pesos) * 100 / PONTOS_PARA_100).round(1)
    out["nivel_risco"] = pd.cut(out["score"], bins=[-1, 0, 30, 60, 101], labels=["nenhum", "baixo", "medio", "alto"])
    out["regras_disparadas"] = [", ".join(nomes[linha]) for linha in flags]
    return out


def aplicar_regras(df: pd.DataFrame, regras: list[Detector] = REGRAS) -> pd.DataFrame:
    """Só as regras determinísticas (v0.1)."""
    return aplicar_detectores(df, regras)


def avaliar(df_pontuado: pd.DataFrame, limiar: float = 1.0) -> dict:
    """Precision/recall globais (qualquer sinal com score >= limiar) vs flag_anomalia."""
    previsto = df_pontuado["score"] >= limiar
    real = df_pontuado["flag_anomalia"] == 1
    tp = int((previsto & real).sum())
    fp = int((previsto & ~real).sum())
    fn = int((~previsto & real).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": round(precision, 3), "recall": round(recall, 3)}


def resumo_por_tipo(df_pontuado: pd.DataFrame, limiar: float = 1.0) -> pd.DataFrame:
    """Recall por tipo de anomalia injetada (qualquer detector)."""
    anom = df_pontuado[df_pontuado["flag_anomalia"] == 1]
    detectado = anom["score"] >= limiar
    return (
        pd.DataFrame({"tipo": anom["tipo_anomalia"], "detectado": detectado})
        .groupby("tipo")["detectado"].agg(n="size", detectados="sum")
        .assign(recall=lambda t: (t["detectados"] / t["n"]).round(3))
        .sort_values("recall")
    )


def resumo_por_detector(df_pontuado: pd.DataFrame, detectores: list[Detector] = TODOS, alvos: dict | None = None) -> pd.DataFrame:
    """Por detector: sinalizados, precision (qualquer anomalia) e recall no tipo-alvo.
    `alvos` (opcional) substitui o tipo-alvo de cada detector, útil para bases externas com outros rótulos."""
    linhas = []
    for d in detectores:
        if f"sinal_{d.nome}" not in df_pontuado.columns:
            continue
        tipo_alvo = (alvos or {}).get(d.nome, d.tipo_alvo)
        sinal = df_pontuado[f"sinal_{d.nome}"]
        real = df_pontuado["flag_anomalia"] == 1
        alvo = df_pontuado["tipo_anomalia"] == tipo_alvo
        linhas.append(
            {
                "detector": d.nome,
                "tipo_alvo": tipo_alvo,
                "sinalizados": int(sinal.sum()),
                "precision": round(float((sinal & real).sum() / max(sinal.sum(), 1)), 3),
                "recall_alvo": round(float((sinal & alvo).sum() / max(alvo.sum(), 1)), 3),
            }
        )
    return pd.DataFrame(linhas)


def curva_por_limiar(df_pontuado: pd.DataFrame, limiares=(1, 30.1, 60.1)) -> pd.DataFrame:
    """Trade-off precision x recall conforme o rigor do limiar (1 = qualquer sinal, 30.1 = médio+, 60.1 = alto)."""
    linhas = []
    for lim in limiares:
        m = avaliar(df_pontuado, lim)
        linhas.append({"limiar_score": lim, "alertas": m["tp"] + m["fp"], **m})
    return pd.DataFrame(linhas)

"""Compara métodos pelo ranking de risco que cada um produz (dados sintéticos com rótulo)."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def _top_k(score: pd.Series, k: int, seed: int = 0) -> np.ndarray:
    ruido = np.random.default_rng(seed).random(len(score)) * 1e-6  # desempate reprodutível
    mascara = np.zeros(len(score), dtype=bool)
    mascara[np.argsort(-(score.to_numpy() + ruido))[:k]] = True
    return mascara


def comparar_metodos(df: pd.DataFrame, scores: dict, k: int | None = None) -> pd.DataFrame:
    """Precision@k, recall@k, AUC e precisão média (AP). Por padrão, k = nº de anomalias reais."""
    real = df["flag_anomalia"].to_numpy() == 1
    k = k or int(real.sum())
    linhas = []
    for nome, s in scores.items():
        top = _top_k(s, k)
        tp = int((top & real).sum())
        linhas.append(
            {
                "metodo": nome,
                f"precision@{k}": round(tp / k, 3),
                f"recall@{k}": round(tp / real.sum(), 3),
                "auc": round(roc_auc_score(real, s), 3),
                "ap": round(average_precision_score(real, s), 3),
            }
        )
    return pd.DataFrame(linhas)


def recall_por_tipo(df: pd.DataFrame, scores: dict, k: int | None = None) -> pd.DataFrame:
    """Recall@k por tipo de anomalia (linhas) e método (colunas)."""
    real = df["flag_anomalia"].to_numpy() == 1
    k = k or int(real.sum())
    colunas = {}
    for nome, s in scores.items():
        top = _top_k(s, k)
        colunas[nome] = pd.Series(top[real]).groupby(df["tipo_anomalia"].to_numpy()[real]).mean().round(2)
    return pd.DataFrame(colunas)


def fila_exclusiva_ml(df: pd.DataFrame, score_regras: pd.Series, score_ml: pd.Series, tamanhos=(100, 200, 300)) -> pd.DataFrame:
    """Segunda fila de revisão: lançamentos SEM sinal de regra/estatística, ordenados pelo score do ML.

    Mede quanto a fila exclusiva do ML acerta (precision) contra a taxa-base de anomalias nesse grupo.
    """
    sem_sinal = score_regras == 0
    taxa_base = float((df.loc[sem_sinal, "flag_anomalia"] == 1).mean())
    ordenados = score_ml[sem_sinal].sort_values(ascending=False)
    linhas = []
    for n in tamanhos:
        topo = df.loc[ordenados.head(n).index]
        acertos = topo[topo["flag_anomalia"] == 1]
        precision = len(acertos) / n
        linhas.append(
            {
                "alertas": n,
                "precision": round(precision, 3),
                "taxa_base": round(taxa_base, 4),
                "lift": round(precision / taxa_base, 1) if taxa_base else float("nan"),
                "tipos_encontrados": ", ".join(f"{t} ({c})" for t, c in acertos["tipo_anomalia"].value_counts().items()),
            }
        )
    return pd.DataFrame(linhas)

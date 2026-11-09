"""Detectores de machine learning não supervisionados (v0.4): Isolation Forest e LOF.

Os modelos NÃO usam os rótulos. O que eles enxergam é decidido pelos atributos (features):
  - brutas: o que está no razão (valor em log, dia da semana, dia do mês, origem, tipo, tamanho do histórico)
  - engenheiradas: brutas + atributos de contexto que um analista escolheria olhar
    (z-score do valor na conta, frequência da conta, raridade do par usuário-conta e conta-centro,
     volume mensal da conta, duplicidade de documento, dia não útil, mesmo usuário criando e aprovando...)

Isso é de propósito: sem engenharia de atributos, o ML sozinho pouco acrescenta.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler

from .calendario import dia_nao_util
from .stats import zscore_robusto

FEATURES_BRUTAS = ["log_valor", "dia_semana", "dia_mes", "origem_manual", "tipo_d", "historico_len"]


def _freq_log(serie: pd.Series) -> pd.Series:
    return np.log(serie.map(serie.value_counts(normalize=True)))


def construir_features(df: pd.DataFrame, engenheiradas: bool = True, extras: bool = False) -> pd.DataFrame:
    """Monta os atributos. Colunas ausentes na base simplesmente não geram o atributo correspondente.

    extras=True acrescenta frequências de categorias (usuário, centro de custo, histórico), úteis em bases
    externas; fica desligado por padrão para o cenário sintético manter os resultados já publicados.
    """
    tem = set(df.columns)
    f = pd.DataFrame(index=df.index)
    f["log_valor"] = np.log(df["valor"].clip(lower=0.01))
    f["dia_semana"] = df["data_lancamento"].dt.day_of_week
    f["dia_mes"] = df["data_lancamento"].dt.day
    if "origem" in tem:
        f["origem_manual"] = (df["origem"] == "manual").astype(int)
    if "tipo" in tem:
        f["tipo_d"] = (df["tipo"] == "D").astype(int)
    if "historico" in tem:
        f["historico_len"] = df["historico"].fillna("").str.len()
    if not engenheiradas:
        return f

    f["z_log_conta"] = zscore_robusto(f["log_valor"], df["conta_cosif"])
    f["nao_util"] = dia_nao_util(df["data_lancamento"]).astype(int)
    f["conta_freq_log"] = np.log(df["conta_cosif"].map(df["conta_cosif"].value_counts()))

    if {"usuario_criacao", "usuario_aprovacao"} <= tem:
        f["criador_eq_aprov"] = (df["usuario_criacao"] == df["usuario_aprovacao"]).astype(int)
    if "usuario_criacao" in tem:
        por_usuario = df.groupby("usuario_criacao")["valor"].transform("size")
        par_usuario_conta = df.groupby(["usuario_criacao", "conta_cosif"])["valor"].transform("size")
        f["par_usuario_conta_log"] = np.log(par_usuario_conta / por_usuario)
    if "centro_custo" in tem:
        por_conta = df.groupby("conta_cosif")["valor"].transform("size")
        par_conta_centro = df.groupby(["conta_cosif", "centro_custo"])["valor"].transform("size")
        f["par_conta_centro_log"] = np.log(par_conta_centro / por_conta)
    if "documento" in tem:
        f["documento_repetido"] = df["documento"].duplicated(keep=False).astype(int)
    chave = [c for c in ["data_lancamento", "conta_cosif", "valor", "tipo", "centro_custo"] if c in tem]
    f["chave_repetida"] = df.duplicated(chave, keep=False).astype(int)
    f["valor_redondo"] = ((df["valor"] >= 1_000) & (df["valor"] % 1_000 == 0)).astype(int)

    mes = df["data_lancamento"].dt.to_period("M")
    volume = df.groupby([df["conta_cosif"], mes])["valor"].transform("size")
    f["volume_mes_z"] = zscore_robusto(np.log(volume), df["conta_cosif"])

    if extras:
        for coluna in ("usuario_criacao", "centro_custo", "historico"):
            if coluna in tem:
                f[f"{coluna}_freq_log"] = _freq_log(df[coluna])
    return f


def pontuar_isolation_forest(
    df: pd.DataFrame, engenheiradas: bool = True, n_estimators: int = 300, seed: int = 0, extras: bool = False
) -> pd.Series:
    """Score de anomalia (quanto maior, mais anômalo) pelo Isolation Forest."""
    X = construir_features(df, engenheiradas, extras)
    modelo = IsolationForest(n_estimators=n_estimators, random_state=seed, n_jobs=-1).fit(X)
    return pd.Series(-modelo.score_samples(X), index=df.index, name="score_if")


def pontuar_lof(df: pd.DataFrame, engenheiradas: bool = True, n_neighbors: int = 35, extras: bool = False) -> pd.Series:
    """Score de anomalia pelo Local Outlier Factor (atributos padronizados de forma robusta)."""
    X = RobustScaler().fit_transform(construir_features(df, engenheiradas, extras))
    modelo = LocalOutlierFactor(n_neighbors=n_neighbors, n_jobs=-1).fit(X)
    return pd.Series(-modelo.negative_outlier_factor_, index=df.index, name="score_lof")


def combinar(score_regras: pd.Series, score_ml: pd.Series, peso_ml: float = 0.5) -> pd.Series:
    """Combina o score de regras+estatística (0 a 100) com o ranking percentual do ML (0 a 1)."""
    return score_regras / 100 + peso_ml * score_ml.rank(pct=True)

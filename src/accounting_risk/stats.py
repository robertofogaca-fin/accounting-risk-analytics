"""Detectores estatísticos (v0.2).

1. Z-score robusto (mediana/MAD) em escala log, por conta.
2. Par usuário-conta raro (frequência relativa).
3. Desvio no perfil de primeiros dígitos por usuário (variante da Lei de Benford).
4. Pico mensal por conta (z-score robusto sobre a quantidade mensal de lançamentos).

Nota sobre Benford: a lei clássica só vale para valores que atravessam várias ordens de
grandeza. Aqui cada conta tem valor típico próprio (faixa estreita), então o detector compara
o usuário com o perfil de dígitos *das suas próprias contas* (baseline empírico).
`benford_mad` fica disponível para dados com faixa ampla.
"""
import numpy as np
import pandas as pd

from .plano_contas import CODIGOS
from .rules import Detector

BENFORD = np.log10(1 + 1 / np.arange(1, 10))


# ---------- utilidades ----------
def zscore_robusto(serie: pd.Series, grupos) -> pd.Series:
    """z = 0,6745 * (x - mediana) / MAD, calculado dentro de cada grupo."""
    mediana = serie.groupby(grupos).transform("median")
    mad = (serie - mediana).abs().groupby(grupos).transform("median")
    z = 0.6745 * (serie - mediana) / mad.replace(0, np.nan)
    return z.fillna(0.0)


def primeiro_digito(valores) -> np.ndarray:
    v = np.abs(np.asarray(valores, dtype=float))
    v = np.where(v == 0, np.nan, v)
    d = np.floor(v / 10 ** np.floor(np.log10(v)))
    return np.nan_to_num(d, nan=1).clip(1, 9).astype(int)


def benford_mad(valores) -> float:
    """Desvio absoluto médio (MAD) entre a distribuição observada e a de Benford.
    Nigrini: < 0,006 conformidade próxima; > 0,015 não conforme."""
    d = primeiro_digito(valores)
    obs = np.bincount(d, minlength=10)[1:10] / len(d)
    return float(np.abs(obs - BENFORD).mean())


# ---------- detectores ----------
def _so_plano(funcao):
    """Executa o detector apenas nas linhas de contas válidas do plano.
    Contas fora do plano são tratadas pela regra `conta_invalida` e distorceriam as estatísticas."""
    def envolvida(df, **kwargs):
        mascara = df["conta_cosif"].isin(CODIGOS)
        resultado = funcao(df[mascara], **kwargs)
        return resultado.reindex(df.index, fill_value=False).astype(bool)
    envolvida.__name__ = funcao.__name__
    envolvida.__doc__ = funcao.__doc__
    return envolvida


def valor_atipico_conta(df: pd.DataFrame, limiar: float = 4.0) -> pd.Series:
    z = zscore_robusto(np.log(df["valor"].clip(lower=0.01)), df["conta_cosif"])
    return z > limiar


def par_usuario_conta_raro(df: pd.DataFrame, share_max: float = 0.02, min_lanc: int = 100) -> pd.Series:
    n_par = df.groupby(["usuario_criacao", "conta_cosif"])["valor"].transform("size")
    n_user = df.groupby("usuario_criacao")["valor"].transform("size")
    return (n_user >= min_lanc) & (n_par / n_user < share_max)


def desvio_digitos_por_usuario(df: pd.DataFrame):
    """Retorna (desvio usuário x dígito, MAD por usuário).

    Desvio = share observado - share esperado. O esperado usa o perfil de dígitos de cada
    conta calculado SEM o próprio usuário (leave-one-out), para que as anomalias dele não
    contaminem a referência."""
    d = primeiro_digito(df["valor"])
    n = len(df)
    um_hot = np.zeros((n, 9))
    um_hot[np.arange(n), d - 1] = 1.0
    um_hot = pd.DataFrame(um_hot)
    conta, usuario = df["conta_cosif"].to_numpy(), df["usuario_criacao"].to_numpy()

    por_conta = um_hot.groupby(conta).sum()
    por_uc = um_hot.groupby([usuario, conta]).sum()
    n_uc = por_uc.sum(axis=1).to_numpy()
    resto = por_conta.reindex(por_uc.index.get_level_values(1)).to_numpy() - por_uc.to_numpy()
    resto_n = resto.sum(axis=1, keepdims=True)
    perfil = resto / np.where(resto_n > 0, resto_n, 1)
    esperado_uc = pd.DataFrame(perfil * n_uc[:, None], index=por_uc.index)

    obs_u = por_uc.groupby(level=0).sum()
    esp_u = esperado_uc.groupby(level=0).sum()
    n_u = obs_u.sum(axis=1)
    desvio = obs_u.div(n_u, axis=0) - esp_u.div(n_u, axis=0)
    desvio.columns = range(1, 10)
    return desvio, desvio.abs().mean(axis=1)


def desvio_digitos_usuario(df: pd.DataFrame, limiar_mad: float = 0.010, excesso_min: float = 0.035,
                           min_lanc: int = 300) -> pd.Series:
    desvio, mad = desvio_digitos_por_usuario(df)
    n_user = df.groupby("usuario_criacao")["valor"].transform("size")
    d = primeiro_digito(df["valor"])
    usuarios = df["usuario_criacao"]
    excesso_linha = desvio.reindex(usuarios).to_numpy()[np.arange(len(df)), d - 1]
    mad_linha = mad.reindex(usuarios).to_numpy()
    return pd.Series((mad_linha > limiar_mad) & (excesso_linha > excesso_min) & (n_user.to_numpy() >= min_lanc),
                     index=df.index)


def pico_mensal(df: pd.DataFrame, limiar: float = 3.5, min_meses: int = 6) -> pd.Series:
    """Sinaliza todos os lançamentos de uma conta/mês cuja QUANTIDADE de lançamentos destoa do histórico.
    Usa contagem (e não soma) para não ser distorcida por poucos valores extremos."""
    mes = df["data_lancamento"].dt.to_period("M")
    total = df.groupby([df["conta_cosif"], mes])["valor"].size()
    z = zscore_robusto(total, total.index.get_level_values(0))
    meses = total.groupby(level=0).transform("size")
    marcados = total.index[(z > limiar) & (meses >= min_meses)]
    chave = pd.MultiIndex.from_arrays([df["conta_cosif"], mes])
    return pd.Series(chave.isin(marcados), index=df.index)


valor_atipico_conta = _so_plano(valor_atipico_conta)
par_usuario_conta_raro = _so_plano(par_usuario_conta_raro)
desvio_digitos_usuario = _so_plano(desvio_digitos_usuario)
pico_mensal = _so_plano(pico_mensal)

ESTATISTICOS = [
    Detector("valor_atipico_conta", "Valor muito acima do padrão da conta (z-score robusto, escala log)", 5,
             valor_atipico_conta, "valor_atipico_conta"),
    Detector("par_usuario_conta_raro", "Usuário lança em conta que raramente movimenta", 3,
             par_usuario_conta_raro, "par_usuario_conta_raro"),
    Detector("desvio_digitos_usuario", "Excesso de um primeiro dígito no perfil do usuário (tipo Benford)", 2,
             desvio_digitos_usuario, "fracionamento_alcada"),
    Detector("pico_mensal", "Volume mensal de lançamentos da conta fora do histórico (triagem por conta/mês)", 1,
             pico_mensal, "pico_mensal"),
]

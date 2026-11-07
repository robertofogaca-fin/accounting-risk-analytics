"""Gerador de razão contábil FICTÍCIO (plano de contas COSIF) com anomalias rotuladas.

Dados normais: dias úteis, valores com centavos, valor típico por conta (lognormal),
aprovador diferente do criador, cada usuário concentrado em 3 contas habituais.

Anomalias CONHECIDAS (pegas por regras simples):
  fim_de_semana, valor_redondo, duplicidade, manual_alto, sem_segregacao,
  historico_vazio, conta_invalida
Anomalias SUTIS (regras fixas não pegam; a estatística precisa pegar):
  valor_atipico_conta, par_usuario_conta_raro, fracionamento_alcada, pico_mensal
"""
import numpy as np
import pandas as pd

from .plano_contas import CODIGOS, DESCRICAO, calcular_dv

# Mediana típica (R$) por conta, na ordem de CODIGOS, e dispersão (log) comum
_MEDIANAS = [900, 1500, 600, 400, 1200, 300, 250, 800, 800]
MEDIANA = dict(zip(CODIGOS, _MEDIANAS))
SIGMA = 0.9

CENTROS_CUSTO = ["CC100", "CC200", "CC300", "CC400"]
USUARIOS = [f"usuario{i:02d}" for i in range(1, 13)]
LIMITE_MANUAL_ALTO = 100_000.0
VALOR_MAX_NORMAL = 80_000.0
ALCADA = 10_000.0
INICIO, FIM = "2025-01-01", "2026-09-30"

TIPOS_CONHECIDOS = [
    "fim_de_semana", "valor_redondo", "duplicidade", "manual_alto",
    "sem_segregacao", "historico_vazio", "conta_invalida",
]
TIPOS_SUTIS = ["valor_atipico_conta", "par_usuario_conta_raro", "fracionamento_alcada", "pico_mensal"]


def _perfis_usuarios(rng):
    return {u: list(rng.choice(CODIGOS, size=3, replace=False)) for u in USUARIOS}


def _aprovadores(rng, criadores):
    return np.array([rng.choice([u for u in USUARIOS if u != c]) for c in criadores])


def _lancamentos_normais(n, rng, habituais):
    datas = pd.bdate_range(INICIO, FIM)
    usuarios = rng.choice(USUARIOS, size=n)
    contas = np.empty(n, dtype=object)
    for u in USUARIOS:
        m = usuarios == u
        k = int(m.sum())
        fora = rng.random(k) < 0.003  # raros lançamentos fora do perfil (ruído realista)
        contas[m] = np.where(fora, rng.choice(CODIGOS, size=k), rng.choice(habituais[u], size=k))
    medianas = np.array([MEDIANA[c] for c in contas])
    valores = np.round(np.clip(medianas * np.exp(SIGMA * rng.standard_normal(n)), 20, VALOR_MAX_NORMAL), 2)
    valores = np.where(valores % 1 == 0, valores + 0.37, valores)  # sem valores "redondos" acidentais
    return pd.DataFrame(
        {
            "data_lancamento": rng.choice(datas, size=n),
            "conta_cosif": contas,
            "descricao_conta": [DESCRICAO[c] for c in contas],
            "centro_custo": rng.choice(CENTROS_CUSTO, size=n),
            "valor": valores,
            "tipo": rng.choice(["D", "C"], size=n),
            "usuario_criacao": usuarios,
            "usuario_aprovacao": _aprovadores(rng, usuarios),
            "origem": rng.choice(["automatico", "manual"], size=n, p=[0.8, 0.2]),
            "historico": "Lançamento contábil rotineiro",
            "flag_anomalia": 0,
            "tipo_anomalia": "NORMAL",
        }
    )


def _novas_linhas(rng, n, contas, usuarios, datas, valores, tipo):
    usuarios = np.asarray(usuarios)
    return pd.DataFrame(
        {
            "data_lancamento": datas,
            "conta_cosif": list(contas),
            "descricao_conta": [DESCRICAO[c] for c in contas],
            "centro_custo": rng.choice(CENTROS_CUSTO, size=n),
            "valor": np.round(valores, 2),
            "tipo": rng.choice(["D", "C"], size=n),
            "usuario_criacao": usuarios,
            "usuario_aprovacao": _aprovadores(rng, usuarios),
            "origem": rng.choice(["automatico", "manual"], size=n, p=[0.8, 0.2]),
            "historico": "Lançamento contábil rotineiro",
            "flag_anomalia": 1,
            "tipo_anomalia": tipo,
        }
    )


def _marcar(df, idx, tipo):
    df.loc[idx, "flag_anomalia"] = 1
    df.loc[idx, "tipo_anomalia"] = tipo


def _injetar_anomalias(df, rng, k, habituais):
    df = df.copy()
    df["documento"] = [f"DOC{i:07d}" for i in range(len(df))]
    livres = set(df.index)

    def pegar(n, elegiveis=None):
        pool = sorted(livres if elegiveis is None else livres & set(elegiveis))
        escolhidos = rng.choice(pool, size=n, replace=False)
        livres.difference_update(int(i) for i in escolhidos)
        return escolhidos

    # ---------- anomalias conhecidas ----------
    idx = pegar(k)  # fim de semana
    sabados = pd.date_range("2025-01-04", "2026-09-26", freq="W-SAT")
    df.loc[idx, "data_lancamento"] = rng.choice(sabados, size=k)
    _marcar(df, idx, "fim_de_semana")

    idx = pegar(k)  # valor redondo
    df.loc[idx, "valor"] = rng.choice([2000.0, 3000.0, 4000.0, 5000.0], size=k)
    _marcar(df, idx, "valor_redondo")

    idx = pegar(k)  # manual de valor alto
    df.loc[idx, "valor"] = np.round(rng.uniform(150_000, 900_000, size=k), 2)
    df.loc[idx, "origem"] = "manual"
    _marcar(df, idx, "manual_alto")

    idx = pegar(k)  # mesmo usuário cria e aprova
    df.loc[idx, "usuario_aprovacao"] = df.loc[idx, "usuario_criacao"]
    _marcar(df, idx, "sem_segregacao")

    idx = pegar(k)  # histórico vazio
    df.loc[idx, "historico"] = ""
    _marcar(df, idx, "historico_vazio")

    idx = pegar(k)  # conta fora do plano (DV errado ou código inexistente)
    invalidas = []
    for i in range(k):
        base = df.loc[idx[i], "conta_cosif"]
        if i % 2 == 0:  # mesmo código com dígito verificador trocado
            digitos = base[0] + base[2] + base[4] + base[6:8] + base[9:11]
            dv_errado = (calcular_dv(digitos) + 1 + i % 8) % 10
            invalidas.append(base[:-1] + str(dv_errado))
        else:  # código estruturalmente válido, mas fora do plano da instituição
            while True:
                novo = "1" + str(rng.integers(1, 9)) + "0" + "0000"
                codigo = f"{novo[0]}.{novo[1]}.{novo[2]}.{novo[3:5]}.{novo[5:7]}-{calcular_dv(novo)}"
                if codigo not in CODIGOS:
                    break
            invalidas.append(codigo)
    df.loc[idx, "conta_cosif"] = invalidas
    _marcar(df, idx, "conta_invalida")

    # ---------- anomalias sutis (alteram linhas existentes) ----------
    baixas = df.index[df["conta_cosif"].map(MEDIANA).le(600).fillna(False)]
    idx = pegar(k, baixas)  # valor muito acima do padrão da conta, mas abaixo do limite manual
    med = df.loc[idx, "conta_cosif"].map(MEDIANA).to_numpy(dtype=float)
    kmax = np.minimum(6.0, np.log(95_000 / med) / SIGMA)
    z = rng.uniform(4.6, kmax)
    df.loc[idx, "valor"] = np.round(med * np.exp(SIGMA * z), 2)
    _marcar(df, idx, "valor_atipico_conta")

    # ---------- anomalias sutis (linhas novas) ----------
    novas = []
    datas_bd = pd.bdate_range(INICIO, FIM)

    # par usuário-conta raro: usuário lança em conta fora do seu perfil
    usu = rng.choice(USUARIOS, size=k)
    contas = [rng.choice([c for c in CODIGOS if c not in habituais[u]]) for u in usu]
    val = [MEDIANA[c] * np.exp(SIGMA * rng.standard_normal()) for c in contas]
    novas.append(_novas_linhas(rng, k, contas, usu, rng.choice(datas_bd, size=k), np.clip(val, 20, 40_000), "par_usuario_conta_raro"))

    # fracionamento abaixo da alçada: muitos lançamentos de 9.5xx a 9.9xx de um mesmo usuário
    grandes = [c for c in CODIGOS if MEDIANA[c] >= 800]
    candidatos = [u for u in USUARIOS if any(c in grandes for c in habituais[u])]
    u = rng.choice(candidatos)
    conta = rng.choice([c for c in habituais[u] if c in grandes])
    novas.append(
        _novas_linhas(rng, k, [conta] * k, [u] * k, rng.choice(datas_bd, size=k),
                      rng.uniform(0.95 * ALCADA, ALCADA - 0.01, size=k), "fracionamento_alcada")
    )

    # pico mensal: rajada de lançamentos moderados em uma conta de baixo volume/mês
    conta = rng.choice([c for c in CODIGOS if MEDIANA[c] <= 400])
    mes = rng.choice(pd.period_range("2025-04", "2026-06", freq="M"))
    dias = pd.bdate_range(mes.start_time, mes.end_time)
    donos = [x for x in USUARIOS if conta in habituais[x]] or USUARIOS
    novas.append(
        _novas_linhas(rng, k, [conta] * k, rng.choice(donos, size=k), rng.choice(dias, size=k),
                      MEDIANA[conta] * rng.uniform(3.0, 6.0, size=k), "pico_mensal")
    )

    novas = pd.concat(novas, ignore_index=True)
    novas["documento"] = [f"DOC{i:07d}" for i in range(len(df), len(df) + len(novas))]
    df = pd.concat([df, novas], ignore_index=True)

    # duplicidade: cópia integral (mesmo documento) de lançamentos normais
    idx = pegar(k, [i for i in livres if i < len(df) - len(novas)])
    copias = df.loc[idx].copy()
    copias["flag_anomalia"] = 1
    copias["tipo_anomalia"] = "duplicidade"
    return pd.concat([df, copias], ignore_index=True)


def gerar_razao(n: int = 20_000, n_por_tipo: int = 100, seed: int = 42) -> pd.DataFrame:
    """Gera `n` lançamentos normais + `n_por_tipo` anomalias de cada um dos 11 tipos."""
    rng = np.random.default_rng(seed)
    habituais = _perfis_usuarios(rng)
    df = _lancamentos_normais(n, rng, habituais)
    df = _injetar_anomalias(df, rng, n_por_tipo, habituais)
    df["data_lancamento"] = pd.to_datetime(df["data_lancamento"])
    df = df.sort_values("data_lancamento", kind="stable").reset_index(drop=True)
    df.insert(0, "id_lancamento", [f"LC{i + 1:06d}" for i in range(len(df))])
    return df[
        ["id_lancamento", "data_lancamento", "conta_cosif", "descricao_conta", "centro_custo", "valor",
         "tipo", "usuario_criacao", "usuario_aprovacao", "origem", "historico", "documento",
         "flag_anomalia", "tipo_anomalia"]
    ]

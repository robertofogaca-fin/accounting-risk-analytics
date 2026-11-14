"""Avalia o teste v0.6 (fracionamento abaixo da alçada): linha de base da v0.5 x detector de fracionamento.

Pré-requisito: python gerar_base_v06.py data/razao_cosif_500k_7pct_anomalias.csv

    python avaliar_v06.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from accounting_risk.carregador import carregar_razao
from accounting_risk.fracionamento import grupos_suspeitos, score_fracionamento
from accounting_risk.genericos import GENERICOS
from accounting_risk.injecao import TIPO
from accounting_risk.ml import pontuar_isolation_forest
from accounting_risk.relatorio import gerar_relatorio_excel
from accounting_risk.scoring import TODOS, aplicar_detectores

DADOS = Path("data")
SAIDA = Path("output/validacao_v06")
TAMANHOS = (100, 300, 1000)
CURVA_GRUPOS = (50, 100, 200, 300, 500, 1000, 2000, 3000, 5000)
TAMANHO_FILA = 1000
ALTA, MEDIA = 100, 300


def metricas(nome: str, score: pd.Series, df: pd.DataFrame, evento: pd.Series) -> dict:
    fraude = (df["tipo_anomalia"] == TIPO).to_numpy()
    s = score.to_numpy() + np.random.default_rng(0).random(len(score)) * 1e-9  # desempate reprodutível
    ordem = np.argsort(-s)
    sinal = score.to_numpy() > 0
    linha = {
        "metodo": nome,
        "alertas_com_sinal": int(sinal.sum()),
        "parcelas_com_sinal": int((sinal & fraude).sum()),
        "eventos_com_sinal": int(evento[sinal & fraude].nunique()),
    }
    for k in TAMANHOS:
        top = np.zeros(len(s), dtype=bool)
        top[ordem[:k]] = True
        linha[f"precision@{k}"] = round((top & fraude).sum() / k, 3)
        linha[f"eventos@{k}"] = int(evento[top & fraude].nunique())
    so_normal_e_fraude = df["tipo_anomalia"].isin(["NORMAL", TIPO]).to_numpy()
    linha["auc_vs_normal"] = round(roc_auc_score(fraude[so_normal_e_fraude], s[so_normal_e_fraude]), 3)
    return linha


def main() -> None:
    t0 = time.time()
    df = carregar_razao(DADOS / "razao_cosif_500k_v06_fracionamento.csv", mapeamento={"usuario": "usuario_criacao"})
    alcadas = pd.read_csv(DADOS / "alcadas_usuarios.csv")
    gabarito = pd.read_csv(DADOS / "gabarito_fracionamento.csv")
    evento = df["documento"].map(gabarito.set_index("documento")["evento"])
    n_eventos = gabarito["evento"].nunique()

    v05 = aplicar_detectores(df, TODOS + GENERICOS, desligar=["dia_nao_util"], plano_proprio=True)["score"]
    scores = {
        "v0.5 regras+estat": v05,
        "v0.5 Isolation Forest": pontuar_isolation_forest(df, engenheiradas=True, extras=True),
        "v0.6 fracionamento": score_fracionamento(df, alcadas),
    }
    resultado = pd.DataFrame([metricas(nome, s, df, evento) for nome, s in scores.items()])

    SAIDA.mkdir(parents=True, exist_ok=True)
    resultado.to_csv(SAIDA / "fracionamento_resultado.csv", index=False)
    frac = scores["v0.6 fracionamento"]
    alertas = df.loc[frac > 0, ["id_lancamento", "data_lancamento", "usuario_criacao", "contraparte", "conta_cosif", "valor", "tipo_anomalia"]]
    alertas.assign(score=frac[frac > 0]).sort_values("score", ascending=False).to_csv(SAIDA / "fracionamento_alertas.csv", index=False)

    # Visão de trabalho: grupos suspeitos (mesmo score, agrupado para revisão)
    grupos, grupo_por_linha = grupos_suspeitos(df, alcadas, score=frac)
    vinculo = pd.DataFrame({"grupo": grupo_por_linha, "evento": evento.reindex(grupo_por_linha.index)}).dropna()
    grupos["fraude"] = grupos.index.isin(vinculo["grupo"].unique())
    grupos.to_csv(SAIDA / "fracionamento_grupos.csv")

    curva = []
    for k in CURVA_GRUPOS:
        topo = grupos.head(k)
        curva.append({"grupos": k, "grupos_fraude": int(topo["fraude"].sum()),
                      "eventos": vinculo.loc[vinculo["grupo"].isin(topo.index), "evento"].nunique()})
    curva = pd.DataFrame(curva)

    achados = set(vinculo.loc[vinculo["grupo"].isin(grupos.head(MEDIA).index), "evento"])
    ev = gabarito.groupby("evento")["parcelas_no_evento"].first().rename("parcelas").reset_index()
    ev["achado"] = ev["evento"].isin(achados)
    ponto_cego = ev.groupby("parcelas").agg(eventos=("evento", "size"), achados=("achado", "sum")).reset_index()

    fila = grupos.head(TAMANHO_FILA)
    posicao = pd.Series(range(1, len(fila) + 1), index=fila.index)
    no_fila = grupo_por_linha[grupo_por_linha.isin(fila.index)]
    lanc = df.loc[no_fila.index, ["id_lancamento", "data_lancamento", "usuario_criacao", "contraparte", "conta_cosif",
                                  "centro_custo", "documento", "valor"]].copy()
    lanc["posicao"] = no_fila.map(posicao).astype(int)
    lanc["alcada"] = lanc["usuario_criacao"].map(alcadas.set_index("usuario")["limite_alcada"])
    lanc = lanc.sort_values(["posicao", "data_lancamento"])

    linha_v06 = resultado[resultado["metodo"] == "v0.6 fracionamento"].iloc[0]
    kpis = [
        ("Lançamentos analisados", len(df), "#,##0"),
        ("Grupos suspeitos (fila completa)", len(grupos), "#,##0"),
        (f"Fraude nos {ALTA} primeiros grupos (prioridade Alta)", grupos.head(ALTA)["fraude"].mean(), "0%"),
        (f"Eventos encontrados nos {MEDIA} primeiros grupos", vinculo.loc[vinculo["grupo"].isin(grupos.head(MEDIA).index), "evento"].nunique() / n_eventos, "0%"),
        ("AUC do detector (fracionamento × normal)", float(linha_v06["auc_vs_normal"]), "0.000"),
    ]
    from accounting_risk.enriquecimento import NIVEIS_ALCADA
    from accounting_risk.injecao import ParametrosFracionamento

    pf = ParametrosFracionamento()
    premissas = [
        ("Base", "Base fictícia COSIF de 500 mil lançamentos (5 tipos de anomalia rotulados por terceiros), acrescida das parcelas injetadas."),
        ("Contraparte", "Criada para todas as linhas: carteira de 120 contrapartes por conta, com concentração tipo Zipf."),
        ("Alçadas", "; ".join(f"{n.title()}: R$ {v['limite']:,.0f} ({v['participacao']:.0%} dos usuários)".replace(",", ".") for n, v in NIVEIS_ALCADA.items())),
        ("Fraude injetada", f"{pf.eventos} pagamentos fracionados em {pf.parcelas_min} a {pf.parcelas_max} parcelas de {pf.faixa_parcela[0]:.0%} a {pf.faixa_parcela[1]:.0%} da alçada, em até {pf.janela_dias_max} dias, com mesma contraparte e conta."),
        ("Detector", "Lançamentos abaixo da alçada, agrupados por usuário, contraparte e conta em janela de 7 dias; suspeito com 2 ou mais lançamentos e soma acima da alçada."),
        ("Score", "(soma ÷ alçada) × (média do valor ÷ alçada): quanto passou do limite e quão perto do limite ficou cada lançamento."),
        ("Prioridade", f"Convenção de capacidade de revisão: Alta = {ALTA} primeiros grupos; Média = até {MEDIA}; Baixa = demais. Fila exportada: {TAMANHO_FILA} grupos."),
        ("Gabarito", "Usado só para medir o resultado; o detector não o lê."),
    ]
    xlsx = gerar_relatorio_excel(
        SAIDA / "relatorio_fracionamento.xlsx", kpis=kpis, metodos=resultado.to_dict("records"), fila=fila,
        lancamentos=lanc, curva=curva, ponto_cego=ponto_cego, premissas=premissas, n_eventos=n_eventos,
        tamanho_alta=ALTA, tamanho_media=MEDIA,
    )

    pd.set_option("display.width", 200)
    print(f"Base: {len(df):,} lançamentos | {int((df['tipo_anomalia'] == TIPO).sum()):,} parcelas em {n_eventos} eventos\n")
    print(resultado.to_string(index=False))
    print("\nTipos entre os alertas do detector de fracionamento:", df.loc[frac > 0, "tipo_anomalia"].value_counts().to_dict())
    print("\nVisão por grupo (fila de revisão):")
    print(curva.assign(precision=lambda c: (c.grupos_fraude / c.grupos).round(2)).to_string(index=False))
    print(f"\nRelatório: {xlsx}")
    print(f"Tempo: {time.time() - t0:.0f}s | arquivos em {SAIDA}")


if __name__ == "__main__":
    main()

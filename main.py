"""Pipeline: gera razão fictício (plano COSIF), aplica regras + estatística + ML e salva relatórios."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd

from accounting_risk.avaliacao import comparar_metodos, fila_exclusiva_ml, recall_por_tipo
from accounting_risk.data_generator import gerar_razao
from accounting_risk.ml import combinar, pontuar_isolation_forest, pontuar_lof
from accounting_risk.scoring import (
    aplicar_detectores, aplicar_regras, avaliar, curva_por_limiar, resumo_por_detector, resumo_por_tipo,
)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
pd.set_option("display.max_colwidth", 70)


def comparar_com_ml(df, pontuado):
    scores = {
        "regras+estat": pontuado["score"],
        "IF bruto": pontuar_isolation_forest(df, engenheiradas=False),
        "IF eng": pontuar_isolation_forest(df, engenheiradas=True),
        "LOF eng": pontuar_lof(df, engenheiradas=True),
    }
    scores["regras+estat+IF"] = combinar(pontuado["score"], scores["IF eng"], peso_ml=1.0)
    return scores, comparar_metodos(df, scores), recall_por_tipo(df, scores)


def main() -> None:
    Path("data").mkdir(exist_ok=True)
    Path("output").mkdir(exist_ok=True)

    # ---------- cenário padrão: 12 tipos de anomalia (todos têm regra ou detector estatístico) ----------
    df = gerar_razao()
    df.to_csv("data/razao_ficticio.csv", index=False)
    pontuado = aplicar_detectores(df)
    sinalizados = pontuado[pontuado["score"] > 0].sort_values("score", ascending=False)
    colunas = [
        "id_lancamento", "data_lancamento", "conta_cosif", "valor", "tipo", "origem", "usuario_criacao",
        "usuario_aprovacao", "score", "nivel_risco", "regras_disparadas",
    ]
    sinalizados[colunas].to_csv("output/lancamentos_sinalizados.csv", index=False)
    resumo_por_tipo(pontuado).to_csv("output/resumo_por_tipo.csv")
    resumo_por_detector(pontuado).to_csv("output/resumo_por_detector.csv", index=False)

    print(f"Lançamentos analisados: {len(df)} | anomalias injetadas: {int(df['flag_anomalia'].sum())}")
    print(f"Lançamentos sinalizados: {len(sinalizados)}")
    print("\nSó regras:             ", avaliar(aplicar_regras(df)))
    print("Regras + estatística:", avaliar(pontuado))
    print("\nTrade-off por limiar de score:")
    print(curva_por_limiar(pontuado).to_string(index=False))

    _, metricas, por_tipo = comparar_com_ml(df, pontuado)
    metricas.to_csv("output/comparacao_metodos_padrao.csv", index=False)
    por_tipo.to_csv("output/recall_por_tipo_e_metodo_padrao.csv")
    print("\n=== CENÁRIO PADRÃO (12 tipos): ranking de risco por método ===")
    print(metricas.to_string(index=False))
    print("\nRecall@k por tipo de anomalia e método:")
    print(por_tipo.to_string())

    # ---------- cenário com anomalia desconhecida: nenhum detector de regra/estatística foi escrito para ela ----------
    df2 = gerar_razao(desconhecidas=True)
    pont2 = aplicar_detectores(df2)
    scores2, metricas2, por_tipo2 = comparar_com_ml(df2, pont2)
    metricas2.to_csv("output/comparacao_metodos_desconhecida.csv", index=False)
    por_tipo2.to_csv("output/recall_por_tipo_e_metodo_desconhecida.csv")
    fila = fila_exclusiva_ml(df2, pont2["score"], scores2["IF eng"])
    fila.to_csv("output/fila_exclusiva_ml.csv", index=False)
    print("\n=== CENÁRIO COM ANOMALIA DESCONHECIDA (13 tipos) ===")
    print(metricas2.to_string(index=False))
    print("\nRecall@k de centro_custo_incomum por método:")
    print(por_tipo2.loc[["centro_custo_incomum"]].to_string())
    print("\nFila exclusiva do ML (lançamentos sem sinal de regra/estatística):")
    print(fila.to_string(index=False))
    print("\nRelatórios salvos em output/")


if __name__ == "__main__":
    main()

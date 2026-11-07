"""Pipeline v0.2: gera razão fictício (plano COSIF), aplica regras + estatística e salva relatórios."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from accounting_risk.data_generator import gerar_razao
from accounting_risk.scoring import (
    aplicar_detectores, aplicar_regras, avaliar, curva_por_limiar, resumo_por_detector, resumo_por_tipo,
)


def main() -> None:
    df = gerar_razao()
    Path("data").mkdir(exist_ok=True)
    Path("output").mkdir(exist_ok=True)
    df.to_csv("data/razao_ficticio.csv", index=False)

    pontuado = aplicar_detectores(df)
    sinalizados = pontuado[pontuado["score"] > 0].sort_values("score", ascending=False)
    colunas = [
        "id_lancamento", "data_lancamento", "conta_cosif", "valor", "tipo", "origem", "usuario_criacao",
        "usuario_aprovacao", "score", "nivel_risco", "regras_disparadas",
    ]
    sinalizados[colunas].to_csv("output/lancamentos_sinalizados.csv", index=False)
    por_tipo = resumo_por_tipo(pontuado)
    por_detector = resumo_por_detector(pontuado)
    por_tipo.to_csv("output/resumo_por_tipo.csv")
    por_detector.to_csv("output/resumo_por_detector.csv", index=False)

    print(f"Lançamentos analisados: {len(df)} | anomalias injetadas: {int(df['flag_anomalia'].sum())}")
    print(f"Lançamentos sinalizados: {len(sinalizados)}")
    print("\nDistribuição por nível de risco:")
    print(pontuado["nivel_risco"].value_counts().to_string())
    print("\nv0.1 (só regras):   ", avaliar(aplicar_regras(df)))
    print("v0.2 (regras + estatística):", avaliar(pontuado))
    print("\nTrade-off por limiar de score:")
    print(curva_por_limiar(pontuado).to_string(index=False))
    print("\nRecall por tipo de anomalia:")
    print(por_tipo.to_string())
    print("\nDesempenho por detector:")
    print(por_detector.to_string(index=False))
    print("\nRelatórios salvos em output/")


if __name__ == "__main__":
    main()

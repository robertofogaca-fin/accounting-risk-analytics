"""Roda o pipeline em uma base externa (CSV ou Excel).

Exemplo (base fictícia COSIF de 500 mil lançamentos):
    python validar_base.py caminho/razao.csv --mapa usuario=usuario_criacao \\
        --desligar conta_invalida,conta_fora_vigencia,dia_nao_util --genericos --ml \\
        --alvo documento_repetido=DUPLICIDADE_DOCUMENTO --alvo centro_custo_raro=CC_INCOMPATIVEL \\
        --alvo historico_raro=HORARIO_INCOMUM --alvo usuario_volume_atipico=USUARIO_INCOMUM \\
        --alvo valor_atipico_conta=VALOR_FORA_PADRAO

Sem as colunas flag_anomalia/tipo_anomalia a base é tratada como real (sem rótulo): o script apenas
pontua e salva os alertas; a revisão passa a ser manual.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd

from accounting_risk.avaliacao import comparar_metodos, recall_por_tipo
from accounting_risk.carregador import carregar_razao
from accounting_risk.genericos import GENERICOS
from accounting_risk.scoring import TODOS, aplicar_detectores, avaliar, curva_por_limiar, resumo_por_detector, resumo_por_tipo

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)


def _pares(lista):
    return dict(item.split("=", 1) for item in (lista or []))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("caminho")
    ap.add_argument("--mapa", action="append", help="coluna_no_arquivo=coluna_padrao (repetível)")
    ap.add_argument("--desligar", default="", help="detectores a desligar, separados por vírgula")
    ap.add_argument("--plano-proprio", action="store_true", help="a base usa contas fora do elenco COSIF do projeto")
    ap.add_argument("--genericos", action="store_true", help="inclui os detectores genéricos de raridade")
    ap.add_argument("--ml", action="store_true", help="roda também o Isolation Forest")
    ap.add_argument("--alvo", action="append", help="detector=TIPO_ANOMALIA, para medir recall por detector")
    ap.add_argument("--saida", default="output/validacao")
    ap.add_argument("--decimal", default=".")
    args = ap.parse_args()

    t0 = time.time()
    df = carregar_razao(args.caminho, mapeamento=_pares(args.mapa), decimal=args.decimal)
    print(f"Base: {len(df):,} lançamentos | {df['data_lancamento'].min().date()} a {df['data_lancamento'].max().date()}")
    print("Colunas opcionais ausentes:", df.attrs["colunas_ausentes"] or "nenhuma")

    from accounting_risk.plano_contas import CODIGOS_SET

    no_plano = df["conta_cosif"].isin(CODIGOS_SET).mean()
    print(f"Contas no elenco COSIF do projeto: {no_plano:.1%} dos lançamentos")
    if no_plano < 0.5 and not args.plano_proprio:
        print("  ATENÇÃO: a maioria das contas não está no elenco; considere --plano-proprio")

    detectores = TODOS + (GENERICOS if args.genericos else [])
    pontuado = aplicar_detectores(
        df, detectores, desligar=[d for d in args.desligar.split(",") if d], plano_proprio=args.plano_proprio
    )
    print("\nDetectores ativos:", ", ".join(pontuado.attrs["detectores_ativos"]))
    for nome, motivo in pontuado.attrs["detectores_desligados"].items():
        print(f"  desligado: {nome} ({motivo})")

    saida = Path(args.saida)
    saida.mkdir(parents=True, exist_ok=True)
    sinalizados = pontuado[pontuado["score"] > 0].sort_values("score", ascending=False)
    colunas = [c for c in ["id_lancamento", "data_lancamento", "conta_cosif", "valor", "usuario_criacao", "score", "nivel_risco", "regras_disparadas", "tipo_anomalia"] if c in sinalizados.columns]
    sinalizados[colunas].head(200_000).to_csv(saida / "lancamentos_sinalizados.csv", index=False)
    print(f"\nSinalizados: {len(sinalizados):,} ({len(sinalizados) / len(df):.1%}) | níveis:", pontuado["nivel_risco"].value_counts().to_dict())

    if "flag_anomalia" not in df.columns:
        print("\nBase sem rótulo: alertas salvos para revisão manual em", saida)
        return

    ativos = [d for d in detectores if d.nome in pontuado.attrs["detectores_ativos"]]
    print("\nDesempenho geral (qualquer sinal):", avaliar(pontuado))
    print(curva_por_limiar(pontuado).to_string(index=False))
    por_tipo = resumo_por_tipo(pontuado)
    por_detector = resumo_por_detector(pontuado, ativos, alvos=_pares(args.alvo))
    por_tipo.to_csv(saida / "resumo_por_tipo.csv")
    por_detector.to_csv(saida / "resumo_por_detector.csv", index=False)
    print("\nRecall por tipo de anomalia:\n", por_tipo.to_string())
    print("\nPor detector:\n", por_detector.to_string(index=False))

    if args.ml:
        from accounting_risk.ml import combinar, pontuar_isolation_forest

        t1 = time.time()
        s_if = pontuar_isolation_forest(df, engenheiradas=True, extras=True)
        scores = {"regras+estat": pontuado["score"], "IF eng": s_if, "regras+estat+IF": combinar(pontuado["score"], s_if, 1.0)}
        metricas = comparar_metodos(df, scores)
        rec = recall_por_tipo(df, scores)
        metricas.to_csv(saida / "comparacao_metodos.csv", index=False)
        rec.to_csv(saida / "recall_por_tipo_e_metodo.csv")
        print(f"\nML (Isolation Forest, {time.time() - t1:.0f}s):\n", metricas.to_string(index=False))
        print("\nRecall@k por tipo e método:\n", rec.to_string())
    print(f"\nTempo total: {time.time() - t0:.0f}s | arquivos em {saida}")


if __name__ == "__main__":
    main()

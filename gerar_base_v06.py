"""Gera a base do teste v0.6: fracionamento abaixo da alçada.

Parte da base COSIF de 500 mil lançamentos e:
  1. atribui uma contraparte a TODAS as linhas (mesma regra para normais e fraudes);
  2. cria a tabela de alçadas por usuário (cadastro separado do razão);
  3. injeta os pagamentos fracionados, usando lançamentos normais do próprio usuário como molde.

Saídas (em data/, fora do Git; a base é reproduzível pela seed):
  razao_cosif_500k_v06_fracionamento.csv  base com a coluna contraparte e as parcelas injetadas
  alcadas_usuarios.csv                    cadastro de alçadas (insumo legítimo do detector)
  gabarito_fracionamento.csv              parcela -> evento (só para avaliação; o detector não lê)

Exemplo:
    python gerar_base_v06.py data/razao_cosif_500k_7pct_anomalias.csv
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd

from accounting_risk.enriquecimento import atribuir_contrapartes, gerar_alcadas
from accounting_risk.injecao import ParametrosFracionamento, injetar_fracionamento


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base")
    ap.add_argument("--saida", default="data")
    ap.add_argument("--coluna-usuario", default="usuario")
    args = ap.parse_args()

    df = pd.read_csv(args.base, parse_dates=["data_lancamento"])
    alcadas = gerar_alcadas(df[args.coluna_usuario])
    base = atribuir_contrapartes(df)
    base, gabarito = injetar_fracionamento(base, alcadas, ParametrosFracionamento(), coluna_usuario=args.coluna_usuario)

    saida = Path(args.saida)
    base["data_lancamento"] = base["data_lancamento"].dt.strftime("%Y-%m-%d")
    base.to_csv(saida / "razao_cosif_500k_v06_fracionamento.csv", index=False)
    alcadas.to_csv(saida / "alcadas_usuarios.csv", index=False)
    gabarito.to_csv(saida / "gabarito_fracionamento.csv", index=False)

    print(f"Base: {len(base):,} lançamentos ({len(gabarito):,} parcelas em {gabarito['evento'].nunique()} eventos)")
    print("Alçadas:", alcadas["nivel"].value_counts().to_dict())
    print("Tipos:", base["tipo_anomalia"].value_counts().to_dict())


if __name__ == "__main__":
    main()

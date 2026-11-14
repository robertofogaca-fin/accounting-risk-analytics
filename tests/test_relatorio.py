import pandas as pd
from openpyxl import load_workbook

from accounting_risk.relatorio import gerar_relatorio_excel


def test_relatorio_tem_as_abas_e_formulas(tmp_path):
    d = pd.Timestamp("2025-03-03")
    fila = pd.DataFrame({"usuario": ["ana"], "contraparte": ["CP1"], "conta_cosif": ["C1"], "data_inicio": [d],
                         "data_fim": [d + pd.Timedelta(days=2)], "lancamentos": [3], "soma": [27_800.0],
                         "alcada": [10_000.0], "fraude": [True]})
    lanc = pd.DataFrame({"posicao": [1], "id_lancamento": [10], "data_lancamento": [d], "usuario_criacao": ["ana"],
                         "contraparte": ["CP1"], "conta_cosif": ["C1"], "centro_custo": ["FIN"], "documento": ["DOC1"],
                         "valor": [9_500.0], "alcada": [10_000.0]})
    caminho = gerar_relatorio_excel(
        tmp_path / "r.xlsx",
        kpis=[("Lançamentos analisados", 3, "#,##0")],
        metodos=[{"metodo": "v0.6", "precision@100": 0.5, "precision@1000": 0.5, "eventos@1000": 1, "auc_vs_normal": 0.9}],
        fila=fila, lancamentos=lanc,
        curva=pd.DataFrame({"grupos": [1], "grupos_fraude": [1], "eventos": [1]}),
        ponto_cego=pd.DataFrame({"parcelas": [3], "eventos": [1], "achados": [1]}),
        premissas=[("Base", "teste")], n_eventos=1,
    )
    wb = load_workbook(caminho)
    assert wb.sheetnames == ["Resumo", "Fila_Revisao", "Lancamentos_Fila", "Curva_Revisao", "Ponto_Cego", "Premissas"]
    fila_ws = wb["Fila_Revisao"]
    assert fila_ws["B5"].value == "Alta"
    assert fila_ws["L5"].value == "=J5-K5"  # excedente é fórmula, não valor fixo

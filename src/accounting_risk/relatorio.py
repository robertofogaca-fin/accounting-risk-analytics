"""Relatório Excel do teste v0.6, no formato de fila de trabalho (como o relatório do conciliador).

Abas: Resumo, Fila_Revisao, Lancamentos_Fila, Curva_Revisao, Ponto_Cego, Premissas.
Colunas derivadas (dias, excedente, percentuais) são fórmulas, para a planilha continuar coerente se editada.
"""
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.chart import BarChart, LineChart, Reference
from openpyxl.formatting.rule import FormulaRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side

# Identidade visual RF Analytics (a mesma das publicações no LinkedIn e do projeto financial_statements)
FONTE = "Arial"
LARANJA = "F56600"      # destaques, indicadores e séries principais dos gráficos
PRETO = "111111"        # cabeçalhos de tabela e títulos
TEXTO_SUAVE = "595959"
CINZA = "F2F2F2"        # linhas alternadas
CINZA_MEDIO = "7F7F7F"  # série secundária dos gráficos
LARANJA_CLARO = "FDE4D0"
BORDA = Side(style="thin", color="BFBFBF")
CORES_PRIORIDADE = {"Alta": LARANJA_CLARO, "Média": CINZA, "Baixa": "FFFFFF"}
MOEDA = "#,##0.00;(#,##0.00);-"
ACAO = "Verificar se é uma mesma obrigação e se houve aprovação da alçada superior"


def _f(**kw):
    return Font(name=FONTE, **kw)


def _titulo(ws, texto, subtitulo=""):
    ws["A1"] = texto
    ws["A1"].font = _f(size=14, bold=True, color=PRETO)
    if subtitulo:
        ws["A2"] = subtitulo
        ws["A2"].font = _f(size=10, italic=True, color=TEXTO_SUAVE)


def _tabela(ws, linha, cabecalho, linhas, formatos=None, zebra=True):
    formatos = formatos or {}
    for j, nome in enumerate(cabecalho, start=1):
        c = ws.cell(row=linha, column=j, value=nome)
        c.font = _f(size=10, bold=True, color="FFFFFF")
        c.fill = PatternFill("solid", fgColor=PRETO)
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        c.border = Border(top=BORDA, bottom=BORDA, left=BORDA, right=BORDA)
    for i, valores in enumerate(linhas, start=linha + 1):
        for j, v in enumerate(valores, start=1):
            c = ws.cell(row=i, column=j, value=v)
            c.font = _f(size=10)
            c.border = Border(top=BORDA, bottom=BORDA, left=BORDA, right=BORDA)
            if j in formatos:
                c.number_format = formatos[j]
            if zebra and (i - linha) % 2 == 0:
                c.fill = PatternFill("solid", fgColor=CINZA)
    ws.row_dimensions[linha].height = 30
    return linha + 1, linha + len(linhas)


def _larguras(ws, larguras):
    for col, w in larguras.items():
        ws.column_dimensions[col].width = w


def _grafico(ch, titulo, eixo_x="", eixo_y="", legenda=True, pct=True):
    ch.title = titulo
    if eixo_x:
        ch.x_axis.title = eixo_x
    if eixo_y:
        ch.y_axis.title = eixo_y
    ch.roundedCorners = False
    ch.x_axis.delete = False
    ch.y_axis.delete = False
    if legenda:
        ch.legend.position = "b"
        ch.legend.overlay = False
    else:
        ch.legend = None
    if pct:
        ch.y_axis.number_format = "0%"
        ch.y_axis.scaling.min, ch.y_axis.scaling.max = 0, 1
    cores = [LARANJA, CINZA_MEDIO, PRETO]
    for i, serie in enumerate(ch.series):
        cor = cores[i % len(cores)]
        if isinstance(ch, LineChart):
            serie.graphicalProperties.line.solidFill = cor
            serie.graphicalProperties.line.width = 28575  # 2,25 pt
            serie.smooth = False
        else:
            serie.graphicalProperties.solidFill = cor
            serie.graphicalProperties.line.solidFill = cor
    ch.height, ch.width = 7.5, 16
    return ch


def gerar_relatorio_excel(caminho, *, kpis, metodos, fila, lancamentos, curva, ponto_cego, premissas, n_eventos,
                          tamanho_alta=100, tamanho_media=300):
    wb = Workbook()

    # Resumo ----------------------------------------------------------------------------------------
    ws = wb.active
    ws.title = "Resumo"
    _titulo(ws, "Fracionamento abaixo da alçada — resultado do teste v0.6",
            "Base fictícia de ~500 mil lançamentos com fraudes injetadas; gabarito usado só para medir o resultado")
    ws["A4"] = "Indicadores"
    ws["A4"].font = _f(bold=True, color=LARANJA)
    for i, (rotulo, valor, formato) in enumerate(kpis, start=5):
        ws.cell(row=i, column=1, value=rotulo).font = _f(size=10)
        c = ws.cell(row=i, column=2, value=valor)
        c.font, c.number_format = _f(size=11, bold=True, color=LARANJA), formato
    ws["D5"] = "Eventos injetados"
    ws["E5"] = n_eventos
    ws["D5"].font, ws["E5"].font = _f(size=10), _f(size=10, bold=True, color="0000FF")
    ws["D6"] = "valor de entrada: gabarito do teste"
    ws["D6"].font = _f(size=8, italic=True, color=TEXTO_SUAVE)

    linha_met = 6 + len(kpis) + 2
    ws.cell(row=linha_met - 1, column=1, value="Comparação com os detectores da v0.5 (por lançamento)").font = _f(bold=True, color=LARANJA)
    cab = ["Método", "Precision@100", "Precision@1.000", "Eventos no top 1.000", "% dos eventos", "AUC × normal"]
    linhas = [[m["metodo"], m["precision@100"], m["precision@1000"], m["eventos@1000"], None, m["auc_vs_normal"]]
              for m in metodos]
    ini, fim = _tabela(ws, linha_met, cab, linhas, {2: "0%", 3: "0%", 4: "0", 5: "0%", 6: "0.000"})
    for i in range(ini, fim + 1):
        ws.cell(row=i, column=5, value=f"=D{i}/$E$5")
    _larguras(ws, {"A": 46, "B": 15, "C": 15, "D": 18, "E": 14, "F": 14})

    notas = [
        "Como ler",
        "• Fila_Revisao: um grupo suspeito por linha (mesmo usuário, contraparte e conta, em até 7 dias), em ordem de prioridade.",
        f"• Prioridade pela posição na fila: Alta = {tamanho_alta} primeiros grupos; Média = até o {tamanho_media}º; Baixa = demais.",
        "• Precision: dos alertas revisados, quantos eram fraude. AUC perto de 0,5 = sorteio; perto de 1 = boa ordenação.",
        "• Os detectores da v0.5 não foram alterados: mostram o que enxergam numa fraude que imita o padrão normal.",
    ]
    for k, texto in enumerate(notas, start=fim + 2):
        ws.cell(row=k, column=1, value=texto).font = _f(size=9, bold=(texto == "Como ler"))

    ch = BarChart()
    ch.type = "bar"
    ch.add_data(Reference(ws, min_col=5, min_row=linha_met, max_row=fim), titles_from_data=True)
    ch.set_categories(Reference(ws, min_col=1, min_row=ini, max_row=fim))
    ws.add_chart(_grafico(ch, "Eventos encontrados nos 1.000 primeiros alertas", legenda=False), "H4")

    # Fila_Revisao ----------------------------------------------------------------------------------
    ws = wb.create_sheet("Fila_Revisao")
    _titulo(ws, "Fila de revisão — grupos suspeitos de fracionamento",
            "Ordenada por score. A coluna Gabarito existe só neste teste: na vida real, a revisão é que responde.")
    cab = ["Posição", "Prioridade", "Usuário", "Contraparte", "Conta COSIF", "Primeiro lançamento",
           "Último lançamento", "Dias", "Lançamentos", "Soma (R$)", "Alçada do usuário (R$)", "Excedente (R$)",
           "Soma / alçada", "Ação sugerida", "Gabarito (teste)"]
    linhas = []
    for pos, (_, g) in enumerate(fila.iterrows(), start=1):
        prioridade = "Alta" if pos <= tamanho_alta else "Média" if pos <= tamanho_media else "Baixa"
        linhas.append([pos, prioridade, g.usuario, g.contraparte, g.conta_cosif, g.data_inicio.to_pydatetime(),
                       g.data_fim.to_pydatetime(), None, int(g.lancamentos), round(g.soma, 2), g.alcada, None, None,
                       ACAO, "Fraude" if g.fraude else "Normal"])
    ini, fim = _tabela(ws, 4, cab, linhas,
                       {6: "dd/mm/yyyy", 7: "dd/mm/yyyy", 8: "0", 9: "0", 10: MOEDA, 11: MOEDA, 12: MOEDA, 13: "0.00x"},
                       zebra=False)
    for i in range(ini, fim + 1):
        ws.cell(row=i, column=8, value=f"=G{i}-F{i}+1")
        ws.cell(row=i, column=12, value=f"=J{i}-K{i}")
        ws.cell(row=i, column=13, value=f"=J{i}/K{i}")
        ws.cell(row=i, column=14).alignment = Alignment(wrap_text=False)
    for nome, cor in CORES_PRIORIDADE.items():
        ws.conditional_formatting.add(f"B{ini}:B{fim}", FormulaRule(
            formula=[f'$B{ini}="{nome}"'], fill=PatternFill("solid", fgColor=cor),
            font=Font(name=FONTE, bold=nome == "Alta", color=LARANJA if nome == "Alta" else PRETO)))
    ws.freeze_panes = f"D{ini}"
    ws.auto_filter.ref = f"A4:O{fim}"
    _larguras(ws, {"A": 9, "B": 11, "C": 13, "D": 13, "E": 15, "F": 13, "G": 13, "H": 7, "I": 12, "J": 14,
                   "K": 14, "L": 14, "M": 10, "N": 70, "O": 12})

    # Lancamentos_Fila ------------------------------------------------------------------------------
    ws = wb.create_sheet("Lancamentos_Fila")
    _titulo(ws, "Lançamentos dos grupos da fila", "Use a Posição para ligar cada lançamento ao seu grupo na Fila_Revisao")
    cab = ["Posição do grupo", "ID lançamento", "Data", "Usuário", "Contraparte", "Conta COSIF", "Centro de custo",
           "Documento", "Valor (R$)", "Valor / alçada"]
    linhas = [[int(r.posicao), int(r.id_lancamento), r.data_lancamento.to_pydatetime(), r.usuario_criacao,
               r.contraparte, r.conta_cosif, r.centro_custo, r.documento, r.valor, r.valor / r.alcada]
              for _, r in lancamentos.iterrows()]
    ini, fim = _tabela(ws, 4, cab, linhas, {3: "dd/mm/yyyy", 9: MOEDA, 10: "0%"})
    ws.freeze_panes = f"A{ini}"
    ws.auto_filter.ref = f"A4:J{fim}"
    _larguras(ws, {"A": 10, "B": 13, "C": 12, "D": 13, "E": 13, "F": 15, "G": 15, "H": 14, "I": 14, "J": 12})

    # Curva_Revisao ---------------------------------------------------------------------------------
    ws = wb.create_sheet("Curva_Revisao")
    _titulo(ws, "Quantos grupos revisar para achar a fraude", "Grupos revisados em ordem de prioridade")
    ws["A3"], ws["B3"] = "Eventos injetados", "=Resumo!E5"
    ws["A3"].font, ws["B3"].font = _f(size=10), _f(size=10, bold=True, color="008000")
    cab = ["Grupos revisados", "Grupos com fraude", "Eventos encontrados", "% dos eventos", "Precision"]
    linhas = [[int(r.grupos), int(r.grupos_fraude), int(r.eventos), None, None] for _, r in curva.iterrows()]
    ini, fim = _tabela(ws, 5, cab, linhas, {1: "#,##0", 2: "#,##0", 3: "#,##0", 4: "0%", 5: "0%"})
    for i in range(ini, fim + 1):
        ws.cell(row=i, column=4, value=f"=C{i}/$B$3")
        ws.cell(row=i, column=5, value=f"=B{i}/A{i}")
    _larguras(ws, {"A": 16, "B": 16, "C": 16, "D": 14, "E": 12})
    ch = LineChart()
    ch.add_data(Reference(ws, min_col=4, min_row=5, max_row=fim), titles_from_data=True)
    ch.add_data(Reference(ws, min_col=5, min_row=5, max_row=fim), titles_from_data=True)
    ch.set_categories(Reference(ws, min_col=1, min_row=ini, max_row=fim))
    ws.add_chart(_grafico(ch, "Eventos encontrados × precision", "Grupos revisados"), "G5")

    # Ponto_Cego ------------------------------------------------------------------------------------
    ws = wb.create_sheet("Ponto_Cego")
    _titulo(ws, f"Eventos encontrados nos {tamanho_media} primeiros grupos, por nº de parcelas",
            "Quanto menos parcelas, mais o fracionamento se parece com o movimento normal")
    cab = ["Parcelas no evento", "Eventos injetados", "Encontrados", "% encontrados"]
    linhas = [[f"{int(r.parcelas)} parcelas", int(r.eventos), int(r.achados), None] for _, r in ponto_cego.iterrows()]
    ini, fim = _tabela(ws, 4, cab, linhas, {2: "0", 3: "0", 4: "0%"})
    for i in range(ini, fim + 1):
        ws.cell(row=i, column=4, value=f"=IFERROR(C{i}/B{i},0)")
    t = fim + 1
    for col, valor in ((1, "Total"), (2, f"=SUM(B{ini}:B{fim})"), (3, f"=SUM(C{ini}:C{fim})"),
                       (4, f"=IFERROR(C{t}/B{t},0)")):
        c = ws.cell(row=t, column=col, value=valor)
        c.font = _f(size=10, bold=True)
        c.border = Border(top=Side(style="thin", color="000000"))
    ws.cell(row=t, column=4).number_format = "0%"
    _larguras(ws, {"A": 18, "B": 16, "C": 14, "D": 14})
    ch = BarChart()
    ch.add_data(Reference(ws, min_col=4, min_row=4, max_row=fim), titles_from_data=True)
    ch.set_categories(Reference(ws, min_col=1, min_row=ini, max_row=fim))
    ws.add_chart(_grafico(ch, "% dos eventos encontrados", legenda=False), "F4")

    # Premissas -------------------------------------------------------------------------------------
    ws = wb.create_sheet("Premissas")
    _titulo(ws, "Premissas e parâmetros do teste", "Detalhes em docs/v0.6_especificacao_fracionamento.md")
    ini, fim = _tabela(ws, 4, ["Item", "Valor"], [list(p) for p in premissas])
    for row in ws.iter_rows(min_row=ini, max_row=fim):
        row[1].alignment = Alignment(wrap_text=True, vertical="top")
    _larguras(ws, {"A": 32, "B": 95})

    for planilha in wb.worksheets:
        planilha.sheet_view.showGridLines = False
    wb.save(caminho)
    return Path(caminho)

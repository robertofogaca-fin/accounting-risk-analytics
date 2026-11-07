from datetime import date, timedelta

import pandas as pd
from dateutil.easter import easter

from accounting_risk.calendario import FERIADOS, NAO_UTEIS, dia_nao_util


def _datas(nome, tipo=None):
    return sorted(d for d, (n, t) in FERIADOS.items() if n == nome and (tipo is None or t == tipo))


def test_feriados_moveis_batem_com_a_pascoa():
    for ano in (2025, 2026, 2027):
        pascoa = easter(ano)
        assert pascoa - timedelta(days=2) in _datas("Sexta-Feira da Paixão")
        assert [pascoa - timedelta(days=48), pascoa - timedelta(days=47)] == [
            d for d in _datas("Carnaval") if d.year == ano
        ]
        assert pascoa - timedelta(days=46) in _datas("Quarta-Feira de Cinzas")
        assert pascoa + timedelta(days=60) in _datas("Corpus Christi")


def test_expediente_parcial_nao_e_dia_nao_util():
    assert pd.Timestamp("2026-02-18") not in NAO_UTEIS  # Cinzas
    assert pd.Timestamp("2026-12-31") not in NAO_UTEIS  # último dia útil do ano
    assert pd.Timestamp("2026-06-04") in NAO_UTEIS  # Corpus Christi


def test_dia_nao_util_marca_fim_de_semana_e_feriado():
    datas = pd.Series(pd.to_datetime(["2026-09-07", "2026-09-08", "2026-09-12", "2026-09-13"]))
    assert dia_nao_util(datas).tolist() == [True, False, True, True]


def test_todo_ano_tem_feriados_nacionais_fixos():
    for ano in (2025, 2026, 2027):
        for mes, dia in [(1, 1), (4, 21), (5, 1), (9, 7), (10, 12), (11, 2), (11, 15), (11, 20), (12, 25)]:
            assert date(ano, mes, dia) in FERIADOS

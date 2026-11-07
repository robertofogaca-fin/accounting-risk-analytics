"""Calendário bancário: feriados e dias sem expediente (2025 a 2027).

Tipos:
  feriado      feriado nacional (sem expediente bancário)
  facultativo  ponto facultativo com bancos sem expediente (Carnaval, Corpus Christi)
  parcial      dia com expediente reduzido; NÃO é dia não útil (Quarta-feira de Cinzas e
               último dia útil do ano, em que o fechamento contábil é feito normalmente)

2026 e 2027 vêm da lista informada pelo autor. 2025 foi montado a partir da data da Páscoa
(20/04/2025) para cobrir o período do razão fictício; confirme antes de uso real.
Não inclui feriados estaduais nem municipais.
"""
from datetime import date

import pandas as pd

FERIADOS = {
    # ---- 2025 ----
    date(2025, 1, 1): ("Confraternização Universal", "feriado"),
    date(2025, 3, 3): ("Carnaval", "facultativo"),
    date(2025, 3, 4): ("Carnaval", "facultativo"),
    date(2025, 3, 5): ("Quarta-Feira de Cinzas", "parcial"),
    date(2025, 4, 18): ("Sexta-Feira da Paixão", "feriado"),
    date(2025, 4, 21): ("Tiradentes", "feriado"),
    date(2025, 5, 1): ("Dia do Trabalhador", "feriado"),
    date(2025, 6, 19): ("Corpus Christi", "facultativo"),
    date(2025, 9, 7): ("Independência do Brasil", "feriado"),
    date(2025, 10, 12): ("Nossa Senhora Aparecida", "feriado"),
    date(2025, 11, 2): ("Finados", "feriado"),
    date(2025, 11, 15): ("Proclamação da República", "feriado"),
    date(2025, 11, 20): ("Consciência Negra", "feriado"),
    date(2025, 12, 25): ("Natal", "feriado"),
    date(2025, 12, 31): ("Último dia útil do ano", "parcial"),
    # ---- 2026 ----
    date(2026, 1, 1): ("Confraternização Universal", "feriado"),
    date(2026, 2, 16): ("Carnaval", "facultativo"),
    date(2026, 2, 17): ("Carnaval", "facultativo"),
    date(2026, 2, 18): ("Quarta-Feira de Cinzas", "parcial"),
    date(2026, 4, 3): ("Sexta-Feira da Paixão", "feriado"),
    date(2026, 4, 21): ("Tiradentes", "feriado"),
    date(2026, 5, 1): ("Dia do Trabalhador", "feriado"),
    date(2026, 6, 4): ("Corpus Christi", "facultativo"),
    date(2026, 9, 7): ("Independência do Brasil", "feriado"),
    date(2026, 10, 12): ("Nossa Senhora Aparecida", "feriado"),
    date(2026, 11, 2): ("Finados", "feriado"),
    date(2026, 11, 15): ("Proclamação da República", "feriado"),
    date(2026, 11, 20): ("Consciência Negra", "feriado"),
    date(2026, 12, 25): ("Natal", "feriado"),
    date(2026, 12, 31): ("Último dia útil do ano", "parcial"),
    # ---- 2027 ----
    date(2027, 1, 1): ("Confraternização Universal", "feriado"),
    date(2027, 2, 8): ("Carnaval", "facultativo"),
    date(2027, 2, 9): ("Carnaval", "facultativo"),
    date(2027, 2, 10): ("Quarta-Feira de Cinzas", "parcial"),
    date(2027, 3, 26): ("Sexta-Feira da Paixão", "feriado"),
    date(2027, 4, 21): ("Tiradentes", "feriado"),
    date(2027, 5, 1): ("Dia do Trabalhador", "feriado"),
    date(2027, 5, 27): ("Corpus Christi", "facultativo"),
    date(2027, 9, 7): ("Independência do Brasil", "feriado"),
    date(2027, 10, 12): ("Nossa Senhora Aparecida", "feriado"),
    date(2027, 11, 2): ("Finados", "feriado"),
    date(2027, 11, 15): ("Proclamação da República", "feriado"),
    date(2027, 11, 20): ("Consciência Negra", "feriado"),
    date(2027, 12, 25): ("Natal", "feriado"),
    date(2027, 12, 31): ("Último dia útil do ano", "parcial"),
}

# Dias sem expediente bancário (exclui os de expediente parcial)
NAO_UTEIS = sorted(pd.Timestamp(d) for d, (_, tipo) in FERIADOS.items() if tipo != "parcial")
_NAO_UTEIS_SET = frozenset(NAO_UTEIS)


def dia_nao_util(datas: pd.Series) -> pd.Series:
    """True para sábado, domingo ou feriado/ponto facultativo bancário."""
    d = pd.to_datetime(datas).dt.normalize()
    return (d.dt.day_of_week >= 5) | d.isin(_NAO_UTEIS_SET)

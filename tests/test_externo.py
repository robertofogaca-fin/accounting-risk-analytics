import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from accounting_risk import genericos
from accounting_risk.carregador import carregar_razao
from accounting_risk.data_generator import CODIGOS as OPERACIONAIS, gerar_razao
from accounting_risk.ml import construir_features
from accounting_risk.scoring import TODOS, aplicar_detectores


def _base_externa(n=400, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "data_lancamento": pd.to_datetime("2025-03-03") + pd.to_timedelta(rng.integers(0, 150, n), unit="D"),
            "conta_cosif": rng.choice(["9.9.9.99.99-9", "8.8.8.88.88-8"], size=n),  # fora do elenco do projeto
            "valor": np.round(5000 * np.exp(0.3 * rng.standard_normal(n)), 2),
            "usuario_criacao": rng.choice([f"u{i:02d}" for i in range(20)], size=n),
        }
    )


def test_carregador_mapeia_colunas_e_detecta_ausentes(tmp_path):
    csv = tmp_path / "razao.csv"
    csv.write_text(
        "data_lancamento;conta_cosif;valor;usuario\n2025-01-02;1.1.1.10.00-1;100,50;ana\n2025-01-03;1.1.1.10.00-1;200,00;bia\n",
        encoding="utf-8",
    )
    df = carregar_razao(csv, mapeamento={"usuario": "usuario_criacao"}, decimal=",")
    assert list(df.columns) == ["data_lancamento", "conta_cosif", "valor", "usuario_criacao"]
    assert df["valor"].tolist() == [100.5, 200.0]
    assert "tipo" in df.attrs["colunas_ausentes"] and "usuario_criacao" not in df.attrs["colunas_ausentes"]


def test_carregador_exige_colunas_obrigatorias(tmp_path):
    csv = tmp_path / "ruim.csv"
    csv.write_text("data,conta,valor\n2025-01-01,x,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="obrigatórias"):
        carregar_razao(csv)


def test_detectores_sem_colunas_ficam_desligados_e_motivo_e_registrado():
    df = _base_externa()
    p = aplicar_detectores(df, TODOS, plano_proprio=True)
    desligados = p.attrs["detectores_desligados"]
    assert "faltam colunas" in desligados["manual_alto"] and "faltam colunas" in desligados["sem_segregacao"]
    assert "conta_invalida" in desligados and "conta_fora_vigencia" in desligados
    assert "valor_atipico_conta" in p.attrs["detectores_ativos"]


def test_sem_plano_proprio_toda_conta_externa_e_invalida():
    p = aplicar_detectores(_base_externa(), TODOS)
    assert p["sinal_conta_invalida"].all()


def test_plano_proprio_deixa_a_estatistica_rodar_em_contas_externas():
    df = _base_externa()
    df.loc[10, "valor"] = 5000 * np.exp(0.3 * 8)  # valor muito atípico para a conta
    com = aplicar_detectores(df, TODOS, plano_proprio=True)
    sem = aplicar_detectores(df, TODOS, plano_proprio=False)
    assert com.loc[10, "sinal_valor_atipico_conta"] and not sem.loc[10, "sinal_valor_atipico_conta"]


def test_resultado_padrao_nao_muda_com_as_novas_opcoes():
    df = gerar_razao(n=3000, n_por_tipo=20, seed=3)
    p = aplicar_detectores(df)
    assert p.attrs["detectores_desligados"] == {}


def test_detectores_genericos():
    rng = np.random.default_rng(1)
    n = 600
    df = pd.DataFrame(
        {
            "conta_cosif": "C1",
            "valor": 1.0,
            "documento": [f"D{i}" for i in range(n)],
            "centro_custo": rng.choice(["A", "B"], size=n),
            "historico": "rotina",
            "usuario_criacao": rng.choice([f"u{i}" for i in range(20)], size=n),
        }
    )
    df.loc[5, "documento"] = df.loc[4, "documento"]  # documento repetido
    df.loc[7, "centro_custo"] = "Z"  # centro que a conta quase não usa
    df.loc[9, "historico"] = "fora do horario"  # texto raro em campo pouco variado
    df.loc[df.index[:200], "usuario_criacao"] = "u_ocupado"  # usuário com volume muito acima dos demais
    assert genericos.documento_repetido(df).sum() == 1 and genericos.documento_repetido(df)[5]
    assert genericos.centro_custo_raro(df)[7] and genericos.centro_custo_raro(df).sum() == 1
    assert genericos.historico_raro(df)[9] and genericos.historico_raro(df).sum() == 1
    assert genericos.usuario_volume_atipico(df).sum() == 200


def test_features_degradam_sem_colunas_opcionais():
    df = _base_externa()
    f = construir_features(df, engenheiradas=True, extras=True)
    assert "origem_manual" not in f.columns and "criador_eq_aprov" not in f.columns
    assert "usuario_criacao_freq_log" in f.columns and np.isfinite(f.to_numpy(dtype=float)).all()


def test_script_validar_base_roda_em_base_rotulada(tmp_path):
    raiz = Path(__file__).resolve().parents[1]
    csv = tmp_path / "razao.csv"
    gerar_razao(n=1500, n_por_tipo=10, seed=5).to_csv(csv, index=False)
    r = subprocess.run(
        [sys.executable, str(raiz / "validar_base.py"), str(csv), "--saida", str(tmp_path / "saida")],
        capture_output=True, text=True, cwd=raiz,
    )
    assert r.returncode == 0, r.stderr
    assert "Desempenho geral" in r.stdout and (tmp_path / "saida" / "resumo_por_tipo.csv").exists()

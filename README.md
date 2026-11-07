# Accounting Risk Analytics

Detecção de lançamentos contábeis anormais, evoluindo de **regras simples** para **estatística** e
**machine learning**, sobre um razão **fictício** com plano de contas no padrão **COSIF** (Banco Central).

## Roadmap

- [x] **v0.1** Gerador de razão fictício + regras determinísticas + score de risco
- [x] **v0.2** Plano de contas COSIF (com validação de dígito verificador) + estatística:
      z-score robusto, perfil de dígitos por usuário (tipo Benford), par usuário-conta raro, pico mensal
- [ ] **v0.3** Machine learning (Isolation Forest / LOF) e comparação regras × estatística × ML
- [ ] **v0.4** Dashboard Streamlit

## Plano de contas COSIF

Código no formato `X.X.X.XX.XX-D`, em que o 1º dígito é o grupo (1 Ativo Realizável, 2 Ativo Permanente,
3 Compensação Ativa, 4 Passivo Exigível, 6 Patrimônio Líquido, 7 Resultado Credor, 8 Resultado Devedor,
9 Compensação Passiva) e `D` é o dígito verificador.

O DV é calculado em módulo 10 com pesos 3-1-7 repetidos sobre os 7 dígitos da base. O algoritmo foi
inferido e conferido contra 17 códigos oficiais (`tests/test_plano_contas.py`).

> As 9 contas do projeto são um subconjunto **ilustrativo**. Valide códigos e títulos contra o elenco
> oficial vigente do Cosif. O BCB está reestruturando o elenco (Res. BCB 320), então confirme qual
> versão a sua instituição utiliza. O lançamento é de uma única perna (sem partida dobrada).

## Detectores

| Detector | Tipo | O que detecta | Peso |
|---|---|---|---|
| fim_de_semana | regra | Lançamento em sábado/domingo | 1 |
| valor_redondo | regra | Valor múltiplo de 1.000 | 2 |
| duplicidade | regra | Mesma data, conta, valor, tipo e centro de custo | 3 |
| manual_alto | regra | Lançamento manual acima do limite | 4 |
| sem_segregacao | regra | Mesmo usuário cria e aprova | 5 |
| historico_vazio | regra | Histórico ausente ou curto | 1 |
| conta_invalida | regra | Conta fora do plano COSIF (código inexistente ou DV errado) | 7 |
| valor_atipico_conta | estatística | z-score robusto (mediana/MAD, escala log) por conta | 5 |
| par_usuario_conta_raro | estatística | Usuário lança em conta que quase nunca movimenta | 3 |
| desvio_digitos_usuario | estatística | Excesso de um 1º dígito no perfil do usuário (ex.: valores logo abaixo da alçada) | 2 |
| pico_mensal | estatística | Quantidade mensal de lançamentos da conta fora do histórico | 1 |

**Score:** soma dos pesos disparados, em que 10 pontos = 100. Níveis: baixo (até 30), médio (até 60), alto (acima de 60).

**Sobre Benford:** a lei clássica só vale para valores que atravessam várias ordens de grandeza. Como cada conta
tem valor típico próprio, o detector compara o usuário com o perfil de dígitos das suas contas, calculado sem o
próprio usuário (leave-one-out). `benford_mad` fica disponível para dados de faixa ampla.

## Anomalias injetadas (11 tipos, rotuladas)

- **Conhecidas** (as regras pegam): fim de semana, valor redondo, duplicidade, manual alto, sem segregação, histórico vazio, conta inválida
- **Sutis** (regras fixas não pegam): valor atípico para a conta, par usuário-conta raro, fracionamento abaixo da alçada, pico mensal

## Resultados (dados sintéticos, seed 42)

| | Recall | Precision |
|---|---|---|
| v0.1: só regras | 63,6% | 87,5% |
| v0.2: regras + estatística | 100% | 64,6% |

Trade-off pelo rigor do alerta: score ≥ 60 gera 207 alertas com 100% de precision e 18,8% de recall;
qualquer sinal gera 1.704 alertas com 100% de recall. O score serve para priorizar a fila de revisão.

## Como rodar

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1      # PowerShell (no cmd: .venv\Scripts\activate.bat)
pip install -r requirements.txt
python main.py
python -m pytest
```

Saídas em `output/`: lançamentos sinalizados, resumo por tipo e resumo por detector.

## Limitações

- O recall de 100% vem de anomalias injetadas com sinal forte. Em dados reais não há rótulo e a calibração dos limiares precisa de revisão manual.
- Os limiares (z > 4, MAD de dígitos, pico) foram calibrados nestes dados sintéticos.
- Feriados ainda não entram na regra de fim de semana.
- Os falsos positivos vêm sobretudo dos detectores de triagem (dígitos, pico mensal), por isso pesam pouco no score.

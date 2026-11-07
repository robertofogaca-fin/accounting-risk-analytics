# Accounting Risk Analytics

Detecção de lançamentos contábeis anormais, evoluindo de **regras simples** para **estatística** e
**machine learning**, sobre um razão **fictício** que usa o **plano de contas COSIF** (Banco Central do Brasil).

## Roadmap

- [x] **v0.1** Gerador de razão fictício + regras determinísticas + score de risco
- [x] **v0.2** Detectores estatísticos: z-score robusto, perfil de dígitos por usuário (tipo Benford),
      par usuário-conta raro, pico mensal
- [x] **v0.3** Plano COSIF completo (4.401 contas, 9 dígitos + DV): validação de DV e de vigência
- [ ] **v0.4** Machine learning (Isolation Forest / LOF) e comparação regras × estatística × ML
- [ ] **v0.5** Dashboard Streamlit

## Plano de contas COSIF

Código no formato `X.X.X.XX.XX.XX-D` (ex.: `1.1.0.00.00.00-2` = DISPONIBILIDADES), com 6 níveis e dígito
verificador (DV). O 1º dígito é o grupo: 1 Ativo Realizável, 2 Ativo Permanente, 3 Compensação Ativa,
4 Passivo Exigível, 6 Patrimônio Líquido, 7 Resultado Credor, 8 Resultado Devedor, 9 Compensação Passiva.

O elenco está em `src/accounting_risk/dados/plano_contas_cosif.csv`: código, nome, início e fim de vigência
(AAAAMM) e natureza (Normal, Retificadora, Duplo posicionamento). Dele o projeto deriva nível e conta "folha"
(sem subcontas).

**DV:** módulo 10, pesos 3-7-1 repetidos da direita para a esquerda sobre os dígitos da base
(`DV = (-soma) mod 10`). O algoritmo foi inferido e conferido em 100% das 4.401 contas do arquivo e em
17 códigos oficiais da estrutura anterior (7 dígitos).

> Confirme a fonte e a versão do elenco no site do BCB antes de qualquer uso real.
> Premissa: o fim de vigência é inclusivo (a conta vale até o fim do mês informado).
> O lançamento do razão é de uma única perna (sem partida dobrada).

## Detectores

| Detector | Tipo | O que detecta | Peso |
|---|---|---|---|
| fim_de_semana | regra | Lançamento em sábado/domingo | 1 |
| valor_redondo | regra | Valor múltiplo de 1.000 | 2 |
| duplicidade | regra | Mesma data, conta, valor, tipo e centro de custo | 3 |
| manual_alto | regra | Lançamento manual acima do limite | 4 |
| sem_segregacao | regra | Mesmo usuário cria e aprova | 5 |
| historico_vazio | regra | Histórico ausente ou curto | 1 |
| conta_fora_vigencia | regra | Conta COSIF usada antes do início ou depois do fim da vigência | 6 |
| conta_invalida | regra | Código fora do elenco COSIF (inexistente ou DV errado) | 7 |
| valor_atipico_conta | estatística | z-score robusto (mediana/MAD, escala log) por conta | 5 |
| par_usuario_conta_raro | estatística | Usuário lança em conta que quase nunca movimenta | 3 |
| desvio_digitos_usuario | estatística | Excesso de um 1º dígito no perfil do usuário (ex.: valores logo abaixo da alçada) | 2 |
| pico_mensal | estatística | Quantidade mensal de lançamentos da conta fora do histórico | 1 |

**Score:** soma dos pesos disparados, em que 10 pontos = 100. Níveis: baixo (até 30), médio (até 60), alto (acima de 60).

**Sobre Benford:** a lei clássica só vale para valores que atravessam várias ordens de grandeza. Como cada conta
tem valor típico próprio, o detector compara o usuário com o perfil de dígitos das suas contas, calculado sem o
próprio usuário (leave-one-out). `benford_mad` fica disponível para dados de faixa ampla.

**Detectores estatísticos** só rodam em contas do elenco e, quando pedem uma base estatística (z-score, dígitos,
pico mensal), em contas com pelo menos 30 lançamentos.

## Anomalias injetadas (12 tipos, rotuladas)

- **Conhecidas** (as regras pegam): fim de semana, valor redondo, duplicidade, manual alto, sem segregação,
  histórico vazio, conta inválida, conta fora de vigência
- **Sutis** (regras fixas não pegam): valor atípico para a conta, par usuário-conta raro, fracionamento abaixo
  da alçada, pico mensal

Os lançamentos normais usam 12 contas folha do COSIF, de natureza Normal e vigentes em todo o período.

## Resultados (dados sintéticos, seed 42)

| | Recall | Precision |
|---|---|---|
| Só regras | 66,7% | 88,9% |
| Regras + estatística | 100% | 69,3% |

Trade-off pelo rigor do alerta: score ≥ 60 gera 301 alertas com 100% de precision e 25,1% de recall;
qualquer sinal gera 1.732 alertas com 100% de recall. O score serve para priorizar a fila de revisão.

Em 10 sementes diferentes, o recall ficou em 100% e a precision de regras + estatística variou de 56,8% a 81,2%.

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
- Os limiares (z > 3,5, MAD de dígitos, pico) foram calibrados nestes dados sintéticos.
- Feriados ainda não entram na regra de fim de semana.
- Os falsos positivos vêm sobretudo dos detectores de triagem (dígitos, pico mensal), por isso pesam pouco no score.
- O plano interno da instituição costuma ter contas próprias mapeadas ao COSIF; aqui o razão usa direto os códigos COSIF.

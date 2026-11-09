# Accounting Risk Analytics

Detecção de lançamentos contábeis anormais, evoluindo de **regras simples** para **estatística** e
**machine learning**, sobre um razão **fictício** que usa o **plano de contas COSIF** (Banco Central do Brasil).

## Roadmap

- [x] **v0.1** Gerador de razão fictício + regras determinísticas + score de risco
- [x] **v0.2** Detectores estatísticos: z-score robusto, perfil de dígitos por usuário (tipo Benford),
      par usuário-conta raro, pico mensal
- [x] **v0.3** Plano COSIF completo (4.401 contas, 9 dígitos + DV): validação de DV e de vigência; calendário de feriados bancários
- [x] **v0.4** Machine learning (Isolation Forest e LOF) e comparação regras × estatística × ML
- [x] **v0.5** Validação externa: carregador de bases, desligamento automático de detectores e detectores genéricos de raridade
- [ ] **v0.6** Dashboard Streamlit

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

## Calendário bancário

`src/accounting_risk/calendario.py` traz os feriados e dias sem expediente de 2025 a 2027. Fim de semana,
feriado nacional e ponto facultativo sem expediente bancário (Carnaval, Corpus Christi) contam como dia não útil.
Quarta-Feira de Cinzas e o último dia útil do ano têm expediente parcial e **não** são marcados, já que o
fechamento contábil acontece normalmente nesses dias.

2026 e 2027 vêm de uma lista informada pelo autor; 2025 foi derivado da data da Páscoa. Os feriados móveis
são conferidos em teste contra a data da Páscoa. Não há feriados estaduais ou municipais.

## Detectores

| Detector | Tipo | O que detecta | Peso |
|---|---|---|---|
| dia_nao_util | regra | Lançamento em fim de semana ou feriado/ponto facultativo bancário | 1 |
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

- **Conhecidas** (as regras pegam): dia não útil (fim de semana e feriado), valor redondo, duplicidade, manual alto, sem segregação,
  histórico vazio, conta inválida, conta fora de vigência
- **Sutis** (regras fixas não pegam): valor atípico para a conta, par usuário-conta raro, fracionamento abaixo
  da alçada, pico mensal

Os lançamentos normais usam 12 contas folha do COSIF, de natureza Normal e vigentes em todo o período.

## Resultados (dados sintéticos, seed 42)

| | Recall | Precision |
|---|---|---|
| Só regras | 66,7% | 88,9% |
| Regras + estatística | 100% | 63,0% |

Trade-off pelo rigor do alerta: score ≥ 60 gera 301 alertas com 100% de precision e 25,1% de recall;
qualquer sinal gera 1.905 alertas com 100% de recall. O score serve para priorizar a fila de revisão.

Em 10 sementes diferentes, o recall ficou em 100% e a precision de regras + estatística variou de 54,0% a 81,4%.

## Machine learning (v0.4)

Isolation Forest e LOF (scikit-learn), **não supervisionados**: o modelo nunca vê o rótulo. O que ele consegue
enxergar depende dos atributos (features) que recebe:

- **Brutos:** valor (log), dia da semana, dia do mês, origem, débito/crédito, tamanho do histórico.
- **Engenheirados:** os brutos + contexto que um analista escolheria olhar: z-score do valor na conta,
  frequência da conta, raridade do par usuário-conta e conta-centro de custo, volume mensal da conta,
  documento repetido, dia não útil, mesmo usuário criando e aprovando.

Como medir sem um limiar fixo: cada método gera um **ranking de risco** e é avaliado por precision@k
(k = número de anomalias reais), AUC e precisão média (AP).

### Resultado no cenário padrão (12 tipos, seed 42)

| Método | Precision@1200 | AUC | AP |
|---|---|---|---|
| Regras + estatística (score) | 77,1% | 0,991 | 0,841 |
| Isolation Forest, atributos brutos | 25,8% | 0,678 | 0,255 |
| Isolation Forest, atributos engenheirados | 66,2% | 0,980 | 0,741 |
| LOF, atributos engenheirados | 38,0% | 0,609 | 0,312 |
| Regras + estatística + Isolation Forest | 82,0% | 0,994 | 0,900 |

- O ML sem engenharia de atributos pouco acrescenta (AUC 0,678).
- Com bons atributos, o Isolation Forest chega perto das regras, mas não as supera nos tipos conhecidos e perde a explicação do alerta.
- Combinar os dois melhora o AP (0,841 para 0,900). Em 7 sementes, o AP combinado ficou entre 0,886 e 0,912.
- O LOF foi fraco nestes dados e só destaca tipos de atributo binário (dia não útil, valor redondo, mesmo usuário criando e aprovando).

### Onde o ML agrega: a anomalia que ninguém previu

`gerar_razao(desconhecidas=True)` cria um 13º tipo, `centro_custo_incomum` (conta usada com um centro de custo que
quase nunca usa), para o qual **nenhuma regra nem detector estatístico foi escrito** (recall de 1%).

A **fila exclusiva do ML** pega só lançamentos sem nenhum sinal de regra ou estatística e os ordena pelo score do
Isolation Forest. Nos 100 primeiros, 34% são anomalias reais, contra uma taxa-base de 0,53% (lift de 63,6 vezes). Em 7
sementes, a precision dos 100 primeiros ficou entre 32% e 50%.

> O ML achou a anomalia porque o atributo de raridade conta-centro de custo estava na lista. Ele não adivinha
> dimensões que ninguém forneceu: o ganho está em cobrir combinações sem escrever uma regra para cada uma.

## Validação externa (v0.5)

O pipeline também roda em uma base que não foi gerada por este projeto. O teste foi feito em uma base fictícia de
500 mil lançamentos com 5 tipos de anomalia rotulados por outra pessoa (35 mil anomalias, 7%).

### Como usar uma base nova

```powershell
python validar_base.py caminho\razao.csv --mapa usuario=usuario_criacao --plano-proprio --genericos --ml
```

- `--mapa coluna_no_arquivo=coluna_padrao` renomeia colunas (repetível). Obrigatórias: `data_lancamento`,
  `conta_cosif`, `valor`. Opcionais: `tipo`, `centro_custo`, `usuario_criacao`, `usuario_aprovacao`, `origem`,
  `historico`, `documento`.
- **Colunas que faltam:** os detectores que dependem delas são desligados e o script informa quais e por quê.
- `--plano-proprio`: a base usa contas fora do elenco COSIF do projeto (desliga `conta_invalida` e
  `conta_fora_vigencia`; os detectores estatísticos passam a rodar em todas as contas).
- `--desligar a,b`: desliga detectores sem poder de discriminação na base.
- `--genericos`: inclui detectores de raridade que não dependem de plano, calendário ou perfil conhecido
  (documento repetido, centro de custo raro para a conta, histórico raro, usuário com volume atípico).
- Com `flag_anomalia` e `tipo_anomalia` na base, o script mede precision e recall; sem rótulo, só pontua e
  salva os alertas para revisão manual (em `output/validacao/`).

Em código: `carregar_razao(caminho, mapeamento=...)` e `aplicar_detectores(df, detectores, desligar=..., plano_proprio=...)`.

### O que aconteceu na base de 500 mil lançamentos

| Configuração | Sinalizados | Precision | Recall |
|---|---|---|---|
| Padrão do projeto, só com o mapeamento de colunas | 100% da base | 7,0% | 100% |
| Calibrada (plano próprio, sem `dia_nao_util`, com detectores genéricos) | 7,0% da base | 99,7% | 99,9% |

Recall por tipo na configuração calibrada: documento duplicado 99,3%; centro de custo incompatível, horário
incomum, usuário incomum e valor fora do padrão, 100% cada. Tempo total: 13 segundos (10 s do Isolation Forest).

O que a validação mostrou:

1. **O padrão do projeto não serve sem calibrar.** Nenhuma conta da base está no elenco COSIF do projeto (os códigos
   estão na estrutura anterior, com DV diferente), então `conta_invalida` marca 100% dos lançamentos. Os detectores
   estatísticos também ficavam sem rodar por depender do elenco.
2. **Há regras sem poder de discriminação.** 28,5% dos lançamentos (normais e anômalos) caem em fim de semana ou
   feriado, então `dia_nao_util` não separa nada.
3. **Só um detector original funcionava**, o `valor_atipico_conta`. Os demais tipos de anomalia (documento, centro de
   custo, histórico, usuário) pediram detectores de raridade, o que motivou os genéricos.
4. **Níveis de risco não transferem.** Os pesos foram calibrados no cenário sintético; aqui só 77 lançamentos chegam
   a nível alto, e nível médio ou alto pega 40% do recall. Em outra base, é preciso recalibrar os pesos.
5. **O Isolation Forest ficou atrás das regras:** AUC 0,993 e AP 0,919, contra 0,999 e 0,993 de regras + estatística.
   Ele pegou só 42% dos usuários incomuns, porque o sinal está no volume do usuário (1,4% contra 1,0% dos demais).
6. **Cuidado ao interpretar:** nesta base as anomalias se entregam por um valor categórico exato (centro de custo
   "INEXISTENTE", um usuário, um texto de histórico). O resultado de ~100% mostra que o pipeline escala e se adapta,
   e não mede a capacidade de achar anomalias sutis.

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
- Na validação externa, os pesos e níveis de risco precisam ser recalibrados para cada base.
- Os resultados de ML dependem do cenário sintético. A anomalia desconhecida foi criada por mim, então o teste mostra o mecanismo, não o desempenho em dados reais.
- O LOF tem custo alto em bases grandes (centenas de milhares de linhas); o Isolation Forest escala melhor.
- O calendário cobre só feriados nacionais bancários (sem estaduais e municipais).
- Os falsos positivos vêm sobretudo dos detectores de triagem (dígitos, pico mensal), por isso pesam pouco no score.
- O plano interno da instituição costuma ter contas próprias mapeadas ao COSIF; aqui o razão usa direto os códigos COSIF.

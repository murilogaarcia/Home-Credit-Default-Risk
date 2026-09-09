# 🏦 Home Credit Default Risk

Projeto de **application scorecard** de risco de crédito desenvolvido para a competição [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) (Kaggle), com o objetivo de melhorar a concessão de crédito de uma instituição que busca ampliar a inclusão financeira da população sem acesso bancário — proporcionando uma experiência de empréstimo positiva e segura, inclusive para quem tem pouco ou nenhum histórico de crédito.

O modelo estima a probabilidade de inadimplência (`TARGET`) no momento da aplicação, combinando dados cadastrais do próprio pedido com o histórico de crédito do cliente em fontes internas e externas à Home Credit.

> 📌 Toda a documentação detalhada do pipeline (decisões, bugs corrigidos, código e resultados de cada etapa) está registrada em um [workspace no Notion](https://app.notion.com/p/Home-Credit-Default-Risk-3cf60c1b202381b6bc42f68040f220f5), organizado em 9 subpáginas, uma por fase do projeto.

---

## 🗂️ Estrutura do repositório

```
Home-Credit-Default-Risk/
├── 01 - EDA Publico e Metadados/     # Análise exploratória e metadados das 8 tabelas do desafio
├── 02- Teste Baseline/               # Baseline com o book público (application_train/test) isolado
├── 03-feature_engineering/           # Construção e cruzamento dos books derivados (Bureau, POS, etc.)
├── 04-Modelagem/                     # Seleção de variáveis, comparação de algoritmos, tuning e avaliação
├── Exemplo_AjustePoliticaCrédito.xlsx # Exemplo de faixas de risco e ponto de corte (Excel-first)
└── submissao_home_credit.csv         # Submissão final gerada para o Kaggle
```

---

## 🧩 Contexto e estrutura dos dados

A base é composta por 8 tabelas relacionadas por `SK_ID_CURR`, com granularidades diferentes:

| Tabela | Granularidade / conteúdo |
|---|---|
| `application_train` / `application_test` | Uma linha por cliente, com o `TARGET` (apenas no train) |
| `bureau` + `bureau_balance` | Histórico de crédito em outras instituições (bureau externo) |
| `previous_application` | Propostas de crédito anteriores dentro da própria Home Credit |
| `POS_CASH_balance` | Histórico de parcelamento de crédito consignado/POS |
| `installments_payments` | Histórico de pagamentos de parcelas |
| `credit_card_balance` | Histórico de utilização de cartão de crédito |

**Característica central do problema:** não existe eixo de calendário nem safra. O `TARGET` já vem cristalizado no momento da aplicação — não há filtro de maturidade nem validação *out-of-time* (OOT). A validação do modelo se limita a um **holdout estratificado**.

---

## 🔬 Pipeline desenvolvido

### 1. EDA (Análise Exploratória)
- Correlação inicial de Pearson com o `TARGET`, liderada por `EXT_SOURCE_1/2/3` (scores de risco externos — correlação negativa é esperada, pois score alto = bom pagador).
- Testes univariados mais robustos (KS, AUC, IV, % missing) por variável, usados como diagnóstico — não como substituto da análise multivariada.
- Decisão de sequência: (1) univariado por book isolado → (2) merge progressivo dos books → (3) modelo multivariado com o conjunto final selecionado.

### 2. Construção dos Books (Feature Engineering)
Feature engineering feito em **PySpark + Spark SQL**, combinando SQL puro (transformações fixas) com a API de DataFrame (geração dinâmica de expressões, quando listas de colunas/categorias dependem dos próprios dados).

Cada book (Bureau, POS_CASH, Installments, Credit Card, Previous Application) segue o mesmo esqueleto:
1. Colunas derivadas de negócio (ratios), antes de qualquer agregação;
2. Flags de janela temporal (`U3, U6, U9, U12, U18, U24, U36`);
3. Flags categóricas geradas dinamicamente (nunca hardcoded);
4. Estatísticas gerais e por janela (sum/mean/median/max/min/std);
5. Razões entre janelas (U3 vs. demais; janelas consecutivas);
6. Range, coeficiente de variação, curtose, aceleração (2ª derivada) e centro de massa (recência ponderada);
7. Join final entre blocos internos, com checagem de colunas duplicadas.

Volume final por book: Bureau (~1.100 variáveis), Installments (~250-300, propositalmente enxuto), Credit Card (ratios de utilização/juros/pagamento mínimo) e Previous Application (~2-3 mil variáveis, o mais aprofundado).

### 3. Cruzamento dos Books
Função reutilizável de *join* progressivo (`application` → Bureau → POS_CASH → Installments → Credit Card → Previous Application), com:
- **Prefixo por book** (`BUR_`, `POS_`, `INST_`, `CC_`, `PREV_`) para evitar colisão de nomes entre tabelas com nomenclatura genérica em comum;
- **Persistência + releitura em Parquet entre cada cruzamento**, resetando o *lineage* do Spark e evitando `StackOverflowError` na cadeia de joins;
- **Downcast para `float32`/`int32`** antes do split treino/teste, reduzindo o uso de memória pela metade;
- Salvamento de um Parquet intermediário a cada etapa, permitindo retomar o pipeline de qualquer ponto sem reprocessar tudo.

### 4. Seleção de Variáveis
Pipeline em 4 etapas:
1. **Baixa variância** — filtro por proporção da moda (calculado nos dados originais), não `VarianceThreshold` após `StandardScaler` (que só captura colunas literalmente constantes);
2. **IV/KS univariado**, com proteção de amostra mínima;
3. **Correlação**, ordenada por IV antes do corte, garantindo que a variável mais forte de um par correlacionado seja sempre preservada;
4. **Importância via LightGBM**, com categóricas tratadas nativamente (sem Target Encoder nessa etapa).

### 5. Modelagem e Comparação de Algoritmos
Framework **Champion-Challenger** comparando LightGBM, CatBoost, Random Forest, Gradient Boosting e Logistic Regression.

**Critério de escolha do campeão:** nunca o maior AUC/KS isolado — priorizar o **menor gap treino-teste** (estabilidade), mesmo que a métrica bruta não seja a mais alta.

**Rodada 1 — Book público isolado (baseline):**

| Modelo | AUC Train | AUC Test | KS Train | KS Test | Gap KS |
|---|---|---|---|---|---|
| LightGBM | 0.7957 | 0.7563 | 0.4394 | 0.3826 | 0.0568 |
| **CatBoost** | 0.7617 | 0.7560 | 0.3896 | 0.3849 | **0.0047** |
| Gradient Boosting | 0.7882 | 0.7559 | 0.4297 | 0.3834 | 0.0463 |
| Random Forest | 0.7371 | 0.7351 | 0.3525 | 0.3440 | 0.0085 |
| Logistic Regression | 0.7178 | 0.7200 | 0.3201 | 0.3236 | -0.0035 |

**🏆 Campeão: CatBoost** — AUC/KS praticamente empatado com o LightGBM, porém com gap de estabilidade ~12x menor.

**Rodada 2 — Incremento de Bureau + Installments** (validação de ganho real por book, não só empilhamento de variáveis):

| Book | Split | AUC | Gini | KS |
|---|---|---|---|---|
| Bureau | Treino | 0.7755 | 0.5510 | 0.4119 |
| Bureau | Teste | 0.7639 | 0.5277 | 0.3947 |
| + Installments | Treino | 0.7899 | 0.5798 | 0.4369 |
| + Installments | Teste | 0.7726 | 0.5452 | 0.4130 |

**Rodada 3 — Após incluir Credit Card:**

| Modelo | AUC Train | AUC Test | KS Train | KS Test | Gap AUC | Gap KS |
|---|---|---|---|---|---|---|
| LightGBM | 0.8026 | 0.7624 | 0.4497 | 0.3899 | 0.0402 | 0.0597 |
| Gradient Boosting | 0.7957 | 0.7625 | 0.4423 | 0.3913 | 0.0332 | 0.0510 |
| **CatBoost** | 0.7673 | 0.7616 | 0.3989 | 0.3898 | **0.0057** | **0.0091** |
| Random Forest | 0.7433 | 0.7386 | 0.3613 | 0.3588 | 0.0047 | 0.0025 |
| Logistic Regression | 0.7374 | 0.7395 | 0.3519 | 0.3568 | -0.0020 | -0.0049 |

CatBoost segue líder em estabilidade em todas as rodadas testadas e foi mantido como modelo campeão.

### 6. Ajuste de Hiperparâmetros (Optuna)
`StratifiedKFold(5)` + early stopping por fold, evitando treinar árvores completas quando o modelo já convergiu.

A função objetivo evoluiu ao longo do projeto:
- **V1** — KS puro, com ranges apertados de regularização;
- **V2** — combinação de KS e *capture rate* nos 2 primeiros decis (`0.4 * KS + 0.6 * capture_rate`), alinhando a otimização ao objetivo de negócio (concentração de risco, não só discriminação);
- **V3** — AUC penalizado pelo gap treino-validação (`SCORE = AUC - GAP_PENALTY * GAP`), monitorado diretamente na função objetivo;
- **V4** — AUC puro, com ranges mais largos de hiperparâmetros (`num_leaves` 2-256, `min_child_samples` 5-100), mantendo o early stopping como proteção.

Resultado de uma rodada de referência: KS (CV) 0.3951 · KS Treino 0.4236 / Teste 0.3950 (gap 0.0286) · ~979 árvores em média no CV.

### 7. Avaliação de Métricas e Controle de Overfitting
- Gap treino-teste como critério central de estabilidade em todas as rodadas.
- **SHAP** para interpretabilidade, validando que as variáveis de maior peso seguem lógica de negócio coerente (ex.: `EXT_SOURCE_*` alto reduz risco, `RATIO_UTILIZACAO_CREDITO` alto aumenta).
- Distinção conceitual entre **discriminação** (KS/AUC — separação ao longo de toda a curva) e **concentração** (capture rate — risco isolado no topo do ranking) como métricas complementares, não substitutas.

### 8. Visões de Negócio (Decis e Concentração de Risco)
- Score de negócio construído como `(1 - probabilidade) * 1000` (score baixo = risco alto, convenção tradicional de bureau) — **atenção:** essa polaridade é invertida em relação à probabilidade bruta usada na submissão do Kaggle.
- Gráfico de Event Rate by Decile (Treino vs. Teste), validando monotonicidade (taxa de evento decrescente do decil 0 ao 9) e proximidade treino-teste.
- Tabela de taxa de maus e taxa acumulada por decil, usada para sustentar o ponto de corte de aprovação.
- Para o objetivo de **concessão** (diferente de cobrança), o valor de negócio está em isolar o menor grupo possível que concentra a maior parte da inadimplência futura.

### 9. Política de Crédito e Ponto de Corte
> 🚧 Etapa aformalizada como política vigente — o conteúdo resume uma simulação de ponto de corte sobre a base de teste.

**Fluxo Excel-first** (ver `Exemplo_AjustePoliticaCrédito.xlsx`): score exportado sem faixa → análise manual da distribuição/taxa de maus por faixa → cortes definidos manualmente sobre o decil de maior risco → aplicados de volta de forma consistente entre treino, holdout e base oficial do Kaggle.

A simulação compara três cenários sobre o split de teste (92.254 clientes, 7.448 maus, taxa de maus de 8,07%): manter a política atual sem corte ("Vigente"), aplicar um corte de recusa usando **apenas o modelo com o book de Bureau**, e aplicar o mesmo corte usando o **modelo com Bureau + Installments Payments + Credit Card**. Premissas de receita: Cash Loans (91% do volume, juros de 22%) e Revolving Loans (9% do volume, juros de 49%), com ticket médio de R$ 599.496.

**Cenário 1 — recusando o decil 0 (corte de ~10% do público):**

| Cenário | Aprovados | % Aprovados | Maus | % Maus | Perda (R$) | Balanço (R$) | % Ganho vs. Vigente |
|---|---|---|---|---|---|---|---|
| Vigente (sem corte) | 92.254 | 100,00% | 7.448 | 8,07% | 4,47 bi | 9,05 bi | — |
| Somente Bureau | 83.028 | 90,00% | 4.918 | 5,92% | 2,95 bi | 9,21 bi | +1,83% |
| Bureau + Installments + Credit Card | 83.028 | 90,00% | 4.759 | 5,73% | 2,85 bi | 9,31 bi | **+2,88%** |

**Cenário 2 — corte mais conservador, de ~6,3% do público:**

| Cenário | Aprovados | % Aprovados | Maus | % Maus | Perda (R$) | Balanço (R$) | % Ganho vs. Vigente |
|---|---|---|---|---|---|---|---|
| Vigente (sem corte) | 92.254 | 100,00% | 7.448 | 8,07% | 4,47 bi | 9,05 bi | — |
| Somente Bureau | 86.488 | 93,75% | 5.626 | 6,50% | 3,37 bi | 9,29 bi | +2,74% |
| Bureau + Installments + Credit Card | 86.488 | 93,75% | 5.516 | 6,38% | 3,31 bi | 9,36 bi | **+3,47%** |

Nos dois cenários, para o **mesmo percentual de corte**, o modelo com Bureau + Installments + Credit Card recusa proporcionalmente mais maus pagadores que o modelo apenas com Bureau (redução adicional de ~150-160 maus por decil no mesmo grupo recusado), gerando um balanço financeiro melhor e quase dobrando o % de ganho sobre o cenário vigente. Isso é consistente com o ganho incremental de KS/AUC já observado na comparação de books da seção de Modelagem: mais fontes de informação concentram melhor o risco no grupo recusado, sem mudar o volume de aprovação.

**⚠️ Limitação central: a base é uma "foto" da carteira já aprovada.**
O `TARGET` só existe para clientes que **já tiveram crédito concedido** — não há informação de desempenho para quem foi recusado no passado (ou nunca solicitou crédito por barreiras de acesso). Isso tem duas implicações diretas para essa simulação:
- **Não há política vigente real para comparar.** O cenário "Vigente" simulado aqui é só a ausência de corte adicional sobre a base observada — não reflete a política de concessão que já existia quando esses clientes foram aprovados, que é desconhecida.
- **Viés de seleção (reject inference):** qualquer corte simulado assume que o comportamento dos clientes no decil recusado seria igual ao observado historicamente para clientes daquele perfil que foram aprovados. Não sabemos como se comportariam solicitantes com perfil semelhante que nunca chegaram a ser aprovados — o que pode subestimar ou superestimar o ganho real de qualquer novo ponto de corte.
- Por isso, os números acima devem ser lidos como uma comparação relativa entre modelos (Bureau vs. Bureau + Installments + Credit Card), e não como uma estimativa definitiva de ganho financeiro caso a política fosse implementada.

**Pendente:**
- Formalizar um ponto de corte de aprovação/recusa único, com justificativa de negócio explícita (não apenas os dois cenários exploratórios acima).
- Investigar técnicas de *reject inference* para mitigar o viés de a base representar só a carteira aprovada.
- Considerar o contexto de inclusão financeira do público-alvo na calibração do corte — evitar que a política penalize desproporcionalmente clientes sem histórico suficiente.

---

## 📈 Resultado da submissão

Pipeline completo submetido ao Kaggle (`submissao_home_credit.csv`), alcançando **~0.770 de AUC-ROC (privado)**, o Kaggle exige a probabilidade bruta da classe positiva (`TARGET=1`), não o score de negócio invertido usado nas visões de decil.

---

## 🛠️ Tecnologias e ferramentas

- **Feature engineering:** PySpark, Spark SQL
- **Modelagem:** LightGBM, CatBoost, Scikit-learn (Random Forest, Gradient Boosting, Logistic Regression)
- **Tuning:** Optuna (`StratifiedKFold` + early stopping)
- **Interpretabilidade:** SHAP
- **Armazenamento intermediário:** Parquet
- **Documentação do processo:** Notion

---

## 📎 Documentação completa

Todas as decisões técnicas, trechos de código e discussões detalhadas de cada etapa estão disponíveis no [workspace do Notion](https://app.notion.com/p/Home-Credit-Default-Risk-3cf60c1b202381b6bc42f68040f220f5).

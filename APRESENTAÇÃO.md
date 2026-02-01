## Apresentação
Responder essas perguntas em perspectiva do projeto em questão:

#### Introdução

◻ O título deve expressar, o mais fielmente possível, o conteúdo temático
do trabalho: 
   * Mitigação de Viés Algorítmico em Cenários de Desbalanceamento Sociodemográfico: Uma Análise Comparativa entre Foundation Models (TabPFN) e Gradient Boosting (XGBoost) nas Eleições Municipais do Ceará

◻ Exposição sintética de como se chegou ao tema de investigação, qual a
gênese do problema, por que se fez tal opção, se houve antecedentes:
   * Contar historia sobre troca do tema de mesarios para esse em detrimento das dificuldades encontradas com o tema proposto, ademais, abordar como os elementos pesquisados durante o processo fomentaram a mudança de tema para o presente.
◻ Único momento em que o pesquisador pode referir-se a motivos de
ordem pessoal;

#### Problema Fomentador da Pesquisa

◻ Exposição mais objetiva e técnica de colocar o problema
◻ Esclarecer a dificuldade específica com a qual se defronta e que se
pretende resolver por intermédio da pesquisa
◻ Como o tema está problematizado e por que ele precisa ainda ser
pesquisado

  * Nesse ponto, acredito que seja possível e cabível falar tanto sobre a necessidade de mitigar viés algorítmico em cenários de desbalanceamento sociodemográfico, como também expor o TabPFN por essa pesctiva em comparação com o XGBoost. É necessario detalhar tanto tecnicamente como socialmente. Ademais é preciso fundamentar corretamente a necessidade de mitigar viés algorítmico em cenários de desbalanceamento sociodemográfico.

◻ Trata-se de delimitar e circunscrever o tema-problema
◻ Para o tema ser problematizado é preciso ter uma ideia muito clara do problema a ser resolvido

  * Modelos como o XGBoost são excelentes em performance, mas 'herdam' o racismo e o sexismo estrutural dos dados históricos. O TabPFN, por ser um modelo de fundação, pode atuar como um neutralizador de viés. Dessa forma é necessario abordar essa problematica de uma forma cientifica e técnica, mas que também seja acessível para pessoas que não são da área de ciência de dados. Portanto para o tema problema é não apenas focar em quem venceu ou perdeu, mas sim em como os modelos se comportam em cenários de desbalanceamento sociodemográfico. E em que medida a arquitetura de aprendizado em contexto do TabPFN reduz a disparidade de gênero e raça em comparação com o XGBoost em cenarios de pequena escala, como os de eleições municipais no Ceará.

◻ O problema, antes de ser considerado, deve ser analisado sob o
aspecto de sua valorização:
    Viabilidade – Pode ser eficazmente resolvido através da pesquisa.
    Relevância – Deve ser capaz de trazer conhecimentos novos.

  * Sobre a relevância, acredito que seja possível argumentar que o TabPFN é um modelo relativamente novo e que ainda não foi amplamente estudado em cenários de desbalanceamento sociodemográfico. Ademais, é possível argumentar que a pesquisa pode trazer conhecimentos novos sobre a capacidade do TabPFN de mitigar viés algorítmico em cenários de desbalanceamento sociodemográfico. Provar que um modelo pré-treinado pode ser mais "justo" que um modelo treinado do zero em um dataset desbalanceado é um avanço significativo para a área de IA responsável. Também é importante ressaltar que a pesquisa destaca como o uso de algoritmos para predição de candidaturas pode perpetuar desigualdades sociais existentes, o que é um problema relevante para a área de ciência de dados e para a sociedade como um todo.

  * Agora sobre a viabilidade, acredito que seja possível argumentar que a pesquisa é viável, pois o TabPFN é um modelo de código aberto e está disponível para uso em pesquisas acadêmicas. Ademais, cabe reafirmar que a pesquisa é viável, pois o dataset utilizado é público e está disponível para uso em pesquisas acadêmicas. Continuando, como o TabPFN opera sobre in-context learning, ele não necessita de treinamento, o que torna a pesquisa viável em termos de tempo e recursos computacionais, eliminando a necessidade de longos processos de treinamento e ajuste de hiperparâmetros. Permitindo o foque na análise comparativa entre os modelos em cenários de desbalanceamento sociodemográfico.

##### Analise sob o aspecto da Relevância

◻ Novidade – Estar adequado ao estado atual da evolução científica.
◻ Exequibilidade – Pode chegar a uma conclusão válida.
◻ Oportunidade – Atender a interesses particulares e gerais.

  * Sobre esses aspectos, acredito que seja possível argumentar que a pesquisa é relevante, pois o TabPFN é um modelo relativamente novo e que ainda não foi amplamente estudado nesse cenario em especifico. Ademais, é possível argumentar que a pesquisa pode trazer conhecimentos novos sobre a capacidade de novos modelos de IA de mitigar viés algorítmico em cenários de desbalanceamento sociodemográfico. Provar que modelos pré-treinados podem ser mais "justos" que modelos treinados do zero em um dataset desbalanceado é um avanço significativo para a área de IA responsável. Também é importante ressaltar que a pesquisa destaca como o uso de algoritmos para predição de candidaturas pode perpetuar desigualdades sociais existentes, o que é um problema relevante para a área de ciência de dados e para a sociedade como um todo. Ademais, cabe pontuar o agnosticismo de treinamento e amortização de interferencia em função da mudança de paradigma.

  * Podemos falar sobre o refinamento metodológico em cima do trabalho já presente, também conseguirmos dizer sobre a mitigação do viés indutivo e tudo isso em small data. Acho necessario também  pontuar especialmente como investigar ferramentas que mitiguem viés algorítmico em cenários de desbalanceamento sociodemográfico é um problema relevante para a área de ciência de dados e para a sociedade como um todo. Contribuindo para uma transparência e justiça algorítmica. 

  * Devemos revisar a matriz de confusão e as métricas de acurácia, precisão, recall para verificar se os resultados estão corretos. Ademais, é importante verificar se as métricas de fairness estão corretas e se os resultados estão de acordo com o esperado. 

#### Hipóteses e Objetivos

1. Hipótese Central

Hipótese: "A arquitetura de inferência bayesiana do TabPFN v2.5 é capaz de amortizar o viés sociodemográfico em datasets eleitorais de pequena escala, resultando em predições com maior equidade (DIR e EOD) do que modelos baseados em otimização de perda (XGBoost)."

2. Objetivo Geral (O Propósito Principal)

Objetivo Geral: Demonstrar a superioridade dos modelos de fundação tabular na mitigação de vieses de gênero e raça em cenários de desbalanceamento sociodemográfico, utilizando as eleições do Ceará como estudo de caso.

* Enfase no Demonstrar, pois exige evidências quantitativas, fugindo do trivial "estudar".

3. Objetivos Específicos 

    1. Objetivo de Avaliar (Medição do Problema):
        Medir o grau de amplificação de viés demográfico executado pelo algoritmo XGBoost quando treinado com dados históricos desbalanceados.

    2. Objetivo de Compreender (Diferenciação Técnica):
        Diferenciar o impacto da arquitetura (Transformadores vs. Árvores) na estabilidade da métrica MCC em ambientes de dados escassos (Small Data).

    3. Objetivo de Avaliar (Contraste):
        Contrastar as métricas de equidade (Impacto Disparate) entre o modelo de fundação "nativo" e o modelo tradicional otimizado.

| Se a Hipótese diz... | O Objetivo Específico deve... | Verbo |
| :--- | :--- | :--- |
| Que o XGBoost tem viés alto. | Calcular esse viés numericamente. | Medir ou Estimar |
| Que o TabPFN é mais estável. | Comparar as variações de performance. | Contrastar ou Diferenciar |
| Que a arquitetura é a causa. | Isolar o efeito da arquitetura nos testes. | Determinar ou Demonstrar |

  * Não podemos usar verbos trivias como "Analisar", "Estudar", "Verificar", "Pesquisar", "Investigar", etc. Devemos usar verbos mais específicos, como "Medir", "Estimar", "Contrastar", "Diferenciar", "Determinar" ou "Demonstrar".

#### Fontes, Procedimentos e Etapas

##### Fontes

* Fontes Documentais: Utilizaremos dados primários extraídos do Repositório de Dados Eleitorais do TSE.

* Fontes Bibliográficas: A pesquisa fundamenta-se em literatura atualizada (2024-2026) sobre Foundation Models para dados tabulares e métricas de Fairness-Aware AI, como as publicações sobre a arquitetura TabPFN v2.5.

* Fontes Empíricas: Os dados gerados pelos próprios experimentos de benchmark (logs de execução, métricas de MCC e coeficientes de equidade) servirão como evidência empírica para a conclusão. 

##### Procedimentos

* Abordagem: Pesquisa quantitativa de natureza aplicada, utilizando o método experimental.

* Procedimento de Comparação (Isolamento de Fatores):

    * Pré-processamento: Codificação de variáveis categóricas e tratamento de valores nulos, mantendo a integridade das variáveis sensíveis (Gênero e Raça).

    * Cenário de Controle (XGBoost): Treinamento do modelo com otimização de hiperparâmetros e aplicação de pesos de classe para compensar o desbalanceamento.

    * Cenário Experimental (TabPFN): Aplicação do modelo de fundação via In-Context Learning com o parâmetro balance_probabilities=True.

* Instrumentos de Medição: A performance será medida pelo MCC e a equidade pelos índices DIR (Impacto Disparate) e EOD (Igualdade de Oportunidade). A significância estatística será validada pelo teste de Wilcoxon (p-valor).

##### Etapas do Processo de Investigação

* Auditoria de Viés nos Dados Brutos: Identificação das disparidades históricas de eleição entre gêneros e raças no Ceará para estabelecer o bias de referência.

* Execução do Benchmark de Performance: Realização de validação cruzada (5−fold) para coletar as métricas de eficácia preditiva (MCC).

* Cálculo dos Índices de Equidade Algorítmica: Extração dos coeficientes DIR e EOD de ambos os modelos para quantificar a mitigação ou amplificação do viés.

* Análise de Significância e Inferência: Aplicação de testes estatísticos para determinar se a superioridade do TabPFN é consistente e não fruto do acaso.

* Sintese e Redação Científica: Consolidação dos achados no formato de artigo acadêmico (Projeto 2).

### Bibliografia

* [1] Tem que por ai, to cansado chefe

### TEMOS QUE 
* O grupo deverá enviar o documento de projeto de pesquisa na data especificada
* O grupo deverá apresentar, através de slides, a proposta de pesquisa para a turma na data especificada
* O tempo reservado para apresentação de cada grupo será de 10
minutos

Itens Exigidos no Documento:
* Título
* Problema de Pesquisa - qualificação do principal problema a ser
abordado
* Justificativa – contribuições técnicas e científicas esperadas
* Objetivos (geral e específicos)   
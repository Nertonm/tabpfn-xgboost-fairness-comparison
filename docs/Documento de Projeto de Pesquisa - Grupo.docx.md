**Mitigação de Viés Algorítmico em Cenários de Desbalanceamento Sociodemográfico: Uma Análise Comparativa entre Foundation Models e Gradient Boosting nas Eleições Municipais do Ceará** 

**Luana Teles Alves**  
**Luann Alves Pereira de Lima**  
**Thiago Nerton Macedo Alves**  
**Vitória do Nascimento Pontes**

**Juazeiro do Norte \- Ceará, 02/2026**

 **Documento de Projeto de Pesquisa**

1. **Identificação da Proposta**

**Título do Projeto:** Mitigação de Viés Algorítmico em Cenários de Desbalanceamento Sociodemográfico: Uma Análise Comparativa entre Foundation Models e Gradient Boosting nas Eleições Municipais do Ceará.  
**Instituição de Execução:** Universidade Federal do Cariri (UFCA) \- Centro de Ciências e Tecnologia (CCT)  
**Proponentes:** Luana Teles Alves ([luana.teles@aluno.ufca.edu.br](mailto:luana.teles@aluno.ufca.edu.br)), Luann Alves Pereira de Lima ([luann.alves@aluno.ufca.edu.br](mailto:luann.alves@aluno.ufca.edu.br)), Thiago Nerton Macedo Alves ([thiago.nerton@aluno.ufca.edu.br](mailto:thiago.nerton@aluno.ufca.edu.br)), Vitória do Nascimento Pontes (vitoria.pontes@aluno.ufca.edu.br).

2. **Problema de Pesquisa**

O uso de algoritmos de aprendizado de máquina para apoiar processos decisórios tem se intensificado em diferentes domínios sociais, incluindo contextos sensíveis como eleições, políticas públicas e representação política. Entre esses algoritmos, modelos baseados em árvores de decisão, como o XGBoost, destacam-se pelo alto desempenho preditivo em dados tabulares e pela ampla adoção em aplicações práticas. Contudo, evidências na literatura indicam que tais modelos podem amplificar vieses sociodemográficas presentes nos dados históricos, especialmente em cenários de desbalanceamento entre grupos minoritários \[1\].

O viés algorítmico, nesse contexto, não emerge como uma falha isolada do modelo, mas como resultado de um processo estrutural no qual desigualdades históricas de raça e gênero são codificadas nos dados de treinamento. Modelos como o XGBoost, ao serem treinados por meio da otimização direta de uma função de perda global, tendem a priorizar padrões majoritários, uma vez que estes dominam estatisticamente o conjunto de dados. Como consequência, a acurácia global pode ser elevada ao custo de erros sistematicamente mais altos para grupos sub-representados, configurando violações de métricas de justiça algorítmica.

Diante desse cenário, a recente emergência dos Modelos de Fundação Tabular, especificamente aqueles baseados em inferência bayesiana via Aprendizado em Contexto, como o TabPFN, introduz um novo paradigma de modelagem. Diferente do treinamento iterativo tradicional, esses modelos constroem suas predições baseando-se em milhões de distribuições prévias. No entanto, ainda não está evidenciado na literatura se essa arquitetura de "conhecimento pré-treinado" atua como um mitigador natural de vieses em conjuntos de dados de pequena escala ou se ela é tão suscetível à reprodução de preconceitos quanto as árvores de decisão.

Portanto, coloca-se como problema central desta pesquisa investigar em que medida a arquitetura de inferência bayesiana do TabPFN reduz a disparidade de predição entre grupos minoritários, comparando a estabilidade de performance preditiva em relação aos modelo XGBoost, utilizando cenários de dados eleitorais municipais de pequena escala.

3. **Justificativa**

A relevância da pesquisa fundamenta-se no crescimento de sistemas de decisão algorítmica em esferas críticas da sociedade, como as estatísticas eleitorais. Diante de um cenário onde dados históricos podem carregar o peso de desigualdades estruturais, a simples aplicação de modelos orientados exclusivamente à otimização de desempenho global, sem auditoria de viés, representa um risco à integridade democrática. A pesquisa justifica-se, portanto, pela necessidade de validar arquiteturas de Inteligência Artificial que não apenas reproduzam, mas que mitiguem tais disparidades em contextos de desequilíbrio de dados.

Este trabalho procura contribuir para o estado da arte ao investigar a transição de paradigmas no Aprendizado de Máquina Tabular: a passagem da Minimização de Perda Empírica para a Inferência Bayesiana Amortizada. A literatura atual ainda carece de estudos empíricos que demonstrem se o TabPFN, pré-treinado em larga escala, possui um viés indutivo capaz de oferecer maior equidade algorítmica do que o XGBoost em conjuntos de dados desbalanceados. Ao contrastar essas arquiteturas, o projeto tenta avançar o conhecimento de uma etapa meramente descritiva, diagnóstico do viés, para uma etapa experimental-explicativa, avaliação da mitigação do viés, oferecendo evidências sobre a robustez dos Transformers em cenários onde técnicas clássicas de reamostragem frequentemente falham.

O projeto entregará um benchmark reprodutível para a auditoria de algoritmos em dados governamentais brasileiros. A contribuição reside na comparação entre a adoção de métricas sensíveis ao desbalanceamento, como o Coeficiente de Correlação de Matthews (MCC), aliada a indicadores de justiça como o Impacto Disparate (DIR) e a Diferença de Oportunidade (EOD), em relação ao uso de métricas tradicionais, como Acurácia ou ROC-AUC, para a validação de modelos eleitorais. Além disso, o trabalho avalia a viabilidade de implementação de modelos que dispensam o ajuste fino de hiperparâmetros, reforçando à comunidade técnica um protocolo para o desenvolvimento de sistemas de suporte à decisão mais eficientes e eticamente alinhados com os princípios de governança algorítmica.

4. **Objetivos**

**Objetivo Geral:** 

* Demonstrar, por meio de experimentação controlada, a diferença técnica da arquitetura de Modelos de Fundação Tabular (Tabular Foundation Models) em relação à robustez intrínseca contra vieses de grupos minoritários, especificamente em cenários de desbalanceamento sociodemográfico severo \[1, 2\]. O estudo busca validar a eficácia do aprendizado em contexto na remoção de efeitos causais de atributos protegidos sem a necessidade de ajuste de pesos \[2\], contrastando essa abordagem com métodos tradicionais de otimização de perda, como o XGBoost, que tendem a reproduzir e amplificar preconceitos históricos latentes nos dados de treinamento \[3, 4\]. A validação empírica será realizada através de um estudo de caso nas eleições municipais do Ceará de 2024, utilizando dados oficiais do Tribunal Superior Eleitoral (TSE) \[5\].

**Objetivos Específicos:**

1) Quantificar a amplificação algorítmica de viés: Medir o grau em que o algoritmo Gradient Boosting exacerba disparidades ao aprender padrões de dados históricos desbalanceados. A auditoria utilizará os indicadores de Impacto Disparate (DIR) e Diferença de Oportunidade (EOD) para demonstrar como a minimização do erro empírico explora correlações espúrias com atributos sensíveis.  
2) Avaliar a robustez estatística em dados escassos: Diferenciar o impacto da arquitetura do modelo na estabilidade preditiva em conjuntos de dados pequenos \[4, 6\]. A análise adotará o Coeficiente de Correlação de Matthews (MCC) como métrica principal, fundamentando-se na literatura recente que aponta a inadequação da ROC-AUC em cenários de alta assimetria de classes (\<3% de classe positiva), onde o MCC demonstra superioridade matemática na distinção de falhas de classificação \[6, 7\].  
3) Isolar o viés indutivo da arquitetura: Contrastar as métricas de equidade entre o modelo de fundação e o modelo tradicional submetido à otimização exaustiva de hiperparâmetros \[2, 4\]. Este objetivo visa comprovar que o pré-treinamento em larga escala (em conjuntos de dados sintéticos e reais) é o fator determinante para a justiça algorítmica, eliminando a dependência de engenharia manual de recursos \[4, 8\].

5. **Etapas da Pesquisa**

Adota-se uma abordagem quantitativa de natureza aplicada, fundamentada no método experimental, com o objetivo de confrontar diferentes paradigmas de aprendizado de máquina em cenários de desbalanceamento sociodemográfico \[1\]. Visando garantir a replicabilidade e a validade científica, o desenho experimental estrutura-se nas seguintes etapas:

1) **Pré-processamento e Auditoria de Dados:**  
   1. A etapa inicial consiste na extração e curadoria dos microdados das Eleições Municipais de 2024, obtidos via Tribunal Superior Eleitoral (TSE) \[5\]. A integridade da distribuição original das classes, como Eleitos e Não Eleitos, será preservada para permitir a auditoria de viés sem contaminação do conjunto de treinamento \[3, 5\].  
2)  **Implementação do Cenário de Controle (Baseline)**  
   1. Configuração do algoritmo XGBoost como modelo de referência, representativo do paradigma de minimização de risco empírico. Este modelo será submetido à otimização de hiperparâmetros e ao ajuste de pesos de classe para mitigar o desbalanceamento. O objetivo é estabelecer um teto de performance para Gradient Boosting Decision Trees, arquitetura que, embora dominante em dados tabulares, tende a amplificar vieses latentes na ausência de regularização específica na função de perda \[3\].  
3) **Execução do Cenário Experimental**  
   1. Aplicação do modelo de fundação TabPFN, operando sob os paradigmas de Aprendizado em Contexto e Inferência Bayesiana Amortizada \[4, 9\]. Diferentemente do controle, este modelo não será submetido a ajuste fino iterativo, testando sua capacidade de generalização zero-shot baseada em priores sintéticos aprendidos previamente \[8\].  
4)  **Avaliação de Robustez e Equidade**  
   1. O cálculo das métricas de desempenho utilizará Validação Cruzada Estratificada (Stratified 5-Fold Cross-Validation). A avaliação priorizará o Coeficiente de Correlação de Matthews (MCC) para aferir a qualidade preditiva, dada a inadequação documentada da ROC-AUC em cenários de desbalanceamento severo (\<3% de classe positiva) \[6, 7\]. Simultaneamente, serão extraídos os indicadores de Impacto Disparate (DIR) e Diferença de Oportunidade (EOD) para mensurar o grau de justiça algorítmica em cada fold \[2, 3\].  
5)  **Validação Estatística**  
   1. Aplicação do Teste de Postos Sinalizados de Wilcoxon (Wilcoxon Signed-Rank Test) para validar a significância das diferenças observadas entre o TabPFN e o XGBoost. Esta etapa visa determinar se as divergências de performance (MCC) e equidade (DIR/EOD) são sistemáticas e estatisticamente relevantes (p\<0.05), descartando a hipótese de aleatoriedade amostral e seguindo protocolos rigorosos de comparação de classificadores em AutoML \[1, 4\].  
6)  **Análise Comparativa e Síntese**  
   1. Confronto final dos dados empíricos com a literatura mais recente, focando na capacidade dos Transformers tabulares atuarem como mitigadores passivos de vieses estruturais. A síntese buscará confirmar se a mudança de arquitetura reduz a dependência de correlações históricas espúrias em contextos de governança algorítmica local \[4, 8\].

6. **Cronograma**

| Etapa da Pesquisa | Data de Conclusão |
| :---- | :---- |
| **Pré-processamento e Auditoria de Dados:** Extração e curadoria de microdados do TSE (2024) e tratamento dos dados. | Semana 1 de janeiro |
| **Implementação do Cenário de Controle (Baseline):** Configuração, treinamento e otimização de hiperparâmetros do modelo XGBoost com ajuste de pesos. | Semana 2 de janeiro |
| **Execução do Cenário Experimental:** Aplicação do modelo TabPFN utilizando Aprendizado em Contexto sem ajuste fino. | Semana 3 de janeiro |
| **Avaliação de Robustez e Equidade:** Realização da Validação Cruzada Estratificada e extração das métricas MCC, DIR e EOD para ambos os cenários. | Semana 4 de janeiro |
| **Validação Estatística:** Aplicação do Teste de Postos Sinalizados de Wilcoxon para verificação de significância das divergências de performance e justiça. | Semana 1 de fevereiro |
| **Análise Comparativa e Síntese:** Discussão dos resultados, confrontando os dados empíricos com o estado da arte sobre *Transformers* tabulares. | Semana 2 de fevereiro |
| **Desenvolvimento do Artigo Científico:** Processo de escrita conforme os padrões de publicação, contemplando os resultados obtidos na pesquisa. | Semanas 3 e 4 de fevereiro |
| **Revisão e Entrega Final:** Formatação final do documento, revisão de referências e submissão. | **08/03/2026** |

7. **Referências Bibliográficas**

\[1\] Grari, V., Ruf, B., Lamprier, S., Detyniecki, M. Achieving Fairness with Decision Trees: An Adversarial Approach. Data Sci. Eng. 5, 99–110, 2020\.  
\[2\] ROBERTSON, J. et al. FairPFN: Transformers Can do Counterfactual Fairness. International Conference on Machine Learning (ICML), 2024\.  
\[3\] RAVICHANDRAN, S. et al. FairXGBoost: Fairness-aware Classification in XGBoost. KDD Workshop on Machine Learning in Finance, 2020\.  
\[4\] GARG, A. et al. Real-TabPFN: Improving Tabular Foundation Models via Continued Pre-training With Real-World Data. arXiv preprint, 2025\.  
\[5\] BASE DOS DADOS. Eleições Brasileiras. Dados do Tribunal Superior Eleitoral (TSE) tratados. 2026\.  
\[6\] CHICCO, D.; JURMAN, G. The Matthews correlation coefficient (MCC) should replace the ROC AUC as the standard metric for assessing binary classification. BioData Mining, 2023\.  
\[7\] IMANI, M. et al. Why ROC-AUC Alone Is Insufficient for Highly Imbalanced Data. Preprints.org, 2025\.  
\[8\] QU, J. et al. TabICL: A Tabular Foundation Model for In-Context Learning on Large Data. arXiv preprint, 2025\.  
\[9\] WIKIPEDIA CONTRIBUTORS. TabPFN. Wikipedia, The Free Encyclopedia, 2025\. (Documentação sobre a versão v2.5 e funcionamento interno).
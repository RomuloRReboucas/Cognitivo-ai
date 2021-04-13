# Cognitivo-ai 
Teste Técnico Data Science
## Análise exploratória para avaliar a consistência dos dados e identificar possíveis variáveis que impactam sua variável resposta.

### Opção: Modelagem para classificação do room type (feature ‘room_type’) com a utilização do Python:<br>

## a. Como foi a definição da sua estratégia de modelagem?

### 1) Análise dos dados <br> 
* Verificação de atributos faltantes,<br>
* Verificação da necessidade de normalização ou padronização.<br>
* Verificação da necessidade de transformação para valores numéricos (one-hot-encoding), se precisar.<br>
* Analise das classes <br>
* Verificação da necessidade de balanceamento das classes<br>

### 2) Modelagem:<br>
* Avaliação de resultados com base de dados sem e com tratamento de balanceamento de classes<br>
* Aplicação dos classificadores Decision Tree, Bagging, Boosting e o Random Forest (RF) <br>
* Aplicação de rede neural ("Cognitivo-ai RNN.ipynb")

### 3) Conclusão:<br>
* Escolha do melhor modelo considerando a melhor acurácia

### Observação: os códigos e os resultados das etapas acima estão disponíveis nos arquivos "Cognitvo AirBnB  - Rômulo Rebouças.ipynb" e "Cognitivo-ai RNN.ipynb". Foi feita a rede neural em arquivo específico processado no ambiente Google Colab. O arquivo "Cognitvo AirBnB  - Rômulo Rebouças.ipynb" foi processado no Jupyter Notbook.

## b. Como foi definida a função de custo utilizada?

Em geral, o MSE (erro do quadrado médio) é utilizado para regressão e a entropia cruzada para classificação. Como o estudo é de classificação foi utilizada a entropia cruzada ('categorical_crossentropy') combinada com o otimizador “Adam” com taxa de aprendizado 0,001. Foram testadas várias taxas de aprendizado no intervalo entre 0,10 e 0,001. O que obteve-se o melhor resultado foi o de 0,001. 

## c. Qual foi o critério utilizado na seleção do modelo final?

No caso dos classificadores Decision Tree, Bagging, Boosting e o Random Forest (RF) foi verificada a acurácia. No caso da rede neural além da acurácia, precisão e revocação, verificou-se o decaimento da função de perda (loss).

Como a base de dados apresentou desbalanceamento acentuado das classes, analizou-se os modelos utilizando-se os métodos de balanceamento Oversampling, Undersampling e Híbrido, com a finalidade de aplicar os modelos com dados em condições mais apropriadas.

Em geral, verificou-se que os modelos aplicados com o uso da base de dados tratada com o método Hibrido (utilizando o oversampling e o undersampling - SMOTEENN) obteverão melhores resultados.

Como resultado, para a base normal (sem tratamento de balanceamento) a classificação "Random Forest" obteve a melhor acurácia, 81%. Considerando a base dados com tratamento (método Hibrido) a rede neural obteve a melhor acurácia, 89%.

## d. Qual foi o critério utilizado para validação do modelo?

O critério utilizado para a validação foi a acurácia considerando o uso da base de testes.

Por que escolheu utilizar este método?

Porque a acurácia mede a proporção de casos que foram corretamente previstos pelo modelo.

Um ponto importante a ser considerado quando pensamos em boa acurácia é a verificação de problemas de Overfitting. Neste estudo, foram feitas simulações para mitigar possível overfitting considerando que hove tratamento de balanceamento de dados e de ajustes de hiperparâmetros nos modelos. 

## e. Quais evidências você possui de que seu modelo é suficientemente bom?

Espera-se que o modelo tenha boa performance considerando-se uma boa acurária, pois significa dizer que exitiu, nos testes, boa acertividade de casos considerados "verdadeiro poistivo" como também de "verdadeiro negativo".





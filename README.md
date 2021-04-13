# Cognitivo-ai 
Teste Técnico Data Science
### Análise exploratória para avaliar a consistência dos dados e identificar possíveis variáveis que impactam sua variável resposta.

### Opção: Modelagem para classificação do room type (feature ‘room_type’) com a utilização do Python:<br>

# a. Como foi a definição da sua estratégia de modelagem?

#### 1) Análise dos dados <br> 
* Verificação de atributos faltantes,<br>
* Verificação da necessidade de normalização ou padronização.<br>
* Verificação da necessidade de transformação para valores numéricos (one-hot-encoding), se precisar.<br>
* Analise das classes <br>
* Verificação da necessidade de balanceamento das classes<br>

#### 2) Modelagem:<br>
* Avaliação de resultados com base sem e com balanceamento<br>
* Aplicação dos classificadores Decision Tree, Bagging, Boosting e o Random Forest (RF) <br>
* Aplicação de rede neural ("Cognitivo-ai RNN.ipynb")

### Observação: os códigos e os resultados das etapas acima estão disponíveis nos arquivos "Cognitvo AirBnB  - Rômulo Rebouças.ipynb" e "Cognitivo-ai RNN.ipynb". Foi feita a rede neural em arquivo específico processado no ambiente Google Colab. O arquivo "Cognitvo AirBnB  - Rômulo Rebouças.ipynb" foi processado no Jupyter Notbook.

# b. Como foi definida a função de custo utilizada?

Em geral, o MSE (erro do quadrado médio) é utilizado para regressão e a entropia cruzada para classificação. Como o estudo é de classificação foi utilizada a entropia cruzada ('categorical_crossentropy') combinada com o otimizador “Adam” com taxa de aprendizado 0,001. Foram testadas várias taxas de aprendizado no intervalo entre 0,10 e 0,001. O que obteve-se o melhor resultado foi o de 0,001. 

c. Qual foi o critério utilizado na seleção do modelo final?
d. Qual foi o critério utilizado para validação do modelo?
Por que escolheu utilizar este método?
e. Quais evidências você possui de que seu modelo é
suficientemente bom?

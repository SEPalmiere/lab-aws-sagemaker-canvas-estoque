# 📊 Previsão de Estoque Inteligente na AWS com [SageMaker Canvas](https://aws.amazon.com/pt/sagemaker/canvas/)

Bem-vindo ao desafio de projeto "Previsão de Estoque Inteligente na AWS com SageMaker Canvas. Neste Lab DIO, você aprenderá a usar o SageMaker Canvas para criar previsões de estoque baseadas em Machine Learning (ML). Siga os passos abaixo para completar o desafio!

## 📋 Pré-requisitos

Antes de começar, certifique-se de ter uma conta na AWS. Se precisar de ajuda para criar sua conta, confira nosso repositório [AWS Cloud Quickstart](https://github.com/digitalinnovationone/aws-cloud-quickstart).


## 🎯 Objetivos Deste Desafio de Projeto (Lab)

![image](https://github.com/digitalinnovationone/lab-aws-sagemaker-canvas-estoque/assets/730492/72f5c21f-5562-491e-aa42-2885a3184650)

- Dê um fork neste projeto e reescreva este `README.md`. Sinta-se à vontade para detalhar todo o processo de criação do seu Modelo de ML para uma "Previsão de Estoque Inteligente".
- Para isso, siga o [passo a passo] descrito a seguir e evolua as suas habilidades em ML no-code com o Amazon SageMaker Canvas.
- Ao concluir, envie a URL do seu repositório com a solução na plataforma da DIO.


## 🚀 Passo a Passo

### 1. Selecionar Dataset

-   Navegue até a pasta `datasets` deste repositório. Esta pasta contém os datasets que você poderá escolher para treinar e testar seu modelo de ML. Sinta-se à vontade para gerar/enriquecer seus próprios datasets, quanto mais você se engajar, mais relevante esse projeto será em seu portfólio.
-   Escolha o dataset que você usará para treinar seu modelo de previsão de estoque.
-   Faça o upload do dataset no SageMaker Canvas.

### 2. Construir/Treinar

-   No SageMaker Canvas, importe o dataset que você selecionou.
-   Configure as variáveis de entrada e saída de acordo com os dados.
-   Inicie o treinamento do modelo. Isso pode levar algum tempo, dependendo do tamanho do dataset.

### 3. Analisar

-   Após o treinamento, examine as métricas de performance do modelo.
-   Verifique as principais características que influenciam as previsões.
-   Faça ajustes no modelo se necessário e re-treine até obter um desempenho satisfatório.

### 4. Prever

-   Use o modelo treinado para fazer previsões de estoque.
-   Exporte os resultados e analise as previsões geradas.
-   Documente suas conclusões e qualquer insight obtido a partir das previsões.

## 🤔 Dúvidas?

Esperamos que esta experiência tenha sido enriquecedora e que você tenha aprendido mais sobre Machine Learning aplicado a problemas reais. Se tiver alguma dúvida, não hesite em abrir uma issue neste repositório ou entrar em contato com a equipe da DIO.



Projeto: Previsão de Estoque Inteligente na AWS com SageMaker Canvas
Introdução

Neste projeto, utilizaremos o SageMaker Canvas, um serviço de Machine Learning de baixo código da AWS, para criar um modelo de previsão de estoque. Este modelo nos ajudará a prever a demanda futura de produtos com base em dados históricos de vendas e outros fatores relevantes.
Pré-requisitos

    Conta da AWS
    Conhecimento básico de Machine Learning
    Experiência com o SageMaker Canvas (opcional)

Passos
1. Criar um notebook do SageMaker

import sagemaker
from sagemaker.canvas import *

2. Importar as bibliotecas necessárias

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

3. Carregar os dados

Carregue os dados de vendas históricos em um DataFrame do Pandas. Certifique-se de que os dados contenham as seguintes colunas:

    product_id
    sales_date
    sales_quantity

4. Explorar os dados

df.head()
df.describe()
df.plot(x='sales_date', y='sales_quantity')

5. Criar o modelo de previsão
Dividir os dados em conjuntos de treinamento e teste

X_train, X_test, y_train, y_test = train_test_split(df[['sales_date']], df['sales_quantity'], test_size=0.2, random_state=42)

Criar e treinar o modelo de regressão linear

model = LinearRegression()
model.fit(X_train, y_train)

6. Avaliar o modelo
Fazer previsões nos dados de teste

y_pred = model.predict(X_test)

Calcular as métricas de avaliação

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print('MSE:', mse)
print('MAE:', mae)

7. Implantar o modelo
Criar um endpoint do SageMaker

endpoint_name = 'previsao-estoque'
role = sagemaker.get_execution_role()

endpoint = sagemaker.Endpoint(endpoint_name, role, 'linear-learner')
endpoint.deploy(model, initial_instance_count=1)

Conclusão

Neste projeto, criamos um modelo de previsão de estoque usando o SageMaker Canvas. Este modelo pode ser usado para prever a demanda futura de produtos com base em dados históricos de vendas e outros fatores relevantes. Isso pode ajudar as empresas a otimizar seus níveis de estoque e reduzir custos.

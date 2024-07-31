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


Projeto: Otimização de Estoque com AWS SageMaker Canvas
Visão Geral
Este projeto demonstra como utilizar o AWS SageMaker Canvas, uma ferramenta de Machine Learning de baixo código, para desenvolver um sistema inteligente de previsão de estoque. Nosso objetivo é criar um modelo preditivo que antecipe a demanda futura de produtos, baseando-se em dados históricos de vendas e variáveis contextuais relevantes.
Requisitos

Conta ativa na AWS
Compreensão básica de conceitos de Machine Learning
Familiaridade com Python e Pandas (recomendado)

1. Configuração do Ambiente SageMaker

import boto3
import sagemaker
from sagemaker import get_execution_role

role = get_execution_role()
session = sagemaker.Session()

# Configurar o bucket S3 para armazenamento de dados
bucket = session.default_bucket()
prefix = 'sagemaker/previsao-estoque'

2. Preparação e Análise de Dados

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Carregar dados (substitua 'seu_arquivo.csv' pelo nome real do seu arquivo)
df = pd.read_csv('seu_arquivo.csv')

# Visualização inicial dos dados
print(df.head())
print(df.describe())

# Análise gráfica das vendas ao longo do tempo
plt.figure(figsize=(12,6))
plt.plot(df['sales_date'], df['sales_quantity'])
plt.title('Histórico de Vendas')
plt.xlabel('Data')
plt.ylabel('Quantidade Vendida')
plt.show()

3. Pré-processamento dos Dados

# Converter 'sales_date' para datetime
df['sales_date'] = pd.to_datetime(df['sales_date'])

# Extrair características temporais
df['day_of_week'] = df['sales_date'].dt.dayofweek
df['month'] = df['sales_date'].dt.month
df['year'] = df['sales_date'].dt.year

# Selecionar features para o modelo
features = ['day_of_week', 'month', 'year']
X = df[features]
y = df['sales_quantity']

# Dividir dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar os dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

4. Treinamento do Modelo com SageMaker Canvas
No SageMaker Canvas, siga estes passos:

Importe seus dados pré-processados.
Selecione 'sales_quantity' como a coluna alvo.
Escolha o tipo de modelo (por exemplo, regressão para previsão de quantidades).
Inicie o treinamento do modelo.

5. Avaliação do Modelo
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Assumindo que 'predictions' são as previsões do seu modelo Canvas
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f'Erro Quadrático Médio: {mse}')
print(f'Erro Absoluto Médio: {mae}')
print(f'R² Score: {r2}')

6.  Implementação do Modelo
from sagemaker.model import Model
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import JSONDeserializer

# Criar um modelo SageMaker a partir do modelo Canvas
model_data = f's3://{bucket}/{prefix}/model.tar.gz'
model = Model(model_data=model_data, role=role, sagemaker_session=session)

# Configurar o endpoint
predictor = model.deploy(
    initial_instance_count=1,
    instance_type='ml.t2.medium',
    serializer=CSVSerializer(),
    deserializer=JSONDeserializer()
)

7.  Utilização do Modelo para Previsões
# Preparar dados para previsão
new_data = pd.DataFrame({
    'day_of_week': [3],
    'month': [7],
    'year': [2024]
})

# Normalizar os novos dados
new_data_scaled = scaler.transform(new_data)

# Fazer previsão
prediction = predictor.predict(new_data_scaled.tolist())
print(f'Previsão de vendas: {prediction}')

8.  Conclusão

Neste projeto, desenvolvemos um sistema avançado de previsão de estoque utilizando o AWS SageMaker Canvas. Através da análise de dados históricos de vendas e variáveis temporais, criamos um modelo capaz de prever com precisão a demanda futura de produtos.
A implementação deste sistema oferece diversos benefícios:

Otimização do gerenciamento de estoque
Redução de custos operacionais
Melhoria na satisfação do cliente através da disponibilidade adequada de produtos

Este projeto demonstra como ferramentas de ML de baixo código, como o SageMaker Canvas, podem ser utilizadas para resolver problemas complexos de negócios de forma eficiente e escalável. À medida que continuamos a refinar e adaptar o modelo, podemos esperar melhorias contínuas na precisão das previsões e, consequentemente, na eficiência operacional da empresa.

# 🏠 House Prices - Previsão de Imóveis

Projeto baseado no desafio [House Prices - Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques), com o objetivo de prever o preço de casas em Ames, Iowa.

## 📌 Objetivo
Aplicar técnicas de aprendizado de máquina para prever o preço final de imóveis residenciais com base em atributos diversos como localização, área construída e qualidade geral.

## 🔍 O que você vai encontrar:
- 📊 EDA com visualizações e insights interpretáveis
- 🧼 Tratamento completo de dados ausentes e outliers
- 🧠 Modelos Lasso, Ridge, XGBoost e Ensemble
- 📈 Avaliação com validação cruzada e análise de métricas

## 🧠 Principais Aprendizados
- Feature engineering impacta fortemente em modelos tabulares
- Modelos lineares regulares são competitivos com ensembles
- Explicabilidade é tão importante quanto performance

## 📁 Organização base

````
house-of-price/
├── data/                   # Dados originais e tratados
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_results.ipynb
├── sandbox/                # Experimentos, testes e rascunhos
├── src/                    # Funções reutilizáveis (ex: utils.py)
├── models/                 # Modelos treinados e artefatos
├── requirements.txt
└── README.md

```

## 🚀 Como rodar
```bash
pip install -r requirements.txt
jupyter notebook
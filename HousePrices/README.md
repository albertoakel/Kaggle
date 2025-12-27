# 🏠 House Prices — Previsão de Imóveis Residenciais

Projeto baseado no desafio **[House Prices – Advanced Regression Techniques (Kaggle)]**.
O objetivo é prever o **preço final de casas em Ames, Iowa**, utilizando técnicas modernas de **Machine Learning para dados tabulares**.

---

## 📌 Objetivo do Projeto

Desenvolver um pipeline completo de **pré-processamento, modelagem e avaliação**, comparando modelos lineares regularizados e métodos ensemble, com foco em:

* desempenho preditivo
* controle de overfitting
* reprodutibilidade
* organização para portfólio profissional

---

## 🔍 O que você vai encontrar neste projeto

* **EDA detalhada** com análise estatística e visual
*  **Pré-processamento robusto** (imputação, normalização, one-hot encoding)
* **Modelos avaliados**:

  * Regressão Linear
  * Ridge e LASSO
  * Random Forest Regressor
  * XGBoost
* **Avaliação comparativa** com MAE, RMSE e R²
*  **Artefatos persistidos** (preprocessador e melhor modelo)

---

## 📊 Resultados dos Modelos

Avaliação realizada sobre o conjunto de teste (target transformado com `log1p`).

| Modelo                   | MAE        | RMSE       | R²         |
| ------------------------ | ---------- | ---------- | ---------- |
| Linear Regression        | 0.0950     | 0.1826     | 0.8035     |
| Ridge (config 0)         | 0.0945     | 0.1679     | 0.8337     |
| Ridge (config 1)         | 0.0963     | 0.1346     | 0.8932     |
| LASSO (config 0)         | 0.1089     | 0.1508     | 0.8660     |
| LASSO (config 1)         | 0.0994     | 0.1384     | 0.8871     |
| Random Forest (config 0) | 0.0934     | 0.1382     | 0.8874     |
| Random Forest (config 1) | 0.0919     | 0.1383     | 0.8872     |
| XGBoost (config 0)       | 0.0976     | 0.1450     | 0.8760     |
| XGBoost (config 1)       | 0.0894     | 0.1320     | 0.8973     |
| **XGBoost (config 2)**   | **0.0838** | **0.1240** | **0.9093** |

➡️ **Melhor desempenho geral:** XGBoost (configuração 2)

---

## 🧠 Principais Aprendizados

* Feature engineering e pré-processamento influenciam mais que o algoritmo em si
* Modelos lineares regularizados são fortes baselines
* XGBoost apresentou o melhor equilíbrio entre viés e variância
* Organização do pipeline é essencial para evitar *data leakage*
* Persistir preprocessadores facilita inferência e deploy

---

## 📁 Estrutura do Projeto

```
HousePrices/
│
├── app/                     # Aplicações futuras (deploy)
│
├── data/
│   ├── raw/                 # Dados originais
│   └── processed/           # Bases pós-processadas
│
├── image/                   # Imagens e figuras
│
├── notebook/
│   ├── eda_HP.ipynb
│   ├── models_Linear.ipynb
│   ├── models_Random_Forest.ipynb
│   ├── models_XGBoost.ipynb
│   ├── XGB2_submission.ipynb
│   └── setup_notebook.py
│
├── sandbox/                 # Testes, rascunhos e experimentos
│
├── src/
│   ├── preprocess_utils.py
│   ├── model_utils.py
│   ├── functions.py
│   ├── preprocess_house_prices_v1.joblib
│   └── melhor_modelo.h5
│
├── requirements.txt
└── README.md
```

---

## 🚀 Como executar o projeto

### 1️⃣ Instalar dependências

```bash
pip install -r requirements.txt
```

### 2️⃣ Executar notebooks

```bash
jupyter notebook
```

### 3️⃣ Pré-processamento automatizado

```bash
python src/preprocess_utils.py
```

---

## 📌 Observações Finais

Este projeto foi estruturado com foco em **boas práticas de ciência de dados**, servindo tanto como **benchmark técnico** quanto como **material de portfólio profissional**.

---
